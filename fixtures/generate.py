"""Build INERT poisoned twins of the clean baseline + a labeled manifest.json.

See SAFETY.md (and CLAUDE.md §SAFETY): every marker planted below is
detectable but non-executable. No component of this script produces a
working payload, a live exploit, or anything that runs.

Techniques (one poisoned artifact each), all derived from clean_cnn.safetensors
except bad_pickle, which is a standalone hand-built pickle (never derived from
the clean model, since it targets the *wrapper*, not the weights):
  - lsb_eicar         EICAR standard AV test string in a tensor's mantissa LSBs.
  - lsb_elf_header    dummy \\x7fELF magic + zero padding in mantissa LSBs.
  - lsb_random_blob   high-entropy random bytes in mantissa LSBs (simulated
                       encrypted payload).
  - plaintext_marker  sentinel strings (/bin/sh, a TEST-NET-1 documentation IP)
                       in mantissa LSBs.
  - bad_pickle        a pickle whose opcode stream references os.system, built
                       via __reduce__ + pickle.dumps and never unpickled.
"""

from __future__ import annotations

import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from safetensors.numpy import save_file as save_safetensors

from assay.intake.loader import iter_tensors

FIXTURES_DIR = Path(__file__).parent
MODELS_DIR = FIXTURES_DIR / "models"
CLEAN_SAFETENSORS = MODELS_DIR / "clean_cnn.safetensors"

EICAR = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"
ELF_HEADER = b"\x7fELF" + b"\x00" * 60  # dummy, non-executable padding
# 192.0.2.1 is in the TEST-NET-1 documentation range (RFC 5737) — never a real host.
PLAINTEXT_MARKERS = b"/bin/sh\x00192.0.2.1\x00"
RANDOM_BLOB_SIZE = 256

LSB_BITS = 8  # low byte of each float32's 23-bit mantissa: 1 payload byte per element


@dataclass
class ManifestEntry:
    file: str
    label: str  # "clean" | "poisoned"
    technique: str | None
    target_tensor: str | None
    # [start, end) flat element index into the target tensor's flattened array.
    ground_truth_region: list[int] | None


def _load_clean_state(path: Path) -> dict[str, np.ndarray]:
    return {info.name: arr.copy() for info, arr in iter_tensors(path)}


def _eligible_targets(state: dict[str, np.ndarray], min_size: int) -> list[str]:
    names = [n for n, arr in state.items() if arr.dtype == np.float32 and arr.size >= min_size]
    return sorted(names, key=lambda n: -state[n].size)


def _embed_bytes_lsb(
    arr: np.ndarray, payload: bytes, bits: int = LSB_BITS
) -> tuple[np.ndarray, tuple[int, int]]:
    """Overwrite the low `bits` bits of each float32 element's bit pattern with
    one payload byte per element (sign + exponent untouched). Returns the
    poisoned array (same shape) and the flat [start, end) element range touched.
    """
    if arr.dtype != np.float32:
        raise ValueError(f"LSB embedding requires float32 tensors, got {arr.dtype}")
    if len(payload) > arr.size:
        raise ValueError("payload does not fit in target tensor")

    flat_u32 = arr.reshape(-1).copy().view(np.uint32)
    mask = np.uint32((~((1 << bits) - 1)) & 0xFFFFFFFF)
    keep_bits = np.uint32((1 << bits) - 1)
    start = 0
    for i, byte in enumerate(payload):
        flat_u32[start + i] = (flat_u32[start + i] & mask) | (np.uint32(byte) & keep_bits)
    poisoned = flat_u32.view(np.float32).reshape(arr.shape)
    return poisoned, (start, start + len(payload))


def _make_lsb_variant(
    clean_state: dict[str, np.ndarray], technique: str, payload: bytes, tensor_index: int
) -> ManifestEntry:
    candidates = _eligible_targets(clean_state, len(payload))
    if not candidates:
        raise RuntimeError(f"no tensor large enough to embed {technique} ({len(payload)} bytes)")
    target_name = candidates[tensor_index % len(candidates)]

    poisoned_state = {k: v.copy() for k, v in clean_state.items()}
    poisoned_arr, region = _embed_bytes_lsb(clean_state[target_name], payload)
    poisoned_state[target_name] = poisoned_arr

    out_name = f"poison_{technique}.safetensors"
    save_safetensors(poisoned_state, str(MODELS_DIR / out_name))
    return ManifestEntry(
        file=out_name,
        label="poisoned",
        technique=technique,
        target_tensor=target_name,
        ground_truth_region=[region[0], region[1]],
    )


class _EvilReduce:
    """Never instantiated/unpickled — pickle.dumps'd only, to produce the
    on-disk opcode bytes a malicious checkpoint would contain. See SAFETY.md:
    pickle opcode sequences that are statically analyzed and never deserialized.
    """

    def __reduce__(self):
        return (os.system, ("echo pwned",))


def _make_bad_pickle_variant() -> ManifestEntry:
    out_name = "poison_bad_pickle.pkl"
    data = pickle.dumps(_EvilReduce(), protocol=2)
    (MODELS_DIR / out_name).write_bytes(data)
    return ManifestEntry(
        file=out_name,
        label="poisoned",
        technique="bad_pickle",
        target_tensor=None,
        ground_truth_region=None,
    )


def main() -> None:
    if not CLEAN_SAFETENSORS.exists():
        raise SystemExit(
            f"{CLEAN_SAFETENSORS} not found — run `python fixtures/train_baseline.py` first"
        )
    clean_state = _load_clean_state(CLEAN_SAFETENSORS)

    entries = [
        ManifestEntry(
            file=CLEAN_SAFETENSORS.name,
            label="clean",
            technique=None,
            target_tensor=None,
            ground_truth_region=None,
        ),
        _make_lsb_variant(clean_state, "lsb_eicar", EICAR, tensor_index=0),
        _make_lsb_variant(clean_state, "lsb_elf_header", ELF_HEADER, tensor_index=1),
        _make_lsb_variant(clean_state, "lsb_random_blob", os.urandom(RANDOM_BLOB_SIZE), tensor_index=0),
        _make_lsb_variant(clean_state, "plaintext_marker", PLAINTEXT_MARKERS, tensor_index=1),
        _make_bad_pickle_variant(),
    ]

    manifest = [asdict(e) for e in entries]
    manifest_path = MODELS_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    n_clean = sum(1 for e in entries if e.label == "clean")
    n_poisoned = sum(1 for e in entries if e.label == "poisoned")
    print(f"wrote {len(manifest)} manifest entries ({n_clean} clean, {n_poisoned} poisoned) to {manifest_path}")


if __name__ == "__main__":
    main()
