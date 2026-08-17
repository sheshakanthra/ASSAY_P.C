"""S5 GATE tests: Layer 2b byte-signature sweep.

- 100% recovery of planted signatures (EICAR, ELF header, plaintext markers)
  with correct offsets
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assay.intake.loader import iter_tensors
from assay.layers.steg_signature import EICAR, analyze_signatures
from assay.models import TensorInfo

MODELS_DIR = Path("fixtures/models")
MANIFEST_PATH = MODELS_DIR / "manifest.json"


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())


def _load_tensor(filename: str, tensor_name: str) -> tuple[TensorInfo, np.ndarray]:
    tensors = {info.name: (info, arr) for info, arr in iter_tensors(MODELS_DIR / filename)}
    return tensors[tensor_name]


def _rules(findings) -> set[str]:
    return {f.rule for f in findings}


# ---------------------------------------------------------------------------
# clean pass
# ---------------------------------------------------------------------------


def test_clean_fixture_all_tensors_produce_zero_findings():
    for info, arr in iter_tensors(MODELS_DIR / "clean_cnn.safetensors"):
        findings = analyze_signatures(info, arr)
        assert findings == [], f"false positive on clean tensor {info.name}: {findings}"


# ---------------------------------------------------------------------------
# 100% recovery of planted signatures, with correct offsets
# ---------------------------------------------------------------------------


def test_eicar_recovered_at_correct_offset(manifest):
    entry = next(e for e in manifest if e["technique"] == "lsb_eicar")
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])
    findings = analyze_signatures(info, arr)

    matches = [f for f in findings if f.rule == "eicar_signature"]
    assert len(matches) == 1
    assert matches[0].value == float(entry["ground_truth_region"][0])
    assert f"offset {entry['ground_truth_region'][0]}" in matches[0].detail


def test_elf_header_recovered_at_correct_offset(manifest):
    entry = next(e for e in manifest if e["technique"] == "lsb_elf_header")
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])
    findings = analyze_signatures(info, arr)

    matches = [f for f in findings if f.rule == "elf_header"]
    assert len(matches) == 1
    assert matches[0].value == float(entry["ground_truth_region"][0])


def test_plaintext_markers_both_recovered_at_correct_offsets(manifest):
    entry = next(e for e in manifest if e["technique"] == "plaintext_marker")
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])
    findings = analyze_signatures(info, arr)

    shell_matches = [f for f in findings if f.rule == "shell_string"]
    ip_matches = [f for f in findings if f.rule == "ipv4_address"]
    assert len(shell_matches) == 1
    assert len(ip_matches) == 1
    # "/bin/sh\x00" is 8 bytes, so the IP marker starts right after it.
    gt_start = entry["ground_truth_region"][0]
    assert shell_matches[0].value == float(gt_start)
    assert ip_matches[0].value == float(gt_start + 8)


def test_random_blob_produces_no_signature_match(manifest):
    """Genuinely random bytes have no reason to match a known signature —
    documented here so the absence is a deliberate, expected result."""
    entry = next(e for e in manifest if e["technique"] == "lsb_random_blob")
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])
    findings = analyze_signatures(info, arr)
    assert findings == []


def test_bad_pickle_is_out_of_scope_for_this_layer(manifest):
    # bad_pickle is a wrapper-level (Layer 1) technique with no target tensor.
    entry = next(e for e in manifest if e["technique"] == "bad_pickle")
    assert entry["target_tensor"] is None


# ---------------------------------------------------------------------------
# unit tests: pattern matching in isolation
# ---------------------------------------------------------------------------


def _tensor_with_low_byte_payload(payload: bytes, n: int = 128) -> tuple[TensorInfo, np.ndarray]:
    rng = np.random.default_rng(1)
    arr = (rng.standard_normal(n) * 0.02).astype(np.float32)
    u32 = arr.view(np.uint32)
    for i, byte in enumerate(payload):
        u32[i] = (u32[i] & 0xFFFFFF00) | byte
    return TensorInfo(name="w", dtype="float32", shape=(n,)), arr


def test_eicar_signature_constant_matches_standard_string():
    assert EICAR == b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"


def test_zip_header_detected():
    info, arr = _tensor_with_low_byte_payload(b"PK\x03\x04rest")
    findings = analyze_signatures(info, arr)
    assert any(f.rule == "zip_header" for f in findings)


def test_macho_header_detected():
    info, arr = _tensor_with_low_byte_payload(b"\xfe\xed\xfa\xce" + b"pad")
    findings = analyze_signatures(info, arr)
    assert any(f.rule == "macho_header" for f in findings)


def test_pe_header_requires_both_mz_and_pe_signature():
    info, arr = _tensor_with_low_byte_payload(b"MZ" + b"\x00" * 30 + b"PE\x00\x00")
    findings = analyze_signatures(info, arr)
    assert any(f.rule == "pe_header" for f in findings)


def test_bare_mz_without_pe_signature_is_not_flagged():
    info, arr = _tensor_with_low_byte_payload(b"MZ" + b"just two bytes, no real PE header follows")
    findings = analyze_signatures(info, arr)
    assert not any(f.rule == "pe_header" for f in findings)


def test_url_detected():
    info, arr = _tensor_with_low_byte_payload(b"https://example.com/payload")
    findings = analyze_signatures(info, arr)
    assert any(f.rule == "url" for f in findings)


def test_base64_blob_detected():
    payload = b"QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbg=="
    info, arr = _tensor_with_low_byte_payload(payload, n=256)
    findings = analyze_signatures(info, arr)
    assert any(f.rule == "base64_blob" for f in findings)


def test_short_base64_like_text_not_flagged():
    info, arr = _tensor_with_low_byte_payload(b"abcd")
    findings = analyze_signatures(info, arr)
    assert not any(f.rule == "base64_blob" for f in findings)
