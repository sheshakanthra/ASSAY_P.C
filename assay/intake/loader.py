"""SAFE tensor loading only.

`iter_tensors(path) -> Iterator[tuple[TensorInfo, np.ndarray]]` is the uniform
entry point every later layer reads through.

Hard rules (see CLAUDE.md §SAFETY + golden rule 7 — never execute an untrusted model):
- safetensors via `safetensors.safe_open` (numpy framework; falls back to the
  torch framework + a bit-preserving view only for bfloat16, which numpy has
  no native dtype for).
- torch via `torch.load(..., weights_only=True, map_location="cpu")` then `.numpy()`.
  This is the sanctioned safe API: its restricted unpickler rejects any GLOBAL
  reference outside a small allowlist instead of executing it.
- Never `pickle.load`; never bare `torch.load` (i.e. never without `weights_only=True`).
- Malformed/truncated files and rejected unsafe globals both raise typed errors,
  not crashes.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open

from assay.intake.detect import detect_format
from assay.models import TensorInfo

# Formats that are, at the byte level, a pickle stream torch.load can parse
# safely under weights_only=True — whether zip-wrapped ("pt") or bare ("pkl"),
# or extension-only ambiguous ("bin", commonly a HF checkpoint of either shape).
_TORCH_LOADABLE_FORMATS = frozenset({"pt", "pkl", "bin"})

# Substring torch's restricted weights_only unpickler includes in its message
# specifically when it refuses a disallowed GLOBAL reference (i.e. the file
# tried to smuggle in code execution), as opposed to plain corruption.
_UNSAFE_GLOBAL_MARKER = "unsupported global"


class UnsafeArtifactError(Exception):
    """Raised when loading an artifact would require unsafe code execution."""


class MalformedArtifactError(Exception):
    """Raised when an artifact is truncated, corrupt, or otherwise unparseable."""


class UnsupportedFormatError(Exception):
    """Raised for a recognized-but-not-yet-loadable format (e.g. onnx, gguf)."""


def iter_tensors(path: str | Path) -> Iterator[tuple[TensorInfo, np.ndarray]]:
    """Safely iterate (TensorInfo, ndarray) pairs from a model artifact.

    Raises MalformedArtifactError for missing/empty/corrupt/truncated files,
    UnsafeArtifactError if the artifact's pickle stream references a global
    outside torch's weights_only allowlist, and UnsupportedFormatError for
    recognized formats this loader doesn't yet know how to read.
    """
    p = Path(path)
    if not p.exists():
        raise MalformedArtifactError(f"no such file: {p}")
    if p.stat().st_size == 0:
        raise MalformedArtifactError(f"empty file: {p}")

    fmt = detect_format(p)
    if fmt == "safetensors":
        yield from _iter_safetensors(p)
    elif fmt in _TORCH_LOADABLE_FORMATS:
        yield from _iter_torch(p)
    else:
        raise UnsupportedFormatError(f"{p}: format '{fmt}' is not supported for tensor iteration")


def _iter_safetensors(p: Path) -> Iterator[tuple[TensorInfo, np.ndarray]]:
    try:
        with safe_open(str(p), framework="numpy") as f:
            names = list(f.keys())
            for name in names:
                try:
                    arr = f.get_tensor(name)
                    dtype_name = str(arr.dtype)
                except TypeError:
                    # numpy has no native bfloat16; re-read that tensor via the torch
                    # framework and preserve its exact bit pattern as uint16 instead.
                    with safe_open(str(p), framework="pt") as f_pt:
                        t = f_pt.get_tensor(name)
                    dtype_name = _torch_dtype_name(t.dtype)
                    arr = _torch_to_numpy(t)
                yield TensorInfo(name=name, dtype=dtype_name, shape=tuple(arr.shape)), arr
    except (MalformedArtifactError, UnsafeArtifactError):
        raise
    except Exception as e:
        raise MalformedArtifactError(f"{p}: failed to parse safetensors file: {e}") from e


def _iter_torch(p: Path) -> Iterator[tuple[TensorInfo, np.ndarray]]:
    try:
        obj = torch.load(str(p), weights_only=True, map_location="cpu")
    except (MalformedArtifactError, UnsafeArtifactError):
        raise
    except Exception as e:
        raise _classify_torch_load_error(e, p) from e

    yield from _walk_tensors(obj)


def _classify_torch_load_error(e: Exception, path: Path) -> Exception:
    if _UNSAFE_GLOBAL_MARKER in str(e).lower():
        return UnsafeArtifactError(f"{path}: refused unsafe pickle content: {e}")
    return MalformedArtifactError(f"{path}: failed to load torch archive: {e}")


def _walk_tensors(obj: Any, prefix: str = "") -> Iterator[tuple[TensorInfo, np.ndarray]]:
    """Recursively find tensors in whatever torch.load returned (a bare tensor,
    a flat state_dict, or a nested checkpoint dict/list); skip non-tensor leaves.
    """
    if isinstance(obj, torch.Tensor):
        name = prefix or "tensor"
        arr = _torch_to_numpy(obj)
        info = TensorInfo(
            name=name,
            dtype=_torch_dtype_name(obj.dtype),
            shape=tuple(int(d) for d in obj.shape),
        )
        yield info, arr
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _walk_tensors(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            yield from _walk_tensors(v, f"{prefix}[{i}]" if prefix else f"[{i}]")


def _torch_dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).removeprefix("torch.")


def _torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    t = t.detach().cpu().contiguous()
    if t.dtype == torch.bfloat16:
        # numpy has no native bfloat16; expose the raw 16-bit pattern instead
        # so later layers (LSB/entropy, S4+) can still inspect the bytes.
        return t.view(torch.uint16).numpy()
    return t.numpy()
