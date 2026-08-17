"""SAFE tensor loading only.

TODO(S1): implement `iter_tensors(path) -> Iterator[tuple[TensorInfo, np.ndarray]]`.

Hard rules (see CLAUDE.md golden rule 7 — never execute an untrusted model):
- safetensors via `safetensors.safe_open` (numpy).
- torch via `torch.load(..., weights_only=True, map_location="cpu")` then `.numpy()`.
- Never `pickle.load`; never bare `torch.load`.
- Raise `UnsafeArtifactError` on anything that would require code execution to load.
- Malformed/truncated files -> typed errors, not crashes.
"""

from __future__ import annotations


class UnsafeArtifactError(Exception):
    """Raised when loading an artifact would require unsafe code execution."""


class MalformedArtifactError(Exception):
    """Raised when an artifact is truncated, corrupt, or otherwise unparseable."""


# TODO(S1): iter_tensors(path) -> Iterator[tuple[TensorInfo, np.ndarray]]
