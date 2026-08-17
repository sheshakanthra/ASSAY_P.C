"""Integrity attestation for disarmed artifacts.

TODO(S8): per-tensor SHA-256 manifest + top-level hash -> signed attestation.json
consumed by the CI gate.
"""

from __future__ import annotations

# TODO(S8): build_attestation(path) -> dict; verify_attestation(path, attestation) -> bool
