"""Remediation: LSB scrub / permutation / quantization.

TODO(S8):
- lsb_scrub(model, k) — zero/re-randomize low-k mantissa bits across tensors
  (destroys LSB payloads).
- permute(model) — permutation-invariant reorder within eligible layers
  (breaks position-dependent stego).
- quantize(model) — optional int8 round-trip.
Save disarmed model; run held-out eval.npz to report accuracy delta.
"""

from __future__ import annotations

# TODO(S8): lsb_scrub / permute / quantize
