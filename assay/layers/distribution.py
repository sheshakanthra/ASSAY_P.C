"""Layer 2c: per-tensor distribution anomaly + sliding-window localization.

TODO(S5): per-tensor stats — mean/std/kurtosis, exact-zero fraction, denormal
fraction, bit-pattern histogram; flag tensors whose distribution deviates from
the model's own layer-type baseline (e.g. EvilModel-style overwritten
"atrophied" weights produce localized anomalous blocks).
"""

from __future__ import annotations

# TODO(S5): analyze_distribution(tensor_info, array, baseline) -> list[Finding]
