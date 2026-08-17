"""Layer 2a: LSB mantissa entropy + monobit/runs randomness + localization.

TODO(S4): for each tensor, extract low-k mantissa bit-planes (k=1..4, configurable)
into a bitstream; compute normalized Shannon entropy per bit-plane; run NIST-style
monobit + runs tests (numpy only, no scipy hard dep); sliding-window entropy across
the flattened tensor to localize suspicious regions (start, end). Emit Findings and
populate TensorReport.tensor_risk using assay.config thresholds.
"""

from __future__ import annotations

# TODO(S4): analyze_entropy(tensor_info, array) -> list[Finding]
