"""Scoring engine — defines the scoring contract (Manual-mode session, S6).

TODO(S6): `score(scan_inputs) -> ScanReport`. Deterministic weighted aggregation
of wrapper + tensor findings -> risk_score (0-100) -> band
(CLEAN < band_clean_max <= SUSPICIOUS < band_suspicious_max <= MALICIOUS), using
weights/thresholds from assay.config. Build `explanations`: ordered list of the
top contributing findings with value vs threshold and a plain-English reason.
"""

from __future__ import annotations

# TODO(S6): score(scan_inputs) -> ScanReport
