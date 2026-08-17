"""ScanReport -> JSON + HTML/MD.

TODO(S7): `to_json(report)` and `to_html(report)`. HTML: verdict header
(score + band), per-layer evidence tables, suspicious-tensor list with
localized regions, wrapper findings. Self-contained inline CSS per DESIGN.md
(dark HUD tokens).
"""

from __future__ import annotations

# TODO(S7): to_json(report: ScanReport) -> str; to_html(report: ScanReport) -> str
