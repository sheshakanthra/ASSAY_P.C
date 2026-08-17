"""S0 smoke test: `scan` on a tiny dummy file returns a valid empty ScanReport."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

DUMMY = Path(__file__).parent / "data" / "dummy.bin"


def test_scan_dummy_file_prints_empty_clean_report():
    result = subprocess.run(
        [sys.executable, "-m", "assay", "scan", str(DUMMY)],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(result.stdout)
    assert payload["band"] == "clean"
    assert payload["risk_score"] == 0
    assert payload["tensor_reports"] == []
    assert payload["wrapper_findings"] == []
    assert payload["artifact"] == str(DUMMY)


def test_scan_missing_file_errors():
    result = subprocess.run(
        [sys.executable, "-m", "assay", "scan", "does/not/exist.bin"],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
