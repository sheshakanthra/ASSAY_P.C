"""Round-trip the DATA CONTRACTS through to_dict()."""

from __future__ import annotations

from assay.models import Finding, RiskBand, ScanReport, Severity, TensorReport


def test_finding_to_dict_round_trip():
    f = Finding(
        layer="wrapper",
        rule="dangerous_global_ref",
        severity=Severity.HIGH,
        detail="GLOBAL opcode references os.system",
        tensor=None,
        value=1.0,
        threshold=0.0,
    )
    d = f.to_dict()
    assert d == {
        "layer": "wrapper",
        "rule": "dangerous_global_ref",
        "severity": "high",
        "tensor": None,
        "detail": "GLOBAL opcode references os.system",
        "value": 1.0,
        "threshold": 0.0,
    }


def test_tensor_report_to_dict_round_trip():
    finding = Finding(
        layer="entropy", rule="high_lsb_entropy", severity=Severity.MEDIUM, detail="entropy 0.97"
    )
    tr = TensorReport(
        name="layer1.weight",
        dtype="float32",
        shape=(4, 4),
        findings=[finding],
        tensor_risk=42.5,
    )
    d = tr.to_dict()
    assert d["name"] == "layer1.weight"
    assert d["shape"] == [4, 4]
    assert d["tensor_risk"] == 42.5
    assert len(d["findings"]) == 1
    assert d["findings"][0]["rule"] == "high_lsb_entropy"


def test_scan_report_to_dict_round_trip():
    report = ScanReport(
        artifact="model.safetensors",
        format="safetensors",
        tensor_reports=[],
        wrapper_findings=[],
        risk_score=0.0,
        band=RiskBand.CLEAN,
        explanations=[],
    )
    d = report.to_dict()
    assert d["artifact"] == "model.safetensors"
    assert d["format"] == "safetensors"
    assert d["tensor_reports"] == []
    assert d["wrapper_findings"] == []
    assert d["risk_score"] == 0.0
    assert d["band"] == "clean"
    assert d["explanations"] == []


def test_severity_and_riskband_values():
    assert Severity.CRITICAL.value == "critical"
    assert RiskBand.MALICIOUS.value == "malicious"
