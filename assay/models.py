"""DATA CONTRACTS — stable after S0.

Defines the shared vocabulary every layer, the scoring engine, and the report
renderer speak: `Severity`, `RiskBand` (enums) and `Finding`, `TensorInfo`,
`TensorReport`, `ScanReport` (data objects). Do not change these signatures
outside a Manual-mode session that updates every consumer (see CLAUDE.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class Severity(StrEnum):
    """Severity of a single Finding, low to high."""

    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RiskBand(StrEnum):
    """Overall verdict band for a ScanReport, derived from risk_score."""

    CLEAN = "clean"
    SUSPICIOUS = "suspicious"
    MALICIOUS = "malicious"


@dataclass
class Finding:
    """A single piece of evidence emitted by a detection layer.

    layer:     which layer produced this (e.g. "wrapper", "entropy", "signature", "distribution").
    rule:      short machine-stable rule id (e.g. "dangerous_global_ref").
    severity:  Severity enum.
    tensor:    tensor name this finding applies to, or None for wrapper/container-level findings.
    detail:    human-readable explanation.
    value:     the observed value that triggered the rule (e.g. measured entropy).
    threshold: the configured threshold it was compared against.
    """

    layer: str
    rule: str
    severity: Severity
    detail: str
    tensor: str | None = None
    value: float | None = None
    threshold: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer": self.layer,
            "rule": self.rule,
            "severity": self.severity.value,
            "tensor": self.tensor,
            "detail": self.detail,
            "value": self.value,
            "threshold": self.threshold,
        }


@dataclass
class TensorInfo:
    """Uniform description of a single tensor read from any supported format."""

    name: str
    dtype: str
    shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "dtype": self.dtype, "shape": list(self.shape)}


@dataclass
class TensorReport:
    """Per-tensor findings + aggregated per-tensor risk score (0-100)."""

    name: str
    dtype: str
    shape: tuple[int, ...]
    findings: list[Finding] = field(default_factory=list)
    tensor_risk: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "findings": [f.to_dict() for f in self.findings],
            "tensor_risk": self.tensor_risk,
        }


@dataclass
class ScanReport:
    """Top-level result of scanning one artifact."""

    artifact: str
    format: str
    tensor_reports: list[TensorReport] = field(default_factory=list)
    wrapper_findings: list[Finding] = field(default_factory=list)
    risk_score: float = 0.0
    band: RiskBand = RiskBand.CLEAN
    explanations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact": self.artifact,
            "format": self.format,
            "tensor_reports": [t.to_dict() for t in self.tensor_reports],
            "wrapper_findings": [f.to_dict() for f in self.wrapper_findings],
            "risk_score": self.risk_score,
            "band": self.band.value,
            "explanations": self.explanations,
        }
