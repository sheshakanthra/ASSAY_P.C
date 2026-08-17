"""Config — thresholds + scoring weights, TOML-overridable.

All numeric thresholds used by the detection layers and the scoring engine
live here so they are documented in one place and tunable without touching
layer code. Values below are placeholders; layers S3-S6 will tune them
against the fixture manifest.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Thresholds:
    """Detection thresholds, one field per layer rule. Placeholders — tuned in S4-S6."""

    # Layer 1: wrapper/opcode/archive analysis (wrapper.py)
    safetensors_header_max_bytes: int = 100_000  # flag implausibly large safetensors JSON headers

    # Layer 2a: LSB entropy / randomness (steg_entropy.py)
    entropy_suspicious: float = 0.85  # normalized Shannon entropy of LSB plane, 0-1
    entropy_malicious: float = 0.95
    monobit_p_value: float = 0.01  # NIST monobit test rejection threshold
    runs_p_value: float = 0.01  # NIST runs test rejection threshold

    # Layer 2c: distribution anomaly (distribution.py)
    kurtosis_z_score: float = 3.0  # z-score vs layer-type baseline
    denormal_fraction: float = 0.01

    # Scoring bands (risk_score 0-100)
    band_clean_max: float = 20.0  # score < this -> CLEAN
    band_suspicious_max: float = 60.0  # this <= score < malicious -> SUSPICIOUS, else MALICIOUS


@dataclass
class ScoringWeights:
    """Per-layer contribution weights to the aggregate Model Risk Score. Placeholders — tuned in S6."""

    wrapper: float = 1.0
    entropy: float = 1.0
    signature: float = 1.5  # confirmed byte-signature matches weigh more than statistical anomaly
    distribution: float = 0.75

    severity_multiplier: dict[str, float] = field(
        default_factory=lambda: {
            "info": 0.0,
            "low": 0.25,
            "medium": 0.5,
            "high": 0.8,
            "critical": 1.0,
        }
    )


@dataclass
class Config:
    """Top-level, immutable-in-practice config object threaded through the pipeline."""

    thresholds: Thresholds = field(default_factory=Thresholds)
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    llm_provider: str = "null"  # "null" (default, offline) or "groq" (opt-in); see ASSAY_LLM env

    @classmethod
    def load(cls, path: str | Path | None = None) -> Config:
        """Load defaults, optionally overridden by a TOML file at `path`.

        TOML shape:
            [thresholds]
            entropy_suspicious = 0.85
            ...
            [weights]
            wrapper = 1.0
            ...
        Unknown keys are ignored; missing keys keep their default.
        """
        cfg = cls()
        if path is None:
            return cfg
        p = Path(path)
        if not p.exists():
            return cfg
        with p.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)
        thresholds_data = data.get("thresholds", {})
        for k, v in thresholds_data.items():
            if hasattr(cfg.thresholds, k):
                setattr(cfg.thresholds, k, v)
        weights_data = data.get("weights", {})
        for k, v in weights_data.items():
            if hasattr(cfg.weights, k):
                setattr(cfg.weights, k, v)
        if "llm_provider" in data:
            cfg.llm_provider = data["llm_provider"]
        return cfg


DEFAULT_CONFIG = Config()
