"""Layer 2c: per-tensor distribution anomaly + sliding-window localization.

`analyze_distribution(info, array, baseline=None, config=None) -> list[Finding]`
computes per-tensor value-domain statistics — mean, std, excess kurtosis,
exact-zero fraction, denormal fraction, low-byte histogram — and flags
localized blocks whose statistics deviate from the *rest of that same tensor*
(a robust median/MAD z-score over non-overlapping blocks, self-referential —
no cross-model context required). When a `baseline` built by `build_baseline()`
is supplied, it additionally compares the tensor's whole-tensor stats against
its own layer-type peers (weight tensors vs bias tensors) elsewhere in the
model.

This is the value-domain counterpart to Layer 2a's bit-domain analysis, aimed
at EvilModel-style attacks that overwrite whole "atrophied" (near-zero,
rarely-updated) weights with payload bytes reinterpreted as float32 — that
produces wild mean/std/kurtosis/denormal swings in the overwritten block,
which is what this layer is built to catch.

Known limitation (see docs/THREAT_MODEL.md, S13; and steg_entropy.py's own
docstring for the bit-domain analog): a pure low-mantissa-byte LSB
substitution — the S2 `lsb_random_blob` fixture's technique — perturbs the
actual float32 *value* by a relative ~2e-5 or less (verified directly against
fixtures/models/: clean vs. poisoned region mean/std are identical to 6
decimal places). No value-domain statistic can see a change that small; this
layer, like Layer 2a, cannot catch that specific fixture. It is a different
attack shape (value-preserving vs. value-overwriting) that this layer isn't
designed to see, not a tuning gap.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from assay.config import DEFAULT_CONFIG, Config
from assay.models import Finding, Severity, TensorInfo

_DENORMAL_MIN = np.finfo(np.float32).tiny  # smallest positive normal float32 (~1.175e-38)


@dataclass
class TensorStats:
    """Per-tensor (or per-block) value-domain statistics."""

    mean: float
    std: float
    kurtosis: float  # excess kurtosis (0 for a normal distribution)
    exact_zero_fraction: float
    denormal_fraction: float
    size: int


def compute_tensor_stats(array: np.ndarray) -> TensorStats:
    flat = array.reshape(-1).astype(np.float64)
    n = flat.size
    if n == 0:
        return TensorStats(0.0, 0.0, 0.0, 0.0, 0.0, 0)

    mean = float(flat.mean())
    std = float(flat.std())
    if std > 0:
        kurtosis = float(np.mean((flat - mean) ** 4) / (std**4) - 3.0)
    else:
        kurtosis = 0.0

    exact_zero_fraction = float(np.sum(flat == 0.0)) / n
    abs_flat = np.abs(flat)
    denormal_mask = (abs_flat > 0) & (abs_flat < _DENORMAL_MIN)
    denormal_fraction = float(np.sum(denormal_mask)) / n

    return TensorStats(mean, std, kurtosis, exact_zero_fraction, denormal_fraction, n)


def low_byte_histogram(array: np.ndarray) -> np.ndarray:
    """256-bin histogram of each element's low mantissa byte (float32 only)."""
    if array.dtype != np.float32 or array.size == 0:
        return np.zeros(256, dtype=np.int64)
    low_bytes = array.reshape(-1).view(np.uint32) & 0xFF
    return np.bincount(low_bytes.astype(np.uint8), minlength=256).astype(np.int64)


def _infer_layer_type(name: str) -> str:
    lname = name.lower()
    if lname.endswith((".bias", "_bias")):
        return "bias"
    if lname.endswith((".weight", "_weight")):
        return "weight"
    return "other"


def build_baseline(tensors: Iterable[tuple[TensorInfo, np.ndarray]]) -> dict[str, dict[str, float]]:
    """Aggregate per-layer-type (weight/bias/other) baseline stats across a
    whole model's tensors, for the cross-tensor comparison in analyze_distribution.
    """
    grouped: dict[str, list[TensorStats]] = {}
    for info, array in tensors:
        if array.dtype != np.float32 or array.size == 0:
            continue
        grouped.setdefault(_infer_layer_type(info.name), []).append(compute_tensor_stats(array))

    baseline: dict[str, dict[str, float]] = {}
    for layer_type, stats_list in grouped.items():
        means = np.array([s.mean for s in stats_list])
        stds = np.array([s.std for s in stats_list])
        kurts = np.array([s.kurtosis for s in stats_list])
        baseline[layer_type] = {
            "n_tensors": len(stats_list),
            "mean_of_mean": float(means.mean()),
            "std_of_mean": float(means.std()),
            "mean_of_std": float(stds.mean()),
            "std_of_std": float(stds.std()),
            "mean_of_kurtosis": float(kurts.mean()),
            "std_of_kurtosis": float(kurts.std()),
        }
    return baseline


def analyze_distribution(
    info: TensorInfo,
    array: np.ndarray,
    baseline: dict[str, dict[str, float]] | None = None,
    config: Config | None = None,
) -> list[Finding]:
    config = config or DEFAULT_CONFIG
    if array.dtype != np.float32 or array.size < 16:
        return []

    thresholds = config.thresholds
    findings: list[Finding] = []

    findings.extend(_localized_block_findings(info, array, thresholds))
    if baseline is not None:
        findings.extend(_layer_type_baseline_findings(info, array, baseline, thresholds))

    return findings


def _blocks(flat: np.ndarray, n_blocks: int) -> list[tuple[int, int, TensorStats]]:
    n = flat.size
    block_size = max(n // n_blocks, 1)
    blocks = []
    for start in range(0, n, block_size):
        chunk = flat[start : start + block_size]
        if chunk.size < 4:
            continue
        blocks.append((start, start + chunk.size, compute_tensor_stats(chunk)))
    return blocks


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = np.median(values)
    mad = np.median(np.abs(values - median)) * 1.4826 + 1e-12
    return np.abs(values - median) / mad


def _localized_block_findings(info: TensorInfo, array: np.ndarray, thresholds) -> list[Finding]:
    flat = array.reshape(-1)
    blocks = _blocks(flat, thresholds.distribution_block_count)
    if len(blocks) < 4:
        return []

    findings: list[Finding] = []

    # NaN/Inf first, and exclude those blocks from the z-score comparisons below:
    # a NaN block's mean/std/kurtosis are themselves NaN, which silently compares
    # False against any threshold and would otherwise hide the anomaly entirely —
    # and raw bytes reinterpreted as float32 (an exponent field of all-1-bits)
    # produce exactly this, so it's also a highly diagnostic signal on its own.
    finite_blocks = []
    for start, end, stat in blocks:
        chunk = flat[start:end]
        n_bad = int(np.sum(~np.isfinite(chunk)))
        if n_bad > 0:
            findings.append(
                Finding(
                    layer="distribution",
                    rule="nan_or_inf_values",
                    severity=Severity.CRITICAL,
                    tensor=info.name,
                    detail=(
                        f"block [{start},{end}) contains {n_bad} non-finite value(s) "
                        "(NaN/Inf) — never produced by trained weights, consistent with "
                        "raw bytes reinterpreted as float32"
                    ),
                    value=float(n_bad),
                    threshold=0.0,
                )
            )
        else:
            finite_blocks.append((start, end, stat))

    # Only mean/std feed the localized z-score scan: per-block kurtosis (a 4th
    # moment) has high sampling variance at these block sizes and produced
    # spurious outliers even on purely synthetic clean data in testing: keep
    # kurtosis computed and reported (TensorStats, layer-type baseline) but not
    # gating here.
    if len(finite_blocks) >= 4:
        for metric_name in ("mean", "std"):
            values = np.array([b[2].mean if metric_name == "mean" else b[2].std for b in finite_blocks])
            z_scores = _robust_z(values)
            for (start, end, stat), z in zip(finite_blocks, z_scores, strict=True):
                if z >= thresholds.kurtosis_z_score:
                    findings.append(
                        Finding(
                            layer="distribution",
                            rule="localized_distribution_anomaly",
                            severity=Severity.MEDIUM,
                            tensor=info.name,
                            detail=(
                                f"block [{start},{end}) has {metric_name}={getattr(stat, metric_name):.6g}, "
                                f"z={z:.2f} vs. this tensor's own other blocks (n={len(finite_blocks)})"
                            ),
                            value=float(z),
                            threshold=thresholds.kurtosis_z_score,
                        )
                    )

    for start, end, stat in blocks:
        if stat.denormal_fraction >= thresholds.denormal_fraction:
            findings.append(
                Finding(
                    layer="distribution",
                    rule="denormal_block",
                    severity=Severity.HIGH,
                    tensor=info.name,
                    detail=(
                        f"block [{start},{end}) denormal-float fraction {stat.denormal_fraction:.4f} "
                        f"exceeds {thresholds.denormal_fraction} — consistent with raw bytes "
                        "reinterpreted as float32 rather than trained weights"
                    ),
                    value=stat.denormal_fraction,
                    threshold=thresholds.denormal_fraction,
                )
            )

    return findings


def _layer_type_baseline_findings(
    info: TensorInfo, array: np.ndarray, baseline: dict[str, dict[str, float]], thresholds
) -> list[Finding]:
    layer_type = _infer_layer_type(info.name)
    peer = baseline.get(layer_type)
    if peer is None or peer["n_tensors"] < 3:
        return []  # not enough peers in this model to form a meaningful baseline

    whole = compute_tensor_stats(array)
    findings: list[Finding] = []
    for metric_name, peer_mean_key, peer_std_key in (
        ("mean", "mean_of_mean", "std_of_mean"),
        ("std", "mean_of_std", "std_of_std"),
        ("kurtosis", "mean_of_kurtosis", "std_of_kurtosis"),
    ):
        peer_std = peer[peer_std_key]
        if peer_std <= 0:
            continue
        value = getattr(whole, metric_name)
        z = abs(value - peer[peer_mean_key]) / peer_std
        if z >= thresholds.kurtosis_z_score:
            findings.append(
                Finding(
                    layer="distribution",
                    rule="layer_type_baseline_outlier",
                    severity=Severity.MEDIUM,
                    tensor=info.name,
                    detail=(
                        f"whole-tensor {metric_name}={value:.6g} is z={z:.2f} from the model's other "
                        f"'{layer_type}' tensors (n={peer['n_tensors']}, baseline "
                        f"{peer_mean_key}={peer[peer_mean_key]:.6g})"
                    ),
                    value=float(z),
                    threshold=thresholds.kurtosis_z_score,
                )
            )
    return findings
