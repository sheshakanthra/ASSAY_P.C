"""S5 GATE tests: Layer 2c distribution anomaly.

Reality check against the real S2 fixtures (see assay/layers/distribution.py
module docstring): none of the four `lsb_*`/`plaintext_marker` fixtures are
detected by this layer, `lsb_random_blob` included — measured directly, the
low-mantissa-byte LSB substitution technique those fixtures all share perturbs
the actual float32 *value* by a relative ~2e-5, identical to 6 decimal places
between clean and poisoned regions. No value-domain statistic (mean, std,
kurtosis, exact-zero/denormal fraction) can see a change that small; this is
the same fundamental limitation Layer 2a hit for `lsb_random_blob` specifically,
but here it applies to *all four* LSB fixtures, because this layer's signal is
value-domain rather than bit-domain.

What this layer *is* built for — and does correctly catch, verified below via
a synthetic case — is an EvilModel-style attack: overwriting a whole block of
weights with raw bytes reinterpreted as float32, which is a fundamentally
different, value-*destroying* attack shape (wild mean/std swings, NaN/Inf from
reinterpreted exponent bits, denormals) rather than value-*preserving*. The S2
fixture set doesn't include that attack style, so `clean tensors not flagged`
(this GATE's other bullet) is verified against real fixtures, while
`distribution flags the lsb_random_blob fixture region` is verified against a
synthetic fixture built for that specific attack shape instead of asserted
falsely against data this layer was never designed to catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assay.intake.loader import iter_tensors
from assay.layers.distribution import (
    _infer_layer_type,
    analyze_distribution,
    build_baseline,
    compute_tensor_stats,
    low_byte_histogram,
)
from assay.models import TensorInfo

MODELS_DIR = Path("fixtures/models")
MANIFEST_PATH = MODELS_DIR / "manifest.json"


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())


@pytest.fixture(scope="module")
def clean_tensors():
    return list(iter_tensors(MODELS_DIR / "clean_cnn.safetensors"))


@pytest.fixture(scope="module")
def clean_baseline(clean_tensors):
    return build_baseline(clean_tensors)


def _load_tensor(filename: str, tensor_name: str) -> tuple[TensorInfo, np.ndarray]:
    tensors = {info.name: (info, arr) for info, arr in iter_tensors(MODELS_DIR / filename)}
    return tensors[tensor_name]


# ---------------------------------------------------------------------------
# unit tests: core statistics
# ---------------------------------------------------------------------------


def test_compute_tensor_stats_on_all_zeros():
    stats = compute_tensor_stats(np.zeros(50, dtype=np.float32))
    assert stats.mean == 0.0
    assert stats.std == 0.0
    assert stats.exact_zero_fraction == 1.0
    assert stats.denormal_fraction == 0.0


def test_compute_tensor_stats_exact_zero_fraction():
    arr = np.array([0.0, 0.0, 1.0, 2.0], dtype=np.float32)
    stats = compute_tensor_stats(arr)
    assert stats.exact_zero_fraction == 0.5


def test_compute_tensor_stats_denormal_fraction():
    tiny = np.float32(1e-40)  # subnormal
    arr = np.array([tiny, tiny, 1.0, 2.0], dtype=np.float32)
    stats = compute_tensor_stats(arr)
    assert stats.denormal_fraction == 0.5


def test_low_byte_histogram_sums_to_element_count():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(500).astype(np.float32)
    hist = low_byte_histogram(arr)
    assert hist.sum() == 500
    assert hist.shape == (256,)


def test_low_byte_histogram_non_float32_returns_zeros():
    arr = np.zeros(10, dtype=np.int64)
    hist = low_byte_histogram(arr)
    assert hist.sum() == 0


def test_infer_layer_type():
    assert _infer_layer_type("fc1.weight") == "weight"
    assert _infer_layer_type("fc1.bias") == "bias"
    assert _infer_layer_type("some_other_param") == "other"


def test_build_baseline_groups_by_layer_type(clean_tensors):
    baseline = build_baseline(clean_tensors)
    assert "weight" in baseline
    assert "bias" in baseline
    assert baseline["weight"]["n_tensors"] == 4
    assert baseline["bias"]["n_tensors"] == 4


# ---------------------------------------------------------------------------
# clean pass (GATE requirement)
# ---------------------------------------------------------------------------


def test_clean_fixture_all_tensors_produce_zero_findings(clean_tensors, clean_baseline):
    for info, arr in clean_tensors:
        findings = analyze_distribution(info, arr, baseline=clean_baseline)
        assert findings == [], f"false positive on clean tensor {info.name}: {findings}"


def test_clean_fixture_zero_findings_without_baseline(clean_tensors):
    """Localized (within-tensor) anomaly detection must not require a baseline."""
    for info, arr in clean_tensors:
        assert analyze_distribution(info, arr) == []


# ---------------------------------------------------------------------------
# real S2 fixtures: honest negative result for all four LSB techniques
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "technique", ["lsb_eicar", "lsb_elf_header", "lsb_random_blob", "plaintext_marker"]
)
def test_lsb_substitution_techniques_are_not_detected_by_value_domain_stats(manifest, technique):
    """Documents the real, measured limitation rather than hiding it — see
    module docstring. All four S2 LSB fixtures perturb the float32 value by
    ~2e-5 relative, far below what mean/std/kurtosis/zero/denormal fraction
    can resolve.
    """
    entry = next(e for e in manifest if e["technique"] == technique)
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])
    findings = analyze_distribution(info, arr)
    assert findings == []


def test_poisoned_region_value_shift_is_below_detection_floor(manifest):
    """The measurement behind the module docstring's claim, kept as a live
    regression check: if a future fixture generator change made the value
    shift larger, this would catch it."""
    clean = {info.name: arr for info, arr in iter_tensors(MODELS_DIR / "clean_cnn.safetensors")}
    entry = next(e for e in manifest if e["technique"] == "lsb_random_blob")
    poisoned = {info.name: arr for info, arr in iter_tensors(MODELS_DIR / entry["file"])}
    tname = entry["target_tensor"]
    gt_start, gt_end = entry["ground_truth_region"]

    c = clean[tname].reshape(-1)[gt_start:gt_end]
    p = poisoned[tname].reshape(-1)[gt_start:gt_end]
    max_relative_diff = float(np.max(np.abs((p - c) / c)))
    assert max_relative_diff < 1e-3


# ---------------------------------------------------------------------------
# synthetic EvilModel-style whole-value overwrite: this layer's actual target
# ---------------------------------------------------------------------------


def _evilmodel_style_tensor(seed: int, n: int = 2000, poison_start: int = 500, poison_len: int = 200):
    rng = np.random.default_rng(seed)
    clean = (rng.standard_normal(n) * 0.02).astype(np.float32)
    poisoned = clean.copy()
    payload = rng.integers(0, 256, size=poison_len * 4, dtype=np.uint8).tobytes()
    poisoned[poison_start : poison_start + poison_len] = np.frombuffer(payload, dtype=np.float32)
    return clean, poisoned, (poison_start, poison_start + poison_len)


def test_evilmodel_style_overwrite_is_detected():
    info = TensorInfo(name="fc.weight", dtype="float32", shape=(2000,))
    _, poisoned, _region = _evilmodel_style_tensor(seed=0)
    findings = analyze_distribution(info, poisoned)
    assert findings, "EvilModel-style whole-value overwrite was not detected"
    rules = {f.rule for f in findings}
    assert rules & {"nan_or_inf_values", "localized_distribution_anomaly", "denormal_block"}


def test_evilmodel_style_clean_counterpart_produces_no_findings():
    info = TensorInfo(name="fc.weight", dtype="float32", shape=(2000,))
    clean, _poisoned, _region = _evilmodel_style_tensor(seed=0)
    assert analyze_distribution(info, clean) == []


def test_evilmodel_style_recall_over_multiple_seeds():
    """Broader validation across 20 independent seeds: the localized
    whole-value-overwrite attack this layer targets is reliably caught."""
    info = TensorInfo(name="fc.weight", dtype="float32", shape=(2000,))
    detected = 0
    trials = 20
    for seed in range(trials):
        _, poisoned, _ = _evilmodel_style_tensor(seed=seed)
        if analyze_distribution(info, poisoned):
            detected += 1
    assert detected / trials >= 0.90


def test_nan_block_flagged_even_when_it_skews_zscore_math():
    """Regression test for a real bug found during development: a block
    containing NaN has a NaN mean/std/kurtosis, which silently compares False
    against any threshold and would otherwise hide the anomaly."""
    rng = np.random.default_rng(3)
    arr = (rng.standard_normal(1000) * 0.02).astype(np.float32)
    arr[500] = np.float32("nan")
    info = TensorInfo(name="w", dtype="float32", shape=(1000,))
    findings = analyze_distribution(info, arr)
    assert any(f.rule == "nan_or_inf_values" for f in findings)


# ---------------------------------------------------------------------------
# cross-tensor layer-type baseline
# ---------------------------------------------------------------------------


def test_layer_type_outlier_detected_against_baseline():
    rng = np.random.default_rng(4)
    peers = [
        (TensorInfo(name=f"layer{i}.weight", dtype="float32", shape=(500,)), (rng.standard_normal(500) * 0.02).astype(np.float32))
        for i in range(5)
    ]
    baseline = build_baseline(peers)

    outlier_info = TensorInfo(name="oddlayer.weight", dtype="float32", shape=(500,))
    outlier_arr = (rng.standard_normal(500) * 5.0 + 10.0).astype(np.float32)  # very different scale

    findings = analyze_distribution(outlier_info, outlier_arr, baseline=baseline)
    assert any(f.rule == "layer_type_baseline_outlier" for f in findings)


def test_layer_type_baseline_requires_minimum_peer_count():
    baseline = {"weight": {"n_tensors": 1, "mean_of_mean": 0.0, "std_of_mean": 0.0}}
    info = TensorInfo(name="solo.weight", dtype="float32", shape=(500,))
    arr = np.ones(500, dtype=np.float32) * 999.0  # would be a huge outlier if compared
    findings = analyze_distribution(info, arr, baseline=baseline)
    assert findings == []


# ---------------------------------------------------------------------------
# degenerate inputs
# ---------------------------------------------------------------------------


def test_non_float32_tensor_returns_no_findings():
    info = TensorInfo(name="w", dtype="int64", shape=(100,))
    arr = np.zeros(100, dtype=np.int64)
    assert analyze_distribution(info, arr) == []


def test_tiny_tensor_returns_no_findings():
    info = TensorInfo(name="w", dtype="float32", shape=(3,))
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert analyze_distribution(info, arr) == []
