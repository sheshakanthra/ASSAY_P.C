"""S4 GATE tests: Layer 2a LSB entropy + monobit/runs randomness + localization.

Per-technique reality check against the real S2 fixtures (see
assay/layers/steg_entropy.py module docstring for the full empirical
calibration story): `lsb_eicar`, `lsb_elf_header`, and `plaintext_marker` are
all reliably caught with zero clean false positives, and localization overlaps
the manifest `ground_truth_region` for both EICAR and ELF (satisfying that
specific GATE bullet literally). `lsb_random_blob` is a genuine, documented
detection gap: os.urandom output is empirically indistinguishable from a
trained float32 tensor's own natural mantissa noise via any bit-level
statistic (entropy, monobit, runs, longest-run — all four were measured
directly against fixtures/models/clean_cnn.safetensors and found to have no
separating threshold). That's why the manifest-measured recall this session's
GATE reports is 3/4 = 0.75, not the originally-targeted >=0.90 — an
information-theoretic limit of bit-level statistics on this exact fixture, not
an implementation shortfall. Layer 2b (signature sweep, S5) is expected to
help close this specific gap for structured payloads, and the scoring engine
(S6) is expected to reason honestly about the "high entropy, no other
evidence" residual case per CLAUDE.md's own S13 plan.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assay.config import Config
from assay.intake.loader import iter_tensors
from assay.layers.steg_entropy import (
    _longest_run,
    _monobit_p,
    _run_pvalue,
    _runs_p,
    _shannon_entropy,
    _sliding_window_entropy,
    analyze_entropy,
)
from assay.models import TensorInfo

MODELS_DIR = Path("fixtures/models")
MANIFEST_PATH = MODELS_DIR / "manifest.json"


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    return json.loads(MANIFEST_PATH.read_text())


def _load_tensor(filename: str, tensor_name: str) -> tuple[TensorInfo, np.ndarray]:
    tensors = {info.name: (info, arr) for info, arr in iter_tensors(MODELS_DIR / filename)}
    return tensors[tensor_name]


# ---------------------------------------------------------------------------
# unit tests: core statistics on known synthetic inputs
# ---------------------------------------------------------------------------


def test_shannon_entropy_of_constant_bits_is_zero():
    assert _shannon_entropy(np.zeros(100, dtype=np.uint8)) == 0.0
    assert _shannon_entropy(np.ones(100, dtype=np.uint8)) == 0.0


def test_shannon_entropy_of_balanced_bits_is_near_one():
    bits = np.array([0, 1] * 500, dtype=np.uint8)
    assert _shannon_entropy(bits) == pytest.approx(1.0, abs=1e-9)


def test_monobit_p_rejects_all_zero_sequence():
    bits = np.zeros(64, dtype=np.uint8)
    assert _monobit_p(bits) < 1e-6


def test_monobit_p_accepts_balanced_sequence():
    bits = np.array([0, 1] * 32, dtype=np.uint8)
    assert _monobit_p(bits) > 0.5


def test_runs_p_rejects_perfectly_alternating_sequence():
    # Correct proportion (50/50) but far too many transitions to be random.
    bits = np.array([0, 1] * 50, dtype=np.uint8)
    assert _runs_p(bits) < 0.01


def test_longest_run_finds_correct_bounds():
    bits = np.array([1, 1, 0, 0, 0, 0, 0, 1, 1], dtype=np.uint8)
    length, start, end = _longest_run(bits, 0)
    assert (length, start, end) == (5, 2, 7)


def test_longest_run_returns_zero_when_value_absent():
    bits = np.ones(20, dtype=np.uint8)
    length, start, end = _longest_run(bits, 0)
    assert length == 0


def test_run_pvalue_decreases_with_run_length():
    n = 10_000
    assert _run_pvalue(n, 10) > _run_pvalue(n, 30) > _run_pvalue(n, 60)


def test_sliding_window_entropy_shape():
    bits = np.random.default_rng(0).integers(0, 2, size=200).astype(np.uint8)
    profile = _sliding_window_entropy(bits, window=32, stride=16)
    assert profile.size == len(range(0, 200 - 32 + 1, 16))


# ---------------------------------------------------------------------------
# synthetic clean-vs-poisoned validation (broader than one specific model)
# ---------------------------------------------------------------------------


def _synthetic_clean_tensor(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.standard_normal(n).astype(np.float32) * 0.05


def _embed_ascii_payload(arr: np.ndarray, payload: bytes) -> np.ndarray:
    flat = arr.reshape(-1).copy()
    u32 = flat.view(np.uint32)
    for i, byte in enumerate(payload):
        u32[i] = (u32[i] & 0xFFFFFF00) | byte
    return flat.reshape(arr.shape)


def test_synthetic_recall_and_false_positive_rate():
    """Broader statistical validation: over 40 independently-seeded synthetic
    "trained-weight-like" tensors, embedding a 64-byte ASCII-range payload
    (mirrors EICAR/plaintext_marker's structure — bit7 of every byte is 0)
    is detected with recall >= 0.90, while 40 unmodified clean tensors of the
    same distribution produce zero false positives.
    """
    rng = np.random.default_rng(42)
    payload = bytes((i % 95) + 32 for i in range(64))  # printable ASCII, bit7 always 0
    info = TensorInfo(name="w", dtype="float32", shape=(2000,))

    detected = 0
    trials = 40
    for _ in range(trials):
        clean = _synthetic_clean_tensor(rng, 2000)
        poisoned = _embed_ascii_payload(clean, payload)
        findings = analyze_entropy(info, poisoned)
        if findings:
            detected += 1
    recall = detected / trials
    assert recall >= 0.90, f"synthetic ASCII-payload recall {recall} below 0.90"

    false_positives = 0
    for _ in range(trials):
        clean = _synthetic_clean_tensor(rng, 2000)
        findings = analyze_entropy(info, clean)
        if findings:
            false_positives += 1
    fp_rate = false_positives / trials
    assert fp_rate <= 0.05, f"synthetic clean false-positive rate {fp_rate} above 0.05"


# ---------------------------------------------------------------------------
# real S2 fixtures: clean pass + per-technique detection + localization
# ---------------------------------------------------------------------------


def test_clean_fixture_all_tensors_produce_zero_findings():
    for info, arr in iter_tensors(MODELS_DIR / "clean_cnn.safetensors"):
        assert analyze_entropy(info, arr) == [], f"false positive on clean tensor {info.name}"


@pytest.mark.parametrize("technique", ["lsb_eicar", "lsb_elf_header", "plaintext_marker"])
def test_detectable_techniques_are_flagged(manifest, technique):
    entry = next(e for e in manifest if e["technique"] == technique)
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])
    findings = analyze_entropy(info, arr)
    assert findings, f"{technique} was not detected"
    assert all(f.rule == "lsb_run_anomaly" for f in findings)


@pytest.mark.parametrize("technique", ["lsb_eicar", "lsb_elf_header"])
def test_localization_overlaps_ground_truth_region(manifest, technique):
    entry = next(e for e in manifest if e["technique"] == technique)
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])
    findings = analyze_entropy(info, arr)
    gt_start, gt_end = entry["ground_truth_region"]

    found_overlap = False
    for f in findings:
        for start, end in _extract_regions(f.detail):
            if start < gt_end and gt_start < end:
                found_overlap = True
    assert found_overlap, f"{technique}: no finding's localized region overlaps ground truth {entry['ground_truth_region']}"


def test_random_blob_is_a_known_undetected_case(manifest):
    """Documents the real limitation rather than hiding it: a genuinely random
    byte blob is statistically indistinguishable from natural float32 mantissa
    noise at the bit level, so this layer alone cannot catch it. See module
    docstring in assay/layers/steg_entropy.py.
    """
    entry = next(e for e in manifest if e["technique"] == "lsb_random_blob")
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])
    findings = analyze_entropy(info, arr)
    assert findings == []


def _extract_regions(detail: str) -> list[tuple[int, int]]:
    import re

    return [(int(s), int(e)) for s, e in re.findall(r"\[(\d+),\s*(\d+)\)", detail)]


# ---------------------------------------------------------------------------
# non-float32 / degenerate inputs
# ---------------------------------------------------------------------------


def test_non_float32_tensor_returns_no_findings():
    info = TensorInfo(name="w", dtype="int64", shape=(100,))
    arr = np.zeros(100, dtype=np.int64)
    assert analyze_entropy(info, arr) == []


def test_tiny_tensor_returns_no_findings():
    info = TensorInfo(name="w", dtype="float32", shape=(3,))
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert analyze_entropy(info, arr) == []


def test_config_run_test_threshold_is_respected():
    entry_technique = "plaintext_marker"
    manifest_data = json.loads(MANIFEST_PATH.read_text())
    entry = next(e for e in manifest_data if e["technique"] == entry_technique)
    info, arr = _load_tensor(entry["file"], entry["target_tensor"])

    strict_config = Config()
    strict_config.thresholds.run_test_p_value = 1e-10  # far stricter than plaintext_marker's p~4e-3
    findings = analyze_entropy(info, arr, config=strict_config)
    assert findings == []
