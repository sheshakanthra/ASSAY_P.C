"""Layer 2a: LSB mantissa entropy + monobit/runs randomness + localization.

`analyze_entropy(info, array, config=None) -> list[Finding]` looks for payload
bytes hidden in the low mantissa bit-planes of float32 tensors.

Calibration note (why this isn't a naive entropy threshold): natural, trained
float32 weights already have near-maximal Shannon entropy in their low mantissa
bits — empirically, whole-tensor entropy and even whole-tensor NIST monobit/runs
p-values on our own S2 clean baseline routinely hit the same range poisoned
regions do (measured directly against fixtures/models/clean_cnn.safetensors;
one clean tensor's whole-tensor monobit p-value was ~1e-251, purely from natural
structure). A fixed entropy/monobit/runs threshold applied globally therefore
cannot separate clean from poisoned without guaranteed false positives on real
models — floating-point rounding noise in the lowest mantissa bits is
information-theoretically close to uniform.

What *does* separate them cleanly (verified against the same fixtures): the
NIST "longest run of identical bits" test, applied per bit-plane across the
whole flattened tensor. An LSB payload that overwrites a contiguous byte range
(as our fixture generator does) produces one implausibly long run of a constant
bit somewhere in that range — 60+ identical bits in a row has probability
~n·2⁻ᴸ under natural noise, astronomically unlikely by chance, and the run's
own [start, end) *is* the localized region, precisely. This is the layer's
primary, gating test. Whole-tensor entropy and the region-restricted monobit/
runs tests are still computed and attached as corroborating evidence (per
CLAUDE.md's spec), but do not independently gate a finding.

Known limitation (see docs/THREAT_MODEL.md, S13): a payload that is itself
statistically indistinguishable from natural mantissa noise (e.g. genuinely
random/encrypted bytes with no long constant run) cannot be caught by this
layer alone — empirically confirmed against the S2 `lsb_random_blob` fixture,
whose injected bytes are byte-for-byte as random-looking as the tensor's own
natural noise. Layer 2b (signature sweep) and Layer 2c (distribution anomaly)
are complementary; the scoring engine (S6) is expected to reason about this
"high entropy, no other evidence" case honestly rather than claim certainty.
"""

from __future__ import annotations

import math

import numpy as np

from assay.config import DEFAULT_CONFIG, Config
from assay.models import Finding, Severity, TensorInfo


def analyze_entropy(info: TensorInfo, array: np.ndarray, config: Config | None = None) -> list[Finding]:
    """Scan one tensor's low mantissa bit-planes for LSB-steganography evidence.

    Only defined for float32 arrays in this MVP (matches the S2 fixture
    generator's own scope; fp16/bf16 support is deferred to S13 per CLAUDE.md).
    """
    config = config or DEFAULT_CONFIG
    if array.dtype != np.float32 or array.size < 8:
        return []

    thresholds = config.thresholds
    flat = array.reshape(-1)
    n = flat.size
    u32 = flat.view(np.uint32)

    findings: list[Finding] = []
    for bit in range(thresholds.lsb_bit_planes):
        bits = _bit_plane(u32, bit)
        if bit < thresholds.lsb_run_test_min_bit:
            continue

        whole_entropy = _shannon_entropy(bits)
        for value in (0, 1):
            length, start, end = _longest_run(bits, value)
            if length == 0:
                continue
            p = _run_pvalue(n, length)
            if p >= thresholds.run_test_p_value:
                continue

            region = bits[start:end]
            region_entropy = _shannon_entropy(region)
            region_monobit_p = _monobit_p(region)
            region_runs_p = _runs_p(region)
            window_profile = _sliding_window_entropy(
                bits, thresholds.entropy_window_size, thresholds.entropy_window_stride
            )
            local_note = _local_entropy_note(
                window_profile, thresholds.entropy_window_size, thresholds.entropy_window_stride, start, end
            )

            severity = Severity.CRITICAL if region_entropy >= thresholds.entropy_malicious else Severity.HIGH
            findings.append(
                Finding(
                    layer="entropy",
                    rule="lsb_run_anomaly",
                    severity=severity,
                    tensor=info.name,
                    detail=(
                        f"bit-plane {bit}: a run of {length} consecutive {value}-bits at elements "
                        f"[{start},{end}) is implausible under natural floating-point noise "
                        f"(p={p:.3e}). whole-tensor entropy={whole_entropy:.3f}, region "
                        f"entropy={region_entropy:.3f}, region monobit p={region_monobit_p:.3e}, "
                        f"region runs p={region_runs_p:.3e}{local_note}"
                    ),
                    value=p,
                    threshold=thresholds.run_test_p_value,
                )
            )

    return findings


# ---------------------------------------------------------------------------
# bit-plane extraction + statistics
# ---------------------------------------------------------------------------


def _bit_plane(u32: np.ndarray, bit: int) -> np.ndarray:
    return ((u32 >> bit) & 1).astype(np.uint8)


def _shannon_entropy(bits: np.ndarray) -> float:
    """Normalized (0-1) Shannon entropy of a binary bitstream."""
    n = bits.size
    if n == 0:
        return 0.0
    p1 = float(bits.mean())
    if p1 <= 0.0 or p1 >= 1.0:
        return 0.0
    return -(p1 * math.log2(p1) + (1 - p1) * math.log2(1 - p1))


def _monobit_p(bits: np.ndarray) -> float:
    """NIST SP 800-22 monobit (frequency) test p-value."""
    n = bits.size
    if n == 0:
        return 1.0
    s = int(np.sum(2 * bits.astype(np.int64) - 1))
    stat = abs(s) / math.sqrt(n)
    return math.erfc(stat / math.sqrt(2))


def _runs_p(bits: np.ndarray) -> float:
    """NIST SP 800-22 runs (transition-count) test p-value."""
    n = bits.size
    if n < 2:
        return 1.0
    pi = float(bits.mean())
    if abs(pi - 0.5) >= 2 / math.sqrt(n):
        return 0.0
    vobs = 1 + int(np.sum(bits[1:] != bits[:-1]))
    denom = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    if denom == 0:
        return 0.0
    z = (vobs - 2 * n * pi * (1 - pi)) / denom
    return math.erfc(abs(z) / math.sqrt(2))


def _longest_run(bits: np.ndarray, value: int) -> tuple[int, int, int]:
    """Longest contiguous run of `value` (0 or 1) in `bits`. Returns (length, start, end)."""
    target = bits == value
    if not target.any():
        return 0, -1, -1
    padded = np.concatenate(([0], target.astype(np.int8), [0]))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    lengths = ends - starts
    idx = int(np.argmax(lengths))
    return int(lengths[idx]), int(starts[idx]), int(ends[idx])


def _run_pvalue(n: int, length: int) -> float:
    """P(a specific bit value's longest run >= `length` in `n` fair coin flips).

    Standard large-n approximation (expected count of run-start positions,
    n * 2**-length); slightly conservative (over-estimates p), which is the
    safe direction for a detector.
    """
    if length <= 0:
        return 1.0
    return min(n * (2.0**-length), 1.0)


def _sliding_window_entropy(bits: np.ndarray, window: int, stride: int) -> np.ndarray:
    """Per-window Shannon entropy profile across the flattened bit-plane."""
    n = bits.size
    if n < window or window <= 0:
        return np.array([])
    starts = range(0, n - window + 1, max(stride, 1))
    return np.array([_shannon_entropy(bits[s : s + window]) for s in starts])


def _local_entropy_note(profile: np.ndarray, window: int, stride: int, start: int, end: int) -> str:
    """Describe the sliding-window entropy immediately around a flagged run,
    as supplementary localization evidence alongside the run's own bounds."""
    if profile.size == 0:
        return ""
    window_starts = list(range(0, profile.size * stride, stride))[: profile.size]
    nearby = [
        profile[i]
        for i, ws in enumerate(window_starts)
        if ws < end + window and ws + window > max(start - window, 0)
    ]
    if not nearby:
        return ""
    nearby_arr = np.array(nearby)
    baseline = float(np.median(profile))
    return f", nearby window entropy {nearby_arr.min():.3f}-{nearby_arr.max():.3f} (tensor median {baseline:.3f})"
