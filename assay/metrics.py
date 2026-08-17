"""Precision/recall/confusion over a fixture manifest.

TODO(S6): extend to run the *whole* pipeline (all layers -> scoring engine ->
ScanReport.band) for model-level precision/recall/F1 + confusion matrix.
`--layer` restricts evaluation to a single layer's raw Finding output, which is
how S4/S5 validate a layer in isolation before the scoring engine (S6) exists.

Usage: python -m assay.metrics fixtures/models/manifest.json [--layer entropy]
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from assay.config import DEFAULT_CONFIG
from assay.intake.loader import iter_tensors
from assay.models import Finding, TensorInfo

_REGION_RE = re.compile(r"\[(\d+),\s*(\d+)\)")


def _layer_entropy(info: TensorInfo, arr, config) -> list[Finding]:
    from assay.layers.steg_entropy import analyze_entropy

    return analyze_entropy(info, arr, config)


LAYERS = {
    "entropy": _layer_entropy,
}


def _findings_regions(findings: list[Finding]) -> list[tuple[int, int]]:
    """Best-effort extraction of the "[start,end)" localized region a Finding's
    detail text reports, for evaluating localization accuracy. Findings that
    don't mention a region (e.g. wrapper-level ones) contribute nothing.
    """
    regions = []
    for f in findings:
        regions.extend((int(s), int(e)) for s, e in _REGION_RE.findall(f.detail))
    return regions


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def evaluate_layer(manifest_path: Path, layer_name: str) -> dict[str, Any]:
    analyze = LAYERS[layer_name]
    manifest: list[dict] = json.loads(manifest_path.read_text())
    models_dir = manifest_path.parent

    clean_results: list[tuple[str, str, bool]] = []  # (file, tensor, flagged)
    technique_results: list[tuple[str, str, str, bool, bool]] = []
    # (technique, file, tensor, detected, localization_overlaps_ground_truth)

    for entry in manifest:
        path = models_dir / entry["file"]
        if entry["label"] == "clean":
            for info, arr in iter_tensors(path):
                findings = analyze(info, arr, DEFAULT_CONFIG)
                clean_results.append((entry["file"], info.name, len(findings) > 0))
        elif entry.get("target_tensor") is not None:
            tensors = {info.name: (info, arr) for info, arr in iter_tensors(path)}
            info, arr = tensors[entry["target_tensor"]]
            findings = analyze(info, arr, DEFAULT_CONFIG)
            detected = len(findings) > 0

            gt_start, gt_end = entry["ground_truth_region"]
            regions = _findings_regions(findings)
            overlaps_gt = any(_overlaps(r, (gt_start, gt_end)) for r in regions)

            technique_results.append((entry["technique"], entry["file"], info.name, detected, overlaps_gt))

    n_clean = len(clean_results)
    n_clean_fp = sum(1 for *_, flagged in clean_results if flagged)
    clean_fp_rate = n_clean_fp / n_clean if n_clean else 0.0

    n_positive = len(technique_results)
    n_detected = sum(1 for *_, detected, _ in technique_results if detected)
    recall = n_detected / n_positive if n_positive else 0.0

    per_technique: dict[str, dict[str, Any]] = {}
    for technique, _file, _tensor, detected, overlaps_gt in technique_results:
        row = per_technique.setdefault(technique, {"total": 0, "detected": 0, "localized": 0})
        row["total"] += 1
        row["detected"] += int(detected)
        row["localized"] += int(overlaps_gt)

    return {
        "layer": layer_name,
        "n_clean_tensors": n_clean,
        "n_clean_false_positives": n_clean_fp,
        "clean_false_positive_rate": clean_fp_rate,
        "n_poisoned_tensors": n_positive,
        "n_detected": n_detected,
        "recall": recall,
        "per_technique": per_technique,
        "clean_results": clean_results,
        "technique_results": technique_results,
    }


def _print_report(report: dict[str, Any]) -> None:
    print(f"=== layer: {report['layer']} ===")
    print()
    print("-- clean tensors (false-positive check) --")
    for file, tensor, flagged in report["clean_results"]:
        mark = "FLAGGED" if flagged else "clean"
        print(f"  {file:32s} {tensor:15s} {mark}")
    print(
        f"clean false-positive rate: {report['n_clean_false_positives']}/{report['n_clean_tensors']} "
        f"= {report['clean_false_positive_rate']:.3f}"
    )
    print()
    print("-- poisoned tensors (recall + localization check) --")
    for technique, file, tensor, detected, overlaps_gt in report["technique_results"]:
        print(
            f"  {technique:18s} {file:32s} {tensor:15s} "
            f"detected={detected!s:5s} localized_overlap={overlaps_gt!s:5s}"
        )
    print()
    print("-- per-technique recall --")
    for technique, row in sorted(report["per_technique"].items()):
        r = row["detected"] / row["total"] if row["total"] else 0.0
        loc = row["localized"] / row["total"] if row["total"] else 0.0
        print(f"  {technique:18s} recall={row['detected']}/{row['total']} = {r:.3f}  localized={loc:.3f}")
    print()
    print(
        f"OVERALL recall on LSB-poisoned tensors: {report['n_detected']}/{report['n_poisoned_tensors']} "
        f"= {report['recall']:.3f}"
    )
    print(f"OVERALL clean per-tensor false-positive rate: {report['clean_false_positive_rate']:.3f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assay.metrics")
    parser.add_argument("manifest", help="path to fixtures manifest.json")
    parser.add_argument("--layer", help="restrict evaluation to a single layer", choices=sorted(LAYERS))
    args = parser.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"error: no such manifest: {manifest_path}")
        return 1

    if args.layer is None:
        print("error: full-pipeline metrics not implemented yet (see S6); pass --layer <name>")
        print(f"available layers: {sorted(LAYERS)}")
        return 1

    report = evaluate_layer(manifest_path, args.layer)
    _print_report(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
