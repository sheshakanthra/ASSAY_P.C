"""ASSAY CLI — `python -m assay scan|disarm|report`.

S0: `scan` returns a valid, empty ScanReport (no analysis logic yet — that
lands in S1-S6). `disarm` and `report` are stubbed for later sessions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from assay.models import RiskBand, ScanReport


def _empty_scan_report(path: str) -> ScanReport:
    """TODO(S6): replace with the real pipeline (intake -> L1 -> L2a/b/c -> score).

    S0 contract: any existing file at `path` yields a valid ScanReport with
    zero findings, risk_score=0, band=CLEAN, format left as "unknown" until
    intake.detect (S1) is wired in.
    """
    return ScanReport(
        artifact=str(path),
        format="unknown",
        tensor_reports=[],
        wrapper_findings=[],
        risk_score=0.0,
        band=RiskBand.CLEAN,
        explanations=[],
    )


def cmd_scan(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.exists():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 1
    report = _empty_scan_report(str(path))
    print(json.dumps(report.to_dict(), indent=2))
    return 0


def cmd_disarm(_args: argparse.Namespace) -> int:
    # TODO(S8): implement disarm/scrub.py + disarm/attest.py wiring.
    print("error: `disarm` is not implemented yet (see S8)", file=sys.stderr)
    return 1


def cmd_report(_args: argparse.Namespace) -> int:
    # TODO(S7): implement report/render.py wiring.
    print("error: `report` is not implemented yet (see S7)", file=sys.stderr)
    return 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="assay", description="ASSAY model weight security scanner")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan_p = subparsers.add_parser("scan", help="Scan a model artifact and print a ScanReport as JSON")
    scan_p.add_argument("path", help="path to the model artifact")
    scan_p.add_argument("--report", help="write an HTML report to this path (S7)")
    scan_p.add_argument("--json", help="write the JSON report to this path (S7)")
    scan_p.set_defaults(func=cmd_scan)

    disarm_p = subparsers.add_parser("disarm", help="Disarm a poisoned artifact (S8)")
    disarm_p.add_argument("path", help="path to the model artifact")
    disarm_p.add_argument("-o", "--output", help="output path for the disarmed artifact")
    disarm_p.add_argument("--method", choices=["lsb", "permute", "quantize"], default="lsb")
    disarm_p.set_defaults(func=cmd_disarm)

    report_p = subparsers.add_parser("report", help="Render a stored report (S7)")
    report_p.add_argument("path", help="path to a stored ScanReport JSON")
    report_p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
