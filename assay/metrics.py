"""Precision/recall/confusion over a fixture manifest.

TODO(S6): run the whole pipeline over fixtures/models/manifest.json ->
precision/recall/F1 + confusion matrix (model-level and per-technique),
printed as a table. `--layer` filter for single-layer eval (S4/S5 use this
in isolation before the scoring engine exists).
Usage: `python -m assay.metrics fixtures/models/manifest.json [--layer entropy]`
"""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="assay.metrics")
    parser.add_argument("manifest", help="path to fixtures manifest.json")
    parser.add_argument("--layer", help="restrict evaluation to a single layer")
    parser.parse_args(argv)
    # TODO(S4+): implement once fixtures (S2) and at least one layer (S3/S4) exist.
    print("error: metrics pipeline not implemented yet (see S2, S4, S6)")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
