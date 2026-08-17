"""S2 GATE tests: fixture generator produces labeled clean/poisoned artifacts.

- >=1 clean + >=5 poisoned artifacts exist; manifest.json validates against a schema;
- smoke test: each poisoned file differs from clean only in the recorded target
  tensor (LSB techniques) / wrapper (bad_pickle).

Requires `python fixtures/train_baseline.py && python fixtures/generate.py` to
have been run first (the S2 GATE runs both before this suite).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from assay.intake.loader import iter_tensors
from assay.intake.pickle_inspect import disassemble

MODELS_DIR = "fixtures/models"
MANIFEST_PATH = f"{MODELS_DIR}/manifest.json"

REQUIRED_KEYS = {"file", "label", "technique", "target_tensor", "ground_truth_region"}
VALID_LABELS = {"clean", "poisoned"}


@pytest.fixture(scope="module")
def manifest() -> list[dict]:
    with open(MANIFEST_PATH, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# manifest schema + counts
# ---------------------------------------------------------------------------


def test_manifest_is_a_list_with_enough_entries(manifest):
    assert isinstance(manifest, list)
    clean = [e for e in manifest if e["label"] == "clean"]
    poisoned = [e for e in manifest if e["label"] == "poisoned"]
    assert len(clean) >= 1
    assert len(poisoned) >= 5


def test_manifest_entries_match_schema(manifest):
    for entry in manifest:
        assert set(entry.keys()) == REQUIRED_KEYS, entry

        assert isinstance(entry["file"], str) and entry["file"]
        assert entry["label"] in VALID_LABELS

        if entry["label"] == "clean":
            assert entry["technique"] is None
            assert entry["target_tensor"] is None
            assert entry["ground_truth_region"] is None
        else:
            assert isinstance(entry["technique"], str) and entry["technique"]
            # target_tensor / ground_truth_region are None for wrapper-level
            # techniques (bad_pickle) and set together for tensor-level ones.
            if entry["target_tensor"] is None:
                assert entry["ground_truth_region"] is None
            else:
                assert isinstance(entry["target_tensor"], str)
                region = entry["ground_truth_region"]
                assert isinstance(region, list) and len(region) == 2
                start, end = region
                assert isinstance(start, int) and isinstance(end, int)
                assert 0 <= start < end

        artifact_path = Path(MODELS_DIR) / entry["file"]
        assert artifact_path.exists(), f"manifest references missing file: {artifact_path}"


def test_manifest_covers_all_five_required_techniques(manifest):
    techniques = {e["technique"] for e in manifest if e["label"] == "poisoned"}
    assert techniques == {
        "lsb_eicar",
        "lsb_elf_header",
        "lsb_random_blob",
        "plaintext_marker",
        "bad_pickle",
    }


# ---------------------------------------------------------------------------
# smoke test: poisoned differs from clean only in the recorded scope
# ---------------------------------------------------------------------------


def _load_state(filename: str) -> dict[str, np.ndarray]:
    return {info.name: arr for info, arr in iter_tensors(f"{MODELS_DIR}/{filename}")}


def test_lsb_poisoned_tensors_match_clean_outside_target(manifest):
    clean_file = next(e["file"] for e in manifest if e["label"] == "clean")
    clean_state = _load_state(clean_file)

    lsb_entries = [e for e in manifest if e["target_tensor"] is not None]
    assert lsb_entries, "expected at least one tensor-level (LSB) poisoned entry"

    for entry in lsb_entries:
        poisoned_state = _load_state(entry["file"])

        assert set(poisoned_state.keys()) == set(clean_state.keys()), entry["file"]

        for name, clean_arr in clean_state.items():
            poisoned_arr = poisoned_state[name]
            if name == entry["target_tensor"]:
                assert not np.array_equal(clean_arr, poisoned_arr), (
                    f"{entry['file']}: target tensor {name} was not modified"
                )
            else:
                np.testing.assert_array_equal(
                    clean_arr,
                    poisoned_arr,
                    err_msg=f"{entry['file']}: non-target tensor {name} unexpectedly changed",
                )


def test_lsb_poisoned_region_confined_to_ground_truth(manifest):
    clean_file = next(e["file"] for e in manifest if e["label"] == "clean")
    clean_state = _load_state(clean_file)

    for entry in (e for e in manifest if e["target_tensor"] is not None):
        poisoned_state = _load_state(entry["file"])
        clean_flat = clean_state[entry["target_tensor"]].reshape(-1)
        poisoned_flat = poisoned_state[entry["target_tensor"]].reshape(-1)

        start, end = entry["ground_truth_region"]
        diff_mask = clean_flat != poisoned_flat
        diff_indices = np.flatnonzero(diff_mask)

        assert diff_indices.size > 0, entry["file"]
        assert diff_indices.min() >= start, entry["file"]
        assert diff_indices.max() < end, entry["file"]


def test_bad_pickle_differs_in_wrapper_not_weights(manifest):
    entry = next(e for e in manifest if e["technique"] == "bad_pickle")
    assert entry["target_tensor"] is None  # not a modified weight — a distinct wrapper

    ops = disassemble(f"{MODELS_DIR}/{entry['file']}")
    op_names = [op.name for op in ops]
    assert "REDUCE" in op_names
    global_ops = [op for op in ops if op.name in ("GLOBAL", "STACK_GLOBAL")]
    assert any("system" in str(op.arg) for op in global_ops)
