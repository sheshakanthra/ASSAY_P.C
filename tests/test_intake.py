"""S1 GATE tests: safe multi-format intake + static pickle disassembly.

- iterate tensors from both .safetensors and .pt, assert shapes/dtypes/count
- disassemble returns opcodes for a crafted pickle
- malformed file raises the typed error (asserted)
"""

from __future__ import annotations

import pickle

import pytest
import torch
from safetensors.torch import save_file as save_safetensors

from assay.intake.detect import detect_format
from assay.intake.loader import (
    MalformedArtifactError,
    UnsafeArtifactError,
    iter_tensors,
)
from assay.intake.pickle_inspect import disassemble

# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def safetensors_path(tmp_path):
    path = tmp_path / "two_tensor.safetensors"
    tensors = {
        "layer1.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layer1.bias": torch.tensor([1.0, 2.0], dtype=torch.float16),
    }
    save_safetensors(tensors, str(path))
    return path


@pytest.fixture
def torch_path(tmp_path):
    path = tmp_path / "two_tensor.pt"
    state_dict = {
        "layer1.weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "layer1.bias": torch.tensor([1.0, 2.0], dtype=torch.float32),
    }
    torch.save(state_dict, str(path))
    return path


class _EvilReduce:
    """Never instantiated/unpickled — only pickle.dumps'd to produce opcode bytes
    an attacker-controlled artifact would contain (see CLAUDE.md SAFETY: static
    opcode sequences that are never deserialized).
    """

    def __reduce__(self):
        import os

        return (os.system, ("echo pwned",))


@pytest.fixture
def bad_pickle_path(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(pickle.dumps(_EvilReduce(), protocol=2))
    return path


# ---------------------------------------------------------------------------
# detect_format
# ---------------------------------------------------------------------------


def test_detect_format_safetensors(safetensors_path):
    assert detect_format(safetensors_path) == "safetensors"


def test_detect_format_pt_zip(torch_path):
    assert detect_format(torch_path) == "pt"


def test_detect_format_bare_pickle(bad_pickle_path):
    assert detect_format(bad_pickle_path) == "pkl"


# ---------------------------------------------------------------------------
# iter_tensors — safetensors
# ---------------------------------------------------------------------------


def test_iter_tensors_safetensors_shapes_and_dtypes(safetensors_path):
    results = list(iter_tensors(safetensors_path))
    assert len(results) == 2

    by_name = {info.name: (info, arr) for info, arr in results}
    weight_info, weight_arr = by_name["layer1.weight"]
    assert weight_info.shape == (2, 3)
    assert weight_info.dtype == "float32"
    assert weight_arr.shape == (2, 3)
    assert weight_arr.dtype.name == "float32"

    bias_info, bias_arr = by_name["layer1.bias"]
    assert bias_info.shape == (2,)
    assert bias_info.dtype == "float16"
    assert bias_arr.dtype.name == "float16"


def test_iter_tensors_safetensors_bfloat16(tmp_path):
    path = tmp_path / "bf16.safetensors"
    save_safetensors({"w": torch.ones(2, 2, dtype=torch.bfloat16)}, str(path))

    results = list(iter_tensors(path))
    assert len(results) == 1
    info, arr = results[0]
    assert info.dtype == "bfloat16"
    assert info.shape == (2, 2)
    # numpy has no native bfloat16 — bytes are preserved as a same-size uint16 view.
    assert arr.dtype.name == "uint16"
    assert arr.shape == (2, 2)


# ---------------------------------------------------------------------------
# iter_tensors — torch .pt
# ---------------------------------------------------------------------------


def test_iter_tensors_torch_shapes_and_dtypes(torch_path):
    results = list(iter_tensors(torch_path))
    assert len(results) == 2

    by_name = {info.name: (info, arr) for info, arr in results}
    weight_info, weight_arr = by_name["layer1.weight"]
    assert weight_info.shape == (2, 3)
    assert weight_info.dtype == "float32"
    assert weight_arr.shape == (2, 3)

    bias_info, _ = by_name["layer1.bias"]
    assert bias_info.shape == (2,)


def test_iter_tensors_torch_bare_tensor(tmp_path):
    path = tmp_path / "bare_tensor.pt"
    torch.save(torch.arange(4, dtype=torch.float32), str(path))

    results = list(iter_tensors(path))
    assert len(results) == 1
    info, arr = results[0]
    assert info.shape == (4,)
    assert arr.shape == (4,)


# ---------------------------------------------------------------------------
# iter_tensors — unsafe / malformed
# ---------------------------------------------------------------------------


def test_iter_tensors_rejects_malicious_pickle(bad_pickle_path):
    with pytest.raises(UnsafeArtifactError):
        list(iter_tensors(bad_pickle_path))


def test_iter_tensors_missing_file_raises():
    with pytest.raises(MalformedArtifactError):
        list(iter_tensors("does/not/exist.safetensors"))


def test_iter_tensors_empty_file_raises(tmp_path):
    path = tmp_path / "empty.safetensors"
    path.write_bytes(b"")
    with pytest.raises(MalformedArtifactError):
        list(iter_tensors(path))


def test_iter_tensors_truncated_torch_archive_raises(torch_path):
    full = torch_path.read_bytes()
    truncated_path = torch_path.parent / "truncated.pt"
    truncated_path.write_bytes(full[: len(full) // 2])

    with pytest.raises(MalformedArtifactError):
        list(iter_tensors(truncated_path))


def test_iter_tensors_garbage_content_raises(tmp_path):
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"this is not a model file at all, just noise" * 4)
    with pytest.raises((MalformedArtifactError, UnsafeArtifactError)):
        list(iter_tensors(path))


# ---------------------------------------------------------------------------
# pickle_inspect.disassemble
# ---------------------------------------------------------------------------


def test_disassemble_bare_pickle_finds_dangerous_global(bad_pickle_path):
    ops = disassemble(bad_pickle_path)
    assert len(ops) > 0
    names = [op.name for op in ops]
    assert "GLOBAL" in names or "STACK_GLOBAL" in names
    assert "REDUCE" in names

    global_ops = [op for op in ops if op.name in ("GLOBAL", "STACK_GLOBAL")]
    assert any("system" in str(op.arg) for op in global_ops)


def test_disassemble_extracts_from_zip_archive(torch_path):
    ops = disassemble(torch_path)
    assert len(ops) > 0
    assert ops[0].name == "PROTO"
    assert ops[-1].name == "STOP"


def test_disassemble_missing_file_raises():
    with pytest.raises(MalformedArtifactError):
        disassemble("does/not/exist.pkl")


def test_disassemble_corrupt_zip_raises(tmp_path):
    path = tmp_path / "corrupt.pt"
    path.write_bytes(b"PK\x03\x04" + b"not actually a zip body")
    with pytest.raises(MalformedArtifactError):
        disassemble(path)
