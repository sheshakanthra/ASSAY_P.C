"""S3 GATE tests: Layer 1 wrapper/opcode/archive analysis.

- flags the bad_pickle fixture (os.system chain) as HIGH
- clean .pt and .safetensors produce zero wrapper findings
- one unit test per rule
"""

from __future__ import annotations

import os
import pickle
import struct
import zipfile
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file as save_safetensors

from assay.config import Config
from assay.layers.wrapper import analyze_wrapper
from assay.models import Severity

# ---------------------------------------------------------------------------
# fixtures: clean artifacts
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_pt(tmp_path):
    path = tmp_path / "clean.pt"
    torch.save({"w": torch.randn(3, 3), "b": torch.randn(3)}, str(path))
    return path


@pytest.fixture
def clean_safetensors(tmp_path):
    path = tmp_path / "clean.safetensors"
    save_safetensors({"w": torch.randn(3, 3), "b": torch.randn(3)}, str(path))
    return path


def _make_zip(tmp_path: Path, name: str, members: dict[str, bytes]) -> Path:
    path = tmp_path / name
    with zipfile.ZipFile(path, "w") as zf:
        for member_name, data in members.items():
            zf.writestr(member_name, data)
    return path


_SAFE_PICKLE = pickle.dumps({"ok": True}, protocol=2)


class _EvilReduce:
    """Never instantiated/unpickled — only pickle.dumps'd. See fixtures/SAFETY.md."""

    def __reduce__(self):
        return (os.system, ("echo pwned",))


_BAD_PICKLE = pickle.dumps(_EvilReduce(), protocol=2)
_REFERENCED_ONLY_PICKLE = pickle.dumps(os.system, protocol=2)  # GLOBAL then STOP — never called


def _write_safetensors_raw(path: Path, header_text: str, data: bytes = b"") -> None:
    header_bytes = header_text.encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        f.write(data)


# ---------------------------------------------------------------------------
# clean pass (GATE requirement)
# ---------------------------------------------------------------------------


def test_clean_pt_produces_zero_wrapper_findings(clean_pt):
    assert analyze_wrapper(clean_pt) == []


def test_clean_safetensors_produces_zero_wrapper_findings(clean_safetensors):
    assert analyze_wrapper(clean_safetensors) == []


# ---------------------------------------------------------------------------
# opcode graph rules
# ---------------------------------------------------------------------------


def test_dangerous_global_invoked_flags_bad_pickle_as_high(tmp_path):
    path = tmp_path / "bad.pkl"
    path.write_bytes(_BAD_PICKLE)

    findings = analyze_wrapper(path)
    invoked = [f for f in findings if f.rule == "dangerous_global_invoked"]
    assert len(invoked) == 1
    assert invoked[0].severity == Severity.HIGH
    assert "system" in invoked[0].detail


def test_dangerous_global_referenced_but_not_invoked_is_medium(tmp_path):
    path = tmp_path / "ref_only.pkl"
    path.write_bytes(_REFERENCED_ONLY_PICKLE)

    findings = analyze_wrapper(path)
    referenced = [f for f in findings if f.rule == "dangerous_global_referenced"]
    assert len(referenced) == 1
    assert referenced[0].severity == Severity.MEDIUM
    assert not any(f.rule == "dangerous_global_invoked" for f in findings)


# ---------------------------------------------------------------------------
# zip archive integrity rules
# ---------------------------------------------------------------------------


def test_zip_crc_mismatch(tmp_path):
    path = _make_zip(tmp_path, "crc.pt", {"archive/data.pkl": _SAFE_PICKLE})
    # Flip a byte inside the stored (uncompressed) pickle content without
    # touching the recorded CRC-32, so verification fails on read.
    raw = bytearray(path.read_bytes())
    idx = raw.find(_SAFE_PICKLE)
    assert idx != -1
    raw[idx] ^= 0xFF
    path.write_bytes(raw)

    findings = analyze_wrapper(path)
    assert any(f.rule == "zip_crc_mismatch" and f.severity == Severity.HIGH for f in findings)


def test_zip_path_traversal(tmp_path):
    path = _make_zip(
        tmp_path, "traversal.pt", {"../evil.pkl": _SAFE_PICKLE, "archive/data.pkl": _SAFE_PICKLE}
    )
    findings = analyze_wrapper(path)
    assert any(f.rule == "zip_path_traversal" and f.severity == Severity.HIGH for f in findings)


def test_zip_executable_member(tmp_path):
    path = _make_zip(
        tmp_path,
        "exe.pt",
        {"archive/data.pkl": _SAFE_PICKLE, "archive/payload.bin": b"\x7fELF" + b"\x00" * 32},
    )
    findings = analyze_wrapper(path)
    assert any(f.rule == "zip_executable_member" and f.severity == Severity.HIGH for f in findings)


def test_zip_multiple_pickles(tmp_path):
    path = _make_zip(
        tmp_path, "multi.pt", {"archive/data.pkl": _SAFE_PICKLE, "archive/data2.pkl": _SAFE_PICKLE}
    )
    findings = analyze_wrapper(path)
    assert any(f.rule == "zip_multiple_pickles" and f.severity == Severity.MEDIUM for f in findings)


def test_non_standard_archive_wrapper(tmp_path):
    path = tmp_path / "fake.pt"
    path.write_bytes(b"7z\xbc\xaf\x27\x1c" + b"not a real archive")  # 7z magic, .pt extension
    findings = analyze_wrapper(path)
    assert any(f.rule == "non_standard_archive_wrapper" and f.severity == Severity.HIGH for f in findings)


def test_corrupt_or_truncated_archive(clean_pt):
    full = clean_pt.read_bytes()
    truncated = clean_pt.parent / "truncated.pt"
    truncated.write_bytes(full[: len(full) // 2])

    findings = analyze_wrapper(truncated)
    assert any(f.rule == "corrupt_or_truncated_archive" and f.severity == Severity.HIGH for f in findings)


# ---------------------------------------------------------------------------
# safetensors header rules
# ---------------------------------------------------------------------------


def test_safetensors_duplicate_metadata_key(tmp_path):
    path = tmp_path / "dup.safetensors"
    header = (
        '{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},'
        '"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    )
    _write_safetensors_raw(path, header, data=b"\x00" * 8)

    findings = analyze_wrapper(path)
    assert any(
        f.rule == "safetensors_duplicate_metadata_key" and f.severity == Severity.HIGH for f in findings
    )


def test_safetensors_offset_length_inconsistency(tmp_path):
    path = tmp_path / "bad_offsets.safetensors"
    # shape [2] float32 should span 8 bytes; claim a 4-byte span instead.
    header = '{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}'
    _write_safetensors_raw(path, header, data=b"\x00" * 8)

    findings = analyze_wrapper(path)
    assert any(
        f.rule == "safetensors_offset_length_inconsistency" and f.severity == Severity.HIGH
        for f in findings
    )


def test_safetensors_hidden_non_tensor_key(tmp_path):
    path = tmp_path / "hidden_key.safetensors"
    header = '{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]},"secret":"hello"}'
    _write_safetensors_raw(path, header, data=b"\x00" * 8)

    findings = analyze_wrapper(path)
    assert any(
        f.rule == "safetensors_hidden_non_tensor_key" and f.severity == Severity.MEDIUM for f in findings
    )


def test_safetensors_oversized_header(tmp_path):
    path = tmp_path / "oversized.safetensors"
    header = '{"w":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}'
    _write_safetensors_raw(path, header, data=b"\x00" * 8)

    tiny_threshold_config = Config()
    tiny_threshold_config.thresholds.safetensors_header_max_bytes = 10  # smaller than our header

    findings = analyze_wrapper(path, config=tiny_threshold_config)
    oversized = [f for f in findings if f.rule == "safetensors_oversized_header"]
    assert len(oversized) == 1
    assert oversized[0].severity == Severity.MEDIUM
    assert oversized[0].value == len(header.encode("utf-8"))
    assert oversized[0].threshold == 10
