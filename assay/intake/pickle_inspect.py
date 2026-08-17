"""Static pickle disassembly — never unpickle.

`disassemble(path) -> list[Opcode]` extracts the raw pickle byte stream (the
whole file for a bare `.pkl`, or the `data.pkl` archive member for a
zip-wrapped `.pt`) and runs it through `pickletools.genops`. This only reads
opcodes; it never constructs objects or calls `pickle.load`.
"""

from __future__ import annotations

import pickletools
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assay.intake.loader import MalformedArtifactError

_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")


@dataclass
class Opcode:
    """One disassembled pickle opcode."""

    name: str
    arg: Any
    pos: int | None

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "arg": self.arg, "pos": self.pos}


def disassemble(path: str | Path) -> list[Opcode]:
    """Return the static opcode sequence of the pickle stream found in `path`.

    Raises MalformedArtifactError if no pickle stream can be located or the
    stream is truncated/corrupt.
    """
    p = Path(path)
    raw = _extract_pickle_bytes(p)
    try:
        return [Opcode(name=op.name, arg=arg, pos=pos) for op, arg, pos in pickletools.genops(raw)]
    except Exception as e:
        raise MalformedArtifactError(f"{p}: failed to disassemble pickle stream: {e}") from e


def _extract_pickle_bytes(path: Path) -> bytes:
    if not path.exists():
        raise MalformedArtifactError(f"no such file: {path}")
    data = path.read_bytes()
    if not data:
        raise MalformedArtifactError(f"empty file: {path}")
    if data[:4] in _ZIP_MAGICS:
        return _extract_pickle_from_zip(path)
    return data


def _extract_pickle_from_zip(path: Path) -> bytes:
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            candidates = [n for n in names if n.endswith("data.pkl")]
            if not candidates:
                candidates = [n for n in names if n.endswith(".pkl")]
            if not candidates:
                raise MalformedArtifactError(f"{path}: zip archive has no pickle member")
            return zf.read(candidates[0])
    except zipfile.BadZipFile as e:
        raise MalformedArtifactError(f"{path}: corrupt zip archive: {e}") from e
