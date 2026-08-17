"""Format sniffing for model artifacts: .pt .pkl .safetensors .onnx .bin .gguf.

`detect_format` reads only a small header (never the whole file) and never
deserializes anything. Magic bytes decide first; file extension is a fallback
for streams with no reliable magic (e.g. bare pickle protocol 0).
"""

from __future__ import annotations

from pathlib import Path

_HEADER_READ_SIZE = 64

_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_PICKLE_PROTO0_OPCODES = (b"(", b"]", b"}", b"c", b"I", b"S")

_EXTENSION_MAP = {
    ".safetensors": "safetensors",
    ".pt": "pt",
    ".pth": "pt",
    ".pkl": "pkl",
    ".onnx": "onnx",
    ".gguf": "gguf",
    ".bin": "bin",
}


def detect_format(path: str | Path) -> str:
    """Sniff the serialization format of a model artifact.

    Returns one of: "safetensors", "pt" (zip-wrapped pickle archive),
    "pkl" (bare pickle stream), "onnx", "gguf", "bin" (extension-only,
    content did not sniff), "unknown".
    """
    p = Path(path)
    with p.open("rb") as f:
        header = f.read(_HEADER_READ_SIZE)

    if header.startswith(b"GGUF"):
        return "gguf"
    if header[:4] in _ZIP_MAGICS:
        return "pt"
    if _looks_like_safetensors(header, p.stat().st_size):
        return "safetensors"
    if header[:1] == b"\x80":  # PROTO opcode, pickle protocol 2+
        return "pkl"
    if header[:1] in _PICKLE_PROTO0_OPCODES:  # pickle protocol 0 stream start
        return "pkl"

    return _EXTENSION_MAP.get(p.suffix.lower(), "unknown")


def _looks_like_safetensors(header: bytes, file_size: int) -> bool:
    """safetensors has no fixed magic: an 8-byte little-endian header length
    followed by that many bytes of a JSON object. Reject anything where the
    declared length is inconsistent with the file size or doesn't start `{`.
    """
    if len(header) < 9:
        return False
    header_len = int.from_bytes(header[:8], "little")
    return 0 < header_len <= file_size - 8 and header[8:9] == b"{"
