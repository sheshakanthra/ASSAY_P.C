"""Format sniffing for model artifacts: .pt .pkl .safetensors .onnx .bin .gguf.

TODO(S1): implement `detect_format(path) -> str` using magic bytes first,
falling back to file extension. Must not require loading/executing the file.
"""

from __future__ import annotations

# TODO(S1): magic-byte / extension based format sniffer.
