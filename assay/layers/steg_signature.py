"""Layer 2b: byte-signature sweep over reconstructed candidate byte streams.

`analyze_signatures(info, array, config=None) -> list[Finding]` reconstructs
two candidate byte streams and scans both for known magic bytes / structured
text patterns:

  (a) the "LSB stream" — one byte per element, taken from the low byte of each
      float32's mantissa (`u32 & 0xFF`). This undoes exactly the interleaving
      our own S2 fixture generator introduces (it substitutes a whole payload
      byte into that position per element), so a payload embedded that way
      recovers as a contiguous, exact match.
  (b) the "raw stream" — the tensor's bytes exactly as stored in memory
      (4 bytes/element for float32), for payloads embedded byte-aligned across
      the full representation rather than restricted to the low byte.

Both streams are scanned for: the EICAR test string, ELF/PE/Mach-O/ZIP magic
bytes, shell strings (/bin/sh, cmd.exe, powershell), URLs, IPv4 addresses, and
base64-looking blobs. Each match is reported with its offset (element index
for the LSB stream, byte index for the raw stream) so it's directly comparable
to a manifest `ground_truth_region`.
"""

from __future__ import annotations

import re

import numpy as np

from assay.config import DEFAULT_CONFIG, Config
from assay.models import Finding, Severity, TensorInfo

EICAR = b"X5O!P%@AP[4\\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"

# (rule, literal pattern, severity)
_LITERAL_SIGNATURES: tuple[tuple[str, bytes, Severity], ...] = (
    ("eicar_signature", EICAR, Severity.CRITICAL),
    ("elf_header", b"\x7fELF", Severity.HIGH),
    ("zip_header", b"PK\x03\x04", Severity.MEDIUM),
    ("macho_header", b"\xfe\xed\xfa\xce", Severity.HIGH),
    ("macho_header", b"\xfe\xed\xfa\xcf", Severity.HIGH),
    ("macho_header", b"\xce\xfa\xed\xfe", Severity.HIGH),
    ("macho_header", b"\xcf\xfa\xed\xfe", Severity.HIGH),
    ("shell_string", b"/bin/sh", Severity.HIGH),
    ("shell_string", b"cmd.exe", Severity.HIGH),
    ("shell_string", b"powershell", Severity.HIGH),
)

# (rule, compiled regex, severity, min_match_length)
_REGEX_SIGNATURES: tuple[tuple[str, re.Pattern[bytes], Severity, int], ...] = (
    ("url", re.compile(rb"https?://[!-~]{4,}"), Severity.HIGH, 0),
    (
        "ipv4_address",
        re.compile(rb"(?:(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})\.){3}(?:25[0-5]|2[0-4][0-9]|1?[0-9]{1,2})"),
        Severity.MEDIUM,
        0,
    ),
    ("base64_blob", re.compile(rb"(?:[A-Za-z0-9+/]{4}){6,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?"), Severity.LOW, 24),
)

# PE files: a short 2-byte "MZ" match alone is common by chance; only report it
# alongside the "PE\0\0" signature the DOS stub always points to.
_PE_SEARCH_WINDOW = 1024


def analyze_signatures(info: TensorInfo, array: np.ndarray, config: Config | None = None) -> list[Finding]:
    config = config or DEFAULT_CONFIG
    findings: list[Finding] = []

    if array.dtype == np.float32 and array.size > 0:
        lsb_stream = (array.reshape(-1).view(np.uint32) & 0xFF).astype(np.uint8).tobytes()
        findings.extend(_scan_stream(lsb_stream, "lsb", info.name))

    raw_stream = array.reshape(-1).view(np.uint8).tobytes()
    findings.extend(_scan_stream(raw_stream, "raw", info.name))

    return findings


def _scan_stream(stream: bytes, stream_label: str, tensor_name: str) -> list[Finding]:
    findings: list[Finding] = []

    for rule, pattern, severity in _LITERAL_SIGNATURES:
        start = 0
        while True:
            offset = stream.find(pattern, start)
            if offset == -1:
                break
            findings.append(
                Finding(
                    layer="signature",
                    rule=rule,
                    severity=severity,
                    tensor=tensor_name,
                    detail=(
                        f"{rule} matched {pattern!r} in the {stream_label} byte stream at offset "
                        f"{offset} (stream length {len(stream)}); region [{offset},{offset + len(pattern)})"
                    ),
                    value=float(offset),
                )
            )
            start = offset + 1

    pe_offset = stream.find(b"MZ")
    while pe_offset != -1:
        window = stream[pe_offset : pe_offset + _PE_SEARCH_WINDOW]
        pe_sig_rel = window.find(b"PE\x00\x00")
        if pe_sig_rel != -1:
            findings.append(
                Finding(
                    layer="signature",
                    rule="pe_header",
                    severity=Severity.HIGH,
                    tensor=tensor_name,
                    detail=(
                        f"pe_header: 'MZ' at {stream_label} offset {pe_offset} followed by 'PE\\0\\0' "
                        f"at offset {pe_offset + pe_sig_rel}; region [{pe_offset},{pe_offset + pe_sig_rel + 4})"
                    ),
                    value=float(pe_offset),
                )
            )
        pe_offset = stream.find(b"MZ", pe_offset + 1)

    for rule, regex, severity, min_length in _REGEX_SIGNATURES:
        for match in regex.finditer(stream):
            if len(match.group(0)) < min_length:
                continue
            findings.append(
                Finding(
                    layer="signature",
                    rule=rule,
                    severity=severity,
                    tensor=tensor_name,
                    detail=(
                        f"{rule} matched in the {stream_label} byte stream at offset "
                        f"{match.start()}; region [{match.start()},{match.end()}): {match.group(0)!r}"
                    ),
                    value=float(match.start()),
                )
            )

    return findings
