"""Layer 1: opcode graph + archive integrity + safetensors-header analysis.

`analyze_wrapper(path, config=None) -> list[Finding]` catches payloads/backdoors
in the *container*, not the weights. Purely static: reads bytes and disassembles
pickle opcodes (via assay.intake.pickle_inspect) without ever deserializing.

Opcode graph analysis is a real (simplified) pickle stack simulation, not a
substring/blocklist grep: it tracks GLOBAL/STACK_GLOBAL/INST pushes through
memoization (PUT/GET) and containers, and only reports a HIGH-severity finding
when a dangerous global is actually consumed by an invocation opcode
(REDUCE/NEWOBJ/NEWOBJ_EX/BUILD/OBJ) — i.e. a real call chain, e.g.
"GLOBAL os.system invoked via REDUCE". A dangerous global that's merely
referenced (pushed, never confirmed invoked) is reported separately at MEDIUM.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from assay.config import DEFAULT_CONFIG, Config
from assay.intake import pickle_inspect
from assay.intake.detect import detect_format
from assay.intake.loader import MalformedArtifactError
from assay.intake.pickle_inspect import Opcode
from assay.models import Finding, Severity

_ZIP_MAGICS = (b"PK\x03\x04", b"PK\x05\x06", b"PK\x07\x08")
_PICKLE_PROTO0_OPCODES = (b"(", b"]", b"}", b"c", b"I", b"S")

_DANGEROUS_MODULES = frozenset({"os", "subprocess", "sys", "socket", "posix", "nt", "importlib"})
_BUILTINS_MODULES = frozenset({"builtins", "__builtin__"})
_DANGEROUS_BUILTINS_NAMES = frozenset({"eval", "exec", "compile", "__import__"})

_SAFETENSORS_DTYPE_SIZES = {
    "F64": 8,
    "F32": 4,
    "F16": 2,
    "BF16": 2,
    "I64": 8,
    "I32": 4,
    "I16": 2,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}

_STR_OPCODES = frozenset(
    {"SHORT_BINUNICODE", "BINUNICODE", "BINUNICODE8", "UNICODE", "SHORT_BINSTRING", "BINSTRING", "STRING"}
)
_GENERIC_PUSH_OPCODES = frozenset(
    {
        "EMPTY_DICT",
        "EMPTY_LIST",
        "EMPTY_SET",
        "EMPTY_TUPLE",
        "NONE",
        "NEWTRUE",
        "NEWFALSE",
        "BININT",
        "BININT1",
        "BININT2",
        "LONG",
        "LONG1",
        "LONG4",
        "BINFLOAT",
        "PERSID",
        "BINPERSID",
        "FROZENSET",
    }
)
_NOOP_OPCODES = frozenset({"STOP", "PROTO", "FRAME"})


def analyze_wrapper(path: str | Path, config: Config | None = None) -> list[Finding]:
    """Dispatch wrapper-level analysis based on the artifact's sniffed format.

    Re-checks raw magic bytes (rather than trusting detect_format's extension
    fallback) so a container that merely *claims* a .pt/.safetensors-style
    extension without matching zip/pickle magic is itself flagged.
    """
    config = config or DEFAULT_CONFIG
    p = Path(path)
    fmt = detect_format(p)

    if fmt == "safetensors":
        return _analyze_safetensors(p, config)

    if fmt in ("pt", "pkl", "bin"):
        with p.open("rb") as f:
            header = f.read(8)
        if header[:4] in _ZIP_MAGICS:
            return _analyze_zip_pickle_archive(p, config)
        if header[:1] == b"\x80" or header[:1] in _PICKLE_PROTO0_OPCODES:
            return _analyze_bare_pickle(p, config)
        return [
            Finding(
                layer="wrapper",
                rule="non_standard_archive_wrapper",
                severity=Severity.HIGH,
                detail=(
                    f"{p}: detected as a pickle-based checkpoint by extension, but the file "
                    "does not start with a recognized zip or pickle-protocol magic byte "
                    "(possible non-standard/renamed wrapper)"
                ),
            )
        ]

    return []  # onnx/gguf/unknown: no wrapper-level checks defined in S3


def _corrupt_finding(detail: str) -> Finding:
    return Finding(layer="wrapper", rule="corrupt_or_truncated_archive", severity=Severity.HIGH, detail=detail)


# ---------------------------------------------------------------------------
# opcode graph: a simplified pickle stack simulation
# ---------------------------------------------------------------------------


class _Global:
    __slots__ = ("module", "name", "pos")

    def __init__(self, module: str, name: str, pos: int | None):
        self.module = module
        self.name = name
        self.pos = pos

    def qualname(self) -> str:
        return f"{self.module}.{self.name}"


class _Str:
    __slots__ = ("value",)

    def __init__(self, value: str):
        self.value = value


_MARK = object()
_UNKNOWN = object()


def _is_dangerous(module: str, name: str) -> bool:
    if module in _DANGEROUS_MODULES:
        return True
    return module in _BUILTINS_MODULES and name in _DANGEROUS_BUILTINS_NAMES


def _simulate_opcode_graph(ops: list[Opcode]) -> tuple[list[tuple[str, _Global, Opcode]], list[_Global]]:
    """Walk the opcode stream tracking GLOBAL/STACK_GLOBAL/INST pushes (through
    PUT/GET memoization) and the invocation opcodes that consume them.

    Returns (invocations, referenced_only):
      invocations      confirmed call chains: (invoking_opcode_name, Global, invoking_op)
      referenced_only   dangerous Globals pushed but never confirmed invoked
    """
    stack: list[object] = []
    memo: dict[int, object] = {}
    memo_counter = 0
    invocations: list[tuple[str, _Global, Opcode]] = []
    seen_dangerous: list[_Global] = []
    invoked_ids: set[int] = set()

    def pop() -> object:
        return stack.pop() if stack else _UNKNOWN

    def pop_to_mark() -> list[object]:
        items: list[object] = []
        while stack:
            top = stack.pop()
            if top is _MARK:
                break
            items.append(top)
        items.reverse()
        return items

    def note_global_push(g: _Global) -> None:
        if _is_dangerous(g.module, g.name):
            seen_dangerous.append(g)

    def note_invocation(op_name: str, callee: object, op: Opcode) -> None:
        if isinstance(callee, _Global) and _is_dangerous(callee.module, callee.name):
            invocations.append((op_name, callee, op))
            invoked_ids.add(id(callee))

    for op in ops:
        name = op.name

        if name == "MARK":
            stack.append(_MARK)

        elif name == "GLOBAL":
            module, _, cls = str(op.arg).partition(" ")
            g = _Global(module, cls, op.pos)
            note_global_push(g)
            stack.append(g)

        elif name == "STACK_GLOBAL":
            name_item = pop()
            module_item = pop()
            module = module_item.value if isinstance(module_item, _Str) else "?"
            cls = name_item.value if isinstance(name_item, _Str) else "?"
            g = _Global(module, cls, op.pos)
            note_global_push(g)
            stack.append(g)

        elif name == "INST":
            pop_to_mark()
            module, _, cls = str(op.arg).partition(" ")
            g = _Global(module, cls, op.pos)
            note_global_push(g)
            note_invocation("INST", g, op)
            stack.append(_UNKNOWN)

        elif name == "OBJ":
            items = pop_to_mark()
            cls = items[0] if items else _UNKNOWN
            note_invocation("OBJ", cls, op)
            stack.append(_UNKNOWN)

        elif name in ("REDUCE", "NEWOBJ"):
            pop()  # args
            callee = pop()
            note_invocation(name, callee, op)
            stack.append(_UNKNOWN)

        elif name == "NEWOBJ_EX":
            pop()  # kwargs
            pop()  # args
            callee = pop()
            note_invocation(name, callee, op)
            stack.append(_UNKNOWN)

        elif name == "BUILD":
            pop()  # state
            obj = pop()
            note_invocation(name, obj, op)
            stack.append(_UNKNOWN)

        elif name in _STR_OPCODES:
            stack.append(_Str(op.arg))

        elif name in ("PUT", "BINPUT", "LONG_BINPUT"):
            if stack:
                memo[op.arg] = stack[-1]

        elif name == "MEMOIZE":
            if stack:
                memo[memo_counter] = stack[-1]
            memo_counter += 1

        elif name in ("GET", "BINGET", "LONG_BINGET"):
            stack.append(memo.get(op.arg, _UNKNOWN))

        elif name in ("TUPLE", "LIST", "DICT", "SETITEMS", "ADDITEMS"):
            pop_to_mark()
            stack.append(_UNKNOWN)

        elif name == "TUPLE1":
            pop()
            stack.append(_UNKNOWN)
        elif name == "TUPLE2":
            pop()
            pop()
            stack.append(_UNKNOWN)
        elif name == "TUPLE3":
            pop()
            pop()
            pop()
            stack.append(_UNKNOWN)

        elif name == "POP":
            pop()
        elif name == "POP_MARK":
            pop_to_mark()
        elif name == "DUP":
            if stack:
                stack.append(stack[-1])
        elif name == "APPEND":
            pop()
        elif name == "APPENDS":
            pop_to_mark()
        elif name == "SETITEM":
            pop()
            pop()

        elif name in _NOOP_OPCODES:
            pass
        elif name in _GENERIC_PUSH_OPCODES:
            stack.append(_UNKNOWN)
        else:
            # Unmodeled opcode: assume it produces one value (keeps the stack
            # roughly balanced for the long tail of rarely-used opcodes).
            stack.append(_UNKNOWN)

    referenced_only = [g for g in seen_dangerous if id(g) not in invoked_ids]
    return invocations, referenced_only


def _findings_from_opcode_graph(
    invocations: list[tuple[str, _Global, Opcode]], referenced_only: list[_Global]
) -> list[Finding]:
    findings = []
    for invoking_op_name, g, op in invocations:
        findings.append(
            Finding(
                layer="wrapper",
                rule="dangerous_global_invoked",
                severity=Severity.HIGH,
                detail=(
                    f"GLOBAL '{g.qualname()}' (opcode offset {g.pos}) is invoked via "
                    f"{invoking_op_name} at opcode offset {op.pos} — a call chain capable of "
                    "code execution on load"
                ),
            )
        )
    for g in referenced_only:
        findings.append(
            Finding(
                layer="wrapper",
                rule="dangerous_global_referenced",
                severity=Severity.MEDIUM,
                detail=(
                    f"GLOBAL '{g.qualname()}' (opcode offset {g.pos}) is referenced but no "
                    "confirmed invocation (REDUCE/NEWOBJ/BUILD/OBJ) was found in this opcode stream"
                ),
            )
        )
    return findings


def _analyze_bare_pickle(path: Path, config: Config) -> list[Finding]:
    try:
        ops = pickle_inspect.disassemble(path)
    except MalformedArtifactError as e:
        return [_corrupt_finding(str(e))]
    invocations, referenced_only = _simulate_opcode_graph(ops)
    return _findings_from_opcode_graph(invocations, referenced_only)


# ---------------------------------------------------------------------------
# zip archive integrity
# ---------------------------------------------------------------------------


def _is_suspicious_member_name(name: str) -> bool:
    if name.startswith("/") or name.startswith("\\"):
        return True
    return ".." in name.replace("\\", "/").split("/")


def _analyze_zip_pickle_archive(path: Path, config: Config) -> list[Finding]:
    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as e:
        return [_corrupt_finding(f"{path}: corrupt zip archive: {e}")]

    findings: list[Finding] = []
    with zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            findings.append(
                Finding(
                    layer="wrapper",
                    rule="zip_crc_mismatch",
                    severity=Severity.HIGH,
                    detail=f"archive member '{bad_member}' fails CRC-32 verification (corrupted or tampered)",
                )
            )

        infos = zf.infolist()
        for info in infos:
            if _is_suspicious_member_name(info.filename):
                findings.append(
                    Finding(
                        layer="wrapper",
                        rule="zip_path_traversal",
                        severity=Severity.HIGH,
                        detail=f"archive member has a suspicious path: '{info.filename}'",
                    )
                )
            if info.filename == bad_member:
                continue
            try:
                head = zf.read(info.filename)[:4]
            except (zipfile.BadZipFile, RuntimeError):
                head = b""
            if head[:2] == b"MZ" or head[:4] == b"\x7fELF":
                findings.append(
                    Finding(
                        layer="wrapper",
                        rule="zip_executable_member",
                        severity=Severity.HIGH,
                        detail=f"archive member '{info.filename}' begins with an executable magic byte header",
                    )
                )

        pkl_members = [i.filename for i in infos if i.filename.lower().endswith(".pkl")]
        if len(pkl_members) > 1:
            findings.append(
                Finding(
                    layer="wrapper",
                    rule="zip_multiple_pickles",
                    severity=Severity.MEDIUM,
                    detail=f"archive contains {len(pkl_members)} pickle members (expected 1): {pkl_members}",
                )
            )

    try:
        ops = pickle_inspect.disassemble(path)
    except MalformedArtifactError as e:
        findings.append(_corrupt_finding(str(e)))
        return findings
    invocations, referenced_only = _simulate_opcode_graph(ops)
    findings.extend(_findings_from_opcode_graph(invocations, referenced_only))
    return findings


# ---------------------------------------------------------------------------
# safetensors header analysis
# ---------------------------------------------------------------------------


def _parse_json_detect_duplicates(text: str) -> tuple[dict, set[str]]:
    dup_keys: set[str] = set()

    def hook(pairs: list[tuple[str, object]]) -> dict:
        seen: set[str] = set()
        d: dict = {}
        for k, v in pairs:
            if k in seen:
                dup_keys.add(k)
            seen.add(k)
            d[k] = v
        return d

    header = json.loads(text, object_pairs_hook=hook)
    return header, dup_keys


def _analyze_safetensors(path: Path, config: Config) -> list[Finding]:
    size = path.stat().st_size
    with path.open("rb") as f:
        raw_len = f.read(8)
        if len(raw_len) < 8:
            return [_corrupt_finding(f"{path}: file too small to contain a safetensors header")]
        header_len = int.from_bytes(raw_len, "little")
        if header_len <= 0 or header_len > size - 8:
            return [
                _corrupt_finding(
                    f"{path}: safetensors header length {header_len} is inconsistent with file size {size}"
                )
            ]
        header_bytes = f.read(header_len)

    findings: list[Finding] = []
    if header_len > config.thresholds.safetensors_header_max_bytes:
        findings.append(
            Finding(
                layer="wrapper",
                rule="safetensors_oversized_header",
                severity=Severity.MEDIUM,
                detail=f"safetensors header is {header_len} bytes, exceeding the configured limit",
                value=float(header_len),
                threshold=float(config.thresholds.safetensors_header_max_bytes),
            )
        )

    try:
        header, dup_keys = _parse_json_detect_duplicates(header_bytes.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        findings.append(_corrupt_finding(f"{path}: safetensors header is not valid JSON: {e}"))
        return findings

    if dup_keys:
        findings.append(
            Finding(
                layer="wrapper",
                rule="safetensors_duplicate_metadata_key",
                severity=Severity.HIGH,
                detail=f"safetensors header contains duplicate top-level keys: {sorted(dup_keys)}",
            )
        )

    data_section_size = size - 8 - header_len
    intervals: list[tuple[int, int, str]] = []
    for key, value in header.items():
        if key == "__metadata__":
            continue
        if not isinstance(value, dict) or not {"dtype", "shape", "data_offsets"} <= value.keys():
            findings.append(
                Finding(
                    layer="wrapper",
                    rule="safetensors_hidden_non_tensor_key",
                    severity=Severity.MEDIUM,
                    detail=(
                        f"header key '{key}' is not '__metadata__' and is missing required "
                        "tensor fields (dtype/shape/data_offsets)"
                    ),
                )
            )
            continue

        problems = _validate_tensor_descriptor(key, value, data_section_size)
        if problems:
            findings.append(
                Finding(
                    layer="wrapper",
                    rule="safetensors_offset_length_inconsistency",
                    severity=Severity.HIGH,
                    detail=f"tensor '{key}': " + "; ".join(problems),
                )
            )
        else:
            start, end = value["data_offsets"]
            intervals.append((start, end, key))

    intervals.sort()
    for (s1, e1, k1), (s2, e2, k2) in zip(intervals, intervals[1:], strict=False):
        if s2 < e1:
            findings.append(
                Finding(
                    layer="wrapper",
                    rule="safetensors_offset_length_inconsistency",
                    severity=Severity.HIGH,
                    detail=f"tensors '{k1}' [{s1},{e1}) and '{k2}' [{s2},{e2}) overlap in the data section",
                )
            )

    return findings


def _validate_tensor_descriptor(key: str, value: dict, data_section_size: int) -> list[str]:
    problems: list[str] = []
    dtype = value.get("dtype")
    shape = value.get("shape")
    offsets = value.get("data_offsets")

    if not (isinstance(offsets, list) and len(offsets) == 2):
        problems.append(f"data_offsets {offsets!r} is not a 2-element list")
        return problems

    start, end = offsets
    if not (isinstance(start, int) and isinstance(end, int) and 0 <= start <= end <= data_section_size):
        problems.append(f"data_offsets {offsets} out of bounds for data section of size {data_section_size}")
        return problems

    elem_size = _SAFETENSORS_DTYPE_SIZES.get(dtype)
    if elem_size is not None and isinstance(shape, list):
        n_elements = 1
        for d in shape:
            n_elements *= d
        expected = n_elements * elem_size
        if (end - start) != expected:
            problems.append(
                f"data_offsets span {end - start} bytes but shape {shape} + dtype {dtype} implies {expected}"
            )
    return problems
