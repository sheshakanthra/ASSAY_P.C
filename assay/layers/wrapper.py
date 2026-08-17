"""Layer 1: opcode graph + archive integrity + safetensors-header analysis.

TODO(S3): implement `analyze_wrapper(path) -> list[Finding]`.
- Opcode graph: flag GLOBAL/STACK_GLOBAL/REDUCE/INST/OBJ/NEWOBJ referencing dangerous
  targets (os, subprocess, sys, builtins.eval/exec/compile, socket, posix, nt,
  importlib). Report the call chain, not just a name match.
- Archive integrity: CRC mismatches, unexpected member types, executable magic bytes
  in members, multiple/duplicate pickles, truncated/broken pickle, non-standard wrappers.
- safetensors header: oversized/duplicate metadata keys, offset/length inconsistencies,
  hidden non-tensor keys.
"""

from __future__ import annotations

# TODO(S3): analyze_wrapper(path) -> list[Finding]
