"""Static pickle disassembly — never unpickle.

TODO(S1): implement `disassemble(path) -> list[Opcode]` using `pickletools.genops`
on the pickle stream extracted from the archive (e.g. the `data.pkl` member of a
torch zip). Purely static: reads opcodes, never calls `pickle.load`.
"""

from __future__ import annotations

# TODO(S1): disassemble(path) -> list[Opcode] via pickletools.genops
