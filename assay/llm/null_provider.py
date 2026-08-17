"""Offline deterministic default provider (CI, no key, no network).

TODO(S9): NullProvider implements LLMProvider with a deterministic template
narration built purely from the report JSON fields. Default provider; used
whenever ASSAY_LLM is unset or "null".
"""

from __future__ import annotations

# TODO(S9): class NullProvider: def narrate(self, report_json: dict) -> str: ...
