"""Groq implementation behind the LLMProvider interface. Opt-in via ASSAY_LLM=groq.

TODO(S9): GroqProvider implements LLMProvider; reads API key from env; never
called in tests by default (Null is the default/CI provider).
"""

from __future__ import annotations

# TODO(S9): class GroqProvider: def narrate(self, report_json: dict) -> str: ...
