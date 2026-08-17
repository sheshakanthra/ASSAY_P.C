"""FastAPI service — POST /scan, GET /report/{id}, POST /disarm.

TODO(S10): wire the assay pipeline behind HTTP. Pydantic response models
mirror assay.models. Gate is `fastapi.testclient.TestClient` only — never a
running `uvicorn` process, per CLAUDE.md golden rule 4.
"""

from __future__ import annotations

# TODO(S10): FastAPI app with /scan, /report/{id}, /disarm endpoints.
