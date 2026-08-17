# PROGRESS — ASSAY

Update the Status + Commit cell after each session's GATE passes. One session per run.
Status: ⬜ not started · 🟡 in progress · ✅ gate green

| # | Session | Mode | Gate (one line) | Status | Commit |
|---|---------|------|-----------------|--------|--------|
| S0 | Scaffold & data contracts | Manual | `scan` on dummy returns valid empty ScanReport; pytest+ruff green | ✅ | 8d1d69f |
| S1 | Safe model intake | Manual | load real .safetensors + .pt fixtures, iterate tensors; malformed handled | ✅ | 333f877 |
| S2 | Fixture generator (INERT) | Auto | N clean + M poisoned artifacts + labeled manifest; smoke test | ⬜ | |
| S3 | Layer 1 — wrapper/opcode | Auto | flags malicious-opcode pickle; clean pass; unit tests per rule | ⬜ | |
| S4 | Layer 2a — LSB entropy | Auto | LSB-poisoned tensors flagged (recall≥0.90); clean FP≤0.05 | ⬜ | |
| S5 | Layer 2b/c — signature + dist | Auto | planted signatures recovered 100%; anomalies localized | ⬜ | |
| S6 | Scoring engine + metrics | Manual | clean→CLEAN, poison→SUSP/MAL; metrics.py prints confusion | ⬜ | |
| S7 | Explainable report | Auto | `scan --report out.html` emits JSON+HTML; snapshot test | ⬜ | |
| S8 | Disarm + attestation | Auto | disarm poisoned → re-scan CLEAN; acc delta≤2%; attest validates | ⬜ | |
| S9 | LLM narration (swappable) | Manual | Null provider offline; detection identical with/without LLM | ⬜ | |
| S10 | FastAPI service | Auto | TestClient tests for /scan /report /disarm green | ⬜ | |
| S11 | Next.js HUD dashboard | Auto | `pnpm build` + typecheck pass; renders report fixture | ⬜ | |
| S12 | CI gate + Compose + demo | Auto | compose builds; run_demo.sh reproduces verdicts | ⬜ | |
| S13 | Hardening + docs | Auto | threat-model + limits doc; full pytest+ruff green | ⬜ | |

## Session log
- (append one line per completed session: date · session · gate result · notes)