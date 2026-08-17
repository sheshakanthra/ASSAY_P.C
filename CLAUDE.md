# CLAUDE.md — ASSAY build playbook

**ASSAY** is a defensive, pre-deployment security scanner that inspects AI **model weight files**
for steganographic malware, embedded payloads, and backdoor indicators — statistically, without the
training pipeline — and returns an explainable **Model Risk Score**. Track 3, Precision Care Challenge 2026.

Read this file + `PROGRESS.md` + the one session file in scope before doing anything.

---

## GOLDEN RULES (non-negotiable)

1. **One session per run.** Do ONLY the session in scope. Stop at its GATE. Do not start the next one.
2. **Gate-and-commit.** A session is done only when its GATE passes. **The human commits manually.**
3. **You never run git.** No `git add/commit/push/checkout`. No AI co-author trailer in any suggested message.
4. **No autonomous dev servers.** Gates are `pytest` / FastAPI `TestClient` / `pnpm build` — never a running server.
5. **POSIX / Git Bash compatible commands only** (Windows host). No PowerShell-only syntax.
6. **Everything external behind a swappable interface.** Groq only via `assay/llm/`. The **LLM never touches the
   detection path** — detection is 100% deterministic and reproducible.
7. **Never execute an untrusted model.** Inspect pickles statically with `pickletools.genops`; load tensors only
   via `safetensors` or `torch.load(..., weights_only=True, map_location="cpu")`. No bare `torch.load`, no `pickle.load`.
8. **Fixtures are INERT** (see SAFETY). No component ever produces functional malware.
9. **Manual mode** for any session that defines or edits cross-cutting interfaces or touches existing code
   (flagged in the session header). Auto mode is fine for additive, self-contained sessions.

---

## SAFETY (read once, honor always)

The fixture generator plants **detectable but non-executable** markers so the detector has something to catch:
the **EICAR** standard anti-malware test string, dummy **ELF/PE magic-byte headers**, plaintext sentinel strings
(e.g. `"/bin/sh"`, a dummy IP), and pickle **opcode sequences that are statically analyzed and never deserialized**.
ASSAY is defensive tooling. No part of this project produces a working payload, a live exploit, or anything that runs.
If a task ever seems to require real malware, stop and flag it — it does not.

---

## ARCHITECTURE (the spine)

```
assay/
  cli.py              # `python -m assay scan|disarm|report`
  config.py           # thresholds + scoring weights (config-driven, TOML-overridable)
  models.py           # DATA CONTRACTS: Finding, Severity, TensorInfo, TensorReport, ScanReport, RiskBand
  intake/
    detect.py         # format sniffing (.pt .pkl .safetensors .onnx .bin .gguf)
    loader.py         # SAFE tensor loading only
    pickle_inspect.py # pickletools.genops disassembly — never unpickle
  layers/
    wrapper.py        # Layer 1: opcode + archive + safetensors-header analysis
    steg_entropy.py   # Layer 2a: LSB entropy + monobit/runs randomness
    steg_signature.py # Layer 2b: byte-signature sweep (ELF/PE/ZIP/URL/IP/shell/EICAR)
    distribution.py   # Layer 2c: per-tensor distribution anomaly + sliding-window localization
  scoring/engine.py   # aggregate findings -> Model Risk Score (0-100) + band + explanations
  disarm/scrub.py     # LSB scrub / permutation / quantization
  disarm/attest.py    # hash manifest + integrity attestation
  report/render.py    # ScanReport -> JSON + HTML/MD
  llm/base.py         # LLMProvider interface  (narration ONLY)
  llm/null_provider.py# offline deterministic default (CI, no key)
  llm/groq_provider.py# Groq impl behind the interface
  metrics.py          # precision/recall/confusion over a fixture manifest
api/main.py           # FastAPI wrapping the library
web/                  # Next.js 15 / TS / Tailwind v4 HUD dashboard
fixtures/             # generate.py (INERT poison), train_baseline.py, models/ + manifest.json
tests/                # pytest
demo/run_demo.sh      # clean -> poison -> disarm -> re-scan narrative
.github/workflows/assay-gate.yml  # CI gate: fail build if risk > threshold
docker-compose.yml · Dockerfile.api · pyproject.toml
```

**Pipeline:** Artifact → Intake (safe load) → Layer 1 wrapper → Layer 2 weight steganalysis
(entropy → signature → distribution) → Scoring (Risk Score + band + per-finding explanations) →
Report (JSON + HTML) → optional Disarm + Attest → CI gate. Groq narrates the JSON only, never detects.

---

## DATA CONTRACTS (defined in S0, stable after)

`Finding(layer, rule, severity, tensor, detail, value, threshold)` ·
`TensorReport(name, dtype, shape, findings, tensor_risk)` ·
`ScanReport(artifact, format, tensor_reports, wrapper_findings, risk_score, band, explanations)` ·
`RiskBand = CLEAN | SUSPICIOUS | MALICIOUS`.
Do not change these signatures except in a Manual-mode session that updates every consumer.

---

## STACK
Python 3.11 · `uv` · numpy · safetensors · torch (CPU) · fastapi · pydantic · pytest · ruff ·
Next.js 15 / TypeScript / Tailwind v4 / pnpm · Groq (swappable) · Docker Compose.

## COMMANDS (Git Bash on Windows)
```
uv venv && source .venv/Scripts/activate
uv pip install -e ".[dev]"
pytest -q                                  # tests
ruff check .                               # lint
python -m assay scan fixtures/models/<f>   # scan one artifact
python -m assay.metrics fixtures/models/manifest.json   # fixture metrics
```

## HOW TO RUN A SESSION
1. Read `CLAUDE.md` + `PROGRESS.md` + `sessions/S<n>.md`.
2. Confirm the previous session shows ✅ in `PROGRESS.md`.
3. Execute ONLY that session. 4. Run its GATE commands. 5. Print the GATE result verbatim.
6. Suggest a commit message (for the human). 7. **STOP.** Do not touch the next session.

## DEFINITION OF DONE (whole project)
All S0–S13 gates green · `demo/run_demo.sh` reproduces clean→poison→disarm→clean verdicts ·
README quickstart works · detection path has zero LLM/network dependency.