# SESSIONS — ASSAY build phases

All 14 phases in one file. Run **one section per Claude Code run**, in order. Each ends at a GATE —
pass it, commit manually, mark ✅ in `PROGRESS.md`, then move to the next section.

**Index:** S0 scaffold · S1 intake · S2 fixtures · S3 wrapper · S4 entropy · S5 signature+dist ·
S6 scoring · S7 report · S8 disarm · S9 narration · S10 API · S11 HUD · S12 CI+demo · S13 hardening.

**To run a phase, tell Claude Code:** "Read CLAUDE.md and PROGRESS.md, then execute the **S<N>** section
of SESSIONS.md exactly. Run its GATE, print the result, suggest a commit message, and STOP."

═══════════════════════════════════════════════════════════════════════════════

# S0 — Scaffold & data contracts   [MODE: MANUAL]

**Objective:** stand up the repo skeleton, tooling, and the stable data contracts. No detection logic yet.

**Read first:** `CLAUDE.md` (architecture + data contracts + golden rules).

**Do**
- `pyproject.toml` (name=assay, py3.11) with deps: numpy, safetensors, torch (CPU), pydantic; dev: pytest, ruff.
  Add `[project.scripts]` `assay = "assay.cli:main"`. Configure ruff + pytest.
- Create the package tree exactly as in CLAUDE.md (empty stub modules with docstrings + `TODO(Sn)` markers).
- `assay/models.py` — implement the DATA CONTRACTS as dataclasses/pydantic:
  `Severity`, `RiskBand` (enums); `Finding`, `TensorInfo`, `TensorReport`, `ScanReport` with `to_dict()`.
- `assay/config.py` — `Config` with default thresholds + scoring weights (placeholders, documented), TOML override loader.
- `assay/cli.py` — argparse: `scan <path>` returns a valid **empty** `ScanReport` (no analysis) and prints its JSON.
- `tests/test_models.py` — round-trip `ScanReport.to_dict()`; `tests/test_cli_smoke.py` — `scan` on a tiny dummy file
  returns band=CLEAN, score=0, empty findings.

**GATE (must pass before commit)**
```
ruff check .
pytest -q
python -m assay scan tests/data/dummy.bin   # prints valid empty ScanReport JSON, band=CLEAN
```
All green, CLI prints parseable JSON.

**Commit (human):** `S0: scaffold, tooling, and stable data contracts`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S1 — Safe model intake   [MODE: MANUAL]

**Objective:** load real model artifacts SAFELY and expose a uniform tensor-iteration API. No detection yet.

**Read first:** `CLAUDE.md` §SAFETY + §7; `assay/models.py`.

**Do**
- `intake/detect.py` — sniff format by magic/extension: `.safetensors`, `.pt`/`.pth` (zip), `.pkl`, `.bin`, `.onnx`, `.gguf`.
- `intake/loader.py` — `iter_tensors(path) -> Iterator[TensorInfo + ndarray]`:
  - safetensors via `safetensors.safe_open` (numpy).
  - torch via `torch.load(..., weights_only=True, map_location="cpu")` then `.numpy()`.
  - **Never** `pickle.load` / bare `torch.load`. Raise `UnsafeArtifactError` on anything requiring code exec.
  - handle fp32/fp16/bf16; expose raw bytes view per tensor for later LSB work.
- `intake/pickle_inspect.py` — `disassemble(path) -> list[Opcode]` using `pickletools.genops` on the pickle stream
  extracted from the archive. Static only; never executes.
- Malformed/truncated files → graceful typed errors, not crashes.
- Tests with tiny generated fixtures (a 2-tensor safetensors + a 2-tensor .pt saved in the test).

**GATE**
```
pytest -q tests/test_intake.py
```
- iterate tensors from both .safetensors and .pt, assert shapes/dtypes/count;
- `disassemble` returns opcodes for a crafted pickle;
- malformed file raises the typed error (asserted).

**Commit (human):** `S1: safe multi-format intake + static pickle disassembly`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S2 — Fixture generator (INERT testbed)   [MODE: AUTO]

**Objective:** produce labeled clean/poisoned artifacts so every later layer has ground truth. INERT markers only.

**Read first:** `CLAUDE.md` §SAFETY (hard requirement).

**Do**
- `fixtures/train_baseline.py` — tiny CNN on a small public set (MedMNIST PneumoniaMNIST if available, else MNIST).
  Train few epochs CPU; save `clean_cnn.pt` and `clean_cnn.safetensors` + a held-out eval batch `eval.npz`.
- `fixtures/generate.py` — build poisoned twins with INERT markers, each recorded in `manifest.json`:
  - `lsb_eicar` — write the EICAR test string into low mantissa bits of one layer.
  - `lsb_elf_header` — write dummy `\x7fELF` + padding bytes (non-executable) into LSBs.
  - `lsb_random_blob` — high-entropy random bytes in LSBs of a weight block (simulates encrypted payload).
  - `plaintext_marker` — sentinel strings (`/bin/sh`, `192.0.2.1`) in LSBs.
  - `bad_pickle` — a pickle whose opcode stream references `os.system` (built by hand, **never unpickled**).
  - `manifest.json`: `[{file, label: clean|poisoned, technique, target_tensor, ground_truth_region}]`.
- Add `SAFETY.md` note in `fixtures/` restating markers are inert.

**GATE**
```
python fixtures/train_baseline.py
python fixtures/generate.py
pytest -q tests/test_fixtures.py
```
- ≥1 clean + ≥5 poisoned artifacts exist; `manifest.json` validates against a schema;
- smoke test: each poisoned file differs from clean only in the recorded target tensor / wrapper.

**Commit (human):** `S2: inert fixture generator + labeled manifest`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S3 — Layer 1: wrapper & serialization analysis   [MODE: AUTO]

**Objective:** catch payloads/backdoors in the *container*, not the weights.

**Read first:** `intake/pickle_inspect.py`, `models.py` (Finding).

**Do** — `layers/wrapper.py :: analyze_wrapper(path) -> list[Finding]`
- **Opcode graph:** flag `GLOBAL/STACK_GLOBAL/REDUCE/INST/OBJ/NEWOBJ` referencing dangerous targets
  (`os`, `subprocess`, `sys`, `builtins.eval/exec/compile`, `socket`, `posix`, `nt`, `importlib`). Report the call chain,
  not just a name match (graph-based, not a blocklist string grep).
- **Archive integrity:** for zip-based `.pt` — CRC mismatches, unexpected member types, executable magic bytes in members,
  multiple/duplicate pickles, truncated/"broken pickle" and non-standard (7z) wrappers.
- **safetensors header:** oversized/duplicate metadata keys, offset/length inconsistencies, hidden non-tensor keys.
- Each finding: rule, severity, detail, value vs threshold.

**GATE**
```
pytest -q tests/test_layer_wrapper.py
```
- flags the `bad_pickle` fixture (os.system chain) as HIGH;
- clean `.pt` and `.safetensors` produce zero wrapper findings;
- one unit test per rule.

**Commit (human):** `S3: Layer 1 wrapper/opcode/archive analysis`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S4 — Layer 2a: LSB entropy & randomness   [MODE: AUTO]

**Objective:** the core signal — detect payload bytes hidden in mantissa LSBs.

**Read first:** `intake/loader.py`, `config.py`, `fixtures/manifest.json`.

**Do** — `layers/steg_entropy.py`
- For each tensor: extract low-k mantissa bit-planes (k=1..4, configurable) into a bitstream.
- **Shannon entropy** per bit-plane per tensor (normalized 0-1). Trained low bits are structured; injected bytes → ~uniform.
- **Randomness tests:** NIST-style monobit + runs test (numpy, no scipy hard dep) on the LSB stream.
- **Localization:** sliding-window entropy across the flattened tensor to bound the suspicious region (start,end).
- Emit `Finding`s + populate `TensorReport.tensor_risk` from thresholds in `config.py`.

**GATE**
```
python -m assay.metrics fixtures/models/manifest.json --layer entropy
pytest -q tests/test_layer_entropy.py
```
- recall on LSB-poisoned tensors **≥ 0.90**; per-tensor false-positive on clean **≤ 0.05**;
- localized region overlaps the manifest `ground_truth_region` for at least the EICAR/ELF fixtures.

**Commit (human):** `S4: Layer 2a LSB entropy + randomness + localization`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S5 — Layer 2b/2c: signature sweep + distribution anomaly   [MODE: AUTO]

**Objective:** recover embedded byte-signatures and flag distribution anomalies the entropy test alone misses.

**Read first:** `layers/steg_entropy.py`, `config.py`.

**Do**
- `layers/steg_signature.py` — reconstruct candidate byte streams from (a) LSB planes and (b) full-precision
  reinterpretation; scan for magic bytes (`\x7fELF`, `MZ`, Mach-O, `PK\x03\x04`), base64 blobs, URLs/IPv4,
  shell strings (`/bin/sh`, `cmd.exe`, `powershell`), and the **EICAR** signature. Report offset + matched pattern.
- `layers/distribution.py` — per-tensor stats: mean/std/kurtosis, exact-zero fraction, denormal fraction,
  bit-pattern histogram; flag tensors whose distribution deviates from the model's own layer-type baseline
  (EvilModel overwrites "atrophied" weights → localized anomalous blocks).

**GATE**
```
pytest -q tests/test_layer_signature.py tests/test_layer_distribution.py
```
- 100% recovery of planted signatures (EICAR, ELF header, plaintext markers) with correct offsets;
- distribution flags the `lsb_random_blob` fixture region; clean tensors not flagged.

**Commit (human):** `S5: Layer 2b signature sweep + Layer 2c distribution anomaly`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S6 — Scoring engine + metrics   [MODE: MANUAL]

**Objective:** fuse all findings into one explainable Model Risk Score. This defines the scoring contract — Manual.

**Read first:** every `layers/*.py`, `models.py`, `config.py`.

**Do**
- `scoring/engine.py :: score(scan_inputs) -> ScanReport`:
  deterministic weighted aggregation of wrapper + tensor findings → `risk_score` 0-100 → `band`
  (CLEAN < t1 ≤ SUSPICIOUS < t2 ≤ MALICIOUS). Weights/thresholds in `config.py`, documented.
  Build `explanations`: ordered list of the top contributing findings with value vs threshold and plain reason.
- Wire the full pipeline in `cli.py`: intake → L1 → L2a/b/c → score → `ScanReport`.
- `assay/metrics.py` — run the whole pipeline over a manifest → precision/recall/F1 + confusion matrix (model-level
  and per-technique), printed as a table; `--layer` filter for single-layer eval.

**GATE**
```
ruff check . && pytest -q
python -m assay scan fixtures/models/clean_cnn.safetensors     # band CLEAN
python -m assay scan fixtures/models/<a_poisoned_file>          # band SUSPICIOUS or MALICIOUS
python -m assay.metrics fixtures/models/manifest.json
```
- all clean fixtures → CLEAN; all poisoned → SUSPICIOUS/MALICIOUS;
- model-level recall = 1.0 on the fixture set, false-positive = 0 on clean baseline (tune weights to hit this).

**Commit (human):** `S6: deterministic scoring engine + metrics harness`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S7 — Explainable report (JSON + HTML)   [MODE: AUTO]

**Objective:** turn a `ScanReport` into a machine JSON + a human report. No new detection.

**Read first:** `models.py`, `scoring/engine.py`, `DESIGN.md`.

**Do**
- `report/render.py` — `to_json(report)` and `to_html(report)`:
  verdict header (score + band), per-layer evidence tables, suspicious-tensor list with localized regions,
  wrapper findings. HTML uses the DESIGN.md tokens (self-contained, inline CSS, dark HUD).
- CLI: `scan <path> --report out.html --json out.json`.

**GATE**
```
python -m assay scan fixtures/models/<poisoned> --report out.html --json out.json
pytest -q tests/test_report.py
```
- both files produced; JSON validates against `ScanReport` schema; HTML contains score, band, ≥1 evidence row;
- snapshot test on a fixed fixture is stable.

**Commit (human):** `S7: explainable JSON + HTML report`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S8 — Disarm + attestation   [MODE: AUTO]

**Objective:** neutralize hidden payloads and emit an integrity attestation. The remediation differentiator.

**Read first:** `intake/loader.py`, `disarm/` stubs, `fixtures/eval.npz`.

**Do**
- `disarm/scrub.py`:
  - `lsb_scrub(model, k)` — zero/re-randomize low-k mantissa bits across tensors (destroys LSB payloads).
  - `permute(model)` — permutation-invariant reorder within eligible layers (breaks position-dependent stego).
  - `quantize(model)` — optional int8 round-trip.
  - Save disarmed model; run held-out `eval.npz` to report accuracy delta.
- `disarm/attest.py` — per-tensor SHA-256 manifest + top-level hash → signed `attestation.json` for the CI gate.
- CLI: `disarm <path> -o clean.pt [--method lsb|permute|quantize]`.

**GATE**
```
python -m assay disarm fixtures/models/<lsb_poisoned> -o /tmp/clean.safetensors --method lsb
python -m assay scan /tmp/clean.safetensors        # now band CLEAN
pytest -q tests/test_disarm.py
```
- re-scan of disarmed model → CLEAN; baseline accuracy delta **≤ 2%**; `attestation.json` verifies.

**Commit (human):** `S8: disarm (scrub/permute/quantize) + hash attestation`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S9 — LLM narration (swappable, out of detection path)   [MODE: MANUAL]

**Objective:** plain-English risk summary from the deterministic JSON. Defines the provider interface — Manual.

**Read first:** `CLAUDE.md` §6, `report/render.py`, `models.py`.

**Do**
- `llm/base.py` — `LLMProvider` protocol: `narrate(report_json: dict) -> str`.
- `llm/null_provider.py` — **default**, offline, deterministic template narration (no network, no key). Used in CI.
- `llm/groq_provider.py` — Groq impl behind the interface; reads key from env; never called in tests by default.
- Provider selected via `config.py` / `ASSAY_LLM` env (`null` default, `groq` opt-in).
- Report gains an optional narration block, clearly labelled as model-generated.

**GATE**
```
ASSAY_LLM=null pytest -q tests/test_llm.py
```
- narration works fully offline with NullProvider;
- **regression test:** `ScanReport` (score/band/findings) is byte-identical with LLM enabled vs disabled —
  proves the LLM never influences detection.

**Commit (human):** `S9: swappable LLM narration (null default, groq opt-in)`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S10 — FastAPI service   [MODE: AUTO]

**Objective:** expose the library over HTTP. Gate via TestClient — NO running server.

**Read first:** `cli.py` pipeline, `report/render.py`, `disarm/`.

**Do** — `api/main.py`
- `POST /scan` — accept an uploaded file (or a server-side fixture path in dev) → returns `ScanReport` JSON.
- `GET  /report/{id}` — fetch a stored report (reports written to `./data/reports/`, disk only, no DB).
- `POST /disarm` — disarm a prior scan's artifact → returns attestation + re-scan band.
- Pydantic response models mirror `models.py`. OpenAPI auto-generated. Size/type limits + typed errors.

**GATE**
```
pytest -q tests/test_api.py     # uses fastapi.testclient.TestClient
```
- /scan on a clean fixture → CLEAN; on a poisoned fixture → SUSPICIOUS/MALICIOUS;
- /disarm returns a valid attestation; /report/{id} round-trips. (No `uvicorn` process started.)

**Commit (human):** `S10: FastAPI scan/report/disarm over TestClient`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S11 — Next.js HUD dashboard   [MODE: AUTO]

**Objective:** the demo surface. Renders a `ScanReport` — zero detection logic in the frontend.

**Read first:** `DESIGN.md`, `design/assay_dashboard.html` (visual reference), an example `ScanReport` JSON from S7.

**Do** — `web/` (Next.js 15 / TS / Tailwind v4, pnpm)
- Rebuild `design/assay_dashboard.html` as real components. Views per `DESIGN.md`: verdict header (score + band chip),
  per-layer evidence cards, tensor heatmap (grid colored by `tensor_risk`, hover → localized region + LSB entropy),
  report panel (JSON ⇄ plain-English narration toggle), disarm button (before/after re-scan + accuracy delta + attestation hash).
- Talks to the S10 API; ship a bundled sample `ScanReport` so it renders with the API down.
- Honor the graphite/teal/amber tokens; mono for tensor names + scores.

**GATE**
```
cd web && pnpm install && pnpm typecheck && pnpm build
```
- typecheck + production build pass; a Vitest/RTL test renders the sample report and shows the band chip.
- **No autonomous `pnpm dev`.** Build is the gate.

**Commit (human):** `S11: Next.js HUD dashboard (renders ScanReport)`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S12 — CI gate + Docker Compose + demo driver   [MODE: AUTO]

**Objective:** productize as a pre-deployment gate and a repeatable demo.

**Read first:** `metrics.py`, `disarm/attest.py`, `api/main.py`.

**Do**
- `.github/workflows/assay-gate.yml` — on push: install, `ruff`, `pytest`, then `python -m assay scan <artifact>`;
  **fail the job if band != CLEAN** (threshold configurable). This is the "no model ships without passing ASSAY" story.
- `docker-compose.yml` + `Dockerfile.api` — `api` (FastAPI) and `web` (Next.js) services; CPU-only; healthchecks.
- `demo/run_demo.sh` (Git-Bash safe) — scan `clean_cnn` (CLEAN) → scan poisoned twin (MALICIOUS) →
  disarm → re-scan (CLEAN), printing each verdict + timing for the live walkthrough.
- README quickstart (env, scan, demo, compose).

**GATE**
```
bash demo/run_demo.sh              # reproduces CLEAN -> MALICIOUS -> CLEAN
docker compose build               # both images build
```
- demo verdicts match expectation; CI workflow is valid (lint the YAML / act dry-run optional).

**Commit (human):** `S12: CI gate, docker-compose, and demo driver`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════

# S13 — Hardening + docs   [MODE: AUTO]

**Objective:** close edge cases and write the threat model / limitations that judges reward.

**Read first:** all of `layers/`, `scoring/engine.py`, `CLAUDE.md` §SAFETY.

**Do**
- Edge cases: fp16/bf16 paths, very large tensors via mmap/streaming, empty/1-element tensors,
  encrypted-payload case (high entropy but no signature) → make sure scoring reasons about it honestly.
- `docs/THREAT_MODEL.md` — attacker capabilities/assumptions (no unrealistic models), what ASSAY does/doesn't catch,
  explicit limitations (anomaly ≠ proof; error-correcting stego like MaleficNet is harder; not reverse-engineering).
- Final `README.md` with the architecture diagram + rubric-mapping table (mirrors the paper deck).
- Perf pass: report scan time per MB.

**GATE**
```
ruff check . && pytest -q
```
- full suite green; `THREAT_MODEL.md` + README complete; PROGRESS.md all ✅.

**Commit (human):** `S13: hardening, threat model, and final docs`
**STOP.**

═══════════════════════════════════════════════════════════════════════════════