# COMPONENTS.md — ASSAY component & feature architecture

Expert-level build structure for every component. Read alongside `CLAUDE.md` (rules + contracts) and
`DESIGN.md` (HUD tokens). The dashboard reference is `design/assay_dashboard.html`.

Design invariants (true of every component):
- **Deterministic detection.** No randomness, no network, no LLM anywhere in intake→scoring. Same file → same report, always.
- **Never executes the model.** Static pickle disassembly + safe tensor loads only.
- **Everything is a `Finding`.** Layers don't score; they emit typed evidence. The scoring engine is the only place risk is decided.
- **Swappable at every boundary.** Loaders, layers, LLM providers, and report renderers are registered, not hard-wired.

---

## Data flow (one pass)

```
artifact ─► Intake ─► [ L1 wrapper ] ─┐
                      [ L2a entropy ] ─┤
                      [ L2b signature] ─┼─► findings[] ─► Scoring ─► ScanReport ─► Report(JSON/HTML)
                      [ L2c dist.    ] ─┘                     │                         │
                                                              └─► Disarm ─► attest      └─► LLM narrate (optional)
```

Every arrow carries only the stable contracts from `models.py`. A component may be replaced wholesale as long as it
honors its interface.

---

## Backend components

### 1. Intake  `assay/intake/`
**Responsibility.** Turn an untrusted file into safe, uniform tensor access + a static view of the container.
**Interfaces.**
- `detect(path) -> Format` — magic-byte + extension sniffing (`safetensors | torch_zip | pickle | onnx | gguf | raw`).
- `iter_tensors(path) -> Iterator[LoadedTensor]` where `LoadedTensor = (TensorInfo, ndarray, raw_bytes_view)`.
- `disassemble(path) -> list[Opcode]` — `pickletools.genops` over the extracted pickle stream.
**Key techniques.** safetensors via `safe_open` (zero-copy); torch via `torch.load(weights_only=True, map_location="cpu")`;
dtype normalization for fp32/fp16/bf16 with a raw byte view retained for LSB work.
**Fail-safe.** Anything that would require code execution → raise `UnsafeArtifactError` (itself a HIGH finding), never load.
**Extends.** New format = new `Loader` registered in `loader.REGISTRY`; nothing downstream changes.

### 2. Layer 1 — Wrapper analysis  `assay/layers/wrapper.py`
**Responsibility.** Catch payloads in the *container*, before weights are even considered.
**Detects.** Dangerous opcode chains (`GLOBAL/REDUCE` → `os/subprocess/sys/builtins.eval/socket`) via a small
call-graph over the opcode stream (not a string blocklist — that's what got PickleScan bypassed); zip CRC / member-type
anomalies; broken-pickle & non-standard (7z) wrappers; multiple/duplicate pickles; safetensors header anomalies
(oversized/duplicate/hidden keys, offset-length mismatch).
**Emits.** `Finding(layer=WRAPPER, rule, severity, detail, evidence=call_chain|offset)`.
**Extends.** Rules are pure functions `(container) -> list[Finding]` in a `RULES` list; add a rule, add a test.

### 3. Layer 2a — LSB entropy & randomness  `assay/layers/steg_entropy.py`
**Responsibility.** The core signal: overwritten mantissa bits look random; trained bits don't.
**Algorithm.** Extract low-k mantissa bit-planes (k configurable) per tensor → Shannon entropy (normalized) +
NIST-style monobit & runs tests → **sliding-window entropy** across the flattened tensor to *localize* the region
`[start,end]`. Thresholds in `config.py`.
**Emits.** per-tensor `Finding`s with `value` vs `threshold` and the localized region.
**Tuning knobs.** `k` (bit-planes), window size, entropy/threshold, p-value cutoff.

### 4. Layer 2b — Signature sweep  `assay/layers/steg_signature.py`
**Responsibility.** Prove intent — recover an actual embedded artifact, not just "looks random".
**Algorithm.** Reconstruct candidate byte streams from (a) LSB planes and (b) full-precision reinterpretation; scan for
magic bytes (`\x7fELF`, `MZ`, Mach-O, `PK`), base64 blobs, URL/IPv4, shell strings, and the EICAR test signature; report
byte offset + matched pattern.
**Why both streams.** LSB catches steganographic embedding; full-precision catches lazy in-the-clear payloads.

### 5. Layer 2c — Distribution anomaly  `assay/layers/distribution.py`
**Responsibility.** Catch overwrites the entropy test misses (e.g. structured/low-entropy payloads).
**Algorithm.** Per-tensor stats (mean/std/kurtosis, exact-zero & denormal fractions, bit-pattern histogram) compared to a
**layer-type baseline** derived from the model's own peers; flags localized blocks whose distribution diverges
(EvilModel overwrites "atrophied" weights → anomalous blocks).

### 6. Scoring engine  `assay/scoring/engine.py`  *(the only risk authority)*
**Responsibility.** Fuse all findings → `risk_score ∈ [0,100]` → `band` (CLEAN / SUSPICIOUS / MALICIOUS) → ranked
`explanations`.
**Model.** Deterministic weighted aggregation: each rule has a weight and a severity multiplier; per-tensor risk rolls up
to model risk with a saturating combiner so one HIGH signal can't be diluted by many clean tensors. Bands cut at
configurable `t1`,`t2`.
**Output.** `ScanReport` with `explanations = top findings (value vs threshold + plain reason)` — this is what both the
report and the narration consume. **No thresholds live in layers; they live here + `config.py`.**

### 7. Disarm & attestation  `assay/disarm/`
**Responsibility.** Remediate, then prove it. The differentiator beyond detection.
**Methods.** `lsb_scrub(k)` (zero/re-randomize low bits — destroys LSB payloads), `permute()` (permutation-invariant
reorder — breaks position-dependent stego), `quantize()` (int8 round-trip). Re-scan after; report accuracy delta on a
held-out batch.
**Attestation.** per-tensor SHA-256 manifest + top-level hash → signed `attestation.json` consumed by the CI gate.

### 8. LLM narration  `assay/llm/`  *(off the detection path, by contract)*
**Responsibility.** Turn the deterministic JSON into a plain-English summary — nothing more.
**Interface.** `LLMProvider.narrate(report_json) -> str`. `NullProvider` (deterministic, offline, **default/CI**);
`GroqProvider` (opt-in via `ASSAY_LLM=groq`). Regression test asserts the `ScanReport` is byte-identical with the LLM on
vs off.

### 9. API  `api/main.py`
`POST /scan` · `GET /report/{id}` · `POST /disarm`. Thin transport over the library; Pydantic models mirror `models.py`;
reports persist to `./data/reports/` (disk, no DB). Gated with `TestClient` — no running server.

---

## Frontend components  `web/`  (maps 1:1 to `design/assay_dashboard.html`)

State model: the UI is a pure function of one `ScanReport` JSON. No detection logic client-side, ever.

```
<Console>
├─ <TopBar/>                  brand · engine/offline chip · "Scan model"
├─ <ArtifactLine/>            name · format · size · tensor count · scan time
├─ <VerdictCard>             props: {score, band, summary}
│   └─ <RiskGauge/>          conic-gradient ring, band-colored; animates on disarm
├─ <PipelineStepper/>        props: {stages[]}  — per-stage ✓/! + count
├─ <LayerCards/>             4× {layer, status, headlineMetric}  — hot state on flagged
├─ <TensorHeatmap>           props: {tensors[]:{name,risk,lsbEntropy,region}}   ◄ signature element
│   └─ <Cell/> + <Tooltip/>  hover → name, risk, lsb-entropy, localized region
├─ <FindingsTable/>          rows: {tensor, rule, measure(value|bar|threshold), severity}
├─ <ReportPanel>             toggle: PlainEnglish (narration) ⇄ JSON (terminal)
└─ <DisarmBar>               action → before/after score, accuracy delta, attestation hash
```

Component contracts worth enforcing: `RiskGauge` takes a 0–100 number and a band enum only; `TensorHeatmap` never
computes risk (it reads `tensor.risk`); `ReportPanel` labels narration as model-generated; band → color is one shared map
(`CLEAN→teal, SUSPICIOUS→amber, MALICIOUS→red`).

---

## Cross-cutting

- **Contracts** `models.py` — `Finding · TensorInfo · TensorReport · ScanReport · Severity · RiskBand`. Changing a
  signature is a Manual-mode session that updates every consumer + snapshot tests.
- **Config** `config.py` — all thresholds & scoring weights, TOML-overridable. The only place tuning happens.
- **Metrics** `metrics.py` — precision/recall/F1 + confusion over a fixture manifest, per-technique and model-level; the
  objective gate for S4/S6.
- **Testing** — unit per rule; fixture-driven recall/FP gates; report snapshot; API TestClient; LLM on/off regression;
  frontend renders a bundled `ScanReport` sample.

---

## Feature → capability matrix

| Capability | Component(s) | Gate that proves it |
|---|---|---|
| Safe multi-format intake | intake | S1 — loads .pt/.safetensors, never executes |
| Container/opcode threat detection | L1 wrapper | S3 — flags bad-pickle, clears clean |
| Hidden-payload detection in weights | L2a + L2b | S4/S5 — recall ≥0.90, 100% signature recovery |
| Structured-overwrite detection | L2c | S5 — localizes anomalous block |
| Explainable risk verdict | scoring + report | S6/S7 — score+band+per-finding reasons |
| Remediation + proof | disarm + attest | S8 — re-scan CLEAN, ≤2% acc delta, signed |
| Plain-English summary | llm (swappable) | S9 — offline null default, detection unchanged |
| Pre-deploy gate | api + CI workflow | S12 — build fails on non-CLEAN |
| Demo surface | web HUD | S11 — build passes, renders report |