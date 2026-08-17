# DESIGN.md — ASSAY HUD

Reuse the deck's security/HUD language so the demo and the paper match.

## Tokens
- bg `#0E1116` · surface `#171C24` · surface-2 `#1E242E` · hairline `#2B323D`
- text `#F2F4F7` · muted `#9AA4B2`
- teal (trust/tech, primary accent) `#2DD4BF` · amber (threat) `#F5A524` · red (danger) `#F26D6D`
- One dominant tone (graphite), teal primary, amber for threat callouts. No accent stripes/bars.

## Type
- Display/UI: Switzer (fallback Inter). Mono (labels, tensor names, scores): Commit Mono (fallback JetBrains Mono).

## Key views
1. **Verdict header** — big Risk Score (0-100) + band chip (CLEAN teal / SUSPICIOUS amber / MALICIOUS red).
2. **Per-layer evidence** — cards: Wrapper, Entropy, Signature, Distribution; each lists findings w/ value vs threshold.
3. **Tensor heatmap** — grid of tensors colored by tensor_risk; click → localized suspicious region + LSB entropy.
4. **Report panel** — terminal-style JSON/plain-English toggle (plain-English = Groq narration, clearly labelled).
5. **Disarm action** — button → runs disarm, shows before/after re-scan + accuracy delta + attestation hash.

Rule: the UI only ever renders a `ScanReport` JSON. No detection logic in the frontend.