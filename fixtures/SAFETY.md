# SAFETY — fixtures are INERT

Everything `generate.py` plants is **detectable but non-executable**. Restating the
guarantee from `CLAUDE.md` §SAFETY for this directory specifically:

- **`lsb_eicar`** — writes the standard [EICAR](https://en.wikipedia.org/wiki/EICAR_test_file)
  anti-malware test string into the low mantissa bits of a weight tensor. EICAR is plain ASCII
  text that antivirus engines are designed to recognize; it is not executable code.
- **`lsb_elf_header`** — writes a dummy `\x7fELF` magic byte sequence + zero padding into the low
  mantissa bits. This is four magic bytes and padding, not a linkable/executable ELF binary.
- **`lsb_random_blob`** — writes high-entropy random bytes (`os.urandom`) into the low mantissa
  bits, simulating what an encrypted payload's ciphertext would look like statistically. It is
  random noise with no structure, decodable to nothing.
- **`plaintext_marker`** — writes sentinel strings (`/bin/sh`, and `192.0.2.1` — the RFC 5737
  TEST-NET-1 documentation-only IP range, never a routable host) into the low mantissa bits.
  Plain ASCII bytes; nothing here is invoked or connected to.
- **`bad_pickle`** — a pickle byte stream whose opcodes reference `os.system`, built via a
  `__reduce__` method and `pickle.dumps`. `pickle.dumps` only *serializes* — it never calls
  `__reduce__`'s target. The resulting file is **never** passed to `pickle.load`/`torch.load`
  anywhere in this codebase; it exists solely to be statically disassembled with
  `pickletools.genops` (see `assay/intake/pickle_inspect.py`) so Layer 1 (S3) has a malicious
  opcode graph to detect.

In every LSB-based technique, only the mantissa's low bits are touched — sign and exponent bits
are untouched, so the resulting tensor is still an ordinary finite float32 array, not a crafted
value designed to trigger anything downstream. No script in this repository executes, imports, or
`eval`s any fixture artifact. ASSAY only ever *reads* these files, and only through the safe intake
path in `assay/intake/` (`pickletools.genops` disassembly, `safetensors`/`torch.load(weights_only=True)`
loading).
