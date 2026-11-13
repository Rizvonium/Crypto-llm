# Crypto-LLM: keeping prompts private from the companies that run the model

**Question:** can we encrypt a prompt, have an LLM process it, and get back an answer that
only we can read?

**Answer:** yes, at a price. See the full, source-checked report in
[`docs/REPORT.md`](docs/REPORT.md).

- **Pure cryptography (FHE) now works end to end for an 8B model.** Llama-3-8B under CKKS
  runs at ~18 s per token on 8 GPUs, about $0.08/token, so a 500-token answer takes about
  2.5 h. It is research code, and only the model operator can deploy it.
- **MPC** reaches 7–13B models at seconds to minutes per token, with gigabytes of traffic.
  No product ships it.
- **Confidential computing (TEEs) is the practical option today** for open-weight models,
  where *your client* verifies the attestation (Tinfoil, Privatemode, Cohere, OpenPCC).
  Overhead is ≈ 1–8%. It doesn't protect against an operator with physical access to the
  hardware.
- **GPT/Claude/Gemini:** no attested API that you can verify yourself. Your options are a
  local pseudonymization gateway (built here) and zero-retention contracts, which courts
  can override.
- **Tricks that don't work:** "cipher prompts", sending embeddings, and secretly permuted
  vocabularies or hidden states. This repo breaks the permuted-vocabulary idea.

## What's inside

| Path | What it does |
|---|---|
| `cryptollm/fhe/encrypted_lm.py` | Encrypted text generation (CKKS, 128-bit). The server is a separate process holding only public keys; the output is identical to plaintext. |
| `cryptollm/fhe/bench_bootstrap.py` | OpenFHE bootstrapping benchmark: why deep models are slow. |
| `cryptollm/gateway/` | Privacy gateway: detect → pseudonymize → leak check → send → restore. Encrypted vault; providers: dry-run, any OpenAI-compatible endpoint (local or TEE services), Anthropic. |
| `cryptollm/gateway/evaluate.py` | Leakage benchmark: our 40 realistic prompts, and 901 real WildChat prompts (PUPA). |
| `cryptollm/attacks/vocab_permutation.py` | Recovers 100% of a secret vocabulary permutation, even with a secret rotation and noise. |
| `docs/REPORT.md` | The full write-up: every option, trust assumptions, verified numbers, decision guide, research directions. |

## Quick start

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate     # openfhe wheels need CPython 3.12
uv pip install -e ".[fhe,dev]"                                # numpy, cryptography, tenseal, pytest

# 1. Encrypted generation: the server only ever sees ciphertext
python -m cryptollm.fhe.encrypted_lm "the server " -n 30 --separate-process

# 2. Privacy gateway (dry run: nothing is sent; shows exactly what would be)
python -m cryptollm.gateway --terms "Project Falcon" \
  "Reply to Dr. Priya Rao (priya.rao@uni.edu) about Project Falcon."
#   better name detection:   uv pip install -e ".[gliner]"  then add  --gliner
#   real providers:          --provider ollama | openai | anthropic --model <id>
#   TEE / other endpoints:   --provider openai --base-url <OpenAI-compatible URL>
#   stable pseudonyms:       --vault me.vault

# 3. Measure what still leaks (add --dataset pupa for 901 real prompts)
python -m cryptollm.gateway.evaluate --configs regex,regex+gliner

# 4. Attack on "secretly permuted vocabulary" encryption
python -m cryptollm.attacks.vocab_permutation

# 5. Bootstrapping cost (uv pip install openfhe; needs ~8 GB RAM)
python -m cryptollm.fhe.bench_bootstrap

python -m pytest -q
```

## Measured here (4-core CPU, no GPU)

| | plaintext | encrypted / protected |
|---|---|---|
| tiny LM, per character (CKKS, 128-bit) | 11 µs | 1.4 s, output identical; ~1 MB up / 8.7 MB down |
| one CKKS bootstrap (N = 2¹⁶) | – | 11.9 s, 7 GB of keys in RAM |
| gateway on 40 realistic prompts: direct identifiers hidden | – | regex 43% → regex+GLiNER **98.5%** (names 21% → 100%) |
| vocabulary-permutation "encryption" | – | attacker recovers **100%** of the mapping |
