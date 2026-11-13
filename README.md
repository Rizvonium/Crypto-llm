# Crypto-LLM: keeping prompts private from the companies that run the model

**Question:** can we encrypt a prompt, have an LLM process it, and get back an answer that
only we can read?

**Answer (details in [`docs/REPORT.md`](docs/REPORT.md)):**

- **Mathematically, yes.** With fully homomorphic encryption (FHE), a server runs the model
  on ciphertexts and never holds the key. This repo does it for a small language model, end
  to end, at 128-bit security.
- **For GPT/Claude/Gemini-scale models, not yet practical.** It costs hours per token on a
  CPU, and only the provider could run it on their own model anyway.
- **What works today:**
  - local open-weight models (nothing leaves your machine);
  - confidential-computing (TEE) services with attestation you verify yourself;
  - a pseudonymization gateway that hides names, emails and IDs from any provider and
    restores them locally.
- **Tricks that don't work:** "cipher prompts", sending embeddings, and secretly permuted
  vocabularies. The operator can undo all of them. This repo includes a working attack on
  the permuted-vocabulary idea.

## What's inside

| Path | What it does |
|---|---|
| `cryptollm/fhe/encrypted_lm.py` | Encrypted text generation (CKKS via TenSEAL). The server is a separate process holding only public keys. |
| `cryptollm/fhe/bench_bootstrap.py` | OpenFHE 128-bit CKKS bootstrapping benchmark. It shows why deep models are slow. |
| `cryptollm/gateway/` | Privacy gateway: detect → pseudonymize → send → restore. Includes an encrypted vault and providers (dry-run, OpenAI-compatible, Anthropic). |
| `cryptollm/attacks/vocab_permutation.py` | Breaks "encrypt tokens with a secret permutation + rotation". |
| `docs/REPORT.md` | The full write-up: options, trust assumptions, measurements, recommendations. |

## Quick start

```bash
uv venv .venv --python 3.12 && source .venv/bin/activate   # openfhe wheels need CPython 3.12
uv pip install numpy cryptography pytest tenseal            # + openfhe for the bootstrap benchmark

# 1. Encrypted generation: the server only ever sees ciphertext
python -m cryptollm.fhe.encrypted_lm "the server " -n 30 --separate-process

# 2. Privacy gateway (dry run: nothing is sent; shows exactly what would be)
python -m cryptollm.gateway --terms "Project Falcon" \
  "Reply to Dr. Priya Rao (priya.rao@uni.edu) about Project Falcon."
#    real providers: --provider ollama | openai | anthropic --model <id>
#    TEE or other OpenAI-compatible services: --provider openai --base-url <url>
#    consistent pseudonyms across sessions: --vault me.vault

# 3. Attack on "secretly permuted vocabulary" encryption
python -m cryptollm.attacks.vocab_permutation

# 4. Bootstrapping cost (needs ~8 GB RAM)
python -m cryptollm.fhe.bench_bootstrap

python -m pytest -q
```

## Measured on a 4-core CPU (no GPU)

| | plaintext | encrypted (CKKS, 128-bit) |
|---|---|---|
| tiny LM, per character | 11 µs | 1.4 s, output identical |
| traffic per character | 8 B | ~1 MB up / 8.7 MB down |
| one CKKS bootstrap (ring 2¹⁶) | – | 11.9 s, 7 GB of keys in RAM |
