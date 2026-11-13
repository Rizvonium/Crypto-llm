# Can we send an encrypted prompt to an LLM and get back an answer only we can read?

**Short answer: yes, it is mathematically possible, and you can run it on a tiny model
today. Nobody can do it yet for a GPT/Claude/Gemini-class model at usable speed.** Five
other things are practical now. They protect different things and rest on different
trust assumptions, and you can combine them.

> **About the sources.** The live web-research agents for this report hit a usage limit
> before they returned, so §§1–2 and §§4–6 come from the author's knowledge up to mid-2026,
> not from a fresh literature check. Product status and paper numbers marked ⚠ change fast
> or are quoted from memory as orders of magnitude; re-check them before citing. Everything
> in §3 was **measured in this repository**.

---

## 1. Why you can't just encrypt the prompt and send it to ChatGPT

An LLM is a function `f` that the provider evaluates on *their* machine. If you send
`Enc(prompt)` to an ordinary model you get `f(Enc(prompt))`, which is garbage. The other
tempting trick is a cipher the model can still understand: a substitution code, "reply in
base64", or permuted tokens. That fails by construction. **Whatever the model can decode,
the operator running that same model can decode.** They have the model, the prompt and
every activation.

So privacy against the party running the model has only five possible sources:

| # | Approach | Who you trust | Works with frontier closed models? | Speed today |
|---|---|---|---|---|
| A | **Don't send it**: run an open-weight model locally | nobody | no (open-weight only) | native |
| B | **Hide the sensitive parts**: pseudonymize, restore locally | the provider sees the rest of the text | **yes** | native |
| C | **Hardware enclave (TEE / confidential computing)** | chip vendors (NVIDIA/AMD/Intel) and the attestation chain | only the vendor's own products, e.g. Apple PCC and Google Private AI Compute ⚠; no general frontier API | ≈ native |
| D | **Split trust (MPC)** | at least one of 2–3 servers stays honest and they don't collude | no | minutes/token ⚠ |
| E | **Compute on ciphertexts (FHE)** | only mathematics (lattice hardness) | no (the provider would have to run it) | hours/token on CPU for an 8B model (§3) |

Two things this table doesn't cover:

- **Anonymity** (who you are): Oblivious HTTP (RFC 9458), Privacy Pass, Duck.ai-style
  proxies. These hide your identity from the provider but not your text, and text often
  identifies its author.
- **Contracts** (zero-data-retention terms, "we don't train on API data"): useful, but
  revocable. The 2025 *NYT v. OpenAI* preservation order forced retention of chats that
  users had deleted ⚠. Policy is not cryptography.

---

## 2. The options in detail

### A. Local / self-hosted open-weight models: the strongest practical answer
If the prompt never leaves your machine, there is nothing to protect. Open-weight models
(e.g. the gpt-oss, Qwen3, Llama, Gemma, DeepSeek and Mistral families ⚠) run via Ollama,
llama.cpp, LM Studio, vLLM or MLX. At 4-bit quantization a ~20–30B model fits in 16–32 GB
of RAM; a ~120B MoE model needs ~64–80 GB. They are still behind the best closed models on
the hardest tasks, but good enough for drafting, summarizing, coding help and analysis of
personal documents. On a rented GPU server the *LLM company* sees nothing, but the *cloud
host* could. Combine with C (a confidential VM) to address that.

### B. Prompt sanitization / pseudonymization: the only option that works with GPT/Claude/Gemini today
Replace names, emails, IDs, account numbers and keys with typed placeholders, send the
sanitized prompt, and put the real values back into the answer on your machine. **This repo
ships a working gateway** (`cryptollm/gateway`):

```
you type:       Draft a reply to Dr. Priya Rao (priya.rao@uni.edu, +91 98765 43210)
                about Project Falcon. Priya asked whether card 4111 1111 1111 1111 was charged.
provider sees:  Draft a reply to Dr. [PERSON_1] ([EMAIL_1], [PHONE_1]) about [TERM_1].
                [PERSON_1] asked whether card [CREDIT_CARD_1] was charged.
you read:       the answer, with the real values restored locally
```

Related research (⚠ from memory): **Prεεmpt** (format-preserving encryption plus metric
differential privacy for numbers), **PAPILLON** (NAACL 2025: a local model rewrites the
query for a remote model), **Hide-and-Seek**, **InferDPT**, **Split-N-Denoise**,
**ConfusionPrompt**. Tools: Microsoft Presidio, LLM Guard, GLiNER-based PII detectors.

**Limits. Be honest with yourself about these:**
- Context re-identifies. "My advisor at a small lab in city X who works on Y" is
  identifying with no names at all. LLMs infer age, location and occupation from writing
  style (Staab et al., ICLR 2024 ⚠).
- Detectors miss things, especially names without titles. Add your own term list or NER,
  and note that the gateway fails closed only on values it already knows.
- It doesn't fit tasks where the private value *is* the question ("is a salary of X fair
  in city Y?").

### C. Confidential computing (TEEs): practical today, if you accept hardware trust
The model runs inside a CPU enclave (AMD SEV-SNP / Intel TDX) with the GPU in
confidential-computing mode (NVIDIA H100/H200/Blackwell). Your client verifies a
**remote attestation**, a hardware-signed measurement of the exact software stack, and
encrypts the prompt to a key that exists only inside that enclave. Overhead is small
(single-digit percent for large models ⚠).

- **Big-provider systems (consumer features, not general APIs):** Apple Private Cloud
  Compute (2024), Meta/WhatsApp Private Processing (2025), Google Private AI Compute
  (Nov 2025), Azure confidential inferencing previews ⚠.
- **Services a developer can use with open-weight models**, mostly via OpenAI-compatible
  APIs: Tinfoil, Privatemode (Edgeless Systems), Phala/RedPill, NEAR AI Cloud, Confident
  Security, Nillion nilAI, among others ⚠. Prefer those that publish open-source clients
  that **verify attestation on your side**. Otherwise you are back to trusting a promise.
  The gateway's `--base-url` points at any such endpoint.
- **Known weaknesses:**
  - The 2025 memory-bus interposer attacks (Battering RAM, WireTap, TEE.fail ⚠) broke
    SGX/TDX/SEV-SNP confidentiality and attestation for an attacker with physical access,
    which a cloud operator has.
  - Response-length and timing side channels on streamed LLM output reveal topics
    (Weiss et al. 2024; Microsoft "Whisper Leak" 2025 ⚠). TEEs don't hide those.

### D. Secure multi-party computation (MPC)
The prompt is secret-shared between 2–3 servers. No single server learns anything unless
they collude. Transformer protocols (Iron, MPCFormer, PUMA, CipherGPT, BOLT, BumbleBee,
SIGMA, Nimbus ⚠) bring 7B-class models to roughly **tens of seconds to minutes per token**,
with **gigabytes of traffic per token** on a fast LAN. Softmax, GELU and LayerNorm dominate
the cost. The deployment question, *who are the non-colluding parties?*, has no
commercial answer for frontier models yet.

### E. Fully homomorphic encryption (FHE): the literal answer to the question
The client encrypts; the server evaluates the model on ciphertexts using plaintext weights,
never holding the secret key; only the client can decrypt the output. CKKS (approximate
arithmetic) suits neural networks. TFHE suits look-up-table non-linearities. Published
systems (THE-X, NEXUS at NDSS 2025, Powerformer, "encryption-friendly LLM architectures" at
ICLR 2025, Zama's encrypted GPT-2 attention head ⚠) handle BERT/GPT-2-scale models in
**minutes per inference** on GPUs. Llama-class generation is not practical yet. The
bottlenecks:
1. bootstrapping (measured below: 12 s per refresh on our CPU);
2. ciphertext × ciphertext products in attention (QKᵀ, softmax·V);
3. polynomial approximations of softmax, GELU and LayerNorm;
4. a KV cache that grows every token;
5. 10³–10⁵× ciphertext expansion.

Dedicated hardware (DARPA DPRIVE accelerators such as Intel's, plus start-up FHE chips ⚠)
targets 100–10,000× speedups. That is the path to practical encrypted LLMs, probably years
out. Homomorphic encryption already runs in production for simpler jobs, such as Apple's
Live Caller ID Lookup and Enhanced Visual Search.

### Things that look like encryption but are not
| Idea | Why it fails |
|---|---|
| "Reply in a cipher / base64 / my secret code" | The operator runs the same model and can decode whatever it decodes. |
| Send embeddings or hidden states instead of text | Inversion attacks recover the text (Vec2Text recovers ~92% of 32-token inputs exactly ⚠). |
| Secretly permute the vocabulary (+ rotate the hidden space) | The operator matches against the public model. **Demonstrated here: 97–100% of the mapping recovered** (`cryptollm/attacks`). |
| "Private mode" / "don't train on my data" toggles | Policy, not cryptography. Retention can be compelled by courts. |

---

## 3. What we built and measured (this repo, 4-core cloud CPU, no GPU)

### 3.1 Encrypted language-model generation, end to end (`cryptollm/fhe`)
A tiny character-level neural LM (64 → 96 → 33, quadratic activation so the encrypted
pass needs no approximation) runs under CKKS at **128-bit security** (TenSEAL/SEAL, ring
dimension 16384). The server runs in a **separate OS process**, receives only the public
evaluation keys, and refuses a context that contains a secret key. The client holds the
key, does the embedding lookup, decrypts the logits and picks the next character.

| | plaintext | encrypted |
|---|---|---|
| output for prompt `"the server "` | `never sees the math on the rig` | **identical** |
| max logit error | – | 7 × 10⁻⁴ |
| latency per character | 11 µs | 1.4 s (≈10⁵× slower) |
| upload / download per character | 8 bytes | 1.05 MB / 8.7 MB |
| one-time evaluation keys sent to the server | – | 180 MB |

What the server still learns: the number of generated characters (one request each),
timing, and sizes. Hiding those requires padding to a fixed length and fixed timing.

### 3.2 The cost of going deeper: bootstrapping (OpenFHE, `bench_bootstrap.py`)
CKKS, 128-bit security, ring dimension 2¹⁶, 1024 slots, 4 threads:
keygen 7.6 s and **7.1 GB of RAM**; **one bootstrap 11.9 s**; ciphertext × ciphertext
multiply 148 ms; ciphertext × plaintext multiply 8.9 ms; rotation 155 ms.

**Back-of-envelope for an 8B-parameter model** (≈7 × 10⁹ multiply-accumulates per token).
Even if every one of the 32,768 slots of a full ciphertext did useful work for every
8.9 ms ct × pt multiply, the weight multiplications alone take ~2 × 10⁵ multiplies ≈
30 minutes per token. That is before the comparable number of rotations at ~155 ms each,
the bootstraps, and the attention and softmax approximations. So it is **hours per token on
this CPU**. GPUs bring 10–100×; FHE ASICs promise more. That matches the literature: real
today for small models, not yet for chat-scale LLMs.

### 3.3 Privacy gateway (`cryptollm/gateway`)
- **Detectors:** checksum-validated structured IDs (Luhn, IBAN mod-97, SSN rules, IPs),
  emails, URLs, phones, API keys/tokens, titled names, your own term list, and optional
  Presidio NER.
- **Placeholders:** each value always gets the same placeholder. A name seen once is
  hidden in later turns too, and so are its first and last names on their own.
- **Fail-closed leak check** before anything is sent.
- **Restoring the answer** works on streamed output, including placeholders split across
  chunks.
- **Vault:** placeholder mappings persist in a vault encrypted with AES-256-GCM under a
  scrypt-derived key.
- **Providers:** dry-run, any OpenAI-compatible endpoint (local or TEE services), and
  Anthropic.

### 3.4 Attack on "keyed vocabulary permutation" (`cryptollm/attacks`)
The attacker uses rotation-invariant row signatures, then orthogonal Procrustes alignment.
Against permutation alone, permutation plus a secret rotation, and up to 15% added noise,
it recovers **97–100%** of the token mapping in seconds.

---

## 4. Recommendations

| Your situation | Do this |
|---|---|
| Very sensitive personal data (health, legal, finances, diaries) | **A**: a local open-weight model. Nothing leaves the machine. |
| Need a strong model but can't run it locally | **C**: a TEE service that lets *you* verify attestation, plus **B** as defense in depth. |
| Need GPT/Claude/Gemini specifically | **B**: the pseudonymization gateway, plus a zero-retention / no-training account. Assume the remaining text is visible. |
| Hiding *who* is asking | add an OHTTP/anonymizing relay, and keep identifying details out of the text. |
| Research direction | FHE/MPC-friendly architectures, gateway leakage evaluation, B+C hybrids, side-channel padding. |

## 5. Open research directions
1. **Measure the gateway's leakage.** Use a re-identification benchmark: can a strong LLM
   recover the hidden values or the user's attributes from the sanitized prompts?
2. **Local–remote decomposition.** A local model splits the task so the remote model only
   ever sees generic sub-questions (in the spirit of PAPILLON). Measure the utility/privacy
   frontier.
3. **FHE-friendly small LMs.** Polynomial activations, softmax-free attention, and
   distillation into depth-budgeted models. §3.1 shows the inference path works; what's
   missing is scale.
4. **Hiding output length and timing** for TEE and FHE services: padding, batching and
   constant-rate streaming.
