# Can we encrypt an LLM prompt so that only we can read the answer?

### Final report, 13 November 2026

Five research agents covered fully homomorphic encryption, multi-party computation and
obfuscation, trusted execution environments, prompt sanitization, and the practical options
(local models, provider terms, law). They reported only what they checked against sources
during this research. The measurements in §9 were run in this repository.

---

## 0. The answer in brief

**Yes, it can be done. It works today for open-weight models, and you pay for it in
speed, money or trust. There is no cryptographic way to do it with ChatGPT, Claude or
Gemini unless the provider builds it.**

| You want… | Best available today | What you trust | Cost |
|---|---|---|---|
| Nobody but you ever sees the prompt | **Run an open-weight model locally.** On a 32 GB laptop: Gemma 4 31B, Qwen3.6-27B, gpt-oss-20b. On an 80 GB GPU: gpt-oss-120b (§7) | your own machine | a quality gap: on the AA Intelligence Index v5, open models score 32–46 vs 58 for the best closed model |
| A server computes the answer and *mathematically* never sees it | **Fully homomorphic encryption (FHE).** Llama-3-8B runs end to end at **18 s per token on 8 GPUs**, about $0.08/token, so a 500-token answer takes ≈ 2.5 h (§2) | maths only (lattice hardness) | ≈ 10³–10⁴× plaintext on GPUs; research code only |
| Same, by splitting trust | **Multi-party computation (MPC).** Llama2-7B: 23 s per forward pass on 2 GPUs plus 255 GB of dealer keys, or 200 s/token on 3 CPU servers (§3) | the servers don't collude | 10²–10⁴×, gigabytes per token; no product |
| A big open model at normal speed, operator locked out | **Confidential computing (TEE)** with attestation **checked by your own client**: Tinfoil, Privatemode, Cohere Model Vault Encrypted (beta), OpenPCC (§4) | Intel/AMD/NVIDIA, and that nobody with physical access has tampered with the machine | ≈ 1–8% overhead |
| GPT/Claude/Gemini specifically | **Hide the identifiers** with a local gateway (§5, built here), plus a zero-retention account (§8) | the provider still sees the remaining text and context. On 901 real prompts our best local detector stack still missed 20% of annotated private units (§9.5) | small quality loss on most tasks (≈ −0.6/10 on writing, < 5 points on benchmarks); fails when the hidden value *is* the question |

**Two facts behind every row:**

1. **Whatever the model can read, its operator can read.** Encrypting the prompt with an
   ordinary cipher, or in a code the model "understands", gives no privacy. The operator
   runs the same model on the same input. So privacy against the operator needs one of
   four things:
   - computing on ciphertext (FHE);
   - splitting the computation (MPC);
   - hardware that locks the operator out (TEE);
   - not sending the secret at all (local model, or sanitization).
2. **Promises are not encryption.** In 2025 a US court ordered OpenAI to keep chats that
   users had deleted. Warrants for chatbot logs are now routine. The newest frontier
   models increasingly *require* 30-day retention (§8).

---

## 1. Why the obvious ideas fail

| Idea | Status | Evidence |
|---|---|---|
| "Encrypt" the prompt with a secret cipher that the model is told how to decode | **No privacy** | The cipher has to travel inside the prompt, so the operator decodes it too. CipherChat (ICLR 2024) shows GPT-4 can work in ciphers *when the prompt explains them* |
| Keyed token / vocabulary substitution with a matching embedding layer (SentinelLMs, AlienLM, STIP) | **Broken** | A ciphertext-only attack recovers 70–95% of tokens (PMA, EMNLP 2026). Under a secret rotation, row-norm matching recovers ~100% (ConjFormer 2026). **Our attack: 100% of a 3,000-token mapping**, including rotation and 15% noise (§9.4) |
| Permute the hidden states (PermLLM, STIP, Centaur) | **Broken** | 97.6–99.9% of prompts decoded perfectly on Llama-3.1-8B and Gemma-2 ("Hidden No More", ICML 2025) |
| Send embeddings or middle-layer activations instead of text | **Broken** | Vec2Text recovers 92% of 32-token inputs exactly. 86.88 token-F1 from a Llama-3 middle layer on a 4,112-token prompt (USENIX Security 2025) |
| Add decoy queries (k-anonymity) | **Weak** | The real sub-prompts still go out in plaintext |
| 2026 heuristic obfuscators (AloePri at ByteDance on a 671B model; GELO; ConjFormer) | **Unproven** | No security proofs. AloePri's own ablation shows >30% token recovery at a weaker noise setting |

---

## 2. Fully homomorphic encryption: the literal answer, now real but slow

The client encrypts and holds the only key. The server runs the model on ciphertexts using
plaintext weights. Only the client can decrypt the output.

**2026 state of the art (verified):**

| System | Model and workload | Speed and hardware |
|---|---|---|
| Park, Cheon, Stehlé et al., arXiv 2601.18511 (v2, Jun 2026) | **Llama-3-8B, full generation**, 128 encrypted tokens; perplexity 6.13, the same as FP16 | 20 s prefill + **18 s per token**, 8× RTX PRO 6000 (confirmed in the v2 abstract). With a 4k-token prompt (last 128 tokens encrypted): 64 s + 22 s/token. v1 (Jan 2026): Llama-2-7B, 33 s/token on 8× RTX 4090 |
| Fan et al. ("Themis"/"Odin"), arXiv 2609.12378 (Sep 2026) | Llama-3-8B, 128-token input, all 32 layers, **described as the first open-source end-to-end implementation** | 366 s server time on **1× H100**, 4.5× faster than THOR's method under the same conditions; per-token decode not reported |
| Cachemir, arXiv 2602.11470 | Llama-3-8B decode; perplexity unchanged | **1.61 min per token** on 1× A100-80GB. The encrypted KV cache is ≈ 17 GB per 1k tokens |
| Cerium (SOSP 2026) | Llama-3-8B prefill, 128 tokens | 43 s to first token on 8× B200; 241 s on 1× B200. Decode not reported |
| BERT-base, 128 tokens: THOR (CCS 2025) → MOAI (ICLR 2026) → Tricycle → Terrazzo (Sep 2026) | encoder classification | 10.4 min → 2.4 min → 100 s → **7 s per input** (batched, RTX 5090) |
| Encryption-friendly LLM (ICLR 2025) | encrypted **LoRA fine-tuning** and inference (2-layer model) | 6.9× faster fine-tuning than the baseline |
| Zama Concrete-ML GPT-2 | 1 attention head, TFHE | 166 s on a 196-core CPU (2023) |

**What it costs:** 18 s/token on 8 GPUs is ≈ 144 GPU-seconds, about **$0.08 per token** at
$2/GPU-hour. A 500-token answer takes about 2.5 h and $40. Plaintext costs fractions of a
cent.

**Why it is slow:**
- bootstrapping (62% of NEXUS's CPU time);
- ciphertext × ciphertext matmuls in attention;
- polynomial approximations of softmax, GELU and RMSNorm, made harder by outlier
  activations;
- weights encoded for FHE, 52–58 GiB for Llama-3-8B (982 GB raw in Cerium);
- a multi-GB encrypted KV cache;
- per-client evaluation keys of 7 GB (measured here) to 50 GB.

Batched "amortized" figures only hold when all inputs belong to one client key.

**What still leaks:**
- With a client in the loop (decrypt, sample, re-encrypt each token), the server sees the
  output length and timing.
- Non-interactive argmax forces greedy decoding and a fixed token budget, and that budget
  is visible.
- There is no integrity guarantee that the server ran the right model.

**Hardware and timeline:**
- Intel's Heracles research chip (ISSCC, Feb 2026) is **1,074–5,547× faster** than a Xeon
  on FHE primitives, but Intel has no product plans.
- No FHE ASIC can be bought in 2026. Niobium has funding and tape-outs; Optalysys sells an
  FPGA node for TFHE; Fabric, Cornami and Chain Reaction have shown no verified LLM
  results.
- GPUs are the practical accelerator: one bootstrap takes 7.5–34 ms on datacenter GPUs,
  against **11.9 s on our 4-core CPU**.
- Zama predicted "encrypted AI within 5 years" in 2023, i.e. about 2028.
- **Our estimate:** ~1 s/token on GPU clusters within 2–3 years. Affordable interactive
  chat needs ASICs, so roughly 2029–2032.

**Production HE today** is shallow lookups with no bootstrapping:
- Apple's Live Caller ID and Enhanced Visual Search;
- Microsoft Edge Password Monitor;
- Google's open-source HEIR compiler (Aug 2026).

**Bottom line:** FHE is the only option whose guarantee rests on maths alone, and an 8B
model now runs end to end. It costs about 10⁴× plaintext, and only the operator can deploy
it on their own model. It is a strong research direction, but not yet a practical tool for
chatting.

---

## 3. Multi-party computation: works up to 13B, no product

| System | Model and result | Traffic | Setting |
|---|---|---|---|
| SIGMA (PETS 2024) | Llama2-7B **23.1 s**, 13B **37.6 s** per 128-token forward pass | 12–19 GB online, plus **255–419 GB of dealer keys** | 2 A6000 GPUs + trusted dealer, 9.4 Gbps LAN; 5.7–8.3 min over WAN |
| PUMA | LLaMA-7B **200 s per token** | 1.8 GB/token | 3 servers (honest majority), 0.1 ms RTT |
| BumbleBee (NDSS 2025) | LLaMA-7B **8 min per token** | 6.3 GB/token | 2 parties (user + provider), 1 Gbps |
| Comet, MPCache, FuseFSS, SHAFT (2025–26) | 1.2–3× gains per paper | – | – |

- **No verified MPC result at 70B.** All LLM-scale results assume semi-honest servers.
- **Deployment needs independent parties who never collude** (another cloud, another
  jurisdiction). The user's machine becomes a full MPC party that moves gigabytes per
  token.
- **No company ships MPC or FHE evaluation of a whole LLM.** "Private LLM" products
  (Nillion nilAI, Duality) actually run in TEEs. Arcium Blackthorn (private alpha, Oct
  2026) uses MPC only to guard the keys of a GPU confidential-computing channel.
- **For experiments:** SecretFlow SPU (PUMA's LLaMA-7B example) and EzPC/GPU-MPC (SIGMA).
  CrypTen was archived in May 2025.

---

## 4. Confidential computing (TEEs): practical now, with hardware trust

**How it works.** The model runs in a confidential VM (Intel TDX or AMD SEV-SNP). The GPU
runs in NVIDIA confidential-computing mode (H100/H200, Blackwell B200/B300). Your client:

1. checks a hardware-signed **attestation** of the exact software (and, ideally, the model
   weights);
2. compares it with reproducible builds in a transparency log;
3. encrypts the prompt to a key that exists only inside the enclave, often through an
   **OHTTP** relay that sees your IP address but not the content.

Throughput overhead is **about 1–8%** when tuned. At batch size 1 it can double the cost
per token.

**Services where *your own client* verifies the attestation (Nov 2026):**

| Service | Models | Notes |
|---|---|---|
| **Tinfoil** | Kimi K3.1 Pro, GLM-5.3, DeepSeek V4.1 Flash, Gemma 4 31B, gpt-oss-120b, Llama 3.3 70B | SDKs in 5 languages, in-browser verifier, model weights measured; OpenAI-compatible; GA |
| **Privatemode** (Edgeless, EU-hosted) | gpt-oss-120b, Gemma 3, Qwen3, GLM-5.3-Flash | local verifying proxy; OpenAI- and Anthropic-compatible; GA. Pin the manifest |
| **Cohere Model Vault Encrypted** | Cohere's own models + open models | client-side OHTTP proxy fails closed; public beta, US only |
| **Confident Security CONFSEC / OpenPCC** | open models | the most Apple-PCC-like open design (Apache-2.0 client); early access |

Phala/RedPill, NEAR AI Cloud and Nillion provide attestation but **don't verify by
default**.
- NEAR AI suffered an SGLang remote-code-execution breach on **8 Nov 2026**.
- Venice's TEE mode delivers plaintext to Venice; only its e2ee mode encrypts on the
  client.
- Brave Leo's TEE path is verified by Brave's gateway, not by your browser.
- **Proton Lumo uses no TEE:** prompts are encrypted to Proton's own servers.

**Frontier closed models:** no API lets a developer reach a GPT-, Claude- or Gemini-class
model with attestation that the developer's own client verifies.
- OpenAI "Private Inference" (29 Sep 2026) is a sales-gated preview with undisclosed
  details.
- Anthropic published a design paper (2025) and licensed Opaque Systems' technology
  (Oct 2026), but has no product.
- Google Private AI Compute (Nov 2025) is for consumer Pixel features only.
- Apple PCC (now also on Google Cloud, 2026 preview) is verified by Apple's OS, not by you.
- Azure confidential inferencing never left preview.

**What breaks TEEs:**
- **Physical memory-bus interposers costing $50–$1,000** read memory and forge
  attestations:
  - WireTap (Sep 2025);
  - Battering RAM ($47.62);
  - TEE.fail (Oct 2025), which forged TDX quotes and "borrowed" NVIDIA GPU attestations;
  - DDRop (Sep 2026), against fully patched TDX.
  
  Intel and AMD call physical attacks out of scope, yet **the cloud operator is exactly the
  party with physical access.**
- **Software bugs inside the enclave**, such as the NEAR breach.
- **Metadata side channels:**
  - Whisper Leak (Nov 2025) classifies the topic from packet sizes and timing at >98%
    AUPRC across 28 LLMs.
  - Token-length analysis reconstructed 29% of streamed responses.
  - Prompt-cache timing leaked across users at 7 of 17 API providers.

**Checklist if you use a TEE service:**
- verify inside your own process;
- pin the measurements and fail closed if they change;
- require that the model weights are measured;
- use an OHTTP relay run by a different party;
- turn off streaming, or pad the output.

---

## 5. Prompt sanitization: the only route to GPT, Claude and Gemini

You hide identifiers locally, send the rest, and restore the identifiers in the answer.
This repo's gateway does exactly that (§9.5).

**Evidence from the literature:**
- **Utility:**
  - Typed placeholders like `[PERSON_1]` cost about −0.6 on a 10-point quality scale for
    writing tasks, and under 5 points on ARC, HellaSwag and IFEval.
  - They collapse multi-document retrieval: RGB fell from 0.80 to 0.32.
  - Stronger models lose more.
  - 85.7% of the PII spans in real ShareGPT/WildChat prompts could be removed without
    hurting the answer.
- **Better methods:**
  - **PAPILLON** (NAACL 2025): a local model rewrites the query. It leaks 7.5% of PII
    units while 85.5% of answers are as good as with the raw query.
  - **Prεεmpt** (NDSS 2026): format-preserving encryption plus metric differential
    privacy, with a cryptographic definition (it still counts NER misses as leakage).
  - For format-preserving encryption, NIST's SP 800-38G Rev. 1 draft keeps **only FF1**;
    FF3 and FF3-1 are broken.
- **The text around the redactions still identifies people:**
  - LLMs infer personal attributes from text at 85.5% top-1 accuracy (Staab et al.,
    ICLR 2024).
  - In 2026, web-search agents **re-identified 26–40 of 53 interview transcripts after
    Presidio redaction**. LLM deanonymization reaches 68% recall at 90% precision
    (USENIX Security 2026).
- **Detectors miss things on hard text:**
  - On REDACT's most sensitive tier, recall is 0.07 for Presidio, 0.42 for GLiNER, 0.57 for
    OpenAI's open Privacy Filter (Apr 2026), and 0.74–0.77 for GPT-4.1 or Claude
    Sonnet 4.6 used as the detector.
  - Uncommon or non-Western names are a weak spot everywhere.

**Our measurement (§9.5):**
- On 901 real WildChat prompts, regex detectors alone hid **4%** of annotated private
  units.
- Adding the GLiNER and Presidio NER models, with organisations and places hidden too,
  raised that to **80%**: 75% of prompts left with nothing exposed, at 8.6% over-redaction.
- Our own hand-written prompts suggested 98.5%. Development sets flatter detectors.

**When it works:** drafting, rewriting, translation, coding, classification, and reasoning
where identities don't matter. **When it fails:** when the private value *is* the question
("is a salary of X fair in city Y?"), when the task needs world knowledge about the hidden
entity, or for medical or legal questions where the sensitive fact is the content.

---

## 6. The decision guide

```
Is the data extremely sensitive (health, legal, identity documents, diaries)?
 ├─ yes → local open-weight model (§7). Done. Nothing leaves the machine.
 └─ no / need a bigger model
     ├─ Is an open-weight model good enough?
     │    └─ yes → TEE service whose SDK verifies attestation in YOUR process (§4)
     │            + gateway as defense in depth + no streaming / padding
     └─ Need a frontier closed model?
          └─ gateway (regex + GLiNER) + zero-retention / no-training account
             + strip context that isn't needed + assume the rest is visible
             + avoid "Covered Models" that force retention (§8)
Research (PhD-scale): FHE decode for 8B models, MPC, attacks on new obfuscators.
```

---

## 7. Local models: how good is "never send it"?

- **Quality.** On the Artificial Analysis Intelligence Index v5 (27 Oct 2026), the best
  closed model, Claude Opus 5.5, scores **58**. Kimi K3.1 Pro scores **46**. Its weights
  were promised for 4 Nov 2026, and Tinfoil now serves it, but Artificial Analysis still
  listed it as proprietary. Among models whose weights were downloadable when v5 launched,
  the best scored **32**.
- **Laptop models.** Models that fit a laptop are much stronger than a year ago, but trail
  on hard agentic tasks.
- **Memory needed:** about 0.58 bytes per parameter at 4-bit, plus the KV cache.

| Hardware | Models | Speed |
|---|---|---|
| 16 GB laptop | gpt-oss-20b (MoE, 3.6B active), Qwen3.5-9B, Gemma 4 12B | gpt-oss-20b: 23–42 tok/s on an M4 with 32 GB |
| 32 GB laptop | Gemma 4 31B / 26B-A4B, Qwen3.6-27B / 3.8-27B (~16–19 GB) | 27–31B dense: ~10–15 tok/s |
| 1× RTX 5090 (32 GB) | Qwen3-32B | **61 tok/s** |
| 80 GB GPU or 128 GB Mac | gpt-oss-120b, Qwen3.5-122B-A10B | – |
| Multi-GPU server | DeepSeek V4, GLM-5.3, Kimi K3.1 Pro (~1.6 TB at 4-bit) | – |

**Tools:** Ollama, llama.cpp, LM Studio, MLX, vLLM. Beware Ollama's `:cloud` models: they
run on Ollama's servers.

**Renting a GPU server** removes the model vendor from the picture but not the host. Only
a TEE changes that.

---

## 8. Contracts, retention and the law (policy, not cryptography)

| Service | Trains on your data? | Default retention | Zero data retention (ZDR) |
|---|---|---|---|
| OpenAI API | no (since Mar 2023) | up to 30 days (abuse logs) | approval via sales |
| Anthropic API | no | within 30 days | by agreement. **"Covered Models" require 30-day retention and are excluded from ZDR** unless expressly authorized; flagged content is kept up to 2 years and classifier scores up to 7 |
| Claude consumer apps | user choice since Oct 2025 | 5 years if training is on; 30 days if off | incognito chats are never used for training |
| Gemini API: free tier / paid tier | free: **yes**, with human review / paid: no | paid: 55 days | – |
| Azure OpenAI | no; content not available to OpenAI | flagged samples reviewed by Microsoft | modified abuse monitoring |
| AWS Bedrock | no; model vendors never see prompts | new `data_retention` setting | `none` for most models; Claude Fable 5.x requires `aws_review` |
| Vertex AI | no | caching ≤ 24 h; some logs 3–30 days | via configuration and account team |

**Legal compulsion (NYT v. OpenAI):**
- **13 May 2025:** the court ordered OpenAI to preserve *all* output logs, including chats
  users had deleted. ZDR and Enterprise traffic were exempt.
- **26 Sep 2025:** the duty ended, but the preserved logs are still held.
- **20M de-identified logs** were ordered produced; the order was affirmed on
  5 Jan 2026.

**Other cases:**
- Federal warrants for ChatGPT and Grok prompts (2025).
- US v. Kim (Jun 2026): the court refused to quash a warrant for ChatGPT records.
- An SDNY ruling held that a defendant's Claude chats were not privileged.

**Status of the vendors' own encryption:**
- OpenAI announced client-side encryption on its roadmap (Nov 2025); not shipped.
- ChatGPT's "Locked Chats" (Oct 2026) are a PIN lock, not end-to-end encryption.

**The anonymity layer** hides *who* you are but not what you wrote:
- OHTTP (RFC 9458) and Privacy Pass;
- Duck.ai, which strips IPs and gets providers to delete within 30 days;
- OpenRouter, which can route only to ZDR endpoints but sees your plaintext;
- Tor/VPN and anonymous payment.

The content itself can still identify you.

---

## 9. What we built and measured (4-core cloud CPU, no GPU)

### 9.1 Encrypted language-model generation (`cryptollm/fhe/encrypted_lm.py`)

The model is a tiny character-level neural LM (64 → 96 → 33, quadratic activation). It runs
under **CKKS at 128-bit security** (TenSEAL/SEAL, N = 16384). The server is a **separate
OS process** that holds only public evaluation keys and refuses a secret key. The client
does the embedding lookup, decryption and sampling.

| | plaintext | encrypted |
|---|---|---|
| output for prompt `"the server "` | `never sees the math on the rig` | **identical** |
| max logit error | – | 7 × 10⁻⁴ |
| latency per character | 11 µs | **1.4 s** (≈ 10⁵×) |
| traffic per character | 8 B | 1.05 MB up / 8.7 MB down |
| one-time evaluation keys | – | 180 MB |

We hit a real-world pitfall worth recording: TenSEAL's `matmul` is only correct on a
*freshly encrypted* (slot-replicated) vector. A second layer applied after any
multiplication silently returned wrong logits (error ≈ 11). We use per-output dot products
for that layer instead.

### 9.2 One encrypted attention decode step, softmax included (`cryptollm/fhe/encrypted_attention.py`)

This is the operation an LLM repeats for every generated token in every head of every
layer: the new token's query attends over an **encrypted KV cache**. It uses OpenFHE CKKS
at 128-bit security (N = 2¹⁶, 1,024 slots).

**Server side:**
- Computes the Q/K/V projections with baby-step/giant-step diagonal matvecs.
- Appends the new key and value to the encrypted cache.
- Computes the scores ⟨q, kⱼ⟩/√d with packed ciphertext × ciphertext products and
  rotate-and-sum.
- Evaluates **softmax homomorphically**: Chebyshev polynomials of exp(x) on [−1.5, 1.5]
  and 1/x on the possible range of the sum.
- Weights the values and applies the output projection.

Nothing is decrypted on the server.

| d_model 32, head dim 16, 8-token context | plaintext | encrypted |
|---|---|---|
| time per decode step (1 head) | 14 µs | **8.5 s** (≈ 6 × 10⁵×) |
| max error vs exact softmax attention | – | **2 × 10⁻¹⁰** |
| multiplicative levels used / budget | – | 21 / 24 (no bootstrapping inside the step) |
| peak RAM (keys + data) | – | 2.9 GB |

**Lessons:**
- FHE needs a **known input range** for every non-polynomial function. We rescale Wq so
  scores stay within [−1.5, 1.5]. Real systems handle LLM outlier activations with extra
  machinery.
- **Each Chebyshev approximation costs 5–7 multiplicative levels.** Our first version ran
  out of levels: decryption failed with "approximation error is too high".
- **A multi-layer model must bootstrap between layers** (§9.3). That is why the 8B results
  in §2 run on 8 GPUs.

### 9.3 Cost of depth: CKKS bootstrapping (`cryptollm/fhe/bench_bootstrap.py`)

Setup: OpenFHE 1.6.1, 128-bit security, N = 2¹⁶, 1,024 slots, 4 threads.

| Operation | Result |
|---|---|
| Key generation | 7.6 s, **7.1 GB RAM** |
| One bootstrap | **11.9 s** |
| ct × ct multiply | 148 ms |
| ct × pt multiply | 8.9 ms |
| Rotation | 155 ms |

GPUs bootstrap 350–800× faster, which is why every 8B result in §2 runs on 8 GPUs.

### 9.4 Attack on "encrypt the tokens so the model can still read them" (`cryptollm/attacks/vocab_permutation.py`)

**Setup:** a 3,000-token public embedding table. The defender applies a secret permutation,
optionally a secret orthogonal rotation of the hidden space, and fine-tuning noise.

**Attack:** rotation-invariant row signatures (norm plus sorted distances to all other
rows), then orthogonal Procrustes on the confident matches, then re-matching.

**Result:** the attacker recovers **100%** of the token mapping in every setting:
permutation only, plus rotation, plus 5% noise, plus 15% noise. Runtime is ~2 s. This agrees
with ConjFormer's row-norm attack, PMA (EMNLP 2026) and "Hidden No More" (ICML 2025) on
real LLM tables.

### 9.5 Privacy gateway and its measured leakage (`cryptollm/gateway`)

**Pipeline:** detect → typed placeholders → leak check → send → restore locally (also for
streamed output).

**Detectors:**
- checksum-validated IDs (Luhn, IBAN mod-97, SSN rules, IPs), emails, URLs and phones;
- API keys: OpenAI, Anthropic, Stripe, AWS, GitHub, Slack, Google, Hugging Face, JWTs,
  PEM private keys, plus a high-entropy token check;
- context-triggered IDs: passport, PAN, MRN, account, booking reference, …;
- account names inside home-directory paths;
- ws/wss/ftp/ssh/git URLs as well as http(s);
- titled names;
- your own term list;
- optional Presidio (spaCy `en_core_web_lg`) and optional **GLiNER** (`urchade/gliner_multi_pii-v1`).

**Name handling:**
- A name, once seen, is hidden in every later turn, including its first and last names on
  their own.
- Names that are also common words are never swept on their own, so "Will" and "Grace"
  don't blank out ordinary words.
- A name taken from an email address is hidden where it appears on its own in the same
  text.

**Safety and storage:** a fail-closed leak check runs before anything is sent. Mappings are
kept in an AES-256-GCM + scrypt vault.

**Leakage, realistic set.** 40 hand-written chat prompts with 67 direct identifiers,
diverse and lowercase names, IDs from several countries, and 4 prompts containing no
personal data. A value counts as hidden only if *every* letter and digit of *every*
occurrence is masked.

| | regex | + Presidio | + GLiNER | + GLiNER + Presidio |
|---|---|---|---|---|
| direct identifiers fully hidden | 43% | 84% | **98.5%** | 98.5% |
| names hidden | 21% | 89% | **100%** | 100% |
| prompts with nothing exposed | 26% | 74% | **97%** | 97% |
| over-redaction (clean text masked) | 0.9% | 3.2% | 3.0% | 5.2% |
| CPU time per prompt | < 1 ms | 7 ms | 131 ms | 141 ms |

**Caveat:** we wrote this set and added the context-ID and secret detectors after seeing
its first results. Before that, IDs were 56% and secrets 50% hidden. Treat it as a
development set.

**Leakage, PUPA (held-out, real prompts).** 901 *real* WildChat user prompts with annotated
privacy units, from the PAPILLON paper (NAACL 2025). None of the gateway was tuned on it.

| 901 real prompts, 2,220 annotated units | regex | + Presidio | + GLiNER | + GLiNER + Presidio | + GLiNER (with orgs/places) + Presidio |
|---|---|---|---|---|---|
| privacy units fully hidden | 4.1% | 42.1% | 38.9% | 54.1% | **80.0%** |
| prompts with nothing exposed | 4.2% | 33.1% | 30.6% | 46.1% | **75.2%** |
| over-redaction (clean text masked) | 0.7% | 3.2% | 3.6% | 5.6% | 8.6% |
| CPU time per prompt | 1 ms | 27 ms | 468 ms | 555 ms | 498 ms |

PUPA counts organisations, places and products as private units. That is why hiding them
(the last column) matters so much.

**What is still exposed in the best setting.** In a random sample of 40 of the 443
exposed units:
- **About one third are real personal identifiers:**
  - lowercase lone first names and surnames ("tim", "aida", "d'amico");
  - usernames;
  - bare ID numbers with no cue word;
  - an account name inside a file path;
  - a `wss://` URL.
- **About a quarter are company and brand names.**
- **About a quarter are annotation noise or not personal:** pronouns such as "her" and
  "his", WildChat's own `presidio_anonymized_…` placeholders, book titles.
- **The rest are code identifiers.**

We then added the home-path and URL-scheme detectors. The PUPA numbers above were measured
*before* those additions, so they remain an unbiased held-out result.

**Reading the numbers:**
- **Our own set overstated the protection.** It reports 98.5% of direct identifiers
  hidden (regex + GLiNER), but on real prompts that setup hid **39%** of annotated units.
  The best stack hid 80% (75% of prompts fully clean) at 8.6% over-redaction. **Evaluate
  on held-out real data.**
- **Regex alone is fine for structured data and secrets, but useless for everything
  else** (4% on PUPA).
- **NER models help a lot and run on CPU in about 0.1–0.5 s per prompt.** On a laptop it
  is worth running GLiNER and Presidio together, with organisations and places hidden too,
  unless the task needs them.
- **Detection is not anonymity.** §5 shows the remaining context can still identify
  people.

---

## 10. Open research directions

1. **FHE decode for open 8B models.** Today it costs 18 s/token on 8 GPUs. Promising
   levers are encryption-friendly fine-tuning (softmax-free attention, low-degree
   activations, LoRA under FHE), KV-cache packing, and GPU libraries (Cheddar, FIDESlib,
   desilofhe).
2. **Attacking 2026 obfuscators.** AloePri (deployed at 671B), GELO and ConjFormer have no
   proofs. Our signature-plus-Procrustes attack is a natural starting point.
3. **Measuring gateway leakage properly.** Combine detector recall (SPY, REDACT, PUPA)
   with *LLM re-identification attacks* on the sanitized prompts, and report the
   privacy/utility frontier.
4. **A hybrid:** local gateway → TEE-hosted open model → frontier model only for de-identified
   sub-questions (in the spirit of PAPILLON).
5. **Side-channel-resistant serving:** padding, batching and constant-rate streaming against
   Whisper Leak-style attacks on TEE and FHE services.

---

## References (selected; all checked by the research agents unless marked *vendor*)

**FHE:**
- Park, Cheon, Stehlé et al. https://arxiv.org/abs/2601.18511
- Fan et al. https://arxiv.org/abs/2609.12378
- Cachemir https://arxiv.org/abs/2602.11470
- Cerium https://arxiv.org/abs/2512.11269
- NEXUS https://www.ndss-symposium.org/wp-content/uploads/2025-868-paper.pdf
- THOR https://eprint.iacr.org/2024/1881
- MOAI https://eprint.iacr.org/2025/991
- Terrazzo https://eprint.iacr.org/2026/1870
- Encryption-friendly LLM https://arxiv.org/abs/2410.02486
- Zama GPT-2 https://huggingface.co/blog/encrypted-llm
- Intel Heracles https://spectrum.ieee.org/fhe-intel
- Apple HE https://machinelearning.apple.com/research/homomorphic-encryption
- Google HEIR https://blog.google/security/how-google-is-making-private-ai-practical-with-homomorphic-encryption/

**MPC / obfuscation:**
- SIGMA https://petsymposium.org/popets/2024/popets-2024-0107.pdf
- PUMA https://arxiv.org/pdf/2307.12533
- BumbleBee https://ndss-symposium.org/wp-content/uploads/2025-57-paper.pdf
- Hidden No More https://arxiv.org/abs/2505.18332
- PMA https://arxiv.org/abs/2609.06749
- ConjFormer https://arxiv.org/abs/2606.16461
- AloePri https://arxiv.org/abs/2603.01499
- Vec2Text https://arxiv.org/abs/2310.06816
- Survey https://eprint.iacr.org/2026/105

**TEE:**
- Apple PCC https://security.apple.com/blog/private-cloud-compute/
- PCC on Google Cloud https://security.apple.com/blog/expanding-pcc
- Google Private AI Compute https://www.infoq.com/news/2025/11/google-private-ai-compute-tee
- WhatsApp Private Processing https://engineering.fb.com/2025/04/29/security/whatsapp-private-processing/
- Anthropic confidential inference https://www.anthropic.com/research/confidential-inference-trusted-vms
- OpenAI Private Intelligence https://venturebeat.com/security/openais-new-private-intelligence-targets-a-growing-enterprise-concern-who-can-see-your-ai-data
- Tinfoil https://tinfoil.sh *(vendor)*
- Privatemode https://docs.privatemode.ai *(vendor)*
- Cohere https://docs.cohere.com/docs/model-vault/encrypted/verifying-deployment *(vendor)*
- OpenPCC https://github.com/openpcc/openpcc
- TEE.fail https://tee.fail
- Battering RAM https://batteringram.eu
- WireTap https://wiretap.fail
- DDRop https://thehackernews.com/2026/09/new-ddrop-attack-breaks-intel-tdx-and.html
- NEAR breach https://thehackernews.com/2026/11/sglang-rce-exploit-breaches-near-ai.html
- Whisper Leak https://arxiv.org/abs/2511.03675
- Token-length attack https://arxiv.org/abs/2403.09751
- Overheads https://arxiv.org/abs/2509.18886
- Provider probe https://amiller.github.io/awesome-private-inference

**Sanitization:**
- Prεεmpt https://arxiv.org/abs/2504.05147
- PAPILLON / PUPA https://arxiv.org/abs/2410.17127
- Staab et al. https://arxiv.org/abs/2310.07298
- Re-identification after redaction https://arxiv.org/abs/2605.30848
- LLM deanonymization https://arxiv.org/abs/2602.16800
- SPY https://arxiv.org/abs/2605.09973
- REDACT https://arxiv.org/abs/2606.19881
- Utility of pseudonymization https://aclanthology.org/2025.privatenlp-main.3.pdf and https://arxiv.org/abs/2609.11335
- PII minimization https://arxiv.org/abs/2510.03662
- NIST SP 800-38G r1 https://csrc.nist.gov/pubs/sp/800/38/g/r1/2pd
- CipherChat https://arxiv.org/abs/2308.06463

**Practical / policy:**
- AA Intelligence Index v5 https://artificialanalysis.ai/articles/artificial-analysis-intelligence-index-v5
- OpenAI data https://developers.openai.com/api/docs/guides/your-data
- Anthropic retention https://platform.claude.com/docs/en/manage-claude/api-and-data-retention
- Gemini terms https://aistudio.google.com/docs/terms
- Bedrock retention https://docs.aws.amazon.com/bedrock/latest/userguide/data-retention.html
- NYT order https://blogs.duanemorris.com/classactiondefense/2025/10/27/new-york-federal-courts-openai-discovery-orders-provide-key-insights-for-companies-navigating-ai-preservation-standards/
- 20M logs https://natlawreview.com/article/when-chats-become-evidence-court-affirms-order-requiring-openai-produce-20-million
- OHTTP https://datatracker.ietf.org/doc/rfc9458
- Duck.ai https://duckduckgo.com/duckai/privacy-terms
- OpenRouter https://openrouter.ai/docs/guides/privacy/logging
