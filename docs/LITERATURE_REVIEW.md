# Fast encrypted LLM inference: a literature review and a publishable gap

*13 November 2026.*

**Scope.** Fully homomorphic encryption (FHE, mainly CKKS) for transformer and LLM inference,
plus the protocol choices around it.

**Method.**
- Four review tracks, each read against primary sources:
  1. non-linear functions (softmax, normalization, activations) without retraining;
  2. model co-design;
  3. linear algebra, bootstrapping, compilers and hardware;
  4. selective encryption and hybrid protocols.
- A fifth, targeted novelty check on the two candidate contributions (§12).
- Original experiments in this repository (§11).

**Evidence tags.**
- Claims are tagged as the reviewers reported them: **[V]** means checked in the primary source,
  **[U]** means secondary or unconfirmed.
- Untagged numbers come from **[V]** sources.

**Limitations.**
- Some IACR ePrint PDFs were behind a bot challenge. For Tricycle, ELLMo, ARION, the THEMIS ePrint
  and Perseus only the abstracts were read.
- Some 2026 preprints are not yet peer reviewed.

---

## 0. Executive summary

1. **Bootstrapping is where the time goes.** Wherever it is reported, bootstrapping takes
   59–80% of FHE transformer runtime:
   - Cachemir, one Llama-3-8B decode block: 75% on CPU and 59% on A100.
   - NEXUS, BERT-base: 62%.
   - Cerium runs 5,890 bootstraps for one 128-token Llama-3-8B prefill, about 184 per layer.

   So the lever is **multiplicative depth**, not FLOPs.
2. **Non-linear depth drives bootstrap count, and softmax is the worst offender.** Softmax's
   share of wall-clock time is modest (4–15%). But how deep its circuit must be depends on the
   **range of attention scores**. LLM scores span tens to hundreds of nats, and every row's
   maximum is different.
   - Methods that shift by a constant offset and divide break at LLM ranges. Cachemir measured
     NEXUS's softmax at 5.7 bits of precision on Llama-3-8B, and MOAI's diverged.
   - The state of the art avoids the maximum entirely with normalize-and-square (Cho et al.,
     CCS'24). Its depth grows with log₂ of a range bound M, and that bound is set by one
     calibrated constant.
3. **Big matrix multiplication is close to solved.** It is now within 4–12× of FP64 BLAS.
   What remains open is decode: vector-matrix products, attention over a growing KV cache, and
   per-user keys.
4. **Precision slack is unused.** Llama-3-8B needs only about 11–12 bits (perplexity 6.14 at
   2⁻¹¹ vs 6.13 in FP16), yet every LLM system runs 40–60-bit scales.
5. **Co-design works, but has mostly been done for encoders and still divides.**
   - Powerformer (fixed-denominator softmax, BERT) cuts softmax depth from 30 to 3.
   - PowerSoftmax (Pythia) still needs a depth-22 division.
   - HEAT (GPT-2, Sep 2026) learns iteration counts and reaches 37.6 s/token.

   **No paper converts a pre-trained decoder to attention with provably bounded scores** (via
   QK-norm or cosine attention) and measures its CKKS cost.
6. **Selective encryption is nearly unexplored.**
   - Only Sylph (Park et al.) mixes public and encrypted tokens, and only as a public prefix
     followed by an encrypted suffix, with no leakage analysis.
   - No paper encrypts only the detected PII spans and computes on them exactly.
7. **Our preliminary evidence (§11).** Exact, zero-depth softmax shifts built from linear
   functions of the keys cut the normalize-and-square rounds by about half: 27→13 on
   TinyStories-1M and 19→9 on GPT-2, on held-out text, with no retraining. A bounded-attention
   fine-tune is being measured (§11.4).
8. **Proposed paper (§12): "Range is the bottleneck."** It has three parts:
   - a training-free, zero-depth shift calibration for max-free softmax;
   - an optional light fine-tune that bounds attention scores by construction;
   - a measured depth, bootstrap and latency study on pre-trained decoders.

   A selective-encryption paper is a strong second option.

---

## 1. The setting

- **What is protected.** A client holds the key. The server holds the weights in plaintext and
  computes on ciphertexts. Prompt and answer stay hidden. The server still learns lengths and
  timing, and gets no integrity guarantee.
- **CKKS basics.** Packed approximate arithmetic. Each multiplication consumes a level.
  Rotations and ciphertext-by-ciphertext multiplications need key switching. Bootstrapping
  restores levels.
- **The two LLM phases differ.** Prefill uses matrix-matrix products and batches well. Decode
  uses vector-matrix products plus attention over an encrypted KV cache, which grows by one
  token per step.

**State of the art for Llama-3-8B under CKKS (end to end):**

| System | Phase | Latency | Hardware | Notes |
|---|---|---|---|---|
| Park et al. / "Sylph", arXiv 2601.18511 v2 | prefill + decode | 20 s + **18 s/token** (128 encrypted tokens); 64 s + 22 s/token with 3,968 public tokens added | 8× RTX PRO 6000 | Perplexity 6.13 = FP16 at 12 bits |
| Cachemir, arXiv 2602.11470 | decode | 1.61 min/token | 1× A100 | Interleaved replicated KV packing; CPU: 478 s per block |
| Cerium, SOSP'26 | prefill | 43 s | 8× B200 | 5,890 bootstraps per prefill |
| Odin, arXiv 2609.12378 | prefill | 366 s | 1× H100 | Open source, per the paper |
| HEAT, arXiv 2609.01730 | decode (GPT-2) | 54.3 → 37.6 s/token | GPU (Perseus) | Learns iteration counts |
| ROSETTA, CCS'26 | decode | 1.5–2.1× faster than Cachemir | CPU/GPU | CKKS + TFHE lookup tables for 1/x and 1/√x |

---

## 2. Where the time goes

| System, workload | Bootstrapping share | Softmax share | Other |
|---|---|---|---|
| Cachemir, Llama-3-8B decode block | **75% CPU (360 / 478 s); 59% A100** | ~4% (19.8 s CPU) | Plaintext × ciphertext matmuls are 78% of non-bootstrap time |
| NEXUS, BERT-base | **62%** | 5% (47 / 857 s CPU) | – |
| Cinnamon, FHE programs (ASPLOS'25) | ~70–80% | – | – |
| Park et al., per layer (8 GPUs) | fused into format conversions | 81 ms of ~550 ms (~15%) | QKV 65, O 40, gate/up 132, QKᵀ 41, PV 63 ms |
| Rho et al., ICLR'25 | – | 43.8% of a layer | the max is ~80% of softmax cost |
| **Ours, 1 attention decode step** (OpenFHE, 4-core CPU) | none (depth 21 of 24) | 36% (1/x 22%, exp 14%) | rotations 37% |

**Takeaway.** Count **levels and bootstraps**, not seconds of softmax. Every level saved inside
a layer delays the next bootstrap, and bootstraps dominate.

---

## 3. Linear algebra and packing

**Building blocks**
- Halevi–Shoup diagonal and baby-step giant-step (BSGS) methods, plus hoisting (CRYPTO'14/'18):
  30–75× faster linear transforms.
- Double hoisting (Bossuat et al., EUROCRYPT'21): about 2× more.
- Jiang et al. (CCS'18): ciphertext-by-ciphertext matmul with O(d) rotations at depth 3.

**Reductions to plaintext BLAS**
- MaMBo (CRYPTO'24).
- "Fast homomorphic linear algebra with BLAS" (J. Cryptol. 2026): within 4–12× of FP64 BLAS.
- Batched small matrices (Cheon–Kang–Lee, CRYPTO'26): 128 batches of 64×64 in 0.20 s
  (plaintext × ciphertext) and 1.08 s (ciphertext × ciphertext).
- Limitation: these need large d or many batched matrices.

**Transformer packings**

| System | Contribution |
|---|---|
| THOR (CCS'25) | Diagonal-major layout |
| MOAI (ICLR'26) | Consistent layouts; 2,448 rotations removed; 52.8% faster than THOR |
| ARION | Fuses softmax with the matmuls; 82.5% fewer rotations than MOAI [U] |
| Tricycle (CCS'26) | Tricyclic encodings; BERT-base 100.5 s on one GPU |
| Odin | Feature-major layout, no repacking between QKᵀ, softmax and PV |
| Park et al. | A different ring dimension per operator, from N=256 for coefficient-encoded plaintext × ciphertext matmul up to N=2¹⁶ |
| Cachemir | Decode vector-matrix products at depth 1. QKV rotations drop from 131,072 (NEXUS) to 52. One ciphertext holds 1,024 KV tokens. |

**GL matrix-native FHE (Gentry–Lee)**
- How it works: an encrypted matmul becomes 4 plaintext matmuls plus a key switch costing 2–4×
  a Hadamard product. Bootstrapping is at CRYPTO'26.
- **Limitation:** the unit size grows ×n because of batching, which hurts single-user decode.
- Status: shown only for encoders and batches (Terrazzo: BERT-base 7.04 s/input amortized on
  an RTX 5090). `desilofhe` is available for non-commercial use.

**Open:** decode vector-matrix products, attention against a growing KV cache, and GL for
decode.

---

## 4. Bootstrapping and level management

**Measured costs at N=2¹⁶:**

| Platform | One bootstrap |
|---|---|
| **Our OpenFHE CPU, 4 cores, 1,024 slots** | 11.9 s |
| OpenFHE CPU (FIDESlib baseline) | 18–29 s |
| FIDESlib, RTX 4090 | 74–146 ms |
| Cheddar | A100 40 ms, H100 31 ms, RTX 5090 22 ms |
| Cerium | 1×B200 14.5 ms; 8×B200 7.5 ms |
| Simulated ASICs | 0.9–6.3 ms |

Two practical points:
- **Sparse slots barely reduce bootstrap time** (64 vs 32,768 slots: 18 vs 29 s), because
  EvalMod runs on the full ring.
- **Bootstrapping to a lower target level is about 2.4× cheaper.**

**Bootstrapping algorithms**
- Sparse-secret encapsulation (ACNS'22).
- META-BTS (CCS'22), and discrete-CKKS bootstrapping at 80-bit precision (CCS'25).
- SHIP (EUROCRYPT'25): bootstrapping at N=2¹³.
- OpenFHE ≥1.5 "StC-First" mode: 1.04–1.53× faster.

**Functional or fused bootstrapping**
- Alexandru–Kim–Polyakov (CRYPTO'25): 8-bit lookup tables. Dumezy et al. (TCHES'26): up to 20
  bits. Both are in OpenFHE ≥1.4.
- RBOOT (USENIX Sec'26) computes ReLU inside EvalMod: 2.77× faster.
- BootNet and NeuJeans target CNNs.
- "Bootstrapping is All You Need" (ePrint 2026/1255) applies this to a transformer (BERT):
  662 → 349 s.
- **Not yet used in any LLM paper.**

**Bootstrap placement compilers**

| Compiler | Approach | Note |
|---|---|---|
| EVA, Hecate, ELASM | Scale management only | – |
| DaCapo (USENIX Sec'24) | First automatic bootstrap placement | 1.21× |
| Fhelipe (PLDI'24) | Packing plus placement | – |
| ReSBM (ASPLOS'25) | Placement by region | – |
| Orion (ASPLOS'25) | Placement between layers | No attention support |
| Orbit (USENIX Sec'26) | Joint rescale and bootstrap placement as an ILP | 1.19–1.73× over prior work |
| HEIR | MLIR-based compiler | – |

These target CNNs. Among LLM papers, only Cachemir extends placement into non-linear layers
and the KV cache.

---

## 5. Softmax and other non-linearities without retraining

### 5.1 Three families of softmax

| Family | Systems | How the cost scales | Status at LLM ranges |
|---|---|---|---|
| **Homomorphic max, then shift** | HETAL (ICML'23), Cerium, ARION | Comparisons cost ~log(range/tolerance) + log α each; a tournament needs log n rounds; all-pairs (Mazzone, USENIX Sec'25) needs n² slots | The max is ~80% of softmax cost. Argmax over 30k logits: 54 s CPU / 2.5 s GPU (NEXUS) |
| **Calibrated constant or "statistical" offset, exp, then divide** | NEXUS, MOAI (per-layer constants 7–13.6), THOR (midpoint of a calibrated range), Tricycle, ELLMo, THEMIS-ePrint (row-wise offset), Odin ("statistical max") | Goldschmidt division needs about log₂(dynamic range × precision) iterations. Every nat by which the offset overshoots the row max multiplies that range by e. | **Breaks:** NEXUS 5.7 bits; MOAI diverges on Llama-3-8B (Cachemir). Both diverge beyond [−8, 8] (ROSETTA). |
| **Max-free normalize-and-square** (Cho, Hanrot, Kim, Park, Stehlé, CCS'24) | Park et al., Odin (k = 2–5 per layer), ROSETTA, THOR's variant | Uses softmax(2x)ᵢ = yᵢ²/Σyⱼ². Rounds k = ⌈log₂M − log₂ln n⌉; main-thread depth ≈ 2 log M − log log n + O(1) | **State of the art.** M is still one calibrated bound per layer. |

**Cho et al. in numbers** (HEaaN, single-thread CPU, 17 s bootstrap):
- M = 256: n = 128 in 124 s; n = 1024 in 182 s; worst-case error 2⁻¹⁵·⁵.
- Batched: 0.05 s per row.
- Bootstrapping is 80% of the time.
- ROSETTA's reimplementation reports precision falling from 17 to 4 bits as n grows from 512 to
  8k, which the original does not show. **This contradiction is unresolved.**

**ROSETTA** (CCS'26) switches from CKKS to TFHE to compute 1/x and 1/√x by lookup table:
softmax 4.8× faster, decode 1.5–2.1× faster than Cachemir. Pegasus-style scheme switching
raised perplexity by 116%.

### 5.2 Normalization and activations

**LayerNorm / RMSNorm (inverse square root)**
- Methods: Newton (NEXUS), THOR's aSOR iteration (2–9× less work), MOAI (4 Newton + 2
  Goldschmidt), Odin (minimax).
- **The input range is the problem:** Llama RMSNorm inputs reach 2,244. Park et al.'s outlier
  suppression (sink prefix plus orthogonal rotations) cuts that to 7.65.

**GELU / SiLU**
- NEXUS: 4 pieces using 3 sign calls. MOAI: 3 pieces. THOR: through tanh.
- Park et al.: SiLU degree 83 → 31 after range reduction.

**Domain-extension polynomials** (Cheon–Kim–Park, TIFS'22) reach O(log R) depth on [−R, R].

### 5.3 Open problems in this area

1. Every normalize-and-square deployment sets k from **one** range M. **Zero-depth exact shifts
   can shrink the gap between the offset and the true row max** (§11.3).
2. Approximate max tuned to the softmax tolerance (α ≈ 9–10 bits, not 20).
3. **How safe are calibrated ranges?** Out-of-range inputs make polynomials diverge, and no
   paper measures failure rates under distribution shift.
4. Published numbers conflict:
   - THOR's softmax depth is reported as 30 by Powerformer and 43 by Cachemir;
   - Cho et al.'s precision under reimplementation (above).

   A systematization-of-knowledge (SoK) paper with an open cost model could resolve these.
5. Use functional bootstrapping for narrow-range non-linearities (GELU, the inverse square
   root).

---

## 6. Co-design: changing the model

### 6.1 FHE/MPC-friendly transformers

| Work | What it replaces | Training | Quality | Measured cost |
|---|---|---|---|---|
| THE-X (Findings ACL'22) | ReLU (client-aided), ReLU-based softmax, learned LayerNorm | 2-stage FT | BERT-tiny −1.5 GLUE | – |
| MPCFormer (ICLR'23) | Quadratic GELU, 2Quad softmax | Layer-wise distillation | BERT-base 80.3 vs 82.8 | 2PC 2.2–5.3× faster |
| Zimerman et al. (ICML'24) | Sigmoid-like attention + range losses | From scratch | Perplexity 16.9 → 18.9 | 6 layers: 305 s (A100) |
| PowerSoftmax (AISTATS'26) | xᵖ/Σxᵖ; division by Goldschmidt (depth 22) | Pythia-70M/1.4B converted | LAMBADA .568 → .487 | Pythia-70M: 93 s per 128 tokens |
| Powerformer (ACL'25) | BPMax (fixed denominator), Batch LN | 1-step distillation | BERT-base matched | **Softmax depth 30 → 3**; 344.5 vs 626 s |
| Encryption-friendly LLM (ICLR'25) | Gaussian kernel (no max, no division) | LoRA + range penalty | GLUE .662 vs .707 (2 layers) | 61.8 → 26.5 s |
| CGF-softmax | Mean/variance shift | Fine-tuned | – | Depth k+6 vs ≥8k+9 |
| AERO | No LN, ReLU, entropy regularizer | From scratch | Perplexity 2.69 → 3.21 | MPC only |
| ENSI (preprint) | Sigmoid with **no** retraining (BitNet-3B) | none | −5.8 PIQA | Sigmoid uses 5 levels |
| ePrint 2026/1255 | DyT replaces LN; functional bootstrapping | Distillation | Matches BERT [U] | 662 → 349 s |
| HEAT (Sep 2026) | Nothing; learns iteration counts | FT (GPT-2) | Perplexity 44.6 → 30.7 | **54.3 → 37.6 s/token**; bootstraps 513 → 326 |

### 6.2 Unencrypted ML techniques that would make FHE cheaper

**Bounded logits**
- QK-norm (Henry'20; ViT-22B): stops logit growth.
- Scaled cosine attention (Swin V2).
- Soft-capping (Gemma 2, cap 50).
- nGPT.

**Softmax alternatives**
- Sigmoid attention (ICLR'25): matches softmax at 1–7B **trained from scratch**, with
  b = −log n.
- Polynomial or Taylor kernels (Based, PolySketchFormer).
- ReLU attention: weak evidence for language models.

**Normalization-free blocks**
- DyT (CVPR'25): matches at 7–70B from scratch.
- FakeLN removal of LayerNorm from pre-trained GPT-2: +0.03–0.07 cross-entropy.

**Converting a pre-trained model**

| Method | Recipe | Quality |
|---|---|---|
| LoLCATs | Attention-output matching + LoRA, 0.4% of prior work's tokens | Closes ~78% of the MMLU gap at 70B/405B |
| Hedgehog | MLP feature maps that mimic softmax | 99% of quality |
| SUPRA | Uplifts a pre-trained model to linear attention (~5% of pre-training) | MMLU gap persists |
| RADLADS | Distillation into linear attention | Under $2k at 72B |

**Lesson:** perplexity recovers easily; in-context recall does not.

### 6.3 What is unclaimed

No FHE or MPC paper uses QK-norm, cosine or soft-capped attention to **guarantee** softmax input
ranges in a **pre-trained decoder**, together with measured CKKS cost and quality against
conversion compute. Each ingredient has precedent. The combination, with systematic ablations,
appears open. Novelty risk is medium; see §12.

---

## 7. Selective encryption and hybrid protocols

**Mixed public/encrypted tokens**
- **Sylph** (Park et al.) supports only a public prefix plus an encrypted suffix. It splits
  attention into PC-attention (encrypted queries against plaintext keys/values) and
  CC-attention (encrypted against encrypted).
  - Per-layer prefill: PC ≈ 1.37 s vs CC ≈ 0.19 s.
  - The authors call the decode gains "modest".
  - **No threat model.** Its "public" retrieved documents and conversation history can depend on
    the private query.
- **Confidential Prompting** (TEE) and **Cascade** (no cryptography) use the same trick:
  partial softmaxes merged by max/sum rescaling.

**Span-level protection without encryption**
- Pr∈∈mpt (NDSS'26): format-preserving encryption plus metric differential privacy, with a
  formal leakage game.
- HERALD: deterministic ciphertext tokens.
- PrivacyRestore, PAPILLON.
- None of these computes on the exact private values under HE or MPC.

**Isolating segments inside attention**
- Block-Attention (ICLR'25): −16 points without fine-tuning; within 0.7–2.5 points after a
  short fine-tune.
- Parallel Context Windows, Prompt Cache, CacheBlend.
- All of these are for efficiency. **None is for confidentiality.**

**Hybrid protocols**
- BOLT: 25.7 GB per BERT-base inference.
- BumbleBee: about 8 min per LLaMA-7B token.
- Client-aided protocols (THE-X, Concrete-ML hybrid, Safhire) leak activations to the client,
  which is harmless for open-weight models.
- Li–Micciancio: decrypted CKKS results must never return to the server.

**Formal leakage analyses**
- Pr∈∈mpt covers plaintext sanitization only.
- CipherPrune argues informally.
- **No analysis of mixed-attention leakage** (positions, lengths, query-dependent context).

---

## 8. Precision and parameters

**Precision needed**
- Llama-3-8B: perplexity 6.33 / 6.15 / 6.14 / 6.13 at 2⁻⁹ / 2⁻¹⁰ / 2⁻¹¹ / 2⁻¹², vs 6.13 in
  FP16.
- Park et al. use 20-bit bootstrapping with range scaling B=128, which leaves 13 effective bits.

**Using lower precision to cheapen parameters**
- Cheddar with Δ=2³⁵.
- BootNet at N=2¹⁵; SWIFT reaches 12 bits at logN=15.
- Grafting and Zephyr: 32-bit primes on GPU.
- Seo & Min (ePrint 2026/2242): 83.6% fewer bootstraps than THOR on BERT-like models.
- **Not yet done for LLM decode.**

---

## 9. Hardware

**Per primitive at N=2¹⁶ (GPU vs our 4-core CPU)**
- GPUs are 140–325× faster per primitive. Cheddar on an RTX 4090: multiply 0.53 ms, rotation
  0.48 ms, vs our 148 / 155 ms.
- GPUs are 300–900× faster for bootstrapping.

**ASICs**
- Simulated: bootstrapping in 0.9–6.3 ms. Cinnamon: BERT-base in 1.7–3.8 s on 4–12 chips.
- Real silicon: Intel Heracles (ISSCC'26) is 1,074–5,547× faster than a Xeon on primitives,
  with 48 GB of HBM. No ML numbers yet.

**Not yet in the literature**
- No paper projects LLM token latency on an ASIC.
- Encoded weights take 52 GiB to 982 GB. That exceeds on-package memory, so decode is likely
  bound by weight traffic unless plaintexts are generated on-chip.

**Contradiction:** Cerium's stated speedups over ASICs conflict with its absolute numbers.

---

## 10. Open problems, ranked for a small team

Ratings assume a CPU-only team with optional GPU access later.

| # | Problem | Why open | Feasibility | Novelty risk |
|---|---|---|---|---|
| 1 | **Range control as the root cause of softmax/norm depth**: training-free shift calibration plus bounded-logit conversion of pre-trained decoders | §5.3, §6.3 | High (plaintext traces + OpenFHE) | Medium |
| 2 | **Selective encryption of private spans** (Redact-then-Bind; secret-isolated attention) with a formal leakage function | §7 | High (our gateway + PUPA + cost model) | Low–medium |
| 3 | Query-dependent "public" context leaks the query in RAG; attack + k-anonymous context defence | §7 | Medium | Low–medium |
| 4 | Precision-aware CKKS parameters for decode (~11 bits needed) | §8 | High | Medium |
| 5 | Softmax SoK with an open, validated cost model | §5.3 | Medium | Low (needs breadth) |
| 6 | How safe are calibrated ranges under distribution shift and adversarial prompts | §5.3 | High | Low–medium |
| 7 | Decode-aware bootstrap placement that merges sparse ciphertexts | §4 | Medium | Medium |
| 8 | GL matrix FHE for decode and the KV cache | §3 | Medium (desilofhe) | Medium–high |

---

## 11. Our preliminary evidence (this repository)

All runs: CPU only, real models, held-out TinyStories text.

### 11.1 TinyStories-1M range calibration (`cryptollm/fhe/tinystories/calibrate.py`)

**Inputs that are already cheap**
- LayerNorm variance ranges are 3–6× → inverse square root needs Chebyshev degree 5–13
  (depth 3–4).
- GELU inputs stay within ±1.7 → degree 5–13.

**The attention problem**
- Scores span −217 to +22 (GPT-Neo does not scale scores).
- Within a single row, scores span up to 74 nats.
- Optimistic total: ~32–35 levels per layer, which means ~7 bootstraps per layer with our
  parameters.

### 11.2 Row maxima make a constant offset impossible

Within a layer, the **row maximum** varies by 15–163 nats across rows. With one constant offset
per layer, the softmax denominator could fall to e⁻¹⁶³. That rules out the shift-then-divide
family and forces a large M (more rounds) on normalize-and-square.

### 11.3 Zero-depth exact shifts halve the normalize-and-square rounds (`shift_gap.py`)

**Idea.** Softmax is shift-invariant, so any **linear functional of the keys** gives an exact
shift that needs only additions:
- the mean-key score q·mean(k);
- the first-token (attention-sink) score q·k₀;
- or a combination fitted on calibration data.

**Calibration protocol**
- Fit on 60 stories, test on 40 held-out stories, n = 128.
- Each shift gets a margin at the 99.9th percentile, so the shift stays at or above the true
  max on 99.9% of rows.

| Total squaring rounds over all layers | Calibrated constant (today's practice) | Best single reference | **Fitted linear shift** |
|---|---|---|---|
| TinyStories-1M (8 layers) | 27 | 18 | **13 (−52%)** |
| GPT-2 small (12 layers) | 19 | 9 (first token) | **9 (−53%)** |

- Each round costs about 2 main-thread levels.
- The fitted shift undershoots the true max on **0.10%** of held-out rows, by construction of
  the margin. This must be absorbed by the first exp polynomial's domain, or certified (§12).
- The best reference depends on the model: GPT-2's first-token attention sink tracks the max,
  TinyStories' mean key does.

### 11.4 Bounded-attention fine-tuning pilot (`bounded_attention.py`)

_Pending: the 1,500-step fine-tune (B = 4) is running; results will be added here._

---

## 12. Proposed publishable work

_Pending: a targeted novelty check of the two candidate contributions is running; the proposal will be finalized from its verdict._

---

## References (by section; full lists in the review-track reports)

**Systems.**
- Sylph https://arxiv.org/abs/2601.18511
- Cachemir https://arxiv.org/abs/2602.11470
- Cerium https://arxiv.org/abs/2512.11269
- Odin https://arxiv.org/abs/2609.12378
- ROSETTA https://arxiv.org/abs/2609.16915
- HEAT https://arxiv.org/abs/2609.01730
- NEXUS https://eprint.iacr.org/2024/136
- THOR https://eprint.iacr.org/2024/1881
- MOAI https://eprint.iacr.org/2025/991
- Tricycle https://eprint.iacr.org/2025/1200
- ARION https://eprint.iacr.org/2025/2271
- THEMIS https://eprint.iacr.org/2026/2264
- Perseus https://eprint.iacr.org/2026/2735

**Softmax / non-linear.**
- Cho et al. https://arxiv.org/abs/2410.11184
- HETAL https://arxiv.org/abs/2403.14111
- CKK20 https://eprint.iacr.org/2019/1234
- Lee–Lee–No–Kim https://eprint.iacr.org/2020/834
- Mazzone et al. https://arxiv.org/abs/2412.15126
- Domain extension https://eprint.iacr.org/2022/280
- aSOR https://eprint.iacr.org/2024/1366
- Pegasus https://eprint.iacr.org/2020/1606
- Functional bootstrapping https://eprint.iacr.org/2024/1623 and https://eprint.iacr.org/2025/1301
- ATLAS https://eprint.iacr.org/2026/1817
- ELLMo https://eprint.iacr.org/2026/198
- CGF-softmax https://arxiv.org/abs/2602.01621
- Seo & Min https://eprint.iacr.org/2026/2242

**Co-design.**
- THE-X https://arxiv.org/abs/2206.00216
- MPCFormer https://arxiv.org/abs/2211.01452
- Zimerman et al. https://arxiv.org/abs/2311.08610
- PowerSoftmax https://arxiv.org/abs/2410.09457
- Powerformer https://aclanthology.org/2025.acl-long.543
- Encryption-friendly LLM https://arxiv.org/abs/2410.02486
- AERO https://arxiv.org/abs/2410.13060
- ENSI https://arxiv.org/abs/2509.09424
- ePrint 2026/1255 https://eprint.iacr.org/2026/1255
- Sigmoid attention https://arxiv.org/abs/2409.04431
- QK-norm https://arxiv.org/abs/2010.04245
- ViT-22B https://proceedings.mlr.press/v202/dehghani23a/dehghani23a.pdf
- Swin V2 https://arxiv.org/abs/2111.09883
- Gemma 2 https://arxiv.org/abs/2408.00118
- DyT https://arxiv.org/abs/2503.10622
- LN removal https://arxiv.org/abs/2409.13710 and https://arxiv.org/abs/2507.02559
- LoLCATs https://arxiv.org/abs/2410.10254
- Hedgehog https://arxiv.org/abs/2402.04347
- Attention sinks https://arxiv.org/abs/2410.10781

**Linear algebra / bootstrapping / hardware.**
- MaMBo https://eprint.iacr.org/2024/1284
- BLAS https://arxiv.org/abs/2503.16080
- Gentry–Lee https://eprint.iacr.org/2025/1935
- GL bootstrapping https://eprint.iacr.org/2026/956
- Cheon–Kang–Lee https://eprint.iacr.org/2025/1957
- Halevi–Shoup https://eprint.iacr.org/2018/244
- Bossuat et al. https://eprint.iacr.org/2020/1203
- DaCapo https://www.usenix.org/conference/usenixsecurity24/presentation/cheon
- Orion https://arxiv.org/abs/2311.03470
- Orbit https://eprint.iacr.org/2026/213
- HEIR https://arxiv.org/abs/2508.11095
- Cheddar https://arxiv.org/abs/2407.13055
- FIDESlib https://arxiv.org/abs/2507.04775
- Cinnamon https://www.pdl.cmu.edu/ftp/BigLearning/cinnamon.pdf
- WHET https://arxiv.org/abs/2606.11541
- Heracles https://spectrum.ieee.org/fhe-intel

**Selective / hybrid.**
- Pr∈∈mpt https://arxiv.org/abs/2504.05147
- Confidential Prompting https://arxiv.org/abs/2409.19134
- Cascade / Hidden No More https://arxiv.org/abs/2505.18332
- Block-Attention https://arxiv.org/abs/2409.15355
- BOLT https://eprint.iacr.org/2023/1893
- BumbleBee https://ndss-symposium.org/ndss-paper/bumblebee-secure-two-party-inference-framework-for-large-transformers
- Safhire https://arxiv.org/abs/2509.01253
- CipherPrune https://arxiv.org/abs/2502.16782
- Li–Micciancio https://eprint.iacr.org/2020/1533
- PAPILLON https://aclanthology.org/2025.naacl-long.173
