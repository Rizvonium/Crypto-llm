"""One decode step of causal self-attention, fully encrypted (CKKS via OpenFHE, 128-bit).

This is the operation an LLM repeats for every generated token, in every head of every
layer: the new token's query attends over the cached keys/values of all earlier tokens.
Here the server holds the projection weights in plaintext and the KV cache *encrypted*;
the client sends one encrypted hidden vector and gets one encrypted output vector back.
Softmax runs homomorphically: exp(x) and 1/x are Chebyshev polynomials, and nothing is
decrypted on the server. Sizes: d_model = 32, head dim = 16, context = 8 tokens.

    server, given Enc(x_new), Enc(K_prev), Enc(V_prev):
      q, k, v   = Wq x, Wk x, Wv x                  (baby-step/giant-step diagonal matvecs)
      K, V      = K_prev + k, V_prev + v            (new token in block 0 of the packed cache)
      s_j       = <q, k_j> / sqrt(16)               (replicate q, multiply, rotate-and-sum)
      p_j       = exp(s_j) / sum_i exp(s_i)         (Chebyshev exp on [-B, B]; Chebyshev 1/x)
      o         = sum_j p_j v_j ;  y = Wo o         (rotate-and-sum, matvec)

Layout: block j = slots [16j, 16j+16) holds token j; 1024 slots, ring dimension 2^16.
Multiplicative depth budget: 24 (no bootstrapping inside the step). A real multi-layer
model must also bootstrap between layers (11.9 s each on our 4-core CPU, see
bench_bootstrap.py). Needs: pip install openfhe (CPython 3.12 wheels) and ~4 GB RAM.
"""

from __future__ import annotations

import math
import resource
import time

import numpy as np
from openfhe import (CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature, ScalingTechnique,
                     SecretKeyDist, SecurityLevel)

D, H, CTX, SLOTS = 32, 16, 8, 1024
B = 1.5                     # scores are kept inside [-B, B] (checked below)
EXP_DEGREE, INV_DEGREE = 13, 59   # Chebyshev degrees (exp error ~1e-9 on [-B, B])
G = 8                       # giant step for the matvecs
ROT_KEYS = [1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32, 64, -1, -2, -4, -8, -16, -32, -64]


def make_context():
    p = CCParamsCKKSRNS()
    p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
    p.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
    p.SetMultiplicativeDepth(24)
    p.SetScalingModSize(50)
    p.SetFirstModSize(60)
    p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
    p.SetBatchSize(SLOTS)
    cc = GenCryptoContext(p)
    for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE,
              PKESchemeFeature.ADVANCEDSHE):
        cc.Enable(f)
    return cc


class Server:
    """Plaintext weights, encrypted data, evaluation keys only."""

    def __init__(self, cc, Wq, Wk, Wv, Wo):
        self.cc = cc
        self.Wq = Wq / math.sqrt(H)  # fold the 1/sqrt(d_head) scaling into Wq
        self.Wk, self.Wv, self.Wo = Wk, Wv, Wo
        # plaintexts are encoded at the level of the ciphertext they meet (FLEXIBLEAUTO scaling)
        self.pt = lambda v, ct: cc.MakeCKKSPackedPlaintext(list(map(float, v)), 1, ct.GetLevel(), None, SLOTS)
        mask = np.zeros(SLOTS)
        self.block_starts = mask.copy(); self.block_starts[[H * j for j in range(CTX)]] = 1
        self.first_block = mask.copy(); self.first_block[:H] = 1
        self.e0 = mask.copy(); self.e0[0] = 1

    def rot(self, ct, k):
        return self.cc.EvalRotate(ct, k)

    def rot_add(self, ct, steps):
        for s in steps:
            ct = self.cc.EvalAdd(ct, self.rot(ct, s))
        return ct

    def babies(self, ct_ext):
        return [ct_ext] + [self.rot(ct_ext, b) for b in range(1, G)]

    def matvec(self, babies, W, n_terms):
        """y[r] = sum_i W[r, (r+i) % n_in] * x[(r+i) % n_in] (diagonal method, BSGS).
        The input must be repeated in the slots so that x_ext[r + i] = x[(r+i) % n_in]."""
        out_dim, in_dim = W.shape
        total = None
        for j in range(n_terms // G):
            inner = None
            for b in range(G):
                i = G * j + b
                diag = np.zeros(SLOTS)
                for r in range(out_dim):
                    diag[r] = W[r, (r + i) % in_dim]
                diag = np.roll(diag, G * j)          # pre-rotate by -G*j
                term = self.cc.EvalMult(babies[b], self.pt(diag, babies[b]))
                inner = term if inner is None else self.cc.EvalAdd(inner, term)
            inner = self.rot(inner, G * j) if j else inner
            total = inner if total is None else self.cc.EvalAdd(total, inner)
        return total

    def attend(self, x_ext, K_prev, V_prev):
        cc = self.cc
        xb = self.babies(x_ext)                 # 7 rotations shared by the Q, K, V projections
        q = self.matvec(xb, self.Wq, D)
        k = self.matvec(xb, self.Wk, D)
        v = self.matvec(xb, self.Wv, D)
        K = cc.EvalAdd(K_prev, k)
        V = cc.EvalAdd(V_prev, v)
        q_rep = self.rot_add(q, [-H, -2 * H, -4 * H])                       # q in every block
        s = self.rot_add(cc.EvalMult(q_rep, K), [1, 2, 4, 8])                # s_j at slot 16j
        s = cc.EvalMult(s, self.pt(self.block_starts, s))
        e = cc.EvalChebyshevFunction(math.exp, s, -B, B, EXP_DEGREE)
        e = cc.EvalMult(e, self.pt(self.block_starts, e))                    # drop exp(0)=1 slots
        z = self.rot_add(e, [H, 2 * H, 4 * H])                               # sum at slot 0
        z_lo, z_hi = CTX * math.exp(-B), CTX * math.exp(B)
        z = cc.EvalMult(z, self.pt(self.e0, z))
        z = cc.EvalAdd(z, self.pt((1 - self.e0) * z_lo, z))                 # keep 1/x in range
        zinv = cc.EvalChebyshevFunction(lambda t: 1.0 / t, z, z_lo * 0.9, z_hi * 1.1, INV_DEGREE)
        zinv = self.rot_add(cc.EvalMult(zinv, self.pt(self.e0, zinv)), [-H, -2 * H, -4 * H])
        p = cc.EvalMult(e, zinv)                                             # p_j at slot 16j
        p_rep = self.rot_add(p, [-1, -2, -4, -8])                            # p_j across block j
        o = self.rot_add(cc.EvalMult(p_rep, V), [H, 2 * H, 4 * H])           # o in slots 0..15
        o = cc.EvalMult(o, self.pt(self.first_block, o))
        o_rep = self.rot_add(o, [-H, -2 * H])                                # 4 copies of o
        return self.matvec(self.babies(o_rep), self.Wo, H)


def reference(x, K_prev_rows, V_prev_rows, Wq, Wk, Wv, Wo):
    q, k, v = Wq @ x / math.sqrt(H), Wk @ x, Wv @ x
    K = np.vstack([k, K_prev_rows]); V = np.vstack([v, V_prev_rows])
    s = K @ q
    p = np.exp(s - s.max()); p /= p.sum()
    return Wo @ (p @ V), s


def main(seed: int = 0, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    Wq, Wk, Wv = (rng.normal(0, 0.12, (H, D)) for _ in range(3))
    Wo = rng.normal(0, 0.25, (D, H))
    x = rng.normal(0, 1, D)
    K_prev_rows = rng.normal(0, 0.9, (CTX - 1, H))   # keys/values of the 7 earlier tokens
    V_prev_rows = rng.normal(0, 1, (CTX - 1, H))
    _, scores = reference(x, K_prev_rows, V_prev_rows, Wq, Wk, Wv, Wo)
    Wq = Wq * min(1.0, 0.8 * B / np.abs(scores).max())   # FHE needs a known range for exp()
    y_ref, scores = reference(x, K_prev_rows, V_prev_rows, Wq, Wk, Wv, Wo)
    assert np.abs(scores).max() < B

    t0 = time.perf_counter()
    cc = make_context()
    keys = cc.KeyGen()
    cc.EvalMultKeyGen(keys.secretKey)
    cc.EvalRotateKeyGen(keys.secretKey, ROT_KEYS)
    log = print if verbose else (lambda *a, **k: None)
    log(f"[client] ring dim {cc.GetRingDimension()}, keygen {time.perf_counter() - t0:.1f}s, "
        f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")

    def enc(vec):
        return cc.Encrypt(keys.publicKey, cc.MakeCKKSPackedPlaintext(list(map(float, vec)), 1, 0, None, SLOTS))

    packed_K = np.zeros(SLOTS); packed_V = np.zeros(SLOTS)
    for j in range(1, CTX):                         # block 0 is reserved for the new token
        packed_K[H * j:H * (j + 1)] = K_prev_rows[j - 1]
        packed_V[H * j:H * (j + 1)] = V_prev_rows[j - 1]
    x_ext = np.zeros(SLOTS); x_ext[:D] = x; x_ext[D:2 * D] = x
    ct_x, ct_K, ct_V = enc(x_ext), enc(packed_K), enc(packed_V)

    server = Server(cc, Wq, Wk, Wv, Wo)              # never touches keys.secretKey
    t1 = time.perf_counter()
    ct_y = server.attend(ct_x, ct_K, ct_V)
    t_server = time.perf_counter() - t1

    pt_y = cc.Decrypt(ct_y, keys.secretKey)
    pt_y.SetLength(D)
    y = np.array(pt_y.GetRealPackedValue())
    err = np.abs(y - y_ref).max()
    t2 = time.perf_counter()
    for _ in range(1000):
        reference(x, K_prev_rows, V_prev_rows, Wq, Wk, Wv, Wo)
    t_plain = (time.perf_counter() - t2) / 1000
    log(f"[server] encrypted attention step (8-token context, 1 head): {t_server:.1f}s "
        f"vs plaintext {t_plain * 1e6:.0f} us (~{t_server / t_plain:,.0f}x)")
    log(f"[client] max |encrypted - exact| = {err:.2e}  (|y| max {np.abs(y_ref).max():.2f}); "
        f"multiplicative levels used: {ct_y.GetLevel()}")
    log(f"peak RAM {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB")
    return {"max_error": err, "server_seconds": t_server, "levels_used": ct_y.GetLevel()}


if __name__ == "__main__":
    main()
