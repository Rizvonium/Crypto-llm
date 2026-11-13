"""Max-free homomorphic softmax by normalize-and-square, with pluggable shifts and round rules.

Identity (Cho, Hanrot, Kim, Park, Stehle, CCS'24): with T(v)_i = v_i^2 / sum_j v_j^2,
    softmax(x) = T^k( exp(x / 2^k) )        for any k >= 0,
and T is scale-invariant, so intermediate normalisations only have to keep magnitudes in a
safe range (~10% accuracy); one final normalisation fixes the scale exactly.

Pipeline (every range and degree below is PUBLIC, fixed from calibration data):
    x  = s - shift                      constant shift: 0 levels; fitted linear shift: 1 level
    y0 = exp(x / 2^k)                   Chebyshev on [-M, over]  (the low tail costs degree, once)
    y  = y0 / sum(y0)                   approximate inverse   (only if k >= 1)
    k times:  y = y^2 / sum(y^2)        approximate inverse on [~1/n, ~1]
    y  = y / sum(y)                     accurate inverse      (the final normalisation)

Round rules:
    M-rule (current practice): k = ceil(log2 M - log2 ln n), M = calibrated width shift - row min
    G-rule (ours):             k = ceil(log2 G - log2 ln n), G = calibrated gap shift - row max
The G-rule only guarantees that the row MAXIMUM survives exp(x/2^k) >= 1/n; the low tail is
absorbed by the (wider) exp polynomial and the first normalisation, both paid once.

Rows are packed interleaved (slot = j*R + r for row r, position j) so that row sums are
rotate-and-add with steps R, 2R, ..., (n/2)R and land in every slot of the row at depth 0.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

# OpenFHE Chebyshev series on [-1, 1]: degree -> multiplicative depth (general [a, b]: +1, see Plan.depth)
DEGREE_DEPTH = {5: 3, 13: 4, 27: 5, 59: 6, 119: 7, 247: 8, 495: 9, 1007: 10}


def cheb_fit(f, a, b, degree):
    return np.polynomial.chebyshev.Chebyshev.interpolate(f, degree, domain=[a, b])


def min_degree(f, a, b, tol, relative=False):
    xs = np.linspace(a, b, 6001)
    fx = f(xs)
    for deg in DEGREE_DEPTH:
        err = np.abs(cheb_fit(f, a, b, deg)(xs) - fx)
        if (err <= tol * np.abs(fx)).all() if relative else err.max() <= tol:
            return deg
    raise ValueError(f"no degree <= 1007 approximates the function on [{a:.3g}, {b:.3g}]")


@dataclass
class Plan:
    """Everything the server needs; all public."""
    n: int
    k: int
    shift_kind: str                    # "constant" or "fitted"
    shift_const: float                 # constant part (incl. margin)
    shift_w: tuple = (0.0, 0.0)        # fitted weights on (row mean, sink score)
    exp_dom: tuple = (0.0, 0.0)        # domain in x for exp(x / 2^k)
    first_dom: tuple = (1.0, 1.0)      # domain of sum(y0)
    round_dom: tuple = (0.0, 0.0)      # domain of sum(y^2) after normalisation
    final_dom: tuple = (0.7, 1.4)      # domain of the final sum
    degrees: dict = field(default_factory=dict)

    def depth(self) -> int:
        # OpenFHE spends one extra level per Chebyshev call to map [a, b] onto [-1, 1]
        d = {deg: dep + 1 for deg, dep in DEGREE_DEPTH.items()}
        total = (1 if self.shift_kind == "fitted" else 0) + d[self.degrees["exp"]]
        if self.k >= 1:
            total += d[self.degrees["first"]] + 1 + self.k * (d[self.degrees["round"]] + 2)
        total += d[self.degrees["final"]] + 1
        return total


def apply_shift(plan: Plan, rows: np.ndarray) -> np.ndarray:
    if plan.shift_kind == "constant":
        return rows - plan.shift_const
    a, b = plan.shift_w
    return rows - (a * rows.mean(1, keepdims=True) + b * rows[:, :1] + plan.shift_const)


def make_plan(cal_rows: np.ndarray, shift_kind: str, rule: str, tol_bits: int = 12) -> Plan:
    """Calibrate shift, k, domains and degrees on calibration rows (shape: rows x n)."""
    n = cal_rows.shape[1]
    rmax, rmin = cal_rows.max(1), cal_rows.min(1)
    if shift_kind == "constant":
        c = np.percentile(rmax, 99.9)
        plan = Plan(n=n, k=0, shift_kind="constant", shift_const=float(c))
    else:
        feats = np.stack([cal_rows.mean(1), cal_rows[:, 0], np.ones(len(cal_rows))], 1)
        w, *_ = np.linalg.lstsq(feats, rmax, rcond=None)
        margin = np.percentile(rmax - feats @ w, 99.9)
        plan = Plan(n=n, k=0, shift_kind="fitted", shift_const=float(w[2] + margin), shift_w=(float(w[0]), float(w[1])))
    x = apply_shift(plan, cal_rows)
    G = max(np.percentile(-x.max(1), 99.9), 1e-6)       # gap shift - row max
    M = np.percentile(-x.min(1), 99.9)                   # width shift - row min
    # allowance for rows whose max exceeds the shift (calibrated at the 99.99th percentile;
    # rarer rows leave the polynomial domain and are reported as failures)
    over = max(0.5, 1.2 * np.percentile(np.maximum(x.max(1), 0), 99.99))
    width = M if rule == "M" else G
    k0 = max(0, math.ceil(math.log2(width) - math.log2(math.log(n))))
    tol = 2.0 ** -tol_bits
    for k in range(k0, k0 + 6):                         # if no polynomial fits, add a round
        try:
            s = 2.0 ** k
            exp_dom = (float(-1.1 * M), float(over))
            sums = np.exp(np.clip(x, -1.1 * M, over) / s).sum(1)
            degrees = {"exp": min_degree(lambda t: np.exp(t / s), *exp_dom, tol / n)}
            if k >= 1:
                first_dom = (float(sums.min() * 0.7), float(sums.max() * 1.4))
                degrees["first"] = min_degree(lambda t: 1 / t, *first_dom, 0.1, relative=True)
                degrees["round"] = min_degree(lambda t: 1 / t, 0.6 / n, 1.6, 0.1, relative=True)
                final_dom = (0.7, 1.4)
            else:
                first_dom = final_dom = (float(sums.min() * 0.7), float(sums.max() * 1.4))
            degrees["final"] = min_degree(lambda t: 1 / t, *final_dom, tol, relative=True)
        except ValueError:
            continue
        plan.k, plan.exp_dom, plan.first_dom, plan.final_dom, plan.degrees = k, exp_dom, first_dom, final_dom, degrees
        plan.round_dom = (0.6 / n, 1.6)
        return plan
    raise ValueError("could not find a feasible plan")


# ----------------------------------------------------------------------------- emulation
def emulate(plan: Plan, rows: np.ndarray):
    """Exactly the polynomial pipeline the encrypted server runs (float64, no CKKS noise).
    Returns probabilities and a per-row flag: did any input leave its polynomial's domain?"""
    deg = plan.degrees
    s = 2.0 ** plan.k
    out_of_domain = np.zeros(len(rows), bool)

    def poly(f, dom, d, v):
        nonlocal out_of_domain
        out_of_domain |= ((v < dom[0]) | (v > dom[1])).reshape(len(rows), -1).any(1)
        return cheb_fit(f, *dom, d)(v)

    x = apply_shift(plan, rows)
    y = poly(lambda t: np.exp(t / s), plan.exp_dom, deg["exp"], x)
    if plan.k >= 1:
        y = y * poly(lambda t: 1 / t, plan.first_dom, deg["first"], y.sum(1, keepdims=True))
        for _ in range(plan.k):
            y2 = y * y
            y = y2 * poly(lambda t: 1 / t, plan.round_dom, deg["round"], y2.sum(1, keepdims=True))
    y = y * poly(lambda t: 1 / t, plan.final_dom, deg["final"], y.sum(1, keepdims=True))
    return y, out_of_domain


def exact_softmax(rows):
    e = np.exp(rows - rows.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


# ----------------------------------------------------------------------------- CKKS
class FheSoftmax:
    """OpenFHE CKKS implementation of the same pipeline for R rows of length n (R*n slots)."""

    def __init__(self, plan: Plan, rows_per_ct: int, secure: bool = True, scale_bits: int = 40):
        from openfhe import (CCParamsCKKSRNS, GenCryptoContext, PKESchemeFeature, ScalingTechnique,
                             SecretKeyDist, SecurityLevel)

        self.plan, self.R = plan, rows_per_ct
        self.slots = plan.n * rows_per_ct
        p = CCParamsCKKSRNS()
        if secure:
            p.SetSecurityLevel(SecurityLevel.HEStd_128_classic)
        else:  # tiny insecure ring, for unit tests only
            p.SetSecurityLevel(SecurityLevel.HEStd_NotSet)
            p.SetRingDim(2 * self.slots)
        p.SetSecretKeyDist(SecretKeyDist.UNIFORM_TERNARY)
        p.SetMultiplicativeDepth(plan.depth() + 2)  # +1 for FLEXIBLEAUTO's pending rescale, +1 margin
        p.SetScalingModSize(scale_bits)
        p.SetFirstModSize(scale_bits + 10)
        p.SetScalingTechnique(ScalingTechnique.FLEXIBLEAUTO)
        p.SetBatchSize(self.slots)
        cc = GenCryptoContext(p)
        for f in (PKESchemeFeature.PKE, PKESchemeFeature.KEYSWITCH, PKESchemeFeature.LEVELEDSHE,
                  PKESchemeFeature.ADVANCEDSHE):
            cc.Enable(f)
        self.cc = cc
        self.keys = cc.KeyGen()
        cc.EvalMultKeyGen(self.keys.secretKey)
        steps = [self.R * 2 ** i for i in range(int(math.log2(plan.n)))]
        # negative steps are only needed to broadcast the sink score (fitted shift)
        cc.EvalRotateKeyGen(self.keys.secretKey, steps + ([-t for t in steps] if plan.shift_kind == "fitted" else []))
        self.steps = steps

    # packing helpers (client side)
    def encrypt(self, rows: np.ndarray):
        v = np.zeros(self.slots)
        for r in range(rows.shape[0]):
            v[r::self.R] = rows[r]
        pt = self.cc.MakeCKKSPackedPlaintext(list(map(float, v)), 1, 0, None, self.slots)
        return self.cc.Encrypt(self.keys.publicKey, pt)

    def decrypt(self, ct, n_rows):
        pt = self.cc.Decrypt(ct, self.keys.secretKey)
        pt.SetLength(self.slots)
        v = np.array(pt.GetRealPackedValue())
        return np.stack([v[r::self.R] for r in range(n_rows)])

    # server side
    def _rowsum(self, ct):
        for t in self.steps:
            ct = self.cc.EvalAdd(ct, self.cc.EvalRotate(ct, t))
        return ct

    def _pt(self, v, ct):
        return self.cc.MakeCKKSPackedPlaintext(list(map(float, v)), 1, ct.GetLevel(), None, self.slots)

    def run(self, ct):
        cc, plan, deg = self.cc, self.plan, self.plan.degrees
        s = 2.0 ** plan.k
        if plan.shift_kind == "constant":
            x = cc.EvalAdd(ct, -plan.shift_const)
        else:
            a, b = plan.shift_w
            mean_term = cc.EvalMult(self._rowsum(ct), a / plan.n)
            sink_mask = np.zeros(self.slots); sink_mask[:self.R] = b      # position j = 0 of every row
            sink = cc.EvalMult(ct, self._pt(sink_mask, ct))
            for t in self.steps:                                          # broadcast along the row
                sink = cc.EvalAdd(sink, cc.EvalRotate(sink, -t))
            x = cc.EvalSub(cc.EvalSub(ct, mean_term), sink)
            x = cc.EvalAdd(x, -plan.shift_const)
        y = cc.EvalChebyshevFunction(lambda t: math.exp(t / s), x, *plan.exp_dom, deg["exp"])
        if plan.k >= 1:
            inv = cc.EvalChebyshevFunction(lambda t: 1 / t, self._rowsum(y), *plan.first_dom, deg["first"])
            y = cc.EvalMult(y, inv)
            for _ in range(plan.k):
                y2 = cc.EvalMult(y, y)
                inv = cc.EvalChebyshevFunction(lambda t: 1 / t, self._rowsum(y2), *plan.round_dom, deg["round"])
                y = cc.EvalMult(y2, inv)
        inv = cc.EvalChebyshevFunction(lambda t: 1 / t, self._rowsum(y), *plan.final_dom, deg["final"])
        return cc.EvalMult(y, inv)

    def softmax(self, rows: np.ndarray):
        ct = self.encrypt(rows)
        t0 = time.perf_counter()
        out = self.run(ct)
        secs = time.perf_counter() - t0
        return self.decrypt(out, rows.shape[0]), out.GetLevel(), secs


def precision_bits(probs, ref):
    err = np.abs(probs - ref).max(1)
    return -np.log2(np.maximum(err, 1e-300))
