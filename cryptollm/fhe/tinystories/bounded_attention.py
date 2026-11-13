"""Pilot: retrofit a pre-trained TinyStories model with *bounded* attention so that softmax
can be evaluated under CKKS without an encrypted max.

Why it works: if every attention score s lies in [-B, B], then each exp(s) lies in
[e^-B, e^B], so the row mean Z/n does too, for ANY row. The context length n is public, so
the server can divide by it for free. Softmax then needs exactly
    exp(.)  on [-B, B]           (fixed Chebyshev polynomial)
    1/x     on [e^-B, e^B]       (fixed Chebyshev polynomial, range ratio e^{2B})
and no comparison or max at all. The price: a bounded score range limits how sharply a head
can focus. This script measures that trade-off.

Modes (patched into GPTNeoSelfAttention._attn):
  exact    original softmax (unbounded scores)
  clamped  softmax(clamp(s, -B, B))           <- used for fine-tuning
  fhe      the polynomial computation CKKS would run: clamp, Chebyshev exp, mean over the
           public n, Chebyshev 1/x, multiply (exactly what the encrypted server computes,
           minus ~1e-10 CKKS noise)

    python -m cryptollm.fhe.tinystories.bounded_attention --B 4 --steps 1500
"""

from __future__ import annotations

import argparse
import math
import time
import types

import numpy as np
import torch
from torch import nn

from .quality import load_model, load_stories, perplexity

DEGREE_DEPTH = {5: 3, 13: 4, 27: 5, 59: 6, 119: 7, 247: 8}


class Cheb:
    """Chebyshev interpolant of f on [a, b], evaluated in torch (float64) like CKKS would."""

    def __init__(self, f, a, b, degree):
        self.a, self.b, self.degree = a, b, degree
        self.c = torch.tensor(np.polynomial.chebyshev.Chebyshev.interpolate(f, degree, domain=[a, b]).coef)

    def __call__(self, x):
        t = (2 * x.double() - (self.a + self.b)) / (self.b - self.a)
        c = self.c
        b1 = torch.zeros_like(t); b2 = torch.zeros_like(t)
        for k in range(len(c) - 1, 0, -1):          # Clenshaw recurrence
            b1, b2 = 2 * t * b1 - b2 + c[k], b1
        return (t * b1 - b2 + c[0]).float()


def min_degree(f, a, b, rel_tol):
    xs = np.linspace(a, b, 4001)
    for deg, depth in DEGREE_DEPTH.items():
        p = np.polynomial.chebyshev.Chebyshev.interpolate(f, deg, domain=[a, b])
        if (np.abs(p(xs) - f(xs)) <= rel_tol * np.abs(f(xs))).all():
            return deg, depth
    return None, None


def patch(model, state):
    """Replace every layer's _attn with a version honouring state['mode'] and state['B']."""

    def _attn(self, query, key, value, attention_mask=None, head_mask=None):
        q, k = query.float(), key.float()
        s = q @ k.transpose(-1, -2)
        ql, kl = q.size(-2), k.size(-2)
        valid = self.bias[:, :, kl - ql:kl, :kl]
        if attention_mask is not None:
            valid = valid & (attention_mask[:, :, :, :kl] > -1)
        B, mode = state["B"], state["mode"]
        if state.get("stats") is not None:
            sv = s.detach()
            state["stats"]["absmax"] = max(state["stats"].get("absmax", 0.0), sv[valid.expand_as(sv)].abs().max().item())
            outside = ((sv.abs() > B) & valid).sum().item()
            state["stats"]["outside"] = state["stats"].get("outside", 0) + outside
            state["stats"]["total"] = state["stats"].get("total", 0) + valid.expand_as(sv).sum().item()
        if state.get("penalty_on"):
            over = torch.relu(s.abs() - B) * valid
            state["penalty"] = state.get("penalty", 0.0) + (over ** 2).sum() / valid.expand_as(s).sum()
        if mode == "exact":
            w = torch.softmax(torch.where(valid, s, torch.finfo(s.dtype).min), -1)
        elif mode == "clamped":
            w = torch.softmax(torch.where(valid, s.clamp(-B, B), torch.finfo(s.dtype).min), -1)
        else:  # "fhe": the fixed polynomial pipeline, no max
            e = state["exp"](s.clamp(-B, B)) * valid
            n = valid.sum(-1, keepdim=True).clamp(min=1)
            z = e.sum(-1, keepdim=True) / n                       # in [e^-B, e^B]
            w = e * state["inv"](z) / n
        w = self.attn_dropout(w.to(value.dtype))
        if head_mask is not None:
            w = w * head_mask
        return w @ value, w

    for blk in model.transformer.h:
        blk.attn.attention._attn = types.MethodType(_attn, blk.attn.attention)


def set_fhe(state, B, exp_degree=None, inv_degree=None, tol=1e-3):
    ed, edep = (exp_degree, None) if exp_degree else min_degree(np.exp, -B, B, tol)
    idg, idep = (inv_degree, None) if inv_degree else min_degree(lambda t: 1 / t, math.exp(-B), math.exp(B), tol)
    state.update(B=B, exp=Cheb(np.exp, -B, B, ed), inv=Cheb(lambda t: 1 / t, math.exp(-B) * 0.98, math.exp(B) * 1.02, idg))
    return {"exp_degree": ed, "exp_depth": edep or DEGREE_DEPTH.get(ed), "inv_degree": idg,
            "inv_depth": idep or DEGREE_DEPTH.get(idg)}


def score_stats(model, tok, texts, state, B):
    state.update(mode="exact", B=B, stats={})
    perplexity(model, tok, texts)
    st = state.pop("stats"); state["stats"] = None
    return st["absmax"], st["outside"] / st["total"]


def finetune(model, tok, state, B, steps, lam, lr, batch=16, max_len=128, log_every=250, train_mode="clamped"):
    texts = load_stories("train", 20000)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1.0, (i + 1) / 100) * 0.5 * (1 + math.cos(math.pi * i / steps)))
    model.train()
    rng = np.random.default_rng(0)
    state.update(mode=train_mode, B=B, penalty_on=lam > 0)
    t0 = time.time()
    for step in range(steps):
        idx = rng.integers(0, len(texts), batch)
        enc = tok([texts[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        state["penalty"] = 0.0
        out = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask,
                    labels=enc.input_ids.masked_fill(enc.attention_mask == 0, -100))
        loss = out.loss + lam * state["penalty"]
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        if (step + 1) % log_every == 0:
            pen = state["penalty"].item() if torch.is_tensor(state["penalty"]) else state["penalty"]
            print(f"  step {step + 1}/{steps}  lm loss {out.loss.item():.3f}  range penalty {pen:.4f}  ({time.time() - t0:.0f}s)", flush=True)
    state["penalty_on"] = False
    model.eval()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--B", type=float, default=4.0, help="score bound")
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lam", type=float, default=1.0, help="weight of the out-of-range penalty")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--n-eval", type=int, default=300)
    ap.add_argument("--control", action="store_true",
                    help="control run: fine-tune with the ordinary softmax and no range penalty")
    args = ap.parse_args()

    torch.manual_seed(0)
    model, tok = load_model(args.model)
    state = {"mode": "exact", "B": args.B, "stats": None}
    patch(model, state)
    val = load_stories("validation", args.n_eval)
    fhe = set_fhe(state, args.B)

    def report(tag):
        absmax, frac = score_stats(model, tok, val, state, args.B)
        state["mode"] = "exact"; ppl_exact = perplexity(model, tok, val)
        state["mode"] = "fhe"; ppl_fhe = perplexity(model, tok, val)
        print(f"{tag:34s} ppl exact {ppl_exact:8.3f} | ppl FHE-softmax(B={args.B:g}) {ppl_fhe:10.3f} | "
              f"max|score| {absmax:6.1f} | scores outside [-B,B] {100 * frac:5.2f}%", flush=True)

    print(f"FHE softmax with B={args.B:g}: exp degree {fhe['exp_degree']} (depth {fhe['exp_depth']}), "
          f"1/x degree {fhe['inv_degree']} (depth {fhe['inv_depth']}) on [e^-B, e^B]; no max needed")
    report("pre-trained model")
    if args.control:
        finetune(model, tok, state, args.B, args.steps, 0.0, args.lr, train_mode="exact")
        report(f"after control fine-tune ({args.steps} steps)")
    else:
        finetune(model, tok, state, args.B, args.steps, args.lam, args.lr)
        report(f"after bounded fine-tune ({args.steps} steps)")


if __name__ == "__main__":
    main()
