"""Step 0 of the speed-up project: how hard is TinyStories-1M to run under CKKS?

FHE can only add and multiply, so every non-linear function must become a polynomial that
is accurate over the *range of values it will see*. This script runs the real model in
plaintext on sample stories, records those ranges per layer, and finds the smallest
Chebyshev degree that approximates each function well enough. Degree -> multiplicative
depth -> bootstraps -> time: this is the baseline cost model the speed-ups must beat.

    python -m cryptollm.fhe.tinystories.calibrate
"""

from __future__ import annotations

import math

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "roneneldan/TinyStories-1M"
PROMPTS = [
    "Once upon a time, there was a little girl named Lily.",
    "Tom had a red ball. One day, the ball rolled into the river.",
    "The sun was shining and the birds were singing in the park.",
    "Sam and his dog went to the beach to build a big sandcastle.",
    "Mom said, \"Let's bake a cake for Grandma's birthday!\"",
    "There was a small cat who was afraid of the dark.",
    "Ben found a shiny key under a big rock in the garden.",
    "Anna wanted to fly like a bird, so she made wings from paper.",
]
# OpenFHE Chebyshev degrees and their multiplicative depth
DEGREE_DEPTH = {5: 3, 13: 4, 27: 5, 59: 6, 119: 7, 247: 8, 495: 9, 1007: 10, 2031: 11}


def min_degree(f, a, b, rel_tol=None, abs_tol=None, ref_scale=1.0):
    """Smallest OpenFHE-friendly Chebyshev degree meeting the tolerance on [a, b]."""
    xs = np.linspace(a, b, 4001)
    for deg, depth in DEGREE_DEPTH.items():
        cheb = np.polynomial.chebyshev.Chebyshev.interpolate(f, deg, domain=[a, b])
        err = np.abs(cheb(xs) - f(xs))
        ok = (err <= rel_tol * np.abs(f(xs))).all() if rel_tol else (err.max() <= abs_tol * ref_scale)
        if ok:
            return deg, depth
    return None, None


def collect():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL).eval()
    cfg = model.config
    L, d, nh = cfg.num_layers, cfg.hidden_size, cfg.num_heads
    stats = {k: [[] for _ in range(L)] for k in ("ln1_var", "ln2_var", "scores", "row_spread", "gelu_in", "resid")}
    caps = {}

    def save(name):
        def hook(_m, _inp, out):
            caps[name] = out.detach()
        return hook

    def save_in(name):
        def hook(_m, inp):
            caps[name] = inp[0].detach()
        return hook

    for i, blk in enumerate(model.transformer.h):
        blk.ln_1.register_forward_pre_hook(save_in(f"ln1_{i}"))
        blk.ln_2.register_forward_pre_hook(save_in(f"ln2_{i}"))
        blk.attn.attention.q_proj.register_forward_hook(save(f"q_{i}"))
        blk.attn.attention.k_proj.register_forward_hook(save(f"k_{i}"))
        blk.mlp.c_fc.register_forward_hook(save(f"fc_{i}"))

    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            ids = model.generate(ids, max_new_tokens=48, do_sample=False, pad_token_id=tok.eos_token_id)
            model(ids)
            T = ids.shape[1]
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
            for i in range(L):
                for key, name in (("ln1_var", "ln1"), ("ln2_var", "ln2")):
                    x = caps[f"{name}_{i}"][0]
                    stats[key][i].append(x.var(dim=-1, unbiased=False).numpy())
                q = caps[f"q_{i}"][0].view(T, nh, d // nh).transpose(0, 1)
                k = caps[f"k_{i}"][0].view(T, nh, d // nh).transpose(0, 1)
                s = (q @ k.transpose(-1, -2))  # GPT-Neo does not scale by 1/sqrt(d_head)
                vals = s[:, causal]
                stats["scores"][i].append(vals.numpy().ravel())
                masked = s.masked_fill(~causal, float("nan"))
                spread = np.nanmax(masked.numpy(), -1) - np.nanmin(masked.numpy(), -1)
                stats["row_spread"][i].append(spread.ravel())
                stats["gelu_in"][i].append(caps[f"fc_{i}"][0].numpy().ravel())
                stats["resid"][i].append(np.abs(caps[f"ln1_{i}"][0].numpy()).max(-1))
    return {k: [np.concatenate(v) for v in vv] for k, vv in stats.items()}, cfg


def main():
    stats, cfg = collect()
    gelu = lambda x: 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
    print(f"TinyStories-1M: {cfg.num_layers} layers, d_model {cfg.hidden_size}, {cfg.num_heads} heads, "
          f"{len(PROMPTS)} stories x ~60 tokens\n")
    print("| layer | LN var range | inv-sqrt degree (depth) | attention scores | max row spread | exp degree (depth) | GELU input | GELU degree (depth) |")
    print("|---|---|---|---|---|---|---|---|")
    total_depth = []
    for i in range(cfg.num_layers):
        v = np.concatenate([stats["ln1_var"][i], stats["ln2_var"][i]])
        va, vb = v.min() * 0.8, v.max() * 1.2
        inv_deg, inv_dep = min_degree(lambda t: 1 / np.sqrt(t + 1e-5), va, vb, rel_tol=1e-3)
        sc = stats["scores"][i]; spread = stats["row_spread"][i].max()
        # softmax after subtracting a per-row bound: exp needed on [-spread, 0]
        ea = -spread * 1.1
        exp_deg, exp_dep = min_degree(np.exp, ea, 0.0, abs_tol=1e-3)
        g = stats["gelu_in"][i]; ga, gb = g.min() * 1.1, g.max() * 1.1
        gel_deg, gel_dep = min_degree(gelu, ga, gb, abs_tol=1e-3, ref_scale=max(abs(ga), abs(gb)))
        print(f"| {i} | {v.min():.3g} – {v.max():.3g} ({v.max()/v.min():.0f}x) | {inv_deg} ({inv_dep}) | "
              f"{sc.min():.1f} – {sc.max():.1f} | {spread:.1f} | {exp_deg} ({exp_dep}) | "
              f"{g.min():.1f} – {g.max():.1f} | {gel_deg} ({gel_dep}) |")
        if None not in (inv_dep, exp_dep, gel_dep):
            # per layer: 2 LayerNorms (square + inv-sqrt + scale), QKV, scores, mask, exp, mask,
            # 1/x (depth ~6), p, p*V, out-proj, fc, GELU, proj
            total_depth.append(2 * (1 + inv_dep + 1) + 1 + 1 + 1 + exp_dep + 1 + 6 + 1 + 1 + 1 + 1 + gel_dep + 1)
    if total_depth:
        print(f"\nestimated multiplicative depth per layer: {min(total_depth)}–{max(total_depth)} "
              f"(our 128-bit bootstrapping parameters leave ~5 usable levels per bootstrap)")


if __name__ == "__main__":
    main()
