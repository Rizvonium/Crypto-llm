"""Experiment: does the gap rule + a fitted zero-depth shift make max-free homomorphic softmax
shallower on REAL attention rows, at equal precision?

For each model and layer, attention-score rows at query position n-1 (context n = 128, every
head) are split by story into calibration and held-out sets. Three strategies are calibrated:
    current  constant shift (per layer), k from the full width M      (current practice)
    g-rule   constant shift,             k from the gap G only
    fitted   fitted linear shift (row mean, sink score, 1) + margin, k from G   (ours)
and evaluated (1) by exact emulation of the polynomial pipeline on all held-out rows, and
(2) under real CKKS encryption (OpenFHE, 128-bit security) on the widest layer of each model.

    python -m cryptollm.fhe.tinystories.nas_experiment            # emulation + encrypted runs
    python -m cryptollm.fhe.tinystories.nas_experiment --no-fhe   # emulation only (fast)
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import pathlib
import resource

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..nas_softmax import FheSoftmax, emulate, exact_softmax, make_plan, precision_bits
from .quality import load_stories

STRATEGIES = {"current": ("constant", "M"), "g-rule": ("constant", "G"), "fitted": ("fitted", "G")}


@torch.no_grad()
def score_rows(name: str, texts: list[str], n: int = 128) -> dict[int, np.ndarray]:
    """Attention-score rows (all heads) at query position n-1 of every text with >= n tokens."""
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager").eval()
    gpt2 = "gpt2" in name
    cfg = model.config
    L = cfg.n_layer if gpt2 else cfg.num_layers
    nh = cfg.n_head if gpt2 else cfg.num_heads
    d = cfg.n_embd if gpt2 else cfg.hidden_size
    dh = d // nh
    scale = 1 / math.sqrt(dh) if gpt2 else 1.0
    caps: dict = {}
    for i, blk in enumerate(model.transformer.h):
        if gpt2:
            blk.attn.c_attn.register_forward_hook(lambda m, inp, out, i=i: caps.__setitem__(i, out))
        else:
            a = blk.attn.attention
            a.q_proj.register_forward_hook(lambda m, inp, out, i=i: caps.__setitem__(("q", i), out))
            a.k_proj.register_forward_hook(lambda m, inp, out, i=i: caps.__setitem__(("k", i), out))
    rows: dict[int, list] = {i: [] for i in range(L)}
    for t in texts:
        ids = tok(t, return_tensors="pt").input_ids[:, :n]
        if ids.shape[1] < n:
            continue
        model(ids)
        for i in range(L):
            if gpt2:
                q, k = caps[i][0].split(d, dim=-1)[:2]
            else:
                q, k = caps[("q", i)][0], caps[("k", i)][0]
            q = q.view(n, nh, dh).transpose(0, 1)[:, -1]          # last query, every head
            k = k.view(n, nh, dh).transpose(0, 1)
            s = torch.einsum("hd,hjd->hj", q, k) * scale          # (heads, n)
            rows[i].append(s.double().numpy())
    return {i: np.concatenate(r) for i, r in rows.items()}


def emulate_layer(cal, test):
    out = {}
    ref = exact_softmax(test)
    for tag, (kind, rule) in STRATEGIES.items():
        plan = make_plan(cal, kind, rule)
        y, ood = emulate(plan, test)
        bits = precision_bits(y, ref)
        ok = ~ood
        out[tag] = {"plan": plan, "k": plan.k, "depth": plan.depth(), "bits_median": float(np.median(bits[ok])),
                    "bits_min": float(bits[ok].min()) if ok.any() else float("nan"), "ood": float(ood.mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-fhe", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--scale-bits", type=int, default=35, help="CKKS scaling-modulus bits for the encrypted runs")
    args = ap.parse_args()

    texts = load_stories("validation", 500)
    results = {}
    for name in ["roneneldan/TinyStories-1M", "openai-community/gpt2"]:
        R = score_rows(name, texts)
        n_rows = len(next(iter(R.values())))
        nh = 16 if "TinyStories" in name else 12
        n_texts = n_rows // nh
        cut = int(n_texts * 0.6) * nh                      # split by story, not by row
        print(f"\n## {name}: {n_texts} stories with >= 128 tokens -> {cut} calibration rows, {n_rows - cut} held-out rows (n = 128)\n")
        print("| layer | current: k / depth | g-rule: k / depth | **fitted: k / depth** | precision bits (median, min) current / fitted | out-of-domain rows current / fitted |")
        print("|---|---|---|---|---|---|")
        per_layer = {}
        tot = {t: 0 for t in STRATEGIES}
        for i, rows in R.items():
            res = emulate_layer(rows[:cut], rows[cut:])
            per_layer[i] = res
            for t in STRATEGIES:
                tot[t] += res[t]["depth"]
            c, g, f = res["current"], res["g-rule"], res["fitted"]
            print(f"| {i} | {c['k']} / {c['depth']} | {g['k']} / {g['depth']} | **{f['k']} / {f['depth']}** | "
                  f"{c['bits_median']:.1f}, {c['bits_min']:.1f} / {f['bits_median']:.1f}, {f['bits_min']:.1f} | "
                  f"{100 * c['ood']:.2f}% / {100 * f['ood']:.2f}% |")
        print(f"| **total depth** | {tot['current']} | {tot['g-rule']} | **{tot['fitted']}** "
              f"({100 * (1 - tot['fitted'] / tot['current']):.0f}% less) | | |")
        results[name] = {"total_depth": tot, "layers": {i: {t: {k: v for k, v in r.items() if k != "plan"} for t, r in res.items()} for i, res in per_layer.items()}}

        if not args.no_fhe:
            widest = max(per_layer, key=lambda i: per_layer[i]["current"]["depth"])
            rows = R[widest]
            test = rows[cut:cut + 8]
            ref = exact_softmax(test)
            print(f"\nEncrypted check (OpenFHE CKKS, 128-bit security, {args.scale_bits}-bit scale), layer {widest}, "
                  f"8 held-out rows packed in one ciphertext:\n")
            print("| strategy | k | planned depth | levels used | ring dimension | precision bits (min) | max abs diff vs emulation | server time | peak RAM (process) |")
            print("|---|---|---|---|---|---|---|---|---|")
            fres = {}
            for tag in ("fitted", "g-rule", "current"):
                plan = per_layer[widest][tag]["plan"]
                fhe = FheSoftmax(plan, rows_per_ct=8, scale_bits=args.scale_bits)
                y, lvl, secs = fhe.softmax(test)
                emu = emulate(plan, test)[0]
                bits = precision_bits(y, ref).min()
                print(f"| {tag} | {plan.k} | {plan.depth()} | {lvl} | 2^{int(math.log2(fhe.cc.GetRingDimension()))} | "
                      f"{bits:.1f} | {np.abs(y - emu).max():.1e} | {secs:.1f} s | "
                      f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.1f} GB |", flush=True)
                fres[tag] = {"k": plan.k, "depth": plan.depth(), "levels": lvl, "ring": fhe.cc.GetRingDimension(),
                             "bits_min": float(bits), "seconds": secs}
                del fhe
                gc.collect()
            results[name]["fhe"] = {"layer": widest, **fres}
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1, default=float))


if __name__ == "__main__":
    main()
