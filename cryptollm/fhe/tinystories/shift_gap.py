"""Preliminary evidence: how far is each zero-depth softmax shift from the true row max?

Max-free homomorphic softmax (normalize-and-square, Cho et al. CCS'24) needs every row shifted
by an upper bound of its max; the number of squaring rounds grows with log2 of the gap G between
that bound and the true max. Softmax is shift-invariant, so any *linear* function of the keys
gives an exact, zero-multiplicative-depth shift. This script compares three:
  constant   one calibrated bound per layer (common practice)
  mean-key   q . mean_j(k_j)   (the mean score of the row; additions only)
  sink       q . k_0           (score of the first token, an attention sink in many LMs)
and reports the 99.9th-percentile G and the implied rounds k ~ ceil(log2 G - log2 ln n).

    python -m cryptollm.fhe.tinystories.shift_gap
"""
import math

import numpy as np
import torch

from .quality import load_stories
from transformers import AutoModelForCausalLM, AutoTokenizer

def traces(name, texts, max_len=128):
    tok = AutoTokenizer.from_pretrained(name); model = AutoModelForCausalLM.from_pretrained(name, attn_implementation="eager").eval()
    caps = {}
    if "gpt2" in name:
        L, nh = model.config.n_layer, model.config.n_head; d = model.config.n_embd; scale = 1 / math.sqrt(d // nh)
        for i, blk in enumerate(model.transformer.h):
            blk.attn.c_attn.register_forward_hook(lambda m, inp, out, i=i: caps.__setitem__(i, out.detach()))
        split = lambda o: o.split(d, dim=-1)[:2]
    else:
        L, nh = model.config.num_layers, model.config.num_heads; d = model.config.hidden_size; scale = 1.0
        for i, blk in enumerate(model.transformer.h):
            a = blk.attn.attention
            a.q_proj.register_forward_hook(lambda m, inp, out, i=i: caps.__setitem__(("q", i), out.detach()))
            a.k_proj.register_forward_hook(lambda m, inp, out, i=i: caps.__setitem__(("k", i), out.detach()))
    rows = {i: {"m": [], "mean": [], "sink": [], "n": []} for i in range(L)}
    with torch.no_grad():
        for t in texts:
            ids = tok(t, return_tensors="pt", truncation=True, max_length=max_len).input_ids
            model(ids); T = ids.shape[1]
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
            for i in range(L):
                if "gpt2" in name:
                    q, k = split(caps[i][0])
                else:
                    q, k = caps[("q", i)][0], caps[("k", i)][0]
                q = q.view(T, nh, d // nh).transpose(0, 1); k = k.view(T, nh, d // nh).transpose(0, 1)
                s = (q @ k.transpose(-1, -2)) * scale                      # (nh, T, T)
                sm = s.masked_fill(~causal, float("-inf"))
                m = sm.max(-1).values                                       # row max
                cnt = causal.sum(-1).float()                                # n per row
                mean = (s * causal).sum(-1) / cnt                           # = q . mean(k_j)  (exact, linear)
                sink = s[:, :, 0]                                           # score of first token
                for key, v in (("m", m), ("mean", mean), ("sink", sink)):
                    rows[i][key].append(v[:, 1:].flatten().numpy())          # skip row 0 (n=1)
                rows[i]["n"].append(cnt[1:].repeat(nh).numpy())
    return {i: {k: np.concatenate(v) for k, v in r.items()} for i, r in rows.items()}

def rounds(G, n):  # normalize-and-square rounds needed for gap G at context n (Cho et al. k ~ log2 G - log2 ln n)
    return max(0, math.ceil(math.log2(max(G, 1e-9)) - math.log2(math.log(n))))

def main() -> None:
    texts = load_stories("validation", 100)
    for name in ["roneneldan/TinyStories-1M", "openai-community/gpt2"]:
        R = traces(name, texts)
        print(f"\n{name}: G = (calibrated bound on max - reference) at the 99.9th percentile, and squaring rounds k at n=128")
        print("| layer | constant shift: G | k | mean-key shift: G | k | sink-token shift: G | k |")
        print("|---|---|---|---|---|---|---|")
        tot = {"const": 0, "mean": 0, "sink": 0}
        for i, r in R.items():
            m = r["m"]
            g_const = np.percentile(m, 99.9) - np.percentile(m, 0.1)     # constant = calibrated max; worst row max far below
            g_mean = np.percentile(m - r["mean"], 99.9)                   # bound on (max - mean) ; shift = mean + bound
            g_sink = np.percentile(m - r["sink"], 99.9) - min(0, np.percentile(m - r["sink"], 0.1))
            ks = {k: rounds(g, 128) for k, g in (("const", g_const), ("mean", g_mean), ("sink", g_sink))}
            for k in tot: tot[k] += ks[k]
            print(f"| {i} | {g_const:.1f} | {ks['const']} | {g_mean:.1f} | {ks['mean']} | {g_sink:.1f} | {ks['sink']} |")
        print(f"| total rounds | | {tot['const']} | | {tot['mean']} | | {tot['sink']} |")


if __name__ == "__main__":
    main()
