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
    rows = {i: {"m": [], "mean": [], "sink": [], "n": [], "min": []} for i in range(L)}
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
                mn = s.masked_fill(~causal, float("inf")).min(-1).values    # row minimum
                for key, v in (("m", m), ("mean", mean), ("sink", sink), ("min", mn)):
                    rows[i][key].append(v[:, 1:].flatten().numpy())          # skip row 0 (n=1)
                rows[i]["n"].append(cnt[1:].repeat(nh).numpy())
    return {i: {k: np.concatenate(v) for k, v in r.items()} for i, r in rows.items()}

def rounds(G, n):  # normalize-and-square rounds needed for gap G at context n (Cho et al. k ~ log2 G - log2 ln n)
    return max(0, math.ceil(math.log2(max(G, 1e-9)) - math.log2(math.log(n))))

def compare(name, cal, test, n_ctx=128):
    """Total squaring rounds on held-out text under three rules:
    sota   : per-layer constant shift (99.9th pct of calibration row maxima), k from the full
             calibrated width M = shift - row min (as in normalize-and-square deployments)
    g_rule : same constant shift, k from the gap G = shift - row max only
    fitted : fitted linear shift w . (mean score, sink score, 1) + calibrated margin, k from G"""
    Rc, Rt = traces(name, cal), traces(name, test)
    tot = {"sota": 0, "g_rule": 0, "fitted": 0}
    under = []
    for i in Rc:
        c = np.percentile(Rc[i]["m"], 99.9)
        M = np.percentile(c - Rt[i]["min"], 99.9)
        G_const = np.percentile(np.maximum(c - Rt[i]["m"], 0), 99.9)
        feats = lambda r: np.stack([r["mean"], r["sink"], np.ones_like(r["mean"])], 1)
        w, *_ = np.linalg.lstsq(feats(Rc[i]), Rc[i]["m"], rcond=None)
        margin = np.percentile(Rc[i]["m"] - feats(Rc[i]) @ w, 99.9)
        g_fit = feats(Rt[i]) @ w + margin - Rt[i]["m"]
        under.append(np.mean(g_fit < 0))
        tot["sota"] += rounds(M, n_ctx)
        tot["g_rule"] += rounds(G_const, n_ctx)
        tot["fitted"] += rounds(np.percentile(np.maximum(g_fit, 0), 99.9), n_ctx)
    return tot, float(np.mean(under))


def main() -> None:
    texts = load_stories("validation", 100)
    cal, test = texts[:60], texts[60:]
    print("Total normalize-and-square rounds over all layers, held-out stories, n = 128")
    print("| model | constant shift, k from full range (current practice) | constant shift, k from gap G | fitted linear shift, k from G | rows where fitted shift < true max |")
    print("|---|---|---|---|---|")
    for name in ["roneneldan/TinyStories-1M", "openai-community/gpt2"]:
        tot, under = compare(name, cal, test)
        print(f"| {name} | {tot['sota']} | {tot['g_rule']} | {tot['fitted']} | {100 * under:.2f}% |")


if __name__ == "__main__":
    main()
