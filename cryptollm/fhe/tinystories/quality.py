"""Quality benchmark for the speed-up project: perplexity on held-out TinyStories text.

Any change that makes a model cheaper under FHE (bounded attention, polynomial
activations, softmax replacements, ...) must be paid for in quality; this measures it.

    python -m cryptollm.fhe.tinystories.quality                 # TinyStories-1M baseline
    python -m cryptollm.fhe.tinystories.quality --model roneneldan/TinyStories-8M
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import time
import urllib.error
import urllib.request

import torch

CACHE = pathlib.Path(os.path.expanduser("~/.cache/cryptollm"))


def load_stories(split: str = "validation", n: int = 500, offset: int = 0) -> list[str]:
    """First n stories of a TinyStories split, fetched from the HF datasets server and cached."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"tinystories_{split}_{offset}_{n}.json"
    if not f.exists():
        texts: list[str] = []
        while len(texts) < n:
            length = min(100, n - len(texts))
            url = ("https://datasets-server.huggingface.co/rows?dataset=roneneldan/TinyStories"
                   f"&config=default&split={split}&offset={offset + len(texts)}&length={length}")
            for attempt in range(6):                      # the datasets server is sometimes 502
                try:
                    with urllib.request.urlopen(url, timeout=60) as r:
                        rows = json.load(r)["rows"]
                    break
                except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
                    if attempt == 5:
                        raise
                    time.sleep(2 ** attempt)
            if not rows:
                break
            texts += [row["row"]["text"] for row in rows]
        f.write_text(json.dumps(texts))
    return json.loads(f.read_text())


@torch.no_grad()
def perplexity(model, tok, texts: list[str], max_len: int = 256, batch: int = 16) -> float:
    """Token-level perplexity (exp of mean next-token NLL), each story truncated to max_len."""
    model.eval()
    nll, count = 0.0, 0
    for i in range(0, len(texts), batch):
        enc = tok(texts[i:i + batch], return_tensors="pt", padding=True, truncation=True, max_length=max_len)
        logits = model(input_ids=enc.input_ids, attention_mask=enc.attention_mask).logits[:, :-1]
        target = enc.input_ids[:, 1:]
        mask = enc.attention_mask[:, 1:].bool()
        lp = torch.log_softmax(logits.float(), -1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        nll -= lp[mask].sum().item()
        count += mask.sum().item()
    return math.exp(nll / count)


def load_model(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return AutoModelForCausalLM.from_pretrained(name).eval(), tok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="roneneldan/TinyStories-1M")
    ap.add_argument("--n", type=int, default=500, help="validation stories")
    args = ap.parse_args()
    model, tok = load_model(args.model)
    texts = load_stories("validation", args.n)
    print(f"{args.model}: perplexity {perplexity(model, tok, texts):.3f} on {len(texts)} held-out stories")


if __name__ == "__main__":
    main()
