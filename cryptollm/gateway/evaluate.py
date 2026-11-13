"""Measure how much a gateway configuration still leaks, and how much it over-redacts.

A labelled sensitive value counts as **hidden** only if every one of its letters/digits,
in every place it occurs, falls inside a redaction span; otherwise it is **exposed** (a
partially masked "14 Maple Street, Apt 3" with "Apt 3" visible counts as exposed).
**Over-redaction** is the share of non-sensitive letters/digits that were masked anyway, a
proxy for lost answer quality.

    python -m cryptollm.gateway.evaluate                              # realistic set, regex only
    python -m cryptollm.gateway.evaluate --configs regex,regex+gliner,regex+presidio,all
    python -m cryptollm.gateway.evaluate --dataset ai4privacy --n 300  # downloads a public sample

Datasets:
* ``realistic``  - 40 hand-written chat prompts (cryptollm/gateway/eval/), diverse names,
                   lowercase names, IDs from several countries, API keys, 5 PII-free controls.
* ``pupa``       - PUPA (Li et al., NAACL 2025): 901 *real* user prompts from WildChat with
                   annotated privacy units (names, places, orgs, ...); no categories or
                   offsets, so every unit is matched case-insensitively (MIT licence).
* ``ai4privacy`` - English rows of the public ai4privacy/pii-masking-400k validation split
                   (synthetic, template-generated; downloaded on demand, not redistributed).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import os
import pathlib
import time
import urllib.request
from collections import defaultdict

from .detectors import DEFAULT_DETECTORS
from .pseudonymizer import Pseudonymizer

HERE = pathlib.Path(__file__).resolve().parent
REALISTIC = HERE / "eval" / "realistic_prompts.jsonl"
DIRECT = ["PERSON", "EMAIL", "PHONE", "ADDRESS", "ID", "SECRET", "IP", "USERNAME", "PII_UNIT"]
QUASI = ["ORG", "LOCATION", "DOB", "OTHER"]

AI4P_MAP = {
    "GIVENNAME": "PERSON", "SURNAME": "PERSON", "EMAIL": "EMAIL", "TELEPHONENUM": "PHONE",
    "STREET": "ADDRESS", "BUILDINGNUM": "ADDRESS", "ZIPCODE": "ADDRESS", "CITY": "LOCATION",
    "IDCARDNUM": "ID", "DRIVERLICENSENUM": "ID", "PASSPORTNUM": "ID", "SOCIALNUM": "ID",
    "TAXNUM": "ID", "CREDITCARDNUMBER": "ID", "ACCOUNTNUM": "ID", "USERNAME": "USERNAME",
    "PASSWORD": "SECRET", "DATEOFBIRTH": "DOB",
}


# ----------------------------------------------------------------------------- data
def load_realistic() -> list[dict]:
    rows = []
    for line in REALISTIC.read_text(encoding="utf-8").splitlines():
        ex = json.loads(line)
        spans = []
        for item in ex["sensitive"]:
            i = ex["text"].find(item["value"])
            while i != -1:
                spans.append((i, i + len(item["value"]), item["category"], item["value"]))
                i = ex["text"].find(item["value"], i + 1)
        rows.append({"text": ex["text"], "spans": spans})
    return rows


def load_ai4privacy(n: int, cache: pathlib.Path) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    f = cache / f"ai4privacy_400k_val_en_{n}.jsonl"
    if not f.exists():
        rows, offset = [], 0
        while len(rows) < n and offset < 20000:
            url = ("https://datasets-server.huggingface.co/rows?dataset=ai4privacy/pii-masking-400k"
                   f"&config=default&split=validation&offset={offset}&length=100")
            with urllib.request.urlopen(url, timeout=60) as r:
                batch = json.load(r)["rows"]
            if not batch:
                break
            for b in batch:
                row = b["row"]
                if row.get("language") == "en":
                    rows.append({"text": row["source_text"], "mask": row["privacy_mask"]})
            offset += 100
        f.write_text("".join(json.dumps(r) + "\n" for r in rows[:n]))
    out = []
    for line in f.read_text().splitlines():
        r = json.loads(line)
        spans = [(m["start"], m["end"], AI4P_MAP.get(m["label"], "OTHER"), m["value"]) for m in r["mask"]]
        out.append({"text": r["text"], "spans": spans})
    return out


def load_pupa(cache: pathlib.Path, limit: int | None = None) -> list[dict]:
    cache.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ("PUPA_New.csv", "PUPA_TNB.csv"):
        f = cache / name
        if not f.exists():
            urllib.request.urlretrieve(
                f"https://huggingface.co/datasets/Columbia-NLP/PUPA/resolve/main/{name}", f)
        with open(f, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                text = r["user_query"]
                units = {u.strip() for u in r["pii_units"].split("||") if len(u.strip()) >= 2}
                spans = [(m.start(), m.end(), "PII_UNIT", u)
                         for u in units for m in re.finditer(re.escape(u), text, re.I)]
                rows.append({"text": text, "spans": spans})
    return rows[:limit] if limit else rows


# ----------------------------------------------------------------------------- configs
def build_configs(names: list[str]) -> dict[str, list]:
    cache: dict[str, object] = {}

    def gliner(quasi=False):
        key = f"gliner{quasi}"
        if key not in cache:
            from .detectors import GlinerDetector
            cache[key] = GlinerDetector(include_quasi=quasi)
        return cache[key]

    def presidio():
        if "presidio" not in cache:
            from .detectors import PresidioDetector
            cache["presidio"] = PresidioDetector()
        return cache["presidio"]

    table = {
        "regex": lambda: list(DEFAULT_DETECTORS),
        "regex+presidio": lambda: [*DEFAULT_DETECTORS, presidio()],
        "regex+gliner": lambda: [*DEFAULT_DETECTORS, gliner()],
        "all": lambda: [*DEFAULT_DETECTORS, gliner(), presidio()],
        "all+quasi": lambda: [*DEFAULT_DETECTORS, gliner(quasi=True), presidio()],
    }
    return {n: table[n]() for n in names}


# ----------------------------------------------------------------------------- scoring
def _alnum_positions(text: str, start: int, end: int) -> set[int]:
    return {i for i in range(start, end) if text[i].isalnum()}


def score(rows: list[dict], detectors: list) -> dict:
    per_cat = defaultdict(lambda: [0, 0])  # category -> [hidden, total]
    misses: list[tuple[str, str]] = []
    over, clean_chars, secs = 0, 0, 0.0
    control_spans, controls = 0, 0
    prompts_with_direct, prompts_protected = 0, 0
    for row in rows:
        text = row["text"]
        p = Pseudonymizer(detectors)
        t0 = time.perf_counter()
        spans = p.find_spans(text)
        secs += time.perf_counter() - t0
        masked = set()
        for s in spans:
            masked |= _alnum_positions(text, s.start, s.end)
        sensitive = set()
        by_value = defaultdict(set)
        for (a, b, cat, val) in row["spans"]:
            pos = _alnum_positions(text, a, b)
            sensitive |= pos
            by_value[(cat, val)] |= pos
        exposed_here = False
        for (cat, val), pos in by_value.items():
            if not pos:
                continue
            ok = pos <= masked
            per_cat[cat][0] += ok
            per_cat[cat][1] += 1
            if not ok and cat in DIRECT:
                misses.append((cat, val))
                exposed_here = True
        if any(cat in DIRECT for (cat, _v) in by_value):
            prompts_with_direct += 1
            prompts_protected += not exposed_here
        non_sensitive = _alnum_positions(text, 0, len(text)) - sensitive
        over += len(masked & non_sensitive)
        clean_chars += len(non_sensitive)
        if not row["spans"]:
            controls += 1
            control_spans += len(spans)

    def agg(cats):
        h = sum(per_cat[c][0] for c in cats if c in per_cat)
        t = sum(per_cat[c][1] for c in cats if c in per_cat)
        return h, t

    return {
        "per_category": {c: tuple(v) for c, v in per_cat.items()},
        "direct": agg(DIRECT),
        "quasi": agg(QUASI),
        "over_redaction": over / max(clean_chars, 1),
        "control_false_positives": (control_spans, controls),
        "ms_per_prompt": 1000 * secs / max(len(rows), 1),
        "prompts_protected": (prompts_protected, prompts_with_direct),
        "misses": misses,
    }


def _pct(h_t) -> str:
    h, t = h_t
    return f"{100 * h / t:5.1f}% ({h}/{t})" if t else "   -   "


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="realistic", choices=["realistic", "pupa", "ai4privacy"])
    ap.add_argument("--limit", type=int, help="evaluate only the first N prompts (pupa)")
    ap.add_argument("--configs", default="regex")
    ap.add_argument("--n", type=int, default=300, help="rows for ai4privacy")
    ap.add_argument("--cache", default=os.path.expanduser("~/.cache/cryptollm"))
    ap.add_argument("--show-misses", action="store_true")
    ap.add_argument("--json", help="write raw results to this file")
    args = ap.parse_args(argv)

    cache = pathlib.Path(args.cache)
    rows = {"realistic": load_realistic,
            "pupa": lambda: load_pupa(cache, args.limit),
            "ai4privacy": lambda: load_ai4privacy(args.n, cache)}[args.dataset]()
    configs = build_configs(args.configs.split(","))
    results = {name: score(rows, dets) for name, dets in configs.items()}

    cats = [c for c in DIRECT + QUASI if any(c in r["per_category"] for r in results.values())]
    print(f"\nDataset: {args.dataset} ({len(rows)} prompts). Share of sensitive values fully hidden:\n")
    header = "| category | " + " | ".join(results) + " |"
    print(header + "\n|" + "---|" * (len(results) + 1))
    for c in cats:
        print(f"| {c} | " + " | ".join(_pct(r["per_category"].get(c, (0, 0))) for r in results.values()) + " |")
    print("| **direct identifiers** | " + " | ".join(_pct(r["direct"]) for r in results.values()) + " |")
    print("| prompts with nothing exposed | " + " | ".join(_pct(r["prompts_protected"]) for r in results.values()) + " |")
    print("| quasi-identifiers | " + " | ".join(_pct(r["quasi"]) for r in results.values()) + " |")
    print("| over-redaction (clean chars masked) | "
          + " | ".join(f"{100 * r['over_redaction']:.1f}%" for r in results.values()) + " |")
    if args.dataset == "realistic":
        print("| spans flagged in PII-free prompts | "
              + " | ".join(f"{r['control_false_positives'][0]} in {r['control_false_positives'][1]}"
                           for r in results.values()) + " |")
    print("| detection time per prompt (CPU) | " + " | ".join(f"{r['ms_per_prompt']:.0f} ms" for r in results.values()) + " |")
    if args.show_misses:
        for name, r in results.items():
            print(f"\nexposed direct identifiers with {name}: {r['misses']}")
    if args.json:
        pathlib.Path(args.json).write_text(json.dumps(results, indent=1, default=list))


if __name__ == "__main__":
    main()
