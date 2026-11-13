"""Privacy gateway CLI.

    python -m cryptollm.gateway "Draft a reply to Dr. Priya Rao (priya@uni.edu) ..."
    python -m cryptollm.gateway --provider ollama --model llama3.1 "..."
    python -m cryptollm.gateway --provider anthropic --model <model-id> --terms "Acme,Project Falcon" "..."
    python -m cryptollm.gateway --vault me.vault "..."      # consistent pseudonyms across runs

Default provider is ``dry-run``: nothing leaves the machine; you see exactly what would.
"""

from __future__ import annotations

import argparse
import getpass
import os
import sys

from .detectors import DEFAULT_DETECTORS, TermsDetector
from .providers import SYSTEM_HINT, Anthropic, DryRunProvider, OpenAICompatible
from .pseudonymizer import Pseudonymizer
from .vault import Vault


def build_provider(args):
    if args.provider == "dry-run":
        return DryRunProvider()
    if args.provider == "openai":
        return OpenAICompatible(args.model or "gpt-4o-mini", args.base_url or "https://api.openai.com/v1")
    if args.provider == "ollama":
        return OpenAICompatible(args.model or "llama3.1", args.base_url or "http://localhost:11434/v1", api_key="ollama")
    if args.provider == "anthropic":
        if not args.model:
            sys.exit("--model is required for the anthropic provider")
        return Anthropic(args.model)
    sys.exit(f"unknown provider {args.provider}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--provider", default="dry-run", choices=["dry-run", "openai", "ollama", "anthropic"])
    ap.add_argument("--model")
    ap.add_argument("--base-url", help="any OpenAI-compatible endpoint (local server, TEE service, ...)")
    ap.add_argument("--terms", default="", help="comma-separated names/terms to always hide")
    ap.add_argument("--ner", action="store_true", help="also use Presidio NER (optional dependency)")
    ap.add_argument("--vault", help="encrypted vault file for consistent pseudonyms across runs")
    args = ap.parse_args(argv)

    detectors = list(DEFAULT_DETECTORS)
    if args.terms:
        detectors.insert(0, TermsDetector([t.strip() for t in args.terms.split(",")], default_label="TERM"))
    if args.ner:
        from .detectors import PresidioDetector
        detectors.append(PresidioDetector())

    passphrase = None
    vault = Vault()
    if args.vault:
        passphrase = os.environ.get("CRYPTOLLM_VAULT_PASSPHRASE") or getpass.getpass("vault passphrase: ")
        if os.path.exists(args.vault):
            vault = Vault.load(args.vault, passphrase)

    p = Pseudonymizer(detectors, vault)
    safe = p.anonymize(args.prompt)
    print("== sent to provider ==\n" + safe)
    reply = build_provider(args).complete([{"role": "system", "content": SYSTEM_HINT},
                                           {"role": "user", "content": safe}])
    print("\n== provider saw/returned ==\n" + reply)
    print("\n== what you read (restored locally) ==\n" + p.deanonymize(reply))
    if args.vault:
        vault.save(args.vault, passphrase)


if __name__ == "__main__":
    main()
