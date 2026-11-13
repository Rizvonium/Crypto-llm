"""Reversible pseudonymization: hide private values before a prompt leaves the machine,
restore them in the answer after it comes back.

    "Email Dr. Priya Rao at priya@uni.edu"  ->  "Email Dr. [PERSON_1] at [EMAIL_1]"
    model answer mentions [PERSON_1]        ->  restored to "Priya Rao" locally

Properties:
* consistent: the same value always maps to the same placeholder (across turns, if you
  persist the vault), so the model can still reason about "who did what";
* sticky: once a value has been seen, later bare mentions are swept even if no detector
  fires again (e.g. "Priya Rao" without the "Dr."), and first/last-name parts of a
  detected person are swept as aliases;
* fail-closed: ``check_no_leak`` refuses to send text that still contains a known original.
"""

from __future__ import annotations

import re
from typing import Iterable, Iterator

from .detectors import DEFAULT_DETECTORS, Detector, Span, detect_all
from .vault import Vault, _norm

# Given names / surnames that are also everyday words: never swept as bare aliases,
# otherwise "Dr. Will Grace" would hide every "will" and "grace" in the conversation.
COMMON_WORDS = frozenset("""
will may mark rose grace hope joy june april august bill pat sue art dawn faith iris ray rich
sky summer sunny victor violet amber ruby pearl crystal holly ivy lily daisy heather autumn
river brook glen dale dean drew frank gene guy jack jean jay jim lane lee long rob bob chase
cliff cole crane fisher gray green grant hunter king knight lord marsh miles mason penny page
park price reed rice rich stone strong sharp short small smith taylor walker wells white wood
young west north south east bell best brown black hall hill love may lake star sun moon
""".split())


def _name_like(tok: str) -> bool:
    """Plausible stand-alone name part: letters (plus inner ' or -), 3+ chars, not a common word."""
    return (len(tok) >= 3 and re.fullmatch(r"[^\W\d_](?:[^\W\d_]|['’-](?=[^\W\d_]))*", tok) is not None
            and tok.casefold() not in COMMON_WORDS)


PLACEHOLDER_RX = re.compile(r"\[\s*([A-Za-z_]+?)_(\d+)\s*\]")


class LeakError(RuntimeError):
    pass


class Pseudonymizer:
    def __init__(self, detectors: Iterable[Detector] | None = None, vault: Vault | None = None,
                 alias_name_parts: bool = True, email_name_heuristic: bool = True):
        self.detectors = list(DEFAULT_DETECTORS if detectors is None else detectors)
        self.vault = vault or Vault()
        self.alias_name_parts = alias_name_parts
        self.email_name_heuristic = email_name_heuristic

    # ---------------------------------------------------------------- outbound
    def _sweep_known(self, text: str) -> list[Span]:
        """Spans for values already in the vault (sticky across turns)."""
        spans = []
        for key, ph in self.vault.originals().items():
            label, value = key.split(":", 1)
            if label in {"PHONE", "CREDIT_CARD", "IBAN", "US_SSN"}:
                continue  # format-variant numbers are re-found by their detectors
            if " " not in value and value in COMMON_WORDS:
                continue
            for m in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)", text, re.I):
                spans.append(Span(m.start(), m.end(), label, m.group(0)))
        return spans

    def _register(self, spans: list[Span], text: str) -> None:
        for s in spans:
            ph = self.vault.by_value.get(_norm(s.label, s.text)) or self.vault.placeholder_for(s.label, s.text)
            if s.label == "PERSON" and self.alias_name_parts:
                # "Dr. Priya Rao" also hides a later bare "Priya" or "Rao"
                for part in s.text.split():
                    part = part.strip(".,;:'’()-")
                    if _name_like(part):
                        self.vault.add_alias("PERSON", part, ph)
            if s.label == "EMAIL" and self.email_name_heuristic:
                # "anjali.k@gmail.com" + a standalone "anjali" elsewhere -> treat as a name
                for tok in re.split(r"[._+\-\d]+", s.text.split("@")[0]):
                    if _name_like(tok) and re.search(rf"(?<![\w@.]){re.escape(tok)}(?![\w@])", text, re.I):
                        if _norm("PERSON", tok) not in self.vault.by_value:
                            self.vault.placeholder_for("PERSON", tok)

    def find_spans(self, text: str) -> list[Span]:
        """The final, non-overlapping spans that will be hidden (detectors run once)."""
        detected = detect_all(text, self.detectors)
        self._register(detected, text)
        return detect_all(text, [_Static(self._sweep_known(text)), _Static(detected)])

    def anonymize(self, text: str) -> str:
        out, pos = [], 0
        for s in self.find_spans(text):
            out += [text[pos:s.start], self.vault.by_value[_norm(s.label, s.text)]]
            pos = s.end
        out.append(text[pos:])
        result = "".join(out)
        self.check_no_leak(result)
        return result

    def check_no_leak(self, outgoing: str) -> None:
        """Fail closed if any known original (or alias) survived substitution."""
        for key, ph in self.vault.originals().items():
            label, value = key.split(":", 1)
            if label in {"PHONE", "CREDIT_CARD", "IBAN", "US_SSN"}:
                hit = value in "".join(c for c in outgoing.upper() if c.isalnum())
            else:
                hit = len(value) >= 3 and re.search(rf"(?<!\w){re.escape(value)}(?!\w)", outgoing, re.I)
            if hit:
                raise LeakError(f"refusing to send: a value hidden as {ph} is still present")

    # ---------------------------------------------------------------- inbound
    def deanonymize(self, text: str) -> str:
        def sub(m: re.Match) -> str:
            ph = f"[{m.group(1).upper()}_{m.group(2)}]"
            return self.vault.by_placeholder.get(ph, m.group(0))

        return PLACEHOLDER_RX.sub(sub, text)

    def deanonymize_stream(self, chunks: Iterable[str]) -> Iterator[str]:
        """Rehydrate streamed output; holds back a trailing partial placeholder like '[PERS'."""
        buf = ""
        for chunk in chunks:
            buf += chunk
            cut = buf.rfind("[")
            if cut != -1 and "]" not in buf[cut:] and len(buf) - cut <= 40:
                ready, buf = buf[:cut], buf[cut:]
            else:
                ready, buf = buf, ""
            if ready:
                yield self.deanonymize(ready)
        if buf:
            yield self.deanonymize(buf)


class _Static:
    def __init__(self, spans: list[Span]):
        self.spans = spans

    def detect(self, text: str) -> list[Span]:
        return self.spans
