"""Detectors for sensitive spans. Pure-Python by default; NER plugs in optionally.

Each detector returns ``Span`` objects. Structured identifiers are validated with their
checksums (Luhn, IBAN mod-97, IP parsing) to keep false positives low. Free-text names are
the hard part: the default ``HonorificNameDetector`` only catches titled names, so for real
use add your own terms (``TermsDetector``) and/or an NER model (``PresidioDetector``).
"""

from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass
from typing import Iterable, Protocol


@dataclass(frozen=True)
class Span:
    start: int
    end: int
    label: str
    text: str


class Detector(Protocol):
    def detect(self, text: str) -> list[Span]: ...


class RegexDetector:
    def __init__(self, label: str, pattern: str, validate=None, flags: int = 0, group: int = 0):
        self.label, self.rx, self.validate, self.group = label, re.compile(pattern, flags), validate, group

    def detect(self, text: str) -> list[Span]:
        out = []
        for m in self.rx.finditer(text):
            s = m.group(self.group)
            if s and (self.validate is None or self.validate(s)):
                out.append(Span(m.start(self.group), m.end(self.group), self.label, s))
        return out


def luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d = d * 2 - 9 if d * 2 > 9 else d * 2
        total += d
    return total % 10 == 0


def iban_ok(s: str) -> bool:
    s = s.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    num = "".join(str(int(c, 36)) for c in s[4:] + s[:4])
    return int(num) % 97 == 1


def ip_ok(s: str) -> bool:
    try:
        ip = ipaddress.ip_address(s)
    except ValueError:
        return False
    return not (ip.is_loopback or ip.is_unspecified)


def ssn_ok(s: str) -> bool:
    a, g, n = s.split("-")
    return a not in ("000", "666") and not a.startswith("9") and g != "00" and n != "0000"


DATE_LIKE = re.compile(r"^\d{4}[-./]\d{1,2}[-./]\d{1,2}$|^\d{1,2}[-./]\d{1,2}[-./]\d{2,4}$")


def phone_ok(s: str) -> bool:
    digits = re.sub(r"\D", "", s)
    return 8 <= len(digits) <= 15 and not DATE_LIKE.match(s.strip())


EMAIL = RegexDetector("EMAIL", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
URL = RegexDetector("URL", r"\bhttps?://[^\s<>\"')\]]+")
CREDIT_CARD = RegexDetector("CREDIT_CARD", r"\b(?:\d[ -]?){12,18}\d\b", luhn_ok)
IBAN = RegexDetector("IBAN", r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]){11,30}\b", iban_ok)
SSN = RegexDetector("US_SSN", r"\b\d{3}-\d{2}-\d{4}\b", ssn_ok)
IPV4 = RegexDetector("IP_ADDRESS", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", ip_ok)
IPV6 = RegexDetector("IP_ADDRESS", r"\b(?:[0-9A-Fa-f]{1,4}:){2,7}[0-9A-Fa-f]{1,4}\b", ip_ok)
# Phones: require a leading + or a typical grouping, so dates and versions are not caught.
PHONE = RegexDetector(
    "PHONE",
    r"(?<![\w.])(?:\+\d{1,3}[ .-]?)?(?:\(\d{2,4}\)[ .-]?)?\d{2,5}(?:[ .-]\d{2,5}){1,4}(?![\w.-])",
    lambda s: phone_ok(s) and (s.startswith("+") or "(" in s or len(re.findall(r"\d+", s)) >= 3),
)
SECRET = RegexDetector(
    "SECRET",
    r"\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}"
    r"|xox[abpr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35})\b",
)
HONORIFIC_NAME = RegexDetector(
    "PERSON",
    r"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof|Sir|Dame)\.?\s+((?:[A-Z][a-z'’-]+)(?:\s+[A-Z][a-z'’-]+){0,2})",
    group=1,
)


class TermsDetector:
    """Always redact these terms (names, projects, places). Case-insensitive, whole word."""

    def __init__(self, terms: dict[str, str] | Iterable[str], default_label: str = "PERSON"):
        if not isinstance(terms, dict):
            terms = {t: default_label for t in terms}
        self.items = [(re.compile(rf"(?<!\w){re.escape(t)}(?!\w)", re.I), lab) for t, lab in terms.items() if t.strip()]

    def detect(self, text: str) -> list[Span]:
        return [Span(m.start(), m.end(), lab, m.group(0)) for rx, lab in self.items for m in rx.finditer(text)]


class PresidioDetector:
    """Optional NER via Microsoft Presidio (pip install presidio-analyzer + a spaCy model)."""

    LABELS = {"PERSON": "PERSON", "LOCATION": "LOCATION", "NRP": "GROUP", "ORGANIZATION": "ORG"}

    def __init__(self, threshold: float = 0.6):
        from presidio_analyzer import AnalyzerEngine  # imported lazily

        self.engine, self.threshold = AnalyzerEngine(), threshold

    def detect(self, text: str) -> list[Span]:
        res = self.engine.analyze(text=text, language="en", entities=list(self.LABELS))
        return [Span(r.start, r.end, self.LABELS[r.entity_type], text[r.start:r.end])
                for r in res if r.score >= self.threshold]


DEFAULT_DETECTORS: list[Detector] = [SECRET, EMAIL, URL, IBAN, CREDIT_CARD, SSN, IPV4, IPV6, PHONE, HONORIFIC_NAME]


def detect_all(text: str, detectors: Iterable[Detector]) -> list[Span]:
    """Run detectors and resolve overlaps (earliest start wins, then the longest span)."""
    spans = sorted((s for d in detectors for s in d.detect(text)), key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            continue
        out.append(s)
    return out
