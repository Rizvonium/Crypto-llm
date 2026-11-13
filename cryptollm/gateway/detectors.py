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
URL = RegexDetector("URL", r"\b(?:https?|wss?|ftps?|sftp|ssh|git)://[^\s<>\"')\]]+")
# The account name inside home-directory paths (/home/alice/..., /Users/alice/..., C:\Users\alice\...)
HOME_PATH_USER = RegexDetector(
    "USERNAME",
    r"(?:/home/|/Users/|[A-Za-z]:\\Users\\|[A-Za-z]:/Users/)([A-Za-z0-9._-]{2,})",
    lambda u: u.lower() not in {"shared", "public", "default", "all users", "admin", "user", "username", "runner", "ubuntu"},
    group=1,
)
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
    r"(?:\b(?:sk-(?:ant-|proj-)?[A-Za-z0-9_-]{20,}|(?:sk|rk|pk)_(?:live|test)_[0-9A-Za-z]{16,}"
    r"|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{40,}"
    r"|xox[abpr]-[A-Za-z0-9-]{10,}|AIza[0-9A-Za-z_-]{35}|hf_[A-Za-z0-9]{30,}"
    r"|eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,})\b"
    r"|-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----)",
)


def _entropy(s: str) -> float:
    from collections import Counter
    from math import log2

    n = len(s)
    return -sum(c / n * log2(c / n) for c in Counter(s).values())


# Random-looking tokens (API secrets, passwords, session cookies): long, mixed upper/lower/digits,
# high Shannon entropy. Lowercase-only hashes and plain paths do not qualify.
HIGH_ENTROPY = RegexDetector(
    "SECRET",
    r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/=_-]{24,}(?![A-Za-z0-9+/=_-])",
    lambda t: (any(c.islower() for c in t) and any(c.isupper() for c in t) and any(c.isdigit() for c in t)
               and _entropy(t) >= 4.0),
)


class ContextualIdDetector:
    """Identifier-looking tokens right after a cue word ("passport no. A07345921",
    "MRN 00482913", "my PAN ABCPK1234L", "booking reference HX-77821-LP"). Cue words make
    this precise enough to catch formats no checksum regex knows about."""

    CUES = (r"passport|pan|aadhaa?r|ssn|social security|national id|id(?:entity)?(?: card)?|licen[cs]e|"
            r"account|acct|a/c|iban|routing|sort code|policy|member(?:ship)?|customer|client|patient|mrn|"
            r"employee|staff|student|roll|registration|reg|enrol(?:l)?ment|booking|reservation|pnr|"
            r"confirmation|reference|ref|order|invoice|ticket|case|claim|tracking|serial|vin|tax|tin|"
            r"gstin|ein|uan|voter|nhs|medicare|insurance|pin")
    RX = re.compile(
        rf"\b(?:{CUES})\b(?:\s*(?:no\.?|number|num|#|id|code|is|was|:|-))*\s*[:#]?\s*"
        r"(?P<id>(?=[A-Za-z0-9-]*\d)[A-Za-z0-9](?:[A-Za-z0-9-]{3,}[A-Za-z0-9])"
        r"|\d[\d -]{3,}\d)",
        re.I,
    )

    def detect(self, text: str) -> list[Span]:
        out = []
        for m in self.RX.finditer(text):
            tok = m.group("id")
            if sum(c.isdigit() for c in tok) >= 3 and not tok.isalpha():
                out.append(Span(m.start("id"), m.end("id"), "ID", tok))
        return out


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
    """Optional NER via Microsoft Presidio + spaCy (pip install presidio-analyzer and a spaCy
    model, e.g. en_core_web_lg). Only the NER entities are used; structured IDs are already
    covered by the checksum-validated detectors above."""

    LABELS = {"PERSON": "PERSON", "LOCATION": "LOCATION", "NRP": "GROUP", "ORGANIZATION": "ORG"}

    def __init__(self, threshold: float = 0.6, entities: Iterable[str] | None = None):
        from presidio_analyzer import AnalyzerEngine  # imported lazily

        self.engine, self.threshold = AnalyzerEngine(), threshold
        supported = set(self.engine.get_supported_entities("en"))
        self.entities = [e for e in (entities or self.LABELS) if e in supported]

    def detect(self, text: str) -> list[Span]:
        res = self.engine.analyze(text=text, language="en", entities=self.entities)
        return [Span(r.start, r.end, self.LABELS.get(r.entity_type, r.entity_type), text[r.start:r.end])
                for r in res if r.score >= self.threshold]


def _chunks(text: str, max_chars: int):
    """Split on sentence/line boundaries into pieces of at most ~max_chars, with offsets."""
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind("\n", start, end))
            if cut > start:
                end = cut + 1
        yield start, text[start:end]
        start = end


class GlinerDetector:
    """Optional zero-shot NER with a GLiNER PII model (pip install gliner; CPU is fine,
    ~0.2 s per short prompt). Catches lowercase and non-Western names that regexes and
    small spaCy models miss. Label names are natural-language prompts to the model."""

    DEFAULT_LABELS = {
        "person": "PERSON",
        "street address": "ADDRESS",
        "phone number": "PHONE",
        "passport number": "ID",
        "national id number": "ID",
        "bank account number": "ID",
        "date of birth": "DOB",
        "username": "USERNAME",
    }
    QUASI_LABELS = {"organization": "ORG", "city": "LOCATION", "location": "LOCATION"}

    def __init__(self, model: str = "urchade/gliner_multi_pii-v1", labels: dict[str, str] | None = None,
                 threshold: float = 0.5, include_quasi: bool = False, max_chars: int = 1000):
        from gliner import GLiNER  # imported lazily

        self.model = GLiNER.from_pretrained(model)
        self.labels = dict(labels or self.DEFAULT_LABELS)
        if include_quasi:
            self.labels.update(self.QUASI_LABELS)
        self.threshold, self.max_chars = threshold, max_chars

    def detect(self, text: str) -> list[Span]:
        spans = []
        for off, chunk in _chunks(text, self.max_chars):
            for e in self.model.predict_entities(chunk, list(self.labels), threshold=self.threshold):
                spans.append(Span(off + e["start"], off + e["end"], self.labels[e["label"]], e["text"]))
        return spans


DEFAULT_DETECTORS: list[Detector] = [SECRET, EMAIL, URL, IBAN, CREDIT_CARD, SSN, IPV4, IPV6, PHONE,
                                     ContextualIdDetector(), HIGH_ENTROPY, HOME_PATH_USER, HONORIFIC_NAME]


def detect_all(text: str, detectors: Iterable[Detector]) -> list[Span]:
    """Run detectors and resolve overlaps (earliest start wins, then the longest span)."""
    spans = sorted((s for d in detectors for s in d.detect(text)), key=lambda s: (s.start, -(s.end - s.start)))
    out: list[Span] = []
    for s in spans:
        if out and s.start < out[-1].end:
            continue
        out.append(s)
    return out
