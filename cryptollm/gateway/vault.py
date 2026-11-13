"""The mapping vault: placeholder <-> original value. It never leaves your machine.

On disk it is encrypted with AES-256-GCM under a key derived from your passphrase with
scrypt, so a stolen laptop backup does not reveal the originals either.
"""

from __future__ import annotations

import json
import os
import pathlib
import re

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

MAGIC = b"CLLMVAULT1"


def _norm(label: str, value: str) -> str:
    v = value.strip()
    if label in {"PHONE", "CREDIT_CARD", "IBAN", "US_SSN"}:
        return label + ":" + re.sub(r"[\s().-]", "", v).upper()
    return label + ":" + " ".join(v.split()).casefold()


class Vault:
    def __init__(self) -> None:
        self.by_placeholder: dict[str, str] = {}
        self.by_value: dict[str, str] = {}
        self.counters: dict[str, int] = {}

    def placeholder_for(self, label: str, value: str) -> str:
        key = _norm(label, value)
        if key not in self.by_value:
            self.counters[label] = self.counters.get(label, 0) + 1
            ph = f"[{label}_{self.counters[label]}]"
            self.by_value[key] = ph
            self.by_placeholder[ph] = value
        return self.by_value[key]

    def add_alias(self, label: str, alias: str, placeholder: str) -> None:
        self.by_value.setdefault(_norm(label, alias), placeholder)

    def originals(self) -> dict[str, str]:
        """normalized-key -> placeholder, for sweeping previously seen values."""
        return dict(self.by_value)

    # ---- encrypted persistence --------------------------------------------
    def to_json(self) -> bytes:
        return json.dumps({"p": self.by_placeholder, "v": self.by_value, "c": self.counters}).encode()

    @classmethod
    def from_json(cls, data: bytes) -> "Vault":
        d, v = json.loads(data), cls()
        v.by_placeholder, v.by_value, v.counters = d["p"], d["v"], d["c"]
        return v

    @staticmethod
    def _key(passphrase: str, salt: bytes) -> bytes:
        return Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode())

    def save(self, path: str | os.PathLike, passphrase: str) -> None:
        salt, nonce = os.urandom(16), os.urandom(12)
        ct = AESGCM(self._key(passphrase, salt)).encrypt(nonce, self.to_json(), MAGIC + salt)
        pathlib.Path(path).write_bytes(MAGIC + salt + nonce + ct)

    @classmethod
    def load(cls, path: str | os.PathLike, passphrase: str) -> "Vault":
        blob = pathlib.Path(path).read_bytes()
        if not blob.startswith(MAGIC):
            raise ValueError("not a vault file")
        salt, nonce, ct = blob[len(MAGIC):len(MAGIC) + 16], blob[len(MAGIC) + 16:len(MAGIC) + 28], blob[len(MAGIC) + 28:]
        try:
            data = AESGCM(cls._key(passphrase, salt)).decrypt(nonce, ct, MAGIC + salt)
        except InvalidTag:
            raise ValueError("wrong passphrase or tampered vault") from None
        return cls.from_json(data)
