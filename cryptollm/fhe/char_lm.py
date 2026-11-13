"""A tiny character-level language model designed to run under homomorphic encryption.

Architecture (Bengio-style neural LM, no attention):

    x      = concat(E[c_{t-k}], ..., E[c_{t-1}])      # client side: embedding lookup
    h      = x @ W1 + b1                              # server: plaintext-weights x ciphertext
    a      = 0.25 * h**2 + 0.5 * h                    # server: quadratic activation
    logits = a @ W2 + b2                              # server: plaintext-weights x ciphertext

The activation is a degree-2 polynomial (a quadratic approximation of SiLU), so the
encrypted forward pass is *exact* up to CKKS noise: no approximation of ReLU/GELU/softmax
is needed on the server. Softmax and sampling happen on the client after decryption.

Training is plain numpy (Adam), a few seconds on a laptop CPU. Weights are cached in
``char_lm_weights.npz`` next to this file.
"""

from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
CORPUS_PATH = HERE / "corpus.txt"
WEIGHTS_PATH = HERE / "char_lm_weights.npz"

ALPHABET = "\nabcdefghijklmnopqrstuvwxyz .,:'-"


def normalize(text: str) -> str:
    """Lowercase and map characters outside the alphabet to a space."""
    text = text.lower()
    return "".join(ch if ch in ALPHABET else " " for ch in text)


@dataclass
class CharLM:
    emb: np.ndarray  # (V, d_e)
    W1: np.ndarray  # (k*d_e, H)
    b1: np.ndarray  # (H,)
    W2: np.ndarray  # (H, V)
    b2: np.ndarray  # (V,)
    context: int

    # ----- vocabulary helpers -------------------------------------------------
    @property
    def vocab(self) -> str:
        return ALPHABET

    def encode(self, text: str) -> list[int]:
        return [ALPHABET.index(ch) for ch in normalize(text)]

    def decode(self, ids: list[int]) -> str:
        return "".join(ALPHABET[i] for i in ids)

    def window(self, ids: list[int]) -> list[int]:
        """Last `context` token ids, left-padded with spaces."""
        pad = ALPHABET.index(" ")
        ids = list(ids)[-self.context :]
        return [pad] * (self.context - len(ids)) + ids

    # ----- forward pass (plaintext reference) --------------------------------
    def features(self, ids: list[int]) -> np.ndarray:
        """Client-side embedding lookup for the current window -> flat feature vector."""
        return self.emb[self.window(ids)].reshape(-1)

    def logits_from_features(self, x: np.ndarray) -> np.ndarray:
        h = x @ self.W1 + self.b1
        a = 0.25 * h * h + 0.5 * h
        return a @ self.W2 + self.b2

    def next_logits(self, ids: list[int]) -> np.ndarray:
        return self.logits_from_features(self.features(ids))

    def generate(self, prompt: str, n: int) -> str:
        ids = self.encode(prompt)
        out = []
        for _ in range(n):
            nxt = int(np.argmax(self.next_logits(ids)))
            ids.append(nxt)
            out.append(nxt)
        return self.decode(out)

    # ----- persistence -------------------------------------------------------
    def save(self, path: pathlib.Path = WEIGHTS_PATH) -> None:
        np.savez_compressed(
            path, emb=self.emb, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
            context=np.array(self.context),
        )

    @classmethod
    def load(cls, path: pathlib.Path = WEIGHTS_PATH) -> "CharLM":
        z = np.load(path)
        return cls(z["emb"], z["W1"], z["b1"], z["W2"], z["b2"], int(z["context"]))


def train(
    context: int = 8,
    d_emb: int = 8,
    hidden: int = 96,
    steps: int = 6000,
    batch: int = 256,
    lr: float = 3e-3,
    seed: int = 0,
    verbose: bool = True,
) -> CharLM:
    rng = np.random.default_rng(seed)
    text = normalize(CORPUS_PATH.read_text())
    ids = np.array([ALPHABET.index(ch) for ch in text])
    V, D = len(ALPHABET), context * d_emb
    pad = ALPHABET.index(" ")
    padded = np.concatenate([np.full(context, pad), ids])
    X = np.stack([padded[i : i + context] for i in range(len(ids))])  # (N, k)
    Y = ids  # (N,)

    params = {
        "emb": rng.normal(0, 0.5, (V, d_emb)),
        "W1": rng.normal(0, 1.0 / np.sqrt(D), (D, hidden)),
        "b1": np.zeros(hidden),
        "W2": rng.normal(0, 1.0 / np.sqrt(hidden), (hidden, V)),
        "b2": np.zeros(V),
    }
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(v) for k, v in params.items()}
    b1_, b2_, eps, wd = 0.9, 0.999, 1e-8, 1e-4

    for step in range(1, steps + 1):
        idx = rng.integers(0, len(X), batch)
        xb, yb = X[idx], Y[idx]
        E = params["emb"][xb]  # (B, k, d_e)
        x = E.reshape(batch, D)
        h = x @ params["W1"] + params["b1"]
        a = 0.25 * h * h + 0.5 * h
        logits = a @ params["W2"] + params["b2"]
        logits -= logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p /= p.sum(axis=1, keepdims=True)
        loss = -np.log(p[np.arange(batch), yb] + 1e-12).mean()

        g = p.copy()
        g[np.arange(batch), yb] -= 1.0
        g /= batch
        grads = {
            "W2": a.T @ g,
            "b2": g.sum(0),
        }
        da = g @ params["W2"].T
        dh = da * (0.5 * h + 0.5)
        grads["W1"] = x.T @ dh
        grads["b1"] = dh.sum(0)
        dx = (dh @ params["W1"].T).reshape(batch, context, d_emb)
        demb = np.zeros_like(params["emb"])
        np.add.at(demb, xb, dx)
        grads["emb"] = demb

        # global-norm clipping keeps the quadratic activation from blowing up
        norm = np.sqrt(sum((gr**2).sum() for gr in grads.values()))
        scale = min(1.0, 1.0 / (norm + 1e-12))
        for k in params:
            gr = grads[k] * scale + wd * params[k]
            m[k] = b1_ * m[k] + (1 - b1_) * gr
            v[k] = b2_ * v[k] + (1 - b2_) * gr * gr
            mh = m[k] / (1 - b1_**step)
            vh = v[k] / (1 - b2_**step)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)

        if verbose and (step % 1000 == 0 or step == 1):
            print(f"step {step:5d}  loss {loss:.3f}")

    return CharLM(params["emb"], params["W1"], params["b1"], params["W2"], params["b2"], context)


def load_or_train() -> CharLM:
    if WEIGHTS_PATH.exists():
        return CharLM.load()
    model = train()
    model.save()
    return model


if __name__ == "__main__":
    model = train()
    model.save()
    print("saved", WEIGHTS_PATH)
    for prompt in ["the server ", "a secret is ", "the key "]:
        print(repr(prompt), "->", repr(model.generate(prompt, 60)))
