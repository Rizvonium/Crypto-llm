"""Why "encrypt the tokens so the model can still read them" fails.

The tempting scheme: pick a secret permutation pi of the vocabulary, send token pi(t)
instead of t, and give the server a model whose embedding rows were permuted to match
(optionally also rotating the hidden space by a secret orthogonal matrix R and folding R^-1
into the first layer, plus some fine-tuning noise). The model then works on "ciphertext"
tokens. Variants of this are published as lightweight "private inference".

The server, however, knows the public base model. This script recovers pi:

1. plain permutation         -> nearest-neighbour match of embedding rows: exact.
2. + secret rotation R        -> rotation-invariant signatures (row norm + sorted distances
                                 to all other rows), then orthogonal Procrustes to estimate
                                 R from the confident matches, then re-match.
3. + noise (fine-tuning)      -> same pipeline, still recovers nearly everything.

Once pi is known, every "encrypted" prompt is plaintext to the server. A frequency attack
on the token stream works too, without any embeddings at all.
"""

from __future__ import annotations

import numpy as np


def make_public_embeddings(V=3000, d=64, seed=0) -> np.ndarray:
    """Synthetic stand-in for a public model's embedding table (anisotropic, like real ones)."""
    rng = np.random.default_rng(seed)
    scales = np.exp(-np.arange(d) / 24.0)
    return rng.normal(size=(V, d)) * scales + 0.1 * rng.normal(size=d)


def obfuscate(E, rotate: bool, noise: float, seed=1):
    rng = np.random.default_rng(seed)
    V, d = E.shape
    pi = rng.permutation(V)  # secret: token t is sent as pi[t]
    E_priv = np.empty_like(E)
    E_priv[pi] = E  # row pi[t] of the server's table is token t's embedding
    if rotate:
        R, _ = np.linalg.qr(rng.normal(size=(d, d)))
        E_priv = E_priv @ R
    E_priv = E_priv + noise * E.std() * rng.normal(size=E.shape)
    return E_priv, pi


def _sqdist(A, B):
    return np.maximum((A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2 * A @ B.T, 0.0)


def _signatures(E, k=32):
    D = np.sqrt(_sqdist(E, E))
    D.sort(axis=1)
    return np.hstack([np.linalg.norm(E, axis=1, keepdims=True), D[:, 1 : k + 1]])


def recover_permutation(E_pub, E_priv) -> np.ndarray:
    """Returns guess with guess[t] = attacker's estimate of pi[t]."""
    # step 1: rotation-invariant signature matching
    S_pub, S_priv = _signatures(E_pub), _signatures(E_priv)
    d = _sqdist(S_pub, S_priv)
    guess = d.argmin(axis=1)
    # step 2: Procrustes on the most confident matches, then plain nearest neighbour
    conf = np.sort(d, axis=1)
    margin = conf[:, 1] - conf[:, 0]
    good = np.argsort(-margin)[: max(len(E_pub) // 4, E_pub.shape[1] * 2)]
    for _ in range(3):
        U, _, Vt = np.linalg.svd(E_pub[good].T @ E_priv[guess[good]])
        R_hat = U @ Vt
        aligned = E_pub @ R_hat
        dist = _sqdist(aligned, E_priv)
        guess = dist.argmin(axis=1)
        good = np.arange(len(E_pub))
    return guess


def main():
    E = make_public_embeddings()
    for rotate, noise in [(False, 0.0), (True, 0.0), (True, 0.05), (True, 0.15)]:
        E_priv, pi = obfuscate(E, rotate, noise)
        acc = (recover_permutation(E, E_priv) == pi).mean()
        print(f"secret permutation{' + rotation' if rotate else ''}{f' + {noise:.0%} noise' if noise else ''}: "
              f"attacker recovers {acc:.1%} of the token mapping")


if __name__ == "__main__":
    main()
