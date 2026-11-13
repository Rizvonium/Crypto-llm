"""Encrypted next-token generation with CKKS (TenSEAL): the server never sees the prompt
or the output.

Protocol (one round trip per generated character):

    client                                       server (no secret key)
    ------                                       ----------------------
    keygen (secret key stays here)
    public context (relin + Galois keys) ----->  load context, check it has no secret key
    x = embed(last k chars); ct = Enc(x) ---->   ct_h = (ct @ W1 + b1) * mask
                                                 ct_a = ct_h^2 + 2*ct_h      (quadratic act.)
                                                 ct_l[i] = <ct_a, W2[:, i]/4>  (per token)
    logits = Dec(ct_l) + b2; pick next char <---  return ct_l
    repeat

Everything crossing the boundary is serialized bytes, so the server object can be put in
another process or another machine without change (see ``--separate-process``).

What the server still learns: how many characters you generate (one request each), timing,
and ciphertext sizes. It learns nothing about their content (under the RLWE assumption,
128-bit security parameters enforced by Microsoft SEAL).
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import struct
import time

import numpy as np
import tenseal as ts

from .char_lm import CharLM, load_or_train

POLY_MODULUS = 16384
COEFF_BITS = [60, 40, 40, 40, 40, 60]  # 4 levels: matmul, slot mask, square, dot products
SCALE = 2**40


class EncryptedLMServer:
    """Holds the model weights in plaintext and only ever sees ciphertexts."""

    def __init__(self, public_context: bytes, model: CharLM):
        self.ctx = ts.context_from(public_context)
        if self.ctx.is_private():
            raise ValueError("server was handed a secret key; refusing")
        self.W1 = model.W1.tolist()
        self.b1 = model.b1.tolist()
        # fold the 0.25 of the activation into W2; b2 is added by the client after decryption
        self.W2q_cols = [col.tolist() for col in (0.25 * model.W2).T]
        # plaintext 0/1 mask: zero every slot past the hidden size so later rotations
        # (inside the dot products) cannot pull stray values into the sums.
        self.mask = [1.0] * model.W1.shape[1]

    def infer(self, ct_bytes: bytes) -> bytes:
        x = ts.ckks_vector_from(self.ctx, ct_bytes)
        h = (x.matmul(self.W1) + self.b1) * self.mask
        a = h.square() + (h + h)  # 4 * (0.25 h^2 + 0.5 h); addition costs no level
        # TenSEAL's matmul is only correct on a freshly encrypted (slot-replicated) vector,
        # so the second layer is one encrypted dot product per vocabulary entry.
        outs = [a.dot(col).serialize() for col in self.W2q_cols]
        return pack(outs)


def pack(chunks: list[bytes]) -> bytes:
    return b"".join(struct.pack(">I", len(c)) + c for c in chunks)


def unpack(blob: bytes) -> list[bytes]:
    out, i = [], 0
    while i < len(blob):
        (n,) = struct.unpack_from(">I", blob, i)
        out.append(blob[i + 4 : i + 4 + n])
        i += 4 + n
    return out


class EncryptedLMClient:
    """Holds the secret key, does the embedding lookup, decrypts, and picks tokens."""

    def __init__(self, model: CharLM):
        self.model = model
        self.ctx = ts.context(ts.SCHEME_TYPE.CKKS, POLY_MODULUS, coeff_mod_bit_sizes=COEFF_BITS)
        self.ctx.global_scale = SCALE
        self.ctx.generate_galois_keys()
        self.ctx.generate_relin_keys()

    def public_context(self) -> bytes:
        return self.ctx.serialize(
            save_public_key=True, save_secret_key=False, save_galois_keys=True, save_relin_keys=True
        )

    def encrypt_window(self, ids: list[int]) -> bytes:
        return ts.ckks_vector(self.ctx, self.model.features(ids).tolist()).serialize()

    def decrypt_logits(self, blob: bytes) -> np.ndarray:
        vals = [ts.ckks_vector_from(self.ctx, c).decrypt()[0] for c in unpack(blob)]
        return np.array(vals) + self.model.b2

    def generate(self, infer, prompt: str, n: int, log=None) -> tuple[str, dict]:
        """`infer` is any callable bytes -> bytes that reaches the server."""
        ids = self.model.encode(prompt)
        out: list[int] = []
        stats = {"up_bytes": 0, "down_bytes": 0, "seconds": 0.0, "max_logit_err": 0.0}
        for _ in range(n):
            t0 = time.perf_counter()
            ct = self.encrypt_window(ids)
            resp = infer(ct)
            logits = self.decrypt_logits(resp)
            stats["seconds"] += time.perf_counter() - t0
            stats["up_bytes"] += len(ct)
            stats["down_bytes"] += len(resp)
            ref = self.model.next_logits(ids)
            stats["max_logit_err"] = max(stats["max_logit_err"], float(np.abs(logits - ref).max()))
            nxt = int(np.argmax(logits))
            ids.append(nxt)
            out.append(nxt)
            if log:
                log(self.model.decode([nxt]))
        stats["chars"] = n
        return self.model.decode(out), stats


def _server_process(conn, public_context: bytes):
    server = EncryptedLMServer(public_context, load_or_train())
    while True:
        msg = conn.recv_bytes()
        if msg == b"STOP":
            break
        conn.send_bytes(server.infer(msg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt", nargs="?", default="the server ")
    ap.add_argument("-n", type=int, default=40, help="characters to generate")
    ap.add_argument("--separate-process", action="store_true",
                    help="run the server in its own OS process; only bytes cross the pipe")
    args = ap.parse_args()

    model = load_or_train()
    t0 = time.perf_counter()
    client = EncryptedLMClient(model)
    pub = client.public_context()
    print(f"[client] keys generated in {time.perf_counter() - t0:.1f}s; "
          f"public evaluation keys sent to server: {len(pub) / 1e6:.1f} MB (secret key stays local)")

    if args.separate_process:
        parent, child = mp.Pipe()
        proc = mp.Process(target=_server_process, args=(child, pub), daemon=True)
        proc.start()

        def infer(b: bytes) -> bytes:
            parent.send_bytes(b)
            return parent.recv_bytes()
    else:
        server = EncryptedLMServer(pub, model)
        infer = server.infer

    sample = client.encrypt_window(model.encode(args.prompt))
    print(f"[server] sees for the prompt only {len(sample) / 1e3:.0f} kB of ciphertext, e.g. "
          f"{sample[200:232].hex()}...")
    print(f"[client] prompt: {args.prompt!r}\n[client] decrypted output: ", end="", flush=True)
    text, st = client.generate(infer, args.prompt, args.n, log=lambda c: print(c, end="", flush=True))
    print()
    if args.separate_process:
        parent.send_bytes(b"STOP")
        proc.join()

    plain = model.generate(args.prompt, args.n)
    t1 = time.perf_counter()
    for _ in range(20):
        model.next_logits(model.encode(args.prompt))
    plain_tok = (time.perf_counter() - t1) / 20
    enc_tok = st["seconds"] / st["chars"]
    print(f"plaintext model output:   {plain!r}  (identical: {plain == text})")
    print(f"max |encrypted - plaintext| logit error: {st['max_logit_err']:.2e}")
    print(f"per-character latency: encrypted {enc_tok * 1e3:.0f} ms vs plaintext {plain_tok * 1e6:.0f} us "
          f"(~{enc_tok / plain_tok:,.0f}x slower)")
    print(f"traffic per character: {st['up_bytes'] / st['chars'] / 1e3:.0f} kB up, "
          f"{st['down_bytes'] / st['chars'] / 1e3:.0f} kB down (plaintext: 8 bytes)")


if __name__ == "__main__":
    main()
