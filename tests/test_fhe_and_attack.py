import numpy as np
import pytest

from cryptollm.attacks.vocab_permutation import make_public_embeddings, obfuscate, recover_permutation

ts = pytest.importorskip("tenseal")


def test_encrypted_generation_matches_plaintext():
    from cryptollm.fhe.char_lm import load_or_train
    from cryptollm.fhe.encrypted_lm import EncryptedLMClient, EncryptedLMServer

    model = load_or_train()
    client = EncryptedLMClient(model)
    server = EncryptedLMServer(client.public_context(), model)
    assert not server.ctx.is_private()
    with pytest.raises(ValueError):  # the server cannot decrypt what it receives
        ts.ckks_vector_from(server.ctx, client.encrypt_window(model.encode("abc"))).decrypt()
    text, stats = client.generate(server.infer, "the key ", 6)
    assert text == model.generate("the key ", 6)
    assert stats["max_logit_err"] < 1e-2


def test_server_refuses_secret_key():
    from cryptollm.fhe.char_lm import load_or_train
    from cryptollm.fhe.encrypted_lm import EncryptedLMClient, EncryptedLMServer

    model = load_or_train()
    client = EncryptedLMClient(model)
    with pytest.raises(ValueError):
        EncryptedLMServer(client.ctx.serialize(save_secret_key=True), model)


def test_vocab_permutation_is_recoverable():
    E = make_public_embeddings(V=800, d=32)
    E_priv, pi = obfuscate(E, rotate=True, noise=0.05)
    assert (recover_permutation(E, E_priv) == pi).mean() > 0.95


def test_encrypted_attention_step_matches_plaintext():
    pytest.importorskip("openfhe")  # CPython 3.12 wheels; needs ~3 GB RAM
    from cryptollm.fhe.encrypted_attention import main

    out = main(verbose=False)
    assert out["max_error"] < 1e-6
