import pytest

from cryptollm.gateway.detectors import DEFAULT_DETECTORS, TermsDetector, detect_all, iban_ok, luhn_ok
from cryptollm.gateway.pseudonymizer import LeakError, Pseudonymizer
from cryptollm.gateway.vault import Vault


def labels(text):
    return [(s.label, s.text) for s in detect_all(text, DEFAULT_DETECTORS)]


def test_structured_identifiers():
    found = labels("mail a.b@example.org, card 4111-1111-1111-1111, IBAN GB82 WEST 1234 5698 7654 32, "
                   "ssn 123-45-6789, ip 192.168.10.7, call +44 20 7946 0958, key AKIAABCDEFGHIJKLMNOP")
    got = {l for l, _ in found}
    assert {"EMAIL", "CREDIT_CARD", "IBAN", "US_SSN", "IP_ADDRESS", "PHONE", "SECRET"} <= got


def test_checksums_reject_lookalikes():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    assert iban_ok("GB82WEST12345698765432") and not iban_ok("GB83WEST12345698765432")


@pytest.mark.parametrize("text", [
    "Released on 2026-11-12 as version 1.6.1.",
    "The value of pi is 3.14159 and the answer is 42.",
    "Meeting room 4B at 10:30, budget 12,500 dollars.",
])
def test_no_false_positives_on_ordinary_text(text):
    assert labels(text) == []


def test_roundtrip_consistency_and_aliases():
    p = Pseudonymizer([TermsDetector(["Project Falcon"], "TERM"), *DEFAULT_DETECTORS])
    out = p.anonymize("Dr. Priya Rao (priya@uni.edu) leads Project Falcon. Priya emailed priya@uni.edu again.")
    assert "Priya" not in out and "priya@uni.edu" not in out and "Falcon" not in out
    assert out.count("[PERSON_1]") == 2 and out.count("[EMAIL_1]") == 2
    reply = "Sure - I will write to [PERSON_1] at [email_1] about [ TERM_1 ]."
    assert p.deanonymize(reply) == "Sure - I will write to Priya Rao at priya@uni.edu about Project Falcon."


def test_sticky_across_turns():
    p = Pseudonymizer()
    p.anonymize("Prof. Arjun Mehta reviewed it.")
    assert p.anonymize("Ask Arjun Mehta, or just Mehta.") == "Ask [PERSON_1], or just [PERSON_1]."


def test_streaming_rehydration_handles_split_placeholders():
    p = Pseudonymizer()
    p.anonymize("Write to Dr. Lena Park.")
    chunks = ["Dear [PER", "SON_1], thanks", " - see [", "PERSON_1]!"]
    assert "".join(p.deanonymize_stream(chunks)) == "Dear Lena Park, thanks - see Lena Park!"


def test_leak_check_fails_closed():
    p = Pseudonymizer()
    p.anonymize("Dr. Lena Park")
    with pytest.raises(LeakError):
        p.check_no_leak("hello Lena Park")


def test_vault_encrypted_roundtrip(tmp_path):
    v = Vault()
    ph = v.placeholder_for("EMAIL", "x@y.com")
    path = tmp_path / "v.vault"
    v.save(path, "correct horse")
    assert b"x@y.com" not in path.read_bytes()
    assert Vault.load(path, "correct horse").by_placeholder[ph] == "x@y.com"
    with pytest.raises(ValueError):
        Vault.load(path, "wrong")
    blob = bytearray(path.read_bytes()); blob[-1] ^= 1; path.write_bytes(bytes(blob))
    with pytest.raises(ValueError):
        Vault.load(path, "correct horse")
