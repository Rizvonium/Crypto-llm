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


# Built at runtime so secret scanners never see a literal provider-format key (it is fake).
FAKE_STRIPE = "sk_" + "live_" + "51Hx9kQ2eZvKYlo2C8f3xT0aB"


@pytest.mark.parametrize("text,label,value", [
    ("passport number is A07345921.", "ID", "A07345921"),
    ("my PAN ABCPK1234L. how do i fix it", "ID", "ABCPK1234L"),
    ("(DOB 1958-07-22, MRN 00482913)", "ID", "00482913"),
    ("booking reference is HX-77821-LP.", "ID", "HX-77821-LP"),
    (f"headers={{'Authorization': 'Bearer {FAKE_STRIPE}'}}", "SECRET", FAKE_STRIPE),
    ("Authorization: Bearer Xq7Lm2Pz9Rt4Vw8Ny3Kd6Hb1Jf5Gc0Sa", "SECRET", "Xq7Lm2Pz9Rt4Vw8Ny3Kd6Hb1Jf5Gc0Sa"),
    ("secret wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY leaked", "SECRET", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"),
])
def test_contextual_ids_and_secrets(text, label, value):
    assert (label, value) in labels(text)


@pytest.mark.parametrize("text", [
    "see /usr/local/lib/python3.12/site-packages",
    "sha256 9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08",
    "Order 66 was executed in episode III",
    "I paid 3000 rupees for the ticket 2 days ago",
])
def test_new_detectors_do_not_fire_on_lookalikes(text):
    assert labels(text) == []


def test_common_word_names_do_not_blank_ordinary_words():
    p = Pseudonymizer()
    out = p.anonymize("Dr. Will Grace said we will need grace and hope. Grace agreed.")
    assert out.startswith("Dr. [PERSON_1] said we will need grace and hope.")


def test_email_local_part_reveals_a_name():
    p = Pseudonymizer()
    out = p.anonymize("ask anjali (anjali.k@gmail.com) to call me")
    assert "anjali" not in out.lower() and "[EMAIL_1]" in out
