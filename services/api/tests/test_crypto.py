"""Unit tests for the PII-at-rest encryption primitive (`app.crypto`).

These tests pin the plan Task 22 / todo Task 23 acceptance criteria (SPEC §6):

- A PII value stored at rest is *ciphertext* — you cannot read the original text
  out of what gets written down.
- The same value can be decrypted back to exactly what went in (a clean
  round-trip inside the app).
- The encryption is *authenticated* and *fails closed*: handing the decryptor a
  value that was locked with a different key, or a value someone has tampered
  with, must raise a clear error — it must never quietly return garbage or, worse,
  the plaintext.

The tests build the cipher directly with real, valid keys so they exercise only
this module's own behaviour (wiring the key up from config and into storage is a
separate slice). Keys are made with Python's built-in `base64` tool — not the
encryption library the implementation uses internally — so the only import that
should be unresolved during the RED phase is `app.crypto` itself.
"""

from __future__ import annotations

# `base64` is part of Python's standard toolbox for turning raw bytes into safe
# text. A Fernet key is exactly "32 raw bytes, written as url-safe base64 text",
# so we use this to mint two valid, distinct keys as plain strings for the tests.
import base64

import pytest

# The one and only unit under test. During RED this import fails (the module does
# not exist yet), which is the signal telling us what to build next.
from app.crypto import DecryptionError, PIICipher

# A valid encryption key: 32 copies of the letter "A" as bytes, written as
# url-safe base64 text (this is exactly the shape a Fernet key must have).
KEY_A = base64.urlsafe_b64encode(b"A" * 32).decode()
# A second, different valid key: 32 copies of "B". Used to prove that a value
# locked with one key cannot be opened with another.
KEY_B = base64.urlsafe_b64encode(b"B" * 32).decode()

# A stand-in for a piece of customer PII we would be storing at rest — an account
# number is exactly the kind of value that must never sit in the database in the
# clear.
SECRET_VALUE = "Account 4111-1111-1111-1111"


def _flip_one_character(token: str, index: int = 20) -> str:
    """Change a single character in a piece of text so it is no longer the same.

    Think of it as smudging one letter on a sealed envelope: everything else is
    untouched, but that one change is enough that the tamper-detection should
    notice. We swap the chosen character for a different one ("A" unless it was
    already "A", in which case "B").

    Its whole job is: "Give me back this text with exactly one letter altered."
    """
    # Pick a replacement that is guaranteed to differ from the current character.
    replacement = "B" if token[index] == "A" else "A"
    # Rebuild the string: everything before the spot, the new character, then the rest.
    return token[:index] + replacement + token[index + 1 :]


def test_encrypt_then_decrypt_round_trips_the_value() -> None:
    """Locking a value and then unlocking it returns exactly the original value."""
    # Make a cipher that locks and unlocks using key A.
    cipher = PIICipher(KEY_A)
    # Lock the secret account number — this is what would be written to the database.
    token = cipher.encrypt(SECRET_VALUE)
    # Unlock it again with the same key.
    recovered = cipher.decrypt(token)
    # Pass only if what came back is character-for-character the original secret.
    assert recovered == SECRET_VALUE


def test_ciphertext_does_not_contain_the_plaintext() -> None:
    """What gets stored is scrambled: the readable original is not sitting inside it."""
    # Make a cipher keyed with key A.
    cipher = PIICipher(KEY_A)
    # Lock the secret; `token` is the ciphertext that would live at rest.
    token = cipher.encrypt(SECRET_VALUE)
    # The stored text must not simply be the original text handed back unchanged.
    assert token != SECRET_VALUE
    # And the tell-tale account digits must not appear in the stored text either —
    # if they did, the value would be readable at rest, which is the whole thing
    # we are trying to prevent.
    assert "4111-1111-1111-1111" not in token


def test_decrypt_with_a_different_key_fails_closed() -> None:
    """A value locked with one key cannot be opened with another — it errors, not leaks."""
    # Lock the secret using key A.
    locking_cipher = PIICipher(KEY_A)
    token = locking_cipher.encrypt(SECRET_VALUE)
    # Now try to open it with a cipher that only knows key B (the "wrong key").
    wrong_key_cipher = PIICipher(KEY_B)
    # The attempt must raise our clear "could not decrypt" error — failing closed —
    # rather than returning made-up bytes or, worst of all, the plaintext.
    with pytest.raises(DecryptionError):
        wrong_key_cipher.decrypt(token)


def test_decrypt_of_tampered_ciphertext_fails_closed() -> None:
    """If the stored ciphertext is altered, decryption refuses it instead of trusting it."""
    # Make a cipher keyed with key A and lock the secret.
    cipher = PIICipher(KEY_A)
    token = cipher.encrypt(SECRET_VALUE)
    # Smudge one character of the stored ciphertext to simulate tampering / corruption.
    tampered = _flip_one_character(token)
    # Because the encryption is authenticated, the same cipher must reject the
    # altered value with our clear error rather than handing back anything at all.
    with pytest.raises(DecryptionError):
        cipher.decrypt(tampered)
