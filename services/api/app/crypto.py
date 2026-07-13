"""Application-layer encryption for PII at rest (SPEC §6, plan Task 22 / todo Task 23).

Sensitive customer data — extracted account numbers, names, amounts, and the like
— must never sit in the database in readable form. This module holds the single
primitive that locks such a value into ciphertext before it is stored and unlocks
it again inside the app when a rep needs to see it. The key is supplied by config
(`ENCRYPTION_KEY`); email_mcp only ever stores the opaque ciphertext and never
holds the key itself.

We use Fernet (from the `cryptography` library), which is *authenticated*
encryption: every locked value carries a built-in tamper check, so a value that
was altered on disk, or one that was locked with a different key, is refused
rather than trusted. Decryption therefore *fails closed* — on any problem it
raises `DecryptionError` instead of returning partial or bogus plaintext.
"""

from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken

__all__ = ["DecryptionError", "PIICipher"]


class DecryptionError(Exception):
    """Raised when a PII value cannot be safely decrypted.

    This is the "fail closed" signal: a wrong key, a tampered ciphertext, or a
    malformed token all surface as this one error, so a caller learns only that
    the value could not be trusted — never why, and never any leaked plaintext.
    """


class PIICipher:
    """Locks and unlocks individual PII strings for storage at rest.

    Hand it the app's encryption key and it gives you two operations: `encrypt`
    turns a readable value into ciphertext safe to store, and `decrypt` turns that
    ciphertext back into the original value inside the app. Because the underlying
    Fernet scheme is authenticated, `decrypt` refuses anything it cannot verify.
    """

    def __init__(self, key: str) -> None:
        """Build a cipher from a url-safe base64 Fernet key (32 bytes of key material).

        The key comes from config as text; Fernet wants it as bytes, so we encode
        it. An malformed key raises here at construction time (fail loud at
        startup) rather than later on first use.
        """
        # Hand the key to Fernet, which does the actual locking/unlocking work.
        self._fernet = Fernet(key.encode())

    def encrypt(self, plaintext: str) -> str:
        """Lock a readable value into ciphertext text that is safe to store.

        The returned string is what gets written to the database; it reveals
        nothing about the original value and carries its own tamper check.
        """
        # Fernet works on bytes: encode the text, lock it, then decode the
        # resulting token back to text so callers/storage handle a plain string.
        return self._fernet.encrypt(plaintext.encode()).decode()

    def decrypt(self, token: str) -> str:
        """Unlock a stored ciphertext back into its original value, or fail closed.

        On any sign of trouble — the wrong key, a tampered or malformed token — it
        raises `DecryptionError` and returns nothing, so bad data can never be
        mistaken for good.
        """
        try:
            # Encode the stored token to bytes and let Fernet verify + unlock it.
            plaintext = self._fernet.decrypt(token.encode())
        except (InvalidToken, ValueError, TypeError) as exc:
            # InvalidToken = wrong key or tampered value; ValueError/TypeError =
            # malformed input. Any of them means "do not trust this" — fail closed.
            raise DecryptionError("could not decrypt PII value") from exc
        # Verified and unlocked: hand back the original readable text.
        return plaintext.decode()
