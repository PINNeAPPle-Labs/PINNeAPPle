"""Password hashing for logins: scrypt from the standard library by default, argon2id when ``argon2-cffi`` is
installed, and verification of bcrypt hashes (``$2b$...``, as written by ``caddy hash-password`` or ``htpasswd -B``)
when ``bcrypt`` is installed.

A stored hash is self-describing, so the scheme can change without a migration:

==================================  ======================================================================
``$scrypt$n=16384,r=8,p=1$<salt>$<key>``  :func:`hashlib.scrypt`, salt and key in URL-safe base64 (no padding)
``$argon2id$v=19$m=...``            argon2-cffi ``PasswordHasher`` format
``$2a$`` / ``$2b$`` / ``$2y$``      bcrypt (verify only)
==================================  ======================================================================

``python -m pinneapple_security hash-password`` prints a hash for an environment variable such as
``HSS_PASSWORD_HASH`` (see ``apps/_shared/README.md``).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import secrets

__all__ = ["hash_password", "verify_password", "needs_rehash", "is_hash", "SCRYPT_N"]

SCRYPT_N, SCRYPT_R, SCRYPT_P = 2 ** 14, 8, 1          # OWASP minimum for scrypt (N=2^14 with r=8 is ~16 MiB)
_PREFIXES = ("$scrypt$", "$argon2", "$2a$", "$2b$", "$2y$")


def _b64(b: bytes) -> str:
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def _unb64(s: str) -> bytes:
    return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))


def is_hash(value: str | None) -> bool:
    """Whether ``value`` looks like a hash this module can verify (as opposed to a plain-text password)."""
    return bool(value) and value.startswith(_PREFIXES)


def hash_password(password: str, scheme: str = "auto") -> str:
    """A salted hash of ``password``. ``scheme``: ``"scrypt"``, ``"argon2"`` (needs argon2-cffi) or ``"auto"``
    (argon2 when installed, else scrypt)."""
    if not isinstance(password, str) or not password:
        raise ValueError("the password must be a non-empty string")
    if scheme in ("auto", "argon2"):
        try:
            from argon2 import PasswordHasher
            return PasswordHasher().hash(password)
        except ImportError:
            if scheme == "argon2":
                raise ImportError("argon2 hashes need `pip install argon2-cffi`") from None
    elif scheme != "scrypt":
        raise ValueError(f"unknown scheme {scheme!r}: scrypt, argon2 or auto")
    salt = secrets.token_bytes(16)
    key = hashlib.scrypt(password.encode(), salt=salt, n=SCRYPT_N, r=SCRYPT_R, p=SCRYPT_P, dklen=32)
    return f"$scrypt$n={SCRYPT_N},r={SCRYPT_R},p={SCRYPT_P}${_b64(salt)}${_b64(key)}"


def _verify_scrypt(password: str, stored: str) -> bool:
    try:
        _, _, params, salt, key = stored.split("$")
        p = dict(kv.split("=", 1) for kv in params.split(","))
        n, r, par = int(p["n"]), int(p["r"]), int(p["p"])
        if not (2 ** 10 <= n <= 2 ** 20 and 1 <= r <= 32 and 1 <= par <= 16):   # refuse absurd (DoS) parameters
            return False
        want = _unb64(key)
        got = hashlib.scrypt(password.encode(), salt=_unb64(salt), n=n, r=r, p=par, dklen=len(want),
                             maxmem=256 * 2 ** 20)
    except (ValueError, KeyError):
        return False
    return hmac.compare_digest(got, want)


def verify_password(password: str, stored: str | None) -> bool:
    """Whether ``password`` matches the hash ``stored``. Unknown formats and missing libraries give False."""
    if not stored or not isinstance(password, str):
        return False
    if stored.startswith("$scrypt$"):
        return _verify_scrypt(password, stored)
    if stored.startswith("$argon2"):
        try:
            from argon2 import PasswordHasher
            from argon2.exceptions import InvalidHashError, VerificationError
        except ImportError:
            return False
        try:
            return PasswordHasher().verify(stored, password)
        except (VerificationError, InvalidHashError):
            return False
    if stored.startswith(("$2a$", "$2b$", "$2y$")):
        try:
            import bcrypt
        except ImportError:
            return False
        try:
            return bcrypt.checkpw(password.encode(), stored.encode())
        except ValueError:
            return False
    return False


def needs_rehash(stored: str) -> bool:
    """True for a scrypt hash weaker than the current parameters (argon2: its own check)."""
    if stored.startswith("$scrypt$"):
        try:
            p = dict(kv.split("=", 1) for kv in stored.split("$")[2].split(","))
            return int(p["n"]) < SCRYPT_N or int(p["r"]) < SCRYPT_R
        except (ValueError, KeyError, IndexError):
            return True
    if stored.startswith("$argon2"):
        try:
            from argon2 import PasswordHasher
            return PasswordHasher().check_needs_rehash(stored)
        except ImportError:
            return False
    return False
