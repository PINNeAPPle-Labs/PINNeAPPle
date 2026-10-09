"""Encryption at rest: AES-256-GCM for bytes, files and arrays, with keys from a passphrase (scrypt) or random.

File format (``PPSEC1``): magic (6 bytes) | salt (16) | nonce (12) | ciphertext+tag. The header is authenticated as
associated data, so changing the salt, nonce or any byte of the ciphertext makes decryption fail instead of returning
corrupted data. GCM authenticates: a wrong key or tampered file raises, it never silently decrypts to garbage.

Needs the ``cryptography`` package (``pip install pinneapple[security]``); key derivation uses the standard library
(``hashlib.scrypt`` with n=2**15, r=8, p=1).
"""
from __future__ import annotations

import hashlib
import io
import os
from typing import Any

__all__ = ["derive_key", "new_key", "encrypt_bytes", "decrypt_bytes", "encrypt_file", "decrypt_file",
           "encrypt_array", "decrypt_array", "DecryptionError", "MAGIC"]

MAGIC = b"PPSEC1"
_SCRYPT = {"n": 2 ** 15, "r": 8, "p": 1}


class DecryptionError(Exception):
    pass


def _aesgcm(key: bytes):
    try:
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    except ImportError as e:  # pragma: no cover
        raise ImportError("encryption needs the 'cryptography' package: pip install pinneapple[security]") from e
    if len(key) != 32:
        raise ValueError("AES-256 key must be 32 bytes")
    return AESGCM(key)


def derive_key(passphrase: str | bytes, salt: bytes) -> bytes:
    if isinstance(passphrase, str):
        passphrase = passphrase.encode("utf-8")
    if len(passphrase) < 8:
        raise ValueError("passphrase must have at least 8 characters")
    return hashlib.scrypt(passphrase, salt=salt, maxmem=2 ** 26, dklen=32, **_SCRYPT)


def new_key() -> bytes:
    return os.urandom(32)


def _key(secret: str | bytes, salt: bytes) -> bytes:
    # 32 raw bytes = a key; anything else = a passphrase
    return secret if isinstance(secret, bytes) and len(secret) == 32 else derive_key(secret, salt)


def encrypt_bytes(data: bytes, secret: str | bytes) -> bytes:
    salt, nonce = os.urandom(16), os.urandom(12)
    header = MAGIC + salt + nonce
    return header + _aesgcm(_key(secret, salt)).encrypt(nonce, data, header)


def decrypt_bytes(blob: bytes, secret: str | bytes) -> bytes:
    if blob[:6] != MAGIC or len(blob) < 34 + 16:
        raise DecryptionError("not a PPSEC1 encrypted blob")
    header, salt, nonce = blob[:34], blob[6:22], blob[22:34]
    try:
        return _aesgcm(_key(secret, salt)).decrypt(nonce, blob[34:], header)
    except Exception as e:
        if isinstance(e, ImportError):
            raise
        raise DecryptionError("wrong key or the data was modified") from None


def encrypt_file(path: str, secret: str | bytes, out: str | None = None, remove_plain: bool = False) -> str:
    out = out or path + ".enc"
    with open(path, "rb") as f:
        blob = encrypt_bytes(f.read(), secret)
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(blob)
    if remove_plain:
        os.remove(path)
    return out


def decrypt_file(path: str, secret: str | bytes, out: str | None = None) -> str:
    out = out or (path[:-4] if path.endswith(".enc") else path + ".dec")
    with open(path, "rb") as f:
        data = decrypt_bytes(f.read(), secret)
    with open(out, "wb") as f:
        f.write(data)
    return out


def encrypt_array(a: Any, secret: str | bytes) -> bytes:
    import numpy as np

    buf = io.BytesIO()
    np.save(buf, np.asarray(a), allow_pickle=False)
    return encrypt_bytes(buf.getvalue(), secret)


def decrypt_array(blob: bytes, secret: str | bytes):
    import numpy as np

    return np.load(io.BytesIO(decrypt_bytes(blob, secret)), allow_pickle=False)
