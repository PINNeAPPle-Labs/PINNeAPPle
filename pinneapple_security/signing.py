"""Signatures for artifacts and records: Ed25519 (public-key, needs ``cryptography``) or HMAC-SHA256 (shared key, stdlib).

Signed JSON uses the DSSE envelope (Dead Simple Signing Envelope, used by in-toto and Sigstore): the payload is signed
together with its type through the pre-authentication encoding ``PAE``, so a signature over one kind of document can
not be replayed as another. Every signature carries a ``keyid`` (SHA-256 of the public key, or of the HMAC key), so a
verifier knows which key to use and a wrong key fails instead of passing by accident.

>>> key = generate_key()                         # Ed25519 private key (keep it secret; give out key.public())
>>> env = sign_json({"model": "burgers_pinn", "sha256": "..."}, key)
>>> verify_json(env, key.public())               # the payload, or VerificationError
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from dataclasses import dataclass
from typing import Any

__all__ = ["SigningKey", "VerifyKey", "generate_key", "hmac_key", "sign_bytes", "verify_bytes", "sign_json",
           "verify_json", "sign_file", "verify_file", "VerificationError", "PAYLOAD_TYPE"]

PAYLOAD_TYPE = "application/vnd.pinneapple+json"


class VerificationError(Exception):
    pass


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode("ascii")


def _unb64(s: str) -> bytes:
    return base64.b64decode(s.encode("ascii"))


def pae(payload_type: str, payload: bytes) -> bytes:
    """DSSE pre-authentication encoding."""
    t = payload_type.encode("utf-8")
    return b"DSSEv1 %d %s %d %s" % (len(t), t, len(payload), payload)


def _crypto():
    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import ed25519
    except ImportError as e:  # pragma: no cover
        raise ImportError("Ed25519 signatures need the 'cryptography' package: pip install pinneapple[security]") from e
    return ed25519, serialization


@dataclass(frozen=True)
class VerifyKey:
    """Public half (Ed25519) or the shared secret (HMAC). ``raw``: 32 bytes."""
    algorithm: str
    raw: bytes

    @property
    def keyid(self) -> str:
        return hashlib.sha256(self.algorithm.encode() + b":" + self.raw).hexdigest()[:16]

    def to_dict(self) -> dict[str, str]:
        if self.algorithm == "hmac-sha256":
            raise ValueError("an HMAC key is secret: do not export it as a public key")
        return {"algorithm": self.algorithm, "public_key": _b64(self.raw), "keyid": self.keyid}

    @classmethod
    def from_dict(cls, d: dict[str, str]) -> VerifyKey:
        return cls(d["algorithm"], _unb64(d["public_key"]))

    def verify(self, data: bytes, sig: bytes) -> bool:
        if self.algorithm == "hmac-sha256":
            return hmac.compare_digest(hmac.new(self.raw, data, hashlib.sha256).digest(), sig)
        ed25519, _ = _crypto()
        try:
            ed25519.Ed25519PublicKey.from_public_bytes(self.raw).verify(sig, data)
            return True
        except Exception:
            return False


@dataclass(frozen=True)
class SigningKey:
    algorithm: str
    raw: bytes          # Ed25519 private seed or HMAC secret

    def public(self) -> VerifyKey:
        if self.algorithm == "hmac-sha256":
            return VerifyKey(self.algorithm, self.raw)
        ed25519, serialization = _crypto()
        pub = ed25519.Ed25519PrivateKey.from_private_bytes(self.raw).public_key()
        return VerifyKey(self.algorithm, pub.public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw))

    @property
    def keyid(self) -> str:
        return self.public().keyid

    def sign(self, data: bytes) -> bytes:
        if self.algorithm == "hmac-sha256":
            return hmac.new(self.raw, data, hashlib.sha256).digest()
        ed25519, _ = _crypto()
        return ed25519.Ed25519PrivateKey.from_private_bytes(self.raw).sign(data)

    def save(self, path: str) -> str:
        """Write the private key (base64) with owner-only permissions."""
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"algorithm": self.algorithm, "private_key": _b64(self.raw)}, f)
        return path

    @classmethod
    def load(cls, path: str) -> SigningKey:
        with open(path) as f:
            d = json.load(f)
        return cls(d["algorithm"], _unb64(d["private_key"]))


def generate_key() -> SigningKey:
    """New Ed25519 key (needs ``cryptography``)."""
    ed25519, serialization = _crypto()
    k = ed25519.Ed25519PrivateKey.generate()
    return SigningKey("ed25519", k.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                                                 serialization.NoEncryption()))


def hmac_key(secret: bytes | None = None) -> SigningKey:
    """Shared-secret key (stdlib only). Anyone who can verify can also sign: use it inside one organisation."""
    secret = secret if secret is not None else os.urandom(32)
    if len(secret) < 16:
        raise ValueError("HMAC secret must be at least 16 bytes")
    return SigningKey("hmac-sha256", secret)


def sign_bytes(data: bytes, key: SigningKey) -> dict[str, str]:
    return {"keyid": key.keyid, "algorithm": key.algorithm, "sig": _b64(key.sign(data))}


def verify_bytes(data: bytes, signature: dict[str, str], key: VerifyKey) -> None:
    if signature.get("keyid") != key.keyid:
        raise VerificationError(f"signature made with key {signature.get('keyid')}, not {key.keyid}")
    if not key.verify(data, _unb64(signature["sig"])):
        raise VerificationError("signature does not match the data")


def sign_json(payload: Any, key: SigningKey, payload_type: str = PAYLOAD_TYPE) -> dict[str, Any]:
    """DSSE envelope over the canonical JSON of ``payload``."""
    from .integrity import canonical_json

    body = canonical_json(payload)
    s = sign_bytes(pae(payload_type, body), key)
    return {"payloadType": payload_type, "payload": _b64(body), "signatures": [{"keyid": s["keyid"], "sig": s["sig"]}]}


def verify_json(envelope: dict[str, Any], key: VerifyKey | SigningKey) -> Any:
    """Check the envelope with ``key``; returns the payload. Raises :class:`VerificationError`."""
    key = key.public() if isinstance(key, SigningKey) else key
    body = _unb64(envelope["payload"])
    data = pae(envelope["payloadType"], body)
    sigs = [s for s in envelope.get("signatures", []) if s.get("keyid") == key.keyid]
    if not sigs:
        raise VerificationError(f"no signature by key {key.keyid}")
    if not any(key.verify(data, _unb64(s["sig"])) for s in sigs):
        raise VerificationError("signature does not match the payload")
    return json.loads(body)


def sign_file(path: str, key: SigningKey, sig_path: str | None = None) -> str:
    """Detached signature ``<path>.sig`` (JSON) over the file's SHA-256 and name."""
    from .integrity import sha256_file

    env = sign_json({"file": os.path.basename(path), "sha256": sha256_file(path)}, key)
    sig_path = sig_path or path + ".sig"
    with open(sig_path, "w") as f:
        json.dump(env, f, indent=1)
    return sig_path


def verify_file(path: str, key: VerifyKey | SigningKey, sig_path: str | None = None) -> dict[str, Any]:
    from .integrity import sha256_file

    with open(sig_path or path + ".sig") as f:
        payload = verify_json(json.load(f), key)
    if payload["sha256"] != sha256_file(path):
        raise VerificationError(f"{path} changed after it was signed")
    return payload
