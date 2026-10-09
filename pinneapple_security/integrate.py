"""pinneapple_security in the rest of the library: signed model versions in the ModelStore, signed model cards and
TrustReports, attested experiments, audited actions.

>>> from pinneapple_security import integrate as sec
>>> key = sec.signing.generate_key()
>>> v = sec.secure_save(store, "burgers", model, signing_key=key, metadata={...}, actor="ana", audit=log)
>>> sec.verify_version(store, "burgers", v, key.public())          # {"ok": True, ...} or what changed
"""
from __future__ import annotations

import json
import os
from typing import Any

from . import attestation, signing
from .audit import AuditLog
from .integrity import Manifest, build_manifest, digest_record
from .model_security import scan_checkpoint

__all__ = ["secure_save", "resign_version", "verify_version", "sign_record", "verify_record", "sign_model_card", "verify_model_card",
           "attest_and_sign_experiment", "signing"]

_MANIFEST, _ENVELOPE = "manifest.json", "manifest.sig.json"


def secure_save(store, problem_id: str, model: Any, signing_key: signing.SigningKey, *,
                metadata: dict[str, Any] | None = None, stage: str = "development", trust_report: Any = None,
                actor: str = "", audit: AuditLog | None = None, reason: str = "") -> str:
    """``ModelStore.save`` + a manifest of the version folder, signed, + a safety scan of the checkpoint + an audit
    record. Returns the version."""
    v = store.save(problem_id, model, metadata=metadata, stage=stage, trust_report=trust_report)
    vdir = store._version_dir(problem_id, v)
    scan = scan_checkpoint(os.path.join(vdir, "model.pt"))
    m = build_manifest(vdir)
    m.save(os.path.join(vdir, _MANIFEST))
    env = signing.sign_json({"problem_id": problem_id, "version": v, "merkle_root": m.root, "scan_safe": scan.safe}, signing_key)
    with open(os.path.join(vdir, _ENVELOPE), "w") as f:
        json.dump(env, f, indent=1)
    if audit is not None:
        audit.append("model.save", actor or "unknown", target=f"{problem_id}/{v}", reason=reason,
                     data={"merkle_root": m.root, "keyid": signing_key.keyid, "stage": stage})
    return v


def resign_version(store, problem_id: str, version: str, signing_key: signing.SigningKey, *, actor: str = "",
                   audit: AuditLog | None = None, reason: str = "") -> str:
    """Re-sign a version after a legitimate change (e.g. ``ModelStore.promote`` rewrites metadata.json); the audit
    record keeps the old and new roots, so the change stays traceable."""
    vdir = store._version_dir(problem_id, version)
    old = None
    try:
        old = Manifest.load(os.path.join(vdir, _MANIFEST)).root
    except (OSError, ValueError):
        pass
    m = build_manifest(vdir)
    m.save(os.path.join(vdir, _MANIFEST))
    scan = scan_checkpoint(os.path.join(vdir, "model.pt"))
    env = signing.sign_json({"problem_id": problem_id, "version": version, "merkle_root": m.root, "scan_safe": scan.safe},
                            signing_key)
    with open(os.path.join(vdir, _ENVELOPE), "w") as f:
        json.dump(env, f, indent=1)
    if audit is not None:
        audit.append("model.resign", actor or "unknown", target=f"{problem_id}/{version}", reason=reason,
                     data={"old_root": old, "new_root": m.root, "keyid": signing_key.keyid})
    return m.root


def verify_version(store, problem_id: str, version: str, key: signing.VerifyKey | signing.SigningKey) -> dict[str, Any]:
    """Signature valid, manifest matches the signed root, files match the manifest, checkpoint scan safe."""
    vdir = store._version_dir(problem_id, version)
    out: dict[str, Any] = {"ok": False, "problem_id": problem_id, "version": version}
    try:
        with open(os.path.join(vdir, _ENVELOPE)) as f:
            payload = signing.verify_json(json.load(f), key)
        m = Manifest.load(os.path.join(vdir, _MANIFEST))
    except (OSError, ValueError, signing.VerificationError) as e:
        out["problem"] = str(e)
        return out
    if payload.get("merkle_root") != m.root or payload.get("version") != version:
        out["problem"] = "the signed manifest root differs from manifest.json"
        return out
    files = m.verify(vdir)
    scan = scan_checkpoint(os.path.join(vdir, "model.pt"))
    out.update(files=files, scan=scan.to_dict(), ok=files["ok"] and scan.safe)
    if not files["ok"]:
        out["problem"] = "files changed since signing"
    elif not scan.safe:
        out["problem"] = "checkpoint is unsafe to load"
    return out


def sign_record(obj: Any, key: signing.SigningKey) -> dict[str, Any]:
    rec = obj.to_dict() if hasattr(obj, "to_dict") else obj
    return signing.sign_json(rec, key)


def verify_record(envelope: dict[str, Any], key, expected: Any = None) -> Any:
    payload = signing.verify_json(envelope, key)
    if expected is not None:
        exp = expected.to_dict() if hasattr(expected, "to_dict") else expected
        if digest_record(exp) != digest_record(payload):
            raise signing.VerificationError("the record differs from the signed one")
    return payload


def sign_model_card(card, key: signing.SigningKey, path: str | None = None) -> dict[str, Any]:
    """Envelope over ``card.to_dict()`` (TrustReport and override included); written next to the card if ``path``."""
    env = sign_record(card, key)
    if path:
        with open(path, "w") as f:
            json.dump(env, f, indent=1)
    return env


def verify_model_card(card, envelope: dict[str, Any], key) -> dict[str, Any]:
    return verify_record(envelope, key, expected=card)


def attest_and_sign_experiment(result: Any, key: signing.SigningKey, directory: str | None = None,
                               inputs: Any = (), repo: str | None = ".") -> dict[str, Any]:
    """In-toto/SLSA statement for a ``pp.Experiment`` result, signed. With ``directory``, the result is saved there
    first, its files become subjects, and ``attestation.json`` is written next to them."""
    artifacts = {}
    if directory:
        result.save(directory)
        artifacts = {n: os.path.join(directory, n) for n in sorted(os.listdir(directory))
                     if n not in ("attestation.json",)}
    st = attestation.attest_experiment(result, artifacts=artifacts, inputs=inputs, repo=repo)
    env = signing.sign_json(st, key, payload_type="application/vnd.in-toto+json")
    if directory:
        with open(os.path.join(directory, "attestation.json"), "w") as f:
            json.dump(env, f, indent=1)
    return env
