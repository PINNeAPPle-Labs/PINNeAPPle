"""Integrity manifests, signatures (Ed25519, HMAC, DSSE), audit chain."""
import json

import numpy as np
import pytest

from pinneapple_security import audit, integrity, signing


def _tree(tmp_path):
    (tmp_path / "a").mkdir(parents=True)
    (tmp_path / "a" / "u.npy").write_bytes(b"\x01\x02")
    (tmp_path / "b.csv").write_text("x,y\n1,2\n")
    return tmp_path


def test_manifest_detects_added_removed_changed(tmp_path):
    d = _tree(tmp_path / "ds")
    m = integrity.build_manifest(str(d))
    assert set(m.files) == {"a/u.npy", "b.csv"} and len(m.root) == 64
    assert m.verify(str(d))["ok"]
    (d / "b.csv").write_text("x,y\n1,3\n")
    (d / "c.txt").write_text("new")
    (d / "a" / "u.npy").unlink()
    r = m.verify(str(d))
    assert not r["ok"] and r["changed"] == ["b.csv"] and r["added"] == ["c.txt"] and r["removed"] == ["a/u.npy"]


def test_manifest_file_can_not_be_edited_silently(tmp_path):
    d = _tree(tmp_path / "ds")
    path = integrity.build_manifest(str(d)).save(str(tmp_path / "m.json"))
    doc = json.load(open(path))
    doc["files"]["b.csv"]["sha256"] = "0" * 64
    json.dump(doc, open(path, "w"))
    with pytest.raises(ValueError, match="edited"):
        integrity.Manifest.load(path)


def test_merkle_root_order_independent_of_walk_and_sensitive_to_content():
    a = integrity.Manifest(files={"x": {"sha256": "aa" * 32, "size": 1}, "y": {"sha256": "bb" * 32, "size": 1}})
    b = integrity.Manifest(files={"y": {"sha256": "bb" * 32, "size": 1}, "x": {"sha256": "aa" * 32, "size": 1}})
    c = integrity.Manifest(files={"x": {"sha256": "aa" * 32, "size": 1}, "y": {"sha256": "cc" * 32, "size": 1}})
    assert a.root == b.root != c.root
    assert integrity.merkle_root([]) == integrity.sha256_bytes(b"")


def test_array_digest_includes_shape_and_dtype():
    x = np.arange(6, dtype=np.float32)
    assert integrity.digest_array(x) != integrity.digest_array(x.reshape(2, 3))
    assert integrity.digest_array(x) != integrity.digest_array(x.astype(np.float64))
    assert integrity.digest_array(x) == integrity.digest_array(x.copy())
    assert integrity.digest_record({"b": 1, "a": 2}) == integrity.digest_record({"a": 2, "b": 1})


@pytest.mark.parametrize("make", [signing.generate_key, signing.hmac_key])
def test_sign_and_verify_json_and_files(tmp_path, make):
    key, other = make(), make()
    env = signing.sign_json({"model": "m", "rel_l2": 0.01}, key)
    assert signing.verify_json(env, key.public())["rel_l2"] == 0.01
    with pytest.raises(signing.VerificationError):
        signing.verify_json(env, other.public())
    forged = dict(env, payload=env["payload"][:-4] + "AAA=")
    with pytest.raises((ValueError, signing.VerificationError)):
        signing.verify_json(forged, key.public())
    f = tmp_path / "w.pt"
    f.write_bytes(b"weights")
    signing.sign_file(str(f), key)
    assert signing.verify_file(str(f), key)["file"] == "w.pt"
    f.write_bytes(b"weightz")
    with pytest.raises(signing.VerificationError, match="changed"):
        signing.verify_file(str(f), key)


def test_payload_type_is_bound_and_keys_round_trip(tmp_path):
    key = signing.generate_key()
    env = signing.sign_json({"a": 1}, key, payload_type="application/x-a")
    with pytest.raises(signing.VerificationError):
        signing.verify_json(dict(env, payloadType="application/x-b"), key)
    path = key.save(str(tmp_path / "k.json"))
    assert (tmp_path / "k.json").stat().st_mode & 0o077 == 0
    assert signing.SigningKey.load(path).keyid == key.keyid
    pub = signing.VerifyKey.from_dict(key.public().to_dict())
    assert signing.verify_json(env, pub) == {"a": 1}
    with pytest.raises(ValueError):
        signing.hmac_key().public().to_dict()


@pytest.mark.parametrize("key", [None, b"k" * 32])
def test_audit_chain_detects_edit_delete_and_reorder(tmp_path, key):
    p = tmp_path / "audit.jsonl"
    log = audit.AuditLog(str(p), key=key)
    for i in range(4):
        log.append("model.train", "ana", target=f"run{i}", data={"epochs": 100 * i}, reason="baseline")
    assert log.verify() == {"ok": True, "n": 4, "head": log.head()}
    assert len(log.query(actor="ana")) == 4 and log.query(target="run2")[0]["data"]["epochs"] == 200
    lines = p.read_text().splitlines()

    rec = json.loads(lines[1])
    rec["data"]["epochs"] = 999
    p.write_text("\n".join([lines[0], json.dumps(rec)] + lines[2:]) + "\n")
    assert log.verify()["bad_seq"] == 1
    p.write_text("\n".join([lines[0]] + lines[2:]) + "\n")
    assert not log.verify()["ok"]
    p.write_text("\n".join([lines[0], lines[2], lines[1], lines[3]]) + "\n")
    assert not log.verify()["ok"]


def test_hmac_audit_chain_resists_recomputed_hashes(tmp_path):
    p = tmp_path / "a.jsonl"
    log = audit.AuditLog(str(p), key=b"s" * 32)
    log.append("x", "ana")
    forger = audit.AuditLog(str(p))            # rewrites the record with plain SHA-256: the keyed check must fail
    rec = json.loads(p.read_text())
    body = {k: v for k, v in rec.items() if k != "hash"}
    body["actor"] = "eve"
    p.write_text(json.dumps({**body, "hash": forger._hash(body)}) + "\n")
    assert not log.verify()["ok"]
