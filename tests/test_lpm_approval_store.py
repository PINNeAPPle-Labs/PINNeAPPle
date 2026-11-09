"""Approval store and campaign isolation checks (no external AWS required)."""
import json
import sqlite3
import threading
import pytest

from pinneapple.lpm.evidence_store import EvidenceStore
from pinneapple.lpm.approval_service import authorize_campaign


def test_campaign_acl(monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_LPM_CAMPAIGN_ACL",json.dumps({
        "alice":{"campaign-a":["approve"]},
        "bob":{"campaign-a":["ingest"],"campaign-b":["approve"]},
    }))
    assert authorize_campaign({"sub":"alice"},"campaign-a","approve")=="alice"
    with pytest.raises(PermissionError):
        authorize_campaign({"sub":"alice"},"campaign-b","approve")
    with pytest.raises(PermissionError):
        authorize_campaign({"sub":"alice"},"campaign-a","ingest")
    with pytest.raises(PermissionError):
        authorize_campaign({"sub":"bob"},"campaign-a","approve")


def test_audit_chain_detects_mutation(tmp_path):
    store=EvidenceStore(tmp_path/"approval.sqlite","object-lock-bucket")
    with store.transaction() as db:
        store.audit(db,"a","alice","evidence.ingested",{"sha256":"abc"})
        store.audit(db,"a","bob","approval.granted",{"sha256":"def"})
    assert store.verify_audit_chain()
    with store.transaction() as db:
        db.execute("UPDATE audit SET actor='mallory' WHERE sequence=1")
    assert not store.verify_audit_chain()


def test_unique_campaign_approval_and_rollback(tmp_path):
    store=EvidenceStore(tmp_path/"approval.sqlite","object-lock-bucket")
    with store.transaction() as db:
        db.execute("INSERT INTO approvals VALUES(?,?,?,?,?)",("c","hash","evidence","alice","now"))
    with pytest.raises(sqlite3.IntegrityError):
        with store.transaction() as db:
            db.execute("INSERT INTO approvals VALUES(?,?,?,?,?)",("c","other","other","bob","later"))
    with store.connection() as db:
        assert db.execute("SELECT COUNT(*) FROM approvals").fetchone()[0]==1


def test_concurrent_approval_serialized(tmp_path):
    store=EvidenceStore(tmp_path/"approval.sqlite","object-lock-bucket")
    results=[]
    barrier=threading.Barrier(2)
    def attempt(actor):
        barrier.wait()
        try:
            with store.transaction() as db:
                db.execute("INSERT INTO approvals VALUES(?,?,?,?,?)",("c",actor,"e",actor,"now"))
            results.append("ok")
        except sqlite3.IntegrityError:
            results.append("conflict")
    a=threading.Thread(target=attempt,args=("alice",))
    b=threading.Thread(target=attempt,args=("bob",))
    a.start();b.start();a.join();b.join()
    assert sorted(results)==["conflict","ok"]
