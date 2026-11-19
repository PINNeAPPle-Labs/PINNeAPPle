"""Durable evidence catalog and serialized approvals.

Production evidence is stored in an S3 bucket with Object Lock COMPLIANCE
retention enabled. SQLite provides local transactional coordination and an
append-only, hash-chained audit trail; deploy with durable database storage.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import uuid

def utcnow():
    return datetime.now(timezone.utc).isoformat()

def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()

class EvidenceStore:
    def __init__(self,db_path=None,bucket=None):
        db_path=db_path or os.environ.get("PINNEAPPLE_LPM_APPROVAL_DB")
        self.bucket=bucket or os.environ.get("PINNEAPPLE_LPM_EVIDENCE_BUCKET")
        if not db_path or not self.bucket:
            raise RuntimeError("Durable approval DB and immutable evidence bucket required")
        self.db_path=Path(db_path)
        self.db_path.parent.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS evidence (
              id TEXT PRIMARY KEY, campaign TEXT NOT NULL, sha256 TEXT NOT NULL,
              s3_key TEXT NOT NULL UNIQUE, version_id TEXT NOT NULL,
              object_lock_until TEXT NOT NULL, uploaded_by TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS approvals (
              campaign TEXT PRIMARY KEY, checkpoint_sha256 TEXT NOT NULL,
              evidence_id TEXT NOT NULL, reviewer TEXT NOT NULL,
              created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS audit (
              sequence INTEGER PRIMARY KEY AUTOINCREMENT,
              timestamp TEXT NOT NULL, campaign TEXT NOT NULL,
              actor TEXT NOT NULL, action TEXT NOT NULL,
              detail TEXT NOT NULL, prev_hash TEXT NOT NULL, hash TEXT NOT NULL
            );
            """)

    @contextmanager
    def connection(self):
        db=sqlite3.connect(self.db_path,timeout=30,isolation_level=None)
        db.execute("PRAGMA busy_timeout=30000")
        db.execute("PRAGMA journal_mode=WAL")
        try:yield db
        finally:db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            try:
                yield db
                db.execute("COMMIT")
            except BaseException:
                db.execute("ROLLBACK")
                raise

    @staticmethod
    def audit(db,campaign,actor,action,detail):
        previous=db.execute("SELECT hash FROM audit ORDER BY sequence DESC LIMIT 1").fetchone()
        prev=previous[0] if previous else "0"*64
        timestamp=utcnow()
        payload=json.dumps(dict(timestamp=timestamp,campaign=campaign,actor=actor,
                                action=action,detail=detail,prev_hash=prev),sort_keys=True)
        digest=sha256_bytes(payload.encode())
        db.execute("INSERT INTO audit(timestamp,campaign,actor,action,detail,prev_hash,hash) VALUES(?,?,?,?,?,?,?)",
                   (timestamp,campaign,actor,action,json.dumps(detail,sort_keys=True),prev,digest))
        return digest

    def client(self):
        import boto3
        return boto3.client("s3")

    def ingest(self,campaign,actor,blob,retention_days=365):
        if len(blob)>100*1024*1024:raise ValueError("Evidence exceeds 100 MiB")
        if retention_days<365:raise ValueError("At least 365 days retention required")
        digest=sha256_bytes(blob)
        key=f"lpm/{campaign}/{uuid.uuid4().hex}/{digest}.npz"
        from datetime import timedelta
        until=datetime.now(timezone.utc)+timedelta(days=retention_days)
        s3=self.client()
        # Bucket must be configured with S3 Object Lock and versioning.
        response=s3.put_object(Bucket=self.bucket,Key=key,Body=blob,
            ContentType="application/octet-stream",ChecksumSHA256=__import__("base64").b64encode(bytes.fromhex(digest)).decode(),
            ObjectLockMode="COMPLIANCE",ObjectLockRetainUntilDate=until)
        version=response.get("VersionId")
        if not version:raise RuntimeError("S3 Object Lock bucket must return a version ID")
        head=s3.head_object(Bucket=self.bucket,Key=key,VersionId=version)
        if head.get("ObjectLockMode")!="COMPLIANCE":
            raise RuntimeError("Evidence was not stored with COMPLIANCE retention")
        evidence_id=uuid.uuid4().hex
        with self.transaction() as db:
            db.execute("INSERT INTO evidence VALUES(?,?,?,?,?,?,?,?)",
                       (evidence_id,campaign,digest,key,version,until.isoformat(),actor,utcnow()))
            self.audit(db,campaign,actor,"evidence.ingested",dict(evidence_id=evidence_id,sha256=digest,version_id=version))
        return dict(id=evidence_id,sha256=digest,version_id=version)

    def load(self,db,campaign,evidence_id):
        row=db.execute("SELECT sha256,s3_key,version_id FROM evidence WHERE id=? AND campaign=?",
                       (evidence_id,campaign)).fetchone()
        if not row:raise PermissionError("Evidence is not registered for this campaign")
        digest,key,version=row
        import io
        blob=self.client().get_object(Bucket=self.bucket,Key=key,VersionId=version)["Body"].read()
        if sha256_bytes(blob)!=digest:raise RuntimeError("Immutable evidence integrity failure")
        return blob,digest

    def verify_audit_chain(self):
        previous="0"*64
        with self.connection() as db:
            for timestamp,campaign,actor,action,detail,prev,digest in db.execute(
                "SELECT timestamp,campaign,actor,action,detail,prev_hash,hash FROM audit ORDER BY sequence"
            ):
                payload=json.dumps(dict(timestamp=timestamp,campaign=campaign,actor=actor,
                    action=action,detail=json.loads(detail),prev_hash=previous),sort_keys=True)
                if prev!=previous or sha256_bytes(payload.encode())!=digest:return False
                previous=digest
        return True
