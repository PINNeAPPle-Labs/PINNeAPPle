"""API keys for app servers: created once (the plain key is shown only then), stored as SHA-256, revocable.

    keys = APIKeyStore("/data/service.sqlite", app="campaign")
    created = keys.create("ci pipeline")                 # {"id": ..., "key": "pk_...", ...}; keep created["key"]
    keys.verify(created["key"])["id"] == created["id"]
    keys.revoke(created["id"])

Keys are 32 random bytes, so a plain SHA-256 is enough to store them (no password stretching needed); lookup is by
hash. Clients send the key in the ``X-API-Key`` header.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any

from .db import LocalDB

PREFIX = "pk_"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY, app TEXT, name TEXT, hash TEXT UNIQUE NOT NULL, hint TEXT,
    scopes TEXT, created REAL, revoked REAL, last_used REAL
);
"""


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


class APIKeyStore:
    def __init__(self, db_path: str | Path, app: str = ""):
        self.app = app
        self._db = LocalDB(db_path, _SCHEMA)

    def create(self, name: str, scopes: list[str] | None = None) -> dict[str, Any]:
        """A new key. The returned ``key`` is the only time the plain key exists outside the client."""
        key = PREFIX + secrets.token_urlsafe(32)
        key_id = secrets.token_hex(4)
        now = time.time()
        self._db.conn.execute("INSERT INTO api_keys(id, app, name, hash, hint, scopes, created) VALUES (?,?,?,?,?,?,?)",
                           (key_id, self.app, str(name)[:200], hash_key(key), key[-4:],
                            json.dumps(scopes or []), now))
        return {**self.get(key_id), "key": key}

    def get(self, key_id: str) -> dict[str, Any]:
        row = self._db.conn.execute("SELECT * FROM api_keys WHERE id=?", (key_id,)).fetchone()
        if row is None:
            raise KeyError(f"unknown API key id {key_id!r}")
        return _public(row)

    def list(self, include_revoked: bool = False) -> list[dict[str, Any]]:
        sql = "SELECT * FROM api_keys WHERE app=?" + ("" if include_revoked else " AND revoked IS NULL")
        return [_public(r) for r in self._db.conn.execute(sql + " ORDER BY created DESC", (self.app,))]

    def revoke(self, key_id: str) -> dict[str, Any]:
        self.get(key_id)
        self._db.conn.execute("UPDATE api_keys SET revoked=? WHERE id=? AND revoked IS NULL", (time.time(), key_id))
        return self.get(key_id)

    def verify(self, key: str | None) -> dict[str, Any] | None:
        """The key's record if it is valid for this app and not revoked, else None."""
        if not key or not key.startswith(PREFIX) or len(key) > 200:
            return None
        row = self._db.conn.execute("SELECT * FROM api_keys WHERE hash=? AND app=? AND revoked IS NULL",
                                 (hash_key(key), self.app)).fetchone()
        if row is None:
            return None
        self._db.conn.execute("UPDATE api_keys SET last_used=? WHERE id=?", (time.time(), row["id"]))
        return _public(row)


def _public(row) -> dict[str, Any]:
    d = dict(row)
    d.pop("hash", None)
    d["scopes"] = json.loads(d.get("scopes") or "[]")
    d["hint"] = "…" + (d.get("hint") or "")
    return d
