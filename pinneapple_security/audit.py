"""Tamper-evident audit trail: an append-only JSON-lines log where every record carries the hash of the previous one.

Editing, deleting or reordering any past record breaks the chain from that point, and :meth:`AuditLog.verify` reports
the first bad line. With a ``key`` the chain is an HMAC chain, so someone who can write the file but does not hold the
key can not rebuild a valid chain after tampering (without a key the log detects accidents and naive edits, not a
forger who recomputes every hash).

Each record holds the ALCOA+ attributes regulated research and industry ask of electronic records (attributable,
legible, contemporaneous, original, accurate): who (``actor``), what (``action`` + ``data``), when (UTC time), why
(``reason``), and on which object (``target``). This is a building block for 21 CFR Part 11 §11.10(e) style audit
trails, not a certified system by itself.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from collections.abc import Iterator
from datetime import datetime, timezone
from typing import Any

from .integrity import canonical_json

__all__ = ["AuditLog", "GENESIS"]

GENESIS = "0" * 64


class AuditLog:
    def __init__(self, path: str, key: bytes | None = None):
        self.path = path
        self.key = key
        d = os.path.dirname(os.path.abspath(path))
        os.makedirs(d, exist_ok=True)

    def _hash(self, body: dict[str, Any]) -> str:
        data = canonical_json(body)
        if self.key:
            return hmac.new(self.key, data, hashlib.sha256).hexdigest()
        return hashlib.sha256(data).hexdigest()

    def records(self) -> Iterator[dict[str, Any]]:
        if not os.path.exists(self.path):
            return
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    def _last(self) -> dict[str, Any] | None:
        last = None
        for rec in self.records():
            last = rec
        return last

    def append(self, action: str, actor: str, target: str = "", data: dict[str, Any] | None = None,
               reason: str = "") -> dict[str, Any]:
        if not action or not actor:
            raise ValueError("an audit record needs an action and an actor")
        last = self._last()
        body = {"seq": (last["seq"] + 1) if last else 0, "time": datetime.now(timezone.utc).isoformat(timespec="microseconds"),
                "actor": actor, "action": action, "target": target, "reason": reason, "data": data or {},
                "prev": last["hash"] if last else GENESIS}
        rec = {**body, "hash": self._hash(body)}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        return rec

    def verify(self) -> dict[str, Any]:
        """``ok``, number of records, and the first broken record (``bad_seq``, ``problem``) if any."""
        prev, n = GENESIS, 0
        for i, rec in enumerate(self.records()):
            body = {k: v for k, v in rec.items() if k != "hash"}
            if rec.get("seq") != i:
                return {"ok": False, "n": n, "bad_seq": i, "problem": "record missing or out of order"}
            if rec.get("prev") != prev:
                return {"ok": False, "n": n, "bad_seq": i, "problem": "chain broken (previous record changed or removed)"}
            if not hmac.compare_digest(self._hash(body), rec.get("hash", "")):
                return {"ok": False, "n": n, "bad_seq": i, "problem": "record content changed"}
            prev, n = rec["hash"], n + 1
        return {"ok": True, "n": n, "head": prev}

    def head(self) -> str:
        """Hash of the last record: publish or sign it to anchor the whole log at a point in time."""
        last = self._last()
        return last["hash"] if last else GENESIS

    def query(self, actor: str | None = None, action: str | None = None, target: str | None = None) -> list[dict[str, Any]]:
        return [r for r in self.records() if (actor is None or r["actor"] == actor)
                and (action is None or r["action"] == action) and (target is None or r["target"] == target)]
