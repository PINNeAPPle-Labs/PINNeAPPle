"""The workbench database: a local SQLite file, or a PostgreSQL server you already run.

It keeps everything needed to resume, replay and later fine-tune:

- ``sessions``: settings and the lead's message history;
- ``events``: every step shown in the UI (messages, tool calls and results, approvals, plans, files);
- ``model_calls``: every request to the model with its exact input messages, the tool specs and the reply, the
  agent, timing and token counts. These are the fine-tuning examples (see :mod:`.finetune`);
- ``feedback``: your rating of a turn (+1 / -1) and an optional note, used to filter the training data.

``Store("~/.pinneapple/workbench/sessions.sqlite")`` or ``Store("postgresql://user:pass@host:5432/db")`` (needs
``pip install "psycopg[binary]"``). The same code and schema run on both.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

__all__ = ["Store", "DEFAULT_DB"]

DEFAULT_DB = os.environ.get("PINNEAPPLE_WORKBENCH_DB") or str(Path.home() / ".pinneapple" / "workbench"
                                                              / "sessions.sqlite")

_SCHEMA = [
    """CREATE TABLE IF NOT EXISTS sessions (
         id TEXT PRIMARY KEY, title TEXT, workspace TEXT, settings TEXT, history TEXT, created REAL, updated REAL)""",
    """CREATE TABLE IF NOT EXISTS events (
         session_id TEXT, seq INTEGER, ts REAL, data TEXT, PRIMARY KEY (session_id, seq))""",
    """CREATE TABLE IF NOT EXISTS model_calls (
         id TEXT PRIMARY KEY, session_id TEXT, turn_id TEXT, run_id TEXT, agent TEXT, model TEXT, ts REAL,
         messages TEXT, tools TEXT, reply TEXT, seconds REAL, prompt_tokens INTEGER, completion_tokens INTEGER,
         ok INTEGER)""",
    """CREATE TABLE IF NOT EXISTS feedback (
         session_id TEXT, turn_id TEXT, rating INTEGER, note TEXT, ts REAL, PRIMARY KEY (session_id, turn_id))""",
    "CREATE INDEX IF NOT EXISTS model_calls_turn ON model_calls (session_id, turn_id)",
]


class Store:
    def __init__(self, url: str | Path = DEFAULT_DB):
        self.url = str(url)
        self.postgres = self.url.startswith(("postgres://", "postgresql://"))
        self._lock = threading.RLock()
        if self.postgres:
            try:
                import psycopg
            except ImportError as e:
                raise ImportError('a PostgreSQL database needs psycopg: pip install "psycopg[binary]"') from e
            self._db = psycopg.connect(self.url, autocommit=True)
        else:
            path = Path(self.url).expanduser()
            if self.url != ":memory:":
                path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(":memory:" if self.url == ":memory:" else str(path), check_same_thread=False)
        for stmt in _SCHEMA:
            self._exec(stmt)

    # -- SQL on either engine ------------------------------------------------------------------------------------
    def _exec(self, sql: str, params: tuple = ()) -> list[tuple]:
        if self.postgres:
            sql = sql.replace("?", "%s")
        with self._lock:
            cur = self._db.cursor()
            cur.execute(sql, params)
            rows = cur.fetchall() if cur.description else []
            if not self.postgres:
                self._db.commit()
            return rows

    # -- sessions ------------------------------------------------------------------------------------------------
    def create(self, workspace: str, settings: dict, title: str = "") -> str:
        sid, now = uuid.uuid4().hex[:12], time.time()
        self._exec("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                   (sid, title or "New session", workspace, json.dumps(settings), "[]", now, now))
        return sid

    def get(self, sid: str) -> dict | None:
        rows = self._exec("SELECT id,title,workspace,settings,history,created,updated FROM sessions WHERE id=?",
                          (sid,))
        if not rows:
            return None
        r = rows[0]
        return {"id": r[0], "title": r[1], "workspace": r[2], "settings": json.loads(r[3]),
                "history": json.loads(r[4]), "created": r[5], "updated": r[6]}

    def list(self) -> list[dict]:
        rows = self._exec("SELECT id,title,workspace,updated FROM sessions ORDER BY updated DESC")
        return [{"id": r[0], "title": r[1], "workspace": r[2], "updated": r[3]} for r in rows]

    def update(self, sid: str, *, title: str | None = None, history: list | None = None,
               settings: dict | None = None) -> None:
        sets, vals = ["updated=?"], [time.time()]
        for col, val in (("title", title), ("history", history), ("settings", settings)):
            if val is not None:
                sets.append(f"{col}=?")
                vals.append(val if col == "title" else json.dumps(val))
        self._exec(f"UPDATE sessions SET {','.join(sets)} WHERE id=?", (*vals, sid))

    def delete(self, sid: str) -> None:
        for table in ("sessions", "events", "model_calls", "feedback"):
            self._exec(f"DELETE FROM {table} WHERE {'id' if table == 'sessions' else 'session_id'}=?", (sid,))

    # -- events --------------------------------------------------------------------------------------------------
    def add_event(self, sid: str, event: dict) -> int:
        with self._lock:
            seq = self._exec("SELECT COALESCE(MAX(seq),0)+1 FROM events WHERE session_id=?", (sid,))[0][0]
            self._exec("INSERT INTO events VALUES (?,?,?,?)", (sid, seq, time.time(), json.dumps(event)))
        return seq

    def events(self, sid: str, after: int = 0) -> list[dict]:
        rows = self._exec("SELECT seq,ts,data FROM events WHERE session_id=? AND seq>? ORDER BY seq", (sid, after))
        return [{"seq": r[0], "ts": r[1], **json.loads(r[2])} for r in rows]

    # -- training data -------------------------------------------------------------------------------------------
    def add_model_call(self, sid: str, call: dict) -> None:
        self._exec("INSERT INTO model_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   (uuid.uuid4().hex, sid, call.get("turn_id", ""), call.get("run_id", ""), call.get("agent", ""),
                    call.get("model", ""), time.time(), json.dumps(call["messages"]), json.dumps(call.get("tools", [])),
                    json.dumps(call.get("reply", {})), float(call.get("seconds") or 0),
                    int(call.get("prompt_tokens") or 0), int(call.get("completion_tokens") or 0),
                    1 if call.get("ok", True) else 0))

    def model_calls(self, sid: str | None = None) -> list[dict]:
        sql = ("SELECT id,session_id,turn_id,run_id,agent,model,ts,messages,tools,reply,seconds,prompt_tokens,"
               "completion_tokens,ok FROM model_calls")
        rows = self._exec(sql + (" WHERE session_id=? ORDER BY ts" if sid else " ORDER BY ts"), (sid,) if sid else ())
        keys = ("id", "session_id", "turn_id", "run_id", "agent", "model", "ts", "messages", "tools", "reply",
                "seconds", "prompt_tokens", "completion_tokens", "ok")
        out = []
        for r in rows:
            d = dict(zip(keys, r, strict=True))
            for k in ("messages", "tools", "reply"):
                d[k] = json.loads(d[k])
            d["ok"] = bool(d["ok"])
            out.append(d)
        return out

    def set_feedback(self, sid: str, turn_id: str, rating: int, note: str = "") -> None:
        rating = 1 if rating > 0 else -1 if rating < 0 else 0
        self._exec("DELETE FROM feedback WHERE session_id=? AND turn_id=?", (sid, turn_id))
        self._exec("INSERT INTO feedback VALUES (?,?,?,?,?)", (sid, turn_id, rating, note, time.time()))

    def feedback(self) -> dict[tuple[str, str], dict]:
        rows = self._exec("SELECT session_id,turn_id,rating,note FROM feedback")
        return {(r[0], r[1]): {"rating": r[2], "note": r[3]} for r in rows}

    def stats(self) -> dict:
        n = lambda sql: self._exec(sql)[0][0]  # noqa: E731
        return {"engine": "postgresql" if self.postgres else "sqlite", "sessions": n("SELECT COUNT(*) FROM sessions"),
                "model_calls": n("SELECT COUNT(*) FROM model_calls"),
                "rated_turns": n("SELECT COUNT(*) FROM feedback WHERE rating<>0"),
                "good_turns": n("SELECT COUNT(*) FROM feedback WHERE rating>0")}
