"""SQLite helpers shared by the service stores: one connection per thread, WAL, immediate transactions."""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path


def connect(db_path: str | Path) -> sqlite3.Connection:
    """A connection in autocommit mode with WAL and a busy timeout (shared by the service stores)."""
    conn = sqlite3.connect(str(db_path), timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection):
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")



class LocalDB:
    """One connection per thread to the same file (a connection must not be shared by concurrent transactions)."""

    def __init__(self, db_path: str | Path, schema: str = ""):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        if schema:
            self.conn.executescript(schema)

    @property
    def conn(self) -> sqlite3.Connection:
        c = getattr(self._local, "conn", None)
        if c is None:
            c = self._local.conn = connect(self.path)
        return c
