"""SQLite persistence for externally sourced scientific metadata.

Ingestion is explicit, transactional and idempotent. No network access occurs here.
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
  source TEXT NOT NULL,
  external_id TEXT NOT NULL,
  title TEXT NOT NULL,
  url TEXT NOT NULL,
  doi TEXT,
  metadata TEXT NOT NULL,
  first_seen TEXT NOT NULL,
  last_seen TEXT NOT NULL,
  PRIMARY KEY (source, external_id)
);
CREATE INDEX IF NOT EXISTS idx_records_doi ON records(doi);
CREATE TABLE IF NOT EXISTS relations (
  source TEXT NOT NULL,
  external_id TEXT NOT NULL,
  target_source TEXT NOT NULL,
  target_id TEXT NOT NULL,
  relation TEXT NOT NULL,
  evidence_url TEXT NOT NULL,
  PRIMARY KEY(source, external_id, target_source, target_id, relation, evidence_url)
);
"""


def canonical_doi(value):
    if not value:
        return None
    doi = str(value).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:", "https://dx.doi.org/"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
            break
    return doi if doi.startswith("10.") and "/" in doi else None


class AtlasStore:
    def __init__(self, path):
        self.path = str(path)
        self.connection = sqlite3.connect(self.path)
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript(SCHEMA)

    def close(self):
        self.connection.close()

    def ingest(self, records):
        now = datetime.now(timezone.utc).isoformat()
        accepted = 0
        with self.connection:
            for item in records:
                source = str(item.get("source") or "").strip()
                external_id = str(item.get("id") or "").strip()
                title = str(item.get("title") or "").strip()
                url = str(item.get("url") or "").strip()
                if not all((source, external_id, title, url)):
                    continue
                parsed = urlparse(url)
                if parsed.scheme not in ("http", "https") or not parsed.netloc:
                    continue
                doi = canonical_doi(item.get("doi"))
                self.connection.execute("""INSERT INTO records
                    (source, external_id, title, url, doi, metadata, first_seen, last_seen)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(source, external_id) DO UPDATE SET
                      title=excluded.title, url=excluded.url, doi=excluded.doi,
                      metadata=excluded.metadata, last_seen=excluded.last_seen""",
                    (source, external_id, title, url, doi,
                     json.dumps(item, sort_keys=True, ensure_ascii=False), now, now))
                accepted += 1
        return accepted

    def list_records(self, limit=100):
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit out of range")
        rows = self.connection.execute("SELECT * FROM records ORDER BY source, external_id LIMIT ?", (limit,))
        return [dict(row) for row in rows]

    def doi_matches(self, doi):
        canonical = canonical_doi(doi)
        if not canonical:
            return []
        rows = self.connection.execute("SELECT source, external_id, title, url FROM records WHERE doi=? ORDER BY source", (canonical,))
        return [dict(row) for row in rows]

    def relate(self, source, external_id, target_source, target_id, relation, evidence_url):
        parsed = urlparse(evidence_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("relation requires evidence URL")
        with self.connection:
            self.connection.execute("INSERT OR IGNORE INTO relations VALUES (?, ?, ?, ?, ?, ?)",
                                    (source, external_id, target_source, target_id, relation, evidence_url))

    def graph(self, limit=100):
        records = self.list_records(limit)
        keys = {(r["source"], r["external_id"]) for r in records}
        nodes = [{"id": f'{r["source"]}:{r["external_id"]}', "type": "scientific_record",
                  "title": r["title"], "url": r["url"], "doi": r["doi"]} for r in records]
        edges = [{"source": f'{r["source"]}:{r["external_id"]}',
                  "target": f'{r["target_source"]}:{r["target_id"]}', "relation": r["relation"],
                  "evidence_url": r["evidence_url"]}
                 for r in self.connection.execute("SELECT * FROM relations")
                 if (r["source"], r["external_id"]) in keys and
                    (r["target_source"], r["target_id"]) in keys]
        return {"nodes": nodes, "edges": edges}
