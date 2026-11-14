# Global Physics AI Atlas: metadata prototype

This is an experimental, **metadata-only** global scientific index, not a verified capability map. It does not claim that a linked publication has a working PINNeAPPle implementation.

## Commands

From the repository root, with Python and pytest available:

```bash
python -m pytest tests/test_atlas_core.py tests/test_atlas_connectors.py tests/test_atlas_store.py tests/test_atlas_reconcile.py -q
python -m pinneapple_atlas.cli --db atlas.sqlite3 sync "physics informed neural networks" --limit 10
python -m pinneapple_atlas.cli --db atlas.sqlite3 reconcile
python -m pinneapple_atlas.cli --db atlas.sqlite3 graph --limit 100
```

Sync contacts public OpenAlex, Crossref and GitHub APIs on demand. These services may enforce quotas or return errors. No external data is bundled or automatically fetched at import time. Repeating sync upserts records. DOI reconciliation only links records with an identical normalized DOI, never similarity of title or an unverified LLM suggestion.

**Limitations:** Crossref and OpenAlex are not full-text indexes here; the graph is metadata only. Current SQLite storage has no schema migrations or automatic retry, checkpointing, scheduled refresh, licensing review, production concurrency handling, or Engineering Studio UI. A matching DOI indicates publication identity, not scientific reproducibility or software equivalence.
