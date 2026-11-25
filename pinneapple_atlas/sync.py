"""Explicit, bounded synchronization of public metadata into AtlasStore."""
from .connectors import PublicMetadataClient
from .store import AtlasStore


def sync_public_query(store: AtlasStore, query: str, *, client=None, limit=10):
    if not query or not query.strip():
        raise ValueError("query cannot be blank")
    client = client or PublicMetadataClient()
    summary = {}
    for source, method in (("openalex", client.openalex), ("crossref", client.crossref),
                           ("github", client.github_repositories)):
        records = method(query, limit=limit)
        summary[source] = store.ingest(records)
    return summary
