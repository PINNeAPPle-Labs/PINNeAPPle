from pinneapple_atlas.store import AtlasStore, canonical_doi
from pinneapple_atlas.sync import sync_public_query


def test_doi_normalization():
    assert canonical_doi("https://doi.org/10.1234/ABC") == "10.1234/abc"
    assert canonical_doi("invalid") is None


def test_idempotent_ingestion_and_doi_matching(tmp_path):
    db = AtlasStore(tmp_path / "atlas.db")
    item = {"source": "openalex", "id": "W1", "title": "FNO",
            "url": "https://openalex.org/W1", "doi": "https://doi.org/10.1234/ABC"}
    assert db.ingest([item]) == 1
    assert db.ingest([item]) == 1
    assert len(db.list_records()) == 1
    assert db.doi_matches("10.1234/abc")[0]["external_id"] == "W1"
    db.close()


def test_evidence_relations_and_graph(tmp_path):
    db = AtlasStore(tmp_path / "atlas.db")
    db.ingest([{"source": "openalex", "id": "W1", "title": "Paper", "url": "https://openalex.org/W1"},
               {"source": "github", "id": "R1", "title": "Code", "url": "https://github.com/a/b"}])
    db.relate("openalex", "W1", "github", "R1", "has_implementation", "https://openalex.org/W1")
    db.relate("openalex", "W1", "github", "R1", "has_implementation", "https://openalex.org/W1")
    graph = db.graph()
    assert len(graph["nodes"]) == 2
    assert len(graph["edges"]) == 1
    db.close()


def test_sync_with_fake_client(tmp_path):
    class Client:
        def openalex(self, query, limit):
            return [{"source": "openalex", "id": "W1", "title": "Paper", "url": "https://openalex.org/W1"}]
        def crossref(self, query, limit):
            return []
        def github_repositories(self, query, limit):
            return []
    db = AtlasStore(tmp_path / "atlas.db")
    assert sync_public_query(db, "physics ai", client=Client()) == {"openalex": 1, "crossref": 0, "github": 0}
    db.close()
