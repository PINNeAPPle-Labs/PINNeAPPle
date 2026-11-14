from pinneapple_atlas.store import AtlasStore
from pinneapple_atlas.reconcile import reconcile_dois, doi_equivalences
from pinneapple_atlas.cli import main


def test_only_identical_dois_match(tmp_path):
    db = AtlasStore(tmp_path / "atlas.sqlite3")
    db.ingest([
        {"source": "openalex", "id": "W1", "title": "Paper A", "url": "https://openalex.org/W1", "doi": "https://doi.org/10.1000/X"},
        {"source": "crossref", "id": "10.1000/x", "title": "Paper A", "url": "https://doi.org/10.1000/x", "doi": "10.1000/x"},
        {"source": "openalex", "id": "W2", "title": "Paper A", "url": "https://openalex.org/W2", "doi": "10.1000/y"},
    ])
    assert len(doi_equivalences(db)) == 1
    assert reconcile_dois(db) == 1
    assert reconcile_dois(db) == 1
    assert len(db.graph()["edges"]) == 1
    db.close()


def test_cli_graph(tmp_path, capsys):
    main(["--db", str(tmp_path / "atlas.sqlite3"), "graph"])
    assert '"nodes": []' in capsys.readouterr().out
