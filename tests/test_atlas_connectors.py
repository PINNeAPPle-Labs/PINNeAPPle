import pytest
from pinneapple_atlas.connectors import PublicMetadataClient
from pinneapple_atlas.inventory import discover_candidates


def test_openalex_normalization_and_limit():
    urls = []
    def fetch(url):
        urls.append(url)
        return {"results": [{"id": "https://openalex.org/W1", "title": "A", "doi": "https://doi.org/10.1/a", "publication_year": 2024}]}
    rows = PublicMetadataClient(fetch=fetch).openalex("neural operator", limit=2)
    assert rows[0]["source"] == "openalex"
    assert "per-page=2" in urls[0]


def test_crossref_and_github_normalization():
    client = PublicMetadataClient(fetch=lambda url: {"message": {"items": [{"DOI": "10.1/x", "title": ["Paper"]}]}})
    assert client.crossref("physics")[0]["doi"] == "https://doi.org/10.1/x"
    client = PublicMetadataClient(fetch=lambda url: {"items": [{"id": 3, "full_name": "a/b", "html_url": "https://github.com/a/b"}]})
    assert client.github_repositories("PINN")[0]["title"] == "a/b"


def test_inventory_never_asserts_implementation():
    client = PublicMetadataClient(fetch=lambda url: {"truncated": False, "tree": [
        {"path": "pinneapple_neural/__init__.py", "type": "blob"},
        {"path": "pinneapple_neural/model.py", "type": "blob"},
        {"path": "docs/readme.md", "type": "blob"}]})
    inventory = client.github_inventory()
    candidates = discover_candidates(inventory)
    assert len(candidates) == 1
    assert candidates[0]["status"] == "unknown"


def test_invalid_limit():
    with pytest.raises(ValueError):
        PublicMetadataClient(fetch=lambda _: {}).openalex("x", limit=101)
