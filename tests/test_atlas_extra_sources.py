import json
import pytest
from pinneapple_atlas.extra_sources import AdditionalSources
from pinneapple_atlas.ontology import ScientificRelation


def test_arxiv_atom():
    xml = b'<feed xmlns="http://www.w3.org/2005/Atom"><entry><id>https://arxiv.org/abs/1234.5678</id><title>Physics AI</title></entry></feed>'
    assert AdditionalSources(fetch=lambda *_: xml).arxiv("physics")[0]["title"] == "Physics AI"


def test_hf_dataset():
    rows = AdditionalSources(fetch=lambda *_: json.dumps([{"id": "org/dataset"}]).encode()).huggingface_datasets("cfd")
    assert rows[0]["url"] == "https://huggingface.co/datasets/org/dataset"


def test_zenodo():
    data = {"hits": {"hits": [{"id": 123, "metadata": {"title": "Simulation"}, "links": {"html": "https://zenodo.org/records/123"}}]}}
    assert AdditionalSources(fetch=lambda *_: json.dumps(data).encode()).zenodo("fluid")[0]["id"] == "123"


def test_relation_requires_evidence():
    with pytest.raises(ValueError):
        ScientificRelation("paper:1", "method:2", "invented", "https://example.org")
