"""Text2Part app (apps/text2part): routes, metering, an STL from a recipe (LLM stubbed), and one real local-LLM run
when an Ollama server with llama3.2:3b is reachable."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("trimesh")
from fastapi.testclient import TestClient  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _ollama_has(model="llama3.2:3b"):
    try:
        import requests
        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        return r.ok and any(m["name"] in (model, f"{model}:latest") for m in r.json().get("models", []))
    except Exception:
        return False


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    os.environ["T2P_DATA_DIR"] = str(tmp_path_factory.mktemp("t2p"))
    os.environ["T2P_FREE_GENERATIONS"] = "3"
    sys.path.insert(0, str(ROOT / "apps" / "text2part"))
    from text2part.api import app
    return TestClient(app)


def test_health_and_meta(client):
    assert client.get("/health").json()["status"] == "ok"
    m = client.get("/api/meta").json()
    assert m["llm"]["provider"] == "ollama" and set(m["tiers"]) == {"quick", "engineering"}


def test_generate_builds_and_measures_the_recipe(client, monkeypatch):
    import pinneapple_llm as pl
    from pinneapple_llm.cad_draft import CadRecipeResult

    box = CadRecipeResult(kind="mesh_recipe", recipe={"name": "box", "params": {"extents": [40, 20, 5]}},
                          reasoning="a 40 x 20 x 5 plate")
    monkeypatch.setattr(pl, "draft_part", lambda *a, **k: (box, 2))
    r = client.post("/v1/generate", json={"description": "a 40 by 20 by 5 mm plate"})
    assert r.status_code == 200, r.text
    assert r.content.startswith(b"solid") or len(r.content) > 84          # ASCII or binary STL
    m = json.loads(r.headers["x-measurements"])
    assert m["extents"] == pytest.approx([40, 20, 5]) and m["volume"] == pytest.approx(4000.0)
    assert r.headers["x-attempts-used"] == "2" and json.loads(r.headers["x-reasoning"]) == "a 40 x 20 x 5 plate"


def test_refused_drafts_and_no_server_are_reported(client, monkeypatch):
    import pinneapple_llm as pl

    def refuse(*a, **k):
        raise ValueError("no valid part after 4 attempts; last: unknown builder 'gear'")
    monkeypatch.setattr(pl, "draft_part", refuse)
    assert client.post("/v1/generate", json={"description": "a gear"}).status_code == 502

    def down(*a, **k):
        raise ConnectionError("could not reach a local Ollama server")
    monkeypatch.setattr(pl, "draft_part", down)
    r = client.post("/v1/generate", json={"description": "a gear"})
    assert r.status_code == 503 and "ollama" in r.json()["detail"]
    assert client.post("/v1/generate", json={"description": "a gear"}).status_code == 429    # 3 free per month
    assert client.get("/api/usage").json()["units"]["generations"]["used"] == 3


@pytest.mark.skipif(not _ollama_has(), reason="no local Ollama server with llama3.2:3b")
def test_real_local_llm_box(tmp_path):
    import pinneapple_llm as pl

    result, attempts = pl.draft_part("a box 30 mm by 20 mm by 10 mm", provider="ollama", model="llama3.2:3b")
    m = pl.measure_part(result)
    assert attempts >= 1 and m["volume"] > 0
