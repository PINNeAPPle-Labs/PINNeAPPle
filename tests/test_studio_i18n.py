"""The Studio's gallery and glossary in Portuguese and English: every gallery entry has a title and a description in
both languages, every glossary term has its English term and definition, and the gallery takes the Portuguese title
and the English fallbacks from the map of the library (``docs/pt/_mapa``)."""
from __future__ import annotations

import json
from pathlib import Path

from pinneapple_studio import gallery
from pinneapple_studio import prose as P

ROOT = Path(__file__).resolve().parents[1]


def test_every_gallery_entry_has_both_languages(monkeypatch):
    monkeypatch.setattr(gallery, "_flow_rows", lambda: ({}, [], {}))
    items = gallery.entries(ROOT)
    assert items
    missing = {k: [e["id"] for e in items if not str(e.get(f) or "").strip()]
               for k, f in (("title", "title"), ("title_pt", "title_pt"), ("description", "description"),
                            ("pt", "pt"))}
    assert missing == {"title": [], "title_pt": [], "description": [], "pt": []}


def test_every_glossary_term_is_in_both_languages():
    g = P.load(ROOT).glossary
    assert len(g) >= 53
    assert [x["termo"] for x in g if not all(str(x.get(k) or "").strip()
                                              for k in ("termo", "definicao", "term", "definition"))] == []


def test_titles_and_english_fallbacks_come_from_the_map(tmp_path, monkeypatch):
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "demo"\nversion = "0.0.1"\n', encoding="utf-8")
    monkeypatch.setenv("PINNEAPPLE_STUDIO_CACHE", str(tmp_path / "cache"))
    ex = tmp_path / "examples" / "viga"
    ex.mkdir(parents=True)
    (ex / "README.md").write_text("# Viga em balanço\n\nExecução de referência.\n", encoding="utf-8")
    (ex / "run.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "examples" / "fins").mkdir()
    (tmp_path / "examples" / "fins" / "README.md").write_text("# Fins\n\nA finned heat sink.\n", encoding="utf-8")
    d = tmp_path / P.PROSE_DIR
    d.mkdir(parents=True)
    (d / "_apps_exemplos.json").write_text(json.dumps({
        "examples": {"viga": "Viga de referência.", "fins": "Um dissipador aletado."},
        "titulos": {"example:viga": "Viga em balanço", "example:fins": "Aletas"},
        "en": {"titulos": {"example:viga": "Cantilever beam"}, "descricoes": {"example:viga": "A reference run."}},
    }), encoding="utf-8")
    monkeypatch.setattr(gallery, "_flow_rows", lambda: ({}, [], {}))
    e = {x["id"]: x for x in gallery.entries(tmp_path)}
    assert (e["example:viga"]["title"], e["example:viga"]["title_pt"]) == ("Cantilever beam", "Viga em balanço")
    assert e["example:viga"]["description"] == "A reference run." and e["example:viga"]["pt"] == "Viga de referência."
    assert (e["example:fins"]["title"], e["example:fins"]["title_pt"]) == ("Fins", "Aletas")   # its own English kept
    assert e["example:fins"]["description"].startswith("A finned heat sink")
