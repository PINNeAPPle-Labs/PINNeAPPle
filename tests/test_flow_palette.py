"""The node palette model (pinneapple_flow.palette): sections in workflow order, every curated node placed, the
generated library calls in the collapsed advanced section, search ranking, per-user favourites in the store."""
import pytest

from pinneapple_flow import REGISTRY, palette


def test_sections_follow_the_engineering_workflow():
    ids = [s["id"] for s in palette.sections()]
    assert ids == ["problem", "equations", "solvers", "physics_ai", "surrogates", "data", "inverse", "validation",
                   "optimization", "twin", "viz", "export", "library"]
    assert [s["collapsed"] for s in palette.sections()] == [False] * 12 + [True]


def test_every_curated_node_has_a_workflow_section_and_auto_nodes_are_advanced():
    cat = REGISTRY.catalog()
    for e in cat:
        p = e["palette"]
        if e["type"].startswith("auto."):
            assert p["section"] == "library" and not p["curated"] and p["group"]
        elif REGISTRY.get(e["type"]).fn.__module__.startswith("pinneapple_flow.") and e["category"] != "Custom":
            assert p["section"] != "library", e["type"]
            assert e["type"] in palette.SECTION_OF, f"{e['type']}: add it to pinneapple_flow.palette.SECTION_OF"
        else:
            continue                                              # node types other tests register
        assert p["title"] and "_" not in p["title"]
        assert len(p["summary"]) <= 110 and "[checked" not in p["summary"]
    lay = palette.layout(cat)
    assert sum(s["count"] for s in lay) == len(cat)
    assert lay[2]["types"][0] == "solver.classical" and lay[3]["types"][0] == "train.pinn"


def test_titles_summaries_and_install_hints():
    assert palette.humanize("generate_eddy_current_synthetic") == "Generate eddy current synthetic"
    assert palette.humanize("cosine_lr") == "Cosine LR"
    assert palette.summary("``f`` does a thing. More text.") == "F does a thing."
    assert palette.install_hint("needs fmpy (ModuleNotFoundError: No module named 'fmpy')") == "pip install fmpy"
    assert palette.install_hint("needs bpy") == "pip install bpy"
    assert "OpenFOAM" in palette.install_hint("OpenFOAM is not installed (blockMesh / icoFoam not on PATH)")


@pytest.mark.parametrize("q,first", [("pinn", "train.pinn"), ("kalman", "assimilate.filter"),
                                     ("fno", "surrogate.train"), ("heat", "solver.classical"),
                                     ("cosine lr", "auto.analysis.cost.cosine_lr")])
def test_search_ranks_the_node_an_engineer_means_first(q, first):
    hits = palette.search(q)
    assert hits and hits[0]["type"] == first


def test_search_puts_curated_nodes_above_library_calls():
    for q in ("pinn", "solver", "sensitivity", "twin", "rom"):
        kinds = [e["palette"]["curated"] for e in palette.search(q)]
        assert kinds == sorted(kinds, reverse=True), q
    assert palette.search("") == [] and palette.search("zzzz-no-such-node") == []


def test_favourites_and_recent_are_stored_per_user(tmp_path):
    from pinneapple_flow.service import FlowService
    svc = FlowService(db_url=f"sqlite:///{tmp_path}/f.db")
    out = svc.set_palette_prefs("user:ana", {"favourites": ["train.pinn", "nope", "train.pinn", 3], "recent": ["viz.plot"]})
    assert out == {"favourites": ["train.pinn"], "recent": ["viz.plot"]}
    assert svc.palette("user:ana")["favourites"] == ["train.pinn"]
    assert svc.palette("local")["favourites"] == []
    svc.set_palette_prefs("user:ana", {"recent": [f"x{i}" for i in range(40)] + ["viz.plot"] * 3})
    assert svc.palette_prefs("user:ana")["recent"] == ["viz.plot"]
    from pinneapple_flow import FlowError
    with pytest.raises(FlowError):
        svc.set_palette_prefs("user:ana", {"favourites": "train.pinn"})
    assert svc.palette_search("kalman")[0] == {"type": "assimilate.filter", "section": "twin"}
