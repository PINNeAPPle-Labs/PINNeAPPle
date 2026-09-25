"""Public-resource catalog: consistent entries and the same license gate as the Noether bridge."""
import re
import warnings

import pytest

from pinneapple_catalog import fetch, get_resource, list_resources, require_allowed
from pinneapple_neural._licencas import ResearchOnlyError, ResearchOnlyWarning

ARXIV = re.compile(r"arXiv:\d{4}\.\d{4,5}")


def test_every_resource_has_a_source_and_a_license():
    for res in list_resources():
        assert res.url.startswith("https://"), res.id
        assert res.license, res.id
        if res.paper and "arXiv" in res.paper:
            assert ARXIV.search(res.paper), res.id


def test_resources_requested_on_2026_09_24_are_present():
    for rid in ["pdebench", "airfrans", "flowbench", "blastnet", "the_well", "realpdebench",
                "meshgraphnets_data", "drivaernet", "mechanical_mnist", "materials_project", "oqmd",
                "open_catalyst", "era5", "usgs_earthquakes", "abc", "fusion360_gallery", "mfcad",
                "mfcadpp", "fabricad", "ucsm", "nist_step_pmi", "deepcad", "cad_steps", "pie2f_cad",
                "thingi10k", "shapenet", "objaverse", "huhb3d", "pressnet", "plaid_rotor37", "geopt",
                "ab_upt", "dpot", "transolver", "domino", "meshgraphnet", "oformer", "disco", "hyena_no",
                "rno", "multigrid_no", "deq_no", "pfem", "fen", "realpdebench_models", "lgno",
                "pinto_kovasznay", "mantle_convection_no", "mace", "aimnet2", "sevennet"]:
        get_resource(rid)


def test_unknown_or_missing_license_counts_as_research_only():
    assert get_resource("pressnet").research_only  # no LICENSE file
    assert get_resource("drivaernet").research_only  # CC BY-NC
    assert not get_resource("airfrans").research_only  # ODbL allows commercial use


def test_commercial_only_filter_excludes_research_only():
    assert all(not r.research_only for r in list_resources(commercial_only=True))


def test_research_only_warns_in_normal_mode(monkeypatch):
    monkeypatch.delenv("PINNEAPPLE_COMMERCIAL_MODE", raising=False)
    with pytest.warns(ResearchOnlyWarning, match="RESEARCH ONLY"):
        require_allowed("flowbench")


def test_research_only_refused_in_commercial_mode(monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    with pytest.raises(ResearchOnlyError):
        require_allowed("fusion360_gallery")
    with pytest.raises(ResearchOnlyError):
        fetch("flowbench")  # gate runs before any download


def test_commercial_resource_passes_silently_in_commercial_mode(monkeypatch):
    monkeypatch.setenv("PINNEAPPLE_COMMERCIAL_MODE", "1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert require_allowed("dpot").id == "dpot"


def test_fetch_rejects_resources_not_on_the_hub(monkeypatch):
    monkeypatch.delenv("PINNEAPPLE_COMMERCIAL_MODE", raising=False)
    with pytest.raises(ValueError, match="not hosted on the Hugging Face Hub"):
        fetch("airfrans")
