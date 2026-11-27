"""The one registry (#528): the external public datasets and the zoo's own artifacts in a single list, with the
licence shown before a download (non-commercial flagged), the citation, the checksums, and the licence and the
citation recorded in a run's dataset card."""
from __future__ import annotations

import json

import pytest

from pinneapple_data import registry as R
from pinneapple_data.datasets import catalog


def test_both_sources_are_in_one_list_with_a_licence_and_a_citation():
    items = R.entries()
    assert len(R.list(kind="dataset")) >= 10 and len(R.list(kind="model")) >= 3
    assert {e.source for e in items} == {"catalog", "zoo"}
    for e in items:
        assert e.licence, e.id                                   # a licence, always, even if it is "not stated"
        assert e.kind in R.KINDS
    # the external ones keep the catalogue's facts instead of a copy
    external = {e.id: e for e in R.list(source="catalog")}
    assert external["pdebench"].citation.startswith("Takamoto")
    assert external["pdebench"].licence_verified and external["pdebench"].licence == "CC BY 4.0"
    for e in external.values():
        c = catalog.get(e.id)
        assert e.citation == str(c.citation or "") and e.homepage == c.homepage


def test_a_non_commercial_licence_is_flagged_before_the_download():
    banner = R.licence("flowbench")
    assert "CC-BY-NC-4.0" in banner and "research use only" in banner
    e = R.get("flowbench")
    assert e.commercial_use is False and "non-commercial" in e.product_warning
    assert [x.id for x in R.list(commercial_only=True)].count("flowbench") == 0


def test_an_unknown_licence_is_not_upgraded_to_usable_in_a_product():
    e = R.get("spe10")                                            # the catalogue says "not stated"
    assert e.commercial_use is None and "not verified" in e.product_warning
    assert R.get("the_well").commercial_use is None                # "code BSD-3, data per scenario": code only


def test_a_zoo_model_carries_its_card_licence_and_citation():
    e = R.get("airfoil_cst_rans_mlp")
    assert e.kind == "model" and e.licence == "apache-2.0"
    assert e.citation.startswith("PINNeAPPle contributors")
    assert e.to_dict()["checksums"]["weights.pt"]                 # the zoo's SHA-256, not a second one


def test_a_registered_entry_loads_with_its_checksums_and_a_missing_reader_says_what_to_use():
    data = R.load("airfoil_rans_cst_re4e6", split="test", offline=True, quiet=True)
    assert len(data) > 3
    meta = R.metadata("airfoil_rans_cst_re4e6")
    assert meta["id"] == "airfoil_rans_cst_re4e6" and meta["licence"] == "cc-by-4.0"
    assert meta["checksums"]["data.npz"]
    # an entry with no reader is registered, not broken: the error names the call that works
    for missing, word in (("airfrans", "no adapter"), ("cfdbench", "local copy")):
        with pytest.raises(R.NoReaderError) as err:
            R.load(missing, quiet=True)
        assert R.get(missing).licence and word in str(err.value)


def test_a_model_loads_through_the_registry():
    model = R.load("airfoil_cst_rans_mlp", offline=True, quiet=True)
    assert hasattr(model, "predict")


def test_the_licence_is_printed_before_the_download_and_the_citation_after(capsys):
    R.load("airfoil_rans_cst_re4e6", split="test", offline=True)
    err = capsys.readouterr().err
    assert err.index("licence cc-by-4.0") < err.index("cite:")    # licence first, citation after the load


def test_a_gated_dataset_is_refused_until_the_caller_says_it_has_access():
    """ERA5 needs registration; the entry says so and ``load`` will not accept somebody else's terms on your behalf.
    The registry has no reader for it today, so it is a NoReaderError -- the gated check comes first, which is what
    the test pins: an entry can be both, and the licence is answered before the reader."""
    e = R.get("era5")
    assert e.gated and e.commercial_use is None
    with pytest.raises(R.RegistryError):
        R.load("era5", quiet=True)


def test_publishing_is_declared_and_returns_the_plan_it_would_upload():
    with pytest.raises(R.PublishError) as err:
        R.publish("airfoil_rans_cst_re4e6", target="hub")
    assert "checksums" in str(err.value) or "data.npz" in str(err.value)
    with pytest.raises(ValueError):
        R.publish("airfoil_rans_cst_re4e6", target="ftp")


def test_the_run_records_the_licence_and_the_citation_in_the_dataset_card(tmp_path):
    from pinneapple_lab.context import RunContext

    ctx = RunContext(str(tmp_path), {"Re": 1e4}, "demo", "demo-1234")
    ctx.dataset("rows")
    writer = R.record(ctx, "airfoil_rans_cst_re4e6", split="test")
    card = writer.card()
    assert card["license"] == "cc-by-4.0"
    assert card["provenance"]["registry"]["citation"].startswith("PINNeAPPle contributors")
    assert card["provenance"]["registry"]["schema"] == R.SCHEMA
    assert card["provenance"]["experiment"] == "demo"              # what was already there is kept
    ctx.close()
    on_disk = json.loads((tmp_path / "datasets" / "airfoil_rans_cst_re4e6" / "card.json").read_text())
    assert on_disk["license"] == "cc-by-4.0"
    with pytest.raises(KeyError):
        R.record(ctx, "not-an-entry")                               # a typo fails here, not in a report later


def test_the_generated_table_names_every_entry_and_the_licence_columns():
    text = R.to_markdown()
    assert text.startswith("# Datasets and models: the registry")
    assert "| id | kind | domain | licence |" in text
    assert "flowbench" in text and "no |" in text                   # the product column says no for CC-BY-NC
    assert text.count("\n") > 20


def test_the_cli_writes_the_page_and_prints_the_banner(tmp_path, capsys):
    out = tmp_path / "registry.md"
    assert R.cli(["--docs", str(out)]) == 0 and "registry" in out.read_text()
    assert R.cli(["--licence", "pdebench"]) == 0
    assert "CC BY 4.0" in capsys.readouterr().out
    assert R.cli(["--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == R.SCHEMA and len(payload["entries"]) >= 13


def test_a_name_that_is_not_a_zoo_dataset_falls_through_to_the_registry():
    from pinneapple_hub.zoo import datasets as Z
    with pytest.raises(R.NoReaderError) as err:                     # no reader: the registry explains
        Z.load("pdebench")
    assert "local copy" in str(err.value)
    with pytest.raises(KeyError):
        Z.load("no-such-dataset-anywhere")
    assert Z.load("airfoil_rans_cst_re4e6", split="test", offline=True) is not None
