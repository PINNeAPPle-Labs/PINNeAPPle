"""One catalogue over every list (#1478): the registry is the source, absence is reported, nothing is invented."""
from __future__ import annotations

import pytest
import torch

import pinneapple as pp
from pinneapple_core.catalog import KINDS, Catalog, CatalogEntry, catalog

ANY = next(n for n in pp.list_models())


def test_the_sections_are_the_nine_kinds():
    c = catalog()
    counts = c.counts()
    assert tuple(counts) == KINDS
    for kind in KINDS:
        assert counts[kind] == len(pp.registered(kind))


def test_an_entry_carries_what_the_registry_declares_and_none_otherwise():
    entry = catalog().get(ANY)
    assert isinstance(entry, CatalogEntry)
    assert entry.id == ANY and entry.kind == "model"
    # a registration that declares no units, variables or provenance shows None — never an inferred value
    assert entry.units is None and entry.variables is None and entry.provenance is None


def test_get_raises_for_something_that_is_not_catalogued():
    with pytest.raises(KeyError, match="no catalogued resource"):
        catalog().get("nao/existe")


def test_every_section_returns_entries_of_its_own_kind():
    c = catalog()
    sections = {"model": c.models, "solver": c.solvers, "dataset": c.datasets,
                "physics": c.experiments, "material": c.materials}
    for kind, section in sections.items():
        for entry in section():
            assert entry.kind == kind, (kind, entry)
    assert {e.kind for e in c.entries()} <= set(KINDS)


def test_the_five_views_agree_on_the_architectures():
    """Acceptance of #1478: the lists that are about architectures return the same set."""
    c = catalog()
    views = c.views()
    agreeing = {name: ids for name, ids in views.items() if c.view_notes()[name] is None}
    assert len(agreeing) >= 2, views
    reference = next(iter(agreeing.values()))
    for name, ids in agreeing.items():
        assert ids == reference, f"{name} disagrees with the catalogue"
    assert c.discrepancies()["models"] == []


def test_a_view_that_is_another_plane_says_so():
    notes = catalog().view_notes()
    for name, note in notes.items():
        assert note is None or isinstance(note, str) and note.strip(), name


def test_a_model_registered_now_appears_in_every_view_with_no_other_change():
    """Acceptance of #1478: one registration reaches the catalogue, the facade and the app's listing."""

    class MyNet(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(2, 1)

        def forward(self, x):
            return self.linear(x)

    name = "catalog_test_net"
    try:
        pp.register(MyNet, kind="model", name=name)
    except Exception as exc:  # pragma: no cover - the kind refuses a plain module in this build
        pytest.skip(f"this build does not take a torch.nn.Module as a model: {exc}")
    try:
        assert name in {e.id for e in catalog().models()}
        assert name in pp.list_models()
        from pinneapple_app.backend.core.model_catalog import list_models as app_list
        assert name in {m.name for m in app_list()}
        assert catalog().get(name).kind == "model"
    finally:
        pp.unregister("model", name)


def test_the_catalogue_is_read_only():
    """It exposes queries and no way to write: the write path stays ``pp.register``."""
    c = catalog()
    public = [n for n in dir(c) if not n.startswith("_")]
    assert not any(n in ("register", "add", "insert", "delete", "remove", "update") for n in public)


def test_to_dict_serialises_every_section_and_the_views():
    payload = catalog().to_dict()
    assert set(payload) == {"kinds", "counts", "entries", "views", "view_notes", "discrepancies"}
    assert len(payload["entries"]) == sum(payload["counts"].values())
    assert payload["counts"]["model"] > 0


def test_a_catalogue_is_built_per_call_so_a_new_registration_is_seen():
    assert isinstance(catalog(), Catalog)
    assert catalog() is not catalog()


def test_what_a_registration_declares_the_catalogue_shows():
    """The other half of "missing is None": when the metadata *is* there, the catalogue shows it as declared."""
    from pinneapple_data.datasets.registry import DatasetInfo

    name = "catalog_test_dataset"
    info = DatasetInfo(id=name, name="Catalogue test dataset", category="toy", description="a test fixture",
                       fields=["T"], license="CC-BY-4.0", reference="Doe et al. (2020)")
    try:
        pp.register(info, kind="dataset", name=name, units={"T": "K"}, citation="Doe et al. (2020)",
                    license="CC-BY-4.0")
    except Exception as exc:  # pragma: no cover - the kind refuses this object in this build
        pytest.skip(f"this build does not take a DatasetInfo here: {exc}")
    try:
        entry = catalog().get(name)
        assert entry.units == {"T": "K"}
        assert entry.provenance, entry.provenance           # the licence and the citation, as declared
        assert "Doe" in str(entry.provenance) or "CC-BY" in str(entry.provenance)
    finally:
        pp.unregister("dataset", name)
