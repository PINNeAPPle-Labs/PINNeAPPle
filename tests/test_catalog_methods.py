"""Method catalog: code paths exist, references are well formed, status covers every item."""
import os
import re

from pinneapple_catalog.methods import get_method, list_methods, method_status, validated_methods

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARXIV = re.compile(r"arXiv:(\d{4}\.\d{4,5})")


def test_every_code_path_exists():
    for m in list_methods():
        for path in m.code:
            if path.startswith("PINNeAPPle-CFD/") and "..." in path:
                continue  # abbreviated sibling-repo path, resolved in the first entry of the group
            base = os.path.dirname(ROOT) if path.startswith("PINNeAPPle-CFD/") else ROOT
            full = os.path.join(base, path)
            if path.startswith("PINNeAPPle-CFD/") and not os.path.isdir(os.path.join(os.path.dirname(ROOT), "PINNeAPPle-CFD")):
                continue  # sibling repo not checked out here
            assert os.path.exists(full), f"{m.id}: {path}"


def test_solvers_training_and_equations_have_probes():
    for m in list_methods():
        assert m.probes, m.id


def test_wrong_xtfc_arxiv_id_is_gone():
    ids = {i for m in list_methods() for r in m.references for i in ARXIV.findall(r)}
    assert "2005.01219" not in ids  # metasurface paper, formerly cited for Leake & Mortari
    assert "1812.08625" in ids


def test_problems_point_at_known_equations():
    for m in list_methods("problem"):
        for eq in m.equations:
            assert get_method(eq).category == "equation", m.id


def test_status_covers_every_item_and_validated_is_a_subset():
    status = method_status()
    assert {m.id for m in list_methods()} <= set(status)
    for m in validated_methods():
        assert status[m.id]["status"] == "validated"
        assert status[m.id]["reference_tests"], m.id


def test_meta_learning_is_in_the_inventory():
    assert "MAML" in get_method("T18").name
