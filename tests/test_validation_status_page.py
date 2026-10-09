"""The public validation status page (docs/validation_status.md) matches the catalog scan (#97)."""
import importlib.util
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _script():
    spec = importlib.util.spec_from_file_location("build_method_status", os.path.join(ROOT, "scripts", "build_method_status.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_public_page_is_in_sync_with_method_status_json():
    mod = _script()
    with open(mod.PUBLIC_PAGE, encoding="utf-8") as f:
        assert f.read() == mod.render_public_page(), "run: python scripts/build_method_status.py --public --markdown"


def test_public_page_counts_every_catalog_item():
    from pinneapple_catalog.methods import list_methods
    with open(os.path.join(ROOT, "pinneapple_catalog", "method_status.json")) as f:
        status = json.load(f)["methods"]
    page = _script().render_public_page()
    n_validated = sum(1 for m in list_methods() if status.get(m.id, {}).get("status") == "validated")
    assert f"| **All** | **{n_validated}** |" in page
    assert all(f"| {m.id} |" in page for m in list_methods())
