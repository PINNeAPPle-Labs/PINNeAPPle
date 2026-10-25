"""pinneapple_arena.external: the catalog of third-party physics-ML libraries is references plus glue (no copied
third-party code), every entry has a recorded licence source, and results never claim a run that did not happen."""
from __future__ import annotations

import ast
from pathlib import Path

from pinneapple_arena import external as ext

RUNNERS_DIR = Path(ext.__file__).parent / "runners"


def test_catalog_counts_and_references():
    assert len(ext.MODELS) == 95 == len(ext.MODELS_BY_SLUG)
    assert set(ext.RUNNERS) == set(ext.MODELS_BY_SLUG)
    assert len({d for m in ext.MODELS for d in m.domains}) == 41
    for m in ext.MODELS:
        assert m.repo.startswith("https://") and m.ref and m.name
        assert m.license_source, m.slug                   # where the licence was read, or that none was found
        assert m.manual_only or m.pip_package, m.slug     # install instructions: pip, or clone + build


def test_licence_counts_and_non_commercial_flags():
    counts = ext.licence_counts()
    assert sum(counts.values()) == 95
    nc = sorted(m.slug for m in ext.MODELS if m.non_commercial)
    assert nc == ["deeponet-fno-benchmark", "deeponet-original", "fluidx3d", "graphcast", "pangu-weather",
                  "reservoirflow"]


def test_runners_hold_no_copied_library_code():
    """Glue only: no class definitions, and every third-party import happens inside a function (at run time)."""
    for path in RUNNERS_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        assert not [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)], path.name
        for node in tree.body:
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mod = node.module if isinstance(node, ast.ImportFrom) else node.names[0].name
                assert mod.split(".")[0] in {"__future__", "numpy", "pinneapple_arena", "inspect", "typing",
                                             "collections"}, (path.name, mod)


def test_results_report_what_happened():
    statuses = {ext.call_infer(s).status for s in ext.RUNNERS if ext.MODELS_BY_SLUG[s].manual_only}
    assert statuses == {"manual_install"}
    r = ext.call_infer("physicsnemo")                     # heavy: skipped unless asked
    assert r.status == "skipped_heavy" and "--include-heavy" in r.local_command
    for slug in ("deepxde", "pina", "sciann"):
        r = ext.call_infer(slug)
        if not ext.is_installed(ext.MODELS_BY_SLUG[slug].import_name):
            assert r.status == "not_installed" and r.local_command.startswith("python -m pinneapple_arena.external")


def test_dataset_probe_without_data_or_stream_is_not_ok():
    spec = ext.DATASETS_BY_SLUG["the-well"]
    r = ext.probe_dataset(spec, spec.subsets[0].name, "valid", stream=False, dataset_root=None)
    assert r.status in ("not_installed", "skipped_heavy") and r.local_command
