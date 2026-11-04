"""Library helpers moved out of the October apps (#1226): series downsampling, cached 3-D views, the OpenFOAM case
root, the DOE design size and the campaign plan view / archive."""
from __future__ import annotations

import json
import zipfile

import numpy as np
import pytest


# ------------------------------------------------------------------------------------------ downsampling (#1235)
def test_minmax_downsample_preserves_the_extrema_that_a_stride_steps_over():
    from pinneapple_core.series import downsample, downsample_indices
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 100_000)
    y[12_345], y[77_777] = 40.0, -35.0                      # isolated spikes
    x = np.arange(y.size) * 0.01
    xs, ys = downsample(x, y, 1000)
    assert len(ys) <= 1000 and ys.max() == 40.0 and ys.min() == -35.0
    assert np.all(np.diff(xs) > 0) and xs[0] == x[0] and xs[-1] == x[-1]
    every = y[downsample_indices(y.size, 1000, "even")]
    assert every.max() < 40.0                               # the evenly spaced pick misses the spike
    for k in (1000, 37):                                    # each bucket's extrema are kept
        idx = downsample_indices(y.size, k, "minmax", y)
        assert y[idx].max() == y.max() and y[idx].min() == y.min()


def test_downsample_methods_match_the_strides_they_replace():
    from pinneapple_core.series import bucket_edges, downsample_indices
    for n, k in [(10, 20), (1500, 1500), (10_001, 1500), (3, 1)]:
        assert np.array_equal(downsample_indices(n, k, "even"), np.linspace(0, n - 1, min(k, n)).astype(int))
        step = max(1, int(np.ceil(n / k)))
        assert np.array_equal(downsample_indices(n, k, "stride"), np.arange(n)[::step])
        assert np.array_equal(bucket_edges(n, k), np.linspace(0, n, min(k, n) + 1).astype(int))
    assert downsample_indices(0, 5, "minmax", []).size == 0
    with pytest.raises(ValueError):
        downsample_indices(10, 5, "nope")
    with pytest.raises(ValueError, match="needs the values"):
        downsample_indices(100, 10, "minmax")


def test_minmax_ignores_non_finite_values():
    from pinneapple_core.series import downsample
    y = np.r_[np.full(50, np.nan), np.linspace(0, 1, 950)]
    _, ys = downsample(None, y, 20)
    assert np.nanmax(ys) == 1.0 and np.nanmin(ys) == 0.0


# ------------------------------------------------------------------------------------------ cached views (#1235)
class _View:
    def __init__(self, tag):
        self.tag = tag

    def glb(self):
        return f"glb-{self.tag}".encode()


def test_cached_view_hits_on_the_same_key_and_rebuilds_on_another(tmp_path):
    from pinneapple_tools.visualization.studio import cached_view
    calls = []

    def build(tag):
        def f():
            calls.append(tag)
            return _View(tag), {"tag": tag, "x": np.float32(0.5), "bad": float("nan")}
        return f

    m1, g1 = cached_view(tmp_path / "views", "a_0", build("a"))
    m2, g2 = cached_view(tmp_path / "views", "a_0", build("other"))   # cached: not rebuilt
    m3, g3 = cached_view(tmp_path / "views", "b_0", build("b"))
    assert calls == ["a", "b"]
    assert m1 == m2 == {"tag": "a", "x": 0.5, "bad": None} and g1 == g2 and g1.read_bytes() == b"glb-a"
    assert m3["tag"] == "b" and g3 != g1
    _, g4 = cached_view(tmp_path / "views", "../evil/x", build("c"))     # a key cannot leave the folder
    assert g4.parent == tmp_path / "views"


def test_a_failed_build_leaves_no_description(tmp_path):
    from pinneapple_tools.visualization.studio import cached_view, view_paths

    def boom():
        raise ValueError("no mesh")
    with pytest.raises(ValueError):
        cached_view(tmp_path, "k", boom)
    assert not any(p.is_file() for p in view_paths(tmp_path, "k"))


def test_saved_views_are_pruned_oldest_first(tmp_path):
    import os

    from pinneapple_tools.visualization.studio import prune_views, save_view
    ids = []
    for i in range(5):
        ids.append(save_view(tmp_path, _View(i)))
        os.utime(tmp_path / f"{ids[-1]}.glb", (1000 + i, 1000 + i))
    assert all(len(v) == 32 for v in ids)
    assert prune_views(tmp_path, 2) == 3
    assert sorted(p.stem for p in tmp_path.glob("*.glb")) == sorted(ids[-2:])


# ------------------------------------------------------------------------------------------ OpenFOAM case root (#1237)
def _zip_case(path, top="mycase", extra=()):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{top}/system/controlDict", "application simpleFoam;\n")
        z.writestr(f"{top}/0/U", "internalField uniform (1 0 0);\n")
        for name, data in extra:
            z.writestr(name, data)
    return path


def test_find_case_root_and_safe_extract(tmp_path):
    from pinneapple_simulation.external_solvers.openfoam.case_root import (
        extract_case,
        find_case_root,
        is_case,
    )
    root = extract_case(_zip_case(tmp_path / "c.zip"), tmp_path / "out")
    assert root == tmp_path / "out" / "mycase" and is_case(root) and find_case_root(root) == root
    assert not is_case(tmp_path / "out")
    with zipfile.ZipFile(tmp_path / "none.zip", "w") as z:
        z.writestr("readme.txt", "no case")
    with pytest.raises(ValueError, match="no OpenFOAM case"):
        extract_case(tmp_path / "none.zip", tmp_path / "out2")
    _zip_case(tmp_path / "evil.zip", extra=[("../escape.txt", "x")])
    with pytest.raises(ValueError, match=r"\.\."):
        extract_case(tmp_path / "evil.zip", tmp_path / "out3")
    assert not (tmp_path / "escape.txt").exists()


def test_extract_case_honours_the_limits(tmp_path):
    from pinneapple_security.archives import ArchiveLimits
    from pinneapple_simulation.external_solvers.openfoam.case_root import extract_case
    with pytest.raises(ValueError):
        extract_case(_zip_case(tmp_path / "c.zip"), tmp_path / "o", ArchiveLimits(max_members=1))


# ------------------------------------------------------------------------------------------ DOE size and plan (#1237)
def _space():
    from pinneapple_data.contracts import ParameterSpace
    return ParameterSpace.from_dict({"name": "s", "parameters": [
        {"name": "a", "kind": "continuous", "low": 0, "high": 1},
        {"name": "b", "kind": "continuous", "low": 1, "high": 10, "log": True},
        {"name": "c", "kind": "integer", "low": 1, "high": 2},
        {"name": "d", "kind": "categorical", "values": ["x", "y", "z"]},
        {"name": "e", "kind": "continuous", "low": -1, "high": 1}]})


@pytest.mark.parametrize("method", ["lhs", "maximin", "sobol", "halton", "random", "full_factorial",
                                    "fractional_factorial"])
@pytest.mark.parametrize("n", [5, 8, 16, 33])
def test_design_size_equals_the_planned_design(method, n):
    from pinneapple_data.doe import design, design_size
    sp = _space()
    for levels in (2, 3, {"a": 4}):
        if method != "full_factorial" and levels != 3:
            continue
        assert design_size(sp, method, n, levels) == len(design(sp, n, method, levels=levels, n_restarts=2))


def test_design_size_of_infill_and_of_a_constrained_factorial():
    import pandas as pd

    from pinneapple_data.doe import design, design_size
    sp = _space()
    ex = pd.DataFrame({"a": [0.5], "b": [3.0], "c": [1], "d": ["x"], "e": [0.0]})
    assert design_size(sp, "infill", 7) == len(design(sp, 7, "infill", existing=ex)) == 7
    feas = lambda df: df["a"] < 0.6                         # noqa: E731
    assert design_size(sp, "full_factorial", None, 3, feasible=feas) == \
        len(design(sp, 0, "full_factorial", levels=3, feasible=feas))
    with pytest.raises(ValueError):
        design_size(sp, "full_factorial", None, 40)       # > 100 000 cases, as design raises


def test_campaign_plan_view_and_archive(tmp_path):
    from pinneapple_data.doe import plan_campaign
    from pinneapple_simulation.external_solvers import campaign_cases as cc
    plan = plan_campaign(_space(), 12, "lhs")
    v = plan.view()
    assert v["names"] == ["a", "b", "c", "d", "e"] and len(v["planned"]) == 12 and "existing" not in v
    assert v["values"] == {"d": ["x", "y", "z"]} and v["bounds"]["b"] == [1, 10, True]
    files = cc.write_campaign(plan, tmp_path / "campaign", archive=tmp_path / "campaign.zip")
    names = zipfile.ZipFile(files["archive"]).namelist()
    on_disk = sorted(p.relative_to(tmp_path).as_posix() for p in (tmp_path / "campaign").rglob("*") if p.is_file())
    assert names == on_disk and "campaign/space.json" in names
    json.dumps(v)


# ------------------------------------------------------------------------------------------ app shims removed (#1239)
def test_library_defaults_do_not_depend_on_an_app_folder():
    import os
    from pathlib import Path

    import pinneapple_analysis.inverse_problems.heat_transfer as ht
    import pinneapple_design.thermal.heatsink.surrogate as hs
    root = Path(__file__).resolve().parents[1]
    if not os.environ.get("PINNEAPPLE_INVERSE_HEAT_RESULTS"):      # the stored runs of the source checkout
        assert Path(ht.EXAMPLE) == root / "examples" / "use_cases" / "fin_convection_inverse"
        assert (Path(ht.EXAMPLE) / "results" / "2d").is_dir()
    assert "apps" not in Path(hs.ARTIFACTS).parts
    found = hs.load_surrogates(str(root / "apps" / "heatsink_sizer" / "artifacts"))
    assert set(found) <= {"forced", "natural"}
