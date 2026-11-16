"""Tests of the verification corpus and of the neutral residual it is scored with.

They check the properties the paper relies on, not the numbers: that the corpus is deterministic,
that each valid approximation hits its declared error level, that each defect is the only deviation
in its case, that the perturbation really vanishes on the declared boundary data, and that the
residual operator gives zero on a solution it should.

Run::

    python -m pytest research/arxiv_paper/experiments/verification/test_corpus.py -q
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import analyze  # noqa: E402
import corpus  # noqa: E402
import run as runner  # noqa: E402

#: The cheapest problems to build; Allen-Cahn spends about 80 s in its reference interpolation and is
#: covered by one test marked slow rather than by every test.
FAST = ("steady_heat", "poisson", "helmholtz")


def cases(name: str) -> list[corpus.Case]:
    return corpus.build_cases(corpus.BY_NAME[name]())


# --------------------------------------------------------------------------- the corpus itself
def test_every_problem_declares_the_same_twelve_cases():
    for name in corpus.BY_NAME:
        names = [c.case_id.split("/")[1] for c in cases(name)]
        assert len(names) == 12, name
        assert "valid_exact" in names
        assert sum(1 for c in cases(name) if c.family == "invalid") == len(corpus.DEFECTS)


@pytest.mark.parametrize("name", FAST)
def test_corpus_is_deterministic(name):
    a = cases(name)
    b = cases(name)
    for ca, cb in zip(a, b, strict=False):
        assert np.array_equal(ca.field, cb.field, equal_nan=True), ca.case_id


@pytest.mark.parametrize("name", FAST)
@pytest.mark.parametrize("level", corpus.VALID_LEVELS)
def test_valid_ladder_hits_its_declared_error(name, level):
    problem = corpus.BY_NAME[name]()
    ref = np.asarray(problem.exact(problem.grid.mesh()), float)
    case = next(c for c in cases(name) if c.case_id.endswith(f"valid_{level:g}"))
    assert corpus.rel_l2(case.field, ref) == pytest.approx(level, rel=1e-9)


@pytest.mark.parametrize("name", FAST)
def test_perturbation_vanishes_on_the_declared_boundary(name):
    """A valid approximation must not break a boundary condition, or the ladder measures the wrong thing."""
    problem = corpus.BY_NAME[name]()
    ref = np.asarray(problem.exact(problem.grid.mesh()), float)
    p = np.asarray(problem.perturbation(problem.grid.mesh()), float)
    names = problem.grid.axes
    faces = list(problem.checker_bc())
    if problem.checker_ic():
        faces.append("t=min")
    for face in faces:
        axis, side = face.split("=")
        sl = [slice(None)] * p.ndim
        sl[names.index(axis)] = 0 if side == "min" else -1
        assert np.max(np.abs(p[tuple(sl)])) < 1e-12, (name, face)
    assert ref.shape == p.shape


@pytest.mark.parametrize("name", FAST)
def test_over_smoothed_keeps_the_declared_boundary_data(name):
    problem = corpus.BY_NAME[name]()
    ref = np.asarray(problem.exact(problem.grid.mesh()), float)
    case = next(c for c in cases(name) if c.defect == "over_smoothed")
    names = problem.grid.axes
    faces = list(problem.checker_bc())
    if problem.checker_ic():
        faces.append("t=min")
    for face in faces:
        axis, side = face.split("=")
        sl = [slice(None)] * ref.ndim
        sl[names.index(axis)] = 0 if side == "min" else -1
        assert np.array_equal(case.field[tuple(sl)], ref[tuple(sl)]), (name, face)
    # and it really removed small scales
    assert corpus.rel_l2(case.field, ref) > 0


@pytest.mark.parametrize("name", FAST)
def test_each_defect_is_the_only_deviation_in_its_case(name):
    problem = corpus.BY_NAME[name]()
    ref = np.asarray(problem.exact(problem.grid.mesh()), float)
    got = {c.defect: c for c in cases(name) if c.family == "invalid"}

    assert np.all(got["zero_field"].field == 0)
    assert got["nonfinite"].nans == 5
    assert not np.isfinite(got["nonfinite"].field).all()
    assert np.allclose(got["unit_scale"].field, ref * 1e3, equal_nan=True)
    # the boundary offset touches the boundary and nothing in the interior
    offset = got["bc_offset"].field - ref
    centre = tuple(len(v) // 2 for v in problem.grid.values)
    assert abs(offset[centre]) < 1e-9
    assert np.max(np.abs(offset.ravel())) > 0


def test_the_declared_expectations_cover_every_defect():
    assert set(analyze.EXPECTED) == set(corpus.DEFECTS)
    for rules in analyze.EXPECTED.values():
        assert rules and all(r in analyze.RULE_TITLES for r in rules)


def test_the_checker_configuration_is_valid_toml():
    import tomllib
    for name in FAST:
        problem = corpus.BY_NAME[name]()
        data = tomllib.loads(corpus.checker_toml(problem, with_reference=True))
        assert data["select"]
        target = data["results"][0]
        assert target["path"] == "prediction.npz"
        assert target["kind"] == "grid"
        assert corpus.REFERENCE_REL_L2 == target["reference"]["rel_l2"]


# --------------------------------------------------------------------------- the residual operator
@pytest.mark.parametrize("name", FAST)
def test_residual_is_at_the_floor_on_the_exact_solution(name):
    """The operator must give the discretisation floor -- not zero, and not a tolerance-sized number.

    The floor is a property of the grid: it is at machine precision for the quadratic steady solution,
    3.5e-3 for Poisson at h = 1e-2, and 3.7e-2 for Helmholtz at h = 2e-2, whose 4*pi mode is the least
    resolved thing in the corpus. All three are far below the 5% the checker tolerates.
    """
    problem = corpus.BY_NAME[name]()
    ref = np.asarray(problem.exact(problem.grid.mesh()), float)
    stat = runner.residual_stat(name, ref, ref, problem.grid)
    assert stat["residual_relative"] < 0.05, (name, stat)


@pytest.mark.parametrize("name", FAST)
def test_residual_of_the_exact_solution_is_the_reported_floor(name):
    """self-consistency: residual_stat(ref, ref) is the floor it reports."""
    problem = corpus.BY_NAME[name]()
    ref = np.asarray(problem.exact(problem.grid.mesh()), float)
    stat = runner.residual_stat(name, ref, ref, problem.grid)
    assert stat["residual_relative"] == pytest.approx(stat["reference_residual_relative"])


@pytest.mark.parametrize("name", FAST)
def test_residual_rises_above_the_floor_with_a_10_percent_error(name):
    """A residual check is only useful where a field error lifts the residual clear of the floor.

    At a 10% error every problem in FAST clears it by at least half again; on Helmholtz, whose floor
    is the highest here, this is the weakest such statement in the corpus, and it still holds.
    """
    problem = corpus.BY_NAME[name]()
    ref = np.asarray(problem.exact(problem.grid.mesh()), float)
    p = np.asarray(problem.perturbation(problem.grid.mesh()), float)
    scale = np.linalg.norm(p.ravel()) / np.linalg.norm(ref.ravel())
    floor = runner.residual_stat(name, ref, ref, problem.grid)["reference_residual_relative"]
    bad = runner.residual_stat(name, ref + p / scale * 1e-1, ref, problem.grid)["residual_relative"]
    assert bad > 1.5 * floor, (name, floor, bad)


@pytest.mark.skipif(os.environ.get("PINNEAPPLE_SLOW") != "1",
                    reason="the Allen-Cahn reference interpolation takes about 80 s; set PINNEAPPLE_SLOW=1")
def test_allen_cahn_reference_matches_the_benchmark():
    """Allen-Cahn reuses the benchmark's own interpolation; this pins the reuse."""
    from pinneapple_tools.benchmark_suite.crossframework.problems import _allen_cahn_interp
    problem = corpus.BY_NAME["allen_cahn"]()
    X = problem.grid.mesh()
    got = np.asarray(problem.exact(X), float)
    want = _allen_cahn_interp(np.asarray(X[0], float).ravel(), np.asarray(X[1], float).ravel(),
                              problem.d).reshape(got.shape)
    assert np.array_equal(got, want)
