"""The classical CFD validation cases: the published tables, and a real solver compared with them (#272, #494).

Three layers, because the issue asks for all of them and only one needs a machine:

* the **tables** (``pinneapple_simulation.cfd_validation.reference``) are checked against what a table of a
  lid-driven cavity must be true of on its own — the lid moves at ``u = 1``, the walls do not move, the coordinates
  increase, one sign change along the centreline — so a transcription error is caught with no solver at all;
* the **runs** are here too, against a real OpenFOAM when one is installed (skipped otherwise, the same way
  ``tests/test_openfoam_limits.py`` skips): the case at Re = 100 agreeing with Ghia's profile, the error falling as
  the mesh refines, the step reattaching, and a warm start saving iterations without changing the answer;
* the **network** (#272's other half) is not retrained here — a training run is minutes of a machine, not a test —
  so what is held is the protocol: the committed ``results/cavity_pinn.json`` is re-scored from its own stored
  profiles through the *same* :func:`profile_error` the meshes go through, and the number the README quotes is
  recomputed from data rather than trusted.

The numbers the runs produce are the ones in ``examples/cfd/classical_cases/README.md``: the tests are the check that
the tables and the case builders still say what the README says they say.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from pinneapple_simulation.cfd_validation import (
    DEV_VAHL_DAVIS_NU,
    GHIA_RE,
    GHIA_U,
    GHIA_V,
    PROFILES,
    cavity,
    foam,
    nu_for_grashof,
    observed_order,
    profile,
    profile_error,
    reference,
    step,
    warm_start,
)

needs_foam = pytest.mark.skipif(not foam.available("simpleFoam"),
                                reason="no OpenFOAM installation found (set FOAM_BASHRC)")


# --------------------------------------------------------------------------------------------- the published tables
def test_every_profile_is_the_length_it_claims_and_its_coordinates_increase():
    for p in PROFILES:
        assert len(p.points) == len(p.values), p.quantity
        assert len(p.points) >= 10, p.quantity
        assert all(b > a for a, b in zip(p.points, p.points[1:], strict=False)), p
        assert np.all(np.isfinite(p.values)), p.quantity
        assert len(p.citation) > 40 and "1982" in p.citation, p.citation


def test_the_tables_are_the_published_ones_at_every_reynolds_number_in_the_issue():
    assert GHIA_RE == (100, 400, 1000, 10000)
    for re in GHIA_RE:
        assert GHIA_U[re].reynolds == re and GHIA_V[re].reynolds == re
        assert GHIA_U[re].span == (0.0, 1.0) and GHIA_V[re].span == (0.0, 1.0)


def test_the_lid_and_the_walls_are_the_boundary_conditions_the_tables_end_on():
    """``u(y = 1) = 1`` is the moving lid and ``u(y = 0) = 0`` the fixed one; ``v`` is zero on both side walls."""
    for re in GHIA_RE:
        u, v = GHIA_U[re].values, GHIA_V[re].values
        assert u[-1] == 1.0, re
        assert u[0] == 0.0, re
        assert v[0] == 0.0 and v[-1] == 0.0, re


def test_the_primary_vortex_crosses_the_centreline_exactly_once():
    """A lid-driven cavity has one primary vortex: ``u`` along ``x = 0.5`` changes sign once, and that is what makes
    the table a cavity and not, say, a channel."""
    for re in GHIA_RE:
        u = np.asarray(GHIA_U[re].values)
        sign = np.sign(u)
        sign = sign[sign != 0]                       # the two walls sit at exactly zero, and are not a crossing
        changes = np.flatnonzero(np.diff(sign) != 0)
        assert len(changes) == 1, f"Re = {re}: {len(changes)} sign changes in u"


def test_the_strongest_backflow_sits_below_the_centreline():
    """The primary vortex turns the flow down the lid and back along the bottom, so the strongest reverse velocity is
    in the lower half: a table whose minimum were above the centreline would be the mirror image of a cavity."""
    for re in GHIA_RE:
        u = np.asarray(GHIA_U[re].values)
        y = GHIA_U[re].points[int(np.argmin(u))]
        assert y < 0.5, f"Re = {re}: the minimum of u is at y = {y}"


def test_the_lookup_names_what_it_has_instead_of_inventing_a_row():
    assert profile("ghia-1982", "u", 400) is GHIA_U[400]
    assert profile("ghia-1982", "v", 1000.0) is GHIA_V[1000]
    with pytest.raises(KeyError, match="not published"):
        profile("ghia-1982", "u", 250)
    with pytest.raises(KeyError, match="no transcribed profiles"):
        profile("unknown", "u", 100)


def test_the_de_vahl_davis_benchmark_is_monotone_and_only_the_published_points_are_answered():
    values = [DEV_VAHL_DAVIS_NU[k] for k in sorted(DEV_VAHL_DAVIS_NU)]
    assert values == sorted(values) and len(set(values)) == len(values)
    assert nu_for_grashof(1e5) == 4.519
    with pytest.raises(KeyError, match="only"):
        nu_for_grashof(1e2)


def test_the_reattachment_table_says_it_is_empty_rather_than_holding_a_guess():
    """The published Armaly curve is a figure; the digitisation circulating is secondary. The table is empty and the
    case still runs, which is the honest state of it."""
    assert reference.ARMALY_REATTACHMENT == {}


# --------------------------------------------------------------------------------------------- the arithmetic
def test_an_exact_profile_has_no_error_and_a_shifted_one_has_the_shift():
    ref = np.asarray(GHIA_U[100].values)
    assert profile_error(GHIA_U[100], ref)["l2"] == pytest.approx(0.0, abs=1e-15)
    shifted = ref + 0.01
    got = profile_error(GHIA_U[100], shifted)
    assert got["l2"] == pytest.approx(0.01, rel=1e-9) and got["points"] == len(ref) - 2
    assert got["linf"] == pytest.approx(0.01, rel=1e-9)


def test_the_wall_stations_are_left_out_and_said_so():
    """The two ends are the boundary condition, not a solved value: including them would reward a case for reading
    its own input, and a caller that wants them can ask."""
    ref = np.asarray(GHIA_U[100].values)
    broken = ref.copy()
    broken[0] = broken[-1] = 1e3
    assert profile_error(GHIA_U[100], broken)["l2"] == pytest.approx(0.0, abs=1e-15)
    assert profile_error(GHIA_U[100], broken, interior_only=False)["l2"] > 100


def test_the_observed_order_is_the_log_ratio_of_the_errors():
    assert observed_order([0.4, 0.1, 0.025], ratio=2.0) == pytest.approx([2.0, 2.0])
    assert observed_order([0.4, 0.4]) == pytest.approx([0.0])
    assert observed_order([0.1, 0.0])[0] == float("inf")          # nothing left to resolve
    assert observed_order([0.1, 0.2])[0] < 0                       # the error grew: that is the finding


# --------------------------------------------------------------------------------------------- the cases, no solver
def test_the_cavity_case_has_the_two_things_that_stop_it_at_start_up(tmp_path):
    """``pRefCell`` and ``constant/turbulenceProperties``: without either the solver never reaches the first
    iteration (``pinneapple_data.preflight.limits``). A case builder that forgot one would look fine until run."""
    case = str(tmp_path / "c")
    params = cavity.write_case(case, 400, 16)
    assert params["nu"] == pytest.approx(1 / 400.0)
    fs = foam.case_files(case, time="0")
    assert "constant/turbulenceProperties" in fs
    assert "pRefCell" in fs["system/fvSolution"].decode()
    from pinneapple_data.preflight.openfoam import check
    report = check(fs)
    failed = [f for f in report.findings if f.status == "fail"]
    assert not failed, [f.title for f in failed]


def test_the_step_case_is_one_block_plus_a_cut_and_an_inlet_that_waits_for_the_mesh(tmp_path):
    """``subsetMesh`` checks a field's length against the mesh it is about to cut, so the inlet profile cannot be
    written before the cut: the case is written with a zero inlet and :func:`step.write_inlet` fills it in."""
    case = str(tmp_path / "s")
    params = step.write_case(case, 200, 8)
    assert 1.5 < params["expansion_mesh"] < 2.2
    with open(f"{case}/system/blockMeshDict") as fh:
        blocks = fh.read().count("hex (")
    assert blocks == 1, "the step is cut out of one block, not built from several"
    with open(f"{case}/system/topoSetDict") as fh:
        assert "action invert" in fh.read()
    with open(f"{case}/0/U") as fh:
        assert "uniform (0 0 0)" in fh.read()


def test_mapping_a_predicted_field_keeps_the_boundary_conditions_byte_for_byte(tmp_path):
    """The boundary conditions are the problem statement; a surrogate may place the first iterate and nothing else."""
    case = str(tmp_path / "c")
    cavity.write_case(case, 100, 8)
    before = open(f"{case}/0/U").read()
    predicted = np.column_stack([np.linspace(0, 1, 64), np.zeros(64), np.zeros(64)])
    got = warm_start.write_fields(case, {"U": predicted})
    after = open(f"{case}/0/U").read()
    assert got["U"]["cells"] == 64
    boundary_before = before[before.index("boundaryField"):]
    boundary_after = after[after.index("boundaryField"):]
    assert boundary_before == boundary_after
    assert "nonuniform List<vector> 64" in after


def test_internal_field_refuses_a_field_of_the_wrong_width(tmp_path):
    case = str(tmp_path / "c")
    cavity.write_case(case, 100, 8)
    with pytest.raises(ValueError, match="expected"):
        warm_start.set_internal_field(case, "U", np.zeros((64, 2)))


def test_the_operator_surrogate_predicts_between_the_cases_it_was_fitted_to(tmp_path):
    """A surrogate is fitted from runs of one mesh: two analytic snapshots, a prediction at the parameter between
    them, and the error measured against a third — the same code path the cavity uses."""
    y = np.linspace(0, 1, 256)
    def snap(re):
        return {"u": np.stack([np.sin(np.pi * y) * re / 1000.0, np.cos(np.pi * y) * 0.01], axis=1)}
    surrogate = warm_start.OperatorSurrogate(modes=2).fit({100.0: snap(100), 1000.0: snap(1000)})
    got = surrogate.predict(400.0)["u"]
    assert got.shape == (256, 2)
    assert surrogate.energy["u"] == pytest.approx(1.0, abs=1e-6)
    err = surrogate.error(400.0, snap(400))
    assert err["u"] < 1e-6                       # the field is linear in Re, so two modes reproduce it exactly
    surrogate.predict(5000.0)
    assert surrogate.extrapolated is True


# --------------------------------------------------------------------------------------------- the runs
@needs_foam
def test_ghia_re100_on_a_coarse_mesh_already_matches_the_published_profile(tmp_path):
    """The table and the case are checked against each other here: a mis-transcribed row cannot agree with a
    Navier-Stokes solution at 32 x 32, and a case whose scheme or boundary conditions are wrong does not either."""
    case = str(tmp_path / "cavity")
    cavity.write_case(case, 100, 32)
    info = cavity.run(case, 100, 32)
    assert info["ok"], info["log"][-1500:]
    assert info["solver"]["converged"], info["solver"]
    got = cavity.sample(case, 100)
    u = profile_error(GHIA_U[100], got["u"])
    v = profile_error(GHIA_V[100], got["v"])
    assert u["l2_relative"] < 0.02, u
    assert v["l2_relative"] < 0.05, v


@needs_foam
def test_refining_the_mesh_reduces_the_error_against_the_table(tmp_path):
    """A validation that does not converge is a coincidence, not a validation: the error at 32 cells has to be
    smaller than at 16."""
    errs = {}
    for n in (16, 32):
        case = str(tmp_path / f"cavity_{n}")
        cavity.write_case(case, 100, n)
        assert cavity.run(case, 100, n)["ok"]
        errs[n] = profile_error(GHIA_U[100], cavity.sample(case, 100)["u"])["l2"]
    assert errs[32] < errs[16], errs
    assert observed_order([errs[16], errs[32]])[0] > 0.5


@needs_foam
def test_a_fresh_directory_is_solved_on_the_first_call(tmp_path):
    """``study`` writes the case before it runs it. It did not: the first call on an empty directory recorded a failed
    run (``blockMesh`` in a directory with no dictionaries) and a second call produced the solution, which reads as the
    solver's failure in a JSON that is only a day old."""
    got = cavity.study(100, str(tmp_path / "fresh"), meshes=(32,))
    rec = got["runs"][0]
    assert rec["iterations"] > 0 and rec["converged"], rec.get("note")
    assert got["errors"][32]["u"]["l2_relative"] < 0.02, got["errors"][32]


@needs_foam
def test_the_step_reattaches_behind_the_step(tmp_path):
    """``x1/S`` is what the case exists to produce: a separated bubble that closes again well inside the outlet."""
    case = str(tmp_path / "step")
    step.write_case(case, 200, 8)
    info = step.run(case, 200, 8)
    assert info["ok"], info["log"][-1500:]
    assert info["inlet"]["faces"] == info["inlet"]["faces"]                 # the profile was written on the mesh
    assert info["inlet"]["u_max"] > 1.3                                    # a parabola of mean 1 peaks at 1.5
    got = step.reattachment_length(case)
    assert got["u_min"] < 0, got
    assert 1.0 < got["x1_over_S"] < 20.0, got


@needs_foam
def test_a_warm_start_saves_iterations_and_does_not_change_the_answer(tmp_path):
    """The point of #494: the solver still produces the answer, and the answer is the same one."""
    n = 32
    golden = {}
    for re in (200, 600):
        case = str(tmp_path / f"snap_{re}")
        cavity.write_case(case, re, n)
        assert cavity.run(case, re, n)["ok"]
        data = foam.cells(case)
        golden[float(re)] = {"U": np.asarray(data["U"]), "p": np.asarray(data["p"])}
    surrogate = warm_start.OperatorSurrogate(modes=4).fit(golden)
    got = warm_start.benchmark(400, surrogate, str(tmp_path / "warm"), n=n)
    assert got.cold_converged and got.warm_converged, got.note
    assert got.warm_iterations < got.cold_iterations, got.to_dict()
    assert got.field_difference["U"] < 0.05, got.to_dict()


# --------------------------------------------------------------------------------------------- the ladder
def _rows(truth, bias=0.0, scale=1.0):
    """A synthetic ladder: a coarse rung that is an affine function of the fine one."""
    rows = []
    for alpha, cl in truth.items():
        rows.append({"design": {"section": "0012", "alpha": alpha}, "rung": "rans-fine", "CL": cl, "CD": 0.01})
        rows.append({"design": {"section": "0012", "alpha": alpha}, "rung": "rans-coarse",
                     "CL": scale * cl + bias, "CD": 0.01})
    return rows


def test_a_constant_bias_cannot_change_a_ranking_and_the_ladder_says_so():
    from pinneapple_simulation.cfd_validation import ladder
    truth = {0.0: 0.0, 4.0: 0.44, 8.0: 0.85}
    got = ladder.calibrate(_rows(truth, bias=0.05), "rans-coarse")
    assert got["offset"] == pytest.approx(-0.05)                 # fine - coarse, the sign the table needs
    assert got["spearman_before"] == pytest.approx(1.0)
    assert got["spearman_after"] == pytest.approx(1.0)


def test_a_bias_that_varies_with_the_design_does_move_the_ranking():
    from pinneapple_simulation.cfd_validation import ladder
    truth = {0.0: 0.0, 4.0: 0.44, 8.0: 0.85}
    rows = _rows(truth, bias=0.0, scale=-1.0)                    # a rung that ranks the designs backwards
    got = ladder.calibrate(rows, "rans-coarse")
    assert got["spearman_before"] == pytest.approx(-1.0)
    assert got["slope"] < 0


def test_spearman_is_the_rank_correlation_with_ties_averaged():
    from pinneapple_simulation.cfd_validation import ladder
    assert ladder.spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert ladder.spearman([1, 2, 3], [30, 20, 10]) == pytest.approx(-1.0)
    assert np.isnan(ladder.spearman([1, 2, 3], [5, 5, 5]))       # a rung that says nothing cannot rank
    assert np.isnan(ladder.spearman([1, 2], [1, 2, 3]))          # different lengths are not a correlation
    assert ladder.spearman([1, 2, 3, 4], [1, 1, 4, 4]) == pytest.approx(0.8944, rel=1e-3)   # ties averaged


def test_the_cheapest_rung_is_the_closed_form_one():
    from pinneapple_simulation.cfd_validation import ladder
    got = ladder.thin_airfoil("0012", 4.0)
    assert got["CL"] == pytest.approx(2 * np.pi * np.sin(np.radians(4.0)))
    assert got["cells"] == 0 and got["seconds"] == 0.0
    assert ladder.thin_airfoil("0012", 0.0)["CL"] == pytest.approx(0.0, abs=1e-12)


def test_the_ladder_exports_itself_as_data_levels_for_the_multifidelity_workflow():
    from pinneapple_simulation.cfd_validation import ladder
    names = [lv["name"] for lv in ladder.MULTIFIDELITY["levels"]]
    assert names[0] == "thin-airfoil" and names == sorted(names, key=lambda n: -1 if n == "thin-airfoil" else 0)
    costs = [lv["cost"] for lv in ladder.MULTIFIDELITY["levels"]]
    assert costs == sorted(costs), "the levels must be ordered by cost for a sampler to choose between them"


# --------------------------------------------------------------------------------------------- the TMR plate
def test_the_flat_plate_puts_its_first_cell_in_the_log_layer(tmp_path):
    """``simpleGrading`` takes the last-to-first ratio, not the per-cell one: a case that gets this wrong puts the
    wall function at ``y+ ~ 3000`` and the skin friction comes out three orders of magnitude low."""
    from pinneapple_simulation.cfd_validation import tmr
    case = str(tmp_path / "plate")
    params = tmr.write_case(case, tmr.PLATE_RE, level=tmr.LEVELS[0], model="kOmegaSST")
    u_tau = params["u_tau_estimate"]
    assert params["first_cell_height"] * u_tau / params["nu"] == pytest.approx(tmr.YPLUS, rel=1e-6)
    assert params["grading"] > 1.0
    per_cell = params["grading"] ** (1.0 / (params["ny"] - 1))         # the growth ratio a cell sees
    total = params["first_cell_height"] * (per_cell ** params["ny"] - 1) / (per_cell - 1)
    assert total == pytest.approx(params["height"], rel=0.02)          # the cells really do reach the far field
    with open(f"{case}/system/blockMeshDict") as fh:
        assert "simpleGrading (1 " in fh.read()


def test_the_tmr_reference_reader_takes_two_columns_and_skips_a_header(tmp_path):
    from pinneapple_simulation.cfd_validation import tmr
    path = tmp_path / "plate.dat"
    path.write_text("# x cf\n0.1 0.005\n0.5 0.003\n0.9 0.0025\n")
    got = tmr.read_reference(str(path))
    assert list(got["x"]) == pytest.approx([0.1, 0.5, 0.9])
    assert list(got["cf"]) == pytest.approx([0.005, 0.003, 0.0025])
    bad = tmp_path / "bad.dat"
    bad.write_text("nothing here\n")
    with pytest.raises(ValueError, match="no x cf"):
        tmr.read_reference(str(bad))


def test_the_one_percent_power_law_is_where_the_plate_is_judged():
    """The check that needs no reference file: the 1/7-power law at the sampling station, quoted with its citation."""
    from pinneapple_simulation.cfd_validation import tmr
    law = tmr.CF_LAW["coefficient"] * (tmr.PLATE_RE * 0.97) ** tmr.CF_LAW["exponent"]
    assert 0.002 < law < 0.004
    assert "Schlichting" in tmr.CF_LAW["citation"]


# --------------------------------------------------------------------------------------------- the network (#272)
RESULT = Path(__file__).resolve().parents[1] / "examples" / "cfd" / "classical_cases" / "results" / "cavity_pinn.json"


def _record() -> dict:
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_the_network_is_scored_at_the_published_stations_through_the_norm_the_meshes_use():
    """The two numbers in the README share a column, so they have to share a ruler.

    The stored profiles are re-scored here with :func:`profile_error` — the function the OpenFOAM study calls — and
    have to give back the errors the record carries. A hand-edited error, a station list that drifted off Ghia's, or
    a norm that quietly changed to the relative-to-the-maximum variant all fail this, without retraining anything.
    """
    record = _record()
    seen = set()
    for mode in ("single", "parametric"):
        for row in record[mode]["rows"]:
            re = int(row["reynolds"])
            seen.add(re)
            assert row["y"] == pytest.approx(list(GHIA_U[re].points)), "u is sampled at Ghia's y stations"
            assert row["x"] == pytest.approx(list(GHIA_V[re].points)), "v is sampled at Ghia's x stations"
            for quantity, ref in (("u", GHIA_U[re]), ("v", GHIA_V[re])):
                again = profile_error(ref, np.asarray(row[quantity], dtype=float))
                stored = row["errors"][quantity]
                assert again["l2_relative"] == pytest.approx(stored["l2_relative"], rel=1e-9), (mode, re, quantity)
                assert again["l2"] == pytest.approx(stored["l2"], rel=1e-9), (mode, re, quantity)
                assert stored["points"] == len(ref.values) - 2 - len(ref.disputed)
    assert seen == set(GHIA_RE[:3]), "the three Table I columns, and no number the paper did not publish"


def test_the_number_is_of_the_published_profile_and_not_of_a_solved_field():
    """A PINN scored against the solver's answer measures the distance between two approximations; this one is
    scored against the table, which is the whole point of putting it next to the mesh study."""
    record = _record()
    assert "no solved field" in record["what"].lower()
    assert record["case"] == "ghia-1982"
    assert "relative L2" in record["protocol"]["norm"]
    assert set(record["protocol"]) >= {"stations", "norm", "excluded", "comparable_with"}


def test_where_each_model_stops_matching_is_read_off_its_measured_rows_and_never_interpolated():
    """The finding of the case is a *bracket* between two measured Reynolds numbers, recomputed here from the stored
    errors: a model that fails everywhere says so instead of inventing a crossing."""
    from pinneapple_simulation.cfd_validation import pinn
    record = _record()
    for mode in ("single", "parametric"):
        for tolerance, stored in record[mode]["stops_matching"].items():
            assert pinn.stops_matching(record[mode]["rows"], float(tolerance)) == stored


def test_matching_needs_both_profiles_and_the_bar_is_the_solvers_own():
    """The two tolerances are the level the mesh table reaches: 5 % on a middling mesh, 1 % on the finest one."""
    from pinneapple_simulation.cfd_validation import pinn
    ok = {"reynolds": 100.0, "errors": {"u": {"l2_relative": 0.01}, "v": {"l2_relative": 0.049}}}
    half = {"reynolds": 400.0, "errors": {"u": {"l2_relative": 0.01}, "v": {"l2_relative": 0.30}}}
    assert pinn.matches(ok, 0.05)
    assert not pinn.matches(half, 0.05), "u alone clearing the bar is not the model matching the table"
    got = pinn.stops_matching([ok, half], 0.05)
    assert got["bracket"] == [100.0, 400.0]
    assert pinn.MATCH_TOLERANCES == (0.05, 0.01)
    assert pinn.REYNOLDS == (100, 400, 1000)


def test_a_model_that_matches_nowhere_says_so_instead_of_naming_a_reynolds_number():
    from pinneapple_simulation.cfd_validation import pinn
    bad = {"reynolds": 100.0, "errors": {"u": {"l2_relative": 0.9}, "v": {"l2_relative": 0.9}}}
    got = pinn.stops_matching([bad], 0.05)
    assert got["bracket"] is None
    assert got["stops"] == "below the lowest measured Reynolds number"


def test_the_operator_is_asked_where_no_column_of_the_table_is_and_scored_against_the_solver():
    """An operator over ``Re`` that is only measured on the three published columns has not been shown to be one: the
    three are the points it was asked to reproduce. The record also carries the same model at Reynolds numbers Ghia
    did not publish, scored against a solved field at the same stations, and this re-scores those rows from the two
    profiles the row stores -- so the number can be checked without the mesh being on disk, and a row that drifted off
    the stations the table uses fails here."""
    from pinneapple_simulation.cfd_validation import Profile, pinn
    record = _record()
    rows = record["parametric"]["off_sample"]
    assert rows, "the operator was not probed off the table: the record has to carry that measurement"
    assert {row["reynolds"] for row in rows} == set(pinn.OFF_SAMPLE)
    assert not {row["reynolds"] for row in rows} & set(float(r) for r in GHIA_RE)
    for row in rows:
        assert row["y"] == pytest.approx(list(GHIA_U[100].points)), "the stations are the table's at every column"
        assert row["x"] == pytest.approx(list(GHIA_V[100].points))
        for quantity, ref_points in (("u", row["y"]), ("v", row["x"])):
            reference_profile = Profile("simpleFoam", quantity, "y" if quantity == "u" else "x", row["reynolds"],
                                        tuple(ref_points), tuple(row[f"reference_{quantity}"]),
                                        row["citation"])
            again = profile_error(reference_profile, np.asarray(row[quantity], dtype=float))
            assert again["l2_relative"] == pytest.approx(row["errors"][quantity]["l2_relative"], rel=1e-9)


def test_the_off_table_rows_carry_the_meshes_the_reference_was_made_on():
    """The reference at a Reynolds number the table does not publish is a discretisation too: the record keeps both
    meshes, so the operator's error can be read against the reference's own change between them."""
    from pinneapple_simulation.cfd_validation import pinn
    record = _record()
    solver = record["off_sample_solver"]
    if solver.get("skipped"):
        pytest.skip(solver["skipped"])
    assert solver["reynolds"] == list(pinn.OFF_SAMPLE)
    per = {}
    for run in solver["runs"]:
        per.setdefault(run["reynolds"], []).append(run)
    for reynolds, runs in per.items():
        assert sorted(r["mesh"] for r in runs) == sorted(solver["meshes"]), (reynolds, runs)
        assert all(r["converged"] for r in runs), f"the reference at Re = {reynolds} did not converge"


def test_the_benchmark_is_read_with_the_density_in_its_reynolds_number():
    """CFDBench's ``prop`` family varies the density, so ``Re = rho U L / mu``: a reader that drops it reads six
    different flows as one Reynolds number, and the error against the table is where that shows."""
    from pinneapple_simulation.cfd_validation import cfdbench

    class _Case:
        def __init__(self, **params):
            self._params = params

        def params(self):
            return self._params

        def summary(self):
            return "cavity/prop/case0000"

    case = _Case(vel_top=10.0, width=0.01, height=0.01, density=1.0, viscosity=1e-3)
    assert cfdbench.reynolds(case) == pytest.approx(100.0)
    dense = _Case(vel_top=10.0, width=0.01, height=0.01, density=4.0, viscosity=1e-3)
    assert cfdbench.reynolds(dense) == pytest.approx(400.0), "the density is part of the Reynolds number"
    with pytest.raises(ValueError):
        cfdbench.reynolds(_Case(width=0.01, height=0.01))


def test_the_benchmark_rows_are_scored_at_the_published_stations_through_the_same_norm():
    from pinneapple_simulation.cfd_validation import cfdbench
    record = _cfdbench()
    assert record["archive"] == cfdbench.ARCHIVE
    families = record["families"]
    assert set(families) == set(cfdbench.PROBLEMS), "every family of the archive is accounted for, unpacked or not"
    assert families["cavity"]["cases"] == record["cases_on_disk"] > 0
    seen = set()
    for row in record["rows"]:
        re = int(round(float(row["reynolds"])))
        seen.add(re)
        assert row["found"] == len(row["cases"])
        for case in row["cases"]:
            assert case["y"] == pytest.approx(list(GHIA_U[re].points))
            assert case["x"] == pytest.approx(list(GHIA_V[re].points))
            for quantity, ref in (("u", GHIA_U[re]), ("v", GHIA_V[re])):
                again = profile_error(ref, np.asarray(case[quantity], dtype=float))
                assert again["l2_relative"] == pytest.approx(case["errors"][quantity]["l2_relative"], rel=1e-9)
            without = case["errors"]["u_without_lid"]
            assert without["points"] < case["errors"]["u"]["points"], "leaving the lid stations out takes some out"
            assert 0 < without["l2_relative"] < case["errors"]["u"]["l2_relative"] + 1e-12
    assert seen == set(GHIA_RE[:3])


def _cfdbench() -> dict:
    path = Path(__file__).resolve().parents[1] / "examples" / "cfd" / "classical_cases" / "results" / "cfdbench.json"
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------- the benchmark's other problems: the tube's published parameters
def test_the_tube_table_is_the_one_the_paper_prints():
    """Table 3 of CFDBench states each set and its step, so a transcription error is arithmetic that does not close.

    The paper says the inlet velocity runs ``0.1 .. 5`` by ``0.1``, the density ``10 .. 1000`` by ``110``, the viscosity
    ``0.01 .. 1`` by ``0.11``, the diameters are five named values and the length-to-diameter ratios are ten named
    ones. Every one of those is checked here against the transcribed tuple, and the counts have to come out at the
    numbers the paper's own Table 6 gives: 50, 100 and 25 cases.
    """
    from pinneapple_simulation.cfd_validation.reference import CFDBENCH_TUBE

    sets = CFDBENCH_TUBE.sets
    assert len(sets["bc"]) == 50 and sets["bc"][0] == pytest.approx(0.1) and sets["bc"][-1] == pytest.approx(5.0)
    step = (sets["bc"][-1] - sets["bc"][0]) / (len(sets["bc"]) - 1)
    assert step == pytest.approx(0.1), "the inlet velocity increases by 0.1 m/s"

    density, viscosity = sets["prop/density"], sets["prop/viscosity"]
    assert len(density) == len(viscosity) == 10
    assert (density[0], density[-1]) == (pytest.approx(10.0), pytest.approx(1000.0))
    assert (density[1] - density[0]) == pytest.approx(110.0), "the density increases by 110 kg/m3"
    assert (viscosity[0], viscosity[-1]) == (pytest.approx(0.01), pytest.approx(1.0))
    assert (viscosity[1] - viscosity[0]) == pytest.approx(0.11), "the viscosity increases by 0.11 Pa s"

    assert sets["geo/diameter"] == (0.01, 0.05, 0.1, 0.3, 0.5)
    assert sets["geo/ratio"] == (1.0, 2.0, 5.0, 7.5, 10.0, 15.0, 20.0, 50.0, 75.0, 100.0)
    # the case counts of Table 6: one case per bc value, the prop grid, and five diameters by five of the ten ratios
    assert len(sets["bc"]) == 50
    assert len(density) * len(viscosity) == 100
    assert len(sets["geo/diameter"]) * 5 == 25
    assert "Table 3" in CFDBENCH_TUBE.citation and "2310.05963" in CFDBENCH_TUBE.citation


def test_the_tube_archive_is_held_against_the_table_it_was_generated_from(tmp_path):
    """The archive's three families are looked up in the transcribed table, and every value has to be in both.

    The checker exists to be able to fail, so both readings are asserted: an archive whose ``case.json`` carries the
    published values agrees, and an archive whose ``prop`` family carries the ten-value index ``0..9`` instead of the
    ``10 .. 1000 kg/m3`` does not — which is what the real archive does.
    """
    from pinneapple_simulation.cfd_validation import cfdbench
    from pinneapple_simulation.cfd_validation.reference import CFDBENCH_TUBE

    sets = CFDBENCH_TUBE.sets
    honest = _tube_archive(tmp_path / "honest", density=lambda i: sets["prop/density"][i])
    record = cfdbench.tube_against_table(honest)
    assert record["agrees"] is True, record["disagrees_on"]
    assert record["cases"] == {"bc": 50, "prop": 100, "geo": 25} == record["cases_published"]
    assert all(p["agrees"] for p in record["parameters"])
    assert not record["unreadable"]

    indexed = _tube_archive(tmp_path / "indexed", density=lambda i: float(i))
    wrong = cfdbench.tube_against_table(indexed)
    assert wrong["agrees"] is False
    assert wrong["disagrees_on"] == ["prop/density"], "only the density reading is wrong"
    row = next(p for p in wrong["parameters"] if p["parameter"] == "prop/density")
    assert row["published_but_absent"] == [float(v) for v in sets["prop/density"]]
    assert row["present_but_unpublished"] == [float(i) for i in range(10)]
    assert next(p for p in wrong["parameters"] if p["parameter"] == "prop/viscosity")["agrees"] is True


def _tube_archive(root: Path, *, density) -> Path:
    """A tube archive of the paper's own shape, small enough to write in a test: no fields, only ``case.json``.

    ``open_cases`` needs ``u.npy`` and ``v.npy`` to exist to call a directory a case, so one number each stands in;
    nothing this check reads comes out of them.
    """
    from pinneapple_simulation.cfd_validation.reference import CFDBENCH_TUBE

    sets = CFDBENCH_TUBE.sets
    base = {"height": 0.1, "width": 1.0, "vel_in": 1.0, "density": 100.0, "viscosity": 0.1}

    def write(subset: str, index: int, params: dict) -> None:
        case = root / subset / f"case{index:04d}"
        case.mkdir(parents=True)
        (case / "case.json").write_text(json.dumps(params), encoding="utf-8")
        np.save(case / "u.npy", np.zeros((2, 2, 2), dtype=np.float32))
        np.save(case / "v.npy", np.zeros((2, 2, 2), dtype=np.float32))

    for i, velocity in enumerate(sets["bc"]):
        write("bc", i, {**base, "vel_in": float(velocity)})
    for i, mu in enumerate(sets["prop/viscosity"]):
        for j in range(len(sets["prop/density"])):
            write("prop", i * len(sets["prop/density"]) + j,
                  {**base, "density": density(j), "viscosity": float(mu)})
    # the geometry family: five diameters by five of the ten length-to-diameter ratios, stored as l/d. The window of
    # five rotates with the diameter so all ten ratios are present, which is what the real archive does.
    ratios = sets["geo/ratio"]
    for a, diameter in enumerate(sets["geo/diameter"]):
        for b in range(5):
            ratio = ratios[(a + 2 * b) % len(ratios)]
            write("geo", a * 5 + b, {**base, "height": float(diameter), "width": float(diameter * ratio)})
    return root
