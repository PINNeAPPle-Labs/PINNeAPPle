"""The generic simulation factory: geometry -> DOE -> solver -> dataset -> quality control (#152).

Every number asserted here is one the run measured: the volume the template's closed form gives for a design, the
outputs the injected simulator returned, the number of rows that reached the dataset and the status of each check.
The simulator is either an analytic :class:`FunctionSimulator` (declared as such, never presented as a solver) or a
small one written here to exercise the metadata the quality control reads. Nothing is asserted about a design the
factory did not evaluate.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os

import pytest

from pinneapple_analysis.design_study import (
    Base,
    Budget,
    DesignStudy,
    Evaluator,
    Objective,
    Quantity,
    Variable,
)
from pinneapple_autolab import FunctionSimulator, SimOutcome, Simulator
from pinneapple_autolab.factory import (
    REJECTED,
    FactoryQc,
    FactoryRow,
    GivenDesigns,
    MaximinDoe,
    ProvenanceError,
    StudyDatasetWriter,
    TemplateGeometry,
    row_verdict,
    run_factory,
    template_names,
)
from pinneapple_autolab.factory.domains import pipe_erosion, ugs


# ---------------------------------------------------------------------------------------------- the case
def bend_spec(n_doe: int = 4, seed: int = 0, outputs=(("dp", "Pa"), ("er", "mm/year"))) -> DesignStudy:
    """The bend case: two variables, the two outputs of the product's dataset and a budget of 1 + n_doe runs."""
    return DesignStudy(
        kind="surrogate", title="bend",
        base=Base(kind="geometry", ref="pp.geometries:pipe_bend_90",
                  variables=[Variable("diameter", 0.05, 0.2, unit="m"),
                             Variable("bend_ratio", 1.0, 3.0, unit="R/D")],
                  baseline={"diameter": 0.0762, "bend_ratio": 1.5}),
        evaluator=Evaluator(tool="toy.bend", inputs={"diameter": "diameter", "bend_ratio": "bend_ratio"},
                            outputs=[Quantity(k, u) for k, u in outputs], conditions={"speed_m_s": 20.0}),
        objectives=[Objective(outputs[0][0], "min")],
        budget=Budget(max_evaluations=1 + n_doe, n_doe=n_doe, n_verify=0, seed=seed))


def bend_model(p):
    """A model, not a solver: dp ~ 900 R/D over D and a wall-loss rate linear in R/D. Both are toy functions."""
    return {"dp": 900.0 * float(p["bend_ratio"]) / float(p["diameter"]), "er": 0.1 * float(p["bend_ratio"])}


def bend_simulator():
    return FunctionSimulator(bend_model, ["dp", "er"], name="toy bend model")


class MetaSim(Simulator):
    """A simulator that records what a real one records: residuals, the mesh and a GCI, in ``SimOutcome.meta``."""

    name, fidelity, outputs = "meta simulator", "reference", ("dp",)

    def __init__(self, meta, ok: bool = True, message: str = ""):
        self.meta, self.ok, self.message = dict(meta), bool(ok), message

    def run(self, params, workdir):
        return SimOutcome({"dp": 1.0 / float(params["diameter"])} if self.ok else {}, ok=self.ok,
                          message=self.message, cost_s=0.0, meta=dict(self.meta))


# ---------------------------------------------------------------------------------------------- geometry
def test_the_pipe_bend_geometry_carries_what_a_solver_needs():
    art = TemplateGeometry("pp.geometries:pipe_bend_90").prepare({"diameter": 0.0762, "bend_ratio": 1.5})
    d, r = 0.0762, 1.5 * 0.0762
    assert art.params["inlet_length"] == pytest.approx(max(10 * d, 0.762))
    assert art.params["outlet_length"] == pytest.approx(max(5 * d, 0.381))
    exact = math.pi * d ** 2 / 4 * (art.params["inlet_length"] + art.params["outlet_length"] + math.pi * r / 2)
    assert art.metrics["volume_m3_exact"] == pytest.approx(exact)                 # Pappus, the template's own
    assert art.metrics["ref_length_m"] == pytest.approx(d)                        # the Reynolds length
    assert art.metrics["min_feature_m"] == pytest.approx(d)                       # what drives the mesh
    assert art.metrics["ports"] == {"inlet": "<X", "outlet": ">Y"}
    assert art.metrics["inside"][0] == pytest.approx(-art.params["inlet_length"] / 2)
    assert art.ok and art.status == "parameters only"                             # no CAD kernel in this gate


def test_the_derived_development_lengths_are_the_ones_built():
    art = TemplateGeometry("pipe_bend_90").prepare({"diameter": 0.2, "bend_ratio": 1.0, "inlet_length": 0.3})
    assert art.params["inlet_length"] == pytest.approx(2.0)                        # 10 D, not the 0.3 asked for
    assert art.params["outlet_length"] == pytest.approx(1.0)                       # 5 D
    assert art.status == "parameters only"


def test_without_the_cad_kernel_the_solid_checks_are_not_run_and_say_so():
    art = TemplateGeometry("pipe_bend_90").prepare({"diameter": 0.0762, "bend_ratio": 1.5})
    by_name = {c.name: c for c in art.checks}
    assert by_name["parameters within the template's validity limits"].status == "supports"
    for name in ("CAD build", "solid volume against the closed form", "STL watertight"):
        assert by_name[name].status == "not_run" and "cadquery" in by_name[name].reason


def test_a_design_outside_the_template_limits_is_refused_with_the_limit():
    art = TemplateGeometry("pipe_bend_90").prepare({"diameter": 0.001, "bend_ratio": 1.5})   # below the 2 mm floor
    assert art.status == "refused" and not art.ok
    assert "diameter = 0.001 m is outside the valid range 0.002-5" in art.reasons[0]


# ---------------------------------------------------------------------------------------------- end to end
def test_the_factory_writes_one_dataset_row_per_run_with_its_provenance(tmp_path):
    run = run_factory(bend_spec(), bend_simulator(), out=str(tmp_path / "out"))
    # a model that records no convergence, mesh or conservation evidence is written but not signed
    assert run.status == "inconclusive" and run.decision == "REVIEW"
    assert len(run.rows) == 5 and len(run.accepted) == 5 and run.dataset["n_rows"] == 5

    from pinneapple_autolab.study.dataset import StudyDataset
    ds = StudyDataset.load(os.path.join(str(tmp_path / "out"), "dataset"))
    assert sorted(ds.ids) == sorted(r.id for r in run.rows)
    assert ds.inputs == ["diameter", "bend_ratio"] and ds.outputs == ["dp", "er"]
    assert ds.units == {"diameter": "m", "bend_ratio": "R/D", "dp": "Pa", "er": "mm/year"}
    # the numbers in the dataset are the ones the simulator returned, design by design
    for row in run.rows:
        j = ds.ids.index(row.id)
        for k, v in row.outputs.items():
            assert ds.Y[j, ds.outputs.index(k)] == pytest.approx(v)

    rows = json.load(open(os.path.join(str(tmp_path / "out"), "rows.json")))
    assert len(rows) == 5 and all(r["provenance"]["commit"] and r["provenance"]["data_hash"] for r in rows)
    assert all(r["provenance"]["solver"] == "toy.bend" and r["provenance"]["seed"] == 0 for r in rows)
    assert all(r["geometry"]["status"] == "parameters only" for r in rows)
    assert all(r["checks"] and any(c["name"] == "simulation ran" for c in r["checks"]) for r in rows)


def test_the_baseline_opens_the_run_and_the_doe_stays_inside_the_budget(tmp_path):
    spec = bend_spec(n_doe=6)
    run = run_factory(spec, bend_simulator(), out=str(tmp_path / "out"), doe=MaximinDoe())
    plan = spec.budget.planned(len(spec.base.names))
    assert len(run.rows) == 1 + plan["doe"] <= spec.budget.max_evaluations
    first = run.rows[0]
    assert first.purpose == "baseline" and first.params == spec.base.baseline_design()
    for row in run.rows[1:]:
        assert 0.05 <= row.params["diameter"] <= 0.2 and 1.0 <= row.params["bend_ratio"] <= 3.0
    assert run.doe["name"] == "maximin" and "discrepancy" in run.doe["coverage"]


def test_a_failed_run_stays_on_record_and_never_reaches_the_dataset(tmp_path):
    def fails(p):
        return "diverged" if float(p["bend_ratio"]) > 2.0 else None
    sim = FunctionSimulator(bend_model, ["dp", "er"], fail=fails)
    run = run_factory(bend_spec(n_doe=6), sim, out=str(tmp_path / "out"))
    assert len(run.rows) == 7
    failed = [r for r in run.rows if not r.accepted]
    assert failed and all(r.verdict == REJECTED and not r.outputs for r in failed)
    assert all("diverged" in " ".join(r.reasons) for r in failed)
    assert run.dataset["n_rows"] == len(run.rows) - len(failed)
    assert next(c for c in run.qc["checks"] if c["name"] == "no rejected row was written")["status"] == "supports"


def test_a_solver_that_raises_is_a_failed_run_not_a_crash(tmp_path):
    def boom(p):
        raise RuntimeError("mesh generation failed")
    run = run_factory(bend_spec(n_doe=3), FunctionSimulator(boom, ["dp", "er"]), out=str(tmp_path / "out"))
    assert run.rows and all(not r.accepted for r in run.rows)
    assert run.status == "refused" and "nothing to put in a dataset" in run.reasons[-1]


def test_the_parameters_the_case_fixes_reach_the_geometry_and_not_the_tool(tmp_path):
    """``Base.fixed`` is the case's own geometry — the runs a line needs before and after a bend — so the factory
    hands it to the CAD arrow, keeps it out of the dataset's columns and leaves the solver's inputs alone."""
    spec = dataclasses.replace(bend_spec(), base=dataclasses.replace(
        bend_spec().base, fixed={"inlet_length": 1.5, "outlet_length": 1.0}))
    run = run_factory(spec, bend_simulator(), out=str(tmp_path / "out"),
                      doe=GivenDesigns([{"diameter": 0.05, "bend_ratio": 1.5}]))
    for row in run.rows:
        d, r = row.params["diameter"], 1.5 * row.params["diameter"]
        assert row.artifact.params["inlet_length"] == pytest.approx(1.5)    # the case's own run, not the template's
        assert row.artifact.params["outlet_length"] == pytest.approx(1.0)   # default of 5 D, 0.25-1.0 m here
        assert row.artifact.metrics["volume_m3_exact"] == pytest.approx(
            math.pi * d ** 2 / 4 * (1.5 + 1.0 + math.pi * r / 2))
        assert "inlet_length" not in row.tool_params                        # the tool's inputs are the evaluator's
    assert [row.params for row in run.rows][-1] == {"diameter": 0.05, "bend_ratio": 1.5}   # the design as given
    from pinneapple_autolab.study.dataset import StudyDataset
    assert StudyDataset.load(os.path.join(str(tmp_path / "out"), "dataset")).inputs == ["diameter", "bend_ratio"]


def test_a_fixed_parameter_the_geometry_does_not_know_refuses_the_row_instead_of_being_dropped(tmp_path):
    spec = dataclasses.replace(bend_spec(), base=dataclasses.replace(bend_spec().base, fixed={"span": 0.1}))
    run = run_factory(spec, bend_simulator(), out=str(tmp_path / "out"))
    assert run.rows and all(not r.accepted and r.artifact.status == "refused" for r in run.rows)
    assert all(r.verdict == REJECTED for r in run.rows)                    # a refused design is rejected, not kept
    assert all("span" in " ".join(r.reasons) for r in run.rows)            # the reason names the parameter
    assert all(not r.outputs for r in run.rows)                            # and nothing was simulated
    assert run.status == "refused"


def test_the_fixed_parameters_are_part_of_what_provenance_records(tmp_path):
    def hashed(inlet: float, folder: str) -> str:
        spec = dataclasses.replace(bend_spec(), base=dataclasses.replace(
            bend_spec().base, fixed={"inlet_length": inlet, "outlet_length": 1.0}))
        run = run_factory(spec, bend_simulator(), out=str(tmp_path / folder),
                          doe=GivenDesigns([{"diameter": 0.05, "bend_ratio": 1.5}]))
        return run.rows[0].provenance["data_hash"]

    assert hashed(1.5, "a") != hashed(1.6, "b")        # two cases that differ only in a fixed size differ in the hash


# ---------------------------------------------------------------------------------------------- quality control
def test_convergence_and_mesh_checks_read_the_simulator_metadata(tmp_path):
    meta = {"final_residuals": {"p": 1e-5, "U": 2e-5}, "mesh": {"cells": 84_223, "ok": True}, "gci": 0.011}
    run = run_factory(bend_spec(n_doe=3, outputs=(("dp", "Pa"),)), MetaSim(meta), out=str(tmp_path / "out"))
    by_name = {c.name: c for c in run.rows[0].checks}
    assert by_name["converged"].status == "supports" and by_name["converged"].value == pytest.approx(2e-5)
    assert by_name["mesh quality"].status == "supports" and by_name["mesh quality"].value == 84_223.0
    assert by_name["mesh convergence (GCI)"].status == "supports"     # 1.1 % against the 10 % limit
    assert run.decision == "APPROVED"
    assert run.rows[0].provenance["mesh"] == {"cells": 84_223}        # the mesh travelled to the provenance


def test_a_run_that_did_not_converge_is_kept_but_the_dataset_is_not_signed(tmp_path):
    meta = {"final_residuals": {"p": 5e-2}, "mesh": {"cells": 1000, "ok": True}}
    run = run_factory(bend_spec(n_doe=3, outputs=(("dp", "Pa"),)), MetaSim(meta), out=str(tmp_path / "out"))
    by_name = {c.name: c for c in run.rows[0].checks}
    assert by_name["converged"].status == "contradicts" and by_name["converged"].critical is False
    assert run.rows[0].accepted                    # a non-critical contradiction keeps the row
    assert run.status == "inconclusive" and run.decision == "REVIEW"
    assert any("convergence evidence" in m for m in run.notes)


def test_the_gci_limit_is_the_declared_one_and_an_absent_gci_is_not_run():
    assert FactoryQc()._gci({}).status == "not_run"
    assert FactoryQc()._gci({"gci": 0.5}).status == "contradicts"            # 50 % of the answer vs 10 % allowed
    assert FactoryQc({"gci": 0.6})._gci({"gci": 0.5}).status == "supports"
    assert FactoryQc()._gci({"gci": 0.05, "gci_limit": 0.01}).status == "contradicts"


def test_the_convergence_check_uses_the_solver_own_residual_limit(tmp_path):
    meta = {"final_residuals": {"p": 1e-2}, "residual_limit": 5e-2}
    run = run_factory(bend_spec(n_doe=3, outputs=(("dp", "Pa"),)), MetaSim(meta), out=str(tmp_path / "out"))
    check = next(c for c in run.rows[0].checks if c.name == "converged")
    assert check.status == "supports" and check.limit == pytest.approx(5e-2)


def test_a_row_without_provenance_makes_the_writer_refuse_the_whole_dataset(tmp_path):
    spec = bend_spec(n_doe=3)
    good = FactoryRow(id="a", purpose="doe", params={"diameter": 0.1, "bend_ratio": 1.5},
                      outputs={"dp": 1.0, "er": 0.1}, ok=True,
                      provenance={"solver": "toy.bend", "library_version": "0", "seed": 0,
                                  "data_hash": "sha256:00", "commit": "abc"})
    bad = FactoryRow(id="b", purpose="doe", params={"diameter": 0.11, "bend_ratio": 1.5},
                     outputs={"dp": 2.0, "er": 0.2}, ok=True, provenance={})
    with pytest.raises(ProvenanceError) as err:
        StudyDatasetWriter().write([good, bad], spec, str(tmp_path / "out"))
    assert "b:" in str(err.value) and "solver is missing" in str(err.value)
    assert not os.path.exists(os.path.join(str(tmp_path / "out"), "dataset.json"))
    assert StudyDatasetWriter().write([good], spec, str(tmp_path / "ok")).n_rows == 1


def test_a_design_outside_its_bounds_is_refused_by_the_doe():
    spec = bend_spec(n_doe=3)
    doe = GivenDesigns([{"diameter": 0.9, "bend_ratio": 1.5}])
    with pytest.raises(ValueError, match="outside"):
        doe.plan(spec.base.variables, 1)


def test_the_doe_can_replay_measured_designs():
    spec = bend_spec(n_doe=3)
    designs = [{"diameter": 0.1, "bend_ratio": 2.0}, {"diameter": 0.15, "bend_ratio": 1.2}]
    assert GivenDesigns(designs).plan(spec.base.variables, 2) == designs


# ---------------------------------------------------------------------------------------------- the report
def test_the_run_renders_as_a_page():
    import tempfile
    run = run_factory(bend_spec(n_doe=3), bend_simulator(), out=tempfile.mkdtemp())
    page = run.to_markdown()
    assert page.startswith("# Factory run: bend") and "Status **inconclusive**" in page
    assert "| baseline000 | baseline |" in page
    assert "## Quality control" in page and "every row has provenance" in page


# ---------------------------------------------------------------------------------------------- the cases
def test_an_unknown_geometry_is_refused_with_the_templates_it_could_name():
    spec = DesignStudy(kind="surrogate", title="typo",
                       base=Base(kind="geometry", ref="pp.geometries:nope",
                                 variables=[Variable("diameter", 0.05, 0.2, unit="m")]),
                       evaluator=Evaluator(tool="toy.bend", inputs={"diameter": "diameter"},
                                           outputs=[Quantity("dp", "Pa")]),
                       objectives=[Objective("dp", "min")])
    run = run_factory(spec, bend_simulator(), out="unused")
    assert run.status == "refused" and "no geometry template" in run.reasons[0]
    assert "pipe_bend_90" in run.reasons[0] and template_names()


def test_the_pipe_erosion_case_runs_up_to_the_solver_and_says_where_it_stops(tmp_path):
    plan = pipe_erosion.plan(seed=1)
    assert plan["n_designs"] == pipe_erosion.spec().budget.planned(2)["doe"]
    assert plan["artifact"]["status"] == "parameters only"
    d = 0.0762
    assert plan["artifact"]["metrics"]["volume_m3_exact"] == pytest.approx(
        math.pi * d ** 2 / 4 * (10 * d + 5 * d + math.pi * 1.5 * d / 2))
    assert all(0.025 <= x["diameter"] <= 0.5 and 1.0 <= x["bend_ratio"] <= 5.0 for x in plan["designs"])

    run = pipe_erosion.run(str(tmp_path / "out"))
    assert run.status == "refused" and run.reasons == [pipe_erosion.SOLVER_MISSING]
    assert run.rows == [] and not os.path.exists(os.path.join(str(tmp_path / "out"), "dataset.json"))


def test_the_pipe_erosion_spec_declares_units_and_the_reference_object():
    spec = pipe_erosion.spec()
    assert spec.evaluator.outputs[0].unit == "Pa" and spec.evaluator.outputs[1].unit == "mm/year"
    assert spec.base.baseline == {"diameter": 0.0762, "bend_ratio": 1.5}     # the Vieira et al. object
    assert "Vieira" in pipe_erosion.REFERENCE
    assert spec.digest() == pipe_erosion.spec().digest()                      # the case is a value, not a run


# ---------------------------------------------------------------------------------------------- the in-house solver
def test_the_reservoir_grid_is_built_and_measured():
    art = ugs.geometry().prepare({"permeability_md": 100.0, "porosity": 0.2})
    m = art.metrics
    assert art.status == "built" and art.ok and m["cells"] == 21 * 21
    assert m["bulk_volume_m3"] == pytest.approx(21 * 21 * 40.0 * 40.0 * 1.0)                 # 40 x 40 x 1 m cells
    assert m["pore_volume_m3"] == pytest.approx(m["bulk_volume_m3"] * 0.2)                  # phi0 = 0.2
    assert m["transmissibility_min"] > 0 and m["transmissibility_min"] == pytest.approx(
        40.0 * 1.0 * 100.0 * ugs.MD / 40.0)                                                 # A k / d
    assert m["well_index_m3"] > 0 and m["ports"]["well"].startswith("P1")
    assert [c.status for c in art.checks] == ["supports", "supports"]


def test_a_porosity_that_is_not_a_porous_medium_is_refused_by_the_grid():
    art = ugs.geometry().prepare({"permeability_md": 100.0, "porosity": 1.0})
    assert art.status == "refused" and "porosity" in art.reasons[0]


def test_the_solver_refuses_a_design_it_cannot_simulate():
    bad = ugs.solver().check({"permeability_md": 0.0, "porosity": 0.2})
    assert bad and "cannot flow" in bad[0]
    assert ugs.solver().check({"permeability_md": 100.0, "porosity": 1.5})                   # outside (0, 1)
    assert ugs.solver().check({"permeability_md": 100.0, "porosity": 0.2,
                               "bhp_floor_Pa": 2.0e7, "initial_pressure_Pa": 1.2e7})          # floor above p0
    assert ugs.solver().check({"permeability_md": 100.0, "porosity": 0.2}) == []


def test_the_storage_case_carries_the_chain_through_the_library_solver(tmp_path):
    run = run_factory(ugs.spec(n_doe=3), ugs.solver(), out=str(tmp_path / "out"), geometry=ugs.geometry())
    assert run.status == "done" and run.decision == "APPROVED" and run.approved
    assert len(run.rows) == 4 and run.dataset["n_rows"] == 4          # 1 baseline + 3 DOE, all written
    assert run.geometry["name"] == "ugs grid" and run.geometry["build"].startswith("pinneapple_simulation.ugs")

    base = run.rows[0]
    assert base.purpose == "baseline" and base.params == {"permeability_md": 100.0, "porosity": 0.2}
    assert base.outputs["peak_deliverability"] == pytest.approx(199_305.5, rel=1e-4)
    assert base.outputs["pressure_drop"] == pytest.approx(1.9, rel=5e-2)
    checks = {c.name: c.status for c in base.checks}
    assert checks["converged"] == "supports" and checks["conservation"] == "supports"
    assert checks["mesh quality"] == "supports" and checks["mesh convergence (GCI)"] == "not_run"
    assert base.provenance["uses_a_mesh"] is True and base.provenance["mesh"] == {"cells": 441}

    ds_dir = os.path.join(str(tmp_path / "out"))
    rows = json.load(open(os.path.join(ds_dir, "rows.json")))
    assert len(rows) == 4 and all(r["outputs"]["peak_deliverability"] > 0 for r in rows)
    # what the dataset carries is what the library's simulator returned for that design
    sim = ugs.solver()
    for row in run.rows[1:]:
        again = sim.run(ugs.spec().evaluator.tool_params(row.params), str(tmp_path / "again"))
        assert again.outputs["peak_deliverability"] == pytest.approx(row.outputs["peak_deliverability"], rel=1e-9)


def test_a_run_that_does_not_conserve_mass_is_rejected():
    class Leaky:
        ok, outputs, message, meta = True, {"peak_deliverability": 1.0, "pressure_drop": 2.0}, "", {
            "converged": True, "mass_balance_error": 0.5, "mesh": {"cells": 441, "ok": True}}

    checks = FactoryQc().row_checks(ugs.geometry().prepare({"permeability_md": 100.0, "porosity": 0.2}), Leaky(),
                                    ugs.spec())
    verdict, reasons = row_verdict(checks)
    assert verdict == "rejected" and any("mass-balance" in r for r in reasons)
