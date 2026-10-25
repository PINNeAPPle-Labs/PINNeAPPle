"""FreeCAD -> OpenFOAM pipelines (pinneapple_simulation.pipelines.freecad_openfoam).

Fast tests use the NumPy stand-in for FreeCAD and only write OpenFOAM dictionaries; the FreeCAD test needs
``freecadcmd`` and the end-to-end run needs OpenFOAM (both skip when the tool is missing).
"""
from __future__ import annotations

import json
import math
import warnings

import numpy as np
import pytest

from pinneapple_physics.closed_form import heat_exchanger, particle_drag, pipe_flow
from pinneapple_simulation.pipelines.freecad_openfoam import (
    MODELS,
    FakeFreeCAD,
    FreeCADBackend,
    MeshSpec,
    PatchSpec,
    Physics,
    PipelineSpec,
    ScalarSpec,
    Sweep,
    assign_patches,
    doe,
    evaluate,
    resolve_physics,
    run_pipeline,
)
from pinneapple_simulation.pipelines.freecad_openfoam.cad import rule_eval, write_stl
from pinneapple_simulation.pipelines.freecad_openfoam.case import write_internal_case
from pinneapple_simulation.pipelines.freecad_openfoam.mesh import (
    allrun_script,
    split_procs,
    write_internal_mesh,
)
from pinneapple_simulation.pipelines.freecad_openfoam.steps import point_inside

HAS_FREECAD = FreeCADBackend().available()[0]


def _openfoam() -> bool:
    from pinneapple_simulation.numerical_solvers.external_flow import openfoam_available
    try:
        return openfoam_available()
    except OSError:
        return False


HAS_OPENFOAM = _openfoam()


# ----------------------------------------------------------------------------------------------- closed forms
def test_reference_correlations():
    assert pipe_flow.dean_number(200, 10, 40) == pytest.approx(200 * math.sqrt(1 / 8))
    assert pipe_flow.curved_pipe_ratio(10.0) == 1.0
    assert pipe_flow.curved_pipe_ratio(70.71) == pytest.approx(1.373, abs=2e-3)   # White (1929)
    assert pipe_flow.curved_pipe_ratio(500) > pipe_flow.curved_pipe_ratio(100) > 1.0
    assert heat_exchanger.hausen_mean_nusselt(100, 0.7, 1e-6) == pytest.approx(3.66, abs=1e-3)
    assert heat_exchanger.hausen_mean_nusselt(100, 2.0, 0.1) == pytest.approx(4.692, abs=1e-3)
    # energy balance inverts: Tout from Nu_m, then Nu_m back
    nu_m, re, pr, dl = 4.5, 100.0, 2.0, 0.1
    t_out = 1 - math.exp(-4 * nu_m / (re * pr * dl))
    assert heat_exchanger.mean_nusselt_from_temperatures(re, pr, dl, 1.0, 0.0, t_out) == pytest.approx(nu_m)
    assert particle_drag.clift_sphere_cd(100) == pytest.approx(1.087, abs=1e-3)
    assert particle_drag.schiller_naumann_cd(100) == pytest.approx(1.092, abs=1e-3)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        particle_drag.clift_sphere_cd(1000)
    assert w and issubclass(w[0].category, RuntimeWarning)


# ----------------------------------------------------------------------------------------------- rules and CAD
def test_rule_eval_is_restricted():
    assert rule_eval("plane and abs(cx + L) < 1e-3 * D", {"plane": True, "cx": -5.0, "L": 5.0, "D": 1.0})
    assert rule_eval("0 < x <= 2", {"x": 2})
    for bad in ("__import__('os')", "x.real", "[i for i in x]", "open('f')"):
        with pytest.raises((ValueError, NameError)):
            rule_eval(bad, {"x": 1})


@pytest.mark.parametrize("name", sorted(MODELS))
def test_fake_templates_are_closed_and_named(tmp_path, name):
    m = MODELS[name]
    fake = FakeFreeCAD()
    fake.build(name, m.defaults(), tmp_path / "doc.json")
    g = fake.export(tmp_path / "doc.json", {}, m.solid, tmp_path / "out")
    assert g.open_edges() == 0
    p = m.defaults()
    exact = {"pipe_bend": math.pi * p.get("D", 0) ** 2 / 4 * (p.get("L_in", 0) + p.get("L_out", 0)
                                                             + math.radians(p.get("angle", 0)) * p.get("Rc", 0)),
             "heated_pipe": math.pi * p.get("D", 0) ** 2 / 4 * (p.get("L_entry", 0) + p.get("L_heated", 0)),
             "spheroid": math.pi / 6 * p.get("D", 0) ** 2 * p.get("L", 0)}[name]
    assert g.volume() == pytest.approx(exact, rel=0.03)        # inscribed polygons: slightly smaller
    a = assign_patches(g, PatchSpec.parse(m.patches), g.info["parameters"])
    assert all(a.values()) and sorted(i for v in a.values() for i in v) == list(range(len(g.faces)))
    stl = write_stl(tmp_path / "g.stl", g, a)
    text = stl.read_text()
    assert all(f"solid {k}" in text for k in a)


def test_fake_parameters_move_the_patches(tmp_path):
    fake = FakeFreeCAD()
    m = MODELS["pipe_bend"]
    fake.build("pipe_bend", m.defaults(), tmp_path / "doc.json")
    g = fake.export(tmp_path / "doc.json", {"Rc": 25.0, "angle": 90.0}, m.solid, tmp_path / "out")
    a = assign_patches(g, PatchSpec.parse(m.patches), g.info["parameters"])
    out = g.faces[a["outlet"][0]]
    assert out.centre[1] == pytest.approx(25.0 + 30.0, abs=0.05)  # 90 degree bend of radius 25 mm, 30 mm leg
    assert out.normal == pytest.approx([0, 1, 0], abs=1e-6)
    with pytest.raises(KeyError):
        fake.export(tmp_path / "doc.json", {"nope": 1.0}, m.solid, tmp_path / "out2")
    V, F = g.mesh()
    assert point_inside(V, F, np.array([-50.0, 0.3, 0.2]))
    assert not point_inside(V, F, np.array([-50.0, 20.0, 0.0]))


def test_unassigned_face_is_an_error(tmp_path):
    fake = FakeFreeCAD()
    fake.build("heated_pipe", MODELS["heated_pipe"].defaults(), tmp_path / "d.json")
    g = fake.export(tmp_path / "d.json", {}, "Fluid", tmp_path / "o")
    with pytest.raises(ValueError, match="matches no patch"):
        assign_patches(g, PatchSpec.parse({"inlet": "plane and cx < 1", "outlet": "plane and cx > 100"}))


# ----------------------------------------------------------------------------------------------- OpenFOAM files
def test_mesh_and_case_dictionaries(tmp_path):
    fake = FakeFreeCAD()
    m = MODELS["heated_pipe"]
    fake.build("heated_pipe", m.defaults(), tmp_path / "d.json")
    g = fake.export(tmp_path / "d.json", {}, "Fluid", tmp_path / "o")
    specs = PatchSpec.parse(m.patches)
    a = assign_patches(g, specs)
    stl = write_stl(tmp_path / "g.stl", g, a)
    case = tmp_path / "case"
    types = {p.name: {"inlet": "patch", "outlet": "patch", "wall": "wall"}[p.kind] for p in specs}
    info = write_internal_mesh(case, stl, types, np.array([0.003, 0.0004, 0.0003]), 0.001, MeshSpec(), procs=4)
    assert info["mesher"] == "snappy" and info["background_cell_m"] == pytest.approx(0.002)
    snappy = (case / "system/snappyHexMeshDict").read_text()
    assert "wall_heated { level (1 1); patchInfo { type wall; } }" in snappy
    assert "locationInMesh (0.003 0.0004 0.0003)" in snappy
    assert "mpirun" in (case / "Allmesh").read_text()
    ph = Physics(velocity=0.01, nu=1e-6, scalar=ScalarSpec("T", 5e-7, 0.0, {"wall_heated": 1.0}))
    write_internal_case(case, {p.name: p.kind for p in specs}, {"inlet": np.array([-1.0, 0, 0])}, ph, 4)
    U = (case / "0.orig/U").read_text()
    assert "value uniform (0.01 0 0)" in U and "noSlip" in U
    T = (case / "0.orig/T").read_text()
    assert "wall_heated\n    {\n        type fixedValue;\n        value uniform 1;" in T
    assert "DT [0 2 -1 0 0 0 0] 5e-07" in (case / "constant/transportProperties").read_text()
    assert "functions" not in (case / "system/controlDict").read_text()      # broken on v1912: see case.py
    allrun = (case / "Allrun").read_text()
    assert "simpleFoam -parallel" in allrun and "scalarTransportFoam" in allrun
    with pytest.raises(ValueError):
        Physics(solver="pimpleFoam", scalar={"name": "T"})
    assert split_procs(4, np.array([10.0, 1.0, 1.0])) == (4, 1, 1)
    assert split_procs(8, np.array([1.0, 1.0, 1.0])) in {(2, 2, 2)}
    assert "pimpleFoam > log.pimpleFoam" in allrun_script("pimpleFoam", 1)


def test_expressions_and_physics():
    assert evaluate("white_ratio(dean(200, D, Rc))", {"D": 10, "Rc": 40}) == pytest.approx(1.373, abs=2e-3)
    ph = resolve_physics({"nu": 1e-6, "velocity": "Re * nu / (D * 1e-3)", "solver": "simpleFoam",
                          "scalar": {"name": "T", "D": "nu / Pr", "walls": {"w": 1, "x": "zeroGradient"}}},
                         {"Re": 100.0, "D": 10.0, "Pr": 2.0})
    assert ph["velocity"] == pytest.approx(0.01) and ph["scalar"]["D"] == pytest.approx(5e-7)
    assert ph["scalar"]["walls"] == {"w": 1, "x": "zeroGradient"} and ph["solver"] == "simpleFoam"


def test_spec_to_workflow_and_doe(tmp_path):
    from pinneapple_tools.automation import known_actions
    spec = PipelineSpec(id="bend", title="Bend", template="pipe_bend", backend="fake",
                        inputs={"Re": {"default": 200.0, "min": 10, "max": 1000}},
                        physics={"nu": 1e-6, "velocity": "Re * nu / (D * 1e-3)"},
                        references={"x": {"measured": "dp_static_pa", "reference": "1", "tol": 0.1}})
    wf = spec.to_workflow(tmp_path)
    assert not wf.validate(known_actions())
    assert [s.action for s in wf.steps] == ["freecadcmd.export", "openfoam.prepare", "openfoam.mesh", "openfoam.solve",
                                            "openfoam.post", "verify.criteria"]
    assert {c.id for c in wf.criteria} == {"converged", "mass_balance", "ref_x"}
    assert PipelineSpec.load(spec.save(tmp_path / "p.json")).to_dict() == spec.to_dict()
    pts = doe({"Rc": (20, 80), "angle": (45, 180)}, 10, seed=1, fixed={"Re": 200.0})
    assert len(pts) == 10 and all(20 <= p["Rc"] <= 80 and p["Re"] == 200.0 for p in pts)
    assert len(doe({"a": (0, 1), "b": (0, 1)}, 3, method="grid")) == 9
    sw = Sweep(spec, pts[:2], tmp_path / "sweep")
    sw.save()
    assert Sweep.load(tmp_path / "sweep").pending() == ["c000", "c001"]
    from pinneapple_tools.automation import WorkflowError
    with pytest.raises(WorkflowError):
        Sweep(spec, [{"Rc": 1.0}], tmp_path / "bad")                      # below the parameter bound


def test_metric_criterion():
    from pinneapple_tools.automation.record import StepRecord
    from pinneapple_tools.automation.verify import evaluate_criterion
    rec = StepRecord("post", "post", "openfoam.post", state="passed", outputs={"e": 0.03})
    ok = evaluate_criterion("metric", {"step": "post", "output": "e", "max": 0.05}, None, {"post": rec})
    bad = evaluate_criterion("metric", {"step": "post", "output": "e", "max": 0.01}, None, {"post": rec})
    assert ok.passed and not bad.passed and ok.value == 0.03


def test_fake_run_blocks_or_runs(tmp_path):
    """The workflow runs the fake CAD step anywhere; without OpenFOAM the run stops, blocked, at the case step."""
    spec = PipelineSpec(id="hp", title="Heated pipe", template="heated_pipe", backend="fake",
                        inputs={"Re": {"default": 100.0, "min": 10, "max": 500}},
                        physics={"nu": 1e-6, "rho": 1000.0, "velocity": "Re * nu / (D * 1e-3)", "iterations": 50})
    if HAS_OPENFOAM:
        pytest.skip("OpenFOAM is installed: test_openfoam_heated_pipe runs the full chain")
    rec = run_pipeline(spec, tmp_path / "run")
    assert rec.step("cad").state == "passed" and rec.state == "blocked"
    assert (tmp_path / "run" / "cad" / "geometry.stl").is_file()
    assert json.loads((tmp_path / "run" / "study_result.json").read_text())["status"]


# ----------------------------------------------------------------------------------------------- real tools
@pytest.mark.skipif(not HAS_FREECAD, reason="freecadcmd not found")
def test_freecad_export_matches_the_fake(tmp_path):
    m = MODELS["pipe_bend"]
    fc = FreeCADBackend()
    fc.build("pipe_bend", m.defaults(), tmp_path / "bend.FCStd")
    g = fc.export(tmp_path / "bend.FCStd", {"Rc": 30.0, "angle": 120.0}, "Fluid", tmp_path / "out")
    assert g.info["valid"] and g.info["closed"] and g.open_edges() == 0 and (tmp_path / "out/geometry.step").is_file()
    assert g.volume() == pytest.approx(g.info["volume_mm3"], rel=0.005)
    a = assign_patches(g, PatchSpec.parse(m.patches), g.info["parameters"])
    out = g.faces[a["outlet"][0]].normal
    assert out == pytest.approx([math.cos(math.radians(120)), math.sin(math.radians(120)), 0], abs=1e-3)


@pytest.mark.skipif(not HAS_OPENFOAM, reason="OpenFOAM not found")
def test_openfoam_heated_pipe(tmp_path):
    """Coarse heated pipe end to end (about 20 s): converged, mass balance, Nu within 10 % of Hausen."""
    spec = PipelineSpec(id="hp", title="Heated pipe", template="heated_pipe", backend="auto", procs=2,
                        inputs={"Re": {"default": 100.0, "min": 10, "max": 500},
                                "Pr": {"default": 2.0, "min": 0.5, "max": 10}},
                        physics={"nu": 1e-6, "rho": 1000.0, "velocity": "Re * nu / (D * 1e-3)", "iterations": 2000,
                                 "residual_tol": 1e-6,
                                 "scalar": {"name": "T", "D": "nu / Pr", "inlet": 0.0, "walls": {"wall_heated": 1.0}}},
                        mesh={"cells_across": 10},
                        derived={"Nu_m": "nu_from_T(Re, Pr, D / L_heated, 1.0, scalar_in, scalar_out)"},
                        references={"Nu": {"measured": "Nu_m", "reference": "hausen_nu(Re, Pr, D / L_heated)",
                                           "tol": 0.10, "source": "Hausen (1943)"}})
    rec = run_pipeline(spec, tmp_path / "run")
    assert rec.state == "passed", rec.message
    m = json.loads((tmp_path / "run/post/measured.json").read_text())["measured"]
    assert m["mass_imbalance"] < 1e-6 and 0 < m["scalar_out"] < 1


# ----------------------------------------------------------------------------------------------- Flow Studio
def test_flow_nodes_and_bundled_flows():
    import pinneapple_flow as pf
    from pinneapple_flow.registry import REGISTRY
    from pinneapple_flow.schema import validate_flow
    assert {"cfd.study", "cfd.geometry", "cfd.run", "cfd.sweep", "cfd.surrogate"} <= set(REGISTRY.types())
    for name in ("freecad_openfoam_bend", "freecad_openfoam_sweep"):
        d = json.loads((pf_flows() / f"{name}.json").read_text())
        assert validate_flow(d) == []
    d = json.loads((pf_flows() / "freecad_openfoam_bend.json").read_text())
    d["nodes"] = [n for n in d["nodes"] if n["id"] != "run"]
    d["edges"] = [e for e in d["edges"] if e["target"] != "run"]
    d["nodes"][0]["params"]["backend"] = "fake"
    r = pf.run(d)
    assert r.output("geom", "metrics")["outputs"]["kinds"] == {"inlet": "inlet", "outlet": "outlet", "walls": "wall"}


def pf_flows():
    from pathlib import Path

    import pinneapple_flow
    return Path(pinneapple_flow.__file__).parent / "flows"


@pytest.mark.skipif(not HAS_OPENFOAM, reason="OpenFOAM not found")
def test_openfoam_pimple_courant(tmp_path):
    """Transient pimpleFoam on a coarse pipe: finishes, conserves mass, keeps the Courant number below 1."""
    spec = PipelineSpec(id="pim", title="Pimple", template="heated_pipe", backend="auto", procs=1,
                        inputs={"Re": {"default": 100.0, "min": 10, "max": 500}},
                        physics={"solver": "pimpleFoam", "nu": 1e-6, "rho": 1000.0, "velocity": "Re * nu / (D * 1e-3)",
                                 "end_time": 2.0, "dt": 0.05, "max_courant": 0.8, "write_interval": 1.0,
                                 "residual_tol": 1.0},
                        mesh={"cells_across": 8})
    rec = run_pipeline(spec, tmp_path / "run")
    assert rec.state == "passed", rec.message
    m = json.loads((tmp_path / "run/post/measured.json").read_text())["measured"]
    assert m["courant_max"] <= 1.0 and m["mass_imbalance"] < 1e-6
