"""PhysicalProblem: one problem description, with lossless adapters to the three existing classes."""
import json

import pytest

import pinneapple as pp
from pinneapple_physics.physical_problem import Parameter, PhysicalProblem, Quantity

# Known, documented exception (see tag_geometry.py): radiation and a material-interface "k" target are not
# expressible as single-field Neumann conditions yet.
KNOWN_INVALID = {"industrial_furnace_thermal"}


@pytest.mark.parametrize("name", pp.list_presets())
def test_every_preset_round_trips_losslessly(name):
    spec = pp.get_preset(name)
    assert PhysicalProblem.from_pde_spec(spec).to_pde_spec() == spec


@pytest.mark.parametrize("name", sorted(set(pp.list_presets()) - KNOWN_INVALID))
def test_every_preset_validates_clean(name):
    assert PhysicalProblem.from_preset(name).validate() == []


def test_the_known_invalid_preset_is_still_reported():
    issues = PhysicalProblem.from_preset("industrial_furnace_thermal").validate()
    assert any("hot_inner_surface" in i for i in issues) and any("insulation_interface" in i for i in issues)


@pytest.mark.parametrize("name", ["pcb_thermal", "cpu_heatsink_thermal", "car_suspension_fatigue",
                                  "datacenter_airflow_2d", "datacenter_cfd_3d", "datacenter_server_thermal"])
def test_presets_whose_bcs_were_fixed_now_compile_and_evaluate(name):
    """These presets declared heat-flux, convection or traction targets without thermal_bc/traction_map,
    so building the loss raised KeyError. They must now evaluate on a batch that hits every condition."""
    torch = pytest.importorskip("torch")
    from pinneapple_physics.pinn_solver.compiler.compile import compile_problem

    spec = pp.get_preset(name)
    nc, nf, n = len(spec.coords), len(spec.fields), 8
    masks = {c.selector["tag"]: [True] * n for c in spec.conditions
             if c.selector_type == "tag" and isinstance(c.selector, dict)}
    normals = torch.zeros(n, nc)
    normals[:, 0] = 1.0
    batch = {"x_col": torch.rand(n, nc, requires_grad=True), "ctx": {"tag_masks": masks}, "n_bc": normals,
             "x_bc": torch.rand(n, nc, requires_grad=True), "y_bc": torch.zeros(n, nf),
             "x_ic": torch.zeros(0, nc), "y_ic": torch.zeros(0, nf), "x_data": torch.zeros(0, nc), "y_data": torch.zeros(0, nf)}
    model = torch.nn.Sequential(torch.nn.Linear(nc, 8), torch.nn.Tanh(), torch.nn.Linear(8, nf))
    out = compile_problem(spec)(model, None, batch)
    total = out["total"] if isinstance(out, dict) else out
    assert torch.isfinite(total)


def test_parameters_and_roles():
    prob = PhysicalProblem.from_preset("burgers_1d", nu=0.01)
    assert prob.parameter_values() == {"nu": 0.01} and prob.parameters["nu"].role == "fixed"
    changed = prob.with_parameters(nu=0.02)
    assert changed.to_pde_spec().pde.params["nu"] == 0.02 and prob.parameters["nu"].value == 0.01
    with pytest.raises(KeyError):
        prob.with_parameters(viscosity=0.1)
    disc = prob.set_role("nu", "unknown", bounds=(1e-3, 1e-1))
    assert list(disc.parameters_by_role("unknown")) == ["nu"] and disc.validate() == []
    design = prob.set_role("nu", "design")
    assert any("needs bounds" in i for i in design.validate())


def test_validate_catches_inconsistencies():
    prob = PhysicalProblem(
        name="bad", coords=("x", "x"), fields=("u",), domain_bounds={"x": (1.0, 0.0), "y": (0, 1)},
        parameters={"k": Parameter("k", value=5.0, unit="W/(m*K)", bounds=(0.0, 1.0)),
                    "q": Parameter("q", value=1.0, unit="furlongs^3"),
                    "r": Parameter("r", role="sideways", value=1.0)},
        units={"u": "m/s", "zzz": "m"}, quantities=(Quantity("peak", field="T"),),
    )
    text = "\n".join(prob.validate())
    for expected in ("duplicate coordinate", "empty or reversed", "unknown coordinate 'y'", "outside bounds",
                     "'furlongs^3' is not recognised", "role 'sideways'", "neither a coordinate nor a field",
                     "unknown field 'T'"):
        assert expected in text, expected


def test_dict_round_trip_is_json_and_keeps_everything_but_callables():
    prob = PhysicalProblem.from_preset("heat_2d_transient") if "heat_2d_transient" in pp.list_presets() \
        else PhysicalProblem.from_preset("burgers_1d")
    prob = prob.set_role(next(iter(prob.parameters)), "uncertain",
                         distribution={"type": "normal", "mean": 1.0, "std": 0.1})
    d = json.loads(json.dumps(prob.to_dict()))
    back = PhysicalProblem.from_dict(d)
    assert back.parameters == prob.parameters and back.pde == prob.pde and back.domain_bounds == prob.domain_bounds
    assert [c.name for c in back.conditions] == [c.name for c in prob.conditions]
    assert back.fingerprint() == prob.fingerprint()
    detached = [c for c in back.conditions if callable(c.value_fn)]
    if detached:  # a callable that was not serialized must fail loudly, never act as "no value"
        with pytest.raises(RuntimeError, match="re-attach"):
            detached[0].value_fn(None, {})


def test_fingerprint_tracks_physics_not_metadata():
    a = PhysicalProblem.from_preset("burgers_1d", nu=0.01)
    b = PhysicalProblem.from_preset("burgers_1d", nu=0.01)
    assert a.fingerprint() == b.fingerprint()
    assert a.with_parameters(nu=0.02).fingerprint() != a.fingerprint()
    a.metadata["note"] = "run on Tuesday"
    assert a.fingerprint() == b.fingerprint()


def test_physics_case_round_trip():
    from pinneapple_data.physics_case import PhysicsCase

    spec = pp.get_preset("burgers_1d")
    case = PhysicsCase(name="case-1", physics=spec, solver="fdm", solver_config={"nx": 64},
                       reference_benchmark="burgers", metadata={"owner": "lab"})
    prob = PhysicalProblem.from_physics_case(case)
    assert prob.solver == "fdm" and prob.to_pde_spec() == spec
    back = prob.to_physics_case()
    assert back.name == "case-1" and back.physics == spec and back.solver_config == {"nx": 64}
    assert back.reference_benchmark == "burgers" and back.metadata == {"owner": "lab"}


def test_problem_design_round_trip_keeps_intent_and_parameter_roles():
    from pinneapple_problemdesign.schema import PhysicsSpec, ProblemSpec as DesignSpec

    design = DesignSpec(title="Heat sink sizing", goal="keep the CPU below 85 C", task_type="inverse_problem",
                        inputs=["x", "y"], outputs=["T"],
                        physics=PhysicsSpec(governing_equations=["steady heat conduction"],
                                            parameters_known=["k"], parameters_unknown=["h"],
                                            units={"k": "W/(m*K)", "h": "W/(m^2*K)", "T": "K"}))
    prob = PhysicalProblem.from_problem_design(design)
    assert prob.task == "inverse" and prob.fields == ("T",) and prob.units == {"T": "K"}
    assert prob.parameters["h"].role == "unknown" and prob.parameters["k"].unit == "W/(m*K)"
    again = prob.to_problem_design()
    assert again.title == design.title and again.goal == design.goal and again.task_type == design.task_type
    assert again.physics.governing_equations == ["steady heat conduction"]
    assert again.physics.parameters_unknown == ["h"] and again.physics.parameters_known == ["k"]
    assert again.physics.units == design.physics.units


def test_to_pde_spec_without_a_pde_explains_what_is_missing():
    with pytest.raises(ValueError, match="has no PDE"):
        PhysicalProblem(name="draft").to_pde_spec()


def test_top_level_exports():
    assert pp.PhysicalProblem is PhysicalProblem and pp.Parameter is Parameter and pp.Quantity is Quantity
