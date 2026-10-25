"""The validation gallery (``examples/validation``) as flows: a model of one family on one physics, validated with
``pp.validate``, and a broken variant of it that one named check must catch
(:func:`pinneapple_analysis.validation.gallery.conclude`: report, trust card, SARIF and the failure record).

Converted: 01 (PINN, heat conduction, boundary loss removed), 05 (POD + DMD, cylinder wake, fitted on the growing
transient), 06 (POD + operator inference, advection, basis cut at 95 % of the energy) and 07 (zoo MLP, heat sink,
queries outside the card's range). The others build their models with helpers written in the scripts."""
from __future__ import annotations

import warnings

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import artifact


def _conclude(ctx, name, good, bad, *, expect, family, physics):
    """Record both reports and the gallery's conclusion (the expected check must catch the broken variant)."""
    from pinneapple_analysis.validation.gallery import conclude
    try:
        res = conclude(name, good, bad, expect=expect, family=family, physics=physics, out_dir=ctx.path(name))
    except AssertionError as exc:
        raise StepFailed(str(exc)) from None
    card = good.to_trust_report()
    ctx.metric("good.decision", good.decision, "pp.validate decision of the model as trained")
    ctx.metric("good.trust_card", card.decisao, "trust card decision of the same evidence")
    ctx.metric("bad.decision", bad.decision, "pp.validate decision of the broken variant")
    ctx.metric("expected_check", expect)
    ctx.metric("caught", ", ".join(res.caught), "checks failing on the broken variant and not on the good model")
    ctx.metric("caught_expected", int(any(c == expect or c.startswith(expect + ".") for c in res.caught)))
    for f, v in good.credibility().items():
        ctx.metric(f"good.credibility.{f}", v["level"])
    bad_st = {c.id: c.status for c in bad.checks}
    ids = [c.id for c in good.checks] + [c.id for c in bad.checks if c.id not in {g.id for g in good.checks}]
    good_st = {c.id: c.status for c in good.checks}
    ctx.table(f"{name}: checks of the good model and of the broken variant", ["check", "good", "broken"],
              [[i, good_st.get(i, "-"), bad_st.get(i, "-")] for i in ids])
    ctx.summary(res.summary())
    return {"metrics": ctx.report["metrics"], "report": artifact(res.files["report"], "json")}


_OUT = [Port("metrics", "metrics"), Port("report", "report", required=False)]


@node("gallery.pinn_heat", "Validation gallery 01: PINN on steady heat conduction (boundary loss removed)",
      category="Validation", outputs=_OUT,
      params=[Param("n_collocation", "int", 2000, minimum=16), Param("adam_steps", "int", 2000, minimum=1),
              Param("lbfgs_steps", "int", 1500, minimum=0)],
      calls="pinneapple_analysis.validation.pinn.poisson_2d, train_pinn; pp.validate; "
            "pinneapple_analysis.validation.gallery.conclude")
def pinn_heat(ctx, inputs, params):
    """-lap T = q on the unit square, T = 0 on the walls (manufactured source): the PINN as trained and the same PINN
    trained with boundary weight 0, both through ``pp.validate(net, {"pinn": problem})``; ``pinn.boundary`` must
    catch the second."""
    import pinneapple as pp
    from pinneapple_analysis.validation.pinn import poisson_2d, train_pinn
    problem = poisson_2d()
    size = dict(n_collocation=int(params["n_collocation"]), adam_steps=int(params["adam_steps"]),
                lbfgs_steps=int(params["lbfgs_steps"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        net, info = train_pinn(problem, **size)
        ctx.check_stop()
        no_bc, _ = train_pinn(problem, **size, weights={"boundary": 0.0})
    good = pp.validate(net, {"pinn": problem}, data={"train_points": info["train_points"]}, level="standard")
    bad = pp.validate(no_bc, {"pinn": problem}, level="standard")
    return _conclude(ctx, "01_pinn_heat_conduction", good, bad, expect="pinn.boundary", family="pinn",
                     physics="heat")


@node("gallery.pod_dmd_wake", "Validation gallery 05: POD + DMD of a cylinder wake (fitted on the growing transient)",
      category="Validation", outputs=_OUT,
      params=[Param("diameter", "int", 20, minimum=4, help="cylinder diameter in lattice cells (LBM)"),
              Param("lbm_steps", "int", 60000, minimum=1000, help="LBM steps (frames every 50 steps)"),
              Param("rank", "int", 8, minimum=1)],
      calls="pinneapple_lab.experiments.rom._wake (LBM, Re 60); "
            "pinneapple_neural.architectures.rom.steppers.pod_dmd_stepper; pp.validate; gallery.conclude")
def pod_dmd_wake(ctx, inputs, params):
    """POD + DMD fitted on the saturated wake (good) and on the growing transient (broken), rolled out from the
    same frame against the LBM reference; ``rollout.invariants`` must catch the amplifying ROM."""
    import pinneapple as pp
    from pinneapple_lab.experiments.rom import _wake
    from pinneapple_neural.architectures.rom.steppers import pod_dmd_stepper
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        _, uy, _, _meta = _wake(60.0, int(params["lbm_steps"]), 50, D=int(params["diameter"]))
    n = len(uy)
    sat, grow, test = slice(int(0.58 * n), int(0.75 * n)), slice(int(0.31 * n), int(0.48 * n)), int(0.75 * n)
    good_step, lam_good = pod_dmd_stepper(uy[sat], r=int(params["rank"]))
    bad_step, lam_bad = pod_dmd_stepper(uy[grow], r=int(params["rank"]))
    ref = uy[test:]
    problem = {"rollout": {"ndim": 2, "dt": 1.0, "periodic": False, "steps": len(ref) - 1},
               "name": "cylinder wake, Re 60 (LBM), cross-stream velocity"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(good_step, problem, data={"u0": ref[0], "reference": ref}, level="standard")
        bad = pp.validate(bad_step, problem, data={"u0": ref[0], "reference": ref}, level="standard")
    ctx.metric("dmd_max_abs_eig_saturated", lam_good, "largest |DMD eigenvalue|, ROM fitted on the saturated wake")
    ctx.metric("dmd_max_abs_eig_transient", lam_bad, "largest |DMD eigenvalue|, ROM fitted on the growing transient")
    return _conclude(ctx, "05_pod_dmd_cylinder_wake", good, bad, expect="rollout.invariants", family="rom",
                     physics="incompressible-flow")


@node("gallery.opinf_advection", "Validation gallery 06: POD + operator inference on advection (basis cut at 95 %)",
      category="Validation", outputs=_OUT,
      params=[Param("n_train", "int", 12, minimum=1, help="training trajectories"),
              Param("energy_bad", "float", 0.95, minimum=0.0, maximum=1.0,
                    help="energy fraction kept by the broken ROM's basis")],
      calls="pinneapple_physics.advection_diffusion_1d.AdvectionDiffusion1D; "
            "pinneapple_neural.architectures.rom.steppers.pod_opinf_stepper; pp.validate; gallery.conclude")
def opinf_advection(ctx, inputs, params):
    """Periodic advection-diffusion of a tracer: ROMs with the full basis (good) and cut at ``energy_bad`` (broken),
    rolled out against the exact solution with mass conserved; ``rollout.spectrum`` must catch the smearing."""
    import numpy as np

    import pinneapple as pp
    from pinneapple_neural.architectures.rom.steppers import pod_opinf_stepper
    from pinneapple_physics.advection_diffusion_1d import AdvectionDiffusion1D
    P = AdvectionDiffusion1D(nx=64, nt=201, t_end=4 * np.pi)
    rng = np.random.default_rng(0)
    train = np.stack([P.exact(P.random_case(rng, nu=0.005, c=1.0, n_modes=6)) for _ in range(int(params["n_train"]))])
    ref = P.exact(P.random_case(np.random.default_rng(99), nu=0.005, c=1.0, n_modes=6))
    problem = {"rollout": {"ndim": 1, "dt": float(P.t[1]), "conserved": ["mass"], "steps": len(ref) - 1},
               "name": "periodic advection-diffusion (tracer in a uniform wind)"}
    good_step, r_good = pod_opinf_stepper(train, energy=1.0 - 1e-10)
    bad_step, r_bad = pod_opinf_stepper(train, energy=float(params["energy_bad"]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(good_step, problem, data={"u0": ref[0], "reference": ref}, level="standard")
        bad = pp.validate(bad_step, problem, data={"u0": ref[0], "reference": ref}, level="standard")
    ctx.metric("pod_rank_good", r_good, "POD rank keeping all the energy")
    ctx.metric("pod_rank_bad", r_bad, f"POD rank keeping {params['energy_bad']:.0%} of the energy")
    return _conclude(ctx, "06_opinf_weather_advection", good, bad, expect="rollout.spectrum", family="rom",
                     physics="advection")


@node("gallery.zoo_range", "Validation gallery 07: zoo heat-sink MLP queried outside its card's range",
      category="Validation", outputs=_OUT,
      params=[Param("model", "str", "heatsink_forced_mlp", help="zoo model"),
              Param("dataset", "str", "heatsink_forced_convection", help="dataset whose test split gives the queries"),
              Param("n_query", "int", 64, minimum=1, help="test designs validated"),
              Param("air_velocity_m_s", "float", 0.2, help="broken query: air velocity"),
              Param("power_w", "float", 600.0, help="broken query: power")],
      calls="pp.zoo.load, pp.datasets.load, pp.validate, pinneapple_analysis.validation.gallery.conclude")
def zoo_range(ctx, inputs, params):
    """The zoo model on test designs of its dataset (good) and on two of them moved to a velocity and power outside
    the card's validity range (broken); ``applicability.range`` must catch the second."""
    import pinneapple as pp
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = pp.zoo.load(params["model"])
        test = pp.datasets.load(params["dataset"], split="test").to_pandas()
        query = test[[i["name"] for i in model.card.inputs]].head(int(params["n_query"]))
        good = pp.validate(model, query=query.to_dict("list"), level="standard")
        out = query.head(2).copy()
        out["air_velocity_m_s"], out["power_w"] = float(params["air_velocity_m_s"]), float(params["power_w"])
        bad = pp.validate(model, query=out.to_dict("list"), level="standard")
    return _conclude(ctx, "07_mlp_zoo_heatsink_convection", good, bad, expect="applicability.range",
                     family="classical-ml", physics="heat")


def _register_capabilities():
    from ..planner.extensions import Capability, register_capability
    common = dict(domain="validation", example="validation", min_score=4)
    register_capability(Capability(
        id="gallery.pinn_heat", title="Validate a PINN of heat conduction and catch a missing boundary loss",
        keywords={"validation gallery": 3, "pinn": 1, "boundary loss": 3, "pinn.boundary": 4,
                  "heat conduction": 2, "validate": 1},
        nodes=("gallery.pinn_heat",), references=("raissi2019pinn", "oberkampf2010vv"),
        template="validation/01_pinn_heat_conduction", quick=True,
        description="examples/validation/01_pinn_heat_conduction.py", **common))
    register_capability(Capability(
        id="gallery.pod_dmd_wake", title="Validate a POD + DMD rollout of a cylinder wake",
        keywords={"validation gallery": 3, "dmd": 2, "dynamic mode decomposition": 3, "cylinder wake": 2,
                  "rollout": 1, "validate": 1},
        nodes=("gallery.pod_dmd_wake", "examples.compare"), references=("berkooz1993pod", "schmid2010dmd"),
        template="validation/05_pod_dmd_cylinder_wake", quick=True,
        description="examples/validation/05_pod_dmd_cylinder_wake.py", **common))
    register_capability(Capability(
        id="gallery.opinf_advection", title="Validate a POD + operator-inference rollout of advection",
        keywords={"validation gallery": 3, "operator inference": 3, "opinf": 3, "advection": 1, "rollout": 1,
                  "validate": 1},
        nodes=("gallery.opinf_advection",), references=("berkooz1993pod", "peherstorfer2016opinf"),
        template="validation/06_opinf_weather_advection", quick=True,
        description="examples/validation/06_opinf_weather_advection.py", **common))
    register_capability(Capability(
        id="gallery.zoo_range", title="Validate a zoo model and catch queries outside its validity range",
        keywords={"validation gallery": 3, "zoo model": 2, "heatsink": 2, "heat sink": 2, "validity range": 3,
                  "out of range": 2, "validate": 1},
        nodes=("gallery.zoo_range",), references=("oberkampf2010vv",),
        template="validation/07_mlp_zoo_heatsink_convection", quick=True,
        description="examples/validation/07_mlp_zoo_heatsink_convection.py", **common))


_register_capabilities()
