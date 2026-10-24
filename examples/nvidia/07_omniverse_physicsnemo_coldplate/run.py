"""Cold plate: PINNeAPPle generates and checks, PhysicsNeMo FNO learns, Omniverse (OpenUSD) shows.

A 100 mm x 100 mm cold plate with copper inserts (k = 400 W/m/K) in an aluminium base (k = 100 W/m/K), a uniform
volumetric heat load and its edges held at the coolant temperature: steady conduction -div(k grad T) = q, T = T_c on
the edges. PINNeAPPle's finite-volume solver makes the training data, ``physicsnemo.models.fno.FNO`` learns the map
layout -> temperature, PINNeAPPle's Darcy-type checks (certified residual bound, boundary condition, energy balance,
maximum principle, reference, training envelope) decide whether the surrogate's fields can be used, and the result is
written as an OpenUSD stage (temperature as a primvar, trust decision on the prim) for Omniverse.

    python examples/nvidia/07_omniverse_physicsnemo_coldplate/run.py            # about 3 min on 4 CPU cores
    python examples/nvidia/07_omniverse_physicsnemo_coldplate/run.py --smoke    # seconds (CI)
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_lab import experiment
from pinneapple_neural.physicsnemo_bridge.field_operators import (
    physicsnemo_fno,
    train_grid_operator,
)
from pinneapple_physics import darcy_checks as dc
from pinneapple_simulation.numerical_solvers import operator_benchmarks as ob
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, curves_figure, field_panels
from pinneapple_tools.nvidia.omniverse import (
    FieldStage,
    grid_surface,
    kit_render_script,
    render_stage_png,
    validate_stage,
)
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
L = 0.10                      # plate side [m]
K0 = 100.0 / 3.0              # W/m/K per unit of the solver's coefficient: 3 -> aluminium (100), 12 -> copper (400)
Q = 2.0e6                     # volumetric heat load [W/m^3] (100 W in a 5 mm plate)
T_COOL = 300.0                # coolant / edge temperature [K]
DT = Q * L ** 2 / K0          # kelvin per unit of the dimensionless solution


def layouts(n_samples, n, seed):
    return ob.threshold_coefficient(ob.periodic_grf(n_samples, (n, n), alpha=2.0, tau=3.0, seed=seed))


def main(smoke=False, base=None):
    """``base``: folder for experiment.json and results/ (default: this folder; tests pass a temporary one)."""
    base = base or HERE
    OUT = os.path.join(base, "results")
    n, n_train, n_test = (16, 24, 8) if smoke else (32, 400, 50)
    budget = ComputeBudget(steps=6) if smoke else ComputeBudget(steps=2700)
    info = RunInfo("07_omniverse_physicsnemo_coldplate")
    os.makedirs(OUT, exist_ok=True)
    architecture_diagram(
        "07 Cold plate: PINNeAPPle + Omniverse + PhysicsNeMo",
        [Block("gen", "Layouts (Gaussian random\nfield, Cu / Al inserts)", "pinneapple", 0, 0),
         Block("fv", "Finite-volume solver\n-div(k grad T) = q", "pinneapple", 1, 0),
         Block("fno", "physicsnemo.models.fno.FNO\ntrained under a ComputeBudget", "physicsnemo", 2, 0),
         Block("chk", "Physics checks: residual bound,\nBC, energy balance, envelope", "pinneapple", 2, 1),
         Block("trust", "Trust decision\nAPPROVED / REVIEW / REJECT", "pinneapple", 3, 1),
         Block("usd", "OpenUSD stage (usd-core)\nT primvar + displayColor + trust", "omniverse", 3, 0),
         Block("val", "UsdValidation + read-back\npreview (matplotlib)", "omniverse", 4, 0),
         Block("kit", "Omniverse Kit / USD Composer\nRTX viewport", "omniverse", 4, 1, ran=False)],
        [("gen", "fv", "k(x, y)"), ("fv", "fno", "(k, T) pairs"), ("fno", "chk", "T predicted"),
         ("chk", "trust", "checks"), ("fno", "usd", "T field"), ("trust", "usd", "decision"),
         ("usd", "val", ".usda"), ("usd", "kit", ".usda")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Data, checks and decision by PINNeAPPle; surrogate by PhysicsNeMo; scene in OpenUSD for Omniverse")

    with experiment("examples/nvidia/07_omniverse_physicsnemo_coldplate", title="Cold plate: PhysicsNeMo FNO, "
                    "PINNeAPPle checks, OpenUSD scene", domain="heat transfer", methods=["FNO (PhysicsNeMo)", "FVM"],
                    modules=["pinneapple_neural.physicsnemo_bridge", "pinneapple_physics.darcy_checks",
                             "pinneapple_tools.nvidia.omniverse"], out_dir=base, register=False,
                    params={"n": n, "n_train": n_train, "n_test": n_test, "budget": budget.__dict__,
                            "k_W_mK": [100, 400], "q_W_m3": Q, "L_m": L}) as ex:
        with info.step("data (PINNeAPPle finite volumes)"):
            a_tr, a_te = layouts(n_train, n, 0), layouts(n_test, n, 1)
            u_tr = np.stack([ob.darcy_2d(a) for a in a_tr])
            u_te = np.stack([ob.darcy_2d(a) for a in a_te])
        pred = physicsnemo_fno(n, latent_channels=32 if not smoke else 8, num_fno_layers=4 if not smoke else 2,
                               num_fno_modes=12 if not smoke else 4, decoder_layer_size=64 if not smoke else 16)
        curve = []

        def evaluate():
            p = pred.predict(a_te)
            e = np.linalg.norm((p - u_te).reshape(len(p), -1), axis=1) / np.linalg.norm(u_te.reshape(len(p), -1), axis=1)
            curve.append(float(e.mean()))
            return {"test_rel_l2": float(e.mean())}

        with info.step("train (PhysicsNeMo FNO)"):
            res = train_grid_operator(pred, a_tr, u_tr, budget, batch_size=8 if not smoke else 4, lr=2e-3,
                                      evaluate=evaluate, eval_at=np.linspace(0.1, 0.9, 9))
        info.mark("PhysicsNeMo FNO training", ran=True, how=f"CPU, {res.run.steps} steps, {res.epochs:.1f} epochs")
        with info.step("checks (PINNeAPPle)"):
            u_hat = pred.predict(a_te)
            env = dc.CoefficientEnvelope.fit(a_tr)
            report, checks = dc.darcy_trust_report(a_te, u_hat, 1.0, model="PhysicsNeMo FNO", reference=u_te,
                                                   envelope=env)
            ref_report, ref_checks = dc.darcy_trust_report(a_te, u_te, 1.0, model="finite volumes")
        decision = report.decisao
        rel = float(np.mean(np.linalg.norm((u_hat - u_te).reshape(n_test, -1), axis=1)
                            / np.linalg.norm(u_te.reshape(n_test, -1), axis=1)))
        tmax_err = float(np.max(np.abs(u_hat.max((1, 2)) - u_te.max((1, 2)))) * DT)
        ex.log_metric("test_rel_l2", rel, "", f"mean relative L2 of T - T_c over {n_test} held-out layouts")
        ex.log_metric("max_hotspot_error_K", tmax_err, "K", "max over test layouts of |max T_FNO - max T_FV|")
        ex.log_metric("train_wall_s", res.run.wall_s, "s", "wall clock of FNO training on this CPU")
        for c in checks:
            if c.passed is not None:
                ex.check(c.name, bool(c.passed), value=c.value, max=c.limit, detail=c.detail, kind=c.kind)
        # pick the test layout with the median error for the pictures and the USD scene
        e_s = np.linalg.norm((u_hat - u_te).reshape(n_test, -1), axis=1) / np.linalg.norm(u_te.reshape(n_test, -1), axis=1)
        i = int(np.argsort(e_s)[len(e_s) // 2])
        est = dc.estimated_error(u_hat[i], a_te[i])
        field_panels([a_te[i] * K0, T_COOL + DT * u_te[i], T_COOL + DT * u_hat[i], DT * np.abs(u_hat[i] - u_te[i]),
                      DT * np.abs(est)],
                     ["conductivity k", "T, finite volumes (reference)", "T, PhysicsNeMo FNO", "|T_FNO - T_FV|",
                      "error estimated from the residual\n(no reference needed)"],
                     os.path.join(OUT, "fields.png"), cmaps=["cividis", "inferno", "inferno", "magma", "magma"],
                     units=["W/m/K", "K", "K", "K", "K"], shared=[(1, 2), (3, 4)], extent=(0, 100, 0, 100),
                     suptitle=f"Held-out layout {i} (median error): FNO relative L2 = {e_s[i]:.3%}")
        hist = [h for h in res.history if h.get("eval")]
        curves_figure([{"title": "FNO training (PhysicsNeMo) on CPU", "xlabel": "step", "ylabel": "relative L2",
                        "logy": True, "series": [("held-out relative L2", [h["step"] for h in hist],
                                                  [h["eval"]["test_rel_l2"] for h in hist], "o-")],
                        "hlines": [(dc.DEFAULT_TOLERANCES["reference"], "reference limit (5 %)", "r")]}],
                      os.path.join(OUT, "training.png"))
        checks_figure(checks, os.path.join(OUT, "checks.png"), title="PhysicsNeMo FNO, PINNeAPPle checks",
                      decision=decision)
        # OpenUSD stage: the plate (cell centres as vertices), T raised slightly in z so the hot spots read in 3-D
        with info.step("OpenUSD stage"):
            c = (np.arange(n) + 0.5) / n * L
            X, Y = np.meshgrid(c, c, indexing="ij")
            T = T_COOL + DT * u_hat[i]
            V, F = grid_surface(X, Y, 0.002 + 0.02 * (T - T_COOL) / max(T.max() - T_COOL, 1e-9))
            usd = os.path.join(OUT, "coldplate.usda")
            st = FieldStage(usd, "Cold plate, PhysicsNeMo FNO temperature")
            path = st.add_mesh("ColdPlate", V, F, fields={"temperature": T.ravel(), "conductivity": (a_te[i] * K0).ravel(),
                                                          "temperature_reference": (T_COOL + DT * u_te[i]).ravel(),
                                                          "estimated_error": (DT * np.abs(est)).ravel()},
                               units={"temperature": "K", "conductivity": "W/m/K", "temperature_reference": "K",
                                      "estimated_error": "K"}, color_by="temperature")
            st.set_trust(path, decision, [c.to_dict() for c in checks], model="physicsnemo.models.fno.FNO",
                         solver="pinneapple finite volumes")
            st.save()
            v = validate_stage(usd, required_primvars={path: ["temperature", "conductivity"]})
            render_stage_png(usd, os.path.join(OUT, "usd_preview.png"), colorbar=("T [K]", float(T.min()),
                             float(T.max()), "inferno"), title="coldplate.usda read back with OpenUSD "
                             "(height = temperature rise)")
            with open(os.path.join(OUT, "kit_render.py"), "w") as f:
                f.write(kit_render_script(usd, os.path.join(OUT, "coldplate_rtx.png")))
        info.mark("OpenUSD stage (usd-core)", ran=True, how="authored and validated with pxr")
        info.mark("Omniverse Kit RTX render", ran=False, why="needs Omniverse Kit with an RTX GPU",
                  how="results/kit_render.py inside USD Composer")
        ex.check("usd_stage_valid", v["ok"], detail="; ".join(v["problems"]) or "UsdValidation clean, primvars sized")
        summary = {"decision": decision, "test_rel_l2": rel, "max_hotspot_error_K": tmax_err,
                   "reference_solver_decision": ref_report.decisao, "fno_parameters": pred.n_parameters(),
                   "training": res.to_dict() | {"history": None}, "usd": {k: v[k] for k in ("ok", "problems",
                   "meshes", "prims")} | {"validators": len(v["compliance"].get("validators", []))},
                   "checks": [c.to_dict() for c in checks]}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "test_rel_l2", "max_hotspot_error_K",
                                              "reference_solver_decision")}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
