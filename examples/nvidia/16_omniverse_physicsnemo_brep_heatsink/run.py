"""The verified heat sink in Omniverse: exact geometry (NVIDIA USD BRep), surrogate search (PhysicsNeMo), physics and
verification (PINNeAPPle), and an OpenUSD scene whose colours are the temperature field the decision rests on.

A shorter version of the design loop of example 10 (60 designs, same duty: 60 W, 40 x 40 mm source, 3 m/s ducted
air, T_source <= 75 C, dp <= 30 Pa) proposes the lightest heat sink; PINNeAPPle verifies it with the exact B-rep
and the closed-form model. The accepted design is tessellated by USD BRep and every vertex gets the temperature of
the closed-form model: the base at the fin-root temperature, the fins along the 1-D fin solution
``theta(z) = theta_root cosh(m (Lc - z)) / cosh(m Lc)``. PINNeAPPle checks that the field in the scene is the one the
rating used (its fin-average equals the fin efficiency, its root and tip match the model) before writing the stage:
native B-rep referenced, mesh with ``temperature``, trust decision, and the RTX capture script.

    PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/16_omniverse_physicsnemo_brep_heatsink/run.py
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_lab import experiment
from pinneapple_physics.closed_form import plate_fin_heatsink as pfh
from pinneapple_tools.nvidia.brep import evaluate_part, geometry_checks
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.heatsink import Duty, design_table, sample_designs, verify
from pinneapple_tools.nvidia.omniverse import (
    FieldStage,
    kit_render_script,
    render_stage_png,
    validate_stage,
)
from pinneapple_tools.nvidia.parts import heat_sink
from pinneapple_tools.nvidia.physicsnemo import MLPSurrogate, relative_errors
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
DUTY = Duty(root_fillet=0.001)
T_MAX, DP_MAX = 75.0, 30.0


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("16_omniverse_physicsnemo_brep_heatsink")
    architecture_diagram(
        "16 Verified heat sink in Omniverse: PINNeAPPle + USD BRep + PhysicsNeMo + Omniverse",
        [Block("doe", "60 designs (LHS)", "pinneapple", 0, 0),
         Block("brep", "USD BRep exact solids\n(mass with root fillets)", "brep", 1, 0),
         Block("cf", "Closed-form rating\n(T_source, dp)", "pinneapple", 1, 1),
         Block("nn", "physicsnemo FullyConnected\nsurrogate + search", "physicsnemo", 2, 0),
         Block("ver", "Verify proposals (B-rep +\nclosed form), accept", "pinneapple", 2, 1),
         Block("field", "Temperature on the B-rep mesh:\nbase + 1-D fin solution", "pinneapple", 3, 1),
         Block("stage", "OpenUSD: B-rep reference +\nmesh with T + trust", "omniverse", 3, 0),
         Block("kit", "Omniverse Kit RTX", "omniverse", 4, 0, ran=False)],
        [("doe", "brep", "recipes"), ("doe", "cf", "designs"), ("brep", "nn", "mass"), ("cf", "nn", "T, dp"),
         ("nn", "ver", "proposals"), ("ver", "field", "accepted design"), ("brep", "stage", "B-rep + mesh"),
         ("field", "stage", "temperature"), ("stage", "kit", ".usda")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Every colour in the scene comes from the model that accepted the design")
    with experiment("examples/nvidia/16_omniverse_physicsnemo_brep_heatsink", title="Verified heat sink in OpenUSD: "
                    "USD BRep, PhysicsNeMo, PINNeAPPle", domain="heat transfer",
                    methods=["B-rep (SMLib)", "FullyConnected (PhysicsNeMo)", "closed-form heat sink", "OpenUSD"],
                    out_dir=base, register=False, modules=["pinneapple_tools.nvidia.heatsink",
                                                           "pinneapple_tools.nvidia.omniverse"],
                    params={"duty": DUTY.__dict__, "T_max_C": T_MAX, "dp_max_Pa": DP_MAX}) as ex:
        with info.step("designs (USD BRep + closed form)"):
            rows = design_table(sample_designs(12 if smoke else 60, seed=2), DUTY)
            test = design_table(sample_designs(4 if smoke else 15, seed=3), DUTY)
        f = lambda rs: (np.array([[r["n_fins"], r["fin_height"], r["fin_t"]] for r in rs]),       # noqa: E731
                        np.array([[r["t_source"] - DUTY.t_ambient_c, r["dp"], r["mass"]] for r in rs]))
        X, Y = f(rows)
        Xt, Yt = f(test)
        net = MLPSurrogate(3, 3, layer_size=64, num_layers=3)
        with info.step("train (PhysicsNeMo)"):
            net.fit(X, Y, ComputeBudget(steps=200 if smoke else 15000), batch_size=32, lr=3e-3)
        err = relative_errors(net.predict(Xt), Yt)
        nn_, hh, tt = np.meshgrid(np.arange(6, 21), np.linspace(0.015, 0.045, 61), np.linspace(0.001, 0.003, 41),
                                  indexing="ij")
        G = np.c_[nn_.ravel(), hh.ravel(), tt.ravel()]
        G = G[(0.1 - G[:, 0] * G[:, 2]) / (G[:, 0] - 1) >= 0.002]
        P = net.predict(G)
        ok = (P[:, 0] + DUTY.t_ambient_c <= T_MAX) & (P[:, 1] <= DP_MAX)
        attempts = []
        with info.step("verify proposals"):
            for i in np.argsort(np.where(ok, P[:, 2], np.inf))[:min(8, int(ok.sum()))]:
                v = verify(int(G[i, 0]), float(G[i, 1]), float(G[i, 2]), DUTY)
                attempts.append({"design": G[i].tolist(), "predicted_t": float(P[i, 0] + DUTY.t_ambient_c),
                                 "verified_t": v["t_source"], "verified_dp": v["dp"], "mass": v["mass"],
                                 "accepted": v["t_source"] <= T_MAX and v["dp"] <= DP_MAX})
                if attempts[-1]["accepted"]:
                    break
        n, H, t = attempts[-1]["design"]
        n = int(n)
        g = pfh.HeatSinkGeometry(DUTY.width, DUTY.length, DUTY.base_t, n, t, H, DUTY.material)
        th = pfh.evaluate(g, pfh.OperatingPoint(DUTY.power_w, DUTY.t_ambient_c, DUTY.air_velocity_m_s, DUTY.source,
                                                DUTY.source))
        native = os.path.join(OUT, "heatsink_brep.usda")
        with info.step("USD BRep: accepted design, tessellation, native USD"):
            rep = evaluate_part(heat_sink(n_fins=n, fin_height=H, fin_t=t, base_t=DUTY.base_t, root_fillet=DUTY.root_fillet),
                                chord_tolerances=[1e-4, 2e-5], max_edge_length=0.002, usd_brep_path=native,
                                density=pfh.MATERIALS[DUTY.material]["rho"])
        V, F = rep.mesh()
        m, lc = th["fin_m_per_m"], th["fin_lc_m"]
        theta_root = th["t_fin_root_c"] - DUTY.t_ambient_c
        z = np.clip(V[:, 2] - DUTY.base_t, 0.0, None)
        T = DUTY.t_ambient_c + theta_root * np.cosh(m * (lc - z)) / np.cosh(m * lc)
        zz = np.linspace(0, lc, 2001)
        prof = np.cosh(m * (lc - zz)) / np.cosh(m * lc)
        eta_field = float(np.trapezoid(prof, zz) / lc)
        tip = float(DUTY.t_ambient_c + theta_root / np.cosh(m * lc))
        on_tip = np.isclose(V[:, 2], DUTY.base_t + H, atol=1e-7)
        g_checks = geometry_checks(rep)             # a display mesh: no solver runs on it, so no element-size rule
        checks = [*g_checks,
                  {"name": "accepted design verified (T_source <= 75 C)", "value": attempts[-1]["verified_t"],
                   "limit": T_MAX, "passed": attempts[-1]["accepted"],
                   "detail": f"{len(attempts)} proposal(s) verified"},
                  {"name": "surrogate held-out error (worst output)", "value": float(err.max()), "limit": 0.03,
                   "passed": err.max() <= 0.03},
                  {"name": "field fin-average = fin efficiency of the rating", "value": abs(eta_field - th["fin_efficiency"]),
                   "limit": 1e-3, "passed": abs(eta_field - th["fin_efficiency"]) <= 1e-3,
                   "detail": f"{eta_field:.4f} vs {th['fin_efficiency']:.4f}"},
                  {"name": "field at the fin root = model root temperature", "value": abs(T[V[:, 2] <= DUTY.base_t + 1e-9].max()
                                                                                          - th["t_fin_root_c"]),
                   "limit": 1e-9, "passed": abs(T[V[:, 2] <= DUTY.base_t + 1e-9].max() - th["t_fin_root_c"]) <= 1e-9},
                  {"name": "field at the fin tips = cosh solution (K)", "value": float(np.abs(T[on_tip] - DUTY.t_ambient_c
                   - theta_root * np.cosh(m * (lc - H)) / np.cosh(m * lc)).max()) if on_tip.any() else float("nan"),
                   "limit": 1e-9, "passed": bool(on_tip.any())}]
        with info.step("OpenUSD stage"):
            usd = os.path.join(OUT, "heatsink.usda")
            st = FieldStage(usd, "Verified heat sink: exact geometry, temperature field, decision")
            st.stage.DefinePrim("/World/BRep").GetReferences().AddReference(os.path.basename(native), "/World")
            mp = st.add_mesh("HeatSink", V, F, fields={"temperature": T}, units={"temperature": "C"},
                             color_by="temperature", cmap="inferno", vrange=(float(T.min()), float(T.max())))
            decision = "APPROVED" if all(c["passed"] for c in checks) else "REVIEW"
            st.set_trust(mp, decision, checks, design={"n_fins": n, "fin_height_m": H, "fin_t_m": t},
                         mass_kg=rep.raw["mass"]["mass"], t_source_c=th["t_source_max_c"], dp_pa=th["pressure_drop_pa"])
            st.save()
            vs = validate_stage(usd, required_primvars={mp: ["temperature"]})
        checks.append({"name": "Omniverse stage valid", "value": None, "limit": None, "passed": vs["ok"],
                       "detail": "; ".join(vs["problems"]) or "UsdValidation clean"})
        decision = "APPROVED" if all(c["passed"] for c in checks) else "REVIEW"
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("accepted_mass", rep.raw["mass"]["mass"] * 1e3, "g", "exact B-rep mass, 6063 density")
        ex.log_metric("accepted_t_source", th["t_source_max_c"], "C", "closed-form source temperature")
        info.mark("NVIDIA USD BRep", ran=True)
        info.mark("PhysicsNeMo FullyConnected", ran=True)
        info.mark("Omniverse Kit RTX render", ran=False, why="needs Omniverse Kit and an RTX GPU")
        render_stage_png(usd, os.path.join(OUT, "usd_temperature.png"), elev=28, azim=-55,
                         colorbar=("temperature [C]", float(T.min()), float(T.max()), "inferno"),
                         title=f"heatsink.usda read back: {n} fins, {H * 1e3:.1f} x {t * 1e3:.2f} mm, "
                               f"T_source {th['t_source_max_c']:.1f} C")
        with open(os.path.join(OUT, "kit_render.py"), "w") as fo:
            fo.write(kit_render_script(usd, os.path.join(OUT, "heatsink_rtx.png")))
        checks_figure(checks, os.path.join(OUT, "checks.png"), title="Design, geometry and field checks",
                      decision=decision)
        summary = {"decision": decision, "design": {"n_fins": n, "fin_height_mm": H * 1e3, "fin_t_mm": t * 1e3},
                   "attempts": attempts, "heldout_rel_rms": err.tolist(), "t_source": th["t_source_max_c"],
                   "dp": th["pressure_drop_pa"], "fin_efficiency": th["fin_efficiency"], "fin_tip_c": tip,
                   "mass": rep.raw["mass"]["mass"], "tessellation": rep.tessellations[-1],
                   "stage": {k: vs[k] for k in ("ok", "problems", "meshes", "prims")}, "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as fo:
            json.dump(summary, fo, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "design", "t_source", "mass")}, indent=1))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
