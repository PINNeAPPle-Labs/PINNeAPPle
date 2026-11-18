"""Cold plate with microchannels: PINNeAPPle generates and checks, PhysicsNeMo FNO learns, Omniverse (OpenUSD) shows.

A 100 mm x 100 mm copper cold plate with microchannels slot-milled into it, two 110 W chips on the top face and water
at 40 C through the slots. The problem the example solves, checks and learns is the *conjugate* one: conduction
through the plate in the plane of the slots, coupled to the coolant through the walls of each slot (``h`` from the
Shah & London slot Nusselt number with Hausen's entry correction, water properties from Incropera Table A.6,
pressure drop from Muzychka-Yovanovich), with the coolant's own temperature rise closed against the heat the slots
actually take. PINNeAPPle's finite volumes make the data (``operator_benchmarks.darcy_reaction_2d``), a
``physicsnemo.models.fno.FNO`` with two input channels -- the conductance map and the slot-to-coolant coupling map --
learns design -> temperature, PINNeAPPle's checks decide whether the surrogate's fields can be used (certified
residual bounds for the reacting operator, boundary condition, the conjugate energy balance, the maximum principle,
a reference comparison and the training envelope), a second surrogate over the design variables answers the
engineering question (hot spot against pumping power, under the uncertainty of the correlation and of the flow),
and the result is written as an OpenUSD stage.

    python examples/nvidia/07_omniverse_physicsnemo_coldplate/run.py            # about 4 min on 4 CPU cores
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
from pinneapple_tools.nvidia import coldplate as cp
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure, field_panels
from pinneapple_tools.nvidia.omniverse import (
    FieldStage,
    grid_surface,
    kit_render_script,
    render_stage_png,
    validate_stage,
)
from pinneapple_tools.nvidia.physicsnemo import MLPSurrogate, relative_errors, sweep_change
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
DUTY = cp.ColdPlate()
T_LIMIT_C, DP_LIMIT_PA, RE_LIMIT = 85.0, 20_000.0, 2300.0
N = 32                        # solver / operator grid over the whole plate, guard ring included
SCALARS = ("rise", "dp", "pump")


def simulate(designs, n=N):
    """The coupled finite-volume solution of each design: the operator's inputs, its output and the three scalars.

    ``A``/``S``/``F``: the dimensionless conductance, slot coupling and source of ``-div(a grad u) + s u = f``;
    ``U``: its solution (``T = T_in + DT u``); ``rise``/``dp``/``pump``: hot spot above the inlet, pressure drop and
    pumping power of the same designs.
    """
    out = {k: [] for k in ("A", "S", "F", "U", "rise", "dp", "pump", "t_max", "dt_f", "energy_closure")}
    for d in designs:
        r = cp.solve(cp.plate_for(d, DUTY), n)
        fl = r["fields"]
        out["A"].append(fl["a"])
        out["S"].append(fl["s"])
        out["F"].append(fl["f0"] + fl["s"] * r["theta_f"])
        out["U"].append(r["u"])
        out["rise"].append(r["t_max_c"] - DUTY.coolant.t_inlet_c)
        out["dp"].append(r["state"].dp)
        out["pump"].append(r["state"].pumping_w)
        out["t_max"].append(r["t_max_c"])
        out["dt_f"].append(r["dt_f"])
        out["energy_closure"].append(r["energy_closure"])
    return {k: np.asarray(v) for k, v in out.items()}


def uq_samples(design, n_mc, seed):
    """Monte-Carlo draws around one design: the flow and the correlation's ``h`` as the plant varies, the chip power
    as the duty varies. Power scales the rise exactly (the plate problem is linear), so it enters as a factor."""
    rng = np.random.default_rng(seed)
    flow = design["flow_lpm"] * np.exp(rng.normal(0.0, 0.10, n_mc))
    h = design["h_scale"] * rng.uniform(0.85, 1.15, n_mc)
    power = rng.uniform(0.95, 1.05, n_mc)
    return flow, h, power


def decision_of(checks, field_decision):
    """The field verdict can only be lowered by what the rest of the run found: any open check gives REVIEW."""
    if field_decision == "REJECT":
        return "REJECT"
    if field_decision == "REVIEW" or any(c.get("passed") is not True for c in checks):
        return "REVIEW"
    return "APPROVED"


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    n_train, n_test = (16, 6) if smoke else (200, 50)
    n_mc = 40 if smoke else 400
    budget = ComputeBudget(steps=6) if smoke else ComputeBudget(steps=2400)
    info = RunInfo("07_omniverse_physicsnemo_coldplate")
    os.makedirs(OUT, exist_ok=True)
    architecture_diagram(
        "07 Cold plate: PINNeAPPle + Omniverse + PhysicsNeMo",
        [Block("gen", "Designs (Latin hypercube):\nchannels, slot width, flow, h", "pinneapple", 0, 0),
         Block("hyd", "Coolant model: Shah & London Nu,\nHausen, Muznycka-Yovanovich dp", "pinneapple", 1, 0),
         Block("fv", "Conjugate finite volumes\n-div(a grad u) + s u = f", "pinneapple", 2, 0),
         Block("fno", "physicsnemo.models.fno.FNO\n(conductance, coupling) -> T", "physicsnemo", 2, 1),
         Block("chk", "Physics checks: certified residual\nbounds, conjugate energy balance", "pinneapple", 3, 1),
         Block("mlp", "FullyConnected on the design\nvariables: hot spot, dp, pump", "physicsnemo", 3, 0),
         Block("uq", "Monte-Carlo the plant:\nP(hot spot > limit), Pareto", "pinneapple", 4, 1),
         Block("trust", "Trust decision\nAPPROVED / REVIEW / REJECT", "pinneapple", 5, 1),
         Block("usd", "OpenUSD stage (usd-core)\nT primvar + coupling + trust", "omniverse", 5, 0),
         Block("kit", "Omniverse Kit / USD Composer\nRTX viewport", "omniverse", 6, 0, ran=False)],
        [("gen", "hyd", "channels"), ("gen", "fv", "a, s, f"), ("hyd", "fv", "h, dp"), ("fv", "fno", "(a, s) -> u"),
         ("fno", "chk", "u predicted"), ("fv", "mlp", "hot spot, dp, pump"), ("mlp", "uq", "surrogate"),
         ("chk", "trust", "checks"), ("uq", "trust", "robustness"), ("fno", "usd", "T field"),
         ("trust", "usd", "decision"), ("usd", "kit", ".usda")],
        os.path.join(OUT, "architecture.png"),
        subtitle="Data, checks and decision by PINNeAPPle; surrogates by PhysicsNeMo; scene in OpenUSD for Omniverse")

    with experiment("examples/nvidia/07_omniverse_physicsnemo_coldplate", title="Cold plate: conjugate coolant model, "
                    "PhysicsNeMo FNO and FullyConnected, PINNeAPPle checks, OpenUSD scene", domain="heat transfer",
                    methods=["FNO (PhysicsNeMo)", "FullyConnected (PhysicsNeMo)", "FVM"],
                    modules=["pinneapple_tools.nvidia.coldplate", "pinneapple_neural.physicsnemo_bridge",
                             "pinneapple_physics.darcy_checks", "pinneapple_tools.nvidia.omniverse"],
                    out_dir=base, register=False,
                    params={"n": N, "n_train": n_train, "n_test": n_test, "n_mc": n_mc, "budget": budget.__dict__,
                            "duty": {"power_w": DUTY.total_power, "t_inlet_c": DUTY.coolant.t_inlet_c,
                                     "material": DUTY.material, "base_t": DUTY.base_t,
                                     "channel_depth": DUTY.channel_depth,
                                     "chips": [c.__dict__ for c in DUTY.chips]},
                            "bounds": cp.BOUNDS, "T_limit_C": T_LIMIT_C, "dp_limit_Pa": DP_LIMIT_PA}) as ex:
        with info.step("designs + conjugate finite volumes (PINNeAPPle)"):
            d_tr = cp.sample_designs(n_train, seed=0, duty=DUTY)
            d_te = cp.sample_designs(n_test, seed=1, duty=DUTY)
            X, tr = cp.design_matrix(d_tr), simulate(d_tr)
            Xt, te = cp.design_matrix(d_te), simulate(d_te)
        ex.log_metric("designs", len(d_tr) + len(d_te), "", "coupled finite-volume solutions behind the surrogates")
        ex.log_metric("energy_closure_max", float(np.abs(np.r_[tr["energy_closure"], te["energy_closure"]]).max()),
                      "", "largest |heat to coolant + heat to walls - chip power| / chip power over all designs")

        # ---- surrogate 1: the field, from the conductance, coupling and source maps ------------------------------
        # Three input channels, not two: the conjugate problem the design solves is
        # -div(a grad u) + s u = f0 + s theta_f, and the source a slot gives back to the plate carries the coolant's
        # own temperature rise, which no map of (a, s) alone can recover. Handing the FNO the source it actually
        # solves is the difference between 5 % and 10 % on the held-out fields.
        pred = physicsnemo_fno(N, in_channels=3, latent_channels=32 if not smoke else 8,
                               num_fno_layers=4 if not smoke else 2, num_fno_modes=12 if not smoke else 4,
                               decoder_layer_size=64 if not smoke else 16)
        curve = []

        def as_input(d):
            return np.stack([d["A"], d["S"], d["F"]], 1)

        def evaluate():
            e = np.linalg.norm((pred.predict(as_input(te)) - te["U"]).reshape(n_test, -1), axis=1) \
                / np.linalg.norm(te["U"].reshape(n_test, -1), axis=1)
            curve.append(float(e.mean()))
            return {"test_rel_l2": float(e.mean())}

        with info.step("train the field operator (PhysicsNeMo FNO)"):
            res = train_grid_operator(pred, as_input(tr), tr["U"], budget,
                                      batch_size=8 if not smoke else 4, lr=2e-3, evaluate=evaluate,
                                      eval_at=np.linspace(0.1, 0.9, 9))
        info.mark("PhysicsNeMo FNO training", ran=True,
                  how=f"CPU, {res.run.steps} steps, {res.epochs:.1f} epochs, {pred.in_channels} input channels")

        # ---- surrogate 2: the design variables, for the engineering question ----------------------------------
        mlp = MLPSurrogate(X.shape[1], len(SCALARS), layer_size=64, num_layers=3)
        Y = np.stack([tr[k] for k in SCALARS], 1)
        Yt = np.stack([te[k] for k in SCALARS], 1)
        with info.step("train the design surrogate (PhysicsNeMo FullyConnected)"):
            mlp_run = mlp.fit(X, Y, ComputeBudget(steps=200 if smoke else 8000), batch_size=64, lr=3e-3)
        scalar_err = relative_errors(mlp.predict(Xt), Yt)
        lo = np.array([cp.BOUNDS[k][0] for k in cp.DESIGN_VARS])
        hi = np.array([cp.BOUNDS[k][1] for k in cp.DESIGN_VARS])
        nominal = np.array([DUTY.n_channels, DUTY.channel_width, DUTY.coolant.flow_lpm, 1.0], float)

        def sweep(i, out, sign):
            a, b = nominal.copy(), nominal.copy()
            a[i], b[i] = lo[i], hi[i]
            return sweep_change(mlp, a, b, out, sign)

        trends = {"hot spot falls with flow": sweep(2, 0, -1),
                  "hot spot falls with the film coefficient": sweep(3, 0, -1),
                  "hot spot falls with the channel count": sweep(0, 0, -1),
                  "pressure drop grows with flow": sweep(2, 1, +1),
                  "pumping power grows with flow": sweep(2, 2, +1)}

        # ---- the surrogate's fields, checked against the conjugate physics -------------------------------------
        with info.step("checks of the surrogate's fields (PINNeAPPle)"):
            u_hat = pred.predict(as_input(te))
            env = dc.CoefficientEnvelope.fit(tr["A"])
            report, checks = dc.darcy_trust_report(te["A"], u_hat, te["F"], model="PhysicsNeMo FNO",
                                                   reference=te["U"], envelope=env, sink=te["S"], exact=True)
            ref_report, _ = dc.darcy_trust_report(te["A"], te["U"], te["F"], model="finite volumes",
                                                 sink=te["S"], exact=True)
        rel = float(np.mean(np.linalg.norm((u_hat - te["U"]).reshape(n_test, -1), axis=1)
                            / np.linalg.norm(te["U"].reshape(n_test, -1), axis=1)))
        fl0 = cp.fields(DUTY, N)                          # the plate's own scales and the guard ring of this duty
        dt, inner = fl0["DT"], fl0["inner"]
        rise_hat = dt * u_hat.reshape(n_test, -1).max(1)
        flips = int(np.sum((rise_hat + DUTY.coolant.t_inlet_c <= T_LIMIT_C)
                           != (te["rise"] + DUTY.coolant.t_inlet_c <= T_LIMIT_C)))
        ex.log_metric("test_rel_l2", rel, "", f"mean relative L2 of the field over {n_test} held-out designs")
        ex.log_metric("max_hotspot_error_K", float(np.abs(rise_hat - te["rise"]).max()), "K",
                      "largest |hot spot from the surrogate - hot spot from the coupled solve|")
        ex.log_metric("train_wall_s", res.run.wall_s, "s", "wall clock of the FNO training on this CPU")
        for c in checks:
            if c.passed is not None:
                ex.check(c.name, bool(c.passed), value=c.value, max=c.limit, detail=c.detail, kind=c.kind)

        # ---- the engineering question: a design, and the same design at the edge ---------------------------------
        with info.step("robust design and uncertainty (PINNeAPPle)"):
            P = mlp.predict(X)
            ok_dp = P[:, 1] <= DP_LIMIT_PA
            rise_lim = T_LIMIT_C - DUTY.coolant.t_inlet_c

            def cheapest(mask, fallback=None):
                idx = np.where(mask, P[:, 2], np.inf)
                return int(np.argmin(idx)) if mask.any() else (fallback if fallback is not None else int(np.argmin(
                    np.where(ok_dp, P[:, 2], np.inf))))

            pick = cheapest(ok_dp & (P[:, 0] <= rise_lim - 3.0))
            pick_edge = cheapest(ok_dp & (P[:, 0] > rise_lim - 2.0) & (P[:, 0] <= rise_lim), fallback=pick)
            design, design_edge = d_tr[pick], d_tr[pick_edge]

            def uq(design, seed, grid):
                """Failure probability of one design over the plant's spread, by the surrogate and by coupled solves."""
                flow, h, power = uq_samples(design, n_mc, seed=seed)
                rise_mlp = power * mlp.predict(np.column_stack([
                    np.full(n_mc, design["n_channels"]), np.full(n_mc, design["channel_width"]), flow, h]))[:, 0]
                rise_fvm = np.empty(n_mc)
                for i in range(n_mc):
                    cfg = cp.plate_for({**design, "flow_lpm": float(flow[i]), "h_scale": float(h[i])}, DUTY)
                    rise_fvm[i] = power[i] * (cp.solve(cfg, grid)["t_max_c"] - DUTY.coolant.t_inlet_c)
                hot = lambda r: r + DUTY.coolant.t_inlet_c > T_LIMIT_C      # noqa: E731
                return {"p_mlp": float(np.mean(hot(rise_mlp))), "p_fvm": float(np.mean(hot(rise_fvm))),
                        "rise_mlp": rise_mlp, "rise_fvm": rise_fvm, "design": design}

            uq_robust = uq(design, 2, 16 if smoke else 24)
            uq_edge = uq(design_edge, 3, 16 if smoke else 24) if pick_edge != pick else uq_robust
            chosen = cp.plate_for(design, DUTY)
            solved = cp.solve(chosen, 48)
            eng = cp.checks(solved, t_limit_c=T_LIMIT_C, dp_budget_pa=DP_LIMIT_PA)
        for c in eng:
            ex.check(f"design: {c['name']}", c["passed"], value=c["value"], max=c["limit"], detail=c["detail"])
        for name, ch in trends.items():
            ex.check(f"trend: {name}", ch <= 0.0, value=float(ch), max=0.0,
                     detail="sign x (value at the top of the range - value at the bottom): positive means the model "
                            "has the direction backwards")
        for o, e in zip(SCALARS, scalar_err, strict=True):
            ex.check(f"held-out error, design surrogate: {o}", bool(e <= 0.05), value=float(e), max=0.05)
        ex.check("the surrogate never flips the design decision (hot spot vs limit)", flips == 0, value=float(flips),
                 max=0.0, detail="held-out designs where the FNO's hot spot and the coupled solve disagree on "
                 "T_max <= limit")
        # The robust pick must agree with the coupled solves. The design at the edge is the opposite kind of check:
        # selecting on the surrogate's own prediction also selects the designs where that prediction is optimistic
        # (regression to the mean), so the interesting statement is not that the two probabilities match there -- they
        # do not -- but that the design the surrogate puts just under the limit is not under it at all.
        for target, u, tol in (("robust design", uq_robust, 0.05),):
            ex.check(f"failure probability, {target}: surrogate vs coupled solves", abs(u["p_mlp"] - u["p_fvm"]) <= tol,
                     value=float(abs(u["p_mlp"] - u["p_fvm"])), max=tol,
                     detail=f"P(hot spot > {T_LIMIT_C:.0f} C) = {u['p_mlp']:.3f} (surrogate) vs {u['p_fvm']:.3f} "
                            f"(coupled solves, {n_mc} draws)")
        ex.check("the design at the edge of the limit really is worse",
                 uq_edge["p_fvm"] >= uq_robust["p_fvm"],
                 value=float(uq_robust["p_fvm"] - uq_edge["p_fvm"]), max=0.0,
                 detail=f"P(fail) = {uq_edge['p_fvm']:.2f} at the edge (the surrogate put it at {uq_edge['p_mlp']:.2f})"
                        f" against {uq_robust['p_fvm']:.2f} for the robust pick: choosing on the surrogate's own "
                        "prediction chooses where that prediction is optimistic")
        ex.log_metric("edge_design_p_surrogate", float(uq_edge["p_mlp"]), "",
                      "P(hot spot > limit) for the design the surrogate put just under the limit")
        ex.log_metric("edge_design_p_fvm", float(uq_edge["p_fvm"]), "",
                      "the same, from coupled solves: the surrogate's optimism at the design it selected")
        ex.check("the chosen design has margin at nominal conditions",
                 bool(solved["t_max_c"] <= T_LIMIT_C - 3.0), value=float(solved["t_max_c"]), max=T_LIMIT_C - 3.0,
                 detail="the design the surrogate picked, re-solved with the coupled model")
        extra = [{"name": f"design: {c['name']}", "passed": c["passed"]} for c in eng]
        extra += [{"name": k, "passed": v <= 0.0} for k, v in trends.items()]
        extra += [{"name": o, "passed": bool(e <= 0.05)} for o, e in zip(SCALARS, scalar_err, strict=True)]
        extra += [{"name": "uq robust", "passed": abs(uq_robust["p_mlp"] - uq_robust["p_fvm"]) <= 0.05},
                  {"name": "uq edge", "passed": uq_edge["p_fvm"] >= uq_robust["p_fvm"]}]
        decision = decision_of(extra, report.decisao)

        # ---- pictures and the OpenUSD stage --------------------------------------------------------------------
        med = int(np.argsort([np.linalg.norm(u_hat[i] - te["U"][i]) / np.linalg.norm(te["U"][i])
                              for i in range(n_test)])[n_test // 2])
        est = dc.estimated_error(u_hat[med], te["A"][med], te["F"][med], sink=te["S"][med])
        T_fv = DUTY.coolant.t_inlet_c + dt * te["U"][med]
        T_fno = DUTY.coolant.t_inlet_c + dt * u_hat[med]
        ext = (0, DUTY.length * 1e3, 0, DUTY.width * 1e3)
        field_panels([te["A"][med], te["S"][med], T_fv, T_fno, np.abs(T_fno - T_fv), dt * np.abs(est)],
                     ["in-plane conductance a", "slot coupling s", "T, coupled finite volumes", "T, PhysicsNeMo FNO",
                      "|T_FNO - T_FV|", "error estimated from the residual"],
                     os.path.join(OUT, "fields.png"), cmaps=["cividis", "viridis", "inferno", "inferno", "magma",
                                                             "magma"],
                     units=["-", "-", "K", "K", "K", "K"], shared=[(2, 3), (4, 5)], extent=ext, ncols=3,
                     suptitle=f"Held-out design (median error): FNO relative L2 = "
                     f"{np.linalg.norm(u_hat[med] - te['U'][med]) / np.linalg.norm(te['U'][med]):.3%}")
        hist = [hh for hh in res.history if hh.get("eval")]
        _figures(OUT, P, pick, pick_edge, solved, uq_robust, uq_edge, hist, checks, decision)
        with info.step("OpenUSD stage"):
            T = T_fno[inner].copy()
            Xf, Yf = np.meshgrid(fl0["X"].ravel(), fl0["Y"].ravel(), indexing="ij")
            V, F = grid_surface(Xf, Yf, 0.002 + 0.02 * (T - T.min()) / max(T.max() - T.min(), 1e-9))
            usd = os.path.join(OUT, "coldplate.usda")
            st = FieldStage(usd, "Cold plate, PhysicsNeMo FNO temperature")
            path = st.add_mesh("ColdPlate", V, F,
                               fields={"temperature": T.ravel(), "conductance": te["A"][med][inner].ravel(),
                                       "coupling": te["S"][med][inner].ravel(),
                                       "temperature_reference": T_fv[inner].ravel(),
                                       "estimated_error": (dt * np.abs(est))[inner].ravel()},
                               units={"temperature": "K", "conductance": "-", "coupling": "-",
                                      "temperature_reference": "K", "estimated_error": "K"}, color_by="temperature")
            st.set_trust(path, decision, [k.to_dict() for k in checks], model="physicsnemo.models.fno.FNO",
                         solver="pinneapple finite volumes, darcy_reaction_2d")
            st.save()
            v = validate_stage(usd, required_primvars={path: ["temperature", "conductance", "coupling"]})
            render_stage_png(usd, os.path.join(OUT, "usd_preview.png"),
                             colorbar=("T [K]", float(T.min()), float(T.max()), "inferno"),
                             title="coldplate.usda read back with OpenUSD (height = temperature)")
            with open(os.path.join(OUT, "kit_render.py"), "w") as f:
                f.write(kit_render_script(usd, os.path.join(OUT, "coldplate_rtx.png")))
        info.mark("OpenUSD stage (usd-core)", ran=True, how="authored and validated with pxr")
        info.mark("Omniverse Kit RTX render", ran=False, why="needs Omniverse Kit with an RTX GPU",
                  how="results/kit_render.py inside USD Composer")
        ex.check("usd_stage_valid", v["ok"], detail="; ".join(v["problems"]) or "UsdValidation clean, primvars sized")

        summary = {"decision": decision, "field_decision": report.decisao,
                   "reference_solver_decision": ref_report.decisao, "test_rel_l2": rel,
                   "max_hotspot_error_K": float(np.abs(rise_hat - te["rise"]).max()),
                   "fno_parameters": pred.n_parameters(), "fno_in_channels": pred.in_channels,
                   "training": res.to_dict() | {"history": None},
                   "design_surrogate": {"heldout_rel_rms": dict(zip(SCALARS, scalar_err.tolist(), strict=True)),
                                        "trend_violations": trends, "steps": mlp_run.steps},
                   "identity": {"chosen": design, "t_max_c": solved["t_max_c"], "dp_pa": solved["state"].dp,
                                "reynolds": solved["state"].re, "h_w_m2k": solved["state"].h,
                                "h_warnings": solved["state"].warnings, "dt_f_k": solved["dt_f"],
                                "energy_closure": solved["energy_closure"], "pumping_w": solved["state"].pumping_w},
                   "uncertainty": {"n_mc": n_mc,
                                   "robust": {"design": uq_robust["design"],
                                              "p_surrogate": uq_robust["p_mlp"], "p_fvm": uq_robust["p_fvm"]},
                                   "edge": {"design": uq_edge["design"],
                                            "p_surrogate": uq_edge["p_mlp"], "p_fvm": uq_edge["p_fvm"]}},
                   "energy_closure_max_all_designs": float(np.abs(np.r_[tr["energy_closure"],
                                                                        te["energy_closure"]]).max()),
                   "usd": {k: v[k] for k in ("ok", "problems", "meshes", "prims")}
                   | {"validators": len(v["compliance"].get("validators", []))},
                   "checks": [k.to_dict() for k in checks] + eng + [{"name": k, "passed": bool(x <= 0.0),
                                                                    "value": float(x)} for k, x in trends.items()]}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "test_rel_l2", "max_hotspot_error_K",
                                              "reference_solver_decision", "uncertainty")}, indent=1, default=float))
    return summary

def _figures(OUT, P, pick, pick_edge, solved, uq_robust, uq_edge, hist, checks, decision):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # training curve
    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.semilogy([h["step"] for h in hist], [h["eval"]["test_rel_l2"] for h in hist], "o-")
    ax.axhline(dc.DEFAULT_TOLERANCES["reference"], color="r", ls="--", lw=1, label="reference limit (5 %)")
    ax.set_xlabel("step")
    ax.set_ylabel("relative L2 on held-out designs")
    ax.set_title("PhysicsNeMo FNO, two input channels (conductance, coupling)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "training.png"), dpi=130)
    plt.close(fig)

    # design space: pumping power against hot spot, the feasible box, and the pick
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    sc = ax.scatter(P[:, 2] * 1e3, P[:, 0] + DUTY.coolant.t_inlet_c, c=P[:, 1], s=14, cmap="viridis", alpha=0.8)
    fig.colorbar(sc, ax=ax, label="pressure drop [Pa]")
    ax.axhline(T_LIMIT_C, color="r", ls="--", lw=1, label=f"hot spot limit {T_LIMIT_C:.0f} C")
    ax.plot(P[pick, 2] * 1e3, P[pick, 0] + DUTY.coolant.t_inlet_c, "r*", ms=15,
            label=f"robust pick, re-solved to {solved['t_max_c']:.1f} C")
    if pick_edge != pick:
        ax.plot(P[pick_edge, 2] * 1e3, P[pick_edge, 0] + DUTY.coolant.t_inlet_c, "k^", ms=10,
                label="lightest design at the edge of the limit")
    ax.set_xlabel("pumping power [mW]")
    ax.set_ylabel("hot spot [C]")
    ax.set_title("The design space through the design surrogate", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "design_space.png"), dpi=130)
    plt.close(fig)

    # uncertainty: the robust pick and the design at the edge, surrogate against coupled solves
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    for ax, u, tag in zip(axs, (uq_robust, uq_edge), ("robust pick", "lightest design at the edge"), strict=True):
        ax.hist(u["rise_mlp"] + DUTY.coolant.t_inlet_c, bins=30, alpha=0.55, color="#1f77b4",
                label=f"design surrogate (P = {u['p_mlp']:.2f})")
        ax.hist(u["rise_fvm"] + DUTY.coolant.t_inlet_c, bins=30, alpha=0.55, color="#d62728",
                label=f"coupled solves (P = {u['p_fvm']:.2f})")
        ax.axvline(T_LIMIT_C, color="k", ls="--", lw=1.2, label=f"limit {T_LIMIT_C:.0f} C")
        ax.set_xlabel("hot spot [C] over the plant's spread")
        ax.set_title(tag, fontsize=9)
        ax.legend(fontsize=7)
    axs[0].set_ylabel("draws")
    fig.suptitle("Uncertainty: flow, film coefficient and duty", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "uncertainty.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="PhysicsNeMo FNO, PINNeAPPle checks",
                  decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
