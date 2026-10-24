"""A PhysicsNeMo model behind PINNeAPPle Model Services: every answer is physics-checked, and a request the surrogate
cannot answer safely is routed to PINNeAPPle's solver.

``physicsnemo.models.fno.FNO`` learns the cold-plate conduction map of example 07 (conductivity layout -> temperature,
32 x 32) and is served on ``/v1/infer`` next to the finite-volume solver. Each answer carries the residual-based
checks (certified error bounds from the PDE residual, boundary condition, energy balance, maximum principle) and the
layout against the training envelope: no reference solution is needed at serving time. A client sends in-distribution
layouts and out-of-distribution ones (other contrast and length scale) and routes to the solver whenever the surrogate
is not APPROVED, then compares latency and accuracy.

    python examples/nvidia/11_model_services_physicsnemo_operator/run.py       # ~2 min on 4 CPU cores
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
from pinneapple_tools.model_services import ModelServiceClient, Registry, serve_in_thread
from pinneapple_tools.model_services.grid import darcy_solver_service, grid_operator_service
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))


def layouts(m, n, seed, alpha=2.0, tau=3.0, hi=12.0, lo=3.0):
    return ob.threshold_coefficient(ob.periodic_grf(m, (n, n), alpha=alpha, tau=tau, seed=seed), high=hi, low=lo)


def cost_scaling(sizes, reps: int = 4) -> list[dict]:
    """Time per layout of the FV solve and of an FNO inference with the served architecture, by grid size. The FNO's
    cost does not depend on its weights, so an untrained copy at each size is timed; no accuracy is claimed."""
    import time
    out = []
    for n in sizes:
        A = layouts(reps, n, 9)
        t = time.perf_counter()
        for a in A:
            ob.darcy_2d(a)
        fv = (time.perf_counter() - t) / reps
        net = physicsnemo_fno(n, latent_channels=32, num_fno_layers=4, num_fno_modes=min(12, n // 2),
                              decoder_layer_size=64)
        net.predict(A[:1])
        t = time.perf_counter()
        for i in range(reps):
            net.predict(A[i:i + 1])
        out.append({"n": n, "fv_ms": 1e3 * fv, "fno_ms": 1e3 * (time.perf_counter() - t) / reps})
    return out


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("11_model_services_physicsnemo_operator")
    architecture_diagram(
        "11 A PhysicsNeMo model served with its physics: PINNeAPPle + Model Services + PhysicsNeMo",
        [Block("data", "Finite-volume data\n(cold-plate layouts)", "pinneapple", 0, 0),
         Block("fno", "physicsnemo.models.fno.FNO\n(trained on CPU)", "physicsnemo", 1, 0),
         Block("svc", "Model Services: FNO and FV\nsolver on /v1/infer", "services", 2, 0),
         Block("chk", "Per-answer checks: certified\nresidual bounds, BC, energy, envelope", "pinneapple", 2, 1),
         Block("route", "Client routing: surrogate first,\nsolver when not APPROVED", "services", 3, 0),
         Block("eval", "Accuracy and latency of\nthe routed answers", "pinneapple", 3, 1)],
        [("data", "fno", "(a, u)"), ("fno", "svc", "model"), ("svc", "chk", "u"), ("chk", "svc", "decision"),
         ("svc", "route", "answer + trust"), ("route", "eval", "")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The service says when its own answer breaks the physics; the client then asks the solver")
    n = 16 if smoke else 32
    with experiment("examples/nvidia/11_model_services_physicsnemo_operator", title="PhysicsNeMo FNO served with "
                    "per-answer physics checks and solver fallback", domain="heat transfer",
                    methods=["FNO (PhysicsNeMo)", "FVM", "model serving"], out_dir=base, register=False,
                    modules=["pinneapple_tools.model_services.grid", "pinneapple_physics.darcy_checks"]) as ex:
        with info.step("data + train (PhysicsNeMo FNO)"):
            a_tr = layouts(24 if smoke else 400, n, 0)
            u_tr = np.stack([ob.darcy_2d(a) for a in a_tr])
            fno = physicsnemo_fno(n, latent_channels=8 if smoke else 32, num_fno_layers=2 if smoke else 4,
                                  num_fno_modes=4 if smoke else 12, decoder_layer_size=16 if smoke else 64)
            tr = train_grid_operator(fno, a_tr, u_tr, ComputeBudget(steps=6 if smoke else 2700),
                                     batch_size=4 if smoke else 8, lr=2e-3)
        env = dc.CoefficientEnvelope.fit(a_tr)
        reg = Registry([grid_operator_service(fno, "coldplate-fno", envelope=env,
                                              card={"architecture": "physicsnemo.models.fno.FNO", "grid": n,
                                                    "train_layouts": len(a_tr), "train_steps": tr.run.steps}),
                        darcy_solver_service()])
        srv, url = serve_in_thread(reg)
        c = ModelServiceClient(url)
        req = {"in-distribution": layouts(4 if smoke else 20, n, 1),
               "higher contrast (k 2 / 20)": layouts(2 if smoke else 8, n, 2, hi=20.0, lo=2.0),
               "finer features (tau 6)": layouts(2 if smoke else 8, n, 3, tau=6.0)}
        rows = []
        try:
            with info.step("requests: surrogate first, solver fallback"):
                for grp, A in req.items():
                    for a in A:
                        r = c.infer("coldplate-fno", {"a": a.tolist()})
                        t_s = c.last_latency_ms
                        routed, t_f = "surrogate", 0.0
                        u = np.asarray(r["outputs"]["u"])
                        if r["trust"]["decision"] != "APPROVED":
                            rf = c.infer("pinneapple-darcy-fv", {"a": a.tolist()})
                            t_f = c.last_latency_ms
                            u_final, routed = np.asarray(rf["outputs"]["u"]), "solver"
                            fv_dec = rf["trust"]["decision"]
                        else:
                            u_final, fv_dec = u, None
                        ref = ob.darcy_2d(a)
                        e_s = float(np.linalg.norm(u - ref) / np.linalg.norm(ref))
                        e_f = float(np.linalg.norm(u_final - ref) / np.linalg.norm(ref))
                        failed = [ch["name"] for ch in r["trust"]["checks"] if ch["passed"] is False]
                        bound = next(ch["value"] for ch in r["trust"]["checks"]
                                     if ch["name"] == "PDE residual (certified L2 error bound)")
                        rows.append({"group": grp, "surrogate_decision": r["trust"]["decision"], "failed": failed,
                                     "routed_to": routed, "solver_decision": fv_dec, "surrogate_rel_l2": e_s,
                                     "final_rel_l2": e_f, "certified_l2_bound": bound,
                                     "surrogate_ms": t_s, "fallback_ms": t_f})
        finally:
            srv.shutdown()
        with info.step("cost against grid size (FV solve vs FNO inference, same architecture)"):
            scaling = cost_scaling((16, 32) if smoke else (32, 64, 128, 256))
        by = {g: [r for r in rows if r["group"] == g] for g in req}
        ood = [r for g, rs in by.items() if g != "in-distribution" for r in rs]
        checks = [{"name": "every surrogate answer carries a trust report", "value": None, "limit": None,
                   "passed": all(r["surrogate_decision"] in ("APPROVED", "REVIEW", "REJECT") for r in rows)},
                  {"name": "out-of-distribution layouts never APPROVED for the surrogate",
                   "value": float(np.mean([r["surrogate_decision"] == "APPROVED" for r in ood])), "limit": 0.0,
                   "passed": not any(r["surrogate_decision"] == "APPROVED" for r in ood)},
                  {"name": "every solver fallback APPROVED", "value": None, "limit": None,
                   "passed": all(r["solver_decision"] == "APPROVED" for r in rows if r["routed_to"] == "solver")},
                  {"name": "routed answers within 1 % of the reference", "value": max(r["final_rel_l2"] for r in rows),
                   "limit": 0.01, "passed": max(r["final_rel_l2"] for r in rows) <= 0.01}]
        decision = "APPROVED" if all(c_["passed"] for c_ in checks) else "REJECT"
        for c_ in checks:
            ex.check(c_["name"], c_["passed"], value=c_["value"], max=c_["limit"])
        for g, rs in by.items():
            ex.log_metric(f"surrogate_rel_l2_{g.split(' ')[0]}", float(np.mean([r["surrogate_rel_l2"] for r in rs])), "",
                          "mean over the requests of the group, against the FV solution")
        info.mark("PhysicsNeMo FNO", ran=True, how=f"{tr.run.steps} steps, CPU")
        info.mark("PINNeAPPle Model Services", ran=True, how=url)
        _figures(OUT, rows, req, checks, decision, scaling)
        summary = {"decision": decision, "requests": rows, "train_steps": tr.run.steps, "train_wall_s": tr.run.wall_s,
                   "cost_scaling": scaling,
                   "groups": {g: {"n": len(rs), "approved": sum(r["surrogate_decision"] == "APPROVED" for r in rs),
                                  "routed_to_solver": sum(r["routed_to"] == "solver" for r in rs),
                                  "surrogate_rel_l2_mean": float(np.mean([r["surrogate_rel_l2"] for r in rs])),
                                  "surrogate_ms_median": float(np.median([r["surrogate_ms"] for r in rs])),
                                  "fallback_ms_median": float(np.median([r["fallback_ms"] for r in rs
                                                                         if r["routed_to"] == "solver"] or [0.0]))}
                              for g, rs in by.items()}, "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "groups")}, indent=1))
    return summary


def _figures(OUT, rows, req, checks, decision, scaling):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    col = {"APPROVED": "#2e9d4f", "REVIEW": "#e0a000", "REJECT": "#c0392b"}
    fig, axs = plt.subplots(1, 3, figsize=(16, 4.3), gridspec_kw={"width_ratios": [1.3, 1, 1]})
    groups = list(req)
    for i, g in enumerate(groups):
        rs = [r for r in rows if r["group"] == g]
        for j, r in enumerate(rs):
            x = i + (j - len(rs) / 2) * 0.025
            axs[0].plot(x, r["surrogate_rel_l2"], "o", color=col[r["surrogate_decision"]], ms=5)
            axs[0].plot(x, r["certified_l2_bound"], "^", mfc="none", color="#555555", ms=5)
    for d, cc in col.items():
        if any(r["surrogate_decision"] == d for r in rows):
            axs[0].plot([], [], "o", color=cc, label=f"true error, answer {d}")
    axs[0].plot([], [], "^", mfc="none", color="#555555", label="certified error bound (from the residual)")
    axs[0].axhline(0.05, color="k", ls="--", lw=1, label="limit on the certified bound (5 %)")
    axs[0].set_xticks(range(len(groups)))
    axs[0].set_xticklabels(groups, fontsize=8)
    axs[0].set_yscale("log")
    axs[0].set_ylabel("relative L2 error")
    axs[0].set_title("FNO answers: true error vs what the service can certify\n(no reference at serving time; "
                     "a bound above 5 % is not approved)", fontsize=9)
    axs[0].legend(fontsize=7, loc="upper left")
    axs[0].grid(alpha=0.3)
    s_ms = [r["surrogate_ms"] for r in rows]
    f_ms = [r["fallback_ms"] for r in rows if r["routed_to"] == "solver"]
    axs[1].hist([s_ms, f_ms], bins=12, label=["FNO request", "FV solver request"], color=["#4f7d00", "#b88a00"])
    axs[1].set_xlabel("round trip [ms] (inference + checks + HTTP)")
    axs[1].set_ylabel("requests")
    axs[1].legend(fontsize=7)
    axs[1].set_title("Latency through /v1/infer at 32 x 32:\nthe FV solve is tiny, the FNO pays torch overhead",
                     fontsize=9)
    n = [r["n"] for r in scaling]
    axs[2].loglog(n, [r["fv_ms"] for r in scaling], "s-", color="#b88a00", label="FV solve")
    axs[2].loglog(n, [r["fno_ms"] for r in scaling], "o-", color="#4f7d00", label="FNO inference (same architecture)")
    axs[2].set_xticks(n)
    axs[2].set_xticklabels([str(k) for k in n])
    axs[2].minorticks_off()
    axs[2].set_xlabel("grid n (n x n cells)")
    axs[2].set_ylabel("time per layout [ms]")
    axs[2].set_title("Where a surrogate pays: cost against grid size\n(measured on this CPU, one layout per call)",
                     fontsize=9)
    axs[2].legend(fontsize=7)
    axs[2].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "routing.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Served model checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
