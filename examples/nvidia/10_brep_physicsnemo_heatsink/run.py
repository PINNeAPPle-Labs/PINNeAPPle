"""Heat-sink design from exact geometry: NVIDIA USD BRep builds every candidate, PINNeAPPle rates it, PhysicsNeMo
learns the design space, PINNeAPPle checks the surrogate and verifies the design it proposes.

An extruded aluminium (6063) plate-fin heat sink, 100 x 100 mm, 1 mm fin-root fillets, cools a 60 W, 40 x 40 mm
source in ducted 3 m/s air. Each candidate (fin count, height, thickness) is built as an exact B-rep (its mass
includes the root fillets a closed form misses) and rated by PINNeAPPle's closed-form model (Teertstra forced
convection, fin efficiency, spreading resistance, pressure drop). ``physicsnemo.models.mlp.FullyConnected`` learns
source temperature, pressure drop and exact mass from 120 designs. PINNeAPPle checks the surrogate on 30 held-out
designs and against the monotonic trends the physics requires, searches it for the lightest design meeting
T_source <= 75 C and dp <= 30 Pa, and accepts a design only after re-evaluating it with the B-rep and the closed
form (lightest first; a proposal that fails verification is rejected and the next one is tried).

    PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/10_brep_physicsnemo_heatsink/run.py   # ~2 min
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_lab import experiment
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.heatsink import Duty, design_table, sample_designs, verify
from pinneapple_tools.nvidia.physicsnemo import MLPSurrogate, relative_errors, sign_violations
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
DUTY = Duty(root_fillet=0.001)
T_MAX, DP_MAX = 75.0, 30.0
OUTS = ("t_rise", "dp", "mass")


def table_xy(rows):
    X = np.array([[r["n_fins"], r["fin_height"], r["fin_t"]] for r in rows], float)
    Y = np.array([[r["t_source"] - DUTY.t_ambient_c, r["dp"], r["mass"]] for r in rows], float)
    return X, Y


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("10_brep_physicsnemo_heatsink")
    architecture_diagram(
        "10 Heat-sink design from exact geometry: PINNeAPPle + NVIDIA USD BRep + PhysicsNeMo",
        [Block("doe", "Design sampling (LHS):\nfins, height, thickness", "pinneapple", 0, 0),
         Block("brep", "USD BRep: exact solid with\nroot fillets -> mass, area", "brep", 1, 0),
         Block("cf", "Closed-form rating: Teertstra,\nfin efficiency, spreading, dp", "pinneapple", 1, 1),
         Block("nn", "physicsnemo FullyConnected:\n(n, H, t) -> T, dp, mass", "physicsnemo", 2, 0),
         Block("chk", "Held-out error, physical\ntrends (monotonicity)", "pinneapple", 2, 1),
         Block("opt", "Search the surrogate: lightest\nwith T <= 75 C, dp <= 30 Pa", "physicsnemo", 3, 0),
         Block("ver", "Verify the proposal: B-rep +\nclosed form -> decision", "pinneapple", 3, 1)],
        [("doe", "brep", "recipes"), ("doe", "cf", "designs"), ("brep", "nn", "exact mass"), ("cf", "nn", "T, dp"),
         ("nn", "chk", "predictions"), ("nn", "opt", "surrogate"), ("opt", "ver", "candidate"), ("chk", "ver", "checks")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The surrogate proposes; the exact geometry and the physics model decide")
    n_train, n_test = (16, 6) if smoke else (120, 30)
    with experiment("examples/nvidia/10_brep_physicsnemo_heatsink", title="Heat-sink design: USD BRep geometry, "
                    "PhysicsNeMo surrogate, PINNeAPPle rating and verification", domain="heat transfer",
                    methods=["B-rep (SMLib)", "FullyConnected (PhysicsNeMo)", "closed-form heat sink"], out_dir=base,
                    register=False, modules=["pinneapple_tools.nvidia.heatsink", "pinneapple_tools.nvidia.physicsnemo"],
                    params={"duty": DUTY.__dict__, "T_max_C": T_MAX, "dp_max_Pa": DP_MAX, "n_train": n_train}) as ex:
        with info.step("designs: USD BRep + closed form"):
            train = design_table(sample_designs(n_train, seed=0), DUTY)
            test = design_table(sample_designs(n_test, seed=1), DUTY)
        X, Y = table_xy(train)
        Xt, Yt = table_xy(test)
        mass_cf_err = max(abs(r["mass"] - r["mass_closed_form"]) / r["mass"] for r in train)
        vol_err = max(abs(r["volume"] - r["volume_closed_form"]) / r["volume"] for r in train)
        net = MLPSurrogate(3, 3, layer_size=64, num_layers=3)
        with info.step("train (PhysicsNeMo FullyConnected)"):
            run = net.fit(X, Y, ComputeBudget(steps=200 if smoke else 15000), batch_size=64, lr=3e-3)
        err = relative_errors(net.predict(Xt), Yt)
        grid_ax = [np.arange(6, 21, 2.0), np.linspace(0.016, 0.044, 8), np.linspace(0.0011, 0.0019, 5)]
        mono = {"T falls with fin height": sign_violations(net, grid_ax, 1, 0, -1),
                "mass grows with fin count": sign_violations(net, grid_ax, 0, 2, +1),
                "mass grows with fin height": sign_violations(net, grid_ax, 1, 2, +1),
                "dp grows with fin count": sign_violations(net, grid_ax, 0, 1, +1)}
        # search the surrogate on a fine grid of valid designs
        nn_, hh, tt = np.meshgrid(np.arange(6, 21), np.linspace(0.015, 0.045, 61), np.linspace(0.001, 0.003, 41),
                                  indexing="ij")
        G = np.c_[nn_.ravel(), hh.ravel(), tt.ravel()]
        G = G[(0.1 - G[:, 0] * G[:, 2]) / (G[:, 0] - 1) >= 0.002]
        P = net.predict(G)
        ok = (P[:, 0] + DUTY.t_ambient_c <= T_MAX) & (P[:, 1] <= DP_MAX)
        order = np.argsort(np.where(ok, P[:, 2], np.inf))[:int(ok.sum())]
        attempts = []
        with info.step("verify proposals (USD BRep + closed form), lightest first"):
            for i in order[:8]:
                c = G[i]
                v = verify(int(c[0]), float(c[1]), float(c[2]), DUTY)
                good = v["t_source"] <= T_MAX and v["dp"] <= DP_MAX
                attempts.append({"n_fins": int(c[0]), "fin_height_mm": c[1] * 1e3, "fin_t_mm": c[2] * 1e3,
                                 "predicted_t": float(P[i, 0] + DUTY.t_ambient_c), "verified_t": v["t_source"],
                                 "verified_dp": v["dp"], "verified_mass": v["mass"], "accepted": bool(good)})
                if good:
                    break
        best = np.array([attempts[-1]["n_fins"], attempts[-1]["fin_height_mm"] / 1e3, attempts[-1]["fin_t_mm"] / 1e3])
        pb = net.predict(best[None])[0]
        checks = [
            {"name": "B-rep volume = closed form incl. fillets (all designs)", "value": vol_err, "limit": 1e-6,
             "passed": vol_err < 1e-6},
            {"name": "B-rep solids manifold (all designs)", "value": None, "limit": None,
             "passed": all(r["manifold"] for r in train + test)},
            *[{"name": f"held-out error: {o}", "value": float(e), "limit": 0.03, "passed": bool(e <= 0.03)}
              for o, e in zip(OUTS, err, strict=True)],
            *[{"name": f"trend: {k}", "value": float(f), "limit": 0.0, "passed": f == 0.0,
               "detail": "fraction of grid cells with the wrong sign"} for k, f in mono.items()],
            {"name": "first surrogate proposal survives verification", "value": attempts[0]["verified_t"],
             "limit": T_MAX, "passed": attempts[0]["accepted"],
             "detail": f"{len(attempts)} proposal(s) verified before one was accepted"},
            {"name": "accepted design meets T_source <= 75 C (verified)", "value": v["t_source"], "limit": T_MAX,
             "passed": v["t_source"] <= T_MAX},
            {"name": "accepted design meets dp <= 30 Pa (verified)", "value": v["dp"], "limit": DP_MAX, "passed": v["dp"] <= DP_MAX},
            {"name": "surrogate vs verified at the proposal (T rise)",
             "value": abs(pb[0] - (v["t_source"] - DUTY.t_ambient_c)) / (v["t_source"] - DUTY.t_ambient_c),
             "limit": 0.03, "passed": abs(pb[0] - (v["t_source"] - DUTY.t_ambient_c)) / (v["t_source"] - DUTY.t_ambient_c) <= 0.03}]
        core = [c for c in checks if not c["name"].startswith("first surrogate")]
        decision = "APPROVED" if all(c["passed"] for c in core) else "REVIEW"
        for c in checks:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("proposal_mass", v["mass"] * 1e3, "g", "exact B-rep mass of the verified design")
        ex.log_metric("proposal_t_source", v["t_source"], "C", "closed-form source temperature of the verified design")
        info.mark("NVIDIA USD BRep", ran=True, how=f"{len(train) + len(test) + 1} exact solids")
        info.mark("PhysicsNeMo FullyConnected", ran=True, how=f"{run.steps} steps on CPU")
        _figures(OUT, train, test, net, G, P, ok, best, v, checks, decision, attempts)
        summary = {"decision": decision, "heldout_rel_rms": dict(zip(OUTS, err.tolist(), strict=True)),
                   "trend_violations": mono, "proposal": {"n_fins": int(best[0]), "fin_height_mm": best[1] * 1e3,
                                                          "fin_t_mm": best[2] * 1e3, "gap_mm": v["gap"] * 1e3},
                   "proposal_surrogate": {"t_source": pb[0] + DUTY.t_ambient_c, "dp": pb[1], "mass": pb[2]},
                   "proposal_verified": {k: v[k] for k in ("t_source", "dp", "mass", "mass_closed_form", "fin_efficiency",
                                                           "warnings")},
                   "attempts": attempts, "fillet_mass_share_max": mass_cf_err, "train_designs": len(train), "train_steps": run.steps,
                   "train_wall_s": run.wall_s, "checks": checks}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        with open(os.path.join(OUT, "designs.json"), "w") as f:
            json.dump([{k: r[k] for k in ("n_fins", "fin_height", "fin_t", "t_source", "dp", "mass", "mass_closed_form")}
                       for r in train + test], f, indent=0, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "heldout_rel_rms", "proposal", "proposal_verified")},
                     indent=1, default=str))
    return summary


def _figures(OUT, train, test, net, G, P, ok, best, v, checks, decision, attempts):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    X, Y = table_xy(train)
    Xt, Yt = table_xy(test)
    Pt = net.predict(Xt)
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, (ax, lab) in enumerate(zip(axs, ("T rise [K]", "pressure drop [Pa]", "exact mass [kg]"), strict=True)):
        ax.plot(Yt[:, j], Pt[:, j], "o", ms=4, label="held-out designs")
        lo, hi = Yt[:, j].min(), Yt[:, j].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xlabel(f"closed form / B-rep: {lab}")
        ax.set_ylabel("PhysicsNeMo")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("PhysicsNeMo surrogate on 30 held-out designs", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "parity.png"), dpi=130)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    sc = ax.scatter(P[:, 2] * 1e3, P[:, 0] + DUTY.t_ambient_c, c=P[:, 1], s=2, cmap="viridis", alpha=0.5)
    fig.colorbar(sc, ax=ax, label="predicted dp [Pa]")
    ax.axhline(T_MAX, color="r", ls="--", lw=1, label="T_source limit")
    for a in attempts[:-1]:
        ax.plot(a["verified_mass"] * 1e3, a["verified_t"], "kx", ms=9)
    ax.plot([], [], "kx", label="proposals rejected by verification")
    ax.plot(v["mass"] * 1e3, v["t_source"], "r*", ms=14, label=f"accepted, verified ({int(best[0])} fins, "
            f"{best[1] * 1e3:.1f} x {best[2] * 1e3:.2f} mm)")
    ax.set_xlabel("mass [g]")
    ax.set_ylabel("source temperature [C]")
    ax.set_title("Design space through the surrogate (grid of valid designs)", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "design_space.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks, os.path.join(OUT, "checks.png"), title="Surrogate and design checks", decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
