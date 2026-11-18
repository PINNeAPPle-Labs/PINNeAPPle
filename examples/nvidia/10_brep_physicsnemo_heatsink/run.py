"""Heat-sink design from exact geometry: NVIDIA USD BRep builds the candidate, PINNeAPPle rates it two ways, PhysicsNeMo
learns the design space, PINNeAPPle checks the surrogate and verifies the design it proposes.

An extruded aluminium plate-fin heat sink, 100 x 100 mm, 1 mm fin-root fillets, with **two** chips on its base -- an
80 W power module on 40 x 40 mm and a 15 W gate driver on 15 x 15 mm -- in 3 m/s air with a partial shroud, so part
of the array sees the room and **radiates**. Each candidate is rated two ways at once:

* PINNeAPPle's closed-form network (Teertstra forced convection, fin efficiency, spreading resistance, pressure
  drop, and radiation as a second surface coefficient weighted by the share of area with a sky view);
* a 3-D finite-volume solve of the base with both footprints on it, which is the only one of the two that can say
  how much hotter the small chip runs than the large one.

The 3-D solve's hot spot is the design quantity; the network's is kept as the cross-check, and the gap between the
two models of the same physics is a check of its own. ``physicsnemo.models.mlp.FullyConnected`` learns hot spot,
pressure drop and mass from seven design variables (fin count, height, thickness, base thickness, material, the
correlation's ``h`` and the interface resistance), so the study spans the uncertainty an engineer lives with rather
than only the nominal geometry. PINNeAPPle checks the surrogate on held-out designs and against the trends the
physics requires, searches it for the lightest design meeting T_source <= 75 C and dp <= 30 Pa, and accepts a design
only after re-evaluating it with the exact solid and both physics models.

    PINNEAPPLE_BREP_PYTHON=/opt/brep/bin/python python examples/nvidia/10_brep_physicsnemo_heatsink/run.py   # ~2 min
    python examples/nvidia/10_brep_physicsnemo_heatsink/run.py --smoke    # seconds, no B-rep needed
"""
import argparse
import json
import os

import numpy as np

from pinneapple_analysis.cost.equal_budget import ComputeBudget
from pinneapple_lab import experiment
from pinneapple_tools.nvidia.brep import brep_available
from pinneapple_tools.nvidia.diagram import Block, architecture_diagram
from pinneapple_tools.nvidia.figures import checks_figure
from pinneapple_tools.nvidia.heatsink import (
    FULL_VARS,
    Duty,
    Source,
    checks,
    evaluate_full,
    sample_designs_full,
    table_rows,
)
from pinneapple_tools.nvidia.physicsnemo import MLPSurrogate, relative_errors, sweep_change
from pinneapple_tools.nvidia.runinfo import RunInfo

HERE = os.path.dirname(os.path.abspath(__file__))
DUTY = Duty(sources=(Source(0.030, 0.050, 0.040, 0.040, 80.0), Source(0.075, 0.050, 0.015, 0.015, 15.0)),
            t_ambient_c=25.0, air_velocity_m_s=3.0, base_t=0.005, material="al6063", root_fillet=0.001,
            emissivity=0.9, sky_fraction=0.25)
T_MAX, DP_MAX = 75.0, 30.0
OUTS = ("rise", "dp", "mass")
GRID_TABLE, GRID_VERIFY = 24, 48


def table_xy(rows):
    X = np.array([[r["n_fins"], r["fin_height"], r["fin_t"], r["base_t"], r["k_material"], r["h_scale"],
                   r["tim_k_m2_w"]] for r in rows], float)
    Y = np.array([[r["rise_k"], r["dp"], r["mass"]] for r in rows], float)
    return X, Y


def main(smoke=False, base=None):
    base = base or HERE
    OUT = os.path.join(base, "results")
    os.makedirs(OUT, exist_ok=True)
    info = RunInfo("10_brep_physicsnemo_heatsink")
    has_brep = brep_available()
    architecture_diagram(
        "10 Heat-sink design from exact geometry: PINNeAPPle + NVIDIA USD BRep + PhysicsNeMo",
        [Block("doe", "Design sampling (LHS): fins,\nheight, thickness, base, material", "pinneapple", 0, 0),
         Block("brep", "USD BRep: exact solid with\nroot fillets -> mass, area", "brep", 1, 0),
         Block("cf", "Closed-form network: Teertstra,\nfin efficiency, spreading, dp, radiation", "pinneapple", 1, 1),
         Block("fvm", "3-D finite volumes of the base:\nboth chips, per-chip hot spot", "pinneapple", 2, 1),
         Block("xchk", "Cross-check: network vs\n3-D solve on every design", "pinneapple", 3, 1),
         Block("nn", "physicsnemo FullyConnected:\n7 design variables -> rise, dp, mass", "physicsnemo", 2, 0),
         Block("chk", "Held-out error, physical\ntrends (monotonicity)", "pinneapple", 3, 0),
         Block("opt", "Search the surrogate: lightest\nwith T <= 75 C, dp <= 30 Pa", "physicsnemo", 4, 0),
         Block("ver", "Verify the proposal: B-rep +\nnetwork + 3-D solve", "pinneapple", 4, 1)],
        [("doe", "brep", "recipes"), ("doe", "cf", "designs"), ("cf", "fvm", "fin coefficient"),
         ("fvm", "xchk", "hot spot"), ("doe", "nn", "design variables"), ("cf", "nn", "rise, dp"), ("brep", "nn", "mass"),
         ("nn", "chk", "predictions"), ("nn", "opt", "surrogate"), ("opt", "ver", "candidate"),
         ("chk", "ver", "checks"), ("xchk", "ver", "model gap")],
        os.path.join(OUT, "architecture.png"),
        subtitle="The surrogate proposes; the exact geometry and two independent physics models decide")
    n_train, n_test = (16, 6) if smoke else (300, 40)
    steps = 300 if smoke else 15000
    with experiment("examples/nvidia/10_brep_physicsnemo_heatsink", title="Heat-sink design: USD BRep geometry, "
                    "PhysicsNeMo surrogate, PINNeAPPle network + 3-D verification", domain="heat transfer",
                    methods=["B-rep (SMLib)", "FullyConnected (PhysicsNeMo)", "closed-form heat sink", "FVM 3-D"],
                    out_dir=base, register=False,
                    modules=["pinneapple_tools.nvidia.heatsink", "pinneapple_design.thermal.heatsink.field",
                             "pinneapple_tools.nvidia.physicsnemo"],
                    params={"duty": {"sources": [s.__dict__ for s in DUTY.all_sources()],
                                     "air_velocity_m_s": DUTY.air_velocity_m_s, "t_ambient_c": DUTY.t_ambient_c,
                                     "emissivity": DUTY.emissivity, "sky_fraction": DUTY.sky_fraction,
                                     "root_fillet": DUTY.root_fillet},
                            "T_max_C": T_MAX, "dp_max_Pa": DP_MAX, "n_train": n_train, "grid_table": GRID_TABLE,
                            "grid_verify": GRID_VERIFY}) as ex:
        with info.step("designs: network + 3-D base solve"):
            d_tr = sample_designs_full(n_train, seed=0, duty=DUTY)
            d_te = sample_designs_full(n_test, seed=1, duty=DUTY)
            train = table_rows(d_tr, DUTY, grid=GRID_TABLE)
            test = table_rows(d_te, DUTY, grid=GRID_TABLE)
        X, Y = table_xy(train)
        Xt, Yt = table_xy(test)
        gap = np.array([abs(r["t_source_network"] - r["t_source_max_c"]) / r["rise_k"] for r in train + test])
        fvm_err = max(r["fvm_energy_error"] for r in train + test)

        net = MLPSurrogate(len(FULL_VARS), len(OUTS), layer_size=64, num_layers=3)
        with info.step("train (PhysicsNeMo FullyConnected)"):
            run = net.fit(X, Y, ComputeBudget(steps=steps), batch_size=64, lr=3e-3)
        err = relative_errors(net.predict(Xt), Yt)
        nominal = np.array([14.0, 0.032, 0.0015, 0.005, 209.0, 1.0, 1e-5])
        lo = np.array([6.0, 0.012, 0.0008, 0.003, 167.0, 0.85, 5e-6])
        hi = np.array([26.0, 0.050, 0.0030, 0.010, 388.0, 1.15, 2e-5])

        def sweep(i, out, expected):
            a, b = nominal.copy(), nominal.copy()
            a[i], b[i] = lo[i], hi[i]
            return sweep_change(net, a, b, out, expected)

        mono = {"hot spot falls with fin height": sweep(1, 0, -1),
                "hot spot falls with fin thickness": sweep(2, 0, -1),
                "hot spot falls with the film coefficient": sweep(5, 0, -1),
                "hot spot falls with base thickness": sweep(3, 0, -1),
                "hot spot falls with the fin count": sweep(0, 0, -1),
                "hot spot grows with the interface resistance": sweep(6, 0, +1),
                "mass grows with fin count": sweep(0, 2, +1),
                "mass grows with fin height": sweep(1, 2, +1),
                "mass grows with base thickness": sweep(3, 2, +1),
                "dp grows with fin count": sweep(0, 1, +1)}

        # search the surrogate on a fine grid of valid designs
        rng = np.random.default_rng(7)
        N = 40000 if not smoke else 400
        G = np.column_stack([rng.integers(6, 27, N), rng.uniform(0.012, 0.050, N), rng.uniform(0.0008, 0.0030, N),
                             rng.uniform(0.003, 0.010, N), rng.choice([167.0, 209.0, 229.0, 388.0], N),
                             rng.uniform(0.85, 1.15, N), rng.uniform(5e-6, 2e-5, N)])
        G = G[(0.1 - G[:, 0] * G[:, 2]) / (G[:, 0] - 1) >= 0.002]
        P = net.predict(G)
        ok = (P[:, 0] + DUTY.t_ambient_c <= T_MAX) & (P[:, 1] <= DP_MAX)
        order = np.argsort(np.where(ok, P[:, 2], np.inf))
        if not ok.any():
            order = np.argsort(P[:, 0])            # nothing is predicted feasible: verify the coolest, and say so
        order = order[:8]
        attempts = []
        with info.step("verify proposals (network + 3-D solve + USD BRep), lightest first"):
            for i in order[:8]:
                d = {"n_fins": int(G[i, 0]), "fin_height": float(G[i, 1]), "fin_t": float(G[i, 2]),
                     "base_t": float(G[i, 3]), "k_material": float(G[i, 4]), "h_scale": float(G[i, 5]),
                     "tim_k_m2_w": float(G[i, 6])}
                v = evaluate_full(d, DUTY, brep=has_brep, grid=GRID_VERIFY)
                good = v["t_source_max_c"] <= T_MAX and v["dp"] <= DP_MAX
                attempts.append({"design": {k: d[k] for k in ("n_fins", "fin_height", "fin_t", "base_t")},
                                 "material": v["material"],
                                 "predicted_t": float(P[i, 0] + DUTY.t_ambient_c),
                                 "verified_t": v["t_source_max_c"], "network_t": v["t_source_network"],
                                 "verified_dp": v["dp"], "verified_mass": v["mass"], "accepted": bool(good)})
                if good:
                    break
        best = d
        pb = net.predict(np.array([[best["n_fins"], best["fin_height"], best["fin_t"], best["base_t"],
                                    best["k_material"], best["h_scale"], best["tim_k_m2_w"]]]))[0]
        eng = checks(v, t_limit_c=T_MAX, dp_limit_pa=DP_MAX)
        info.mark("NVIDIA USD BRep", ran=has_brep,
                  how=f"{len(train) + len(test) + 1} exact solids" if has_brep else "",
                  why="" if has_brep else "usd-brep is not installed here (CPython 3.12 wheel): the mass is the "
                                          "closed form's, set PINNEAPPLE_BREP_PYTHON to run it")
        info.mark("3-D base finite volumes", ran=True, how=f"{len(train) + len(test) + 1} solves at "
                                                          f"{GRID_TABLE} and {GRID_VERIFY} in-plane cells")
        info.mark("PhysicsNeMo FullyConnected", ran=True, how=f"{run.steps} steps on CPU")
        checks_all = [
            {"name": "3-D base solve closes its energy balance (all designs)", "value": fvm_err, "limit": 1e-6,
             "passed": fvm_err < 1e-6},
            *[{"name": f"held-out error: {o}", "value": float(e), "limit": 0.05, "passed": bool(e <= 0.05)}
              for o, e in zip(OUTS, err, strict=True)],
            *[{"name": f"trend: {k}", "value": float(f), "limit": 0.0, "passed": f <= 0.0,
               "detail": "-expected x (value at the top of the range - value at the bottom): positive means the model "
                         "has the direction backwards"} for k, f in mono.items()],
            {"name": "network and 3-D solve agree on every design", "value": float(np.percentile(gap, 90)),
             "limit": 0.20, "passed": bool(np.percentile(gap, 90) <= 0.20),
             "detail": "p90 of |T_network - T_3D| / rise: two independent models of the same physics"},
            {"name": "accepted design meets T_source <= 75 C (verified 3-D)", "value": v["t_source_max_c"],
             "limit": T_MAX, "passed": v["t_source_max_c"] <= T_MAX},
            {"name": "accepted design meets dp <= 30 Pa (verified)", "value": v["dp"], "limit": DP_MAX,
             "passed": v["dp"] <= DP_MAX},
            {"name": "surrogate vs verified at the proposal (rise)",
             "value": abs(pb[0] - v["rise_k"]) / v["rise_k"], "limit": 0.05,
             "passed": bool(abs(pb[0] - v["rise_k"]) / v["rise_k"] <= 0.05)},
            *[{"name": f"design: {c['name']}", "value": c["value"], "limit": c["limit"],
               "passed": c["passed"], "detail": c["detail"]} for c in eng]]
        decision = "APPROVED" if all(c["passed"] for c in checks_all) else "REVIEW"
        for c in checks_all:
            ex.check(c["name"], c["passed"], value=c["value"], max=c["limit"], detail=c.get("detail", ""))
        ex.log_metric("proposal_mass", v["mass"] * 1e3, "g", "mass of the verified design")
        ex.log_metric("proposal_t_source", v["t_source_max_c"], "C", "3-D hot spot of the verified design")
        ex.log_metric("network_vs_3d_gap_p90", float(np.percentile(gap, 90)) * 100, "%",
                      "p90 gap between the closed-form network and the 3-D base solve")
        _figures(OUT, train, test, net, G, P, ok, best, v, checks_all, decision, attempts, gap)
        summary = {"decision": decision, "heldout_rel_rms": dict(zip(OUTS, err.tolist(), strict=True)),
                   "trend_violations": mono, "network_vs_3d_gap": {"median": float(np.median(gap)),
                                                                   "p90": float(np.percentile(gap, 90)),
                                                                   "max": float(gap.max())},
                   "proposal": {**{k: best[k] for k in ("n_fins", "fin_height", "fin_t", "base_t")},
                                "material": v["material"], "gap_mm": v["gap"] * 1e3},
                   "proposal_surrogate": {"t_source": float(pb[0] + DUTY.t_ambient_c), "dp": float(pb[1]),
                                          "mass": float(pb[2])},
                   "proposal_verified": {"t_source_3d": v["t_source_max_c"], "t_chip": v["t_chip_c"],
                                         "t_source_network": v["t_source_network"], "dp": v["dp"], "mass": v["mass"],
                                         "fin_efficiency": v["fin_efficiency"], "radiation_share": v["radiation_share"],
                                         "h_w_m2k": v["h_w_m2k"], "warnings": v["warnings"]},
                   "attempts": attempts, "brep_ran": has_brep, "train_designs": len(train), "train_steps": run.steps,
                   "train_wall_s": run.wall_s, "checks": checks_all}
        with open(os.path.join(OUT, "summary.json"), "w") as f:
            json.dump(summary, f, indent=1, default=float)
        with open(os.path.join(OUT, "designs.json"), "w") as f:
            json.dump([{k: r[k] for k in ("n_fins", "fin_height", "fin_t", "base_t", "material", "h_scale",
                                          "tim_k_m2_w", "t_source_max_c", "t_source_network", "dp", "mass")}
                       for r in train + test], f, indent=0, default=float)
        info.save(os.path.join(OUT, "run_info.json"))
    print(json.dumps({k: summary[k] for k in ("decision", "heldout_rel_rms", "network_vs_3d_gap", "proposal",
                                              "proposal_verified")}, indent=1, default=str)[:1500])
    return summary


def _figures(OUT, train, test, net, G, P, ok, best, v, checks_all, decision, attempts, gap):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Pt = net.predict(table_xy(test)[0])
    Yt = table_xy(test)[1]
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.8))
    for j, (ax, lab) in enumerate(zip(axs, ("rise [K]", "pressure drop [Pa]", "mass [kg]"), strict=True)):
        ax.plot(Yt[:, j], Pt[:, j], "o", ms=4, label="held-out designs")
        lo, hi = Yt[:, j].min(), Yt[:, j].max()
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
        ax.set_xlabel(f"network + 3-D solve: {lab}")
        ax.set_ylabel("PhysicsNeMo")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)
    fig.suptitle("PhysicsNeMo surrogate on the held-out designs", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "parity.png"), dpi=130)
    plt.close(fig)

    # the two physics models against each other: the cross-check the example is about
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    x = np.array([r["t_source_network"] for r in train + test])
    y = np.array([r["t_source_max_c"] for r in train + test])
    ax.plot(x, y, "o", ms=4, label="designs")
    lo, hi = min(x.min(), y.min()), max(x.max(), y.max())
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="y = x")
    ax.set_xlabel("closed-form network [C]")
    ax.set_ylabel("3-D base solve, hottest chip [C]")
    ax.set_title(f"The same physics two ways: p90 gap {100 * np.percentile(gap, 90):.1f} %", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "models.png"), dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    sc = ax.scatter(P[:, 2] * 1e3, P[:, 0] + DUTY.t_ambient_c, c=P[:, 1], s=2, cmap="viridis", alpha=0.4)
    fig.colorbar(sc, ax=ax, label="predicted dp [Pa]")
    ax.axhline(T_MAX, color="r", ls="--", lw=1, label="T_source limit")
    for a in attempts[:-1]:
        ax.plot(a["verified_mass"] * 1e3, a["verified_t"], "kx", ms=9)
    ax.plot([], [], "kx", label="proposals rejected by verification")
    ax.plot(v["mass"] * 1e3, v["t_source_max_c"], "r*", ms=15,
            label=f"accepted, verified ({best['n_fins']} fins, {best['fin_height'] * 1e3:.1f} x "
                  f"{best['fin_t'] * 1e3:.2f} mm, {v['material']})")
    ax.set_xlabel("mass [g]")
    ax.set_ylabel("source temperature [C]")
    ax.set_title("Design space through the surrogate", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "design_space.png"), dpi=130)
    plt.close(fig)
    checks_figure(checks_all, os.path.join(OUT, "checks.png"), title="Surrogate, models and design checks",
                  decision=decision)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    main(ap.parse_args().smoke)
