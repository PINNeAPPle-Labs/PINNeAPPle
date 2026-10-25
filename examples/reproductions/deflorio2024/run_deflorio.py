"""Reproduce De Florio, Schiassi, Calabrò, Furfaro, J. Comput. Appl. Math. 436 (2024) 115396 with the protocol
frozen in TARGETS.md (PIELM and X-TFC at full scale; PINN and Deep-TFC at a declared reduced scale).

    python examples/reproductions/deflorio2024/run_deflorio.py --part elm [--nu 0.01]
    python examples/reproductions/deflorio2024/run_deflorio.py --part dnn [--nu 1] [--method pinn]
    python examples/reproductions/deflorio2024/run_deflorio.py --part baselines
    python examples/reproductions/deflorio2024/run_deflorio.py --part report
    ... --smoke                      # tiny sizes, outputs suffixed _smoke

Every run is written to runs/<part>/<method>_nu<nu>_s<seed>.json and skipped when it exists (resumable).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import pinneapple_lab  # noqa: E402
from pinneapple_analysis.reproduction import seed_stats, status  # noqa: E402
from pinneapple_analysis.validation.pinn import PINNProblem, d, mlp, train_pinn  # noqa: E402
from pinneapple_simulation.numerical_solvers import gated_xtfc as gx  # noqa: E402
from pinneapple_simulation.numerical_solvers.xtfc_general import (  # noqa: E402
    ConstrainedExpression1D,
    Constraint,
    XTFCProblem,
    sample_calabro,
    solve_xtfc,
)

HERE = os.path.dirname(os.path.abspath(__file__))
NUS = (1.0, 0.1, 0.01, 0.001, 0.0001)
ELM_NEURONS = {1.0: 100, 0.1: 100, 0.01: 1000, 0.001: 10_000, 0.0001: 10_000}       # Tables 2-6 superscripts
ELM_SEEDS = {1.0: 10, 0.1: 10, 0.01: 10, 0.001: 3, 0.0001: 3}                        # TARGETS.md item 6
DNN_WIDTH = {1.0: 20, 0.1: 20, 0.01: 50, 0.001: 50, 0.0001: 50}                     # item 7 (reduced)
X_TRAIN = (np.arange(1, 10_001) / 10_001)[:, None]                                  # item 3
X_TEST = np.linspace(0.0, 1.0, 20_001)                                               # item 4
# Tables 2-6, PIELM / X-TFC: (max, mean, eps_G)
TARGET = {
    ("pielm", 1.0): (6.11e-15, 2.10e-15, 2.99e-15), ("xtfc", 1.0): (3.89e-16, 1.17e-16, 1.57e-16),
    ("pielm", 0.1): (3.18e-13, 1.64e-13, 1.90e-13), ("xtfc", 0.1): (1.49e-13, 5.93e-14, 7.31e-14),
    ("pielm", 0.01): (2.18e-08, 2.18e-08, 2.16e-08), ("xtfc", 0.01): (1.73e-12, 9.45e-14, 6.70e-13),
    ("pielm", 0.001): (4.89e-05, 4.89e-05, 4.88e-05), ("xtfc", 0.001): (2.83e-12, 7.09e-13, 1.18e-12),
    ("pielm", 0.0001): (5.0e-01, 5.0e-01, 5.0e-01), ("xtfc", 0.0001): (2.89e-03, 1.39e-03, 1.74e-03),
}
# PINN_A, PINN_L, DeepTFC_A, DeepTFC_L max|u* - u|
TARGET_DNN = {1.0: (3.15e-05, 9.96934e-08, 1.04022e-06, 1.619e-07), 0.1: (5.7474e-05, 1.47079e-05, 1.03e-04, 1.10e-04),
              0.01: (5.00e-01, 5.00e-01, 2.59e-02, 6.18332e-05), 0.001: (5.00e-01, 5.00e-01, 9.58e-01, 9.58e-01),
              0.0001: (5.00e-01, 5.00e-01, 9.58e-01, 9.58e-01)}


def exact(x, nu):
    pe = 1.0 / nu
    return -np.expm1(pe * (np.asarray(x) - 1.0)) / -np.expm1(-pe)


def errors(pred, nu, resid_sq_mean=None, bc=None):
    e = np.abs(np.asarray(pred).ravel() - exact(X_TEST, nu))
    out = {"max": float(e.max()), "mean": float(e.mean()), "std": float(e.std()),
           "eps_G": float(np.sqrt(np.trapezoid(e * e, X_TEST))), "e_x0": float(e[0]), "e_xf": float(e[-1])}
    if resid_sq_mean is not None:
        out["eps_T"] = float(np.sqrt(resid_sq_mean + (sum(b * b for b in bc) if bc else 0.0)))
    return out


def _path(part, name, smoke):
    folder = os.path.join(HERE, "runs" + ("_smoke" if smoke else ""), part)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, name + ".json")


def problem(nu):
    pe = 1.0 / nu
    return XTFCProblem([(0.0, 1.0)], {"u": [Constraint(0, 0.0), Constraint(0, 1.0)]}, {"u": lambda X: 1 - X[:, 0]},
                       lambda X, D: D[("u", (2,))] - pe * D[("u", (1,))], [("u", (1,)), ("u", (2,))], linear=True)


def _chunked(fn, x, size=4000):
    return np.concatenate([fn(x[i:i + size]) for i in range(0, len(x), size)])


def part_elm(nus, smoke, methods=("pielm", "xtfc")):
    for nu in nus:
        n = 50 if smoke else ELM_NEURONS[nu]
        Xtr = X_TRAIN[::100] if smoke else X_TRAIN
        for s in range(2 if smoke else ELM_SEEDS[nu]):
            for m in methods:
                path = _path("elm", f"{m}_nu{nu:g}_s{s}", smoke)
                if os.path.exists(path):
                    continue
                w = sample_calabro(n, np.random.default_rng(s))
                t0 = time.perf_counter()
                r = solve_xtfc(problem(nu), Xtr, n_neurons=n, activation="gaussian", weights=w,
                               constraints="expression" if m == "xtfc" else "rows",
                               face_points=np.array([[0.0], [1.0]]), lstsq_driver="gelsy")
                t = time.perf_counter() - t0
                pred = _chunked(lambda x, r=r: r.predict(x[:, None]), X_TEST)
                res = _chunked(lambda x, r=r: r.residual(x)[0], Xtr)
                bc = [float(r.predict([[0.0]])[0] - 1.0), float(r.predict([[1.0]])[0])] if m == "pielm" else None
                rec = {"method": m, "nu": nu, "seed": s, "neurons": n, "solve_s": t, "lstsq_s": r.lstsq_seconds,
                       **errors(pred, nu, float(np.mean(res * res)), bc)}
                json.dump(rec, open(path, "w"), indent=1)
                print(m, nu, s, f"max {rec['max']:.3g} mean {rec['mean']:.3g} epsT {rec['eps_T']:.3g} {t:.1f}s",
                      flush=True)


class _AdamSnapshot:
    """Monitor for train_pinn: records the test prediction at the end of the Adam phase."""
    stop = False

    def __init__(self, adam_steps):
        self.adam_steps, self.pred = adam_steps, None

    def __call__(self, net, step):
        if step == self.adam_steps:
            with torch.no_grad():
                self.pred = net(torch.tensor(X_TEST[:, None], dtype=torch.float32)).numpy().ravel()


def part_dnn(nus, smoke, methods=("pinn", "deeptfc")):
    adam, lbfgs = (20, 5) if smoke else (2000, 500)
    for nu in nus:
        pe = 1.0 / nu
        for s in range(1 if smoke else 3):
            for m in methods:
                path = _path("dnn", f"{m}_nu{nu:g}_s{s}", smoke)
                if os.path.exists(path):
                    continue
                prob = PINNProblem(
                    f"advection_diffusion_nu{nu:g}", [(0.0, 1.0)],
                    lambda u, X, pe=pe: d(u, X, 0, 2) - pe * d(u, X, 0, 1),
                    boundary=(lambda n, rng: (np.array([[0.0], [1.0]]), np.array([[1.0], [0.0]])))
                    if m == "pinn" else None)
                torch.manual_seed(s)
                net = mlp(1, 1, DNN_WIDTH[nu], 10, activation="gaussian")
                if m == "deeptfc":
                    net = ConstrainedExpression1D(net, 0.0, 1.0, 1.0, 0.0)
                snap = _AdamSnapshot(adam)
                t0 = time.perf_counter()
                net, info = train_pinn(prob, net, n_collocation=200 if smoke else 10_000, adam_steps=adam,
                                       lbfgs_steps=lbfgs, lr=1e-3, weights={"boundary": 1.0}, monitor=snap, seed=s)
                t = time.perf_counter() - t0
                with torch.no_grad():
                    pred = net(torch.tensor(X_TEST[:, None], dtype=torch.float32)).numpy().ravel()
                rec = {"method": m, "nu": nu, "seed": s, "width": DNN_WIDTH[nu], "depth": 10, "adam": adam,
                       "lbfgs": lbfgs, "train_s": t, "final_loss": info["loss"], "A": errors(snap.pred, nu),
                       "L": errors(pred, nu)}
                json.dump(rec, open(path, "w"), indent=1)
                print(m, nu, s, f"A max {rec['A']['max']:.3g} L max {rec['L']['max']:.3g} {t:.0f}s", flush=True)


def part_baselines(nus, smoke):
    """Same points / neurons / solver: tanh X-TFC with unif(-1, 1) weights; Gated X-TFC with the same number of
    basis functions (nested gate search, TARGETS.md of arXiv:2510.01039)."""
    x_val = (np.arange(800) + 0.5) / 800
    for nu in nus:
        n = 50 if smoke else ELM_NEURONS[nu]
        Xtr = X_TRAIN[::100] if smoke else X_TRAIN
        for s in range(2 if smoke else ELM_SEEDS[nu]):
            path = _path("baselines", f"xtfc_tanh_nu{nu:g}_s{s}", smoke)
            if not os.path.exists(path):
                r = solve_xtfc(problem(nu), Xtr, n_neurons=n, activation="tanh", seed=s, lstsq_driver="gelsy")
                rec = {"method": "xtfc_tanh_unif(-1,1)", "nu": nu, "seed": s,
                       **errors(_chunked(lambda x, r=r: r.predict(x[:, None]), X_TEST), nu)}
                json.dump(rec, open(path, "w"), indent=1)
                print("xtfc_tanh", nu, s, f"max {rec['max']:.3g}", flush=True)
        path = _path("baselines", f"gated_nu{nu:g}", smoke)
        if not os.path.exists(path) and (smoke or n <= 2000):
            p = gx.LinearBVP1D(gx._const(1.0), gx._const(-1.0 / nu), gx._const(0.0), gx._const(0.0), 1.0, 0.0, nu,
                               lambda x, nu=nu: exact(x, nu))
            r = gx.search_gates_nested(p, x_val, n_centers=n // 2, xs_bounds=(0.5, 0.999),
                                       xs_tol=1e-2 if smoke else 1e-4, eps_tol=10.0 if smoke else 1e-1)
            rec = {"method": "gated_xtfc", "nu": nu, "basis": n, "xs": r["xs"], "eps_scale": r["eps_scale"],
                   "seconds": r["seconds"], **errors(r["sol"].predict(X_TEST), nu)}
            json.dump(rec, open(path, "w"), indent=1)
            print("gated", nu, f"max {rec['max']:.3g}", flush=True)


def _load(part):
    return [json.load(open(f)) for f in sorted(glob.glob(os.path.join(HERE, "runs", part, "*.json")))]


def report():
    elm, dnn, base = _load("elm"), _load("dnn"), _load("baselines")
    rows = ["| ν | Method | Neurons | Seeds | max: paper / ours median [min, max] | mean: paper / ours | "
            "ε_G: paper / ours | ε_T ours | solve | Status (max) |", "|" + "---|" * 10]
    statuses = {}
    meds = {}
    for nu in NUS:
        for m in ("pielm", "xtfc"):
            rs = [r for r in elm if r["method"] == m and r["nu"] == nu]
            if not rs:
                continue
            st = {k: seed_stats([r[k] for r in rs]) for k in ("max", "mean", "eps_G", "eps_T", "solve_s")}
            meds[(m, nu)] = st["max"].median
        for m in ("pielm", "xtfc"):
            rs = [r for r in elm if r["method"] == m and r["nu"] == nu]
            if not rs:
                continue
            st = {k: seed_stats([r[k] for r in rs]) for k in ("max", "mean", "eps_G", "eps_T", "solve_s")}
            pmax, pmean, peg = TARGET[(m, nu)]
            ordering = None
            if ("xtfc", nu) in meds and ("pielm", nu) in meds:
                ordering = meds[("xtfc", nu)] <= meds[("pielm", nu)]
            stat = status(st["max"].median, pmax, ordering_holds=ordering)
            statuses[f"{m} nu={nu:g}"] = stat
            rows.append(f"| {nu:g} | {'X-TFC' if m == 'xtfc' else 'PIELM'} | {rs[0]['neurons']} | {len(rs)} | "
                        f"{pmax:.3g} / {st['max'].median:.3g} [{st['max'].min:.2g}, {st['max'].max:.2g}] | "
                        f"{pmean:.3g} / {st['mean'].median:.3g} | {peg:.3g} / {st['eps_G'].median:.3g} | "
                        f"{st['eps_T'].median:.3g} | {st['solve_s'].median:.1f} s | **{stat}** |")
    drows = ["| ν | Method | width x depth | Seeds | max after Adam: paper / ours | max after L-BFGS: paper / ours | "
             "learned? paper / ours | train time |", "|" + "---|" * 8]
    for nu in NUS:
        for j, m in enumerate(("pinn", "deeptfc")):
            rs = [r for r in dnn if r["method"] == m and r["nu"] == nu]
            if not rs:
                continue
            a = seed_stats([r["A"]["max"] for r in rs])
            lb = seed_stats([r["L"]["max"] for r in rs])
            pa, pl = TARGET_DNN[nu][2 * j], TARGET_DNN[nu][2 * j + 1]
            drows.append(f"| {nu:g} | {'PINN' if m == 'pinn' else 'Deep-TFC'} | {rs[0]['width']} x 10 | {len(rs)} | "
                         f"{pa:.3g} / {a.median:.3g} | {pl:.3g} / {lb.median:.3g} | "
                         f"{'yes' if pl < 1e-2 else 'no'} / {'yes' if lb.median < 1e-2 else 'no'} | "
                         f"{np.median([r['train_s'] for r in rs]):.0f} s | reduced-scale |")
    brows = ["| ν | Method | max abs error (median) | Seeds |", "|---|---|---|---|"]
    for nu in NUS:
        for m, label in (("xtfc", "X-TFC, Gaussian, [9] weights (this reproduction)"),
                         ("xtfc_tanh_unif(-1,1)", "X-TFC, tanh, unif(-1, 1) weights"),
                         ("gated_xtfc", "Gated X-TFC (same basis count, gate search)")):
            src = elm if m == "xtfc" else base
            rs = [r for r in src if r["method"] == m and r["nu"] == nu]
            if rs:
                brows.append(f"| {nu:g} | {label} | {np.median([r['max'] for r in rs]):.3g} | {len(rs)} |")
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for m, mk in (("pielm", "s"), ("xtfc", "o")):
        xs = [nu for nu in NUS if (m, nu) in meds]
        ax.loglog(xs, [meds[(m, nu)] for nu in xs], mk + "-", label=f"{m} ours (median)")
        ax.loglog(NUS, [TARGET[(m, nu)][0] for nu in NUS], mk + "--", mfc="none", label=f"{m} paper")
    ax.set_xlabel("ν")
    ax.set_ylabel("max |u* - u|")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "max_error_vs_nu.png"), dpi=120)
    plt.close(fig)
    with open(os.path.join(HERE, "RESULTS.md"), "w") as fh:
        fh.write("# De Florio et al. 2024 reproduction: measured results\n\nGenerated by `run_deflorio.py --part "
                 f"report` from runs/*. Machine: {platform.platform()}, Python {platform.python_version()}.\n\n"
                 "## PIELM and X-TFC (full scale)\n\n" + "\n".join(rows)
                 + "\n\n## PINN and Deep-TFC (reduced scale, TARGETS.md item 7 and Amendment 1)\n\n"
                 + "\n".join(drows) + "\n\n## Baselines at equal budget\n\n" + "\n".join(brows)
                 + "\n\n![max error vs nu](max_error_vs_nu.png)\n")
    with pinneapple_lab.experiment(
            "examples/reproductions/deflorio2024", title="De Florio et al. 2024 reproduction",
            description="PINN, Deep-TFC, PIELM and X-TFC on the 1D steady advection-diffusion BVP with sharp "
                        "gradients (J. Comput. Appl. Math. 436, 115396) against the frozen targets of TARGETS.md.",
            domain="physics-informed ML", methods=["X-TFC", "PIELM", "PINN", "Deep-TFC"],
            modules=["pinneapple_simulation.numerical_solvers.xtfc_general", "pinneapple_analysis.reproduction",
                     "pinneapple_analysis.validation.pinn"], out_dir=HERE, register=False,
            links={"targets": "examples/reproductions/deflorio2024/TARGETS.md"}) as ex:
        for name, st in statuses.items():
            ex.check(f"{name}: {st}", st == "reproduced", kind="reference", label="reproduction")
        for r in elm:
            if r["method"] == "xtfc" and r["seed"] == 0:
                ex.check(f"X-TFC nu={r['nu']:g}: boundary values exact ({max(r['e_x0'], r['e_xf']):.1g})",
                         max(r["e_x0"], r["e_xf"]) < 1e-12, kind="physics")
                ex.log_metric(f"xtfc_max_nu_{r['nu']:g}_seed0", r["max"], "abs error", "max |u - exact|, 20001 pts")
    print("\n".join(rows + [""] + drows + [""] + brows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", required=True, choices=["elm", "dnn", "baselines", "report"])
    ap.add_argument("--nu", type=float, nargs="*")
    ap.add_argument("--method", nargs="*")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    nus = tuple(a.nu) if a.nu else NUS
    if a.part == "report":
        report()
    elif a.part == "baselines":
        part_baselines(nus, a.smoke)
    else:
        kw = {"methods": tuple(a.method)} if a.method else {}
        globals()[f"part_{a.part}"](nus, a.smoke, **kw)


if __name__ == "__main__":
    main()
