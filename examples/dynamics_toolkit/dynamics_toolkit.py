"""The dynamics toolkit end to end: five studies, each set against an independent reference.

1. first_passage  pp.solve trains a PINN on the absorbing Fokker-Planck (heat) equation u_t = D u_xx on (0, 1);
                  the mass of its density is the survival function of Brownian motion leaving the interval. Set
                  against the closed form exp(-D pi^2 t) and a Monte Carlo of the SDE (Kaplan-Meier).
2. koopman        the spectrum of a PINN: DMD of the PINN's snapshots of u_t = D u_xx with three Fourier modes,
                  against the exact decay rates -D k^2 pi^2, and DMD of the exact snapshots as the control.
3. fisher         when to read a thermocouple to estimate the diffusivity: D-optimal times against equally spaced
                  ones, the Cramer-Rao bound checked by refitting simulated data.
4. dimensional    Buckingham pi for a cooling slab, then the collapse of centre temperatures of six slabs
                  (different L and D) on t D / L^2, against a permutation control.
5. early_warning  critical slowing down before a saddle-node, with two controls: a stable system and a stable
                  system whose noise grows (a known false positive of variance).

Run: ``python examples/dynamics_toolkit/dynamics_toolkit.py [--case first_passage ...] [--epochs 2000]``.
It writes ``figures/<case>.png`` and ``experiment.json`` here. Every number in the manifest is measured by this run.
"""
from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import pinneapple as pp  # noqa: E402
import pinneapple_lab  # noqa: E402
from pinneapple_analysis import dynamics as dyn  # noqa: E402
from pinneapple_analysis.dynamics import early_warning as ew  # noqa: E402
from pinneapple_analysis.dynamics.first_passage import survival_from_solution  # noqa: E402
from pinneapple_analysis.dynamics.koopman import DMD  # noqa: E402
from pinneapple_analysis.inverse_problems.forward import (  # noqa: E402
    CallableModel,
    Observations,
    Parameter,
)
from pinneapple_physics import ProblemBuilder  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
CASES = ("first_passage", "koopman", "fisher", "dimensional", "early_warning")


def heat_problem(name: str, D: float, t_end: float, ic):
    return (ProblemBuilder(name).domain(x=(0.0, 1.0), t=(0.0, t_end)).fields("u").pde("heat_equation", alpha=D)
            .ic(field="u", fn=ic).bc(kind="dirichlet", field="u", value=0.0, on="x_boundary").build())


def first_passage(ex, epochs: int, seed: int) -> None:
    D, t_end = 0.1, 2.0
    spec = heat_problem("brownian_exit_density", D, t_end, lambda X: 0.5 * np.pi * np.sin(np.pi * X[:, 0:1]))
    sol = pp.solve(spec, method="pinn", epochs=epochs, seed=seed)
    x, t = np.linspace(0.0, 1.0, 201), np.linspace(0.0, t_end, 41)
    fp = survival_from_solution(sol, x, t)
    S_exact = np.exp(-D * np.pi ** 2 * t)
    rmean_exact = (1 - np.exp(-D * np.pi ** 2 * t_end)) / (D * np.pi ** 2)

    rng = np.random.default_rng(seed)
    x0 = np.arccos(1 - 2 * rng.uniform(size=20_000)) / np.pi      # inverse CDF of (pi/2) sin(pi x)
    mc = dyn.FirstPassageAnalyzer(dyn.BoundaryCondition.interval(0.0, 1.0, name="exit")).monte_carlo(
        dyn.SDE.brownian(0.0, np.sqrt(2 * D)), x0[:, None], t_max=t_end, dt=1e-3, n_paths=x0.size, seed=seed)
    dist = mc.distributions["exit"]
    S_mc = dist.survival(t)

    err_pinn = float(np.max(np.abs(fp.estimate["S"] - S_exact)))
    err_mc = float(np.max(np.abs(S_mc - S_exact)))
    m_mc, se_mc = mc.estimate["exit"]["mean"], mc.uncertainty["exit"]["mean_se"]
    ex.log_metric("first_passage_pinn_max_abs_error_S", err_pinn, "-",
                  "max over 41 times of |S_PINN - exp(-D pi^2 t)|, S_PINN = trapezoid of the PINN density on 201 x")
    ex.log_metric("first_passage_mc_max_abs_error_S", err_mc, "-",
                  "max over the same times of |Kaplan-Meier S - exp(-D pi^2 t)|, 20 000 Euler-Maruyama paths, "
                  "dt 1e-3, Brownian bridge")
    ex.log_metric("first_passage_restricted_mean_pinn", fp.estimate["restricted_mean"], "time",
                  f"integral of S_PINN over [0, {t_end}] (exact {rmean_exact:.6f})")
    ex.log_metric("first_passage_restricted_mean_mc", m_mc, "time",
                  f"Kaplan-Meier restricted mean up to {t_end}, standard error {se_mc:.2e}")
    ex.log_metric("first_passage_pinn_wall_time", sol.metadata()["wall_time_s"], "s", "pp.solve wall time")
    ex.check("PINN survival matches the closed form", value=err_pinn, max=0.01, kind="reference",
             detail="max |S_PINN - exp(-D pi^2 t)| <= 0.01")
    ex.check("Monte Carlo survival matches the closed form", value=err_mc, max=0.01, kind="reference",
             detail="max |S_MC - exp(-D pi^2 t)| <= 0.01 (20 000 paths: pointwise sd <= 0.0035)")
    ex.check("PINN restricted mean within 3 Monte Carlo standard errors", value=abs(fp.estimate["restricted_mean"]
             - m_mc), max=3 * se_mc, kind="verification")
    ex.check("Fokker-Planck diagnostics are ok", all(d.status in ("ok", "info") for d in fp.diagnostics),
             kind="sanity", detail="; ".join(f"{d.name}={d.value:.3g} {d.status}" for d in fp.diagnostics))

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ax.plot(t, S_exact, "k-", lw=2, label="exact exp(-D pi^2 t)")
    ax.plot(t, fp.estimate["S"], "r--", label="PINN density (pp.solve), integrated")
    sf = dist.survival_function()
    ax.step(sf.t, sf.S, "b:", where="post", label="Monte Carlo, Kaplan-Meier")
    ax.set(xlabel="t", ylabel="S(t) = P(tau > t)", title="Exit of Brownian motion from (0, 1)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "first_passage.png"), dpi=110)
    plt.close(fig)
    print(f"first_passage: max|dS| PINN {err_pinn:.2e}, MC {err_mc:.2e}; restricted mean exact {rmean_exact:.4f}, "
          f"PINN {fp.estimate['restricted_mean']:.4f}, MC {m_mc:.4f} +- {se_mc:.4f}")


def koopman(ex, epochs: int, seed: int) -> None:
    D, t_end, amps = 0.05, 1.0, (1.0, 0.5, 0.25)

    def exact(x, t):
        return sum(a * np.exp(-D * (k * np.pi) ** 2 * t)[:, None] * np.sin(k * np.pi * x)[None, :]
                   for k, a in enumerate(amps, start=1))

    spec = heat_problem("heat_three_modes", D, t_end,
                        lambda X: sum(a * np.sin(k * np.pi * X[:, 0:1]) for k, a in enumerate(amps, start=1)))
    sol = pp.solve(spec, method="pinn", epochs=epochs, seed=seed)
    x, t = np.linspace(0.0, 1.0, 101), np.linspace(0.0, t_end, 51)
    T, X = np.meshgrid(t, x, indexing="ij")
    U_pinn = sol.predict(np.column_stack([X.ravel(), T.ravel()]).astype(np.float32))[:, 0].reshape(T.shape)
    U_exact = exact(x, t)
    ref = [-D * (k * np.pi) ** 2 for k in (1, 2, 3)]

    rows = {}
    for label, U in (("exact", U_exact), ("pinn", U_pinn)):
        r = DMD(rank=3).fit(dyn.from_snapshots(t, U, kind="pinn" if label == "pinn" else "analytic"))
        ev = r.details["operator"].eigenvalues
        om = np.sort(ev.continuous.real)[::-1]
        rows[label] = {"rates": om, "rel": np.abs(om - ref) / np.abs(ref), "residuals": r.uncertainty["residuals"],
                       "gd_rank": DMD().fit(dyn.from_snapshots(t, U)).estimate["rank"]}
    field_err = float(np.linalg.norm(U_pinn - U_exact) / np.linalg.norm(U_exact))
    ex.log_metric("koopman_pinn_field_rel_l2", field_err, "-", "relative L2 of the PINN on the 101 x 51 grid")
    for k in range(3):
        ex.log_metric(f"koopman_pinn_rate_rel_error_mode{k + 1}", rows["pinn"]["rel"][k], "-",
                      f"|omega_DMD - (-D ({k + 1} pi)^2)| / |.|, DMD rank 3 on the PINN snapshots")
        ex.log_metric(f"koopman_exact_rate_rel_error_mode{k + 1}", rows["exact"]["rel"][k], "-",
                      "the same on the exact snapshots (control)")
    ex.log_metric("koopman_pinn_gavish_donoho_rank", rows["pinn"]["gd_rank"], "modes",
                  "Gavish-Donoho rank of the PINN snapshot matrix (3 for the exact field)")
    ex.log_metric("koopman_pinn_max_residual", float(np.max(rows["pinn"]["residuals"])), "-",
                  "largest ResDMD residual of the 3 PINN eigenpairs")
    ex.check("DMD recovers the exact decay rates (control)", value=float(np.max(rows["exact"]["rel"])), max=1e-6,
             kind="reference")
    ex.check("PINN spectrum: every decay rate within 5 %", value=float(np.max(rows["pinn"]["rel"])), max=0.05,
             kind="reference", detail="DMD of the PINN rollout against -D k^2 pi^2, k = 1, 2, 3")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    ks = np.arange(1, 4)
    ax.plot(ks, -np.array(ref), "ko-", label="exact D k^2 pi^2")
    ax.plot(ks, -rows["pinn"]["rates"], "r^--", label="DMD of the PINN")
    ax.set(xlabel="mode k", ylabel="decay rate", title="Spectrum of a PINN (u_t = D u_xx)", xticks=ks)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "koopman.png"), dpi=110)
    plt.close(fig)
    print(f"koopman: field rel L2 {field_err:.2e}; rate errors PINN {np.round(rows['pinn']['rel'], 5)}, exact "
          f"{rows['exact']['rel']}; GD rank PINN {rows['pinn']['gd_rank']}")


def fisher(ex, epochs: int, seed: int) -> None:
    D, xs, sigma, n = 0.05, 0.5, 0.01, 4
    ps = [Parameter("A", 1.0, 0.1, 10.0), Parameter("D", D, 1e-3, 1.0)]
    model = CallableModel(lambda th, inp: th["A"] * np.sin(np.pi * xs) * np.exp(-th["D"] * np.pi ** 2 * inp["t"]),
                          inputs=["t"], parameters=ps)
    sch = dyn.ObservationScheduler(model, (0.0, 10.0), n_grid=401, sigma=sigma)
    opt = sch.schedule(n, "D")
    t_opt = np.sort(np.asarray(opt.estimate["times"], dtype=float))
    t_uni = np.linspace(0.0, 10.0, n)
    out = {}
    for label, tt in (("optimal", t_opt), ("uniform", t_uni)):
        obs = Observations({"t": tt}, sigma)
        crlb = dyn.FisherInformation.gaussian(model, obs).crlb()
        sd_bound = float(np.sqrt(crlb.cov[1, 1]))
        perf = dyn.fisher.estimator_performance(model, obs, n_rep=500, seed=seed)
        out[label] = {"bound": sd_bound, "empirical": float(perf.estimate["D"]["empirical_std"]),
                      "efficiency": float(perf.estimate["D"]["efficiency"])}
    ratio = out["uniform"]["bound"] / out["optimal"]["bound"]
    ex.log_metric("fisher_optimal_times", float(np.mean(t_opt)), "time",
                  f"mean of the {n} D-optimal times {np.round(t_opt, 3).tolist()} (grid of 401 on [0, 10])")
    for label in ("optimal", "uniform"):
        ex.log_metric(f"fisher_{label}_crlb_sd_D", out[label]["bound"], "length^2/time",
                      "sqrt of the (D, D) entry of the inverse Fisher matrix, sigma = 0.01")
        ex.log_metric(f"fisher_{label}_empirical_sd_D", out[label]["empirical"], "length^2/time",
                      "sd of 500 least-squares fits of simulated data")
    ex.log_metric("fisher_sd_ratio_uniform_over_optimal", ratio, "-", "CRLB sd of D, uniform / optimal")
    ex.check("D-optimal design is at least as informative as the uniform one", value=ratio, min=1.0,
             kind="verification")
    for label in ("optimal", "uniform"):
        ex.check(f"least squares attains the bound ({label} times): efficiency in [0.8, 1.2]",
                 value=out[label]["efficiency"], min=0.8, max=1.2, kind="reference",
                 detail="Cramer-Rao bound against 500 refits")

    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    tt = np.linspace(0, 10, 300)
    ax.plot(tt, np.exp(-D * np.pi ** 2 * tt), "k-", label="u(0.5, t)")
    ax.plot(t_opt, np.exp(-D * np.pi ** 2 * t_opt), "ro", ms=8, label=f"D-optimal (sd {out['optimal']['bound']:.1e})")
    ax.plot(t_uni, np.exp(-D * np.pi ** 2 * t_uni), "bs", mfc="none", ms=8,
            label=f"uniform (sd {out['uniform']['bound']:.1e})")
    ax.set(xlabel="t", ylabel="u", title="When to measure to estimate D")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fisher.png"), dpi=110)
    plt.close(fig)
    print(f"fisher: optimal times {np.round(t_opt, 3)}, sd(D) optimal {out['optimal']['bound']:.3e} "
          f"(empirical {out['optimal']['empirical']:.3e}), uniform {out['uniform']['bound']:.3e} "
          f"(empirical {out['uniform']['empirical']:.3e}); ratio {ratio:.2f}")


def dimensional(ex, epochs: int, seed: int) -> None:
    Q = dyn.PhysicalQuantity.of
    pi_an = dyn.BuckinghamPiAnalyzer([Q("u", Theta=1), Q("u0", Theta=1), Q("x", L=1), Q("L", L=1), Q("t", T=1),
                                      Q("D", L=2, T=-1)], target="u", repeating=["u0", "L", "D"])
    groups = pi_an.groups()
    names = sorted(str(g.formula()) for g in groups)
    ex.log_metric("dimensional_n_groups", len(groups), "groups", "Buckingham pi: 6 quantities, rank 3")

    slabs = [(1.0, 0.1), (2.0, 0.1), (1.0, 0.4), (0.5, 0.02), (3.0, 1.0), (0.8, 0.05)]  # 6! = 720 permutations
    rng = np.random.default_rng(seed)
    members = []
    for L, D in slabs:
        t = np.sort(rng.uniform(0, 1.2 * L ** 2 / D, 120))           # irregular sampling, own window per slab
        n = np.arange(1, 400, 2)
        u = (4 / np.pi) * np.sum((np.sin(n * np.pi / 2) / n)[None, :]
                                 * np.exp(-D * (n * np.pi / L) ** 2 * t[:, None]), axis=1)
        members.append(dyn.from_array(t, u, names=["u_centre"], kind="analytic"))
    cc = dyn.CurveCollapseAnalyzer(dyn.TrajectoryEnsemble(members))
    res = cc.analyze(x_scales=[L ** 2 / D for L, D in slabs], label="t D / L^2", seed=seed)
    raw, scaled, p = res.estimate["raw"], res.estimate["collapse"], res.estimate["p_value"]
    ex.log_metric("dimensional_collapse_raw", raw, "-", "mean_x Var_k[y] / Var_x[mean_k y] against t")
    ex.log_metric("dimensional_collapse_scaled", scaled, "-", "the same against t D / L^2")
    ex.log_metric("dimensional_permutation_p", p, "-", "permutation control, 999 reassignments of the scales")
    ex.check("three groups: u/u0, x/L, t D/L^2", len(groups) == 3, kind="reference", detail=", ".join(names))
    ex.check("centre temperatures collapse on t D / L^2", value=scaled, max=1e-3, kind="reference",
             detail="exact series: the collapse is exact up to interpolation on irregular samples")
    ex.check("the collapse beats the permutation control", value=p, max=0.05, kind="verification")

    fig, axs = plt.subplots(1, 2, figsize=(8, 3.4))
    for (L, D), m in zip(slabs, members, strict=True):
        axs[0].plot(m.t, m.x[:, 0], ".", ms=3, label=f"L={L}, D={D}")
        axs[1].plot(m.t * D / L ** 2, m.x[:, 0], ".", ms=3)
    axs[0].set(xlabel="t", ylabel="u(L/2) / u0", title="raw")
    axs[1].set(xlabel="t D / L^2", title="rescaled")
    axs[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "dimensional.png"), dpi=110)
    plt.close(fig)
    print(f"dimensional: groups {names}; collapse raw {raw:.3g} -> scaled {scaled:.3g}, permutation p {p:.3g}")


def early_warning(ex, epochs: int, seed: int) -> None:
    an = dyn.EarlyWarningAnalyzer()
    runs = {"saddle_node": ew.ramp_series("saddle_node", sigma=0.002, seed=seed),
            "stable": ew.ramp_series("stable", seed=seed),
            "stable_growing_noise": ew.ramp_series("stable_growing_noise", seed=seed)}
    res = {k: an.analyze(dyn.from_array(r["t"], r["x"]), seed=seed) for k, r in runs.items()}
    for k, r in res.items():
        for ind in ("variance", "ac1"):
            ex.log_metric(f"early_warning_{k}_{ind}_tau", r.estimate[ind]["tau"], "-",
                          f"Kendall tau of rolling {ind}, LOESS detrend, window 0.5")
            ex.log_metric(f"early_warning_{k}_{ind}_p", r.estimate[ind]["p_value"], "-",
                          "one-sided p from 199 AR(1) surrogates")
    sn, st, gn = res["saddle_node"].estimate, res["stable"].estimate, res["stable_growing_noise"].estimate
    ex.check("saddle-node: variance and AC1 rise significantly", sn["n_significant_rising"] == 2,
             kind="reference", detail=f"tau var {sn['variance']['tau']:.2f}, ac1 {sn['ac1']['tau']:.2f}")
    ex.check("stable control: no significant rise", st["n_significant_rising"] == 0, kind="verification")
    ex.check("growing noise: AC1 does not rise significantly (variance may: a known false positive)",
             gn["ac1"]["p_value"] > 0.05, kind="verification",
             detail=f"variance p {gn['variance']['p_value']:.3f}, ac1 p {gn['ac1']['p_value']:.3f}")

    fig, axs = plt.subplots(2, 3, figsize=(10, 4.6), sharex="col")
    for j, (k, r) in enumerate(res.items()):
        ind = r.details["indicators"]
        axs[0, j].plot(runs[k]["t"], runs[k]["x"], lw=0.6)
        axs[0, j].set_title(k, fontsize=9)
        axs[1, j].plot(ind["ac1"].t, ind["ac1"].values, "r-", lw=1)
        axs[1, j].set_xlabel("t")
    axs[1, 0].set_ylabel("rolling AC1")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "early_warning.png"), dpi=110)
    plt.close(fig)
    print("early_warning: " + "; ".join(f"{k} tau(var, ac1) = ({r.estimate['variance']['tau']:.2f}, "
                                        f"{r.estimate['ac1']['tau']:.2f}), significant {r.estimate['n_significant_rising']}"
                                        for k, r in res.items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=CASES, action="append")
    ap.add_argument("--epochs", type=int, default=2000, help="PINN epochs (cases first_passage, koopman)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(FIG, exist_ok=True)
    with pinneapple_lab.experiment(
            "examples/dynamics_toolkit", title="Dynamics toolkit end to end",
            description="First passage from a pp.solve PINN, the Koopman spectrum of a PINN, sensor times by Fisher "
                        "information, Buckingham pi with curve collapse, and early-warning indicators with controls; "
                        "each against an independent reference.",
            domain="dynamical systems", params={"epochs": a.epochs, "seed": a.seed, "cases": a.case or list(CASES)},
            methods=["pinneapple_analysis.dynamics", "pp.solve(method='pinn')"],
            modules=["pinneapple_analysis.dynamics", "pinneapple"], out_dir=HERE, register=False,
            links={"guide": "docs/guides/dynamics_toolkit.md",
                   "script": "examples/dynamics_toolkit/dynamics_toolkit.py"}) as ex:
        for name in a.case or CASES:
            globals()[name](ex, a.epochs, a.seed)


if __name__ == "__main__":
    main()
