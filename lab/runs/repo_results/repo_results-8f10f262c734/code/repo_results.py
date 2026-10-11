"""Results already produced in this repository, imported into the lab as validated runs and datasets.

Each source reads the artefacts a repository script wrote (benchmarks/_out, examples/*/results, ...), re-checks
them against their reference (exact solution, published value, solver baseline), and records metrics, figures and
datasets, so the lab database also holds the long runs that are too expensive to repeat on every sweep. The
scripts that produced the artefacts are snapshotted with each run (``code_files``).

    python -m pinneapple_lab sweep repo_results -g source=burgers_pinn,fin_inverse_2d,fin_inverse_3d,...
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np

from ..spec import Experiment, register

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

SOURCES = {
    "burgers_pinn": "PINN for viscous Burgers (nu = 0.01) vs the exact Cole-Hopf solution",
    "fin_inverse_2d": "PINN inverse problem: heat-transfer coefficient of a plate from 8 noisy sensors",
    "fin_inverse_3d": "PINN inverse problem: heat-transfer coefficient of a 3D block from sensors",
    "lbm_strouhal": "Lattice-Boltzmann vortex shedding: Strouhal numbers vs the published cylinder value",
    "meshgraphnet": "MeshGraphNet rollouts (synthetic diffusion, DeepMind cylinder flow) vs frozen baseline",
    "concorde_aoa": "LBM-LES slender delta wing (parametric Concorde stand-in): CL/CD/CM vs angle of attack",
    "bh_forecast": "U-Net forecast of black-hole accretion flows: lead-time skill vs persistence",
}

_SCRIPTS = {
    "burgers_pinn": ["examples/benchmark_suite/03_pinn_burgers_full_pipeline.py"],
    "fin_inverse_2d": ["examples/use_cases/fin_convection_inverse/plate_2d.py",
                       "examples/use_cases/fin_convection_inverse/fv_reference.py"],
    "fin_inverse_3d": ["examples/use_cases/fin_convection_inverse/block_3d.py"],
    "lbm_strouhal": ["benchmarks/lbm_vortex_shedding.py"],
    "meshgraphnet": ["examples/meshgraphnet/01_synthetic_diffusion.py",
                     "examples/meshgraphnet/02_cylinder_flow_deepmind.py"],
    "concorde_aoa": ["examples/use_cases/concorde_high_aoa/concorde_high_aoa_pipeline.py"],
    "bh_forecast": ["examples/black_hole_weather/evaluate.py"],
}


def _p(rel: str) -> str:
    return os.path.join(_REPO, rel)


def _json(rel: str):
    with open(_p(rel)) as f:
        return json.load(f)


def _figures(ctx, pattern: str, prefix: str = "") -> None:
    for f in sorted(glob.glob(_p(pattern))):
        ctx.figure_file(f, prefix + os.path.basename(f))


def _burgers(ctx):
    from pinneapple_physics.closed_form.burgers import burgers_sine_exact
    d = "data/artifacts/examples/burgers_pinn/"
    u, x, t = (np.load(_p(d + n)) for n in ("u_pred.npy", "x_grid.npy", "t_grid.npy"))
    nu = 0.01
    X, T = np.meshgrid(x, t)                                  # u_pred is [t, x], as the script saves it
    ue = burgers_sine_exact(X, T, nu).astype(np.float32)
    rel = float(np.linalg.norm(u - ue) / np.linalg.norm(ue))
    hist = _json(d + "training_history.json")
    for k, v in hist.items():
        ctx.metric(f"final_loss_{k}", v[-1])
    ctx.metric("rel_l2_vs_exact", rel)
    ctx.metric("max_abs_error", float(np.abs(u - ue).max()))
    ctx.check("rel_l2_vs_exact", value=rel, max=0.1, detail="exact Cole-Hopf solution, nu = 0.01")
    ds = ctx.dataset("fields", description="Burgers u(t, x): PINN prediction and exact solution",
                     units={"x": "-", "t": "-", "u": "-"})
    ds.add(x=x, t=t, u_pred=u, u_exact=ue, nu=nu, model="pinn_mlp", source="burgers_pinn")
    _plot_pair(ctx, "burgers_pinn_vs_exact", u, ue, extent=[x[0], x[-1], t[0], t[-1]], xlabel="x", ylabel="t",
               cmap="RdBu_r", title=("PINN", "exact (Cole-Hopf)"))
    ctx.figure_file(_p(d + "burgers_pinn_results.png"))


def _fin(ctx, dim: str):
    d = f"examples/use_cases/fin_convection_inverse/results/{dim}/"
    s = _json(d + "summary.json")
    h_true = s["setup"]["h_true_W_m2K"]
    h = s["h_pinn"]["mean"]
    ctx.input("setup", s["setup"])
    ctx.metric("h_true", h_true)
    ctx.metric("h_pinn_mean", h)
    ctx.metric("h_pinn_std", s["h_pinn"]["std"])
    ctx.metric("h_fv_fit_mean", s["h_fv_fit"]["mean"])
    ctx.metric("h_initial_guess", s["setup"].get("h_initial_guess"))
    ctx.metric("max_field_error_C", s["max_field_error_C"])
    ctx.metric("source_seconds", s.get("seconds"))
    ctx.check("h_fv_refit", value=s["h_fv_fit"]["mean"], reference=h_true, rtol=0.05,
              detail="h from re-running the finite-volume model against the sensors")
    ctx.check("h_trainable", value=h, reference=h_true, rtol=0.1,
              detail="the network's own h; the README documents -7 % in 3D (weak flux constraint)")
    ctx.check("field_error", value=s["max_field_error_C"], max=2.0, detail="max |T_pinn - T_fv| in C")
    ref = np.load(_p(d + "field_reference.npy"))
    ds = ctx.dataset(f"temperature_fields_{dim}", description=f"{dim} temperature field: finite-volume reference and "
                     "PINN reconstructions from noisy sensors (one per noise draw)", units={"T": "C", "h": "W/m2K"})
    for i, f in enumerate(sorted(glob.glob(_p(d + "field_pinn_*.npy")))):
        run = s["runs"][i] if i < len(s.get("runs", [])) else {}
        ds.add(T_pinn=np.load(f), T_reference=ref, draw=i, h_pinn=float(run.get("h_pinn", np.nan)),
               h_true=h_true, case=s["case"])
    _figures(ctx, d + "*.png")


def _lbm(ctx):
    rows = []
    for f in sorted(glob.glob(_p("benchmarks/_out/lbm_vortex_shedding_*.json"))):
        r = json.load(open(f))
        tag = os.path.basename(f)[len("lbm_vortex_shedding_"):-5]
        rows.append((tag, r))
        ctx.metric(f"St_{tag}", r["St"])
        if r.get("reference_St"):
            ctx.check(f"St_{tag}", value=r["St"], reference=r["reference_St"], rtol=0.2,
                      detail=f"Re {r['Re']:g}, D = {r['length_scale_lattice']:g} cells; reference St 0.1647")
        else:
            ctx.check(f"St_{tag}_bluff_body_range", passed=0.12 <= r["St"] <= 0.25, detail=r.get("reference_note", ""))
    ds = ctx.dataset("strouhal", description="LBM vortex-shedding runs: Strouhal number, set-up and probe amplitude")
    for tag, r in rows:
        ds.add(case=r["case"], tag=tag, Re=float(r["Re"]), St=float(r["St"]),
               St_reference=float(r["reference_St"]) if r.get("reference_St") else None,
               length_scale_cells=float(r["length_scale_lattice"]), u_in=float(r["u_in"]),
               steps=int(r["steps"]), periods=float(r["periods_analysed"]),
               probe_uy_amplitude=float(r["probe_uy_amplitude"]))


def _mgn(ctx):
    sets = {"synthetic_diffusion": _json("examples/meshgraphnet/_out/synthetic_diffusion.json"),
            "cylinder_flow": _json("examples/meshgraphnet/_out/cylinder_flow/metrics.json")}
    ds = ctx.dataset("rollout_scores", description="MeshGraphNet rollout and one-step RMSE vs the frozen-initial-"
                     "condition baseline")
    for k, m in sets.items():
        ctx.metric(f"{k}_rollout_rmse", m["rollout_rmse"])
        ctx.metric(f"{k}_baseline_rmse", m["frozen_ic_baseline_rmse"])
        ctx.check(f"{k}_beats_baseline", value=m["rollout_rmse"], max=m["frozen_ic_baseline_rmse"])
        ds.add(case=k, rollout_rmse=m["rollout_rmse"], one_step_rmse=m["one_step_rmse"],
               baseline_rmse=m["frozen_ic_baseline_rmse"], n_steps=int(m["n_steps"]))
    _figures(ctx, "examples/meshgraphnet/_out/cylinder_flow/*.png")


def _concorde(ctx):
    m = _json("examples/use_cases/concorde_high_aoa/outputs/metrics.json")
    ctx.input("geometry", m["geometry"])
    ctx.input("lbm_config", m["lbm_config"])
    sweep = sorted(m["aoa_sweep"], key=lambda r: r["aoa_deg"])
    a = np.array([r["aoa_deg"] for r in sweep])
    cl = np.array([r["CL"] for r in sweep])
    slope = float(np.polyfit(np.radians(a), cl, 1)[0])
    v = m["validation"]
    ref = np.interp(a, v["reference_alpha_deg"], v["reference_CL"])
    ctx.metric("CL_slope_per_rad", slope)
    ctx.metric("CL_max", float(cl.max()))
    ctx.metric("CL_rms_vs_polhamus", float(np.sqrt(np.mean((cl - ref) ** 2))))
    ctx.check("lift_increases_with_aoa", passed=bool(np.all(np.diff(cl) > -0.02)))
    ctx.check("CL_within_polhamus_band", value=float(np.sqrt(np.mean((cl - ref) ** 2))), max=0.5,
              detail="low-fidelity cross-check vs the Polhamus slender-wing theory, not test data")
    ds = ctx.dataset("polar", description="LBM-LES aerodynamic coefficients vs angle of attack (Re 300, coarse)")
    for r, rcl in zip(sweep, ref, strict=True):
        ds.add(aoa_deg=r["aoa_deg"], CL=r["CL"], CD=r["CD"], CM=r["CM"], CY=r["CY"], Cp_min=r["Cp_min"],
               Cp_max=r["Cp_max"], enstrophy=r.get("enstrophy"), CL_polhamus=float(rcl))
    _figures(ctx, "examples/use_cases/concorde_high_aoa/outputs/*.png")


def _bh(ctx):
    ds = ctx.dataset("lead_time_skill", description="Mean absolute error vs lead time of the black-hole flow "
                     "forecasters and of persistence, in and out of distribution")
    for f in sorted(glob.glob(_p("examples/black_hole_weather/results/one_*.json"))):
        r = json.load(open(f))
        tag = os.path.basename(f)[4:-5]
        sc = r["scores"]
        mae, pers = np.asarray(sc["mae"]), np.asarray(sc["persistence"])
        lead = np.asarray(r["lead"], dtype=float)
        ctx.metric(f"{tag}_one_block_mae", r["one_block_mae"])
        ctx.metric(f"{tag}_one_block_persistence", r["one_block_persistence"])
        ctx.metric(f"{tag}_horizon_beats_persistence", r["horizon_mean"]["beats_persistence"])
        ds.add(model=tag.split("_")[0], test=r["test"], lead=lead, mae=mae, persistence=pers,
               climatology=np.asarray(sc["climatology"]), acc=np.asarray(sc["acc"]),
               horizon_beats_persistence=float(r["horizon_mean"]["beats_persistence"]))
        if tag.startswith("res"):
            ctx.check(f"{tag}_beats_persistence_first_block", value=r["one_block_mae"],
                      max=r["one_block_persistence"])
    _figures(ctx, "docs/assets/blackhole/*skill*.png")


def _plot_pair(ctx, name, a, b, *, extent, xlabel, ylabel, cmap, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
    lo, hi = float(min(a.min(), b.min())), float(max(a.max(), b.max()))
    for k, (z, tt) in enumerate(zip((a, b), title, strict=True)):
        im = ax[k].imshow(z, origin="lower", aspect="auto", extent=extent, cmap=cmap, vmin=lo, vmax=hi)
        ax[k].set_title(tt)
        ax[k].set_xlabel(xlabel)
        ax[k].set_ylabel(ylabel)
    fig.colorbar(im, ax=ax[:2], shrink=0.9)
    im = ax[2].imshow(np.abs(a - b), origin="lower", aspect="auto", extent=extent, cmap="magma")
    ax[2].set_title("|error|")
    ax[2].set_xlabel(xlabel)
    fig.colorbar(im, ax=ax[2], shrink=0.9)
    ctx.figure(name, fig)


_RUN = {"burgers_pinn": _burgers, "fin_inverse_2d": lambda c: _fin(c, "2d"), "fin_inverse_3d": lambda c: _fin(c, "3d"),
        "lbm_strouhal": _lbm, "meshgraphnet": _mgn, "concorde_aoa": _concorde, "bh_forecast": _bh}


@register
class RepoResults(Experiment):
    name = "repo_results"
    version = "1"
    description = ("Results already produced by repository scripts (PINNs, inverse problems, LBM, MeshGraphNet, "
                   "black-hole forecasts), re-validated against their references and stored as datasets.")
    tags = ["import", "benchmark", "dataset"]
    params = {"source": "burgers_pinn"}
    space = {"source": list(SOURCES)}
    code_files = [_p(s) for v in _SCRIPTS.values() for s in v]

    def run(self, ctx):
        src = ctx.params["source"]
        if src not in _RUN:
            raise KeyError(f"unknown source {src!r}; one of {sorted(_RUN)}")
        ctx.log(SOURCES[src])
        ctx.input("source", {"name": src, "description": SOURCES[src], "scripts": _SCRIPTS[src]})
        with ctx.stage("import"):
            _RUN[src](ctx)
