"""The cases on the PINNeAPPle Labs landing page that live in public companion repositories, as lab runs.

``benchmark_case`` imports one finished experiment of PINNeAPPle-Benchmark (or PINNeAPPle-Climate): it re-checks the
headline claim against the numbers in the experiment's own ``results.json`` (the model against its baseline, the
reproduction against the improvement), records the headline metrics, the figures, the arrays the paper's figures read
(``results/*.npz`` as the ``arrays`` dataset), the paper link, and the experiment's source as the code that ran.
The experiments themselves take hours to days (100 crash simulations, ten years of hourly weather), so the lab
stores their published results instead of repeating them.

The companion repository is found at ``$PINNEAPPLE_BENCHMARK`` / ``$PINNEAPPLE_CLIMATE``, next to this checkout
(``../pinneapple-benchmark``), or is cloned (shallow, public) into ``<lab>/_external``.

    python -m pinneapple_lab sweep benchmark_case -g case=soil_twin,bumper_crash,terramechanics,heated_channel,pdr,shock_train,sst
"""
from __future__ import annotations

import glob
import json
import os
import subprocess

import numpy as np

from ..spec import Experiment, register
from .examples import REPO

_REPOS = {
    "benchmark": ("PINNeAPPle-Benchmark", "https://github.com/PINNeAPPle-Labs/PINNeAPPle-Benchmark",
                  "PINNEAPPLE_BENCHMARK"),
    "climate": ("PINNeAPPle-Climate", "https://github.com/PINNeAPPle-Labs/PINNeAPPle-Climate", "PINNEAPPLE_CLIMATE"),
}

CASES = {
    "soil_twin": ("benchmark", "experiments/02_physics_digital_twin",
                  "Soil-temperature digital twin from ten years of real hourly weather, Natal (RN): identified "
                  "physics ODE, PINN, ML baselines, anomaly detection"),
    "bumper_crash": ("benchmark", "experiments/03_bumper_beam_transolver",
                     "Bumper-beam crash surrogate: 100 OpenRadioss simulations, Transolver reproduced then improved, "
                     "classical baselines"),
    "terramechanics": ("benchmark", "experiments/04_terramechanics_robust",
                       "Wheel-soil terramechanics surrogate: constraint audit, robust PINN with hard constraints"),
    "heated_channel": ("benchmark", "experiments/05_thermal_channel_twin",
                       "Digital twin of an actuated heated channel: verified CFD, POD/FNO/DeepONet surrogates, "
                       "assimilation and control"),
    "pdr": ("benchmark", "experiments/07_pdr_thesis_reproduction",
            "Pedestrian dead reckoning from a phone IMU: a thesis reproduced, then improved"),
    "shock_train": ("benchmark", "experiments/08_supersonic_shock_train",
                    "Shock train in a supersonic duct with a fixed-pressure outlet: back-pressure sweep, grid and 3-D"),
    "sst": ("climate", "experiments/01_sst_satellite_forecasting",
            "Sea-surface temperature forecasts from satellite data, Brazil-Malvinas Confluence: FNO and CNN "
            "ensembles against persistence on unseen years"),
}


def source_root(key: str, lab_root: str | None = None) -> str | None:
    """Local checkout of a companion repository (environment variable, sibling folder, or a shallow clone)."""
    name, url, env = _REPOS[key]
    for cand in (os.environ.get(env), os.path.join(os.path.dirname(REPO), name.lower()),
                 os.path.join(os.path.dirname(REPO), name)):
        if cand and os.path.isdir(os.path.join(cand, "experiments")):
            return cand
    if lab_root is None:
        return None
    dst = os.path.join(lab_root, "_external", name)
    if not os.path.isdir(os.path.join(dst, "experiments")):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        r = subprocess.run(["git", "clone", "-q", "--depth", "1", url, dst], capture_output=True, timeout=1800)
        if r.returncode != 0:
            return None
    return dst


def _get(d, path, default=None):
    for k in path:
        if isinstance(d, dict) and k in d:
            d = d[k]
        elif isinstance(d, list) and isinstance(k, int) and -len(d) <= k < len(d):
            d = d[k]
        else:
            return default
    return d


# ---------------------------------------------------------------------- per-case headline claims
def _soil(ctx, r):
    t = r["test"]
    pers = t["Persistence"]["metrics"]["h6"]["RMSE"]
    ode = t["Physics ODE (identified)"]["metrics"]["h6"]["RMSE"]
    pinn = t["PINN (PINNeAPPle)"]["metrics"]["h6"]["RMSE"]
    best = min((v["metrics"]["h6"]["RMSE"], k) for k, v in t.items())
    ctx.metric("rmse_h6_persistence_K", pers)
    ctx.metric("rmse_h6_physics_ode_K", ode)
    ctx.metric("rmse_h6_pinn_K", pinn)
    ctx.metric("rmse_h6_best_K", best[0])
    ctx.output("best_model_h6", best[1])
    an = r["anomaly"]["by_type"]["spike"]
    ctx.metric("spike_recall_twin", an["recall_twin_anomaly"])
    ctx.metric("spike_recall_naive_band", an["recall_naive"])
    ctx.metric("false_alarm_rate_twin", r["anomaly"]["false_alarm_rate"]["twin_ANOMALY"])
    ctx.check("physics_ode_beats_persistence_6h", value=ode, max=pers)
    ctx.check("twin_detects_more_spikes_than_band", value=an["recall_twin_anomaly"], min=an["recall_naive"])
    # published as a negative result: at 6 h the PINN does not beat persistence (recorded, not hidden)
    ctx.metric("pinn_over_persistence_h6", pinn / pers)


def _bumper(ctx, r):
    rep, imp = r["reproduction"]["test"], r["improved"]["test"]
    for tag, t in (("reproduction", rep), ("improved", imp), ("pod_gp", r["classical"]["pod_gp"]),
                   ("nearest", r["classical"]["nearest"])):
        ctx.metric(f"{tag}_field_rel_l2", t["field_rel_l2"])
        ctx.metric(f"{tag}_node_rmse_mm", t["node_rmse_mm"])
        ctx.metric(f"{tag}_node_r2", t["node_r2"])
    ctx.metric("simulations", r["data"]["n_runs"])
    ctx.metric("mesh_nodes", r["data"]["nodes"])
    ctx.check("improved_beats_reproduction", value=imp["field_rel_l2"], max=rep["field_rel_l2"])
    ctx.check("improved_node_r2", value=imp["node_r2"], min=0.95)


def _terra(ctx, r):
    rob = r["robust2d"]["Robust PINN (hard R2,R4 + soft R3,R5)"]
    mlp = r["robust2d"]["Data-only MLP (ModifiedMLP)"]
    orig = r["original"]
    for out in ("Fx", "Fz", "My"):
        ctx.metric(f"original_{out}_rel_l2", orig["test"][out]["rel_L2"])
        ctx.metric(f"robust_pinn_{out}_rel_l2", rob["test"][out]["rel_L2"])
        ctx.metric(f"data_only_{out}_rel_l2", mlp["test"][out]["rel_L2"])
    for k, v in rob["constraint_violations"].items():
        ctx.metric(f"robust_pinn_violations_{k}", v)
    ctx.check("hard_constraints_hold", passed=rob["constraint_violations"]["R2_mohr_coulomb"] == 0
              and rob["constraint_violations"]["R4_torque"] == 0, detail="Mohr-Coulomb and torque, built in")
    ctx.check("robust_beats_original_on_torque", value=rob["test"]["My"]["rel_L2"], max=orig["test"]["My"]["rel_L2"])


def _channel(ctx, r, src):
    sel = r["selected_on_val"]
    for name, m in r["models"].items():
        for split in ("test_same_family", "test_unseen_sinusoid"):
            v = _get(m, ["test", split, "rel_l2", "T"])
            if v is not None and np.isfinite(v) and v < 1e3:
                ctx.metric(f"{name}|{split}|T_rel_l2", v)
    ctx.output("selected_on_validation", sel)
    t_sel = r["models"][sel]["test"]["test_unseen_sinusoid"]["rel_l2"]["T"]
    ctx.check("selected_surrogate_T_unseen_inputs", value=t_sel, max=0.15, detail=sel)
    try:
        with open(os.path.join(src, "results", "verification.json")) as f:
            ver = json.load(f)
        nu = ver["nusselt"][-1]
        ctx.metric("cfd_nusselt_rel_error", nu["rel_err"])
        ctx.check("cfd_nusselt_vs_reference", value=nu["rel_err"], max=0.01, detail="Nu = 5.385, fully developed")
        po = ver["poiseuille"][-1]
        ctx.metric("cfd_poiseuille_profile_max_error", po["u_profile_max_err"])
    except (OSError, KeyError, IndexError, ValueError):
        pass


def _pdr(ctx, r):
    rep = r["ch3_tlio"]["dead_reckoning_90s"]["drift_median_pct"]
    imu = r["imu_only_dead_reckoning_90s"]["drift_median_pct"]
    var = r["improvements"]["variants"]
    best = min((v["dead_reckoning_90s"]["drift_median_pct"], k) for k, v in var.items())
    ctx.metric("drift_90s_imu_only_pct", imu)
    ctx.metric("drift_90s_reproduction_pct", rep)
    ctx.metric("drift_90s_improved_pct", best[0])
    ctx.output("best_variant", best[1])
    ctx.check("network_beats_raw_imu", value=rep, max=imu)
    ctx.check("improvement_beats_reproduction", value=best[0], max=rep, detail=best[1])


def _shock(ctx, r, src):
    sweep = []
    for f in sorted(glob.glob(os.path.join(src, "results", "case_sweep_*.json"))):
        d = json.load(open(f))
        sweep.append((d["pb_ratio"], d["x_s_final"], bool(d.get("unstarted")), bool(d.get("no_train"))))
    trained = [s for s in sweep if not s[2] and not s[3]]
    for pb, x_s, _, _ in sweep:
        ctx.metric(f"x_shock_pb{pb:g}", x_s)
    xs = [s[1] for s in trained]
    ctx.check("shock_moves_upstream_with_back_pressure", passed=all(a > b for a, b in zip(xs, xs[1:], strict=False)),
              detail="shock-train leading edge vs back-pressure ratio")
    ctx.check("unstarts_at_highest_back_pressure", passed=bool(sweep and sweep[-1][2]))
    grid = []
    for f in sorted(glob.glob(os.path.join(src, "results", "case_grid_*.json"))):
        d = json.load(open(f))
        grid.append(d["x_s_final"])
    if grid:
        ctx.metric("grid_x_shock_coarse_to_fine", ",".join(f"{g:.3g}" for g in grid))
    ds = ctx.dataset("sweep", description="Shock-train leading-edge position against back-pressure ratio")
    for pb, x, uns, no in sweep:
        ds.add(pb_ratio=pb, x_shock=x, unstarted=uns, no_train=no)


def _sst(ctx, r):
    t = r["scores"]["test"]
    sel = r["selected_model_on_val"]
    pers = t["Persistence"]["rmse"][-1]
    best = t[sel]["rmse"][-1]
    for name, m in t.items():
        ctx.metric(f"{name}|rmse_lead7_degC", m["rmse"][-1])
    ctx.metric("rmse_reduction_vs_persistence_lead7", 1 - best / pers)
    ctx.output("selected_on_validation", sel)
    ctx.check("selected_beats_persistence_lead7", value=best, max=pers, detail=sel)
    ds = ctx.dataset("lead_skill", description="Test RMSE (degC) per forecast lead for every model")
    for name, m in t.items():
        ds.add(model=name, rmse=np.asarray(m["rmse"]), mae=np.asarray(m.get("mae", m["rmse"])))


_RUN = {"soil_twin": _soil, "bumper_crash": _bumper, "terramechanics": _terra, "heated_channel": _channel,
        "pdr": _pdr, "shock_train": _shock, "sst": _sst}


@register
class BenchmarkCase(Experiment):
    name = "benchmark_case"
    version = "1"
    description = ("Cases from the PINNeAPPle Labs landing page that live in the public PINNeAPPle-Benchmark and "
                   "PINNeAPPle-Climate repositories, imported with their headline claim re-checked, figures, the "
                   "arrays behind the paper's figures and the source that produced them.")
    tags = ["landing", "benchmark", "surrogate", "digital-twin", "dataset"]
    params = {"case": "heated_channel"}
    space = {"case": list(CASES)}

    @classmethod
    def code_for(cls, params):
        key, rel, _ = CASES[params["case"]]
        src = source_root(key)
        if not src:
            return []
        folder = os.path.join(src, rel)
        return sorted(glob.glob(os.path.join(folder, "src", "*.py")) + glob.glob(os.path.join(folder, "src", "*.sh")))

    def run(self, ctx):
        case = ctx.params["case"]
        key, rel, desc = CASES[case]
        lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ctx.dir))))
        src_root = source_root(key, lab_root)
        if src_root is None:
            raise FileNotFoundError(f"{_REPOS[key][0]} not found and could not be cloned")
        src = os.path.join(src_root, rel)
        commit = subprocess.run(["git", "-C", src_root, "rev-parse", "HEAD"], capture_output=True, text=True).stdout
        repo_name, url, _ = _REPOS[key]
        ctx.log(desc)
        ctx.input("source", {"repository": url, "path": rel, "commit": commit.strip(), "description": desc,
                             "papers": [f"{url}/blob/main/{rel}/{os.path.relpath(p, src)}"
                                        for p in sorted(glob.glob(os.path.join(src, "paper", "*.pdf")))]})
        with open(os.path.join(src, "results", "results.json")) as f:
            res = json.load(f)
        ctx.output("results", res)
        with ctx.stage("headline"):
            fn = _RUN[case]
            if case in ("heated_channel", "shock_train"):
                fn(ctx, res, src)
            else:
                fn(ctx, res)
        with ctx.stage("artifacts"):
            for f in sorted(glob.glob(os.path.join(src, "figures", "*.png")))[:12]:
                ctx.figure_file(f)
            ds = None
            for f in sorted(glob.glob(os.path.join(src, "results", "*.npz"))):
                if os.path.getsize(f) > 64 * 2 ** 20:
                    continue
                with np.load(f, allow_pickle=False) as z:
                    for k in z.files:
                        a = z[k]
                        if a.dtype.kind in "biuf" and a.size:
                            if ds is None:
                                ds = ctx.dataset("arrays", description=f"Arrays behind the figures of {repo_name}/{rel}")
                            ds.add(array=a, file=os.path.basename(f), key=k, case=case)
