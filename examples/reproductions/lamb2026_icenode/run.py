"""Reproduce Lamb et al. (2026), "Discovering how ice crystals grow using neural ODEs and symbolic regression".

    python examples/reproductions/lamb2026_icenode/run.py --quick          # minutes: reduced-scale, labelled as such
    python examples/reproductions/lamb2026_icenode/run.py                  # the paper's 500 epochs on both models
    python examples/reproductions/lamb2026_icenode/run.py --out /tmp/icenode

What runs here, in the paper's order:

1. the two parameterised baselines of Table 1 -- no surface kinetics (``G = G_c``) and the Nelson & Baker (1996)
   deposition coefficient -- integrated with the same RK4 solver;
2. the strongly- and weakly-constrained neural ODEs of Eqs. 3-4, trained on the 286 public 1 Hz mass-ratio series
   over the 500 s window and scored on t < 500 s (training) and t < 1000 s (extrapolation), Table 1;
3. the same strongly-constrained model trained on *synthetic* series with a known alpha (Nelson & Baker), to check
   that the method recovers a known function (the paper's Figure 2), before any claim about the real data;
4. symbolic regression on the trained weakly-constrained network (the paper's Figure 1c, Table S2), giving an
   explicit closed form for G, and the paper's own Eq. 6 scored on the same data through the same solver;
5. the paper's Eq. 6 integrated as a growth law, i.e. the paper's proposed model applied to our 286 series.

Not attempted: the AIDA Aerosol and Cloud Chamber comparison (the paper's Figure 4), which needs the IsoCloud
archive and a bin-microphysics parcel model; TARGETS.md records that and why.

Everything is written to ``--out`` (default this folder): ``results/*.json`` with every number quoted in the
report, ``figures/*.png``, and the metrics registered in the folder's experiment manifest.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import bel  # noqa: E402
import node  # noqa: E402
import symbolic  # noqa: E402
import synthetic  # noqa: E402

#: Table 1 of Lamb et al. (2026), frozen in TARGETS.md: MSE loss over all series, and the count of experiments on
#: which each model performs best (t < 500 s, t < 1000 s).
TABLE1 = {
    "none": {"sse_short": 41396, "sse_long": 222750, "best_short": 62, "best_long": 55},
    "nelson_baker": {"sse_short": 40751, "sse_long": 220930, "best_short": 29, "best_long": 31},
    "strong": {"sse_short": 30714, "sse_long": 203645, "best_short": 61, "best_long": 62},
    "weak": {"sse_short": 16495, "sse_long": 184723, "best_short": 138, "best_long": 142},
}
ORDER = ("none", "nelson_baker", "strong", "weak")

#: Eq. 6 of the paper (its discovered equation 8 of Table S2), in SI units.
EQ6 = {"a0": 688.267, "a1": 1.3153, "a2": 0.85601, "a3": 2.6606e-12, "a4": 0.1123e-9}


def paper_eq6(r, T, p, m):
    """``G = a0 Gc^a1 (a2 + a3/m)^-1 + a4`` (paper Eq. 6) as a kinetics callable for the same RK4 solver."""
    Gc = node._Gc(r, T, p)
    return EQ6["a0"] * Gc ** EQ6["a1"] / (EQ6["a2"] + EQ6["a3"] / m.clamp_min(1e-20)) + EQ6["a4"]


def eq6_G(Gc, m):
    """The same equation on numpy arrays, for the G samples."""
    return EQ6["a0"] * np.asarray(Gc) ** EQ6["a1"] / (EQ6["a2"] + EQ6["a3"] / np.maximum(np.asarray(m), 1e-20)) \
        + EQ6["a4"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--epochs", type=int, default=500, help="Adam epochs per neural ODE (the paper's 500)")
    ap.add_argument("--synthetic-epochs", type=int, default=500)
    ap.add_argument("--dt", type=float, default=2.0, help="RK4 step in s (the paper leaves it unstated)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="20 epochs, one seed: plumbing, not a result")
    ap.add_argument("--no-synthetic", action="store_true")
    ap.add_argument("--model", choices=list(ORDER) + ["eq6", "synthetic"], default=None,
                    help="train one model into results/checkpoints and stop (runs can be split across processes)")
    ap.add_argument("--out", default=HERE)
    args = ap.parse_args(argv)
    epochs = 20 if args.quick else args.epochs
    syn_epochs = 20 if args.quick else args.synthetic_epochs
    out = os.path.abspath(args.out)
    os.makedirs(os.path.join(out, "results"), exist_ok=True)
    os.makedirs(os.path.join(out, "figures"), exist_ok=True)
    cfg = node.TrainConfig(epochs=epochs, dt=args.dt, seed=args.seed)

    print("1/5 data: levitation diffusion chamber archives")
    data = bel.load()
    print(f"    {len(data)} series, T {data.T.min():.1f}-{data.T.max():.1f} K, S_i {data.Si.min():.3f}-"
          f"{data.Si.max():.3f}, {data.length().sum()} samples")

    print("2/5 baselines and neural ODEs (Table 1)")
    ckpt_dir = os.path.join(out, "results", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    models, scores, preds, rhss = {}, {}, {}, {}
    if args.model == "synthetic":
        _train_or_load("synthetic", synthetic.make(data, seed=args.seed),
                       node.TrainConfig(epochs=syn_epochs, dt=args.dt, t_max=500.0, seed=args.seed),
                       ckpt_dir, args, kinetics="strong")
        return 0
    todo = [args.model] if args.model else list(ORDER)
    for kin in todo:
        print(f"    {kin} ...", flush=True)
        rhs, hist, pred = _train_or_load(kin, data, cfg, ckpt_dir, args)
        s = node.score(pred, data, dt=args.dt)
        models[kin], scores[kin], preds[kin], rhss[kin] = hist, s, pred, rhs
        print(f"      sse(500s) {s['sse_short']:9.0f}  sse(1000s) {s['sse_long']:9.0f}   "
              f"[paper {TABLE1[kin]['sse_short']} / {TABLE1[kin]['sse_long']}]  {hist.seconds:.0f} s")
    if args.model:                                    # training only: the assembly run picks the checkpoint up
        return 0
    for kin in ORDER:
        if kin not in preds:
            raise SystemExit(f"{kin} has no checkpoint in {ckpt_dir}: run `--model {kin}` first")

    print("    the paper's Eq. 6 as a growth law ...", flush=True)
    rhs6, hist6, pred6 = _train_or_load("eq6", data, cfg, ckpt_dir, args)
    s6 = node.score(pred6, data, dt=args.dt)
    models["eq6"], scores["eq6"], preds["eq6"] = hist6, s6, pred6

    # G along the measured trajectories, for the symbolic step and its figure
    samples = node.sample_G(rhss["weak"], data)
    samples["G_eq6"] = eq6_G(samples["Gc"], samples["m"])

    # which model wins an individual experiment, on both windows
    short = np.stack([scores[k]["per_experiment_short"] for k in ORDER])
    long = np.stack([scores[k]["per_experiment_long"] for k in ORDER])
    best_short = {k: int(np.sum(np.argmin(short, axis=0) == i)) for i, k in enumerate(ORDER)}
    best_long = {k: int(np.sum(np.argmin(long, axis=0) == i)) for i, k in enumerate(ORDER)}
    for k in ORDER:
        print(f"    best on {best_short[k]:3d} experiments (t<500 s), {best_long[k]:3d} (t<1000 s)   "
              f"[paper {TABLE1[k]['best_short']} / {TABLE1[k]['best_long']}]")

    print("3/5 synthetic data with a known alpha (paper Figure 2)")
    synth_result = {}
    if not args.no_synthetic:
        syn = synthetic.make(data, seed=args.seed)
        rhs_syn, syn_hist, _ = _train_or_load("synthetic", syn, node.TrainConfig(epochs=syn_epochs, dt=args.dt,
                                                                                 t_max=500.0, seed=args.seed),
                                              ckpt_dir, args, kinetics="strong")
        import torch
        with torch.no_grad():
            alpha_nn = rhs_syn.net(rhs_syn._x_env).squeeze(-1).numpy()
        rel = np.abs(alpha_nn / syn.alpha - 1.0)
        synth_result = {"n": int(len(syn)), "median_rel_error": float(np.median(rel)),
                        "mean_rel_error": float(np.mean(rel)), "epochs": syn_epochs,
                        "alpha_nn": alpha_nn, "alpha_true": syn.alpha,
                        "noise_std_median": float(np.median(np.nanstd(syn.noise, axis=1)))}
        print(f"    median |alpha_NN/alpha_NB - 1| = {synth_result['median_rel_error']:.3f}, "
              f"mean {synth_result['mean_rel_error']:.3f}")

    print("4/5 symbolic regression on the trained networks")
    sym_result = _symbolic(samples, args)
    fig_paths = _figures(out, data, models, preds, samples, synth_result, args.dt)

    print("5/5 results")
    best_model = min(("strong", "weak", "eq6"), key=lambda k: scores[k]["sse_short"])
    result = {
        "paper": "Lamb et al. 2026, Science Advances 12(39) eaee6174 (arXiv:2510.17935)",
        "data": {"n_series": len(data), "n_samples": int(data.length().sum()),
                 "T_K": [float(data.T.min()), float(data.T.max())], "Si": [float(data.Si.min()), float(data.Si.max())],
                 "sources": {str(k): int(v) for k, v in zip(*np.unique(data.source, return_counts=True), strict=True)},
                 "note": "286 of the paper's 290: its 4 raw-voltage 'cold' series are not in the archives"},
        "config": {"epochs": epochs, "dt": args.dt, "seed": args.seed, "quick": args.quick,
                   "synthetic_epochs": syn_epochs},
        "table1": {k: {"sse_short": scores[k]["sse_short"], "sse_long": scores[k]["sse_long"],
                       "mse_short": scores[k]["mse_short"], "mse_long": scores[k]["mse_long"],
                       "best_short": best_short.get(k), "best_long": best_long.get(k),
                       "paper": TABLE1.get(k), "seconds": models[k].seconds,
                       "losses": models[k].losses[::max(1, len(models[k].losses) // 200)]}
                   for k in list(ORDER) + ["eq6"]},
        "synthetic": {k: v for k, v in synth_result.items() if not k.startswith("alpha")},
        "symbolic": sym_result,
        "best_model": best_model,
        "figures": [os.path.relpath(p, out) for p in fig_paths],
    }
    path = os.path.join(out, "results", f"icenode_seed{args.seed}.json")
    json.dump(_plain(result), open(path, "w"), indent=1)
    print(f"    wrote {os.path.relpath(path, out)} and {len(fig_paths)} figures")
    _manifest(result, out)
    return 0


# ------------------------------------------------------------------------------------------- train or reuse
def _ckpt_path(ckpt_dir, name, cfg):
    return os.path.join(ckpt_dir, f"{name}_ep{cfg.epochs}_dt{cfg.dt:g}_t{cfg.t_max:g}_seed{cfg.seed}.npz")


def _build_rhs(kinetics, data, net=None):
    import torch
    to = lambda a: torch.as_tensor(np.asarray(a), dtype=torch.float32)                     # noqa: E731
    return node.IceGrowthODE(Si=to(data.Si), T=to(data.T), P=to(data.P), m0=to(data.m0),
                             kinetics=kinetics, net=net)


def _train_or_load(name, data, cfg, ckpt_dir, args, kinetics=None):
    """Train a model once and reuse it from ``results/checkpoints`` afterwards.

    Each of the four Table 1 models takes tens of minutes on one CPU thread, so a full run can be split across
    processes (``--model strong``, ``--model weak``) and the assembly run then reads their checkpoints instead of
    retraining. A checkpoint records the prediction, the loss history and (for a network) its weights.
    """
    import torch
    kin = kinetics if kinetics is not None else (paper_eq6 if name == "eq6" else name)
    path = _ckpt_path(ckpt_dir, name, cfg)
    net = None
    if kin in ("strong", "weak"):
        net = node.KineticsMLP(n_in=2 if kin == "strong" else 3, out_scale=1.0 if kin == "strong" else 2.0,
                               width=cfg.width, seed=cfg.seed)
    if os.path.exists(path):
        with np.load(path) as z:
            pred = z["pred"]
            hist = node.TrainHistory(losses=z["losses"].tolist(), epochs=int(z["epochs"]),
                                     seconds=float(z["seconds"]), config={"kinetics": name, **cfg.__dict__})
        if net is not None and os.path.exists(path.replace(".npz", ".pt")):
            net.load_state_dict(torch.load(path.replace(".npz", ".pt"), weights_only=True))
        print(f"      (from {os.path.basename(path)})", flush=True)
        return _build_rhs(kin, data, net), hist, pred
    rhs, hist, pred = node.train(data, kin, cfg, verbose=not args.quick and name in ("strong", "weak"))
    np.savez_compressed(path, pred=pred.astype(np.float32), losses=np.asarray(hist.losses, dtype=float),
                        epochs=hist.epochs, seconds=hist.seconds)
    if net is not None:
        torch.save(rhs.net.state_dict(), path.replace(".npz", ".pt"))
    return rhs, hist, pred


# ------------------------------------------------------------------------------------------------- symbolic step
def _symbolic(samples, args) -> dict:
    """Distil the trained weakly-constrained network's ``G`` into a closed form and score the paper's Eq. 6."""
    G, Gc, m = samples["G"], samples["Gc"], samples["m"]
    ratio = G / Gc
    x = {"m": m / 1e-11, "Si": samples["Si"], "T": (samples["T"] - 273.15) / 10.0, "Gc": Gc / 1.8e-9}
    front = symbolic.discover(x, ratio, generations=60, population=240, seed=args.seed, parsimony=0.08,
                              max_nodes=21)
    best = symbolic.refine(front.best_within(1.10), x, ratio)
    paper = eq6_G(Gc, m)
    out = {
        "target": "G/Gc from the weakly-constrained network, features m/1e-11 kg, S_i, (T-273.15)/10, Gc/1.8e-9",
        "expression": best.render(6), "nodes": best.nodes,
        "rmse_ratio": best.rmse(ratio, x), "std_ratio": float(np.std(ratio)),
        "front": [{"nodes": e.nodes, "rmse": rr, "expression": e.render(5)} for e, rr in front.pareto][:12],
        "seconds": round(front.seconds, 1),
        "paper_eq6": {"expression": "G = a0 Gc^a1 (a2 + a3/m)^-1 + a4", "constants": EQ6,
                      "rmse_ratio": float(np.sqrt(np.mean((paper / Gc - ratio) ** 2))),
                      "rmse_G": float(np.sqrt(np.mean((paper - G) ** 2)))},
        "g_spread": float(np.std(G)), "n_samples": int(G.size),
    }
    print(f"    discovered {best.render(5)} ({best.nodes} nodes, rmse {out['rmse_ratio']:.4f} of a spread of "
          f"{out['std_ratio']:.4f}); paper Eq. 6 rmse {out['paper_eq6']['rmse_ratio']:.4f}")
    return out


# ------------------------------------------------------------------------------------------------------- figures
def _figures(out, data, models, preds, samples, synth, dt=1.0):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths = []

    # --- mass-ratio time series, the paper's Figure S5 analog
    t = data.time
    order = np.argsort(-np.array(data.length()))[:4]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, i in zip(axes.ravel(), order, strict=True):
        n = int(data.length()[i])
        ax.plot(t[i, :n], data.mass_ratio[i, :n], "k.", ms=2.5, label="measured")
        for k, style in (("none", ":"), ("nelson_baker", "--"), ("strong", "-."), ("weak", "-")):
            y = preds[k][i]
            ax.plot(np.arange(y.size) * dt, y, style, lw=1.4, label=k)      # the RK4 grid, step dt
        ax.set_title(f"experiment {i}: T={data.T[i] - 273.15:.1f} C, S_i={data.Si[i]:.3f}", fontsize=9)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("m/m0")
    axes[0, 0].legend(fontsize=7, frameon=False)
    fig.suptitle("Depositional ice growth: measured mass ratio against the four models")
    fig.tight_layout()
    paths.append(os.path.join(out, "figures", "mass_ratio_fits.png"))
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    # --- loss curves
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for k in ("strong", "weak"):
        ax.semilogy(models[k].losses, lw=1.4, label=k)
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (sum of squared errors)")
    ax.set_title("Neural ODE training on the chamber experiments")
    ax.legend(frameon=False)
    fig.tight_layout()
    paths.append(os.path.join(out, "figures", "training_loss.png"))
    fig.savefig(paths[-1], dpi=130)
    plt.close(fig)

    # --- learned G against Gc, the paper's Figure 3a analog
    if samples:
        fig, ax = plt.subplots(figsize=(6.5, 4.4))
        ax.loglog(samples["m"], samples["G"], ".", ms=2, alpha=0.30, label="weak NODE, G")
        ax.loglog(samples["m"], samples["Gc"], ".", ms=2, alpha=0.30, label="G_c (continuum)")
        ax.loglog(samples["m"], samples["G_eq6"], ".", ms=2, alpha=0.55, label="paper Eq. 6")
        ax.set_xlabel("crystal mass (kg)")
        ax.set_ylabel("transfer coefficient G (s/m)")
        ax.set_title("Transfer coefficient learned from the experiments")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        paths.append(os.path.join(out, "figures", "learned_G.png"))
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)

    # --- synthetic alpha, the paper's Figure 2 analog
    if synth:
        fig, ax = plt.subplots(figsize=(4.8, 4.4))
        ax.scatter(synth["alpha_true"], synth["alpha_nn"], s=14, alpha=0.8)
        lim = [0, 1.02]
        ax.plot(lim, lim, "k--", lw=1, label="alpha_NN = alpha_NB")
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xlabel("alpha that generated the synthetic series (Nelson & Baker)")
        ax.set_ylabel("alpha learned by the strong NODE")
        ax.set_title(f"synthetic data: median |error| {synth['median_rel_error']:.1%}")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        paths.append(os.path.join(out, "figures", "alpha_synthetic.png"))
        fig.savefig(paths[-1], dpi=130)
        plt.close(fig)
    return paths


# ----------------------------------------------------------------------------------------------- bookkeeping
def _plain(obj):
    """JSON-safe copy: numpy scalars to Python, arrays to lists."""
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _plain(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj


def _manifest(result, out):
    """Register the metrics in the folder manifest (``python -m pinneapple_lab register .`` regenerates it)."""
    try:
        from pinneapple_lab import experiment
    except Exception:                                                          # pragma: no cover - optional
        return
    try:
        with experiment("lamb2026_icenode", title="Ice crystals grow: neural ODE and symbolic regression",
                        domain="atmospheric physics", methods=["neural ODE", "symbolic regression", "RK4"],
                        params=result["config"], out_dir=out, register=False) as ex:
            for k in ORDER + ("eq6",):
                row = result["table1"][k]
                ex.log_metric(f"sse_500s_{k}", round(row["sse_short"], 1), "",
                              f"sum of squared m/m0 errors over {result['data']['n_series']} series, t<500 s")
                ex.log_metric(f"sse_1000s_{k}", round(row["sse_long"], 1), "", "same, t<1000 s")
            ex.log_metric("synthetic_alpha_median_rel_error", round(result["synthetic"].get("median_rel_error", -1), 4),
                          "", "median |alpha_NN/alpha_NB - 1| on synthetic series")
            ex.log_metric("symbolic_rmse_ratio", round(result["symbolic"]["rmse_ratio"], 5), "",
                          "rmse of the discovered closed form for G/Gc")
            ex.log_metric("paper_eq6_rmse_ratio", round(result["symbolic"]["paper_eq6"]["rmse_ratio"], 5), "",
                          "rmse of the paper's Eq. 6 on the same G samples")
        from pinneapple_lab import manifest as M
        m = M.load(out)                                     # fill in what the run itself knows
        m.description = ("Lamb et al. (2026): the depositional ice growth law learned from 286 levitation diffusion "
                         "chamber mass-ratio series with a neural ODE, the strongly- and weakly-constrained models of "
                         "its Table 1, and a closed form for the transfer coefficient from symbolic regression.")
        m.modules = ["pinneapple_physics.cloud"]
        m.scripts = ["bel.py", "node.py", "symbolic.py", "synthetic.py", "run.py"]
        m.files["figures"] = [f for f in result["figures"] if os.path.exists(os.path.join(out, f))]
        m.files["data"] = [f"results/icenode_seed{result['config']['seed']}.json"]
        m.links = {"targets": os.path.join(os.path.relpath(os.path.abspath(out), REPO), "TARGETS.md"),
                   "readme": os.path.join(os.path.relpath(os.path.abspath(out), REPO), "README.md")}
        m.references = ["Lamb et al., Science Advances 12(39) eaee6174 (2026), arXiv:2510.17935",
                        "Harrington et al., JAS 76, 1609 (2019)", "Nelson & Baker, JAS 53, 2853 (1996)",
                        "Murphy & Koop, QJRMS 131, 1539 (2005)",
                        "Pokrifka et al. 2018 doi:10.26208/z7bf-nq20, 2023 doi:10.26208/htw5-q166"]
        M.save(m, out)
    except Exception as e:                                                     # pragma: no cover - optional
        print(f"    (manifest not written: {e})")


if __name__ == "__main__":
    raise SystemExit(main())
