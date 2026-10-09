"""Regime change or sensor noise? The five-family ensemble on a stream where 15 % of the readings are noisy (the
measured input and the measured response), with the current-case physics residual alone (left) and with a physics
check of the reading itself (right, ``measurement_check``): a physical state here is smooth, so energy at high
wavenumbers in the measured profile means the sensor, not the process.

Writes ``sensor_noise_comparison.gif`` (both runs side by side, the measured field drawn as the reference, so the
noisy readings are visible) and ``sensor_noise_timeline.png`` (model chosen on every case against the best model in
hindsight, noisy readings marked). Trains the five models once (about 3 minutes on a laptop CPU).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from compare_lookahead_gif import stitch
from five_model_families_gif import AD, REGIMES, random_case, train_experts, training_baseline

from pinneapple_physics.ensemble import PhysicsEnsemble
from pinneapple_physics.ensemble_viz import animate_ensemble


def physically_smooth(case, max_share=1e-4) -> bool:
    """True when the measured initial profile is NOT physically plausible (energy beyond the 4 Fourier modes the
    process has)."""
    f = np.abs(np.fft.rfft(case["u0"])) ** 2
    return bool(f[8:].sum() / f.sum() > max_share)


def noisy_stream(seed=7, cases_per_regime=16, share=0.15, amplitude=0.3):
    rng = np.random.default_rng(seed)
    cases = [random_case(rng, r) for r in REGIMES for _ in range(cases_per_regime)]
    truth = [AD.exact(c) for c in cases]
    n = len(cases)
    candidates = [i for i in range(n) if i % cases_per_regime not in (0, 1)]   # keep the regime changes clean
    noisy = np.zeros(n, bool)
    noisy[rng.choice(candidates, int(share * n), replace=False)] = True
    measured, refs = [], []
    for i, (c, t) in enumerate(zip(cases, truth, strict=True)):
        if noisy[i]:
            c = dict(c, u0=c["u0"] + rng.normal(0, amplitude * np.std(c["u0"]), c["u0"].shape))
            t = t + rng.normal(0, amplitude * np.std(t), t.shape)
        measured.append(c)
        refs.append(t)
    return cases, truth, measured, refs, noisy


def timeline(runs: dict, best: np.ndarray, names, noisy, cases_per_regime, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 1.9 * len(runs) + 0.7), sharex=True)
    colors = {n: plt.get_cmap("tab10")(i) for i, n in enumerate(names)}
    k = np.arange(len(best))
    for ax, (label, (run, err)) in zip(axes, runs.items(), strict=True):
        pick = np.array([names.index(a) for a in run.active])
        for i in np.nonzero(noisy)[0]:
            ax.axvspan(i - 0.45, i + 0.45, color="#ff5d5d", alpha=0.15, lw=0)
        ax.scatter(k, best, s=60, marker="s", facecolors="none", edgecolors="0.6", label="best model in hindsight")
        ax.scatter(k, pick, s=14, c=[colors[names[i]] for i in pick], label="model chosen")
        wrong_noisy = int(np.sum((pick != best) & noisy))
        ax.set_title(f"{label}: wrong model on {wrong_noisy} of {int(noisy.sum())} noisy readings, "
                     f"mean error against the true field {err:.3f}", fontsize=10, loc="left")
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for r in range(1, len(REGIMES)):
            ax.axvline(r * cases_per_regime - 0.5, color="0.8", ls="--", lw=0.8)
    h, lab = axes[0].get_legend_handles_labels()
    import matplotlib.patches as mpatches

    h.append(mpatches.Patch(color="#ff5d5d", alpha=0.3))
    lab.append("noisy reading")
    fig.legend(h, lab, fontsize=8, loc="upper right", ncol=3, frameon=False)
    axes[-1].set_xlabel("case (the regime changes at the dashed lines)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(out_dir=".", scale=1.0, cases_per_regime=16, frames_per_case=2, seed=7, dpi=56):
    out_dir = Path(out_dir)
    experts = train_experts(scale)
    names = [e.name for e in experts]
    cases, truth, measured, refs, noisy = noisy_stream(seed, cases_per_regime)
    base = dict(mode="select", residual_fn=AD.residual, residual_lookahead=10.0,
                residual_baseline=training_baseline(experts))
    runs = {}
    for label, extra in (("residual only", {}), ("+ physics check of the reading", {"measurement_check": physically_smooth})):
        run = PhysicsEnsemble(experts, **base, **extra).run(measured, refs, keep_predictions=True)
        err = float(np.mean([np.linalg.norm(p["prediction"] - t) / np.linalg.norm(t)
                             for p, t in zip(run.predictions, truth, strict=True)]))
        runs[label] = (run, err)
    clean_errors = np.array([[np.linalg.norm(e.predict(c) - t) / np.linalg.norm(t) for e in experts]
                             for c, t in zip(cases, truth, strict=True)])
    best = clean_errors.argmin(1)
    regimes = [(k * cases_per_regime, f"{r} regime") for k, r in enumerate(REGIMES)]
    with tempfile.TemporaryDirectory() as tmp:
        gifs = [animate_ensemble(run, refs, Path(tmp) / f"{i}.gif", x=AD.x, times=AD.t,
                                 frames_per_case=frames_per_case, regimes=regimes,
                                 title=f"{label} (error {err:.3f})", dpi=dpi)
                for i, (label, (run, err)) in enumerate(runs.items())]
        gif = stitch(gifs[0], gifs[1], out_dir / "sensor_noise_comparison.gif")
    png = timeline(runs, best, names, noisy, cases_per_regime, out_dir / "sensor_noise_timeline.png")
    for label, (run, err) in runs.items():
        pick = np.array([names.index(a) for a in run.active])
        print(f"{label:32s} error {err:.4f} | wrong model on noisy readings {int(np.sum((pick != best) & noisy))} "
              f"of {int(noisy.sum())} | flagged {int(run.suspect.sum())}")
    print("wrote", gif, png)
    return runs


if __name__ == "__main__":
    main()
