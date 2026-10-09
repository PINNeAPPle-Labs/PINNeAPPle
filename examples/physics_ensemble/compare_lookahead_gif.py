"""Side by side: the five-family ensemble reacting only to past errors (left) and with the current-case physics
residual against each model's own history (right, ``residual_lookahead``), on the same stream of cases.

Writes ``lookahead_comparison.gif`` (both animations stitched frame by frame) and ``lookahead_timeline.png`` (the
model chosen on every case by each version against the best model in hindsight). Trains the five models once
(about 3 minutes on a laptop CPU).
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from five_model_families_gif import AD, REGIMES, random_case, train_experts

from pinneapple_physics.ensemble import PhysicsEnsemble
from pinneapple_physics.ensemble_viz import animate_ensemble


def stitch(left: Path, right: Path, out: Path, fps: int = 10) -> Path:
    """Two GIFs with the same number of frames → one, side by side."""
    from PIL import Image, ImageSequence

    a = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(left))]
    b = [f.convert("RGB") for f in ImageSequence.Iterator(Image.open(right))]
    frames = []
    for fa, fb in zip(a, b, strict=True):
        im = Image.new("RGB", (fa.width + fb.width, max(fa.height, fb.height)), "white")
        im.paste(fa, (0, 0))
        im.paste(fb, (fa.width, 0))
        frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=int(1000 / fps), loop=0, optimize=True)
    return out


def timeline(runs: dict, oracle: np.ndarray, names: list, cases_per_regime: int, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(runs), 1, figsize=(10, 1.9 * len(runs) + 0.6), sharex=True)
    colors = {n: plt.get_cmap("tab10")(i) for i, n in enumerate(names)}
    k = np.arange(len(oracle))
    for ax, (label, run) in zip(axes, runs.items(), strict=True):
        pick = np.array([names.index(a) for a in run.active])
        ax.scatter(k, oracle, s=60, marker="s", facecolors="none", edgecolors="0.6", label="best model in hindsight")
        ax.scatter(k, pick, s=14, c=[colors[names[i]] for i in pick], label="model chosen")
        wrong = int(np.sum(pick != oracle))
        ax.set_title(f"{label}: wrong model on {wrong} of {len(k)} cases, mean error {np.nanmean(run.ensemble_errors):.3f}",
                     fontsize=10, loc="left")
        ax.set_yticks(range(len(names)), names, fontsize=8)
        for r in range(1, len(REGIMES)):
            ax.axvline(r * cases_per_regime - 0.5, color="0.8", ls="--", lw=0.8)
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, fontsize=8, loc="upper right", ncol=2, frameon=False)
    axes[-1].set_xlabel("case (the regime changes at the dashed lines)")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(out_dir=".", scale=1.0, cases_per_regime=16, frames_per_case=2, seed=7, lookahead=3.0, dpi=56):
    out_dir = Path(out_dir)
    experts = train_experts(scale)
    names = [e.name for e in experts]
    rng = np.random.default_rng(seed)
    cases = [random_case(rng, r) for r in REGIMES for _ in range(cases_per_regime)]
    refs = [AD.exact(c) for c in cases]
    runs = {"past errors only": PhysicsEnsemble(experts, mode="select").run(cases, refs, keep_predictions=True),
            "+ current-case residual": PhysicsEnsemble(experts, mode="select", residual_fn=AD.residual,
                                                       residual_lookahead=lookahead).run(cases, refs,
                                                                                         keep_predictions=True)}
    oracle = np.nanargmin(runs["past errors only"].errors, axis=1)
    regimes = [(k * cases_per_regime, f"{r} regime") for k, r in enumerate(REGIMES)]
    with tempfile.TemporaryDirectory() as tmp:
        gifs = [animate_ensemble(run, refs, Path(tmp) / f"{i}.gif", x=AD.x, times=AD.t, frames_per_case=frames_per_case,
                                 regimes=regimes, title=f"{label} (error {np.nanmean(run.ensemble_errors):.3f})", dpi=dpi)
                for i, (label, run) in enumerate(runs.items())]
        gif = stitch(gifs[0], gifs[1], out_dir / "lookahead_comparison.gif")
    png = timeline(runs, oracle, names, cases_per_regime, out_dir / "lookahead_timeline.png")
    for label, run in runs.items():
        print(f"{label:24s} mean error {np.nanmean(run.ensemble_errors):.4f} | wrong model on "
              f"{int(np.sum(np.array([names.index(a) for a in run.active]) != oracle))} of {len(cases)} cases")
    print("wrote", gif, png)
    return runs


if __name__ == "__main__":
    main()
