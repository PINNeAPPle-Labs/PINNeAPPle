"""Adaptive forecasting on a series that changes regime: seasonal -> trend -> random walk.

Prints the error of the ensemble, of every model and of the best single model in hindsight, shows when the leading
model changes, and plots the forecasts, the interval and the model weights (adaptive_forecasting.png).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pinneapple_systems.time_series import AdaptiveForecaster, default_experts


def make_series(n=900, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    y = np.empty(n)
    a, b = n // 3, 2 * n // 3
    y[:a] = 10 + 5 * np.sin(2 * np.pi * t[:a] / 12) + rng.normal(0, 0.5, a)
    y[a:b] = y[a - 1] + 0.4 * np.arange(1, b - a + 1) + rng.normal(0, 0.5, b - a)
    y[b:] = y[b - 1] + np.cumsum(rng.normal(0, 1.5, n - b))
    return y


def main(out="adaptive_forecasting.png"):
    y = make_series()
    run = AdaptiveForecaster(default_experts(season_length=12), horizon=3).run(y, start=60)
    err = run.errors(h=1)
    for name in sorted(err, key=lambda k: err[k]["mae"]):
        print(f"{name:26s} MAE {err[name]['mae']:.3f}")
    print("interval coverage (90 % target):", round(run.coverage(1), 3))
    print("leading model changes (first 10):", run.switches()[:10])

    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    o = run.origins + 1
    ax[0].plot(np.arange(len(y)), y, "k", lw=0.8, label="observed")
    ax[0].plot(o, run.forecast[:, 0], "C1", lw=0.8, label="adaptive forecast (h = 1)")
    ax[0].fill_between(o, run.lower[:, 0], run.upper[:, 0], color="C1", alpha=0.2, label="90 % interval")
    ax[0].legend(loc="upper left")
    top = np.argsort(-run.weights.mean(0))[:5]
    ax[1].stackplot(run.origins, run.weights[:, top].T, labels=[run.expert_names[i] for i in top])
    ax[1].set(ylabel="weight (h = 1)", xlabel="time")
    ax[1].legend(loc="upper left", fontsize=8)
    for x in (300, 600):
        for a_ in ax:
            a_.axvline(x, color="grey", ls=":")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    return err


if __name__ == "__main__":
    main()
