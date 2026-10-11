"""PDE experiments solved with the library's solvers, checked against exact solutions."""
from __future__ import annotations

import numpy as np

from ..spec import Experiment, register


@register
class HeatXTFC(Experiment):
    name = "heat_xtfc"
    version = "1"
    description = ("1D heat equation u_t = alpha u_xx on [0,1] x [0,1], u0 = sin(pi x), solved by X-TFC "
                   "(pinneapple_simulation.numerical_solvers.xtfc_pde) vs exp(-alpha pi^2 t) sin(pi x).")
    tags = ["pde", "xtfc", "verification"]
    limitations = ["at alpha = 1 the smaller bases miss the accuracy target (failed runs kept)"]
    params = {"alpha": 0.1, "n_basis": 60, "n_collocation": 25, "activation": "tanh", "seed": 0}
    space = {"alpha": ("log", 0.01, 1.0), "n_basis": [20, 40, 60, 100], "n_collocation": [10, 20, 30],
             "activation": ["tanh", "sigmoid", "sin"]}

    def run(self, ctx):
        from pinneapple_simulation.numerical_solvers.xtfc_pde import solve_xtfc_pde
        p = ctx.params
        with ctx.stage("solve"):
            sol = solve_xtfc_pde("heat_1d", param=p["alpha"], n_basis=p["n_basis"], n_collocation_x=p["n_collocation"],
                                 n_collocation_t=p["n_collocation"], activation=p["activation"], seed=p["seed"])
        x = np.linspace(0, 1, 101)
        t = np.linspace(0, 1, 51)
        X, T = np.meshgrid(x, t, indexing="ij")
        u = np.asarray(sol["predict"](X.ravel(), T.ravel())).reshape(X.shape)
        ref = np.exp(-p["alpha"] * np.pi ** 2 * T) * np.sin(np.pi * X)
        err = float(np.sqrt(np.mean((u - ref) ** 2)) / np.sqrt(np.mean(ref ** 2)))
        ctx.output("u", u.astype(np.float32))
        ctx.metric("rel_l2_error", err)
        ctx.metric("max_abs_error", float(np.max(np.abs(u - ref))))
        ctx.check("rel_l2_below_1e-3", value=err, max=1e-3, detail="vs the exact solution", kind="reference")
        ds = ctx.dataset("fields", description="u(x, t) of the heat equation on a 101 x 51 grid", units={"u": "-"})
        ds.add(u=u.astype(np.float32), u_exact=ref.astype(np.float32), alpha=float(p["alpha"]))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(7, 2.6))
        ax[0].imshow(u.T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="inferno")
        ax[0].set_title("X-TFC u(x, t)", fontsize=9)
        im = ax[1].imshow(np.abs(u - ref).T, origin="lower", extent=[0, 1, 0, 1], aspect="auto", cmap="viridis")
        ax[1].set_title(f"|error|, rel L2 {err:.1e}", fontsize=9)
        fig.colorbar(im, ax=ax[1])
        ctx.figure("solution", fig)
