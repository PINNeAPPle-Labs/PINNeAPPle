"""Physics-informed networks on the electromagnetics presets with closed-form solutions (``em_*``): electrostatics
across a dielectric interface, magnetostatics of a round wire, a TM cavity mode and the skin effect.

The examples ``examples/electrodynamics/01..06`` train their own loops and commit only figures; this experiment
solves the same physics through the library (``pp.solve(problem, "pinn")``) and scores it against the exact field
(``pp.compare(..., reference="analytic")``) on 4096 points, so the error is measured, not read from a plot.

    python -m pinneapple_lab sweep em_pinn -g case=em_two_dielectric_capacitor,em_wire_az,em_tm_cavity,em_skin_effect
"""
from __future__ import annotations

import numpy as np

from ..spec import Experiment, register

CASES = {
    "em_two_dielectric_capacitor": "Electrostatics: parallel plates with two dielectrics (Gauss across the interface; "
                                   "exact D uniform, E_i = D / eps_i)",
    "em_wire_az": "Magnetostatics: A_z of a round conductor (exact from Ampere's law)",
    "em_tm_cavity": "Electrodynamics: TMz standing wave in a PEC square cavity (exact Ez = sin sin cos)",
    "em_skin_effect": "Magneto-quasistatics: eddy currents into a conductor (exact A0 exp(-(1 + j) x / delta))",
}


@register
class EMPinn(Experiment):
    name = "em_pinn"
    check_labels = {
        "rel_l2_Az_vs_closed_form": {"en": "A_z vs Amp\u00e8re's law (exact)", "pt": "A_z vs lei de Amp\u00e8re (exata)"},
        "rel_l2_Ar_vs_closed_form": {"en": "skin effect vs exact solution", "pt": "efeito pelicular vs solu\u00e7\u00e3o exata",
                                     "also": ["rel_l2_Ai_vs_closed_form"]},
        "rel_l2_phi_vs_closed_form": {"en": "potential vs exact solution", "pt": "potencial vs solu\u00e7\u00e3o exata"},
        "rel_l2_Ez_vs_closed_form": {"en": "TM mode vs exact solution", "pt": "modo TM vs solu\u00e7\u00e3o exata",
                                     "also": ["rel_l2_Hx_vs_closed_form", "rel_l2_Hy_vs_closed_form"]},
    }
    version = "1"
    description = ("Physics-informed networks on the electromagnetics presets with closed-form solutions "
                   "(electrostatics, magnetostatics, a TM cavity, the skin effect), scored against the exact field.")
    tags = ["pinn", "electromagnetics", "pde"]
    references = ["D. J. Griffiths, Introduction to Electrodynamics, ex. 4.7 (two-dielectric capacitor)",
                  "D. M. Pozar, Microwave Engineering, sec. 6.3 (rectangular cavity)"]
    limitations = ["one seed and one budget per case; the error is that of this budget, not the method's best"]
    case_param = "case"
    case_descriptions = CASES
    params = {"case": "em_wire_az", "epochs": 3000, "n_collocation": 2048, "seed": 0, "tolerance": 0.05}
    space = {"case": list(CASES)}

    def run(self, ctx):
        import pinneapple as pp
        p = ctx.params
        prob = pp.PhysicalProblem.from_preset(p["case"])
        ctx.input("problem", {"preset": p["case"], "fields": list(prob.fields), "coords": list(prob.coords)})
        with ctx.stage("train"):
            sol = pp.solve(prob, "pinn", epochs=int(p["epochs"]), n_collocation=int(p["n_collocation"]),
                           seed=int(p["seed"]))
        with ctx.stage("score"):
            cmp = pp.compare(prob, [sol], reference="analytic").to_dict()
        row = cmp["rows"][0]
        for f, e in row["metrics"]["relative_l2"].items():
            ctx.metric(f"rel_l2_{f}", e)
            ctx.check(f"rel_l2_{f}_vs_closed_form", value=e, max=float(p["tolerance"]), kind="reference",
                      label=f"{f} vs closed form", detail=f"{cmp['n_points']} points, exact solution of the preset; "
                                                          f"tolerance {p['tolerance']:g}")
        ctx.metric("train_seconds", row["wall_time_s"])
        ctx.metric("n_parameters", row["info"].get("n_parameters"))
        hist = getattr(sol, "history", None) or {}
        loss = hist.get("loss") if isinstance(hist, dict) else None
        if loss:
            ctx.check("training_loss_decreased", passed=float(loss[-1]) < float(loss[0]), kind="sanity")
        self._figure(ctx, prob, sol, pp)

    @staticmethod
    def _figure(ctx, prob, sol, pp):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        ex = pp.solve(prob, "analytic")
        b = prob.domain_bounds
        names = list(prob.coords)
        lo, hi = (np.array([b[c][i] for c in names], float) for i in (0, 1))
        f0 = list(prob.fields)[0]
        if len(names) == 1:
            x = np.linspace(lo[0], hi[0], 400)[:, None]
            yp, ye = np.asarray(sol.predict(x)), np.asarray(ex.predict(x))
            fig, ax = plt.subplots(figsize=(6, 3.4))
            ax.plot(x[:, 0], ye[:, 0], "k-", lw=2, label="exact")
            ax.plot(x[:, 0], yp[:, 0], "--", c="#d62728", label="PINN")
            ax.set_xlabel(names[0])
            ax.set_ylabel(f0)
            ax.legend()
        else:
            n = 120
            g = np.meshgrid(*[np.linspace(lo[i], hi[i], n) for i in range(2)], indexing="ij")
            X = np.stack([g[0].ravel(), g[1].ravel()] + [np.full(n * n, lo[i]) for i in range(2, len(names))], 1)
            yp = np.asarray(sol.predict(X))[:, 0].reshape(n, n)
            ye = np.asarray(ex.predict(X))[:, 0].reshape(n, n)
            fig, ax = plt.subplots(1, 3, figsize=(10, 3.2), constrained_layout=True)
            ext = [lo[0], hi[0], lo[1], hi[1]]
            vmin, vmax = float(ye.min()), float(ye.max())
            for a, z, t in ((ax[0], yp, "PINN"), (ax[1], ye, "exact")):
                im = a.imshow(z.T, origin="lower", extent=ext, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")
                a.set_title(f"{t}: {f0}")
                a.set_xlabel(names[0])
            fig.colorbar(im, ax=ax[:2], shrink=0.85)
            im = ax[2].imshow(np.abs(yp - ye).T, origin="lower", extent=ext, cmap="magma", aspect="auto")
            ax[2].set_title("|error|")
            fig.colorbar(im, ax=ax[2], shrink=0.85)
        ctx.figure(f"{ctx.params['case']}_pinn_vs_exact", fig)
