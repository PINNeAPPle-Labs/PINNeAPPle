"""Reduced-order models on the lab's own solvers (``pinneapple_neural.architectures.rom``).

``rom_study`` runs one case:

* ``cylinder_wake``   -- vortex shedding behind a cylinder at Re 100 (lattice Boltzmann). POD of the velocity
                         fluctuations (the travelling-wave mode pair), DMD (the shedding frequency from the spectrum
                         of the linear operator) and Operator Inference (quadratic latent dynamics) forecasting the
                         wake beyond the training window, against persistence.
* ``beam_parametric`` -- a parametric ROM of the 3-D solid FEM: a cantilever whose height, width and load direction
                         vary; POD of the displacement and von Mises fields with the coefficients regressed by RBF
                         and Gaussian processes, with and without the amplitude / shape split, against the nearest
                         training design; error on unseen designs and the speed-up over the solver.

    python -m pinneapple_lab sweep rom_study -g case=cylinder_wake,beam_parametric
"""
from __future__ import annotations

import math
import time

import numpy as np

from ..spec import Experiment, register

CASES = {
    "cylinder_wake": "Vortex shedding behind a cylinder at Re 100 (LBM): POD modes, DMD frequency and OpInf/DMD "
                     "forecasts of the wake against persistence.",
    "beam_parametric": "Parametric ROM of a 3-D cantilever (height, width, load direction): POD + RBF / GPR on 80 "
                       "FEM solutions, checked on 20 unseen designs against the nearest training design.",
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _wake(Re: float, steps: int, save_every: int, D: int = 20):
    import warnings

    import torch

    from pinneapple_simulation.numerical_solvers.lbm import (
        LBMSolver,
        _d2q9_tensors,
        _equilibrium_2d,
        cylinder_mask,
    )
    nx, ny, u_in = 12 * D, 5 * D, 0.1
    cx, cy = 3.0 * D, ny / 2 + 0.5
    mask = cylinder_mask(nx, ny, cx, cy, D / 2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        solver = LBMSolver(nx=nx, ny=ny, Re=Re * (ny - 2) / D, u_in=u_in, obstacle_mask=mask, Cs=0.0)
    cxv, cyv, wv, _ = _d2q9_tensors(torch.device("cpu"))
    ux0 = torch.where(mask.bool(), 0.0, u_in)
    f0 = _equilibrium_2d(torch.ones(nx, ny), ux0, torch.zeros(nx, ny), cxv, cyv, wv)
    with torch.no_grad():
        out = solver(f0, steps=steps, save_every=save_every)
    ux, uy = out.extras["trajectory_ux"], out.extras["trajectory_uy"]
    ux = np.asarray(torch.stack(ux) if isinstance(ux, list) else ux, np.float64)
    uy = np.asarray(torch.stack(uy) if isinstance(uy, list) else uy, np.float64)
    return ux, uy, mask.numpy().astype(bool), {"nx": nx, "ny": ny, "D": D, "u_in": u_in, "cx": cx, "cy": cy}


def _vort(ux, uy, D, u):
    return (np.gradient(uy, axis=-2) - np.gradient(ux, axis=-1)) / u * D


@register
class ROMStudy(Experiment):
    name = "rom_study"
    version = "1"
    description = ("Reduced-order models on the lab's own solvers: POD / DMD / Operator Inference forecasting a "
                   "cylinder wake, and a parametric POD-RBF / POD-GPR model of a 3-D FEM cantilever, each against "
                   "its baseline (persistence, nearest design).")
    tags = ["rom", "reduced-order-model", "pod", "dmd", "surrogate", "physics-ai", "dataset"]
    references = ["B. R. Noack et al., J. Fluid Mech. 497 (2003) 335 (cylinder wake POD)",
                  "P. J. Schmid, J. Fluid Mech. 656 (2010) 5 (DMD)",
                  "B. Peherstorfer and K. Willcox, CMAME 306 (2016) 196 (Operator Inference)",
                  "J. S. Hesthaven and S. Ubbiali, J. Comput. Phys. 363 (2018) 55 (POD with regression)"]
    limitations = {
        "cylinder_wake": ["2-D, single Reynolds number, periodic regime (the easy case for linear ROMs)",
                          "the transient is excluded from the training window"],
        "beam_parametric": ["linear elasticity: displacement is linear in the load magnitude, so only the geometry "
                            "and the load direction are varied", "80 training designs in a three-parameter box; "
                            "test designs inside the box (no extrapolation)"],
    }
    case_param = "case"
    case_descriptions = CASES
    params = {"case": "cylinder_wake", "Re": 100.0, "steps": 30000, "save_every": 50, "n_train": 80, "n_test": 20,
              "seed": 0}
    space = {"case": list(CASES)}

    def run(self, ctx):
        if ctx.params["case"] == "cylinder_wake":
            self._wake(ctx)
        elif ctx.params["case"] == "beam_parametric":
            self._beam(ctx)
        else:
            raise ValueError(f"case must be one of {list(CASES)}")

    # ------------------------------------------------------------------ cylinder wake
    def _wake(self, ctx):
        import torch

        from pinneapple_neural.architectures.rom import (
            POD,
            DynamicModeDecomposition,
            OperatorInference,
        )
        p = ctx.params
        with ctx.stage("simulate"):
            ux, uy, solid, g = _wake(p["Re"], int(p["steps"]), int(p["save_every"]))
        ctx.input("setup", g | {"Re": p["Re"], "steps": p["steps"], "save_every": p["save_every"]})
        dt = float(p["save_every"])                                   # lattice steps between snapshots
        start = int(0.4 * len(ux))                                     # past the start-up transient
        U = np.concatenate([ux[start:].reshape(len(ux) - start, -1), uy[start:].reshape(len(uy) - start, -1)], 1)
        T = len(U)
        n_tr = int(0.6 * T)
        # shedding frequency from a wake probe on the whole periodic record (the reference for DMD)
        probe = uy[start:, int(g["cx"] + 2 * g["D"]), int(g["ny"] / 2)]
        pr = probe - probe.mean()
        nfft = 16 * len(pr)
        spec = np.abs(np.fft.rfft(pr * np.hanning(len(pr)), nfft))
        fr = np.fft.rfftfreq(nfft, d=dt)
        f_probe = float(fr[np.argmax(spec[1:]) + 1])
        St_probe = f_probe * g["D"] / g["u_in"]
        period = 1.0 / f_probe / dt                                    # snapshots per shedding period
        Xtr = torch.tensor(U[:n_tr], dtype=torch.float64)
        pod = POD(r=20, center=True).fit(Xtr)
        evr = pod.explained_variance_ratio_.numpy()
        sv = pod.sv_.numpy()
        ctx.metric("St_probe", St_probe)
        ctx.metric("pod_energy_modes_1_2", float(evr[:2].sum()))
        ctx.metric("pod_energy_modes_1_4", float(evr[:4].sum()))
        ctx.check("pod_mode_pair_holds_the_energy", value=float(evr[:2].sum()), min=0.8,
                  detail="modes 1 and 2 (the travelling vortex street) hold most of the fluctuation energy",
                  kind="physics")
        ctx.check("pod_modes_come_in_pairs", value=float(sv[1] / sv[0]), min=0.85,
                  detail="sigma_2 / sigma_1: a travelling wave needs two modes of nearly equal energy", kind="physics")
        dmd = DynamicModeDecomposition(r=12).fit(Xtr[None])
        lam, _, _ = dmd.eig()
        lam = lam.numpy()
        fq = np.abs(np.angle(lam)) / (2 * np.pi * dt)
        amp = np.abs(lam)
        lead = np.argsort(-(amp * (fq > 0.2 * f_probe)))[0]           # the least damped oscillating mode
        St_dmd = float(fq[lead] * g["D"] / g["u_in"])
        ctx.metric("St_dmd", St_dmd)
        ctx.check("dmd_frequency_vs_probe_spectrum", value=St_dmd, reference=St_probe, rtol=0.03,
                  detail="leading DMD eigenvalue against the zero-padded FFT of a wake probe", kind="physics")
        # forecasts beyond the training window
        H = T - n_tr
        truth = U[n_tr - 1:]
        x0 = torch.tensor(U[n_tr - 1][None], dtype=torch.float64)
        pred_dmd = dmd.rollout(x0, H)[0].numpy()
        # Operator Inference: rank and quadratic regularisation chosen on the last 25 % of the training window
        A_all = pod.encode(Xtr)
        n_fit = int(0.75 * n_tr)
        val_truth = U[n_fit - 1:n_tr]
        best_cfg, best_err = None, np.inf
        for r_ in (4, 6, 8, 10):
            for lq in (1e-2, 1e0, 1e2):
                a_ = A_all[:n_fit, :r_]
                try:
                    m_ = OperatorInference(r=r_, use_quadratic=True, l2_linear=1e-6, l2_quad=lq, scale=True).fit(a_[None])
                    av = m_.rollout(a_[-1][None], n_tr - n_fit + H)[0]          # through the whole horizon
                    amp = float(av.abs().max()) / float(A_all[:, :r_].abs().max())
                    pv = (av[: n_tr - n_fit + 1] @ pod.basis_[:, :r_].T + pod.mean_).numpy()
                    ev = float(np.mean(np.linalg.norm(pv - val_truth, axis=1)))
                    if not np.isfinite(amp) or amp > 2.0:                       # unstable: reject
                        ev = np.inf
                except Exception:                                  # noqa: BLE001 - a diverging candidate
                    ev = np.inf
                if np.isfinite(ev) and ev < best_err:
                    best_cfg, best_err = (r_, lq), ev
        if best_cfg is None:                                   # every candidate unstable: the most regularised one
            best_cfg = (4, 1e2)
            ctx.log("no stable Operator Inference candidate; using r = 4, l2_quad = 100")
        r, lq = best_cfg
        ctx.output("opinf_selection", {"rank": r, "l2_quad": lq, "validation": "last 25 % of the training window"})
        a_tr = A_all[:, :r]
        oi = OperatorInference(r=r, use_quadratic=True, l2_linear=1e-6, l2_quad=lq, scale=True).fit(a_tr[None])
        a_pred = oi.rollout(a_tr[-1][None], H)[0]
        basis = pod.basis_[:, :r]
        pred_oi = (a_pred @ basis.T + pod.mean_).numpy()
        pers = np.repeat(U[n_tr - 1][None], H + 1, 0)

        def rel(Pr):
            fl = truth - U[:n_tr].mean(0)
            return np.linalg.norm(Pr - truth, axis=1) / np.linalg.norm(fl, axis=1).mean()
        e_dmd, e_oi, e_p = rel(pred_dmd), rel(pred_oi), rel(pers)
        k3 = min(H, int(round(3 * period)))
        ctx.metric("snapshots_per_period", period)
        ctx.metric("forecast_periods", H / period)
        for nm, e in (("dmd", e_dmd), ("opinf", e_oi), ("persistence", e_p)):
            ctx.metric(f"error_3_periods_{nm}", float(e[: k3 + 1].mean()))
            ctx.metric(f"error_end_{nm}", float(e[-1]))
        best = min(float(e_dmd[: k3 + 1].mean()), float(e_oi[: k3 + 1].mean()))
        ctx.check("rom_forecast_beats_persistence", value=float(e_p[: k3 + 1].mean()) / max(best, 1e-12), min=3.0,
                  detail="mean relative error over 3 shedding periods, persistence / best ROM", kind="baseline")
        ctx.check("opinf_bounded_over_horizon", value=float(np.nanmax(np.abs(a_pred.numpy())) / float(A_all[:, :r].abs().max()))
                  if np.isfinite(a_pred.numpy()).all() else float("inf"), max=2.0,
                  detail="largest latent amplitude of the forecast / largest in training (no blow-up)", kind="sanity")
        ctx.check("rom_forecast_error_3_periods", value=best, max=0.15,
                  detail="relative to the fluctuation norm", kind="generalization")
        # figures
        plt = _plt()
        ny, nx = g["ny"], g["nx"]
        modes = pod.basis_.numpy()
        fig, axs = plt.subplots(3, 2, figsize=(11, 6.6))
        axs[0, 0].semilogy(np.arange(1, len(evr) + 1), evr, "o-", color="#1b9e77", ms=4)
        axs[0, 0].set_title("POD energy per mode (the pair 1-2, then 3-4)", fontsize=9)
        axs[0, 0].set_xlabel("mode")
        th = np.linspace(0, 2 * np.pi, 200)
        axs[0, 1].plot(np.cos(th), np.sin(th), color="0.7", lw=0.8)
        axs[0, 1].scatter(lam.real, lam.imag, c=np.where(np.arange(len(lam)) == lead, "#d95f02", "#7570b3"), s=18)
        axs[0, 1].set_aspect("equal")
        axs[0, 1].set_title(f"DMD eigenvalues: St = {St_dmd:.4f} (probe {St_probe:.4f})", fontsize=9)
        for k, ax in enumerate(axs[1:].ravel()):
            mx, my = modes[: nx * ny, k].reshape(nx, ny), modes[nx * ny:, k].reshape(nx, ny)
            w = _vort(mx, my, g["D"], 1.0)
            w[solid] = np.nan
            v = np.nanmax(np.abs(w))
            ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-v, vmax=v, aspect="equal")
            ax.set_title(f"POD mode {k + 1} (vorticity), {100 * evr[k]:.1f} % of the energy", fontsize=9)
            ax.set_axis_off()
        fig.tight_layout()
        ctx.figure("pod_modes_and_dmd_spectrum", fig)
        fig, ax = plt.subplots(figsize=(8, 3.6))
        tp = np.arange(H + 1) / period
        ax.plot(tp, e_p, color="0.5", label="persistence")
        ax.plot(tp, e_dmd, color="#7570b3", label="POD-DMD (linear, r = 12)")
        ax.plot(tp, e_oi, color="#d95f02", label=f"POD-OpInf (quadratic, r = {r})")
        ax.set_xlabel("forecast horizon (shedding periods)")
        ax.set_ylabel("relative error")
        ax.legend(frameon=False)
        ax.set_title("Forecasting the wake beyond the training window", fontsize=10)
        fig.tight_layout()
        ctx.figure("forecast_error", fig)
        frames = []
        D, u0 = g["D"], g["u_in"]
        for i in range(0, H + 1, max(1, H // 40)):
            fig, axs = plt.subplots(2, 1, figsize=(7, 3.6))
            for ax, X, lab in ((axs[0], truth[i], "LBM"), (axs[1], pred_dmd[i], "POD-DMD forecast")):
                w = _vort(X[: nx * ny].reshape(nx, ny), X[nx * ny:].reshape(nx, ny), D, u0)
                w[solid] = np.nan
                ax.imshow(w.T, origin="lower", cmap="RdBu_r", vmin=-3, vmax=3, aspect="equal")
                ax.set_title(f"{lab}, {i / period:.1f} periods after the training window", fontsize=8)
                ax.set_axis_off()
            fig.tight_layout()
            fig.canvas.draw()
            frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0)
            plt.close(fig)
        ctx.gif("wake_truth_vs_rom", frames, duration_ms=110)
        ds = ctx.dataset("wake", description="Velocity snapshots (periodic regime) and POD modes / singular values",
                         units={"ux": "lattice", "uy": "lattice"})
        ds.add(ux=ux[start:].astype(np.float32), uy=uy[start:].astype(np.float32),
               pod_modes=modes[:, :8].astype(np.float32), singular_values=sv.astype(np.float32),
               dt_lattice=dt, St_probe=St_probe)

    # ------------------------------------------------------------------ parametric beam
    def _beam(self, ctx):
        from pinneapple_neural.architectures.rom import ParametricPOD, latin_hypercube
        from pinneapple_simulation.numerical_solvers.solid_fem import (
            FEMResult,
            SolidFEM,
            box_mesh,
            fea_figure,
        )
        p = ctx.params
        L, nx, ny, nz, F = 2.0, 30, 3, 6, 1e4
        box = {"H": (0.12, 0.3), "W": (0.06, 0.2), "angle_deg": (0.0, 90.0)}
        inner = {k: (a + 0.1 * (b - a), b - 0.1 * (b - a)) for k, (a, b) in box.items()}

        def solve(H, W, ang):
            m = box_mesh(L, W, H, nx, ny, nz)
            f = SolidFEM(m)
            f.fix(m.nodes_on(x=0.0))
            a = math.radians(ang)
            f.load_face(m.face_nodes("x+"), (0.0, F * math.sin(a), -F * math.cos(a)))
            return f.solve()
        Ptr, names = latin_hypercube(int(p["n_train"]), box, int(p["seed"]))
        Pte, _ = latin_hypercube(int(p["n_test"]), inner, int(p["seed"]) + 7)
        ctx.input("design_space", {"parameters": box, "test_inside": inner, "L": L, "mesh": [nx, ny, nz], "load_N": F})
        with ctx.stage("fem_snapshots"):
            t0 = time.time()
            Rtr = [solve(*q) for q in Ptr]
            t_fem = (time.time() - t0) / len(Ptr)
            Rte = [solve(*q) for q in Pte]
        feats = lambda Q: np.c_[np.log(Q[:, 0]), np.log(Q[:, 1]), Q[:, 2] / 90.0]   # noqa: E731
        fields = {"displacement": (np.array([r.u.ravel() for r in Rtr]), np.array([r.u.ravel() for r in Rte])),
                  "von_mises": (np.array([r.von_mises for r in Rtr]) / 1e6, np.array([r.von_mises for r in Rte]) / 1e6)}
        table = {}
        best = {}
        for fld, (Xa, Xb) in fields.items():
            for reg in ("nearest", "rbf", "gpr"):
                for norm in (False, True):
                    rom = ParametricPOD(r=16, energy=None, regressor=reg, normalize=norm, seed=int(p["seed"]))
                    import warnings
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        rom.fit(feats(Ptr), Xa)
                        t0 = time.time()
                        e = rom.error(feats(Pte), Xb)
                        t_rom = (time.time() - t0) / len(Pte)
                    key = f"{reg}{'+split' if norm else ''}"
                    table[(fld, key)] = {"median": float(np.median(e)), "max": float(e.max()), "t": t_rom,
                                         "errors": e}
                    if reg != "nearest" and (fld not in best or np.median(e) < table[(fld, best[fld])]["median"]):
                        best[fld] = key
            nn = min(table[(fld, "nearest")]["median"], table[(fld, "nearest+split")]["median"])
            b = table[(fld, best[fld])]
            ctx.metric(f"{fld}_best_rom", best[fld])
            ctx.metric(f"{fld}_median_error", b["median"])
            ctx.metric(f"{fld}_max_error", b["max"])
            ctx.metric(f"{fld}_nearest_design_median_error", nn)
            ctx.check(f"{fld}_rom_beats_nearest_design", value=nn / max(b["median"], 1e-12), min=3.0,
                      detail="median relative L2 error on unseen designs, nearest training design / best ROM",
                      kind="baseline")
            ctx.check(f"{fld}_rom_error_unseen_designs", value=b["median"], max=0.03,
                      detail=f"median relative L2 error on {len(Pte)} unseen designs ({best[fld]})",
                      kind="generalization")
        # uncertainty: the GP's predictive standard deviation must cover the unseen designs at its nominal level
        from ..uq import coverage_check
        for fld in fields:
            g_ = ParametricPOD(r=16, energy=None, regressor="gpr", normalize=fld == "von_mises").fit(feats(Ptr), fields[fld][0])
            mu, sd = g_.predict(feats(Pte), return_std=True)
            # the GP covers the coefficient regression; add the basis truncation error measured on training designs
            trunc = np.sqrt(np.mean((g_.projection_error(fields[fld][0]) *
                                     np.linalg.norm(fields[fld][0], axis=1)) ** 2) / fields[fld][0].shape[1])
            coverage_check(ctx, fld, fields[fld][1], mu, np.sqrt(sd ** 2 + trunc ** 2), level=0.9, tol=0.1)
        speed = t_fem / max(table[("displacement", best["displacement"])]["t"], 1e-9)
        ctx.metric("fem_seconds_per_design", t_fem)
        ctx.metric("speedup_over_fem", speed)
        ctx.output("errors", {f"{k[0]}|{k[1]}": {"median": v["median"], "max": v["max"]} for k, v in table.items()})
        plt = _plt()
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
        for ax, fld in zip(axs, fields, strict=True):
            keys = [k for (f_, k) in table if f_ == fld]
            med = [table[(fld, k)]["median"] for k in keys]
            mx = [table[(fld, k)]["max"] for k in keys]
            y = np.arange(len(keys))
            from matplotlib.ticker import NullFormatter
            ax.barh(y, med, color=["#999" if k.startswith("nearest") else "#1b9e77" for k in keys])
            ax.plot(mx, y, "k|", ms=12, label="worst design")
            ax.set_yticks(y, keys)
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.set_title(f"{fld}: median relative L2 error (bars) on {len(Pte)} unseen designs\n"
                         "grey: nearest training design (baseline)", fontsize=9)
            ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        ctx.figure("rom_errors", fig)
        # the worst unseen design: FEM against ROM, post-processor style
        fld = "von_mises"
        e = table[(fld, best[fld])]["errors"]
        i = int(np.argmax(e))
        reg, norm = best[fld].split("+")[0], best[fld].endswith("split")
        rom_vm = ParametricPOD(r=16, energy=None, regressor=reg, normalize=norm).fit(feats(Ptr), fields[fld][0])
        rom_u = ParametricPOD(r=16, energy=None, regressor=best["displacement"].split("+")[0],
                              normalize=best["displacement"].endswith("split")).fit(feats(Ptr), fields["displacement"][0])
        q = Pte[i:i + 1]
        vm_hat = rom_vm.predict(feats(q))[0] * 1e6
        u_hat = rom_u.predict(feats(q))[0].reshape(-1, 3)
        true = Rte[i]
        rom_res = FEMResult(true.mesh, u_hat, np.c_[vm_hat, np.zeros((len(vm_hat), 5))], true.reactions,
                            true.info | {"element": "POD-ROM"})
        vr = (float(true.von_mises.min()) / 1e6, float(true.von_mises.max()) / 1e6)
        _, sc = fea_figure(true, title="x")
        plt.close("all")
        for nm, rr in (("fem", true), ("rom", rom_res)):
            fig, _ = fea_figure(rr, title=f"{'FEM' if nm == 'fem' else 'ROM (' + best[fld] + ')'}: H = {q[0, 0]:.3f} m, "
                                f"W = {q[0, 1]:.3f} m, load at {q[0, 2]:.0f} deg (worst unseen design)",
                                scale=sc, vrange=vr)
            ctx.figure(f"worst_design_{nm}", fig, dpi=120)
            plt.close(fig)
        ds = ctx.dataset("beam_designs", description="Parametric FEM snapshots: design parameters, nodal "
                         "displacement and von Mises on a fixed-topology mesh", units={"u": "m", "von_mises": "MPa"})
        for split, P_, R_ in (("train", Ptr, Rtr), ("test", Pte, Rte)):
            for q_, r_ in zip(P_, R_, strict=True):
                ds.add(split=split, H=float(q_[0]), W=float(q_[1]), angle_deg=float(q_[2]),
                       u=r_.u.astype(np.float32), von_mises=(r_.von_mises / 1e6).astype(np.float32))
