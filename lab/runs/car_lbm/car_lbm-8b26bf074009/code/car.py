"""Road-vehicle aerodynamics with a parametric car body and a Physics AI surrogate.

* ``car_lbm``       -- one design of a 2-D car side profile (``pinneapple_design.geometry.gen.car2d``, Ahmed-type,
                       no wheels) in a virtual wind tunnel: lattice-Boltzmann D2Q9 with Smagorinsky LES, moving
                       ground at the free-stream speed, slip ceiling. Drag and lift by momentum exchange, mean
                       fields, a vorticity movie. Dataset: geometry (mask, SDF), mean velocity and pressure,
                       coefficients; plus vorticity snapshots.
* ``car_surrogate`` -- trains on every ``car_lbm`` run in the database: an FNO (``pinneapple_neural``) maps the
                       geometry to the mean flow, with and without a physics loss (mass conservation, no-slip on
                       the body), and an MLP ensemble maps the six design parameters to drag and lift. Both are
                       scored on designs they never saw, against simple baselines. The surrogate then searches
                       the design space for the lowest drag and the winner is verified by a new LBM run.

    python -m pinneapple_lab sweep car_lbm -n 48 -j 2
    python -m pinneapple_lab run car_surrogate
"""
from __future__ import annotations

import os

import numpy as np

from ..spec import Experiment, register

_SHAPE_KEYS = ("slant_deg", "windshield_deg", "hood", "clearance", "diffuser_deg", "nose")


def simulate_car(shape: dict, *, length_cells: int = 64, Re: float = 500.0, u_in: float = 0.08, Cs: float = 0.1,
                 steps: int = 8000, save_every: int = 50, keep_frames: int = 48, tunnel_height: float = 2.5,
                 solid_mask=None, frontal_height: float | None = None):
    """Run the virtual wind tunnel for one car profile (or any ``solid_mask``, with its ``frontal_height`` in
    cells). Returns a dict of fields, histories and coefficients (Cd on the frontal height, Cl on the length)."""
    import torch

    from pinneapple_design.geometry.gen.car2d import CarProfile
    from pinneapple_simulation.numerical_solvers import lbm as L
    torch.set_num_threads(1)
    Lc = int(length_cells)
    nx, ny = 5 * Lc, int(tunnel_height * Lc)
    x_nose = 1.2 * Lc
    car = CarProfile(**{k: float(shape[k]) for k in _SHAPE_KEYS})
    solid_np = car.mask(nx, ny, Lc, x_nose) if solid_mask is None else np.asarray(solid_mask, bool)
    nx, ny = solid_np.shape
    solid = torch.tensor(solid_np)
    fluid = ~solid
    nu = u_in * Lc / Re
    omega = 1.0 / (3 * nu + 0.5)
    cx, cy, w, opp = L._d2q9_tensors(torch.device("cpu"))
    cxf, cyf = cx.float(), cy.float()
    wl = [float(v) for v in w]
    links = torch.stack([solid & torch.roll(fluid, shifts=(int(cx[i]), int(cy[i])), dims=(0, 1)) for i in range(9)])
    ux0 = torch.where(solid, 0.0, u_in)
    f = L._equilibrium_2d(torch.ones(nx, ny), ux0, torch.zeros(nx, ny), cx, cy, w)
    up = [i for i, d in enumerate(L._C2Q9_Y) if d > 0]
    down = [i for i, d in enumerate(L._C2Q9_Y) if d < 0]
    mirror = {i: next(j for j in range(9) if L._C2Q9_X[j] == L._C2Q9_X[i] and L._C2Q9_Y[j] == -L._C2Q9_Y[i])
              for i in down}
    Fx, Fy, frames = [], [], []
    mux = torch.zeros(nx, ny)
    muy = torch.zeros(nx, ny)
    mrho = torch.zeros(nx, ny)
    muu, muv, mvv = torch.zeros(nx, ny), torch.zeros(nx, ny), torch.zeros(nx, ny)
    nmean = 0
    for s in range(steps):
        rho, ux, uy = L._macroscopic_2d(f, cx, cy)
        feq = L._equilibrium_2d(rho, ux, uy, cx, cy, w)
        om = L._smagorinsky_omega(f, feq, rho, omega, Cs, cx, cy) if Cs > 0 else omega
        f_post = f - om * (f - feq)
        f_new = L._stream_2d(f_post, cx, cy)
        for i in up:            # moving road (u_wall = u_in): bounce-back with the wall-momentum correction
            f_new[i, :, 0] = f_post[L._OPP_D2Q9[i], :, 0] + 6.0 * wl[i] * L._C2Q9_X[i] * u_in
        for i in down:          # slip ceiling: specular reflection
            f_new[i, :, -1] = f_post[mirror[i], :, -1]
        g = (f_new * links).sum(dim=(1, 2))
        Fx.append(float(2 * (cxf * g).sum()))
        Fy.append(float(2 * (cyf * g).sum()))
        f_new = L._bounce_back_2d(f_post, f_new, solid, opp)
        L._zou_he_inlet(f_new, u_in)
        L._zou_he_outlet(f_new, 1.0)
        f = f_new
        if s >= steps // 2:
            mux += ux
            muy += uy
            mrho += rho
            muu += rho * ux * ux
            muv += rho * ux * uy
            mvv += rho * uy * uy
            nmean += 1
        if (s + 1) % save_every == 0 and s >= steps - keep_frames * save_every:
            vort = (np.gradient(uy.numpy(), axis=0) - np.gradient(ux.numpy(), axis=1)) * Lc / u_in
            vort[solid_np] = 0.0
            frames.append(vort.astype(np.float32))
    H = car.frontal_height() * Lc if frontal_height is None else float(frontal_height)
    q = 0.5 * u_in ** 2
    cd = np.asarray(Fx) / (q * H)
    cl = np.asarray(Fy) / (q * Lc)
    mux, muy, mrho, muu, muv, mvv = (t / nmean for t in (mux, muy, mrho, muu, muv, mvv))
    # time-mean force on the body by a momentum balance over a box around it (pressure + momentum flux, including
    # the turbulent stresses; viscous stresses on the box are negligible at these Reynolds numbers)
    ys_, xs_ = np.nonzero(solid_np.T)
    i0, i1 = max(1, xs_.min() - Lc // 3), min(nx - 2, xs_.max() + Lc // 2)
    j0, j1 = 1, min(ny - 2, ys_.max() + Lc // 3)
    pr = mrho.numpy() / 3.0
    uu, uv, vv = muu.numpy(), muv.numpy(), mvv.numpy()
    fx_cv = (-(pr[i1, j0:j1 + 1] + uu[i1, j0:j1 + 1]).sum() + (pr[i0, j0:j1 + 1] + uu[i0, j0:j1 + 1]).sum()
             - uv[i0:i1 + 1, j1].sum() + uv[i0:i1 + 1, j0].sum())
    fy_cv = (-(pr[i0:i1 + 1, j1] + vv[i0:i1 + 1, j1]).sum() + (pr[i0:i1 + 1, j0] + vv[i0:i1 + 1, j0]).sum()
             - uv[i1, j0:j1 + 1].sum() + uv[i0, j0:j1 + 1].sum())
    sdf = car.sdf(nx, ny, Lc, x_nose) / Lc if solid_mask is None else np.zeros((nx, ny), np.float32)
    return {"cd_cv": float(fx_cv) / (q * H), "cl_cv": float(fy_cv) / (q * Lc), "car": car, "solid": solid_np, "sdf": sdf, "mean_ux": mux.numpy() / u_in, "mean_uy": muy.numpy() / u_in,
            "mean_p": (mrho.numpy() - 1.0) / 3.0 / q, "cd_t": cd, "cl_t": cl, "frames": frames, "nx": nx, "ny": ny,
            "Lc": Lc, "x_nose": x_nose, "u_in": u_in, "frontal_height": H, "finite": bool(torch.isfinite(f).all())}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _flow_figure(sim, title):
    plt = _plt()
    sp = np.hypot(sim["mean_ux"], sim["mean_uy"])
    sp[sim["solid"]] = np.nan
    Lc = sim["Lc"]
    x = (np.arange(sim["nx"]) - sim["x_nose"]) / Lc
    y = np.arange(sim["ny"]) / Lc
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(sp.T, origin="lower", extent=[x[0], x[-1], y[0], y[-1]], cmap="magma", vmin=0, vmax=1.4)
    ax.streamplot(x, y, sim["mean_ux"].T, sim["mean_uy"].T, color="w", linewidth=0.5, density=1.4, arrowsize=0.5)
    P = sim["car"].outline()
    ax.fill(P[:, 0], P[:, 1], color="#c9d3d6", ec="#1b2a30", lw=1)
    ax.axhline(0, color="#7a8a90", lw=2)
    ax.set_xlim(-0.8, 3.2)
    ax.set_ylim(0, y[-1])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("x / car length")
    fig.colorbar(im, ax=ax, shrink=0.8, label="|u| / U (time mean)")
    fig.tight_layout()
    return fig


@register
class CarLBM(Experiment):
    name = "car_lbm"
    version = "1"
    description = ("Virtual wind tunnel for a parametric 2-D car body (Ahmed-type, six design parameters): "
                   "lattice-Boltzmann LES with moving road; drag and lift by momentum exchange, mean flow, wake "
                   "vorticity movie. The training data of car_surrogate.")
    tags = ["automotive", "cfd", "lbm", "geometry", "dataset"]
    params = {"slant_deg": 25.0, "windshield_deg": 35.0, "hood": 0.62, "clearance": 0.08, "diffuser_deg": 4.0,
              "nose": 0.5, "length_cells": 64, "Re": 500.0, "u_in": 0.08, "Cs": 0.1, "steps": 8000, "tunnel_height": 2.5}
    space = {"slant_deg": (5.0, 40.0), "windshield_deg": (20.0, 55.0), "hood": (0.45, 0.75),
             "clearance": (0.04, 0.12), "diffuser_deg": (0.0, 10.0), "nose": (0.0, 1.0)}

    def run(self, ctx):
        p = ctx.params
        shape = {k: p[k] for k in _SHAPE_KEYS}
        ctx.input("design", shape)
        with ctx.stage("simulate"):
            sim = simulate_car(shape, length_cells=p["length_cells"], Re=p["Re"], u_in=p["u_in"], Cs=p["Cs"],
                               steps=p["steps"], tunnel_height=p["tunnel_height"])
        n = len(sim["cd_t"])
        q3, q4 = slice(n // 2, 3 * n // 4), slice(3 * n // 4, n)
        # coefficients from the control-volume momentum balance (validated on a cylinder: Cd 2.11 at Re 20 and
        # 1.64 at Re 40 against 2.05-2.09 and 1.52-1.60 unconfined, blockage 1/12); the bounce-back momentum
        # exchange series over-predicts the level about 2x here and is used only for its unsteadiness
        cd, cl = float(sim["cd_cv"]), float(sim["cl_cv"])
        ctx.metric("Cd", cd)
        ctx.metric("Cl", cl)
        ctx.metric("Cd_rms_fluctuation", float(sim["cd_t"][n // 2:].std() / sim["cd_t"][n // 2:].mean() * cd))
        sig = sim["cl_t"][n // 2:] - cl
        spec = np.abs(np.fft.rfft(sig))
        k = int(np.argmax(spec[1:]) + 1)
        st = float(np.fft.rfftfreq(len(sig))[k] * sim["frontal_height"] / p["u_in"])
        ctx.metric("Strouhal_frontal_height", st)
        div = np.gradient(sim["mean_ux"], axis=0) + np.gradient(sim["mean_uy"], axis=1)
        fl = ~sim["solid"]
        fl[:2], fl[-2:], fl[:, :2], fl[:, -2:] = False, False, False, False
        ctx.metric("mean_flow_divergence", float(np.abs(div[fl]).mean()))
        ctx.check("finite_fields", sim["finite"])
        ctx.check("drag_statistically_converged", value=abs(sim["cd_t"][q3].mean() / sim["cd_t"][q4].mean() - 1),
                  max=0.05, detail="mean drag over the 3rd vs 4th quarter of the run")
        ctx.check("drag_in_bluff_body_range", value=cd, min=0.1, max=2.5,
                  detail="2-D bluff bodies near a moving ground: Cd on the frontal height ~0.3-2")
        ds = ctx.dataset("flow", description="Car geometry (mask, signed distance) and the time-mean flow it "
                         "produces (velocity / U, pressure coefficient), with drag and lift",
                         units={"sdf": "car lengths", "mean_ux": "U", "mean_uy": "U", "mean_cp": "-"})
        ds.add(mask=sim["solid"].astype(np.uint8), sdf=sim["sdf"].astype(np.float32),
               mean_ux=sim["mean_ux"].astype(np.float32), mean_uy=sim["mean_uy"].astype(np.float32),
               mean_cp=sim["mean_p"].astype(np.float32), Cd=cd, Cl=cl,
               **{k: float(v) for k, v in shape.items()})
        snaps = ctx.dataset("vorticity", description="Wake vorticity snapshots (normalised by U / car length)",
                            shard_size=64)
        for i, fr in enumerate(sim["frames"]):
            snaps.add(vorticity=fr.astype(np.float16), frame=i, Cd=cd, **{k: float(v) for k, v in shape.items()})
        ctx.figure("mean_flow", _flow_figure(sim, f"Cd = {cd:.3f}, Cl = {cl:.3f}   slant {shape['slant_deg']:.0f} deg, "
                                                   f"clearance {shape['clearance']:.3f} L"))
        plt = _plt()
        fig, ax = plt.subplots(figsize=(6, 2.6))
        t = np.arange(n) * p["u_in"] / p["length_cells"]
        ax.plot(t, sim["cd_t"], lw=0.8, label="Cd")
        ax.plot(t, sim["cl_t"], lw=0.8, label="Cl")
        ax.axvspan(t[n // 2], t[-1], color="#999", alpha=0.12)
        ax.set_xlabel("convective time  t U / L")
        ax.legend(frameon=False)
        ax.set_ylim(-1, 4)
        fig.tight_layout()
        ctx.figure("forces", fig)
        # wake movie
        cmap = plt.get_cmap("RdBu_r")
        P = sim["car"].outline()
        Lc = sim["Lc"]
        xs = np.clip((P[:, 0] * Lc + sim["x_nose"]).astype(int), 0, sim["nx"] - 1)
        ys = np.clip((P[:, 1] * Lc).astype(int), 0, sim["ny"] - 1)
        gif = []
        for fr in sim["frames"]:
            img = cmap(np.clip(fr.T[::-1] / 12 + 0.5, 0, 1))[..., :3]
            img[(~sim["solid"]).T[::-1] == 0] = (0.78, 0.82, 0.84)
            img[sim["ny"] - 1 - ys, xs] = (0.1, 0.16, 0.19)
            gif.append(img)
        ctx.gif("wake_vorticity", gif, duration_ms=70)


@register
class CarSurrogate(Experiment):
    name = "car_surrogate"
    version = "1"
    description = ("Physics AI on car geometry: an FNO maps the car shape to its mean flow (data-only and with a "
                   "mass-conservation / no-slip physics loss), an MLP ensemble maps the design parameters to drag "
                   "and lift; scored on unseen designs against baselines, then used to find a low-drag shape that "
                   "a new LBM run verifies.")
    tags = ["automotive", "physics-ai", "surrogate", "fno", "optimization"]
    params = {"holdout": 0.2, "epochs": 300, "width": 24, "modes": 16, "physics_weight": 0.5, "ensemble": 5,
              "search_samples": 4000, "verify": True, "seed": 0}

    def run(self, ctx):
        import torch

        from ..store import LabStore
        p = ctx.params
        torch.manual_seed(p["seed"])
        torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
        lab_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(ctx.dir))))
        store = LabStore(lab_root)
        samples = list(store.samples("car_lbm", "flow"))
        if len(samples) < 10:
            raise RuntimeError(f"car_surrogate needs at least 10 completed car_lbm runs, found {len(samples)}")
        samples.sort(key=lambda s: s["_run_id"])
        rng = np.random.default_rng(p["seed"])
        idx = rng.permutation(len(samples))
        n_test = max(2, int(round(p["holdout"] * len(samples))))
        test, train = idx[:n_test], idx[n_test:]
        ctx.input("split", {"train": [samples[i]["_run_id"] for i in train],
                            "test": [samples[i]["_run_id"] for i in test]})
        ctx.metric("n_train", len(train))
        ctx.metric("n_test", len(test))
        X = np.stack([np.stack([s["mask"].astype(np.float32), s["sdf"]]) for s in samples])
        Y = np.stack([np.stack([s["mean_ux"], s["mean_uy"], s["mean_cp"]]) for s in samples]).astype(np.float32)
        Pm = np.array([[s[k] for k in _SHAPE_KEYS] for s in samples], dtype=np.float64)
        C = np.array([[s["Cd"], s["Cl"]] for s in samples])
        fluid = 1.0 - X[:, :1]

        # ------------------------------------------------------------ field surrogate (FNO)
        def train_fno(lam):
            from pinneapple_neural.architectures.neural_operators.fno import FNO2d
            torch.manual_seed(p["seed"])
            net = FNO2d(2, 3, width=p["width"], modes1=p["modes"], modes2=p["modes"], layers=4)
            opt = torch.optim.Adam(net.parameters(), lr=3e-3)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, p["epochs"])
            xt, yt, ft = (torch.tensor(a[train]) for a in (X, Y, fluid))
            hist = []
            for _ in range(p["epochs"]):
                perm = torch.randperm(len(train))
                tot = 0.0
                for b in range(0, len(train), 4):
                    j = perm[b:b + 4]
                    out = net(xt[j]).y * ft[j]                         # no-slip: zero velocity inside the body
                    loss = ((out - yt[j]) ** 2 * ft[j]).mean()
                    if lam > 0:                                         # mean flow is divergence-free
                        dudx = out[:, 0, 2:, 1:-1] - out[:, 0, :-2, 1:-1]
                        dvdy = out[:, 1, 1:-1, 2:] - out[:, 1, 1:-1, :-2]
                        loss = loss + lam * (((dudx + dvdy) * ft[j][:, 0, 1:-1, 1:-1]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                    tot += float(loss)
                sched.step()
                hist.append(tot)
            net.eval()
            with torch.no_grad():
                pred = (net(torch.tensor(X)).y * torch.tensor(fluid)).numpy()
            return net, pred, hist

        def field_err(pred, ids):
            sp_t = np.hypot(Y[ids, 0], Y[ids, 1])
            sp_p = np.hypot(pred[ids, 0], pred[ids, 1])
            return float(np.mean([np.linalg.norm(a - b) / np.linalg.norm(b) for a, b in zip(sp_p, sp_t, strict=True)]))

        def divergence(pred, ids):
            d = (pred[ids, 0, 2:, 1:-1] - pred[ids, 0, :-2, 1:-1] + pred[ids, 1, 1:-1, 2:] - pred[ids, 1, 1:-1, :-2])
            return float(np.abs(d * fluid[ids, 0, 1:-1, 1:-1]).mean())

        with ctx.stage("train_fno"):
            _, pred0, h0 = train_fno(0.0)
            _, pred1, h1 = train_fno(p["physics_weight"])
        # baseline: the flow of the nearest training design in parameter space
        z = (Pm - Pm[train].mean(0)) / Pm[train].std(0)
        nn_of = {int(i): int(train[np.argmin(((z[train] - z[i]) ** 2).sum(1))]) for i in test}
        pred_nn = Y.copy()
        for i, j in nn_of.items():
            pred_nn[i] = Y[j] * fluid[i]
        e_data, e_phys, e_nn = field_err(pred0, test), field_err(pred1, test), field_err(pred_nn, test)
        ctx.metric("field_speed_rel_l2_fno_data_only", e_data)
        ctx.metric("field_speed_rel_l2_fno_physics", e_phys)
        ctx.metric("field_speed_rel_l2_nearest_design", e_nn)
        ctx.metric("divergence_fno_data_only", divergence(pred0, test))
        ctx.metric("divergence_fno_physics", divergence(pred1, test))
        ctx.metric("divergence_lbm", divergence(Y, test))
        best_field = min(e_data, e_phys)
        ctx.check("fno_beats_nearest_design", value=best_field, max=e_nn)
        ctx.check("fno_field_error", value=best_field, max=0.15, detail="relative L2 of |u| on unseen designs")

        # ------------------------------------------------------------ coefficient surrogate (MLP ensemble)
        with ctx.stage("train_mlp"):
            zt = torch.tensor(z, dtype=torch.float32)
            cm, cs = C[train].mean(0), C[train].std(0)
            ct = torch.tensor((C - cm) / cs, dtype=torch.float32)
            members = []
            for m in range(p["ensemble"]):
                torch.manual_seed(100 + m)
                net = torch.nn.Sequential(torch.nn.Linear(6, 64), torch.nn.Tanh(), torch.nn.Linear(64, 64),
                                          torch.nn.Tanh(), torch.nn.Linear(64, 2))
                opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
                tr = torch.tensor(train)
                for _ in range(3000):
                    loss = ((net(zt[tr]) - ct[tr]) ** 2).mean()
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
                members.append(net)

            def coef(zz):
                with torch.no_grad():
                    out = torch.stack([m(torch.tensor(zz, dtype=torch.float32)) for m in members]).numpy()
                return out.mean(0) * cs + cm, out.std(0) * cs
            cpred, csd = coef(z)
        A = np.c_[z, np.ones(len(z))]
        lin = np.linalg.lstsq(A[train], C[train], rcond=None)[0]
        clin = A @ lin
        mape = lambda pr: float(np.mean(np.abs(pr[test, 0] / C[test, 0] - 1)))   # noqa: E731
        ctx.metric("Cd_mape_mlp_ensemble", mape(cpred))
        ctx.metric("Cd_mape_linear", mape(clin))
        ctx.metric("Cd_mape_nearest_design", float(np.mean([abs(C[j, 0] / C[i, 0] - 1) for i, j in nn_of.items()])))
        ctx.metric("Cl_mae_mlp_ensemble", float(np.mean(np.abs(cpred[test, 1] - C[test, 1]))))
        ctx.check("Cd_surrogate_error", value=mape(cpred), max=0.08, detail="mean |error| on unseen designs")
        ctx.check("Cd_surrogate_beats_linear", value=mape(cpred), max=mape(clin))

        plt = _plt()
        # parity
        fig, ax = plt.subplots(1, 2, figsize=(8, 3.6))
        for k, name in enumerate(("Cd", "Cl")):
            ax[k].errorbar(C[test, k], cpred[test, k], yerr=2 * csd[test, k], fmt="o", color="#d95f02", ms=4,
                           label="unseen designs (±2σ ensemble)")
            ax[k].plot(C[train, k], cpred[train, k], ".", color="#999", ms=3, label="training designs")
            lo, hi = C[:, k].min(), C[:, k].max()
            ax[k].plot([lo, hi], [lo, hi], "k-", lw=0.6)
            ax[k].set_xlabel(f"{name} (LBM)")
            ax[k].set_ylabel(f"{name} (surrogate)")
        ax[0].legend(frameon=False, fontsize=7)
        fig.tight_layout()
        ctx.figure("coefficient_parity", fig)
        # fields: three unseen designs
        Lc = 64
        nxg, nyg = Y.shape[2], Y.shape[3]
        ext = [-1.2, (nxg - 1.2 * Lc) / Lc, 0, nyg / Lc]
        show = list(test[:3])
        fig, axes = plt.subplots(len(show), 3, figsize=(12, 2.1 * len(show)))
        best_pred = pred1 if e_phys <= e_data else pred0
        for r, i in enumerate(show):
            sp_t = np.hypot(Y[i, 0], Y[i, 1])
            sp_p = np.hypot(best_pred[i, 0], best_pred[i, 1])
            for c, (img, ttl, cm_) in enumerate(((sp_t, "LBM (truth)", "magma"), (sp_p, "FNO surrogate", "magma"),
                                                 (np.abs(sp_p - sp_t), "|error|", "viridis"))):
                a = axes[r, c] if len(show) > 1 else axes[c]
                im = np.where(X[i, 0] > 0, np.nan, img)
                a.imshow(im.T, origin="lower", extent=ext, cmap=cm_, vmin=0, vmax=1.4 if c < 2 else 0.3)
                a.set_xlim(-0.8, 3.2)
                a.set_title(f"{ttl}  design {samples[i]['_run_id'][-6:]}", fontsize=8)
                a.set_xticks([])
                a.set_yticks([])
        fig.tight_layout()
        ctx.figure("field_predictions_unseen", fig)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.semilogy(h0, label="data only")
        ax.semilogy(h1, label="with physics loss")
        ax.set_xlabel("epoch")
        ax.set_ylabel("training loss")
        ax.legend(frameon=False)
        fig.tight_layout()
        ctx.figure("fno_training", fig)

        # ------------------------------------------------------------ design search with the surrogate, verified
        if p["verify"]:
            from pinneapple_design.geometry.gen.car2d import sample_designs
            cands = sample_designs(p["search_samples"], seed=p["seed"] + 1)
            Pc = np.array([[d[k] for k in _SHAPE_KEYS] for d in cands])
            zc = (Pc - Pm[train].mean(0)) / Pm[train].std(0)
            cp_, sd_ = coef(zc)
            # stay inside the data: penalise ensemble disagreement (a trust region on the surrogate)
            score = cp_[:, 0] + 2 * sd_[:, 0]
            best = int(np.argmin(score))
            design = cands[best]
            ctx.output("surrogate_optimum", {"design": design, "Cd_predicted": float(cp_[best, 0]),
                                             "Cd_sigma": float(sd_[best, 0])})
            with ctx.stage("verify_lbm"):
                sim = simulate_car(design)
            cd_true = float(sim["cd_cv"])
            ctx.metric("optimum_Cd_predicted", float(cp_[best, 0]))
            ctx.metric("optimum_Cd_lbm", cd_true)
            ctx.metric("optimum_vs_median_training_Cd", cd_true / float(np.median(C[train, 0])))
            ctx.metric("optimum_vs_best_training_Cd", cd_true / float(C[train, 0].min()))
            ctx.check("optimum_prediction_verified", value=abs(cp_[best, 0] / cd_true - 1), max=0.1,
                      detail="surrogate vs a new LBM run of the chosen design")
            ctx.check("optimum_below_median_design", value=cd_true, max=float(np.median(C[train, 0])))
            ctx.figure("optimum_flow", _flow_figure(sim, f"surrogate optimum: Cd predicted {cp_[best, 0]:.3f}, "
                                                        f"LBM {cd_true:.3f}"))
        ds = ctx.dataset("predictions", description="Surrogate predictions on every design (train and unseen)")
        for i in range(len(samples)):
            ds.add(run_id=samples[i]["_run_id"], split="test" if i in set(test.tolist()) else "train",
                   Cd=float(C[i, 0]), Cd_pred=float(cpred[i, 0]), Cl=float(C[i, 1]), Cl_pred=float(cpred[i, 1]),
                   speed_pred=np.hypot(best_pred[i, 0], best_pred[i, 1]).astype(np.float16))
