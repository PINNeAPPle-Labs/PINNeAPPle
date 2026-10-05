"""3D: how good is the fan? h on the top of a steel block from nine thermocouples, and the chip temperature under it.

A 10 W device (10 x 10 mm) is bolted under a carbon-steel block (40 x 40 x 10 mm); a fan blows over the top face, with
an unknown h. Nine thermocouples sit on the top face (+/-0.5 °C): the bottom, where the device is, cannot be reached.
The PINN learns T(x, y, z) and h together from Laplace's equation, the device flux on the bottom, the convective top,
the adiabatic sides and the nine readings, and so gives the temperature under the device, which nobody measured.

    T_xx + T_yy + T_zz = 0,   -k T_z = q''(x, y) at z = 0,   -k T_z = h (T - T_air) at z = H

Checks (the PINN never sees them): h vs the true value, the 3D field and the device temperature vs an independent
finite-volume solution (fv_reference.py), and the classical alternative of fitting h by re-running that model.

What the run shows: the field and the device temperature come out right (within 0.4 °C), but the network's own h
parameter settles ~7 % low, because the flux that pins h is a ~1 K gradient across 10 mm of steel and a small field
error is a large error on it. h from the energy balance on the learned field (P / integral of T - T_air over the top)
lands on the true value; both are reported.

Run:  python examples/use_cases/fin_convection_inverse/block_3d.py     (about 30 minutes on a laptop CPU)
Figures: python examples/use_cases/fin_convection_inverse/plots_2d3d.py   (the 3D render needs pyvista)
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from fv_reference import _footprint_area, block_3d, convergence  # noqa: E402
from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, PINNFactory, PINNProblemSpec  # noqa: E402

OUT = os.path.join(_HERE, "results", "3d")

LX, LY, LZ, K = 0.040, 0.040, 0.010, 50.0            # m, W/m.K (carbon steel)
POWER, T_AIR = 10.0, 25.0                             # W, °C
DEVICE = (0.020, 0.020, 0.005, 0.0025)                # centre x, centre y, half side, edge width (m)
H_TRUE, H_GUESS = 150.0, 40.0                         # W/m2.K: true (hidden, fan-cooled) and the starting value
SENSORS_MM = np.array([[x, y] for y in (8.0, 20.0, 32.0) for x in (8.0, 20.0, 32.0)])   # on the top face
NOISE_C, N_DRAWS = 0.5, 3
L, DT = LX, 50.0
FV = dict(lx=LX, ly=LY, lz=LZ, k=K, power=POWER, t_air=T_AIR, device=DEVICE)

Q_FLUX = POWER / _footprint_area(*DEVICE, LX, LY)
C_Q = Q_FLUX * L / (K * DT)                           # scaled bottom flux
C_T = L / K                                           # top Biot number = C_T * h
ZH = LZ / L                                           # scaled thickness


def source(xi, eta, *_):
    xc, yc, half, w = (v / L for v in DEVICE)
    sig = lambda s: 0.5 * (1 + torch.tanh(s / w))  # noqa: E731
    return sig(half - torch.abs(xi - xc)) * sig(half - torch.abs(eta - yc))


def top_readings(seed: int, ref):
    from scipy.interpolate import RegularGridInterpolator
    x, y, z, t = ref
    # the reference's top cell centres (0.42 mm under the surface); the PINN is given the same depth
    f = RegularGridInterpolator((y, x), t[-1], bounds_error=False, fill_value=None)
    rng = np.random.default_rng(seed)
    t_s = f(np.column_stack([SENSORS_MM[:, 1] / 1000, SENSORS_MM[:, 0] / 1000]))
    return t_s + rng.normal(0.0, NOISE_C, size=t_s.shape), float(z[-1])


class SideAdiabaticNet(torch.nn.Module):
    """MLP on (cos(pi xi), cos(pi eta), zeta): dT/dxi = 0 at xi = 0, 1 and dT/deta = 0 at eta = 0, 1 by construction,
    so the four adiabatic sides hold exactly instead of as four soft loss terms."""

    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers, d = [], 3
        for _ in range(depth):
            layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
            d = width
        self.mlp = torch.nn.Sequential(*layers, torch.nn.Linear(d, 1))

    def forward(self, x):
        return self.mlp(torch.cat([torch.cos(math.pi * x[:, :1]), torch.cos(math.pi * x[:, 1:2]), 4 * x[:, 2:3]], 1))


def identify(t_meas, z_sensor: float, seed: int, steps: int = 3000, lbfgs_iters: int = 800):
    torch.manual_seed(seed)
    d = lambda v: f"Derivative(T(xi, eta, zeta), {v})"  # noqa: E731
    spec = PINNProblemSpec(
        pde_residuals=[" + ".join(f"Derivative(T(xi, eta, zeta), ({v}, 2))" for v in ("xi", "eta", "zeta"))],
        conditions=[{"name": "device (bottom)", "equation": f"{d('zeta')} + c_q*S(xi, eta, zeta)", "weight": 10.0},
                    {"name": "fan (top)", "equation": f"{d('zeta')} + c_t*h*T(xi, eta, zeta)", "weight": 10.0}],
        independent_vars=["xi", "eta", "zeta"], dependent_vars=["T"], inverse_params=["h"],
        constants={"c_q": C_Q, "c_t": C_T}, exogenous={"S": source},
        loss_weights={"pde": 5.0, "conditions": 10.0, "data": 50.0},
    )
    loss_fn = PINNFactory(spec).generate_loss_function()
    net = SideAdiabaticNet()
    model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": H_GUESS})
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": 2e-3},
                            {"params": model.inverse_params.parameters(), "lr": 0.5}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1300, gamma=0.3)
    g = torch.Generator().manual_seed(seed)
    r = lambda n, hi=1.0: hi * torch.rand(n, 1, generator=g)  # noqa: E731

    def face(n, axis, value):
        pts = [r(n), r(n), r(n, ZH)]
        pts[axis] = torch.full((n, 1), value)
        return tuple(t.requires_grad_(True) for t in pts)

    def interior(n):
        m = n // 3                                                      # a third under/around the device
        xc, half = DEVICE[0] / L, DEVICE[2] / L
        xi = torch.cat([r(n - m), xc + (2 * r(m) - 1) * 1.8 * half]).clamp(0, 1)
        eta = torch.cat([r(n - m), xc + (2 * r(m) - 1) * 1.8 * half]).clamp(0, 1)
        zeta = torch.cat([r(n - m, ZH), r(m, ZH / 2)])
        return tuple(t.requires_grad_(True) for t in (xi, eta, zeta))

    def bottom(n):
        m = n // 2
        xc, half = DEVICE[0] / L, DEVICE[2] / L
        xi = torch.cat([r(n - m), xc + (2 * r(m) - 1) * 1.8 * half]).clamp(0, 1)
        eta = torch.cat([r(n - m), xc + (2 * r(m) - 1) * 1.8 * half]).clamp(0, 1)
        return tuple(t.requires_grad_(True) for t in (xi, eta, torch.zeros_like(xi)))

    xs = torch.tensor(SENSORS_MM / 1000 / L, dtype=torch.float32)
    data_in = (xs[:, :1], xs[:, 1:], torch.full((len(xs), 1), z_sensor / L))
    th_d = torch.tensor(((t_meas - T_AIR) / DT).reshape(-1, 1), dtype=torch.float32)
    history = []
    for step in range(steps + 1):
        if step % 250 == 0:
            conds = [bottom(600), face(400, 2, ZH)]
            batch = {"collocation": interior(2500), "conditions": conds, "data": (data_in, th_d)}
        opt.zero_grad(set_to_none=True)
        loss, comps = loss_fn(model, batch)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0:
            history.append((step, float(model.inverse_params["h"].detach())))
    # L-BFGS polish on a larger fixed batch: Adam leaves a residual that leaks ~10 % of the heat, and h follows it
    conds = [bottom(1500), face(800, 2, ZH)]
    batch = {"collocation": interior(8000), "conditions": conds, "data": (data_in, th_d)}
    lbfgs = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=lbfgs_iters, history_size=50,
                              line_search_fn="strong_wolfe", tolerance_grad=1e-9, tolerance_change=1e-12)
    it = [steps]

    def closure():
        lbfgs.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model, batch)
        loss.backward()
        it[0] += 1
        if it[0] % 25 == 0:
            history.append((it[0], float(model.inverse_params["h"].detach())))
        return loss
    lbfgs.step(closure)
    _, comps = loss_fn(model, batch)
    return model, float(model.inverse_params["h"].detach()), history, comps


def fit_with_fv(t_meas) -> tuple:
    from scipy.interpolate import RegularGridInterpolator
    n = [0]

    def sse(h):
        n[0] += 1
        x, y, z, t = block_3d(h, nx=20, ny=20, nz=6, **FV)
        f = RegularGridInterpolator((y, x), t[-1], bounds_error=False, fill_value=None)
        return float(np.sum((f(np.column_stack([SENSORS_MM[:, 1] / 1000, SENSORS_MM[:, 0] / 1000])) - t_meas) ** 2))
    a, b, gr = 20.0, 400.0, (math.sqrt(5) - 1) / 2
    for _ in range(25):
        c, d = b - gr * (b - a), a + gr * (b - a)
        a, b = (a, d) if sse(c) < sse(d) else (c, b)
    return (a + b) / 2, n[0]


def main():
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    ref = block_3d(H_TRUE, **FV)
    x, y, z, t_ref = ref
    grid_err = convergence(block_3d, dict(nx=20, ny=20, nz=6), dict(nx=40, ny=40, nz=12), h=H_TRUE, **FV)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    runs, t0 = [], time.time()
    for draw in range(N_DRAWS):
        t_meas, z_s = top_readings(100 + draw, ref)
        model, h_hat, hist, comps = identify(t_meas, z_s, seed=draw)
        with torch.no_grad():
            th = model(*(torch.tensor(a.reshape(-1, 1) / L, dtype=torch.float32) for a in (xx, yy, zz)))
        t_pinn = T_AIR + DT * th.numpy().reshape(t_ref.shape)
        h_fv, n_solves = fit_with_fv(t_meas)
        # energy balance on the learned field: the known power leaves through the top, h = P / integral of (T - T_air)
        n = 80
        g1 = (torch.arange(n, dtype=torch.float32) + 0.5) / n
        gx, gy = torch.meshgrid(g1, g1, indexing="ij")
        with torch.no_grad():
            top = model(gx.reshape(-1, 1), gy.reshape(-1, 1), torch.full((n * n, 1), ZH))
        h_energy = POWER / (float(top.mean()) * DT * LX * LY)
        runs.append({"draw": draw, "readings_C": t_meas.round(3).tolist(), "h_pinn": h_hat, "h_fv_fit": h_fv,
                     "h_energy_balance": h_energy,
                     "fv_solves": n_solves, "max_field_error_C": float(np.max(np.abs(t_pinn - t_ref))),
                     "device_max_pinn_C": float(t_pinn[0].max()), "top_max_pinn_C": float(t_pinn[-1].max()),
                     "history": hist, "final_loss": comps})
        np.save(os.path.join(OUT, f"field_pinn_{draw}.npy"), t_pinn.astype(np.float32))
        print(f"draw {draw}: h = {h_hat:6.1f} (energy balance {h_energy:6.1f}, FV fit {h_fv:6.1f}), device {t_pinn[0].max():.1f} °C "
              f"(ref {t_ref[0].max():.1f}), max field error {runs[-1]['max_field_error_C']:.2f} °C")
    np.save(os.path.join(OUT, "field_reference.npy"), t_ref.astype(np.float32))
    hp = np.array([r["h_pinn"] for r in runs])
    hf = np.array([r["h_fv_fit"] for r in runs])
    dv = np.array([r["device_max_pinn_C"] for r in runs])
    he = np.array([r["h_energy_balance"] for r in runs])
    summary = {
        "case": "3d_block",
        "setup": {"L_m": [LX, LY, LZ], "k_W_mK": K, "power_W": POWER, "T_air_C": T_AIR, "device": DEVICE,
                  "h_true_W_m2K": H_TRUE, "h_initial_guess": H_GUESS, "sensors_mm": SENSORS_MM.tolist(),
                  "sensor_depth_mm": round((LZ - z_s) * 1000, 3), "noise_C": NOISE_C, "draws": N_DRAWS},
        "reference": {"grid": [40, 40, 12], "grid_convergence_K": grid_err, "device_max_C": float(t_ref[0].max()),
                      "top_min_C": float(t_ref[-1].min()), "top_max_C": float(t_ref[-1].max())},
        "h_pinn": {"mean": float(hp.mean()), "std": float(hp.std(ddof=1))},
        "h_fv_fit": {"mean": float(hf.mean()), "std": float(hf.std(ddof=1)), "solves_per_fit": runs[0]["fv_solves"]},
        "h_energy_balance": {"mean": float(he.mean()), "std": float(he.std(ddof=1))},
        "device_max_pinn_C": {"mean": float(dv.mean()), "std": float(dv.std(ddof=1))},
        "device_max_if_h_guess_C": float(block_3d(H_GUESS, **FV)[3][0].max()),
        "max_field_error_C": float(max(r["max_field_error_C"] for r in runs)),
        "seconds": round(time.time() - t0, 1), "runs": runs,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nh = {hp.mean():.1f} ± {hp.std(ddof=1):.1f} (true {H_TRUE}); FV fit {hf.mean():.1f} ± {hf.std(ddof=1):.1f}; "
          f"device {dv.mean():.1f} ± {dv.std(ddof=1):.1f} °C (ref {t_ref[0].max():.1f}); worst field error "
          f"{summary['max_field_error_C']:.2f} °C; {summary['seconds']} s")


if __name__ == "__main__":
    main()
