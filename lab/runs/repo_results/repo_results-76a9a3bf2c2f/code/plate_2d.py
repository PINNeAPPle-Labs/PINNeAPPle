"""2D: convection coefficient of a heat-spreader plate from eight thermocouples, and the hot spot it implies.

An aluminium plate (100 x 60 x 1 mm) carries an 8 W device (20 x 20 mm) and cools by natural convection from both
faces; h is unknown. Eight thermocouples on the plate (+/-0.5 °C). There is no formula for this field: the PINN learns
T(x, y) and h together from the plate equation, the adiabatic edges and the readings.

    k t (T_xx + T_yy) - 2 h (T - T_air) + q''(x, y) = 0

Checks (the PINN never sees them): h vs the true value, the temperature map and the hot spot vs an independent
finite-volume solution (fv_reference.py, grid-converged to 0.01 K), and the classical alternative, fitting h by
re-running the finite-volume model until it matches the readings (needs a simulation model and dozens of solves).

Run:  python examples/use_cases/fin_convection_inverse/plate_2d.py     (about 20 minutes on a laptop CPU)
Figures: python examples/use_cases/fin_convection_inverse/plots_2d3d.py
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

from fv_reference import _footprint_area, convergence, interp_grid, plate_2d  # noqa: E402
from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, NeuralNetwork, PINNFactory, PINNProblemSpec  # noqa: E402

OUT = os.path.join(_HERE, "results", "2d")

LX, LY, THK, K = 0.100, 0.060, 0.001, 200.0          # m, m, m, W/m.K (aluminium)
POWER, T_AIR = 8.0, 25.0                              # W, °C
DEVICE = (0.030, 0.030, 0.010, 0.002)                 # centre x, centre y, half side, edge width (m)
H_TRUE, H_GUESS = 15.0, 60.0                          # W/m2.K: true (hidden) and the network's starting value
SENSORS_MM = np.array([[10, 10], [50, 10], [90, 10], [70, 30], [90, 50], [50, 50], [10, 50], [30, 55]], float)
NOISE_C, N_DRAWS = 0.5, 5
L, DT = LX, 50.0                                      # length and temperature scales
FV = dict(lx=LX, ly=LY, t=THK, k=K, power=POWER, t_air=T_AIR, device=DEVICE)

Q_FLUX = POWER / _footprint_area(*DEVICE, LX, LY)     # W/m2 under the device
C_H = 2 * L**2 / (K * THK)                            # coefficient of h in the scaled equation
C_Q = Q_FLUX * L**2 / (K * THK * DT)                  # scaled source


def source(xi, eta):
    """Scaled device footprint as a function of the network inputs (torch)."""
    xc, yc, half, w = (v / L for v in DEVICE)
    sig = lambda s: 0.5 * (1 + torch.tanh(s / w))  # noqa: E731
    return sig(half - torch.abs(xi - xc)) * sig(half - torch.abs(eta - yc))


def readings(seed: int, field):
    x, y, t = field
    rng = np.random.default_rng(seed)
    t_s = interp_grid(x, y, t, SENSORS_MM[:, 0] / 1000, SENSORS_MM[:, 1] / 1000)
    return t_s + rng.normal(0.0, NOISE_C, size=t_s.shape)


def identify(t_meas, seed: int, steps: int = 3000):
    torch.manual_seed(seed)
    ar = LY / L
    spec = PINNProblemSpec(
        pde_residuals=["Derivative(T(xi, eta), (xi, 2)) + Derivative(T(xi, eta), (eta, 2)) - c_h*h*T(xi, eta)"
                       " + c_q*S(xi, eta)"],
        conditions=[{"name": n, "equation": e, "weight": 1.0} for n, e in (
            ("left", "Derivative(T(xi, eta), xi)"), ("right", "Derivative(T(xi, eta), xi)"),
            ("bottom", "Derivative(T(xi, eta), eta)"), ("top", "Derivative(T(xi, eta), eta)"))],
        independent_vars=["xi", "eta"], dependent_vars=["T"], inverse_params=["h"],
        constants={"c_h": C_H, "c_q": C_Q}, exogenous={"S": source},
        loss_weights={"pde": 1.0, "conditions": 1.0, "data": 50.0},
    )
    loss_fn = PINNFactory(spec).generate_loss_function()
    net = NeuralNetwork(num_inputs=2, num_outputs=1, num_layers=4, num_neurons=48, activation=torch.nn.Tanh())
    model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": H_GUESS})
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": 2e-3},
                            {"params": model.inverse_params.parameters(), "lr": 0.2}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.3)
    g = torch.Generator().manual_seed(seed)

    def col(n):
        """Uniform points plus a third of them around the device, where the source changes fastest."""
        m = n // 3
        xc, yc, half = DEVICE[0] / L, DEVICE[1] / L, DEVICE[2] / L
        xi = torch.cat([torch.rand(n - m, 1, generator=g), xc + (2 * torch.rand(m, 1, generator=g) - 1) * 1.6 * half])
        eta = torch.cat([ar * torch.rand(n - m, 1, generator=g), yc + (2 * torch.rand(m, 1, generator=g) - 1) * 1.6 * half])
        return (xi.clamp(0, 1).requires_grad_(True), eta.clamp(0, ar).requires_grad_(True))

    def edge(n, fixed, value):
        s = torch.rand(n, 1, generator=g)
        v = torch.full((n, 1), value)
        return ((v, s * ar) if fixed == "xi" else (s, v))

    xi_d = torch.tensor(SENSORS_MM[:, :1] / 1000 / L, dtype=torch.float32)
    eta_d = torch.tensor(SENSORS_MM[:, 1:] / 1000 / L, dtype=torch.float32)
    th_d = torch.tensor(((t_meas - T_AIR) / DT).reshape(-1, 1), dtype=torch.float32)
    history = []
    for step in range(steps + 1):
        if step % 250 == 0:                                            # fresh collocation points
            conds = [tuple(t.requires_grad_(True) for t in e) for e in
                     (edge(200, "xi", 0.0), edge(200, "xi", 1.0), edge(200, "eta", 0.0), edge(200, "eta", ar))]
            batch = {"collocation": col(3000), "conditions": conds, "data": ((xi_d, eta_d), th_d)}
        opt.zero_grad(set_to_none=True)
        loss, comps = loss_fn(model, batch)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0:
            history.append((step, float(model.inverse_params["h"].detach())))
    return model, float(model.inverse_params["h"].detach()), history, comps


def fit_with_fv(t_meas) -> tuple:
    """Classical inverse: golden-section search on h, one finite-volume solve per evaluation."""
    n = [0]

    def sse(h):
        n[0] += 1
        x, y, t = plate_2d(h, nx=100, ny=60, **FV)
        return float(np.sum((interp_grid(x, y, t, SENSORS_MM[:, 0] / 1000, SENSORS_MM[:, 1] / 1000) - t_meas) ** 2))
    a, b, gr = 2.0, 80.0, (math.sqrt(5) - 1) / 2
    for _ in range(30):
        c, d = b - gr * (b - a), a + gr * (b - a)
        a, b = (a, d) if sse(c) < sse(d) else (c, b)
    return (a + b) / 2, n[0]


def main():
    torch.set_num_threads(2)
    os.makedirs(OUT, exist_ok=True)
    ref = plate_2d(H_TRUE, **FV)                                       # 200 x 120 cells
    x, y, t_ref = ref
    grid_err = convergence(plate_2d, dict(nx=100, ny=60), dict(nx=200, ny=120), h=H_TRUE, **FV)
    xx, yy = np.meshgrid(x, y)
    runs, t0 = [], time.time()
    for draw in range(N_DRAWS):
        t_meas = readings(100 + draw, ref)
        model, h_hat, hist, comps = identify(t_meas, seed=draw)
        with torch.no_grad():
            th = model(torch.tensor(xx.reshape(-1, 1) / L, dtype=torch.float32),
                       torch.tensor(yy.reshape(-1, 1) / L, dtype=torch.float32)).numpy().reshape(t_ref.shape)
        t_pinn = T_AIR + DT * th
        h_fv, n_solves = fit_with_fv(t_meas)
        runs.append({"draw": draw, "readings_C": t_meas.round(3).tolist(), "h_pinn": h_hat, "h_fv_fit": h_fv,
                     "fv_solves": n_solves, "max_field_error_C": float(np.max(np.abs(t_pinn - t_ref))),
                     "hot_spot_pinn_C": float(t_pinn.max()), "history": hist, "final_loss": comps})
        np.save(os.path.join(OUT, f"field_pinn_{draw}.npy"), t_pinn[::2, ::2].astype(np.float32))
        print(f"draw {draw}: h = {h_hat:5.2f} (FV fit {h_fv:5.2f}), hot spot {t_pinn.max():.1f} °C "
              f"(ref {t_ref.max():.1f}), max field error {runs[-1]['max_field_error_C']:.2f} °C")
    np.save(os.path.join(OUT, "field_reference.npy"), t_ref[::2, ::2].astype(np.float32))
    hp = np.array([r["h_pinn"] for r in runs])
    hf = np.array([r["h_fv_fit"] for r in runs])
    hs = np.array([r["hot_spot_pinn_C"] for r in runs])
    summary = {
        "case": "2d_plate",
        "setup": {"Lx_m": LX, "Ly_m": LY, "thickness_m": THK, "k_W_mK": K, "power_W": POWER, "T_air_C": T_AIR,
                  "device": DEVICE, "h_true_W_m2K": H_TRUE, "h_initial_guess": H_GUESS,
                  "sensors_mm": SENSORS_MM.tolist(), "noise_C": NOISE_C, "draws": N_DRAWS},
        "reference": {"grid": [200, 120], "grid_convergence_K": grid_err, "hot_spot_C": float(t_ref.max()),
                      "min_C": float(t_ref.min())},
        "h_pinn": {"mean": float(hp.mean()), "std": float(hp.std(ddof=1))},
        "h_fv_fit": {"mean": float(hf.mean()), "std": float(hf.std(ddof=1)), "solves_per_fit": runs[0]["fv_solves"]},
        "hot_spot_pinn_C": {"mean": float(hs.mean()), "std": float(hs.std(ddof=1))},
        "hot_spot_if_h_guess_C": float(plate_2d(H_GUESS, **FV)[2].max()),
        "max_field_error_C": float(max(r["max_field_error_C"] for r in runs)),
        "seconds": round(time.time() - t0, 1), "runs": runs,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nh = {hp.mean():.2f} ± {hp.std(ddof=1):.2f} (true {H_TRUE}); FV fit {hf.mean():.2f} ± {hf.std(ddof=1):.2f}; "
          f"hot spot {hs.mean():.1f} ± {hs.std(ddof=1):.1f} °C (ref {t_ref.max():.1f}); worst field error "
          f"{summary['max_field_error_C']:.2f} °C; {summary['seconds']} s")


if __name__ == "__main__":
    main()
