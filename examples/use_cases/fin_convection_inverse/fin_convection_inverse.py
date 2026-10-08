"""How hot is the air side? Find the convection coefficient h of a pin fin from five thermocouples.

A stainless-steel pin fin (k = 16 W/m.K, D = 5 mm, L = 50 mm) sits on an 80 °C wall in 25 °C air. Nobody knows the
convection coefficient h: textbook correlations can easily be off by 20-30 % in a real installation. Five thermocouples along the
fin (+/-0.5 °C noise) are all we have. A physics-informed network learns the temperature field and h together, from the
fin equation and the readings, without needing the analytic solution.

The fin does have an analytic solution, which is why it was chosen: it lets us check every number.

    theta'' - m^2 theta = 0,        m^2 = 4 h / (k D),      theta = (T - T_inf) / (T_b - T_inf)
    theta(0) = 1                    (base)
    -k theta'(L) = h theta(L)       (convective tip)

What is checked, over 10 independent noise draws of the thermocouple readings:
  * h identified vs the true h (hidden from the network),
  * the reconstructed temperature field vs the analytic one, everywhere along the fin,
  * the heat the fin dissipates, Q = -k A dT/dx at the base (from the network's own derivative), vs the analytic Q,
  * a classical least-squares fit of the analytic profile to the same readings, as a reference. It needs the formula;
    the PINN does not, so the same script works for geometries that have no formula.

Run:  python examples/use_cases/fin_convection_inverse/fin_convection_inverse.py   (about 5 minutes on a laptop CPU)
Writes results/summary.json and the figures in results/.
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
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, NeuralNetwork, PINNFactory, PINNProblemSpec  # noqa: E402

OUT = os.path.join(_HERE, "results")

# --------------------------------------------------------------------------- the fin (SI units)
K, D, L = 16.0, 0.005, 0.050            # W/m.K (stainless steel), m, m
T_BASE, T_AIR = 80.0, 25.0              # °C
H_TRUE = 25.0                           # W/m2.K: hidden from the network, used only to make the readings
H_GUESS = 100.0                         # where the network starts (4x off)
SENSORS_MM = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
NOISE_C = 0.5                           # thermocouple noise (1 sigma), °C
N_DRAWS = 10                            # independent noise draws
AREA, PERIM = math.pi * D**2 / 4, math.pi * D

C_FIN = 4 * L**2 / (K * D)              # (mL)^2 = C_FIN * h, in the coordinate xi = x/L
C_TIP = L / K                           # tip Biot number = C_TIP * h


def theta_exact(xi: np.ndarray, h: float) -> np.ndarray:
    """Analytic fin profile with a convective tip (Incropera, Table 3.4, case A)."""
    m = math.sqrt(h * PERIM / (K * AREA))
    r = h / (m * K)
    return (np.cosh(m * L * (1 - xi)) + r * np.sinh(m * L * (1 - xi))) / (np.cosh(m * L) + r * np.sinh(m * L))


def q_exact(h: float) -> float:
    """Heat dissipated by the fin, W."""
    m = math.sqrt(h * PERIM / (K * AREA))
    r = h / (m * K)
    return math.sqrt(h * PERIM * K * AREA) * (T_BASE - T_AIR) * (math.sinh(m * L) + r * math.cosh(m * L)) / (
        math.cosh(m * L) + r * math.sinh(m * L))


def readings(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = T_AIR + (T_BASE - T_AIR) * theta_exact(SENSORS_MM / 1000 / L, H_TRUE)
    return t + rng.normal(0.0, NOISE_C, size=t.shape)


# --------------------------------------------------------------------------- PINN inverse problem
def identify(t_meas: np.ndarray, seed: int, steps: int = 2500):
    """Learn theta(xi) and h from the fin equation, the base and tip conditions and the five readings."""
    torch.manual_seed(seed)
    spec = PINNProblemSpec(
        pde_residuals=["Derivative(T(xi), (xi, 2)) - c_fin*h*T(xi)"],
        conditions=[{"name": "base", "equation": "T(xi) - 1", "weight": 1.0},
                    {"name": "tip", "equation": "Derivative(T(xi), xi) + c_tip*h*T(xi)", "weight": 1.0}],
        independent_vars=["xi"], dependent_vars=["T"], inverse_params=["h"],
        constants={"c_fin": C_FIN, "c_tip": C_TIP},
        loss_weights={"pde": 1.0, "conditions": 10.0, "data": 100.0},
    )
    loss_fn = PINNFactory(spec).generate_loss_function()
    net = NeuralNetwork(num_inputs=1, num_outputs=1, num_layers=3, num_neurons=32, activation=torch.nn.Tanh())
    model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": H_GUESS})
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": 3e-3},
                            {"params": model.inverse_params.parameters(), "lr": 0.5}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1000, gamma=0.3)

    xi_col = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)
    xi_base = torch.zeros((1, 1), requires_grad=True)
    xi_tip = torch.ones((1, 1), requires_grad=True)
    xi_d = torch.tensor(SENSORS_MM / 1000 / L, dtype=torch.float32).reshape(-1, 1)
    th_d = torch.tensor((t_meas - T_AIR) / (T_BASE - T_AIR), dtype=torch.float32).reshape(-1, 1)
    batch = {"collocation": (xi_col,), "conditions": [(xi_base,), (xi_tip,)], "data": ((xi_d,), th_d)}

    history = []
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, comps = loss_fn(model, batch)
        loss.backward()
        opt.step()
        sched.step()
        if step % 50 == 0:
            history.append((step, float(model.inverse_params["h"].detach())))
    h_hat = float(model.inverse_params["h"].detach())

    xi = torch.linspace(0, 1, 401).reshape(-1, 1).requires_grad_(True)
    th = model(xi)
    dth = torch.autograd.grad(th.sum(), xi)[0]
    q_pinn = float(-K * AREA * (T_BASE - T_AIR) / L * dth[0, 0])            # Fourier's law at the base
    return h_hat, xi.detach().numpy().ravel(), th.detach().numpy().ravel(), q_pinn, history, comps


def least_squares_fit(t_meas: np.ndarray) -> float:
    """Reference: fit h in the analytic profile to the readings (golden-section search on the squared error)."""
    xi_s = SENSORS_MM / 1000 / L
    sse = lambda h: float(np.sum((T_AIR + (T_BASE - T_AIR) * theta_exact(xi_s, h) - t_meas) ** 2))  # noqa: E731
    a, b, g = 1.0, 200.0, (math.sqrt(5) - 1) / 2
    for _ in range(100):
        c, d = b - g * (b - a), a + g * (b - a)
        a, b = (a, d) if sse(c) < sse(d) else (c, b)
    return (a + b) / 2


def main() -> None:
    torch.set_num_threads(1)               # a 1-input network: threads cost more than they save
    os.makedirs(OUT, exist_ok=True)
    runs = []
    t0 = time.time()
    for draw in range(N_DRAWS):
        t_meas = readings(seed=100 + draw)
        h_hat, xi, th, q, hist, comps = identify(t_meas, seed=draw)
        exact = theta_exact(xi, H_TRUE)
        field_err = float(np.max(np.abs(th - exact)) * (T_BASE - T_AIR))                   # °C, worst point
        runs.append({"draw": draw, "readings_C": t_meas.round(3).tolist(), "h_pinn": h_hat,
                     "h_least_squares": least_squares_fit(t_meas), "q_pinn_W": q, "max_field_error_C": field_err,
                     "history": hist, "profile": {"xi": xi[::4].tolist(), "theta": th[::4].tolist()},
                     "final_loss": comps})
        print(f"draw {draw}: h = {h_hat:6.2f} W/m2K (LSQ {runs[-1]['h_least_squares']:6.2f}), "
              f"Q = {q:.3f} W, max field error {field_err:.2f} °C")
    hp = np.array([r["h_pinn"] for r in runs])
    hl = np.array([r["h_least_squares"] for r in runs])
    qp = np.array([r["q_pinn_W"] for r in runs])
    summary = {
        "setup": {"k_W_mK": K, "D_m": D, "L_m": L, "T_base_C": T_BASE, "T_air_C": T_AIR, "h_true_W_m2K": H_TRUE,
                  "h_initial_guess": H_GUESS, "sensors_mm": SENSORS_MM.tolist(), "noise_C": NOISE_C,
                  "draws": N_DRAWS, "mL": math.sqrt(C_FIN * H_TRUE)},
        "h_pinn": {"mean": float(hp.mean()), "std": float(hp.std(ddof=1)), "min": float(hp.min()), "max": float(hp.max())},
        "h_least_squares": {"mean": float(hl.mean()), "std": float(hl.std(ddof=1))},
        "q_true_W": q_exact(H_TRUE),
        "q_pinn_W": {"mean": float(qp.mean()), "std": float(qp.std(ddof=1))},
        "q_if_h_guess_W": q_exact(H_GUESS),
        "max_field_error_C": float(max(r["max_field_error_C"] for r in runs)),
        "seconds": round(time.time() - t0, 1),
        "runs": runs,
    }
    with open(os.path.join(OUT, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"\nh = {hp.mean():.2f} ± {hp.std(ddof=1):.2f} W/m2K (true {H_TRUE}); least squares "
          f"{hl.mean():.2f} ± {hl.std(ddof=1):.2f}; Q = {qp.mean():.3f} ± {qp.std(ddof=1):.3f} W "
          f"(true {q_exact(H_TRUE):.3f}); worst field error {summary['max_field_error_C']:.2f} °C; "
          f"{summary['seconds']} s")
    try:
        from plots import make_figures  # noqa: WPS433 - optional, needs matplotlib
        make_figures(summary, OUT)
    except ImportError as exc:
        print(f"figures skipped ({exc})")


if __name__ == "__main__":
    sys.path.insert(0, _HERE)
    main()
