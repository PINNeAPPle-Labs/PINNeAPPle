"""Inverse heat transfer with PINNeAPPle: the convection coefficient h from a few thermocouples.

1D pin fin and 2D heat-spreader plate are trained on request (the fin also accepts the user's own readings); the 3D
block takes ~30 minutes to train, so its result (examples/use_cases/fin_convection_inverse/results/3d) is served as is.
Every number is checked: the fin against its analytic solution, the plate and the block against an independent
finite-volume solver (examples/use_cases/fin_convection_inverse/fv_reference.py).
"""
from __future__ import annotations

import json
import math
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch

from pinneapple_physics.pinn_solver.factory.pinn_factory import PINN, NeuralNetwork, PINNFactory, PINNProblemSpec

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
EXAMPLE = os.path.join(_ROOT, "examples", "use_cases", "fin_convection_inverse")
if EXAMPLE not in sys.path:
    sys.path.insert(0, EXAMPLE)

from fv_reference import _footprint_area, interp_grid, plate_2d  # noqa: E402

__all__ = ["FinInput", "run_fin", "run_plate", "plate_precomputed", "block_precomputed", "fin_exact", "fin_q_exact"]

Progress = Optional[Callable[[int, int, float], None]]       # (step, total, current h)


class InputError(ValueError):
    pass


# --------------------------------------------------------------------------- 1D pin fin
def fin_exact(xi: np.ndarray, h: float, k: float, d: float, length: float) -> np.ndarray:
    """theta = (T - T_air) / (T_base - T_air) of a pin fin with a convective tip (Incropera, Table 3.4 case A)."""
    area, perim = math.pi * d * d / 4, math.pi * d
    m = math.sqrt(h * perim / (k * area))
    r = h / (m * k)
    return (np.cosh(m * length * (1 - xi)) + r * np.sinh(m * length * (1 - xi))) / (
        np.cosh(m * length) + r * np.sinh(m * length))


def fin_q_exact(h: float, k: float, d: float, length: float, dt: float) -> float:
    area, perim = math.pi * d * d / 4, math.pi * d
    m = math.sqrt(h * perim / (k * area))
    r = h / (m * k)
    return math.sqrt(h * perim * k * area) * dt * (math.sinh(m * length) + r * math.cosh(m * length)) / (
        math.cosh(m * length) + r * math.sinh(m * length))


def _golden(f: Callable[[float], float], a: float, b: float, n: int = 80) -> float:
    g = (math.sqrt(5) - 1) / 2
    for _ in range(n):
        c, d = b - g * (b - a), a + g * (b - a)
        a, b = (a, d) if f(c) < f(d) else (c, b)
    return (a + b) / 2


def run_fin(*, k: float, d_mm: float, length_mm: float, t_base: float, t_air: float, sensors_mm: Sequence[float],
            readings: Sequence[float], h_guess: float = 100.0, steps: int = 2000, seed: int = 0,
            progress: Progress = None, h_true: Optional[float] = None) -> Dict:
    """Learn the temperature profile and h of a pin fin from thermocouple readings (°C at ``sensors_mm``)."""
    d, length = d_mm / 1000, length_mm / 1000
    xs = np.asarray(sensors_mm, float)
    tm = np.asarray(readings, float)
    if not (k > 0 and d > 0 and length > 0):
        raise InputError("conductivity, diameter and length must be positive")
    if len(xs) != len(tm) or not 2 <= len(xs) <= 12:
        raise InputError("give between 2 and 12 thermocouples, each with a position and a reading")
    if np.any(xs <= 0) or np.any(xs > length_mm):
        raise InputError(f"thermocouple positions must be in (0, {length_mm:g}] mm from the wall")
    if abs(t_base - t_air) < 1:
        raise InputError("the wall and the air must differ by at least 1 °C")
    span = t_base - t_air
    theta = (tm - t_air) / span
    if np.any(theta < -0.1) or np.any(theta > 1.1):
        raise InputError("every reading must lie between the air and the wall temperature")

    torch.manual_seed(seed)
    c_fin, c_tip = 4 * length**2 / (k * d), length / k
    spec = PINNProblemSpec(
        pde_residuals=["Derivative(T(xi), (xi, 2)) - c_fin*h*T(xi)"],
        conditions=[{"name": "wall", "equation": "T(xi) - 1"},
                    {"name": "tip", "equation": "Derivative(T(xi), xi) + c_tip*h*T(xi)"}],
        independent_vars=["xi"], dependent_vars=["T"], inverse_params=["h"],
        constants={"c_fin": c_fin, "c_tip": c_tip},
        loss_weights={"pde": 1.0, "conditions": 10.0, "data": 100.0},
    )
    loss_fn = PINNFactory(spec).generate_loss_function()
    net = NeuralNetwork(num_inputs=1, num_outputs=1, num_layers=3, num_neurons=32, activation=torch.nn.Tanh())
    model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": h_guess})
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": 3e-3},
                            {"params": model.inverse_params.parameters(), "lr": 0.005 * h_guess}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps * 2 // 5), gamma=0.3)
    xi_col = torch.linspace(0, 1, 200).reshape(-1, 1).requires_grad_(True)
    batch = {"collocation": (xi_col,),
             "conditions": [(torch.zeros((1, 1), requires_grad=True),), (torch.ones((1, 1), requires_grad=True),)],
             "data": ((torch.tensor(xs / length_mm, dtype=torch.float32).reshape(-1, 1),),
                      torch.tensor(theta, dtype=torch.float32).reshape(-1, 1))}
    history = []
    for step in range(steps + 1):
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model, batch)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            model.inverse_params["h"].clamp_(min=1e-3)
        if step % 20 == 0:
            h_now = float(model.inverse_params["h"].detach())
            history.append([step, h_now])
            if progress:
                progress(step, steps, h_now)
    h_hat = float(model.inverse_params["h"].detach())
    xi = torch.linspace(0, 1, 201).reshape(-1, 1).requires_grad_(True)
    th = model(xi)
    dth = torch.autograd.grad(th.sum(), xi)[0]
    area = math.pi * d * d / 4
    q_pinn = float(-k * area * span / length * dth[0, 0])
    xi_np, th_np = xi.detach().numpy().ravel(), th.detach().numpy().ravel()

    # cross-check: least squares on the analytic profile (needs the formula, which exists for this fin)
    sse = lambda h: float(np.sum((t_air + span * fin_exact(xs / length_mm, h, k, d, length) - tm) ** 2))  # noqa: E731
    h_lsq = _golden(sse, 0.01, max(2000.0, 10 * h_guess))
    out = {
        "h_pinn": h_hat, "h_least_squares": h_lsq, "q_pinn_W": q_pinn,
        "q_least_squares_W": fin_q_exact(h_lsq, k, d, length, span),
        "x_mm": (xi_np * length_mm).round(3).tolist(), "T_pinn": (t_air + span * th_np).round(3).tolist(),
        "T_least_squares": (t_air + span * fin_exact(xi_np, h_lsq, k, d, length)).round(3).tolist(),
        "sensors_mm": xs.tolist(), "readings": tm.round(3).tolist(), "history": history,
        "mL": math.sqrt(c_fin * h_hat), "biot_cross_section": h_hat * d / (2 * k),
    }
    if h_true is not None:
        out.update(h_true=h_true, q_true_W=fin_q_exact(h_true, k, d, length, span),
                   T_true=(t_air + span * fin_exact(xi_np, h_true, k, d, length)).round(3).tolist())
    return out


def fin_synthetic_readings(*, k, d_mm, length_mm, t_base, t_air, sensors_mm, h_true, noise, seed=0) -> List[float]:
    rng = np.random.default_rng(seed)
    xs = np.asarray(sensors_mm, float)
    t = t_air + (t_base - t_air) * fin_exact(xs / length_mm, h_true, k, d_mm / 1000, length_mm / 1000)
    return (t + rng.normal(0, noise, size=t.shape)).round(2).tolist()


# --------------------------------------------------------------------------- 2D plate
PLATE = dict(lx=0.100, ly=0.060, t=0.001, k=200.0, t_air=25.0, device=(0.030, 0.030, 0.010, 0.002))
PLATE_SENSORS_MM = [[10, 10], [50, 10], [90, 10], [70, 30], [90, 50], [50, 50], [10, 50], [30, 55]]


def run_plate(*, power: float, h_true: float, noise: float, h_guess: float = 60.0, steps: int = 1500, seed: int = 0,
              progress: Progress = None) -> Dict:
    """Synthetic readings from the finite-volume model, then the PINN learns T(x, y) and h; compared to the model."""
    if not (0.5 <= power <= 50 and 2 <= h_true <= 200 and 0 <= noise <= 3 and 1 <= h_guess <= 500):
        raise InputError("power 0.5-50 W, true h 2-200 W/m2K, noise 0-3 °C, starting h 1-500")
    lx, ly, thk, k, t_air = PLATE["lx"], PLATE["ly"], PLATE["t"], PLATE["k"], PLATE["t_air"]
    dev = PLATE["device"]
    fv = dict(lx=lx, ly=ly, t=thk, k=k, power=power, t_air=t_air, device=dev)
    x, y, t_ref = plate_2d(h_true, nx=100, ny=60, **fv)
    s = np.asarray(PLATE_SENSORS_MM, float)
    rng = np.random.default_rng(seed + 100)
    t_meas = interp_grid(x, y, t_ref, s[:, 0] / 1000, s[:, 1] / 1000) + rng.normal(0, noise, len(s))
    length, dtemp = lx, max(10.0, float(t_ref.max() - t_air))
    q_flux = power / _footprint_area(*dev, lx, ly)
    c_h, c_q = 2 * length**2 / (k * thk), q_flux * length**2 / (k * thk * dtemp)
    ar = ly / length

    def source(xi, eta):
        xc, yc, half, w = (v / length for v in dev)
        sig = lambda z: 0.5 * (1 + torch.tanh(z / w))  # noqa: E731
        return sig(half - torch.abs(xi - xc)) * sig(half - torch.abs(eta - yc))

    torch.manual_seed(seed)
    dxi, deta = "Derivative(T(xi, eta), xi)", "Derivative(T(xi, eta), eta)"
    spec = PINNProblemSpec(
        pde_residuals=["Derivative(T(xi, eta), (xi, 2)) + Derivative(T(xi, eta), (eta, 2)) - c_h*h*T(xi, eta)"
                       " + c_q*S(xi, eta)"],
        conditions=[{"name": n, "equation": e} for n, e in (("left", dxi), ("right", dxi), ("bottom", deta),
                                                          ("top", deta))],
        independent_vars=["xi", "eta"], dependent_vars=["T"], inverse_params=["h"],
        constants={"c_h": c_h, "c_q": c_q}, exogenous={"S": source},
        loss_weights={"pde": 1.0, "conditions": 1.0, "data": 50.0},
    )
    loss_fn = PINNFactory(spec).generate_loss_function()
    net = NeuralNetwork(num_inputs=2, num_outputs=1, num_layers=4, num_neurons=48, activation=torch.nn.Tanh())
    model = PINN(net, inverse_params_names=["h"], initial_guesses={"h": h_guess})
    opt = torch.optim.Adam([{"params": net.parameters(), "lr": 2e-3},
                            {"params": model.inverse_params.parameters(), "lr": 0.2 * h_guess / 60}])
    sched = torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, steps // 3), gamma=0.3)
    g = torch.Generator().manual_seed(seed)
    xc, yc, half = dev[0] / length, dev[1] / length, dev[2] / length

    def col(n):
        m = n // 3
        xi = torch.cat([torch.rand(n - m, 1, generator=g), xc + (2 * torch.rand(m, 1, generator=g) - 1) * 1.6 * half])
        eta = torch.cat([ar * torch.rand(n - m, 1, generator=g), yc + (2 * torch.rand(m, 1, generator=g) - 1) * 1.6 * half])
        return xi.clamp(0, 1).requires_grad_(True), eta.clamp(0, ar).requires_grad_(True)

    def edge(n, fixed, value):
        sv, v = torch.rand(n, 1, generator=g), torch.full((n, 1), value)
        pts = (v, sv * ar) if fixed == "xi" else (sv, v)
        return tuple(p.requires_grad_(True) for p in pts)

    data = ((torch.tensor(s[:, :1] / 1000 / length, dtype=torch.float32),
             torch.tensor(s[:, 1:] / 1000 / length, dtype=torch.float32)),
            torch.tensor(((t_meas - t_air) / dtemp).reshape(-1, 1), dtype=torch.float32))
    history = []
    for step in range(steps + 1):
        if step % 250 == 0:
            batch = {"collocation": col(2500), "data": data,
                     "conditions": [edge(150, "xi", 0.0), edge(150, "xi", 1.0), edge(150, "eta", 0.0),
                                    edge(150, "eta", ar)]}
        opt.zero_grad(set_to_none=True)
        loss, _ = loss_fn(model, batch)
        loss.backward()
        opt.step()
        sched.step()
        with torch.no_grad():
            model.inverse_params["h"].clamp_(min=0.1)
        if step % 25 == 0:
            h_now = float(model.inverse_params["h"].detach())
            history.append([step, h_now])
            if progress:
                progress(step, steps, h_now)
    xx, yy = np.meshgrid(x, y)
    with torch.no_grad():
        th = model(torch.tensor(xx.reshape(-1, 1) / length, dtype=torch.float32),
                   torch.tensor(yy.reshape(-1, 1) / length, dtype=torch.float32)).numpy().reshape(t_ref.shape)
    t_pinn = t_air + dtemp * th
    h_hat = float(model.inverse_params["h"].detach())
    return {"h_pinn": h_hat, "h_true": h_true, "power_W": power, "steps": steps,
            "hot_spot_pinn": float(t_pinn.max()), "hot_spot_reference": float(t_ref.max()),
            "max_field_error": float(np.max(np.abs(t_pinn - t_ref))),
            "field_pinn": t_pinn.round(2).tolist(), "field_reference": t_ref.round(2).tolist(),
            "x_mm": (x * 1000).round(2).tolist(), "y_mm": (y * 1000).round(2).tolist(),
            "sensors_mm": PLATE_SENSORS_MM, "readings": np.round(t_meas, 2).tolist(), "history": history,
            "device_mm": [v * 1000 for v in dev[:3]],
            "hot_spot_if_guess": float(plate_2d(h_guess, nx=100, ny=60, **fv)[2].max())}


# --------------------------------------------------------------------------- precomputed full runs
def _summary(case: str) -> Dict:
    with open(os.path.join(EXAMPLE, "results", case, "summary.json")) as fh:
        return json.load(fh)


def _typical(summary: Dict) -> int:
    hp = summary["h_pinn"]["mean"]
    return min(range(len(summary["runs"])), key=lambda i: abs(summary["runs"][i]["h_pinn"] - hp))


def plate_precomputed() -> Dict:
    s = _summary("2d")
    i = _typical(s)
    field = np.load(os.path.join(EXAMPLE, "results", "2d", f"field_pinn_{i}.npy"))
    ref = np.load(os.path.join(EXAMPLE, "results", "2d", "field_reference.npy"))
    st = s["setup"]
    ny, nx = field.shape
    return {"summary": {k: v for k, v in s.items() if k != "runs"}, "run": s["runs"][i],
            "field_pinn": field.round(2).tolist(), "field_reference": ref.round(2).tolist(),
            "x_mm": ((np.arange(nx) + 0.5) * st["Lx_m"] * 1000 / nx).round(2).tolist(),
            "y_mm": ((np.arange(ny) + 0.5) * st["Ly_m"] * 1000 / ny).round(2).tolist(),
            "draws": [{"h_pinn": r["h_pinn"], "hot_spot": r["hot_spot_pinn_C"], "h_fv_fit": r["h_fv_fit"]}
                      for r in s["runs"]]}


def block_precomputed() -> Dict:
    s = _summary("3d")
    i = _typical(s)
    field = np.load(os.path.join(EXAMPLE, "results", "3d", f"field_pinn_{i}.npy"))
    ref = np.load(os.path.join(EXAMPLE, "results", "3d", "field_reference.npy"))
    return {"summary": {k: v for k, v in s.items() if k != "runs"}, "run": s["runs"][i],
            "shape": list(field.shape), "field_pinn": field.round(2).ravel().tolist(),
            "field_reference": ref.round(2).ravel().tolist(),
            "draws": [{"h_pinn": r["h_pinn"], "h_energy_balance": r["h_energy_balance"],
                       "device": r["device_max_pinn_C"]} for r in s["runs"]]}
