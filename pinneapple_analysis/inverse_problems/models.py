"""Physical forward models for identifiability, calibration and inversion (heat transfer and kinetics).

=====================  ==========================================================================================
``PinFinModel``        steady pin fin with a convective tip, analytic (Incropera, Table 3.4 case A); the Inverse
                       Heat Lab case. Observables: thermocouple temperatures ``T`` and the base heat flow
                       ``q_base``. Autograd, and a PINN (``heat_transfer.run_fin``) for h.
``RodModel``           the same fin discretised in finite volumes, with h piecewise constant over zones (a field),
                       unknown conductivity, base temperature and air temperature. Exact forward sensitivities,
                       one-solve adjoint gradient, a torch twin for autograd; reconstructs T(x) and h(x).
``PlateModel``         the Inverse Heat Lab's 2D heat-spreader plate (finite volumes,
                       ``pinneapple_simulation.numerical_solvers.heat_fv_reference.plate_2d``): h and the device
                       power from thermocouples; reconstructs T(x, y); PINN via ``heat_transfer.run_plate``.
``LumpedCoolingModel`` lumped-capacitance cooling curve ``T(t) = T_air + (T0 - T_air) exp(-h A t / C)``.
``ArrheniusModel``     first-order conversion ``X(t, T) = 1 - exp(-A exp(-Ea / (R T)) t)``.
=====================  ==========================================================================================

Each has ``default_parameters()`` (bounds, units, log scale) and ``candidate_measurements`` over its physical
domain, so the experimental design proposes positions, times, temperatures or another observable.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from functools import lru_cache

import numpy as np

from .forward import ForwardModel, Observations, Parameter

__all__ = ["PinFinModel", "RodModel", "PlateModel", "LumpedCoolingModel", "ArrheniusModel", "R_GAS"]

R_GAS = 8.314462618          # J/(mol K), CODATA 2018 exact value


def _obs_mask(observable, name: str, n: int) -> np.ndarray:
    if observable is None:
        return np.ones(n, bool) if name in ("T", "") else np.zeros(n, bool)
    o = np.asarray(observable, dtype=object)
    return np.array([str(v) in (name, "") if name == "T" else str(v) == name for v in o])


# --------------------------------------------------------------------------- analytic pin fin
class PinFinModel(ForwardModel):
    """Pin fin of diameter ``d_mm`` and length ``length_mm`` on a wall at ``T_base``, in air at ``T_air``.

    Temperatures depend on h and k only through ``m^2 = 4 h / (k d)`` and the tip ratio ``h / (m k)``, i.e. only
    through ``h/k``: from thermocouples alone h and k are *not* separately identifiable. The base heat flow
    ``q_base = sqrt(h P k A) (T_base - T_air) (...)`` scales with ``sqrt(h k)`` and resolves it."""
    name = "pin_fin"
    inputs = ("x_mm",)
    outputs = ("T", "q_base")
    description = "Steady pin fin with a convective tip (analytic). Inputs: position x_mm from the wall."

    def __init__(self, d_mm: float = 5.0, length_mm: float = 50.0):
        self.d_mm, self.length_mm = float(d_mm), float(length_mm)

    def default_parameters(self) -> list[Parameter]:
        return [Parameter("h", 50.0, 1.0, 2000.0, "W/m²K", description="convection coefficient"),
                Parameter("k", 16.0, 1.0, 500.0, "W/m·K", fixed=True, description="fin conductivity"),
                Parameter("T_base", 80.0, -50.0, 1000.0, "°C", log=False, fixed=True),
                Parameter("T_air", 25.0, -50.0, 500.0, "°C", log=False, fixed=True)]

    def _parts(self, th, lib):
        d, L = self.d_mm / 1000, self.length_mm / 1000
        area, perim = math.pi * d * d / 4, math.pi * d
        h, k = th["h"], th["k"]
        m = lib.sqrt(h * perim / (k * area))
        r = h / (m * k)
        return d, L, area, perim, h, k, m, r

    def _eval(self, th, x_mm, observable, lib):
        d, L, area, perim, h, k, m, r = self._parts(th, lib)
        xi = x_mm / self.length_mm
        den = lib.cosh(m * L) + r * lib.sinh(m * L)
        theta = (lib.cosh(m * L * (1 - xi)) + r * lib.sinh(m * L * (1 - xi))) / den
        T = th["T_air"] + (th["T_base"] - th["T_air"]) * theta
        q = lib.sqrt(h * perim * k * area) * (th["T_base"] - th["T_air"]) * \
            (lib.sinh(m * L) + r * lib.cosh(m * L)) / den
        return T, q

    def predict(self, theta, inputs, observable=None):
        x = np.asarray(inputs["x_mm"], float)
        th = {k: float(v) for k, v in theta.items()}
        T, q = self._eval(th, np.nan_to_num(x), observable, np)
        is_q = _obs_mask(observable, "q_base", len(x))
        return np.where(is_q, q, T)

    def predict_torch(self, theta, inputs, observable=None):
        import torch

        x = torch.as_tensor(np.nan_to_num(np.asarray(inputs["x_mm"], float)), dtype=torch.float64)
        T, q = self._eval(theta, x, observable, torch)
        is_q = torch.as_tensor(_obs_mask(observable, "q_base", len(x)))
        return torch.where(is_q, q * torch.ones_like(x), T)

    def candidate_measurements(self, obs, n=20):
        sig_T = float(np.median(obs.sigma[_obs_mask(obs.observable, "T", obs.n)])) if \
            _obs_mask(obs.observable, "T", obs.n).any() else 0.5
        x = np.linspace(self.length_mm / n, self.length_mm, n)
        c = Observations({"x_mm": x}, sig_T, observable=np.array(["T"] * n, dtype=object))
        q = Observations({"x_mm": [0.0]}, 0.05 * 0.5, observable=np.array(["q_base"], dtype=object))
        return c.concat(q)

    def fields(self, theta):
        x = np.linspace(0, self.length_mm, 201)
        T = self.predict(theta, {"x_mm": x}, np.array(["T"] * len(x), dtype=object))
        return {"T": {"x": x.tolist(), "values": T.tolist(), "unit": "°C", "axis": "x_mm"}}

    def pinn(self, obs: Observations, parameters: Mapping[str, Parameter], steps: int = 2000, seed: int = 0,
             progress=None, h_true: float | None = None) -> dict:
        """The Inverse Heat Lab's physics-informed network for h (k, T_base, T_air known), on the T readings."""
        from .heat_transfer import run_fin

        if not parameters["k"].fixed or not parameters["T_base"].fixed or not parameters["T_air"].fixed:
            raise ValueError("the fin PINN estimates h only: fix k, T_base and T_air")
        m = _obs_mask(obs.observable, "T", obs.n)
        out = run_fin(k=parameters["k"].value, d_mm=self.d_mm, length_mm=self.length_mm,
                      t_base=parameters["T_base"].value, t_air=parameters["T_air"].value,
                      sensors_mm=obs.inputs["x_mm"][m].tolist(), readings=obs.values[m].tolist(),
                      h_guess=parameters["h"].value, steps=steps, seed=seed, progress=progress, h_true=h_true)
        return {"theta": {"h": out["h_pinn"]}, "history": out["history"],
                "field": {"x": out["x_mm"], "values": out["T_pinn"]}, "raw": out}


# --------------------------------------------------------------------------- discretised rod (adjoint)
class RodModel(ForwardModel):
    """Finite-volume pin fin with ``n_zones`` convection zones (h1..hN; ``h`` when one zone).

    Cell balance ``k A (T_{i+1} - 2 T_i + T_{i-1}) / dx - h_i P dx (T_i - T_air) = 0``; the base face is held at
    ``T_base`` (half-cell conductance), the tip loses heat through the half cell and ``h`` of the last zone in
    series. The system ``A(theta) T = b(theta)`` is symmetric tridiagonal, solved banded. Derivatives are exact:
    forward sensitivities ``dT/dtheta = A^-1 (db/dtheta - dA/dtheta T)`` (one solve per parameter) for Jacobians,
    and the adjoint ``A^T lambda = G^T w`` (one solve, whatever the number of parameters) for the misfit gradient.
    """
    name = "rod_fv"
    inputs = ("x_mm",)
    outputs = ("T", "q_base")
    description = "Finite-volume pin fin with zone-wise h(x); exact sensitivities and adjoint gradient."

    def __init__(self, d_mm: float = 5.0, length_mm: float = 100.0, n_zones: int = 1, n_cells: int = 200):
        self.d_mm, self.length_mm, self.n_zones, self.N = float(d_mm), float(length_mm), int(n_zones), int(n_cells)
        L = self.length_mm / 1000
        self.dx = L / self.N
        self.xc = (np.arange(self.N) + 0.5) * self.dx
        self.zone = np.minimum((self.xc / L * self.n_zones).astype(int), self.n_zones - 1)
        d = self.d_mm / 1000
        self.area, self.perim = math.pi * d * d / 4, math.pi * d
        self.h_names = ["h"] if self.n_zones == 1 else [f"h{j + 1}" for j in range(self.n_zones)]

    def default_parameters(self) -> list[Parameter]:
        ps = [Parameter(n, 50.0, 1.0, 2000.0, "W/m²K", description=f"convection coefficient, zone {j + 1}")
              for j, n in enumerate(self.h_names)]
        return ps + [Parameter("k", 16.0, 1.0, 500.0, "W/m·K", fixed=True),
                     Parameter("T_base", 80.0, -50.0, 1000.0, "°C", log=False, fixed=True),
                     Parameter("T_air", 25.0, -50.0, 500.0, "°C", log=False, fixed=True)]

    def field_groups(self) -> list[list[str]]:
        """Parameters that form one field, in spatial order (the h zones), for smoothness priors."""
        return [list(self.h_names)] if self.n_zones > 1 else []

    def zone_edges_mm(self) -> list[float]:
        return list(np.linspace(0, self.length_mm, self.n_zones + 1))

    # assembly ---------------------------------------------------------------
    def _coeffs(self, th, lib=np):
        hz = [th[n] for n in self.h_names]
        k, Tb, Ta = th["k"], th["T_base"], th["T_air"]
        if lib is np:
            h = np.asarray(hz, float)[self.zone]
        else:
            h = lib.stack([lib.as_tensor(v, dtype=lib.float64) for v in hz])[lib.as_tensor(self.zone)]
        c = k * self.area / self.dx                         # interior face conductance
        cb = 2 * k * self.area / self.dx                    # base half cell
        hl = hz[-1]
        gt = 1.0 / (self.dx / (2 * k * self.area) + 1.0 / (hl * self.area))   # tip: half cell + film
        conv = h * self.perim * self.dx
        return h, c, cb, gt, conv, k, Tb, Ta, hl

    def _system(self, th):
        h, c, cb, gt, conv, k, Tb, Ta, hl = self._coeffs(th)
        N = self.N
        diag = conv + 2 * c
        diag[0] += cb - c
        diag[-1] += gt - c
        b = conv * Ta
        b[0] += cb * Tb
        b[-1] += gt * Ta
        return diag, np.full(N - 1, -c), b

    def _solve(self, diag, off, rhs):
        from scipy.linalg import solve_banded

        ab = np.zeros((3, self.N))
        ab[0, 1:], ab[1], ab[2, :-1] = off, diag, off
        return solve_banded((1, 1), ab, rhs)

    def temperature(self, theta) -> np.ndarray:
        diag, off, b = self._system({k: float(v) for k, v in theta.items()})
        return self._solve(diag, off, b)

    def _obs_operator(self, th, obs: Observations):
        """g = G T + e(theta): rows of G (n, N) and the explicit part, with its derivatives."""
        x = np.nan_to_num(obs.inputs["x_mm"]) / 1000
        n, N = obs.n, self.N
        G = np.zeros((n, N))
        wb = np.zeros(n)                                     # weight on T_base (between the wall and cell 0)
        is_q = _obs_mask(obs.observable, "q_base", n)
        for r in range(n):
            if is_q[r]:
                continue
            xr = x[r]
            if xr <= self.xc[0]:
                t = xr / self.xc[0]
                G[r, 0], wb[r] = t, 1 - t
            elif xr >= self.xc[-1]:
                G[r, -1] = 1.0
            else:
                i = int(np.searchsorted(self.xc, xr) - 1)
                t = (xr - self.xc[i]) / self.dx
                G[r, i], G[r, i + 1] = 1 - t, t
        cb = 2 * th["k"] * self.area / self.dx
        G[is_q, 0] = -cb
        wb = np.where(is_q, cb, wb)
        return G, wb, is_q

    def _residual_derivs(self, th, T, names: Sequence[str]) -> np.ndarray:
        """(N, p): d b/d theta_j - d A/d theta_j T, exact."""
        h, c, cb, gt, conv, k, Tb, Ta, hl = self._coeffs(th)
        N = self.N
        out = np.zeros((N, len(names)))
        dT = np.diff(T)
        for j, nm in enumerate(names):
            v = np.zeros(N)
            if nm == "k":                                    # conductances are proportional to k (tip: series)
                dc = c / k
                v[:-1] += dc * dT                            # -(dA/dk T): interior faces
                v[1:] -= dc * dT
                v[0] += (cb / k) * (Tb - T[0])
                dgt = gt**2 * self.dx / (2 * k**2 * self.area)
                v[-1] += dgt * (Ta - T[-1])
            elif nm == "T_base":
                v[0] = cb
            elif nm == "T_air":
                v = h * self.perim * self.dx
                v[-1] += gt
            elif nm in self.h_names:
                zj = self.h_names.index(nm)
                m = self.zone == zj
                v[m] = self.perim * self.dx * (Ta - T[m])
                if zj == self.n_zones - 1:
                    dgt = gt**2 / (hl**2 * self.area)
                    v[-1] += dgt * (Ta - T[-1])
            else:
                raise KeyError(nm)
            out[:, j] = v
        return out

    def _explicit(self, th, T, obs, names, G, wb, is_q):
        """(n, p): partial derivatives of g at fixed T."""
        n = obs.n
        E = np.zeros((n, len(names)))
        cb = 2 * th["k"] * self.area / self.dx
        for j, nm in enumerate(names):
            if nm == "T_base":
                E[:, j] = wb
            elif nm == "k":
                E[is_q, j] = cb / th["k"] * (th["T_base"] - T[0])
        return E

    def predict(self, theta, inputs, observable=None):
        th = {k: float(v) for k, v in theta.items()}
        obs = Observations(dict(inputs), 1.0, observable=observable)
        T = self.temperature(th)
        G, wb, _ = self._obs_operator(th, obs)
        return G @ T + wb * th["T_base"]

    def sensitivities(self, theta, obs: Observations, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        th = {k: float(v) for k, v in theta.items()}
        diag, off, b = self._system(th)
        T = self._solve(diag, off, b)
        G, wb, is_q = self._obs_operator(th, obs)
        R = self._residual_derivs(th, T, names)
        S = np.column_stack([self._solve(diag, off, R[:, j]) for j in range(len(names))])
        return G @ T + wb * th["T_base"], G @ S + self._explicit(th, T, obs, names, G, wb, is_q)

    def adjoint_gradient(self, theta, obs: Observations, names: Sequence[str]) -> tuple[float, np.ndarray]:
        th = {k: float(v) for k, v in theta.items()}
        diag, off, b = self._system(th)
        T = self._solve(diag, off, b)
        G, wb, is_q = self._obs_operator(th, obs)
        g = G @ T + wb * th["T_base"]
        w = (g - obs.values) / obs.sigma**2
        lam = self._solve(diag, off, G.T @ w)                # A is symmetric: A^T lambda = G^T w
        R = self._residual_derivs(th, T, names)
        grad = lam @ R + w @ self._explicit(th, T, obs, names, G, wb, is_q)
        return 0.5 * float(np.sum(((g - obs.values) / obs.sigma) ** 2)), grad

    def predict_torch(self, theta, inputs, observable=None):
        import torch

        h, c, cb, gt, conv, k, Tb, Ta, hl = self._coeffs(theta, torch)
        N = self.N
        e0 = torch.zeros(N, dtype=torch.float64)
        e0[0] = 1.0
        eN = torch.zeros(N, dtype=torch.float64)
        eN[-1] = 1.0
        diag = conv + 2 * c + (cb - c) * e0 + (gt - c) * eN
        off = -c * torch.ones(N - 1, dtype=torch.float64)
        A = torch.diag(diag) + torch.diag(off, 1) + torch.diag(off, -1)
        b = conv * Ta + cb * Tb * e0 + gt * Ta * eN
        T = torch.linalg.solve(A, b)
        obs = Observations(dict(inputs), 1.0, observable=observable)
        G, wb, is_q = self._obs_operator({"k": 1.0}, obs)          # geometry only; q rows rebuilt below
        Gt, wbt = torch.as_tensor(G), torch.as_tensor(wb)
        isq = torch.as_tensor(is_q)
        gT = Gt @ T + wbt * Tb
        q = cb * (Tb - T[0])
        return torch.where(isq, q * torch.ones_like(gT), gT)

    def candidate_measurements(self, obs, n=24):
        m = _obs_mask(obs.observable, "T", obs.n)
        sig = float(np.median(obs.sigma[m])) if m.any() else 0.5
        x = np.linspace(self.length_mm / n, self.length_mm, n)
        c = Observations({"x_mm": x}, sig, observable=np.array(["T"] * n, dtype=object))
        return c.concat(Observations({"x_mm": [0.0]}, 0.05, observable=np.array(["q_base"], dtype=object)))

    def fields(self, theta):
        th = {k: float(v) for k, v in theta.items()}
        T = self.temperature(th)
        x = np.concatenate([[0.0], self.xc * 1000])
        hz = np.array([th[n] for n in self.h_names])
        return {"T": {"x": x.tolist(), "values": np.concatenate([[th["T_base"]], T]).tolist(), "unit": "°C",
                      "axis": "x_mm"},
                "h": {"x": (self.xc * 1000).tolist(), "values": hz[self.zone].tolist(), "unit": "W/m²K",
                      "axis": "x_mm"}}


# --------------------------------------------------------------------------- 2D plate
PLATE_SETUP = dict(lx=0.100, ly=0.060, t=0.001, k=200.0, device=(0.030, 0.030, 0.010, 0.002))


@lru_cache(maxsize=256)
def _plate_unit(h: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Temperature rise per watt of device power at convection ``h`` (the plate equation is linear in power)."""
    from pinneapple_simulation.numerical_solvers.heat_fv_reference import plate_2d

    x, y, T = plate_2d(h, nx=nx, ny=ny, power=1.0, t_air=0.0, **PLATE_SETUP)
    return x, y, T


class PlateModel(ForwardModel):
    """The Inverse Heat Lab's heat-spreader plate (100 x 60 x 1 mm aluminium, 20 mm device, convection on both
    faces): ``T(x, y) = T_air + power * u(x, y; h)``, with ``u`` from the finite-volume solver (``nx x ny`` cells,
    cached per h). Inputs ``x_mm``, ``y_mm``; observable ``T``."""
    name = "plate_2d"
    inputs = ("x_mm", "y_mm")
    outputs = ("T",)
    cost = "moderate"
    description = "Heat-spreader plate (2D finite volumes, the Inverse Heat Lab case): h and device power."

    def __init__(self, nx: int = 60, ny: int = 36):
        self.nx, self.ny = int(nx), int(ny)

    def default_parameters(self) -> list[Parameter]:
        return [Parameter("h", 40.0, 2.0, 300.0, "W/m²K"),
                Parameter("power", 8.0, 0.5, 50.0, "W", fixed=True),
                Parameter("T_air", 25.0, -50.0, 200.0, "°C", log=False, fixed=True)]

    def _field(self, th):
        from scipy.interpolate import RegularGridInterpolator

        x, y, u = _plate_unit(round(float(th["h"]), 12), self.nx, self.ny)
        return x, y, u, RegularGridInterpolator((y, x), u, bounds_error=False, fill_value=None)

    def predict(self, theta, inputs, observable=None):
        _, _, _, f = self._field(theta)
        px, py = np.asarray(inputs["x_mm"], float) / 1000, np.asarray(inputs["y_mm"], float) / 1000
        return float(theta["T_air"]) + float(theta["power"]) * f(np.column_stack([py, px]))

    def candidate_measurements(self, obs, n=8):
        xs, ys = np.linspace(5, 95, n + 2)[1:-1], np.linspace(5, 55, max(3, n * 6 // 10))
        X, Y = np.meshgrid(xs, ys)
        return Observations({"x_mm": X.ravel(), "y_mm": Y.ravel()}, float(np.median(obs.sigma)),
                            observable=np.array(["T"] * X.size, dtype=object))

    def fields(self, theta):
        x, y, u, _ = self._field(theta)
        T = float(theta["T_air"]) + float(theta["power"]) * u
        return {"T": {"x": (x * 1000).round(3).tolist(), "y": (y * 1000).round(3).tolist(),
                      "values": T.round(3).tolist(), "unit": "°C", "axis": "x_mm,y_mm"}}

    def pinn(self, obs: Observations, parameters: Mapping[str, Parameter], steps: int = 1500, seed: int = 0,
             progress=None, h_true: float | None = None) -> dict:
        """The Inverse Heat Lab's plate PINN on these readings (h unknown, power known)."""
        from .heat_transfer import run_plate

        if not parameters["power"].fixed:
            raise ValueError("the plate PINN estimates h only: fix the power")
        sensors = np.column_stack([obs.inputs["x_mm"], obs.inputs["y_mm"]]).tolist()
        out = run_plate(power=parameters["power"].value, h_true=h_true or parameters["h"].value, noise=0.0,
                        h_guess=parameters["h"].value, steps=steps, seed=seed, progress=progress,
                        readings=obs.values.tolist(), sensors_mm=sensors, t_air=parameters["T_air"].value)
        return {"theta": {"h": out["h_pinn"]}, "history": out["history"],
                "field": {"x": out["x_mm"], "y": out["y_mm"], "values": out["field_pinn"]}, "raw": out}


# --------------------------------------------------------------------------- lumped cooling and kinetics
class LumpedCoolingModel(ForwardModel):
    """``T(t) = T_air + (T0 - T_air) exp(-h A t / C)`` (Biot number well below 0.1). ``area_m2`` is known; the
    curve depends on h and the heat capacity C only through ``h A / C``: both free is structurally ambiguous."""
    name = "lumped_cooling"
    inputs = ("t_s",)
    outputs = ("T",)
    description = "Lumped-capacitance cooling curve of a part in air."

    def __init__(self, area_m2: float = 0.01):
        self.area = float(area_m2)

    def default_parameters(self) -> list[Parameter]:
        return [Parameter("h", 20.0, 0.5, 500.0, "W/m²K"),
                Parameter("C", 450.0, 1.0, 1e5, "J/K", fixed=True, description="heat capacity m·c"),
                Parameter("T0", 90.0, -50.0, 1000.0, "°C", log=False),
                Parameter("T_air", 25.0, -50.0, 500.0, "°C", log=False)]

    def _eval(self, th, t, lib):
        return th["T_air"] + (th["T0"] - th["T_air"]) * lib.exp(-th["h"] * self.area * t / th["C"])

    def predict(self, theta, inputs, observable=None):
        return self._eval({k: float(v) for k, v in theta.items()}, np.asarray(inputs["t_s"], float), np)

    def predict_torch(self, theta, inputs, observable=None):
        import torch

        return self._eval(theta, torch.as_tensor(np.asarray(inputs["t_s"], float)), torch)

    def candidate_measurements(self, obs, n=30):
        t = obs.inputs["t_s"]
        hi = float(np.nanmax(t)) * 3
        return Observations({"t_s": np.linspace(0, hi, n)}, float(np.median(obs.sigma)),
                            observable=np.array(["T"] * n, dtype=object))


class ArrheniusModel(ForwardModel):
    """First-order conversion ``X = 1 - exp(-A exp(-Ea / (R T)) t)`` at temperature ``T_K``. At one temperature
    only the rate ``A exp(-Ea/RT)`` is seen: A and Ea are structurally ambiguous until a second temperature is
    measured (and strongly correlated over a narrow temperature range)."""
    name = "arrhenius"
    inputs = ("t_s", "T_K")
    outputs = ("X",)
    description = "First-order reaction conversion with an Arrhenius rate constant."

    def default_parameters(self) -> list[Parameter]:
        return [Parameter("A", 1e6, 1.0, 1e14, "1/s", description="pre-exponential factor"),
                Parameter("Ea", 60e3, 1e3, 3e5, "J/mol", description="activation energy")]

    def _eval(self, th, t, T, lib):
        return 1 - lib.exp(-th["A"] * lib.exp(-th["Ea"] / (R_GAS * T)) * t)

    def predict(self, theta, inputs, observable=None):
        return self._eval({k: float(v) for k, v in theta.items()}, np.asarray(inputs["t_s"], float),
                          np.asarray(inputs["T_K"], float), np)

    def predict_torch(self, theta, inputs, observable=None):
        import torch

        return self._eval(theta, torch.as_tensor(np.asarray(inputs["t_s"], float)),
                          torch.as_tensor(np.asarray(inputs["T_K"], float)), torch)

    def candidate_measurements(self, obs, n=8):
        t = np.unique(obs.inputs["t_s"])
        T = obs.inputs["T_K"]
        temps = np.linspace(float(T.min()) - 40, float(T.max()) + 40, n)
        tt, TT = np.meshgrid(np.linspace(t.min(), t.max(), 6), temps)
        return Observations({"t_s": tt.ravel(), "T_K": TT.ravel()}, float(np.median(obs.sigma)),
                            observable=np.array(["X"] * tt.size, dtype=object))
