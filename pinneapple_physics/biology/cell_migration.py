"""Cell migration in scratch assays: data loading, the reaction-diffusion forward solver and the generalized least
squares error of Lagergren et al. (BINNs), with the classical FKPP, Generalized Porous-FKPP and the BINN-guided
delay-reaction-diffusion models.

Equations and numerics follow Lagergren, Nardini, Baker, Simpson and Flores, "Biologically-informed neural networks
guide mechanistic modeling from sparse experimental data", PLoS Comput Biol 16(12):e1008462 (2020), arXiv:2005.13073:

* model ``u_t = T(t) (D(u) u_x)_x + G(u) u`` (Eq 11), no-flux ends, initial profile from the data at t = 0;
* forward solver (Eqs 17-20): method of lines on 201 equispaced points of the data's x range, the flux at
  ``x_{i+1/2}`` the average of ``D u_x`` evaluated with ``u_i`` and ``u_{i+1}``, ghost points for no flux, an
  explicit Runge-Kutta integrator;
* GLS error (Eq 6) ``mean(((u_model - u_data) / |u_model|^gamma)^2)``, gamma = 0.2.

The data are those of Jin et al., J Theor Biol 390:136-145 (2016), as distributed with the BINNs code
(``Data/cell_density_profiles.npy``, github.com/jlager/BINNs): a pickled dict that :func:`load_scratch_assay` reads
with a restricted unpickler (NumPy arrays, dicts, lists and numbers only; anything else is refused), never with
``allow_pickle=True``.
"""
from __future__ import annotations

import io
import math
import pickle
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

__all__ = ["ScratchAssay", "load_scratch_assay", "solve_reaction_diffusion", "gls_error", "fkpp", "porous_fkpp",
           "delay_reaction_diffusion", "fit_mechanistic", "K_CARRYING", "DENSITIES", "PARAMETER_BOUNDS"]

K_CARRYING = 1.7e3                    # cells/mm^2 (Jin et al. 2016; fixed in Lagergren et al.)
DENSITIES = (10000, 12000, 14000, 16000, 18000, 20000)
# Lagergren et al., Methods, "Parameter estimation" (mm^2/day, 1/day)
PARAMETER_BOUNDS = {"D": (0.0, 0.096), "D0": (0.0, 0.096), "m": (0.0, 4.0), "r": (0.0, 2.4),
                    "beta0": (-10.0, 10.0), "beta1": (-10.0, 10.0)}

_ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"), ("numpy.core.multiarray", "_reconstruct"),
            ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "scalar"),
            ("numpy._core.multiarray", "scalar"), ("builtins", "dict"), ("builtins", "list"),
            ("collections", "OrderedDict"), ("_codecs", "encode")}  # bytes in protocol-2 pickles


class _SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in _ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refused to load {module}.{name} from a data file")


def _read_npy_object(path) -> object:
    with open(path, "rb") as fh:
        raw = fh.read()
    buf = io.BytesIO(raw)
    version = np.lib.format.read_magic(buf)
    read_header = {(1, 0): np.lib.format.read_array_header_1_0, (2, 0): np.lib.format.read_array_header_2_0}
    if version not in read_header:
        raise ValueError(f"unsupported .npy version {version}")
    shape, fortran, dtype = read_header[version](buf)
    if dtype.hasobject:
        obj = _SafeUnpickler(buf).load()
        return obj.item() if isinstance(obj, np.ndarray) and obj.shape == () else obj
    return np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)


@dataclass
class ScratchAssay:
    """One experiment: positions ``x`` (mm, without the left-most profile, as in the paper), times ``t`` (days),
    densities ``U`` (cells/mm^2, shape (len(x), len(t)))."""
    density: int
    x: np.ndarray
    t: np.ndarray
    U: np.ndarray


def load_scratch_assay(path, density: int) -> ScratchAssay:
    """Scratch-assay data for an initial seeding ``density`` (cells per well) from the BINNs data file. Units are
    converted from um and hours to mm and days, and the left-most profile is removed (Lagergren et al., Methods)."""
    data = _read_npy_object(path)
    d = data[f"dens_{int(density)}"]
    x = np.asarray(d["x"], float).ravel()[1:] / 1000.0
    t = np.asarray(d["t"], float).ravel() / 24.0
    U = np.asarray(d["U_mean"], float)[1:, :] * 1e6
    return ScratchAssay(int(density), x, t, U)


def solve_reaction_diffusion(D: Callable, G: Callable, x: np.ndarray, t_eval: np.ndarray, u0: np.ndarray,
                             T: Callable | None = None, *, n: int = 201, method: str = "RK45",
                             rtol: float = 1e-6, atol: float = 1e-6) -> np.ndarray:
    """``u_t = T(t) (D(u) u_x)_x + G(u) u`` with no-flux ends on ``n`` points spanning ``[x.min(), x.max()]``; the
    initial profile ``u0`` (on ``x``) is linearly interpolated. Returns ``u`` on ``x`` at ``t_eval``,
    shape (len(x), len(t_eval)); ``nan`` if the integration fails."""
    from scipy.integrate import solve_ivp

    xg = np.linspace(float(np.min(x)), float(np.max(x)), n)
    dx = xg[1] - xg[0]
    ug0 = np.interp(xg, x, u0)

    def rhs(t, u):
        ue = np.concatenate([[u[1]], u, [u[-2]]])                     # ghost points: u_-1 = u_1, u_n = u_n-2
        ux = (ue[1:] - ue[:-1]) / dx                                   # at i+1/2
        Dl, Dr = D(ue[:-1]), D(ue[1:])
        P = 0.5 * (Dl * ux + Dr * ux)                                  # Eq 20
        diff = (P[1:] - P[:-1]) / dx
        tt = 1.0 if T is None else T(t)
        return tt * diff + G(u) * u

    sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), ug0, t_eval=t_eval, method=method,
                    rtol=rtol, atol=atol)
    if not sol.success or sol.y.shape[1] != len(t_eval):
        return np.full((len(x), len(t_eval)), np.nan)
    return np.stack([np.interp(x, xg, sol.y[:, j]) for j in range(len(t_eval))], axis=1)


def gls_error(u_model: np.ndarray, u_data: np.ndarray, gamma: float = 0.2) -> float:
    """Eq 6: ``mean(((u_model - u_data) / |u_model|^gamma)^2)``."""
    um = np.asarray(u_model, float)
    if not np.all(np.isfinite(um)):
        return math.inf
    w = np.abs(um) ** gamma
    w = np.where(w > 0, w, 1e-12)
    return float(np.mean(((um - np.asarray(u_data, float)) / w) ** 2))


def fkpp(D: float, r: float, K: float = K_CARRYING):
    """Classical FKPP (Eq 13): constant diffusivity, logistic growth. Returns (D(u), G(u), None)."""
    return (lambda u: np.full_like(u, D)), (lambda u: r * (1 - u / K)), None


def porous_fkpp(D: float, m: float, r: float, K: float = K_CARRYING):
    """Generalized Porous-FKPP (Eq 14): ``D (u/K)^m``."""
    return (lambda u: D * np.maximum(u / K, 0.0) ** m), (lambda u: r * (1 - u / K)), None


def delay_reaction_diffusion(D0: float, D: float, m: float, r: float, beta0: float, beta1: float,
                             K: float = K_CARRYING):
    """BINN-guided model (Eqs 11-12): ``D0 + D (u/K)^m``, logistic growth, delay ``1 / (1 + exp(-(beta1 t + beta0)))``
    (t in days, beta1 in 1/day)."""
    return ((lambda u: D0 + D * np.maximum(u / K, 0.0) ** m), (lambda u: r * (1 - u / K)),
            (lambda t: 1.0 / (1.0 + math.exp(-(beta1 * t + beta0)))))


MODELS: dict[str, tuple[Callable, tuple[str, ...]]] = {
    "fkpp": (fkpp, ("D", "r")),
    "porous_fkpp": (porous_fkpp, ("D", "m", "r")),
    "delay_reaction_diffusion": (delay_reaction_diffusion, ("D0", "D", "m", "r", "beta0", "beta1")),
}


def model_prediction(name: str, params: Mapping[str, float], assay: ScratchAssay, **kw) -> np.ndarray:
    build, names = MODELS[name]
    Dfn, Gfn, Tfn = build(*[params[k] for k in names])
    return solve_reaction_diffusion(Dfn, Gfn, assay.x, assay.t, assay.U[:, 0], Tfn, **kw)


def fit_mechanistic(name: str, assay: ScratchAssay, x0: Mapping[str, float], *, gamma: float = 0.2,
                    bounds: Mapping[str, tuple[float, float]] | None = None, maxiter: int = 200, **kw) -> dict:
    """Minimise the GLS error of model ``name`` over its parameters with L-BFGS-B in the paper's bounds (SciPy
    defaults otherwise, as in the paper). Parameters are optimised in units scaled by their bound widths."""
    from scipy.optimize import minimize

    _, names = MODELS[name]
    bnd = {**PARAMETER_BOUNDS, **(bounds or {})}
    lo = np.array([bnd[k][0] for k in names])
    hi = np.array([bnd[k][1] for k in names])
    scale = hi - lo

    def obj(z):
        p = dict(zip(names, lo + z * scale, strict=True))
        return gls_error(model_prediction(name, p, assay, **kw), assay.U, gamma)

    z0 = np.clip((np.array([x0[k] for k in names]) - lo) / scale, 0.0, 1.0)
    res = minimize(obj, z0, method="L-BFGS-B", bounds=[(0.0, 1.0)] * len(names), options={"maxiter": maxiter})
    p = dict(zip(names, (lo + res.x * scale).tolist(), strict=True))
    return {"params": p, "gls": float(res.fun), "nfev": int(res.nfev), "success": bool(res.success),
            "message": str(res.message)}
