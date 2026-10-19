"""Physics checks of a 2-D Darcy field, ``-div(a grad u) = f`` on the unit square with ``u = 0`` on the boundary.

Every check works on a predicted field alone (no reference solution) and uses the same discretisation as the
reference solver ``pinneapple_simulation.numerical_solvers.operator_benchmarks.darcy_2d`` and the zoo dataset
``darcy_2d_li2021``: cell-centred values on an ``n x n`` grid, ``h = 1/n``, two-point flux finite volumes with
harmonic-mean face coefficients and half-cell Dirichlet faces. A field that solves that discrete problem passes all
of them to round-off; a surrogate's prediction passes them only as far as it respects the physics.

``darcy_residual``
    ``f - A_a u`` per cell (``A_a`` the finite-volume operator): the PDE residual, consistent with the solver.
``residual_error_bound``
    a guaranteed bound on the error of ``u`` in the energy norm of the discrete problem, from the residual only:
    ``||u_h - u||_A <= ||r||_{L^-1} / sqrt(min a)``, with ``L`` the constant-coefficient (``a = 1``) operator, whose
    inverse is applied by a fast sine transform. It holds because ``A_a - min(a) L`` is positive semi-definite
    (every face coefficient is a harmonic mean, never below ``min a``). Returned relative to ``||u||_A``.
``boundary_flux_balance``
    net outward flux through the four sides (``2 a u`` per boundary face, the solver's own boundary flux) against
    the integral of the source: the discrete conservation statement.
``boundary_values``
    the value of ``u`` on the boundary, extrapolated linearly from the first two cells (second order).
``check_nodal_darcy_batch`` (and ``nodal_*``, ``solve_nodal_darcy``)
    the same kind of checks for training data stored at the nodes of a finite-difference grid (PhysicsNeMo's
    ``Darcy2D`` datapipe layout), before anyone trains on the data: residual of the documented equation in that
    layout, boundary values, a control-volume flux balance, positivity, and the diversity measures of
    ``pinneapple_data.audit``. For cell-centred data in the layout of ``darcy_2d_li2021``, use ``audit_dataset``.
``residual_l2_bound``
    the same certificate for the error of the field itself (discrete Poincare inequality).
``PROFILES`` / ``profile_verdicts``
    use profiles, set a priori with the reason for each limit: ``fields`` (maps of ``u``: L2 bound, ``u = 0``,
    positivity) and ``fluxes`` (flow rates and wall gradients: energy bound, flux balance); per profile the
    decision and the graded ratio value / limit.
``sample_checks`` / ``sample_verdict``
    the checks of one prediction exactly as ``pinneapple check`` runs them, so a verdict shown next to a
    prediction is the CI's verdict.
``estimated_error`` / ``estimated_flux_error`` / ``regional_errors`` / ``use_verdicts``
    where the error is, from the residual alone: (uncertified) error maps of ``u`` and of the flux ``a grad u``, its size near the walls and in the
    interior, and which uses a prediction can serve and why (interior field values; walls and boundary fluxes;
    inputs inside the training envelope).
``physics_metrics`` / ``calibrate_tolerances``
    the per-sample values of the physics checks, and limits calibrated on a model's own predictions for held-out
    in-distribution inputs (no labels): a drift monitor that flags inputs where the model is less physical than
    where it was validated.
``darcy_checks`` / ``darcy_trust_report``
    the checks above (plus positivity, the maximum principle for ``f >= 0``, an optional reference comparison and
    an optional applicability check of the coefficient against the training data) for one field or a batch, as
    :class:`pinneapple_analysis.trust.trust_report.Check` objects and a deploy decision.

numpy and scipy only (the checker of ``pinneapple check`` loads this module without torch).
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy import fft as _fft

__all__ = [
    "DEFAULT_TOLERANCES", "darcy_operator_apply", "darcy_residual", "residual_error_bound", "boundary_flux_balance",
    "boundary_values", "coefficient_stats", "CoefficientEnvelope", "DarcyCheck", "darcy_checks",
    "darcy_trust_report", "residual_l2_bound", "UseProfile", "PROFILES", "profile_verdicts", "SAMPLE_CHECKS",
    "sample_checks", "sample_verdict", "estimated_error", "flux_magnitude", "estimated_flux_error", "region_masks", "regional_errors", "USES",
    "use_verdicts", "CHECK_TOLERANCE_KEYS", "check_config_entry", "save_darcy_npz", "nodal_darcy_residual", "nodal_boundary_values", "nodal_flux_balance",
    "solve_nodal_darcy", "check_nodal_darcy_batch", "physics_metrics", "calibrate_tolerances", "trust_report",
]

#: Default limits. ``energy_bound``: certified relative energy-norm error; ``bc``: RMS boundary value over max |u|
#: (the solver's own solution reaches 0.004 at 64 x 64 and 0.002 at 128 x 128: the extrapolation is only first
#: order where the coefficient jumps next to the boundary);
#: ``flux``: relative net flux imbalance; ``positivity``: most negative value over max |u|; ``reference``: relative L2;
#: ``l2_bound``: certified relative L2 error, at the same 5 % as ``reference``.
DEFAULT_TOLERANCES: dict[str, float] = {
    "energy_bound": 0.10,
    "l2_bound": 0.05,
    "bc": 0.02,
    "flux": 0.02,
    "positivity": 0.01,
    "reference": 0.05,
}


def _batch(x: np.ndarray) -> tuple[np.ndarray, bool]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 2:
        return x[None], True
    if x.ndim != 3 or x.shape[-1] != x.shape[-2]:
        raise ValueError(f"expected (n, n) or (batch, n, n), got {x.shape}")
    return x, False


def _pair(a, u) -> tuple[np.ndarray, np.ndarray, bool]:
    a, sa = _batch(a)
    u, su = _batch(u)
    if a.shape[-2:] != u.shape[-2:]:
        raise ValueError(f"a {a.shape} and u {u.shape} are on different grids")
    a, u = np.broadcast_arrays(a, u)
    return a, u, sa and su


def darcy_operator_apply(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``A_a u``: the finite-volume operator of ``-div(a grad u)`` (with ``u = 0`` on the boundary) applied to ``u``.

    ``u``, ``a``: ``(n, n)`` or ``(batch, n, n)`` at the cell centres. Identical, cell by cell, to the matrix that
    ``operator_benchmarks.darcy_2d`` assembles.
    """
    a, u, single = _pair(a, u)
    n = u.shape[-1]
    h = 1.0 / n
    out = np.zeros_like(u)
    for ax in (1, 2):
        sl_l = [slice(None)] * 3
        sl_r = [slice(None)] * 3
        sl_l[ax], sl_r[ax] = slice(None, -1), slice(1, None)
        al, ar = a[tuple(sl_l)], a[tuple(sl_r)]
        t = 2 * al * ar / (al + ar) / h ** 2
        flux = t * (u[tuple(sl_l)] - u[tuple(sl_r)])      # from the left cell to the right one
        out[tuple(sl_l)] += flux
        out[tuple(sl_r)] -= flux
    tb = 2 * a / h ** 2
    out[:, 0, :] += tb[:, 0, :] * u[:, 0, :]
    out[:, -1, :] += tb[:, -1, :] * u[:, -1, :]
    out[:, :, 0] += tb[:, :, 0] * u[:, :, 0]
    out[:, :, -1] += tb[:, :, -1] * u[:, :, -1]
    return out[0] if single else out


def darcy_residual(u: np.ndarray, a: np.ndarray, f: float | np.ndarray = 1.0) -> np.ndarray:
    """Finite-volume residual ``f - A_a u`` per cell (same shape as ``u``); zero for the solver's own solution."""
    Au = darcy_operator_apply(u, a)
    return np.broadcast_to(np.asarray(f, dtype=np.float64), Au.shape) - Au


def _laplace_inverse(r: np.ndarray) -> np.ndarray:
    """``L^-1 r`` for the constant-coefficient operator (a = 1) on the last two axes, by a type-II sine transform.

    The cell-centred Dirichlet operator (ghost value ``-u`` behind each boundary face) has the eigenvectors
    ``sin(k pi (i + 1/2) / n)``, ``k = 1..n``, with eigenvalues ``(4 / h^2) sin^2(k pi / (2 n))`` per axis.
    """
    n = r.shape[-1]
    h = 1.0 / n
    k = np.arange(1, n + 1)
    lam = 4.0 / h ** 2 * np.sin(k * np.pi / (2 * n)) ** 2
    R = _fft.dstn(r, type=2, axes=(-2, -1), norm="ortho")
    R = R / (lam[:, None] + lam[None, :])
    return _fft.idstn(R, type=2, axes=(-2, -1), norm="ortho")


def residual_error_bound(u: np.ndarray, a: np.ndarray, f: float | np.ndarray = 1.0) -> np.ndarray:
    """Certified relative error of ``u`` in the energy norm of the discrete problem, from its residual alone.

    Returns ``||r||_{L^-1} / sqrt(min a) / ||u||_A`` per sample (a scalar for one field), an upper bound on
    ``||u_h - u||_A / ||u||_A`` where ``u_h`` is the exact finite-volume solution on the same grid. Inner products
    carry the cell area ``h^2``. The bound is about the discrete solution: the discretisation error of ``u_h`` itself
    (about 0.5 % in L2 at 128 x 128 for the zoo dataset) is not included.
    """
    a, u, single = _pair(a, u)
    h2 = 1.0 / u.shape[-1] ** 2
    r = darcy_residual(u, a, f)
    dual = np.sqrt(np.maximum(h2 * np.sum(r * _laplace_inverse(r), axis=(-2, -1)), 0.0))
    energy = np.sqrt(np.maximum(h2 * np.sum(u * darcy_operator_apply(u, a), axis=(-2, -1)), 0.0))
    amin = a.reshape(a.shape[0], -1).min(axis=1)
    out = dual / np.sqrt(amin) / np.maximum(energy, 1e-300)
    return out[0] if single else out


def residual_l2_bound(u: np.ndarray, a: np.ndarray, f: float | np.ndarray = 1.0) -> np.ndarray:
    """Certified relative L2 error of ``u`` (to the discrete solution), from its residual alone.

    ``||u_h - u||_2 <= ||r||_{L^-1} / (min a * sqrt(lambda_min(L))) / ||u||_2``: the energy bound of
    :func:`residual_error_bound` with the discrete Poincare inequality ``v^T A v >= min(a) lambda_min(L) ||v||^2``,
    ``lambda_min(L) = (8 / h^2) sin^2(pi / (2 n))`` (about ``2 pi^2``). It bounds the error of the field itself, the
    quantity a map of ``u`` is judged by, where the energy bound bounds the error of its gradient (the flux).
    """
    a, u, single = _pair(a, u)
    n = u.shape[-1]
    h2 = 1.0 / n ** 2
    r = darcy_residual(u, a, f)
    dual = np.sqrt(np.maximum(h2 * np.sum(r * _laplace_inverse(r), axis=(-2, -1)), 0.0))
    lam = 8.0 * n ** 2 * np.sin(np.pi / (2 * n)) ** 2
    amin = a.reshape(a.shape[0], -1).min(axis=1)
    norm = np.sqrt(h2 * np.sum(u * u, axis=(-2, -1)))
    out = dual / (amin * np.sqrt(lam)) / np.maximum(norm, 1e-300)
    return out[0] if single else out


def boundary_flux_balance(u: np.ndarray, a: np.ndarray, f: float | np.ndarray = 1.0) -> dict[str, np.ndarray]:
    """Net outward flux through the boundary against the integral of the source (per sample).

    The outward flux through a boundary face of cell ``c`` is ``a_c (u_c - 0) / (h/2) * h = 2 a_c u_c``, as in the
    solver. Returns ``{"outflow", "source", "imbalance"}`` with ``imbalance = |outflow - source| / |source|``.
    """
    a, u, single = _pair(a, u)
    n = u.shape[-1]
    h = 1.0 / n
    out = 2 * (np.sum(a[:, 0, :] * u[:, 0, :], axis=-1) + np.sum(a[:, -1, :] * u[:, -1, :], axis=-1)
               + np.sum(a[:, :, 0] * u[:, :, 0], axis=-1) + np.sum(a[:, :, -1] * u[:, :, -1], axis=-1))
    src = np.sum(np.broadcast_to(np.asarray(f, dtype=np.float64), u.shape), axis=(-2, -1)) * h * h
    imb = np.abs(out - src) / np.maximum(np.abs(src), 1e-300)
    res = {"outflow": out, "source": src, "imbalance": imb}
    return {k: v[0] for k, v in res.items()} if single else res


def boundary_values(u: np.ndarray) -> np.ndarray:
    """``u`` on the four sides, ``1.5 u_0 - 0.5 u_1`` from the two cells next to each face: ``(batch, 4, n)``
    (order x_min, x_max, y_min, y_max; ``(4, n)`` for one field)."""
    u, single = _batch(u)
    b = np.stack([1.5 * u[:, 0, :] - 0.5 * u[:, 1, :], 1.5 * u[:, -1, :] - 0.5 * u[:, -2, :],
                  1.5 * u[:, :, 0] - 0.5 * u[:, :, 1], 1.5 * u[:, :, -1] - 0.5 * u[:, :, -2]], axis=1)
    return b[0] if single else b


def coefficient_stats(a: np.ndarray) -> dict[str, np.ndarray]:
    """Per-sample descriptors of a coefficient field used for the applicability check.

    ``a_min``, ``a_max``, ``contrast`` (max / min), ``high_fraction`` (share of cells above the mid value) and
    ``interface_density`` (share of interior faces across which ``a`` jumps: inversely proportional to the
    correlation length of the field).
    """
    a, _ = _batch(a)
    amin = a.reshape(a.shape[0], -1).min(axis=1)
    amax = a.reshape(a.shape[0], -1).max(axis=1)
    mid = 0.5 * (amin + amax)[:, None, None]
    jumps = (np.abs(np.diff(a, axis=1)) > 1e-9 * amax[:, None, None]).sum(axis=(1, 2)) + \
            (np.abs(np.diff(a, axis=2)) > 1e-9 * amax[:, None, None]).sum(axis=(1, 2))
    n = a.shape[-1]
    return {"a_min": amin, "a_max": amax, "contrast": amax / np.maximum(amin, 1e-300),
            "high_fraction": (a > mid).mean(axis=(1, 2)), "interface_density": jumps / (2.0 * n * (n - 1))}


@dataclass
class CoefficientEnvelope:
    """What the training coefficients looked like: the values ``a`` took and the range of the field statistics.

    ``fit(a_train)`` records the extreme values of ``a`` and the ``[q, 1 - q]`` quantiles of the contrast, high
    fraction and interface density; ``contains(a)`` says, per sample, whether a new coefficient lies inside.
    """
    a_min: float
    a_max: float
    ranges: dict[str, tuple[float, float]] = field(default_factory=dict)
    rel_tol: float = 1e-6

    @classmethod
    def fit(cls, a_train: np.ndarray, q: float = 0.005) -> CoefficientEnvelope:
        s = coefficient_stats(a_train)
        rng = {k: (float(np.quantile(s[k], q)), float(np.quantile(s[k], 1 - q)))
               for k in ("contrast", "high_fraction", "interface_density")}
        return cls(float(s["a_min"].min()), float(s["a_max"].max()), rng)

    def violations(self, a: np.ndarray) -> list[list[str]]:
        """Per sample, the descriptors outside the envelope (empty list = inside)."""
        s = coefficient_stats(a)
        tol = self.rel_tol * max(abs(self.a_max), 1.0)
        out: list[list[str]] = []
        for i in range(len(s["a_min"])):
            bad = []
            if s["a_min"][i] < self.a_min - tol:
                bad.append(f"min a {s['a_min'][i]:.3g} < {self.a_min:.3g}")
            if s["a_max"][i] > self.a_max + tol:
                bad.append(f"max a {s['a_max'][i]:.3g} > {self.a_max:.3g}")
            for k, (lo, hi) in self.ranges.items():
                v = float(s[k][i])
                if v < lo - 1e-12 or v > hi + 1e-12:
                    bad.append(f"{k} {v:.3g} outside [{lo:.3g}, {hi:.3g}]")
            out.append(bad)
        return out

    def contains(self, a: np.ndarray) -> np.ndarray:
        return np.array([not v for v in self.violations(a)])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_check_config(self) -> dict[str, Any]:
        """The ``envelope`` table of a ``pinneapple check`` result (``physics = "darcy"``)."""
        return {"a-min": self.a_min, "a-max": self.a_max,
                **{k.replace("_", "-"): [float(lo), float(hi)] for k, (lo, hi) in self.ranges.items()}}

    @classmethod
    def from_check_config(cls, d: Mapping[str, Any]) -> CoefficientEnvelope:
        """Inverse of :meth:`to_check_config`."""
        d = {str(k).replace("-", "_"): v for k, v in d.items()}
        ranges = {k: (float(v[0]), float(v[1])) for k, v in d.items() if k not in ("a_min", "a_max")}
        bad = set(ranges) - {"contrast", "high_fraction", "interface_density"}
        if bad:
            raise ValueError(f"unknown envelope key(s) {sorted(bad)}")
        return cls(float(d.get("a_min", -np.inf)), float(d.get("a_max", np.inf)), ranges)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> CoefficientEnvelope:
        return cls(float(d["a_min"]), float(d["a_max"]), {k: tuple(v) for k, v in d.get("ranges", {}).items()},
                   float(d.get("rel_tol", 1e-6)))


@dataclass
class DarcyCheck:
    """One check over a batch: the per-sample values, the aggregate compared with the limit, pass / fail."""
    name: str
    kind: str                 # physics | applicability | reference
    values: np.ndarray        # per sample
    value: float              # aggregate (quantile over the batch)
    limit: float | None
    passed: bool | None       # None = not run
    critical: bool = True
    detail: str = ""

    @property
    def failing_fraction(self) -> float:
        if self.limit is None or self.values.size == 0:
            return 0.0
        return float(np.mean(~(self.values <= self.limit)))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "value": self.value, "limit": self.limit,
                "passed": self.passed, "critical": self.critical, "failing_fraction": self.failing_fraction,
                "detail": self.detail}


def darcy_checks(a: np.ndarray, u: np.ndarray, f: float | np.ndarray = 1.0, *,
                 reference: np.ndarray | None = None, envelope: CoefficientEnvelope | None = None,
                 tolerances: Mapping[str, float] | None = None, quantile: float = 0.9,
                 physics_kind: str = "physics") -> list[DarcyCheck]:
    """Physics checks of predicted Darcy fields ``u`` for coefficients ``a`` (``(n, n)`` or ``(batch, n, n)``).

    Each check is computed per sample and aggregated with the ``quantile`` over the batch (0.9: nine samples in ten
    must pass); the per-sample values are kept. Physics checks (no reference needed): certified energy-error bound
    from the residual, boundary condition, flux balance, positivity (``f >= 0`` only). ``reference`` adds the
    relative L2 error; ``envelope`` adds the applicability of the coefficients (inside the training envelope).
    ``physics_kind`` labels the physics checks (e.g. ``"drift"`` when ``tolerances`` come from
    :func:`calibrate_tolerances`).
    """
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    a, u, _ = _pair(a, u)
    agg = lambda v: float(np.quantile(v, quantile)) if v.size else float("nan")   # noqa: E731
    checks: list[DarcyCheck] = []

    def add(name, kind, values, limit, detail, critical=True):
        values = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(values).all()
        v = agg(values) if finite else float("inf")
        checks.append(DarcyCheck(name, kind, values, v, limit, bool(finite and v <= limit), critical, detail))

    finite = np.isfinite(u).all(axis=(-2, -1))
    if not finite.all():
        checks.append(DarcyCheck("finite values", physics_kind, (~finite).astype(float), float((~finite).mean()), 0.0,
                                 False, True, f"{int((~finite).sum())} of {len(finite)} fields have NaN or inf"))
        return checks
    pm = physics_metrics(a, u, f)
    add("PDE residual (certified energy-error bound)", physics_kind, pm["energy_bound"], tol["energy_bound"],
        "||r||_{L^-1} / sqrt(min a) / ||u||_A from the finite-volume residual f + div(a grad u); bounds the "
        "relative energy-norm distance to the discrete solution")
    add("PDE residual (certified L2 error bound)", physics_kind, pm["l2_bound"], tol["l2_bound"],
        "the energy bound with the discrete Poincare inequality: bounds the relative L2 distance of u to the "
        "discrete solution")
    add("boundary condition u = 0", physics_kind, pm["bc"], tol["bc"], "RMS of u on the boundary (linear "
        "extrapolation from the first two cells) over max |u|")
    add("flux balance (net boundary flux = source)", physics_kind, pm["flux"], tol["flux"],
        "|outflow - integral of f| / |integral of f|, outflow = sum of 2 a u over the boundary faces")
    fmin = float(np.min(np.asarray(f, dtype=np.float64)))
    if fmin >= 0:
        add("positivity (maximum principle)", physics_kind, pm["positivity"], tol["positivity"],
            "most negative value over max |u| (f >= 0 and u = 0 on the boundary give u >= 0)")
    else:
        checks.append(DarcyCheck("positivity (maximum principle)", physics_kind, np.zeros(0), float("nan"), None, None,
                                 False, "not run: the source changes sign"))
    if reference is not None:
        ref, _ = _batch(reference)
        ref = np.broadcast_to(ref, u.shape)
        err = np.linalg.norm((u - ref).reshape(len(u), -1), axis=1) / \
            np.maximum(np.linalg.norm(ref.reshape(len(u), -1), axis=1), 1e-300)
        add("reference (relative L2)", "reference", err, tol["reference"], "relative L2 error against the reference "
            "solution (needs labels: evaluation only)")
    if envelope is not None:
        viol = envelope.violations(a)
        outside = np.array([1.0 if v else 0.0 for v in viol])
        first = next((v for v in viol if v), [])
        add("applicability (coefficient inside the training envelope)", "applicability", outside, 0.0,
            f"{int(outside.sum())} of {len(outside)} coefficients outside the training envelope"
            + (f", e.g. {'; '.join(first)}" if first else ""))
    return checks


def physics_metrics(a: np.ndarray, u: np.ndarray, f: float | np.ndarray = 1.0) -> dict[str, np.ndarray]:
    """Per-sample values of the physics checks, keyed like :data:`DEFAULT_TOLERANCES` (``energy_bound``,
    ``l2_bound``, ``bc``, ``flux``, ``positivity``); no reference solution needed."""
    a, u, _ = _pair(a, u)
    scale = np.maximum(np.max(np.abs(u), axis=(-2, -1)), 1e-300)
    return {"energy_bound": residual_error_bound(u, a, f),
            "l2_bound": residual_l2_bound(u, a, f),
            "bc": np.sqrt(np.mean(boundary_values(u) ** 2, axis=(-2, -1))) / scale,
            "flux": boundary_flux_balance(u, a, f)["imbalance"],
            "positivity": np.maximum(-np.min(u, axis=(-2, -1)), 0.0) / scale}


def calibrate_tolerances(a: np.ndarray, u: np.ndarray, f: float | np.ndarray = 1.0, *, quantile: float = 0.99,
                         floor: Mapping[str, float] | None = None) -> dict[str, float]:
    """Limits for :func:`darcy_checks` from a model's own predictions ``u`` on held-out in-distribution inputs ``a``.

    The fixed :data:`DEFAULT_TOLERANCES` say whether a prediction obeys the physics to an engineering tolerance.
    These say whether a new prediction is as physical as the model was where it was validated: per check, the
    ``quantile`` of the per-sample values on the calibration inputs (at least ``floor``, default 1e-12). No
    reference solution is used, so the limits can be refreshed on any batch of in-distribution inputs. With the
    default aggregation of :func:`darcy_checks` (the 0.9 quantile of a batch), a batch from the calibration
    distribution passes, and a batch that departs from it is flagged.
    """
    m = physics_metrics(a, u, f)
    fl = {k: 1e-12 for k in m} | dict(floor or {})
    return {k: max(float(np.quantile(v, quantile)), fl[k]) for k, v in m.items()}


@dataclass(frozen=True)
class UseProfile:
    """What a prediction is used for, and the physics limits that use needs. ``limits``: metric (a key of
    :func:`physics_metrics`) -> limit; ``reasons``: why each limit, written before any model is checked."""
    name: str
    use: str
    limits: Mapping[str, float]
    reasons: Mapping[str, str]


#: Use profiles. The limits are the library's fixed engineering tolerances (:data:`DEFAULT_TOLERANCES`), grouped by
#: the quantity a use depends on; none is chosen from a model's values.
PROFILES: dict[str, UseProfile] = {
    "fields": UseProfile(
        "fields", "maps and point values of u (pressure / head)",
        {"l2_bound": 0.05, "bc": 0.02, "positivity": 0.01},
        {"l2_bound": "5 % relative L2, the tolerance the library applies to a reference comparison; here certified "
                     "from the residual, without the reference",
         "bc": "u = 0 to 2 % of max |u| (RMS), the Dirichlet tolerance of pp.trust_card; the reference solver "
               "reaches 0.2 % at 128 x 128",
         "positivity": "maximum principle (f >= 0): no value below -1 % of max |u|, the max-principle tolerance of "
                       "pp.trust_card"}),
    "fluxes": UseProfile(
        "fluxes", "flow rates through the boundary and wall gradients (Darcy velocity -a grad u)",
        {"energy_bound": 0.10, "flux": 0.02},
        {"energy_bound": "10 % error of the flux field a grad u in the energy norm, certified from the residual; "
                         "the dataset's 512^2 -> 128^2 averaging alone leaves 2 to 9 %",
         "flux": "net outflow equal to the source integral within 2 %; the reference solver reaches 1 %"}),
}


def profile_verdicts(a: np.ndarray, u: np.ndarray, f: float | np.ndarray = 1.0, *,
                     profiles: Mapping[str, UseProfile] | None = None, quantile: float = 0.9) -> dict[str, Any]:
    """Per use profile, over a batch: each metric's ``quantile`` value, its limit and ``ratio = value / limit``,
    the profile's ``ratio`` (the worst one, 1 = at the limit) and its decision (APPROVED when every metric is
    within its limit, else REJECT)."""
    pm = physics_metrics(a, u, f)
    out: dict[str, Any] = {}
    for name, prof in (profiles or PROFILES).items():
        rows = {}
        for k, lim in prof.limits.items():
            v = float(np.quantile(pm[k], quantile))
            rows[k] = {"value": v, "limit": lim, "ratio": v / lim, "failing_fraction": float(np.mean(pm[k] > lim))}
        worst = max(rows, key=lambda k: rows[k]["ratio"])
        out[name] = {"decision": "APPROVED" if rows[worst]["ratio"] <= 1 else "REJECT", "ratio": rows[worst]["ratio"],
                     "worst": worst, "metrics": rows, "use": prof.use}
    return out


#: Names of the per-sample checks of :func:`sample_checks` (also the subjects ``pinneapple check`` reports).
SAMPLE_CHECKS = ("energy_bound", "l2_bound", "flux", "bc", "positivity", "a_min", "a_max", "contrast",
                 "high_fraction", "interface_density")


def sample_checks(a: np.ndarray, u: np.ndarray, f: float = 1.0, *, tolerances: Mapping[str, float] | None = None,
                  envelope: CoefficientEnvelope | None = None) -> list[dict[str, Any]]:
    """The checks of one prediction (``(n, n)``), each ``{"key", "value", "limit", "passed"}``, as ``pinneapple
    check`` runs them (``physics = "darcy"``): the physics metrics against ``tolerances`` (default
    :data:`DEFAULT_TOLERANCES`) and, with ``envelope``, the coefficient against the training envelope (values of
    ``a``, contrast, high fraction, interface density). A figure that shows a verdict next to a prediction uses this
    function, so it says what the CI says."""
    tol = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    pm = physics_metrics(a, u, f)
    out = []
    for k in ("energy_bound", "l2_bound", "flux", "bc", "positivity"):
        if k == "positivity" and float(np.min(np.asarray(f, dtype=np.float64))) < 0:
            continue
        v = float(np.ravel(pm[k])[0])
        out.append({"key": k, "value": v, "limit": tol[k], "passed": bool(np.isfinite(v) and v <= tol[k])})
    if envelope is not None:
        st = {k: float(v[0]) for k, v in coefficient_stats(a).items()}
        tol_a = envelope.rel_tol * max(abs(envelope.a_max), 1.0)
        out.append({"key": "a_min", "value": st["a_min"], "limit": envelope.a_min,
                    "passed": st["a_min"] >= envelope.a_min - tol_a})
        out.append({"key": "a_max", "value": st["a_max"], "limit": envelope.a_max,
                    "passed": st["a_max"] <= envelope.a_max + tol_a})
        for k, (lo, hi) in envelope.ranges.items():
            out.append({"key": k, "value": st[k], "limit": [lo, hi],
                        "passed": lo - 1e-12 <= st[k] <= hi + 1e-12})
    return out


def sample_verdict(checks: Sequence[Mapping[str, Any]]) -> str:
    """PASS when every check of :func:`sample_checks` passes, else FAIL (the exit status of ``pinneapple check``)."""
    return "PASS" if all(c["passed"] for c in checks) else "FAIL"


# --------------------------------------------------------------------------------------------- where it works
def estimated_error(u: np.ndarray, a: np.ndarray, f: float | np.ndarray = 1.0) -> np.ndarray:
    """Map of the error of ``u`` estimated from its residual alone: ``e ~ L^-1 r / a_h``.

    ``A_a e = r`` holds exactly for the error ``e = u_h - u`` to the discrete solution. Replacing ``A_a`` by the
    constant-coefficient operator ``a_h L`` (``a_h`` the harmonic mean of ``a``, one fast sine transform) gives this
    estimate. It is not a certified bound (see :func:`residual_error_bound` for that), but it says where the error
    is. With the harmonic mean it tends to over-estimate: on the PhysicsNeMo Darcy example it is 1.4 to 2.7 times the
    true error, with a correlation of 0.7 to 0.85 to the true error map.
    """
    a, u, single = _pair(a, u)
    ah = 1.0 / np.mean(1.0 / a, axis=(-2, -1))
    e = _laplace_inverse(darcy_residual(u, a, f)) / ah[:, None, None]
    return e[0] if single else e


def flux_magnitude(u: np.ndarray, a: np.ndarray) -> np.ndarray:
    """``|a grad u|`` per cell from the solver's own face fluxes (harmonic-mean faces, half-cell Dirichlet faces):
    each component is the mean of the cell's two face flux densities."""
    a, u, single = _pair(a, u)
    n = u.shape[-1]
    h = 1.0 / n
    comps = []
    for ax in (1, 2):
        sl_l = [slice(None)] * 3
        sl_r = [slice(None)] * 3
        sl_l[ax], sl_r[ax] = slice(None, -1), slice(1, None)
        al, ar = a[tuple(sl_l)], a[tuple(sl_r)]
        q_in = 2 * al * ar / (al + ar) * (u[tuple(sl_l)] - u[tuple(sl_r)]) / h        # interior faces, + direction
        lo = [slice(None)] * 3
        hi = [slice(None)] * 3
        lo[ax], hi[ax] = slice(0, 1), slice(-1, None)
        q_lo = -2 * a[tuple(lo)] * u[tuple(lo)] / h                                  # into the domain at the low wall
        q_hi = 2 * a[tuple(hi)] * u[tuple(hi)] / h
        faces = np.concatenate([q_lo, q_in, q_hi], axis=ax)
        f_l = [slice(None)] * 3
        f_r = [slice(None)] * 3
        f_l[ax], f_r[ax] = slice(None, -1), slice(1, None)
        comps.append(0.5 * (faces[tuple(f_l)] + faces[tuple(f_r)]))
    q = np.hypot(*comps)
    return q[0] if single else q


def estimated_flux_error(u: np.ndarray, a: np.ndarray, f: float | np.ndarray = 1.0) -> np.ndarray:
    """Map of the error of the flux ``a grad u`` estimated from the residual alone (:func:`estimated_error`),
    relative to the largest flux of ``u``: ``|a grad e| / max |a grad u|``."""
    a, u, single = _pair(a, u)
    q = flux_magnitude(estimated_error(u, a, f), a) / np.maximum(flux_magnitude(u, a).max(axis=(-2, -1)),
                                                                   1e-300)[:, None, None]
    return q[0] if single else q


def region_masks(n: int, wall_cells: int = 2, interior_from: int = 8) -> dict[str, np.ndarray]:
    """Boolean masks of an ``n x n`` cell grid: ``wall`` (cells closer than ``wall_cells`` to the boundary) and
    ``interior`` (at least ``interior_from`` cells away)."""
    i = np.arange(n)
    d = np.minimum.outer(np.minimum(i, n - 1 - i), np.minimum(i, n - 1 - i))
    return {"wall": d < wall_cells, "interior": d >= interior_from}


def regional_errors(u: np.ndarray, a: np.ndarray, f: float | np.ndarray = 1.0, *, wall_cells: int = 2,
                    interior_from: int = 8) -> dict[str, np.ndarray]:
    """Relative error estimated without a reference (:func:`estimated_error`) in the ``wall`` and ``interior``
    regions of :func:`region_masks`, per sample: ``||e||_region / ||u||_region``."""
    a, u, single = _pair(a, u)
    e = estimated_error(u, a, f)
    out = {}
    for k, m in region_masks(u.shape[-1], wall_cells, interior_from).items():
        out[k] = np.linalg.norm((e * m).reshape(len(u), -1), axis=1) / \
            np.maximum(np.linalg.norm((u * m).reshape(len(u), -1), axis=1), 1e-300)
    return {k: v[0] for k, v in out.items()} if single else out


#: Uses of a Darcy prediction and what each needs; limits fixed a priori (the library's tolerances).
USES: dict[str, dict[str, Any]] = {
    "interior field": {"needs": {"interior": 0.05}, "why": "values of u away from the walls (maps, maxima): "
                       "estimated interior error within the 5 % of a reference comparison"},
    "walls and fluxes": {"needs": {"wall": 0.05, "flux": 0.02}, "why": "values and gradients next to the walls, "
                         "boundary flow rates: estimated wall-band error within 5 % and flux balance within 2 %"},
}


def use_verdicts(a: np.ndarray, u: np.ndarray, f: float = 1.0, *, envelope: CoefficientEnvelope | None = None,
                 wall_cells: int = 2, interior_from: int = 8) -> dict[str, dict[str, Any]]:
    """For one prediction (``(n, n)``): which uses it can serve, and why (:data:`USES`), from the residual alone,
    plus ``inputs`` (the coefficient inside the training envelope). Each entry: ``ok``, the measured ``values``, the
    ``limits`` and a one-line ``reason``."""
    reg = regional_errors(u, a, f, wall_cells=wall_cells, interior_from=interior_from)
    vals = {"interior": float(reg["interior"]), "wall": float(reg["wall"]),
            "flux": float(boundary_flux_balance(u, a, f)["imbalance"])}
    names = {"interior": "interior error", "wall": "wall-band error", "flux": "flux imbalance"}
    out: dict[str, dict[str, Any]] = {}
    for use, spec in USES.items():
        lim = spec["needs"]
        bad = [k for k in lim if not vals[k] <= lim[k]]
        worst = max(lim, key=lambda k: vals[k] / lim[k])
        reason = (f"{names[worst]} ~{100 * vals[worst]:.1f} % > {100 * lim[worst]:g} %" if bad else
                  f"{names[worst]} ~{100 * vals[worst]:.1f} % <= {100 * lim[worst]:g} %")
        out[use] = {"ok": not bad, "values": {k: vals[k] for k in lim}, "limits": dict(lim), "reason": reason,
                    "why": spec["why"]}
    if envelope is not None:
        viol = envelope.violations(a)[0]
        out["inputs"] = {"ok": not viol, "values": {}, "limits": {}, "why": "coefficient inside the training "
                         "envelope", "reason": "inside the training envelope" if not viol else viol[0]}
    return out


def trust_report(checks: Sequence[DarcyCheck], *, model: str = "",
                 kinds: Sequence[str] | None = None):
    """The deploy decision (``TrustReport``: APPROVED / REVIEW / REJECT) of a list of :class:`DarcyCheck`, e.g. the
    checks of several :func:`darcy_checks` calls (fixed limits and calibrated ones); ``kinds`` keeps only those."""
    from pinneapple_analysis.trust.trust_report import Check, TrustReport
    out = []
    for c in checks:
        if kinds is not None and c.kind not in kinds:
            continue
        status = "not_run" if c.passed is None else ("supports" if c.passed else "contradicts")
        name = c.name if c.kind in ("physics", "reference", "applicability") else f"{c.name} [{c.kind}]"
        out.append(Check(name, status, critical=c.critical, value=None if not np.isfinite(c.value) else c.value,
                         limit=c.limit, source=c.kind, reason=c.detail))
    return TrustReport(out, model=model)


def darcy_trust_report(a: np.ndarray, u: np.ndarray, f: float | np.ndarray = 1.0, *, model: str = "",
                       kinds: Sequence[str] = ("physics", "reference", "applicability"), **kwargs):
    """``(TrustReport, checks)``: the deploy decision (APPROVED / REVIEW / REJECT) of :func:`darcy_checks`.

    ``kinds`` selects which checks enter the decision (``("physics",)`` gives the verdict the physics alone reach,
    without labels or the training envelope).
    """
    checks = [c for c in darcy_checks(a, u, f, **kwargs) if c.kind in kinds]
    return trust_report(checks, model=model), checks


# --------------------------------------------------------------------------------------------- export
def save_darcy_npz(path, a: np.ndarray, u: np.ndarray, **extra: np.ndarray):
    """One Darcy field as the grid result ``pinneapple check`` reads (``physics = "darcy"``): cell-centre axes
    ``x``, ``y`` of the unit square, coefficient ``a`` and solution ``u`` (float32)."""
    from pathlib import Path
    a, u = np.asarray(a), np.asarray(u)
    if a.shape != u.shape or a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"a and u must be the same (n, n) grid, got {a.shape} and {u.shape}")
    c = (np.arange(a.shape[0]) + 0.5) / a.shape[0]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, x=c, y=c, a=a.astype(np.float32), u=u.astype(np.float32), **extra)
    return path


#: Tolerance names of ``pinneapple check`` for the metrics of :func:`physics_metrics`.
CHECK_TOLERANCE_KEYS = {"energy_bound": "darcy-energy", "l2_bound": "darcy-l2", "flux": "darcy-flux",
                        "bc": "darcy-bc", "positivity": "darcy-positivity"}


def check_config_entry(name: str, path: str, *, f: float = 1.0, tolerances: Mapping[str, float] | None = None,
                       envelope: CoefficientEnvelope | None = None, coefficient: str = "a",
                       solution: str = "u") -> str:
    """One ``[[result]]`` table of a ``pinneapple-check.toml`` for a Darcy prediction saved with
    :func:`save_darcy_npz`: ``physics = "darcy"``, the source, the tolerances (metric keys of
    :func:`physics_metrics`) and the coefficient envelope (bounds of ``a`` and the envelope statistics)."""
    lines = ["[[result]]", f'name = "{name}"', f'path = "{path}"', 'physics = "darcy"', f"parameters = {{ f = {f:g} }}",
             f'units = {{ {coefficient} = "1", {solution} = "1" }}']
    if envelope is not None:
        lines.append(f"bounds = {{ {coefficient} = [{envelope.a_min:g}, {envelope.a_max:g}] }}")
        env = ", ".join(f"{k} = {v!r}" if not isinstance(v, list) else f"{k} = [{v[0]:.6g}, {v[1]:.6g}]"
                        for k, v in envelope.to_check_config().items())
        lines.append(f"envelope = {{ {env} }}")
    if tolerances:
        lines.append("tolerances = { " + ", ".join(f"{CHECK_TOLERANCE_KEYS[k]} = {v:.6g}"
                                                    for k, v in tolerances.items()) + " }")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------------- node-based data
# Datasets that store Darcy fields at the nodes of a finite-difference grid (spacing ``dx``) with zero Dirichlet
# values on ghost nodes outside the stored block, e.g. PhysicsNeMo's ``Darcy2D`` datapipe: ghost at index -1 and,
# after its one-node crop, at index n + 1 on the high side (``cropped = 1``). The documented equation
# -div(k grad u) = f is written as -(k lap u + grad k . grad u) = f with central differences, the form such solvers
# iterate; ``cross_denominator`` is the denominator of the product of the two central differences in grad k . grad u
# (``4 dx^2`` for the equation; another value reproduces a stencil that differs from it).

def _nodal_pad(u: np.ndarray, k: np.ndarray):
    U = np.pad(u, ((0, 0), (1, 1), (1, 1)))                       # zero ghosts (high side used only if not cropped)
    K = np.pad(k, ((0, 0), (1, 1), (1, 1)), mode="edge")          # coefficient clamped at the edges
    return U, K


def nodal_darcy_residual(u: np.ndarray, k: np.ndarray, f: float = 1.0, dx: float = 1.0, *, cropped: int = 1,
                         cross_denominator: float | None = None) -> np.ndarray:
    """``-(k lap u + grad k . grad u) - f`` at the stored nodes whose four neighbours are known: ``(batch, m, m)``
    with ``m = n - 1`` when the high side is cropped, ``n`` otherwise."""
    u, single = _batch(u)
    k, _ = _batch(k)
    U, K = _nodal_pad(u, k)
    cd = 4.0 * dx * dx if cross_denominator is None else float(cross_denominator)
    c = (slice(None), slice(1, -1), slice(1, -1))
    e, w = (slice(None), slice(2, None), slice(1, -1)), (slice(None), slice(None, -2), slice(1, -1))
    nn, s = (slice(None), slice(1, -1), slice(2, None)), (slice(None), slice(1, -1), slice(None, -2))
    lap = (U[e] + U[w] + U[nn] + U[s] - 4 * U[c]) / (dx * dx)
    cross = ((K[e] - K[w]) * (U[e] - U[w]) + (K[nn] - K[s]) * (U[nn] - U[s])) / cd
    r = -(K[c] * lap + cross) - f
    if cropped:
        r = r[:, :-1, :-1]
    return r[0] if single else r


def nodal_boundary_values(u: np.ndarray, *, cropped: int = 1) -> np.ndarray:
    """``u`` extrapolated linearly from the two nearest stored nodes to the zero-ghost nodes on the four sides:
    ``(batch, 4, n)`` (x_min, x_max, y_min, y_max). Low ghost at index -1, high ghost at index ``n + cropped``."""
    u, single = _batch(u)
    g = 1 + cropped                                       # distance from the last stored node to the high ghost
    b = np.stack([2 * u[:, 0, :] - u[:, 1, :], u[:, -1, :] + g * (u[:, -1, :] - u[:, -2, :]),
                  2 * u[:, :, 0] - u[:, :, 1], u[:, :, -1] + g * (u[:, :, -1] - u[:, :, -2])], axis=1)
    return b[0] if single else b


def nodal_flux_balance(u: np.ndarray, k: np.ndarray, f: float = 1.0, dx: float = 1.0, *,
                       cropped: int = 1) -> np.ndarray:
    """Relative imbalance of a control-volume budget around the stored nodes (whole block if not cropped, all but
    the last row and column if cropped): net outward flux (harmonic-mean face coefficients) against ``f`` times the
    area. Zero for a conservative solution; per sample."""
    u, single = _batch(u)
    k, _ = _batch(k)
    U, K = _nodal_pad(u, k)
    m = u.shape[-1] - (1 if cropped else 0)
    hm = lambda p, q: 2 * p * q / (p + q)                          # noqa: E731
    i0, i1 = 1, m                                                  # padded indices of the first / last node in the box
    out = np.zeros(len(u))
    sl = slice(i0, i1 + 1)
    for ax in (1, 2):
        def take(i, ax=ax):
            idx = [slice(None), sl, sl]
            idx[ax] = i
            return tuple(idx)
        out += np.sum(hm(K[take(i0)], K[take(i0 - 1)]) * (U[take(i0)] - U[take(i0 - 1)]), axis=-1)
        out += np.sum(hm(K[take(i1)], K[take(i1 + 1)]) * (U[take(i1)] - U[take(i1 + 1)]), axis=-1)
    src = f * (m * dx) ** 2
    imb = np.abs(out - src) / abs(src)
    return imb[0] if single else imb


def solve_nodal_darcy(k: np.ndarray, f: float = 1.0, dx: float = 1.0, *,
                      cross_denominator: float | None = None) -> np.ndarray:
    """Direct solve of ``-(k lap u + grad k . grad u) = f`` on an ``n x n`` node block with zero ghosts at index -1
    and ``n`` (coefficient clamped at the edges): the reference for :func:`nodal_darcy_residual` with ``cropped=0``."""
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla
    k = np.asarray(k, dtype=np.float64)
    n = k.shape[0]
    K = np.pad(k, 1, mode="edge")
    cd = 4.0 * dx * dx if cross_denominator is None else float(cross_denominator)
    idx = np.arange(n * n).reshape(n, n)
    kc = K[1:-1, 1:-1]
    rows, cols, vals = [idx.ravel()], [idx.ravel()], [(4 * kc / dx ** 2).ravel()]
    for (di, dj), dk in (((1, 0), K[2:, 1:-1] - K[:-2, 1:-1]), ((-1, 0), -(K[2:, 1:-1] - K[:-2, 1:-1])),
                         ((0, 1), K[1:-1, 2:] - K[1:-1, :-2]), ((0, -1), -(K[1:-1, 2:] - K[1:-1, :-2]))):
        coef = -(kc / dx ** 2 + dk / cd)
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        ok = (ii + di >= 0) & (ii + di < n) & (jj + dj >= 0) & (jj + dj < n)
        rows.append(idx[ok])
        cols.append(idx[(ii + di)[ok], (jj + dj)[ok]])
        vals.append(coef[ok])
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n * n, n * n))
    return spla.spsolve(A.tocsc(), np.full(n * n, float(f))).reshape(n, n)


def _groups(x: np.ndarray, rtol: float = 1e-6) -> list[int]:
    """Index of the first field of each group of identical fields (equal to ``rtol`` of their scale everywhere)."""
    return [g[0] for g in _group_fields(x, rtol)]


def _group_fields(x: np.ndarray, rtol: float = 1e-6) -> list[list[int]]:
    x, _ = _batch(x)
    flat = x.reshape(len(x), -1)
    scale = max(float(np.max(np.abs(flat))), 1e-300)
    groups: list[list[int]] = []
    for i in range(len(flat)):
        for g in groups:
            if np.max(np.abs(flat[i] - flat[g[0]])) <= rtol * scale:
                g.append(i)
                break
        else:
            groups.append([i])
    return groups


def _audit():
    """``pinneapple_data.audit`` (numpy only), without the package ``__init__`` when the checker's loader is there."""
    try:
        from pinneapple_check._reuse import leaf
        return leaf("pinneapple_data.audit")
    except ImportError:
        import importlib
        return importlib.import_module("pinneapple_data.audit")


def check_nodal_darcy_batch(k: np.ndarray, u: np.ndarray, f: float = 1.0, dx: float = 1.0, *, cropped: int = 1,
                            alt_cross_denominators: Mapping[str, float] | None = None,
                            tolerances: Mapping[str, float] | None = None,
                            floor_factor: float = 2.0, n_reference: int = 8) -> dict[str, Any]:
    """Physics checks of a batch of node-based Darcy training pairs ``(k, u)`` before anyone trains on them.

    Residual of the documented equation (relative RMS over the batch, ``residual`` limit), boundary condition
    (``bc``), control-volume flux balance (``flux``), positivity and the diversity of the coefficient samples. For
    reference, the same equation solved directly for up to ``n_reference`` distinct coefficients
    (:func:`solve_nodal_darcy`, coefficient clamped over the cropped edge) is put through the same checks; the boundary and flux limits are at least
    ``floor_factor`` times what that exact solve reaches on the same grid. ``alt_cross_denominators`` (name -> value) gives the
    residual of other stencils, to say which equation the data do satisfy. Returns a JSON-able dict with the values and pass / fail.
    """
    tol = {"residual": 1e-2, "bc": 2e-2, "flux": 2e-2, "positivity": 1e-2, **(tolerances or {})}
    k = np.asarray(k, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)

    def metrics(kk, uu):
        r = nodal_darcy_residual(uu, kk, f, dx, cropped=cropped)
        scale = np.maximum(np.max(np.abs(uu), axis=(-2, -1)), 1e-300)
        return {"residual": np.sqrt(np.mean(r ** 2, axis=(-2, -1))) / abs(f),
                "bc": np.sqrt(np.mean(nodal_boundary_values(uu, cropped=cropped) ** 2, axis=(-2, -1))) / scale,
                "flux": nodal_flux_balance(uu, kk, f, dx, cropped=cropped),
                "positivity": np.maximum(-np.min(uu, axis=(-2, -1)), 0.0) / scale}

    def summary(m, limits):
        out = {}
        for name, v in m.items():
            v = np.atleast_1d(v)
            lim = limits[name]
            out[name] = {"median": float(np.median(v)), "max": float(np.max(v)), "limit": lim,
                         "failing_fraction": float(np.mean(v > lim)), "passed": bool(np.median(v) <= lim)}
        return out

    n = k.shape[-1]
    reps = _groups(k)[:n_reference]                 # one direct solve per distinct coefficient
    us = np.stack([solve_nodal_darcy(np.pad(k[i], ((0, cropped), (0, cropped)), mode="edge") if cropped else k[i],
                                     f, dx)[:n, :n] for i in reps])
    ref_m = metrics(k[reps], us)
    # the boundary extrapolation and the control-volume budget of a non-conservative stencil have a floor on a
    # coarse grid: the limit is the larger of the tolerance and twice what the exact solves reach on this grid
    limits = {name: max(tol[name], floor_factor * float(np.max(ref_m[name]))) if name in ("bc", "flux")
              else tol[name] for name in ref_m}
    ref = summary(ref_m, limits)
    data = summary(metrics(k, u), limits)
    audit = _audit()                                               # diversity: the dataset audit's own measures
    dup = audit.near_duplicates(k)
    div = {"n_fields": len(k), "n_distinct": len(_groups(k)), "exact_copies": dup["exact"],
           "near_copies": dup["near"], "effective_sample_count": audit.effective_sample_count(k)}
    data["distinct_coefficients"] = {**div, "passed": dup["near"] == 0                 # centred: rank <= n - 1
                                     and div["effective_sample_count"] >= min(2.0, len(k) - 1) - 1e-9}
    alt = {}
    for name, cd in (alt_cross_denominators or {}).items():
        r = nodal_darcy_residual(u, k, f, dx, cropped=cropped, cross_denominator=cd)
        alt[name] = float(np.median(np.sqrt(np.mean(r ** 2, axis=(-2, -1))) / abs(f)))
    diff = np.linalg.norm((u[reps] - us).reshape(len(reps), -1), axis=1) / \
        np.maximum(np.linalg.norm(us.reshape(len(reps), -1), axis=1), 1e-300)
    return {"checks": data, "direct_solve_of_the_documented_equation": {"checks": ref,
                                                                        "n_solved": len(reps),
                                                                        "relative_l2_data_vs_solve":
                                                                            float(np.median(diff))},
            "residual_of_other_stencils": alt,
            "passed": all(v["passed"] for v in data.values())}
