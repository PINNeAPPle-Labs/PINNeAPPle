"""PINN validation (#703): the residual at points never used in training, detectors for the known failure modes, the
exact-solution gate and a training-time monitor.

A PINN's training loss is measured on its own collocation points and mixes the residual with boundary and initial
terms, so a low loss says little about the solution. :func:`validate_pinn` evaluates a trained network on fresh points
and names what went wrong:

``pinn.residual``
    The PDE residual on a dense uniform grid, on fresh random points and on adaptive points (the neighbourhood of the
    largest residuals of a large random sample): RMS and maximum, normalised by the problem's residual scale; the map
    on the grid is in the evidence.
``pinn.trivial``
    Near-trivial solution: the spread and the gradient energy of the field against the exact solution's (or the
    problem's expected scale). A network that stopped near its initial, nearly constant state fails here.
``pinn.causality``
    Propagation failure in time (compare Wang, Sankaran & Perdikaris 2022, "Respecting causality is all you need"):
    the residual concentrated at late times, or -- with an exact solution -- an error that grows in time while the
    residual stays small (a solution of the PDE that did not come from the initial condition).
``pinn.boundary`` / ``pinn.initial``
    Boundary and initial conditions on fresh points, against the solution scale. When they fail while the residual
    passes, the condition was traded off against the residual (a boundary weight too small).
``pinn.spectrum``
    Frequency content against the exact solution (``validation.spectral``), or of the operator ``L[u]`` against the
    forcing ``f`` when there is no exact solution: a PINN's spectral bias is missing high-band energy.
``pinn.exact``
    Relative L2 and maximum error against the exact solution, where one exists (manufactured solutions, Cole-Hopf for
    Burgers, Kovasznay).

Built-in problems: :func:`burgers_1d` (Cole-Hopf by quadrature), :func:`poisson_2d` (manufactured) and
:func:`kovasznay` (steady Navier-Stokes). Any other problem is a :class:`PINNProblem` with a residual written with
:func:`d` (autograd derivatives). :class:`PINNMonitor` runs the same checks every N epochs and stops a training run
on a failure mode.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .spectral import spectral_comparison

__all__ = ["PINNProblem", "PINNReport", "PINNMonitor", "FAILURE_MODES", "d", "validate_pinn", "train_pinn", "mlp",
           "burgers_1d",
           "cole_hopf", "poisson_2d", "kovasznay", "PROBLEMS"]

FAILURE_MODES = {
    "pinn.residual": "PDE residual too large at fresh points",
    "pinn.trivial": "near-trivial solution",
    "pinn.causality": "propagation failure in time",
    "pinn.boundary": "boundary condition not satisfied",
    "pinn.initial": "initial condition not satisfied",
    "pinn.spectrum": "spectral bias: frequency content differs from the reference",
    "pinn.exact": "error against the exact solution",
}

TOLERANCES = {"residual": 5e-2, "residual_max": 0.3, "trivial": 0.2, "late": 10.0, "boundary": 1e-2,
              "spectrum": 0.25, "spectrum_floor": 1e-6, "exact": 2e-2}


def d(u, X, i: int, order: int = 1):
    """``d^order u / d X_i^order`` by autograd (``u``: ``(N,)`` or ``(N, 1)``, ``X``: ``(N, dim)`` with grad)."""
    import torch
    for _ in range(order):
        u = torch.autograd.grad(u.sum(), X, create_graph=True)[0][:, i:i + 1]
    return u


@dataclass
class PINNProblem:
    """What :func:`validate_pinn` needs to know about a problem.

    ``residual(u, X)``: the PDE residual(s), ``(N, k)``, from the network output ``u`` ``(N, m)`` at ``X`` (``(N, dim)``,
    requires grad; use :func:`d`). ``boundary(n, rng)`` / ``initial(n, rng)``: fresh points and target values
    ``(n, m)`` (NaN where an output is unconstrained). ``exact(X)``: ``(N, m)`` numpy. ``lhs`` / ``forcing``: the
    operator and right-hand side, ``L[u] = f``, for the spectrum check without an exact solution. ``scale``: the
    magnitude of the solution; ``residual_scale``: of the terms of the PDE (the residual is reported relative to it).
    """
    name: str
    bounds: Sequence[tuple[float, float]]
    residual: Callable[[Any, Any], Any]
    boundary: Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]] | None = None
    initial: Callable[[int, np.random.Generator], tuple[np.ndarray, np.ndarray]] | None = None
    exact: Callable[[np.ndarray], np.ndarray] | None = None
    lhs: Callable[[Any, Any], Any] | None = None
    forcing: Callable[[np.ndarray], np.ndarray] | None = None
    time_axis: int | None = None
    scale: float = 1.0
    residual_scale: float = 1.0
    grid: Sequence[int] = (64, 64)
    outputs: Sequence[str] = ("u",)
    gauge: Sequence[int] = ()           # outputs defined up to a constant (pressure): compared after removing means
    # residual conditions (Neumann, Robin): (name, sampler(n, rng) -> X, fn(u, X) -> residual); train_pinn adds
    # weights["constraint"] x their mean square
    constraints: Sequence[tuple[str, Callable, Callable]] = ()

    @property
    def dim(self) -> int:
        return len(self.bounds)

    def mesh(self, grid: Sequence[int] | None = None) -> tuple[np.ndarray, tuple[int, ...]]:
        g = tuple(grid or self.grid)
        axes = [np.linspace(lo, hi, n) for (lo, hi), n in zip(self.bounds, g, strict=True)]
        M = np.meshgrid(*axes, indexing="ij")
        return np.stack([m.ravel() for m in M], 1), g

    def uniform(self, n: int, rng: np.random.Generator) -> np.ndarray:
        lo, hi = np.array(self.bounds, float).T
        return lo + (hi - lo) * rng.random((n, self.dim))


# ----------------------------------------------------------------------------------------------- evaluation helpers
def _forward(model, X):
    out = model(X)
    if isinstance(out, (tuple, list)):
        out = out[0]
    if hasattr(out, "y"):
        out = out.y
    return out if out.ndim == 2 else out.reshape(len(X), -1)


def _predict(model, X: np.ndarray, chunk: int = 8192) -> np.ndarray:
    import torch
    dt = next((p.dtype for p in getattr(model, "parameters", lambda: [])()), torch.float32)
    with torch.no_grad():
        return np.concatenate([_forward(model, torch.tensor(X[i:i + chunk], dtype=dt)).double().numpy()
                               for i in range(0, len(X), chunk)])


def _apply(model, fn, X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """``fn(u, X)`` (residual or operator) at ``X``, by autograd, as ``(N, k)`` numpy."""
    import torch
    dt = next((p.dtype for p in getattr(model, "parameters", lambda: [])()), torch.float32)
    out = []
    for i in range(0, len(X), chunk):
        Xt = torch.tensor(X[i:i + chunk], dtype=dt, requires_grad=True)
        r = fn(_forward(model, Xt), Xt)
        r = r if r.ndim == 2 else r.reshape(len(Xt), -1)
        out.append(r.detach().double().numpy())
    return np.concatenate(out)


def _rms(a: np.ndarray) -> float:
    a = a[np.isfinite(a)]
    return float(np.sqrt(np.mean(a ** 2))) if a.size else float("nan")


def _cond_error(model, sampler, n: int, rng) -> float:
    X, target = sampler(n, rng)
    pred = _predict(model, np.asarray(X, float))
    target = np.asarray(target, float).reshape(len(X), -1)
    m = np.isfinite(target)
    return _rms(pred[:, :target.shape[1]][m] - target[m])


def _check(cid: str, status: str, summary: str, value=None, threshold=None) -> dict[str, Any]:
    return {"id": cid, "status": status, "summary": summary, "value": value, "threshold": threshold}


@dataclass
class PINNReport:
    problem: str
    checks: list[dict[str, Any]]
    evidence: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, cid: str) -> dict[str, Any]:
        return next(c for c in self.checks if c["id"] == cid)

    @property
    def failure_modes(self) -> list[str]:
        """The failing checks, most specific diagnosis first (the residual and the exact error are symptoms)."""
        order = ["pinn.trivial", "pinn.causality", "pinn.boundary", "pinn.initial", "pinn.spectrum", "pinn.residual",
                 "pinn.exact"]
        bad = {c["id"] for c in self.checks if c["status"] == "fail"}
        return [FAILURE_MODES[i] for i in order if i in bad]

    @property
    def passed(self) -> bool:
        return all(c["status"] != "fail" for c in self.checks)

    def summary(self) -> str:
        lines = [f"PINN validation: {self.problem} -- " + ("pass" if self.passed else
                                                           "FAIL: " + "; ".join(self.failure_modes))]
        lines += [f"  [{c['status']:>4}] {c['id']}: {c['summary']}" for c in self.checks]
        return "\n".join(lines)


# ----------------------------------------------------------------------------------------------- validate_pinn
def validate_pinn(model, problem: PINNProblem | str, *, n_points: int = 4096, grid: Sequence[int] | None = None,
                  train_points: np.ndarray | None = None, tolerances: dict[str, float] | None = None,
                  seed: int = 0) -> PINNReport:
    """Validate a trained PINN ``model`` (a torch module or callable ``X -> u``) on ``problem``.

    ``train_points``: the collocation points used in training, if known, to report the residual there next to the
    fresh-point residual (a ratio well above 1 is overfitting the collocation set). ``tolerances`` override
    :data:`TOLERANCES`."""
    if isinstance(problem, str):
        problem = PROBLEMS[problem]()
    tol = {**TOLERANCES, **(tolerances or {})}
    rng = np.random.default_rng(seed)
    P = problem
    checks: list[dict[str, Any]] = []
    ev: dict[str, Any] = {}

    # ---- residual at fresh points: grid, random, adaptive
    Xg, g = P.mesh(grid)
    rg = np.linalg.norm(_apply(model, P.residual, Xg), axis=1) / P.residual_scale
    Xr = P.uniform(n_points, rng)
    rr = np.linalg.norm(_apply(model, P.residual, Xr), axis=1) / P.residual_scale
    Xc = P.uniform(8 * n_points, rng)
    rc = np.linalg.norm(_apply(model, P.residual, Xc), axis=1)
    top = Xc[np.argsort(rc)[-max(n_points // 8, 1):]]
    lo, hi = np.array(P.bounds, float).T
    Xa = np.clip(top + 0.01 * (hi - lo) * rng.standard_normal(top.shape), lo, hi)
    ra = np.linalg.norm(_apply(model, P.residual, Xa), axis=1) / P.residual_scale
    rms, mx = _rms(rr), float(np.nanmax(np.concatenate([rg, rr, ra])))
    ev["residual"] = {"rms_random": rms, "rms_grid": _rms(rg), "max": mx, "max_adaptive": float(np.nanmax(ra)),
                      "map": rg.reshape(g)}
    if train_points is not None:
        rt = _rms(np.linalg.norm(_apply(model, P.residual, np.asarray(train_points, float)), axis=1)
                  / P.residual_scale)
        ev["residual"]["rms_train"] = rt
        ev["residual"]["fresh_over_train"] = rms / max(rt, 1e-300)
    finite = np.isfinite(mx)
    ok = finite and rms <= tol["residual"] and mx <= tol["residual_max"]
    checks.append(_check("pinn.residual", "pass" if ok else "fail",
                         f"relative residual at fresh points: RMS {rms:.2e} (<= {tol['residual']:g}), max {mx:.2e} "
                         f"(<= {tol['residual_max']:g}, adaptive points included)", rms, tol["residual"]))

    # ---- the field on the grid
    U = _predict(model, Xg)
    Ue = P.exact(Xg) if P.exact is not None else None
    for j in P.gauge:
        U[:, j] -= U[:, j].mean()
        if Ue is not None:
            Ue[:, j] -= Ue[:, j].mean()

    # ---- near-trivial solution
    spread = float(np.mean(U.std(0)))
    gU = _grad_energy(U, g, P)
    if Ue is not None:
        s_ref, g_ref = float(np.mean(Ue.std(0))), _grad_energy(Ue, g, P)
    else:
        s_ref, g_ref = P.scale, None
    ratios = [spread / max(s_ref, 1e-300)] + ([gU / max(g_ref, 1e-300)] if g_ref else [])
    r_triv = float(min(ratios))
    ev["trivial"] = {"spread": spread, "spread_ref": s_ref, "grad": gU, "grad_ref": g_ref}
    checks.append(_check("pinn.trivial", "pass" if r_triv >= tol["trivial"] else "fail",
                         f"spread and gradient energy {r_triv:.2f} x the "
                         + ("exact solution's" if Ue is not None else "problem scale")
                         + f" (>= {tol['trivial']:g})", r_triv, tol["trivial"]))

    # ---- causality / propagation in time
    if P.time_axis is not None:
        T = np.moveaxis(rg.reshape(g), P.time_axis, 0)
        prof = np.sqrt(np.mean(T.reshape(len(T), -1) ** 2, axis=1))
        q = max(len(prof) // 4, 1)
        early, late = float(np.sqrt(np.mean(prof[:q] ** 2))), float(np.sqrt(np.mean(prof[-q:] ** 2)))
        conc = late / max(early, 1e-300)
        ev["causality"] = {"residual_by_time": prof, "late_over_early": conc}
        bad = conc > tol["late"] and late > tol["residual"] and early <= tol["residual"]
        why = (f"residual in the last quarter of the time interval {late:.1e}, first quarter {early:.1e} "
               f"(late/early {conc:.1f}, <= {tol['late']:g} unless the late residual passes)")
        if bad:
            why += ": the residual is resolved early and concentrated at late times"
        if Ue is not None:
            E = np.moveaxis((np.linalg.norm(U - Ue, axis=1)).reshape(g), P.time_axis, 0)
            eprof = np.sqrt(np.mean(E.reshape(len(E), -1) ** 2, axis=1)) / max(P.scale, 1e-300)
            ev["causality"]["error_by_time"] = eprof
            e_late, e_early = float(np.max(eprof[-q:])), float(np.max(eprof[:q]))
            grows = e_late > tol["exact"] and e_late > 5 * e_early and ok
            bad = bad or grows
            why += f"; error early {e_early:.1e} -> late {e_late:.1e}" + (
                " with a small residual: a solution of the PDE that did not propagate from the initial condition"
                if grows else "")
        checks.append(_check("pinn.causality", "fail" if bad else "pass", why, conc, tol["late"]))
    else:
        checks.append(_check("pinn.causality", "not-applicable", "steady problem (no time axis)"))

    # ---- boundary and initial conditions on fresh points
    for cid, sampler, what in (("pinn.boundary", P.boundary, "boundary"), ("pinn.initial", P.initial, "initial")):
        if sampler is None:
            checks.append(_check(cid, "not-applicable", f"the problem has no {what} condition to check"))
            continue
        e = _cond_error(model, sampler, n_points // 4, rng) / max(P.scale, 1e-300)
        ev[what] = {"rms_error": e, "balance": e / max(rms, 1e-300)}
        fine = e <= tol["boundary"]
        msg = f"{what} condition error {e:.2e} of the solution scale (<= {tol['boundary']:g})"
        if not fine and ok:
            msg += (f"; the residual passes ({rms:.1e}): the {what} condition was traded off against the residual "
                    f"(error / residual {e / max(rms, 1e-300):.0f})")
        checks.append(_check(cid, "pass" if fine else "fail", msg, e, tol["boundary"]))

    # ---- frequency content
    sp_axes = [i for i in range(P.dim) if i != P.time_axis]
    try:
        if Ue is not None:
            a, b, what = U, Ue, "the exact solution"
        elif P.lhs is not None and P.forcing is not None:
            a, b, what = _apply(model, P.lhs, Xg), P.forcing(Xg).reshape(len(Xg), -1), "the forcing (L[u] vs f)"
        else:
            a = None
        if a is None:
            checks.append(_check("pinn.spectrum", "not-applicable", "no exact solution and no forcing to compare"))
        else:
            biases = []
            for j in range(a.shape[1]):
                fa, fb = (np.moveaxis(x[:, j].reshape(g), sp_axes, range(P.dim - len(sp_axes), P.dim))
                          for x in (a, b))
                if np.ptp(fb) < 1e-12 * max(P.scale, 1.0):
                    continue
                biases.append(spectral_comparison(fa, fb, len(sp_axes), periodic=False,
                                                  floor=tol["spectrum_floor"]).hf_bias)
            hf = float(max(biases, key=abs)) if biases else 0.0
            ev["spectrum"] = {"hf_bias": biases}
            checks.append(_check("pinn.spectrum", "pass" if abs(hf) <= tol["spectrum"] else "fail",
                                 f"high-wavenumber energy {hf:+.0%} relative to {what} "
                                 f"(within +/-{tol['spectrum']:.0%})", hf, tol["spectrum"]))
    except ValueError as exc:                                          # a reference with no resolved high band
        checks.append(_check("pinn.spectrum", "not-applicable", f"spectrum not resolved: {exc}"))

    # ---- exact solution gate
    if Ue is not None:
        rel = float(np.linalg.norm(U - Ue) / max(np.linalg.norm(Ue), 1e-300))
        emax = float(np.abs(U - Ue).max() / max(P.scale, 1e-300))
        ev["exact"] = {"rel_l2": rel, "max": emax}
        checks.append(_check("pinn.exact", "pass" if rel <= tol["exact"] else "fail",
                             f"relative L2 error {rel:.2e} (<= {tol['exact']:g}), max {emax:.2e} of the scale",
                             rel, tol["exact"]))
    else:
        checks.append(_check("pinn.exact", "not-applicable", "no exact solution for this problem"))
    return PINNReport(P.name, checks, ev)


def _grad_energy(U: np.ndarray, g: tuple[int, ...], P: PINNProblem) -> float:
    """RMS of the gradient over the spatial axes (time excluded), times the domain size: a dimensionless amplitude."""
    tot = 0.0
    for j in range(U.shape[1]):
        F = U[:, j].reshape(g)
        for i in range(P.dim):
            if i == P.time_axis:
                continue
            lo, hi = P.bounds[i]
            h = (hi - lo) / (g[i] - 1)
            tot += float(np.mean(np.gradient(F, h, axis=i) ** 2)) * (hi - lo) ** 2
    return math.sqrt(tot / U.shape[1])


# ----------------------------------------------------------------------------------------------- training monitor
class PINNMonitor:
    """The checks of :func:`validate_pinn` every ``every`` epochs during training.

    Call ``monitor(model, epoch)`` from the training loop; it returns the report when it ran (else None) and sets
    ``stop`` once a failure mode in ``stop_on`` (check ids, e.g. ``"pinn.boundary"``) has failed ``patience`` times
    in a row after ``warmup`` epochs (early on, every network is near-trivial). ``history`` keeps (epoch, failing
    ids). Small ``n_points`` and ``grid`` keep it cheap."""

    def __init__(self, problem: PINNProblem | str, *, every: int = 100, warmup: int = 0, patience: int = 2,
                 stop_on: Sequence[str] = ("pinn.trivial", "pinn.causality", "pinn.boundary", "pinn.initial"),
                 n_points: int = 1024, grid: Sequence[int] | None = (32, 32), **kw):
        self.problem = PROBLEMS[problem]() if isinstance(problem, str) else problem
        self.every, self.warmup, self.patience, self.stop_on = every, warmup, patience, tuple(stop_on)
        self.kw = {"n_points": n_points, "grid": grid, **kw}
        self.history: list[tuple[int, list[str]]] = []
        self.stop, self.reason, self._streak = False, None, 0

    def __call__(self, model, epoch: int) -> PINNReport | None:
        if epoch % self.every:
            return None
        was = getattr(model, "training", None)
        rep = validate_pinn(model, self.problem, **self.kw)
        if was:
            model.train()
        failing = [c["id"] for c in rep.checks if c["status"] == "fail"]
        self.history.append((epoch, failing))
        hit = [i for i in failing if i in self.stop_on]
        self._streak = self._streak + 1 if hit and epoch >= self.warmup else 0
        if self._streak >= self.patience:
            self.stop, self.reason = True, "; ".join(FAILURE_MODES[i] for i in hit)
        return rep


# ----------------------------------------------------------------------------------------------- a reference trainer
def mlp(n_in: int, n_out: int, width: int = 32, depth: int = 4):
    """A tanh multilayer perceptron, the usual PINN network."""
    import torch
    layers: list = [torch.nn.Linear(n_in, width), torch.nn.Tanh()]
    for _ in range(depth - 1):
        layers += [torch.nn.Linear(width, width), torch.nn.Tanh()]
    return torch.nn.Sequential(*layers, torch.nn.Linear(width, n_out))


def train_pinn(problem: PINNProblem | str, model=None, *, n_collocation: int = 2000, n_condition: int = 400,
               adam_steps: int = 1500, lbfgs_steps: int = 1000, lr: float = 2e-3, weights: dict[str, float] | None
               = None, monitor: PINNMonitor | None = None, seed: int = 0, data: tuple | None = None,
               callback: Callable[[int, float], None] | None = None):
    """Train a PINN on ``problem`` with soft constraints: Adam, then L-BFGS; loss = mean squared residual (relative to
    the residual scale) + ``weights["boundary"]`` x boundary + ``weights["initial"]`` x initial misfit (default 10
    each; 0 drops the term). ``monitor`` is called every Adam step and ends training when it sets ``stop``.

    ``problem.constraints`` (residual conditions such as Neumann) add ``weights["constraint"]`` (default 10) x their
    mean square; ``data=(X, Y)`` adds ``weights["data"]`` (default 10) x the misfit to observations (NaN in ``Y``:
    unobserved), which with trainable coefficients inside ``problem.residual`` makes an inverse PINN. ``callback(step,
    loss)`` is called every Adam step.

    Returns ``(model, info)`` with ``info = {"train_points", "steps", "stopped", "loss", "history"}``. A plain
    reference trainer for testing the checks, not a tuned one."""
    import torch
    P = PROBLEMS[problem]() if isinstance(problem, str) else problem
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = model if model is not None else mlp(P.dim, len(P.outputs))
    w = {"boundary": 10.0, "initial": 10.0, "constraint": 10.0, "data": 10.0, **(weights or {})}
    Xc = P.uniform(n_collocation, rng)
    X = torch.tensor(Xc, dtype=torch.float32, requires_grad=True)
    terms = []
    for key, sampler in (("boundary", P.boundary), ("initial", P.initial)):
        if sampler is not None and w[key] > 0:
            Xb, T = sampler(n_condition, rng)
            T = torch.tensor(np.asarray(T, float).reshape(len(Xb), -1), dtype=torch.float32)
            terms.append((torch.tensor(Xb, dtype=torch.float32), T, torch.isfinite(T), w[key]))
    if data is not None and w["data"] > 0:
        Xd, Yd = (np.asarray(a, float) for a in data)
        Yd = torch.tensor(Yd.reshape(len(Xd), -1), dtype=torch.float32)
        terms.append((torch.tensor(Xd, dtype=torch.float32), Yd, torch.isfinite(Yd), w["data"]))
    cons = []
    for _name, sampler, fn in (getattr(P, "constraints", ()) or ()):
        if w["constraint"] > 0:
            cons.append((torch.tensor(np.asarray(sampler(n_condition, rng), float), dtype=torch.float32,
                                      requires_grad=True), fn))

    def loss():
        total = (P.residual(_forward(net, X), X) ** 2).mean() / P.residual_scale ** 2
        for Xb, T, m, wk in terms:
            total = total + wk * ((_forward(net, Xb)[:, :T.shape[1]] - torch.nan_to_num(T)) ** 2)[m].mean()
        for Xk, fn in cons:
            total = total + w["constraint"] * (fn(_forward(net, Xk), Xk) ** 2).mean()
        return total

    opt = torch.optim.Adam(net.parameters(), lr)
    step, stopped = 0, False
    history: list[float] = []
    for step in range(1, adam_steps + 1):
        opt.zero_grad()
        lv = loss()
        lv.backward()
        opt.step()
        history.append(float(lv.detach()))
        if callback is not None:
            callback(step, history[-1])
        if monitor is not None:
            monitor(net, step)
            if monitor.stop:
                stopped = True
                break
    if lbfgs_steps and not stopped:
        lb = torch.optim.LBFGS(net.parameters(), max_iter=lbfgs_steps, line_search_fn="strong_wolfe",
                               tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            lb.zero_grad()
            lv = loss()
            lv.backward()
            return lv
        lb.step(closure)
    return net, {"train_points": Xc, "steps": step, "stopped": stopped, "loss": float(loss().detach()),
                 "history": history}


# ----------------------------------------------------------------------------------------------- built-in problems
def cole_hopf(x: np.ndarray, t: np.ndarray, nu: float, n: int = 2001) -> np.ndarray:
    """Exact solution of ``u_t + u u_x = nu u_xx``, ``u(x, 0) = -sin(pi x)``, by the Cole-Hopf transform (the
    heat-kernel integral, trapezoidal rule over 10 standard deviations, exponent shifted for range)."""
    x, t = np.broadcast_arrays(np.asarray(x, float), np.asarray(t, float))
    z = np.linspace(-10.0, 10.0, n)
    out = -np.sin(np.pi * x)
    m = t > 0
    xs, ts = x[m][:, None], t[m][:, None]
    y = xs - np.sqrt(4 * nu * ts) * z[None]
    lf = -np.cos(np.pi * y) / (2 * np.pi * nu) - z[None] ** 2
    w = np.exp(lf - lf.max(axis=1, keepdims=True))
    out[m] = -(np.sin(np.pi * y) * w).sum(1) / w.sum(1)
    return out


def burgers_1d(nu: float = 0.05 / math.pi) -> PINNProblem:
    """``u_t + u u_x = nu u_xx`` on ``x in [-1, 1], t in [0, 1]``, ``u(x, 0) = -sin(pi x)``, ``u(+-1, t) = 0``
    (Raissi et al. 2019 with a larger viscosity by default). Inputs ``(x, t)``; exact by Cole-Hopf."""
    def residual(u, X):
        return d(u, X, 1) + u * d(u, X, 0) - nu * d(u, X, 0, 2)

    def boundary(n, rng):
        t = rng.random(n)
        x = np.where(rng.random(n) < 0.5, -1.0, 1.0)
        return np.stack([x, t], 1), np.zeros((n, 1))

    def initial(n, rng):
        x = rng.uniform(-1, 1, n)
        return np.stack([x, np.zeros(n)], 1), -np.sin(np.pi * x)[:, None]

    return PINNProblem("burgers_1d", [(-1.0, 1.0), (0.0, 1.0)], residual, boundary, initial,
                       exact=lambda X: cole_hopf(X[:, 0], X[:, 1], nu)[:, None], time_axis=1, scale=1.0,
                       residual_scale=math.pi, grid=(128, 64))


def poisson_2d(kx: int = 1, ky: int = 2, exact: bool = True) -> PINNProblem:
    """``-lap u = f`` on the unit square, ``u = 0`` on the boundary, manufactured ``u = sin(kx pi x) sin(ky pi y)``.
    ``exact=False`` hides the exact solution (the spectrum check then compares ``-lap u`` with ``f``)."""
    c = (kx ** 2 + ky ** 2) * math.pi ** 2

    def ue(X):
        return (np.sin(kx * math.pi * X[:, 0]) * np.sin(ky * math.pi * X[:, 1]))[:, None]

    def lhs(u, X):
        return -(d(u, X, 0, 2) + d(u, X, 1, 2))

    def f(X):
        return c * ue(X)

    def residual(u, X):
        import torch
        fx = c * torch.sin(kx * math.pi * X[:, :1]) * torch.sin(ky * math.pi * X[:, 1:2])
        return lhs(u, X) - fx

    def boundary(n, rng):
        s = rng.random(n)
        side = rng.integers(0, 4, n)
        X = np.stack([np.choose(side, [s, s, np.zeros(n), np.ones(n)]),
                      np.choose(side, [np.zeros(n), np.ones(n), s, s])], 1)
        return X, np.zeros((n, 1))

    return PINNProblem(f"poisson_2d({kx},{ky})", [(0.0, 1.0), (0.0, 1.0)], residual, boundary, None,
                       exact=ue if exact else None, lhs=lhs, forcing=f, scale=1.0, residual_scale=c / 2,
                       grid=(64, 64))


def kovasznay(re: float = 40.0) -> PINNProblem:
    """Steady incompressible Navier-Stokes, Kovasznay (1948) flow on ``[-0.5, 1] x [-0.5, 1.5]``; outputs
    ``(u, v, p)``, velocity prescribed on the boundary (the same flow as the dataset ``kovasznay_ns``); the pressure
    is compared up to a constant."""
    nu = 1.0 / re
    lam = re / 2 - math.sqrt(re ** 2 / 4 + 4 * math.pi ** 2)

    def ue(X):
        x, y = X[:, 0], X[:, 1]
        e = np.exp(lam * x)
        return np.stack([1 - e * np.cos(2 * math.pi * y), lam / (2 * math.pi) * e * np.sin(2 * math.pi * y),
                         0.5 * (1 - np.exp(2 * lam * x))], 1)

    def residual(out, X):
        import torch
        u, v, p = out[:, :1], out[:, 1:2], out[:, 2:3]
        ux, uy, vx, vy = d(u, X, 0), d(u, X, 1), d(v, X, 0), d(v, X, 1)
        r1 = u * ux + v * uy + d(p, X, 0) - nu * (d(u, X, 0, 2) + d(u, X, 1, 2))
        r2 = u * vx + v * vy + d(p, X, 1) - nu * (d(v, X, 0, 2) + d(v, X, 1, 2))
        return torch.cat([r1, r2, ux + vy], 1)

    def boundary(n, rng):
        s = rng.random(n)
        side = rng.integers(0, 4, n)
        x = np.choose(side, [-0.5 + 1.5 * s, -0.5 + 1.5 * s, np.full(n, -0.5), np.ones(n)])
        y = np.choose(side, [np.full(n, -0.5), np.full(n, 1.5), -0.5 + 2 * s, -0.5 + 2 * s])
        X = np.stack([x, y], 1)
        T = ue(X)
        T[:, 2] = np.nan                                               # pressure is not prescribed
        return X, T

    return PINNProblem(f"kovasznay(Re={re:g})", [(-0.5, 1.0), (-0.5, 1.5)], residual, boundary, None, exact=ue,
                       scale=1.0, residual_scale=1.0, grid=(64, 64), outputs=("u", "v", "p"), gauge=(2,))


PROBLEMS: dict[str, Callable[[], PINNProblem]] = {"burgers_1d": burgers_1d, "poisson_2d": poisson_2d,
                                                  "kovasznay": kovasznay}
