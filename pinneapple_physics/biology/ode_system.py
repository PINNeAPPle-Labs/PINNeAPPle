"""A well-mixed biological model as symbolic ODEs: simulate it, export it, turn it into a problem or a forward model.

:class:`ODESystem` is the common form of every model this package reads (SBML, :mod:`.sbml`) or builds (the PK/PD
and endocrine models in :mod:`.models`): state variables with ``d x / dt = f(x, p, t)`` as SymPy expressions,
algebraic variables (assignment rules) evaluated in order, parameters with values, and initial values that may be
expressions of the parameters (SBML initial assignments).

From one system:

* :meth:`ODESystem.simulate` integrates it with SciPy (``BDF`` by default: biological networks are often stiff), with
  the analytic Jacobian, and returns every state and algebraic variable on the requested times;
* :meth:`ODESystem.rate_fn` is the right-hand side in PyTorch for the ``reaction_kinetics_network`` residual of the
  PINN compiler, and :meth:`ODESystem.to_problem_spec` the :class:`ProblemSpec` that uses it;
* :meth:`ODESystem.forward_model` is a :class:`~pinneapple_analysis.inverse_problems.forward.ForwardModel` for
  identifiability, calibration and inversion: observables are states or algebraic variables, the parameters are any
  subset of the system's (initial values included, as ``<state>_0``), and the sensitivities come from the forward
  sensitivity equations (exact for the ODE, not finite differences);
* :meth:`ODESystem.to_sbml` writes it back as SBML Level 3 (rate rules), so a model can be checked by another
  simulator (libRoadRunner, COPASI) and re-imported.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sympy as sp

__all__ = ["ODESystem", "Trajectory", "TIME"]

TIME = sp.Symbol("t", real=True)


@dataclass
class Trajectory:
    """Solution of :meth:`ODESystem.simulate` on the requested times: ``values[name]`` for every state and algebraic
    variable, ``success`` and the solver message."""
    t: np.ndarray
    values: dict[str, np.ndarray]
    success: bool = True
    message: str = ""
    n_rhs: int = 0

    def __getitem__(self, name: str) -> np.ndarray:
        return self.values[name]

    def matrix(self, names: Sequence[str]) -> np.ndarray:
        return np.column_stack([self.values[n] for n in names])


def _sym(name: str) -> sp.Symbol:
    return TIME if name == "t" else sp.Symbol(name, real=True)


def _jacobian(f: sp.Matrix, X) -> sp.Matrix:
    """Jacobian with ``x**n / x`` combined into ``x**(n-1)``: SymPy differentiates ``x**n`` (symbolic ``n``, e.g. a
    Hill coefficient) to ``n*x**n/x``, which is 0/0 at ``x = 0``, a common initial value."""
    J = f.jacobian(X)
    return J.applyfunc(lambda e: sp.powsimp(e, force=True)
                       if any(not p.exp.is_Integer for p in e.atoms(sp.Pow)) else e)


def _integrate(fun, t_eval, z0, *, breakpoints=(), jac=None, **kw):
    """``solve_ivp`` restarted at each breakpoint inside the interval (a dose, a meal, a switched input): across a
    jump of the right-hand side an adaptive step otherwise shrinks to nothing, above all for sensitivities that start
    at exactly zero. Returns ``(Y (n, len(t_eval)), success, message, nfev)``."""
    from scipy.integrate import solve_ivp

    t_eval = np.asarray(t_eval, dtype=float)
    t0, t1 = float(t_eval[0]), float(t_eval[-1])
    cuts = [t0] + sorted(b for b in set(breakpoints) if t0 < b < t1) + [t1]
    Y = np.full((len(z0), t_eval.size), np.nan)
    z = np.asarray(z0, dtype=float)
    nfev, msg = 0, ""
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        last = b == t1
        m = (t_eval >= a) & ((t_eval <= b) if last else (t_eval < b))
        te = np.unique(np.concatenate([t_eval[m], [b]]))
        # inside a segment the right-hand side takes its one-sided limits at the ends (Heaviside(0) is 0 or 1 by
        # convention; the segment's own value is the one that holds between the breakpoints)
        lo, hi = a + 1e-9 * (b - a), b - 1e-9 * (b - a)
        def f_seg(t, y, lo=lo, hi=hi):
            return fun(min(max(t, lo), hi), y)
        j_seg = None if jac is None else (lambda t, y, lo=lo, hi=hi: jac(min(max(t, lo), hi), y))
        sol = solve_ivp(f_seg, (a, b), z, t_eval=te, jac=j_seg, **kw)
        nfev += int(sol.nfev)
        msg = str(sol.message)
        if not sol.success:
            return Y, False, msg, nfev
        idx = np.searchsorted(te, t_eval[m])
        Y[:, m] = sol.y[:, idx]
        z = sol.y[:, -1]
    return Y, True, msg, nfev


@dataclass
class ODESystem:
    """``d states / dt = rhs`` with algebraic ``assignments`` and ``parameters``.

    ``rhs`` and ``assignments`` map names to SymPy expressions in the symbols ``Symbol(name, real=True)`` and the
    time :data:`TIME`; ``assignments`` is ordered so each one only uses earlier ones (``topological`` order, as SBML
    requires). ``initial`` maps each state to a number or an expression of parameters (and earlier assignments).
    ``inputs`` are known functions of time (e.g. a dosing schedule) given as SymPy expressions in :data:`TIME`; they
    are substituted like assignments.
    """
    name: str
    states: list[str]
    rhs: dict[str, sp.Expr]
    initial: dict[str, Any]
    parameters: dict[str, float] = field(default_factory=dict)
    assignments: dict[str, sp.Expr] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        missing = [s for s in self.states if s not in self.rhs]
        if missing:
            raise ValueError(f"no right-hand side for state(s) {missing}")
        missing = [s for s in self.states if s not in self.initial]
        if missing:
            raise ValueError(f"no initial value for state(s) {missing}")
        self.parameters = {k: float(v) for k, v in self.parameters.items()}
        self._cache: dict[Any, Any] = {}
        unknown = self.free_symbols() - {_sym(n) for n in [*self.states, *self.parameters, *self.assignments]} \
            - {TIME}
        if unknown:
            raise ValueError(f"{self.name}: symbol(s) with no value, state or rule: {sorted(map(str, unknown))}")

    # ------------------------------------------------------------------ symbolic forms
    @classmethod
    def from_strings(cls, name: str, rhs: Mapping[str, str], initial: Mapping[str, Any],
                     parameters: Mapping[str, float], assignments: Mapping[str, str] | None = None,
                     **kw) -> ODESystem:
        """Build from text, e.g. ``rhs={"x": "-k*x"}``. The text is parsed by SymPy's parser with only the system's
        names in scope (a model written in code; for untrusted text use :mod:`pinneapple_physics.symbolic_pde.text`)."""
        names = [*rhs, *parameters, *(assignments or {})]
        local = {n: _sym(n) for n in names}
        local["t"] = TIME
        conv = lambda s: sp.sympify(s, locals=local) if isinstance(s, str) else sp.sympify(s)   # noqa: E731
        return cls(name, list(rhs), {k: conv(v) for k, v in rhs.items()},
                   {k: (conv(v) if isinstance(v, str) else v) for k, v in initial.items()},
                   dict(parameters), {k: conv(v) for k, v in (assignments or {}).items()}, **kw)

    def free_symbols(self) -> set:
        out: set = set()
        for e in [*self.rhs.values(), *self.assignments.values()]:
            out |= sp.sympify(e).free_symbols
        for v in self.initial.values():
            out |= sp.sympify(v).free_symbols
        return out

    def expanded(self, expr) -> sp.Expr:
        """``expr`` with every assignment substituted (in reverse order, so nested rules resolve)."""
        e = sp.sympify(expr)
        for name in reversed(list(self.assignments)):
            e = e.subs(_sym(name), self.assignments[name])
        return e

    def expanded_rhs(self) -> list[sp.Expr]:
        key = "rhs_expanded"
        if key not in self._cache:
            self._cache[key] = [self.expanded(self.rhs[s]) for s in self.states]
        return self._cache[key]

    def breakpoints(self) -> list[float]:
        """Times where the right-hand side jumps: zeros of ``Heaviside`` arguments and of ``Piecewise`` conditions
        that depend on time only (doses, meals, pulses), evaluated at the nominal parameters."""
        key = "breakpoints"
        if key not in self._cache:
            subs = {_sym(k): v for k, v in self.parameters.items()}
            out = set()
            for e in self.expanded_rhs():
                args = [h.args[0] for h in e.atoms(sp.Heaviside)]
                for pw in e.atoms(sp.Piecewise):
                    for _, c in pw.args:
                        if isinstance(c, sp.core.relational.Relational):
                            args.append(c.lhs - c.rhs)
                for a in args:
                    a = sp.sympify(a).subs(subs)
                    if a.free_symbols == {TIME} and sp.degree(a, TIME) == 1:
                        out.add(float(sp.solve(a, TIME)[0]))
            self._cache[key] = sorted(out)
        return self._cache[key]

    def state_symbols(self) -> list[sp.Symbol]:
        return [_sym(s) for s in self.states]

    def parameter_symbols(self) -> list[sp.Symbol]:
        return [_sym(p) for p in self.parameters]

    def with_parameters(self, **values: float) -> ODESystem:
        """A copy with some parameter values changed."""
        unknown = set(values) - set(self.parameters)
        if unknown:
            raise KeyError(f"unknown parameter(s) {sorted(unknown)}")
        return ODESystem(self.name, list(self.states), dict(self.rhs), dict(self.initial),
                         {**self.parameters, **values}, dict(self.assignments), dict(self.units), dict(self.meta))

    # ------------------------------------------------------------------ numeric forms
    def _numeric(self):
        if "numeric" not in self._cache:
            X, P = self.state_symbols(), self.parameter_symbols()
            f = sp.Matrix(self.expanded_rhs())
            J = _jacobian(f, X)
            args = (TIME, X, P)
            self._cache["numeric"] = (sp.lambdify(args, list(f), "numpy", cse=True),
                                      sp.lambdify(args, J.tolist(), "numpy", cse=True))
        return self._cache["numeric"]

    def initial_values(self, parameters: Mapping[str, float] | None = None, t0: float = 0.0) -> np.ndarray:
        """Initial state vector, evaluating initial assignments with these parameter values."""
        p = {**self.parameters, **(parameters or {})}
        subs = {_sym(k): v for k, v in p.items()}
        subs[TIME] = t0
        out = []
        for s in self.states:
            e = self.expanded(sp.sympify(self.initial[s]))
            # an initial value may refer to other states' initial values (SBML initial assignments)
            for _ in range(len(self.states)):
                syms = e.free_symbols & set(self.state_symbols())
                if not syms:
                    break
                e = e.subs({x: self.expanded(sp.sympify(self.initial[str(x)])) for x in syms})
            out.append(float(e.subs(subs)))
        return np.asarray(out, dtype=float)

    def simulate(self, t_eval, parameters: Mapping[str, float] | None = None, *, y0=None, t0: float | None = None,
                 method: str = "BDF", rtol: float = 1e-8, atol: float = 1e-10, max_step: float = math.inf,
                 outputs: Sequence[str] | None = None) -> Trajectory:
        """Integrate from ``t0`` (default ``min(0, t_eval[0])``: initial values are those at t = 0, as in SBML) and
        return states and algebraic variables at ``t_eval``.

        ``max_step`` should be set when an input has short pulses (a bolus, a meal) that an adaptive step could jump
        over."""
        t_eval = np.asarray(t_eval, dtype=float)
        t0 = min(0.0, float(t_eval[0])) if t0 is None else float(t0)
        p = {**self.parameters, **(parameters or {})}
        pv = [p[k] for k in self.parameters]
        f, J = self._numeric()
        y0 = self.initial_values(p, t0) if y0 is None else np.asarray(y0, dtype=float)
        n = len(self.states)
        grid = t_eval if t0 >= t_eval[0] else np.concatenate([[t0], t_eval])
        Y, ok, msg, nfev = _integrate(
            lambda t, y: np.asarray(f(t, y, pv), dtype=float).reshape(n), grid, y0, breakpoints=self.breakpoints(),
            method=method, rtol=rtol, atol=atol, max_step=max_step,
            jac=None if method in ("RK45", "RK23", "DOP853") else
            (lambda t, y: np.asarray(J(t, y, pv), dtype=float).reshape(n, n)))
        Y = Y[:, -t_eval.size:]
        values = {s: Y[i] for i, s in enumerate(self.states)}
        names = list(self.assignments) if outputs is None else [o for o in outputs if o in self.assignments]
        if names:
            values.update(self.evaluate(names, t_eval, values, p))
        return Trajectory(t_eval, values, ok, msg, nfev)

    def evaluate(self, names: Sequence[str], t, states: Mapping[str, np.ndarray],
                 parameters: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
        """Algebraic variables (or any assignment) on a trajectory."""
        p = {**self.parameters, **(parameters or {})}
        X, P = self.state_symbols(), self.parameter_symbols()
        out = {}
        for n in names:
            fn = sp.lambdify((TIME, X, P), self.expanded(self.assignments[n]), "numpy")
            v = fn(np.asarray(t, dtype=float), [np.asarray(states[s]) for s in self.states],
                   [p[k] for k in self.parameters])
            out[n] = np.broadcast_to(np.asarray(v, dtype=float), np.shape(t)).copy()
        return out

    def residual(self, t, states: Mapping[str, np.ndarray], derivatives: Mapping[str, np.ndarray],
                 parameters: Mapping[str, float] | None = None) -> dict[str, np.ndarray]:
        """``dx/dt - f(x)`` per state for a trajectory from any method (a PINN, a surrogate, measured data)."""
        p = {**self.parameters, **(parameters or {})}
        f, _ = self._numeric()
        t = np.asarray(t, dtype=float)
        F = f(t, [np.asarray(states[s], dtype=float) for s in self.states], [p[k] for k in self.parameters])
        return {s: np.asarray(derivatives[s], dtype=float) - np.broadcast_to(np.asarray(F[i], dtype=float), t.shape)
                for i, s in enumerate(self.states)}

    # ------------------------------------------------------------------ torch / PINN
    def torch_rhs(self, trainable: Sequence[str] = ()) -> Callable:
        """``rhs(t, x, theta)`` in PyTorch: ``t`` (N,1), ``x`` dict of (N,1) tensors per state, ``theta`` dict of
        tensors for the names in ``trainable`` (the other parameters take their values). Returns a dict of rates."""
        import torch

        X, P = self.state_symbols(), self.parameter_symbols()
        def wrap(tf, mf):
            return lambda *a: tf(*a) if any(isinstance(x, torch.Tensor) for x in a) else mf(*a)

        def heaviside(x, *_):
            return (x > 0).to(x.dtype) if isinstance(x, torch.Tensor) else float(x > 0)

        mods = [{"exp": wrap(torch.exp, math.exp), "log": wrap(torch.log, math.log),
                 "sqrt": wrap(torch.sqrt, math.sqrt), "sin": wrap(torch.sin, math.sin),
                 "cos": wrap(torch.cos, math.cos), "tanh": wrap(torch.tanh, math.tanh), "Abs": wrap(torch.abs, abs),
                 "Heaviside": heaviside, "Max": wrap(lambda a, b: torch.clamp(a, min=b) if not isinstance(b, torch.Tensor)
                                                     else torch.maximum(torch.as_tensor(a), b), max),
                 "Min": wrap(lambda a, b: torch.clamp(a, max=b) if not isinstance(b, torch.Tensor)
                             else torch.minimum(torch.as_tensor(a), b), min)}, "math"]
        fns = [sp.lambdify((TIME, X, P), e, modules=mods) for e in self.expanded_rhs()]
        fixed = dict(self.parameters)
        trainable = list(trainable)

        def rhs(t, x, theta):
            pv = [theta[k] if k in trainable else fixed[k] for k in self.parameters]
            xv = [x[s] for s in self.states]
            out = {}
            for s, fn in zip(self.states, fns, strict=True):
                v = fn(t, xv, pv)
                out[s] = v if isinstance(v, torch.Tensor) else torch.full_like(t, float(v))
            return out
        return rhs

    def rate_fn(self) -> Callable:
        """The ``rate_fn(fields, coords, params)`` of the compiler's ``reaction_kinetics_network`` residual."""
        rhs = self.torch_rhs()

        def rate_fn(fields, coords, params):
            return rhs(coords["t"], fields, {})
        return rate_fn

    def to_problem_spec(self, t_end: float, *, t0: float = 0.0, ic_weight: float = 10.0):
        """A ``reaction_kinetics_network`` :class:`ProblemSpec` on ``t in [t0, t_end]`` with one initial condition
        per state, ready for ``pp.PhysicalProblem.from_pde_spec`` and ``pp.solve(problem, "pinn")``."""
        from pinneapple_physics.pde_environment.conditions import InitialCondition
        from pinneapple_physics.pde_environment.scales import ScaleSpec
        from pinneapple_physics.pde_environment.spec import PDETermSpec, ProblemSpec

        y0 = self.initial_values(t0=t0)
        conds = []
        for s, v in zip(self.states, y0, strict=True):
            conds.append(InitialCondition(
                name=f"ic_{s}", fields=(s,), selector_type="callable",
                selector=lambda X, ctx, t0=t0: np.isclose(X[:, 0], t0),
                value_fn=lambda X, ctx, v=float(v): np.full((X.shape[0], 1), v, dtype=np.float32),
                weight=ic_weight))
        pde = PDETermSpec(kind="reaction_kinetics_network", fields=tuple(self.states), coords=("t",),
                          params=dict(self.parameters), meta={"rate_fn": self.rate_fn(), "source": self.name})
        return ProblemSpec(name=self.name, dim=0, coords=("t",), fields=tuple(self.states), pde=pde,
                           conditions=tuple(conds), sample_defaults={"n_col": 2000, "n_ic": 50},
                           scales=ScaleSpec(L=float(t_end - t0), U=float(max(1.0, np.abs(y0).max()))),
                           domain_bounds={"t": (float(t0), float(t_end))},
                           solver_spec={"name": "scipy", "method": "solve_ivp", "params": {"method": "BDF"}},
                           meta={"ode_system": self})

    # ------------------------------------------------------------------ sensitivities / forward model
    def _sensitivity_system(self, names: Sequence[str]):
        key = ("sens", tuple(names))
        if key not in self._cache:
            X, P = self.state_symbols(), self.parameter_symbols()
            f = sp.Matrix(self.expanded_rhs())
            Jx = _jacobian(f, X)
            pnames = [n for n in names if n in self.parameters]
            Jp = _jacobian(f, [_sym(n) for n in pnames]) if pnames else sp.zeros(len(X), 0)
            fn = sp.lambdify((TIME, X, P), [list(f), Jx.tolist(), Jp.tolist()], "numpy", cse=True)
            self._cache[key] = (fn, pnames)
        return self._cache[key]

    def sensitivities(self, t_eval, names: Sequence[str], parameters: Mapping[str, float] | None = None, *,
                      y0=None, method: str = "BDF", rtol: float = 1e-8, atol: float = 1e-10,
                      max_step: float = math.inf):
        """States and ``d state / d name`` on ``t_eval`` by the forward sensitivity equations
        ``dS/dt = J_x S + J_p``. ``names`` are parameters or ``<state>_0`` (initial values; their values then come
        from ``y0``). Returns ``(Trajectory, S)`` with ``S`` of shape (len(t_eval), n_states, len(names))."""
        t_eval = np.asarray(t_eval, dtype=float)
        names = list(names)
        p = {**self.parameters, **(parameters or {})}
        pv = [p[k] for k in self.parameters]
        fn, pnames = self._sensitivity_system(names)
        n, m = len(self.states), len(names)
        base = self.initial_values(p, float(t_eval[0]))
        y0 = base if y0 is None else np.asarray(y0, dtype=float)
        explicit = [i for i, s in enumerate(self.states) if f"{s}_0" in names]
        S0 = np.zeros((n, m))
        for j, nm in enumerate(names):
            if nm in self.parameters:
                # initial values given by initial assignments depend on the parameters
                dp = 1e-6 * max(1.0, abs(p[nm]))
                dy = (self.initial_values({**p, nm: p[nm] + dp}, float(t_eval[0])) - base) / dp
                dy[explicit] = 0.0
                S0[:, j] = dy
            elif nm.endswith("_0") and nm[:-2] in self.states:
                S0[self.states.index(nm[:-2]), j] = 1.0
            else:
                raise KeyError(f"{nm} is neither a parameter nor <state>_0")
        pcol = [names.index(k) for k in pnames]

        def F(t, z):
            y, S = z[:n], z[n:].reshape(n, m)
            fv, Jx, Jp = fn(t, y, pv)
            dS = np.asarray(Jx, dtype=float).reshape(n, n) @ S
            if pcol:
                dS[:, pcol] += np.asarray(Jp, dtype=float).reshape(n, len(pcol))
            return np.concatenate([np.asarray(fv, dtype=float).reshape(n), dS.ravel()])

        Z, ok, msg, nfev = _integrate(F, t_eval, np.concatenate([y0, S0.ravel()]), breakpoints=self.breakpoints(),
                                      method=method, rtol=rtol, atol=atol, max_step=max_step)
        traj = Trajectory(t_eval, {s: Z[i] for i, s in enumerate(self.states)}, ok, msg, nfev)
        return traj, Z[n:].T.reshape(t_eval.size, n, m)

    def assignment_sensitivities(self, name: str, traj: Trajectory, S: np.ndarray, names: Sequence[str],
                                 parameters: Mapping[str, float] | None = None):
        """Value and ``d name / d names`` of an algebraic variable on a trajectory from :meth:`sensitivities`
        (chain rule ``dg/dx S + dg/dp``)."""
        p = {**self.parameters, **(parameters or {})}
        g = self.expanded(self.assignments[name])
        X, P = self.state_symbols(), self.parameter_symbols()
        gx = _jacobian(sp.Matrix([g]), X)
        gp = _jacobian(sp.Matrix([g]), [_sym(k) for k in names]) if len(names) else sp.zeros(1, 0)
        gp = [gp[0, j] if names[j] in self.parameters else sp.Integer(0) for j in range(len(names))]
        f = sp.lambdify((TIME, X, P), [g, list(gx), gp], "numpy")
        tt = traj.t
        val, dx, dp = f(tt, [traj.values[s] for s in self.states], [p[k] for k in self.parameters])
        b = lambda v: np.broadcast_to(np.asarray(v, float), tt.shape)        # noqa: E731
        dx = np.stack([b(v) for v in dx], axis=1)
        dp = np.stack([b(v) for v in dp], axis=1) if len(names) else np.zeros((tt.size, 0))
        return b(val).copy(), np.einsum("tn,tnm->tm", dx, S) + dp

    def forward_model(self, observables: Sequence[str], parameters: Sequence[str], *, method: str = "BDF",
                      rtol: float = 1e-8, atol: float = 1e-10, max_step: float = math.inf,
                      bounds: Mapping[str, tuple[float, float]] | None = None, log: bool = True):
        """A :class:`ForwardModel` predicting ``observables`` (states or assignments) at input ``t`` from the named
        parameters (and ``<state>_0`` initial values). Observations carry ``inputs={"t": ...}`` and an
        ``observable`` per row. Parameters default to log scale with bounds (value/10, value*10) unless given."""
        from pinneapple_analysis.inverse_problems.ode_model import ODEForwardModel

        return ODEForwardModel(self, observables, parameters, method=method, rtol=rtol, atol=atol,
                               max_step=max_step, bounds=bounds, log=log)

    # ------------------------------------------------------------------ export
    def to_sbml(self, path: str | None = None) -> str:
        """SBML Level 3 Version 2 text with one parameter per state and a rate rule for it, assignment rules for the
        algebraic variables and initial assignments for expression initial values. Written to ``path`` if given."""
        from .sbml import write_sbml

        text = write_sbml(self)
        if path is not None:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(text)
        return text

    def describe(self) -> dict[str, Any]:
        return {"name": self.name, "states": list(self.states), "parameters": dict(self.parameters),
                "assignments": list(self.assignments), "meta": {k: v for k, v in self.meta.items()
                                                                  if isinstance(v, (str, int, float, list))}}

    def __repr__(self) -> str:
        return (f"ODESystem({self.name!r}, {len(self.states)} states, {len(self.parameters)} parameters, "
                f"{len(self.assignments)} assignments)")
