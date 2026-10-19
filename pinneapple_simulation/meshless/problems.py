"""Linear 2-D problems with Dirichlet conditions, defined by expressions in ``x``, ``y`` (and ``t``).

``Problem("poisson", exact="sin(pi*x)*sin(pi*y)")`` derives the source and the boundary values from a manufactured
solution (with sympy), so every method can be scored against the exact solution. Without ``exact``, give ``source``
and ``boundary``; the study then uses a fine reference solution.

==========  =============================================  ========================================
kind        equation                                       data
==========  =============================================  ========================================
poisson     -Δu = f                                        u = g on the boundary
helmholtz   -Δu - k² u = f                                 u = g on the boundary
heat        u_t = α Δu + f  on (0, t_end]                  u = g on the boundary, u(t=0) = u0
==========  =============================================  ========================================
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

__all__ = ["Problem", "KINDS"]

KINDS = ("poisson", "helmholtz", "heat")


def _expr(s: str, time: bool):
    import sympy as sp
    x, y, t = sp.symbols("x y t", real=True)
    names = {"x": x, "y": y, "t": t, "pi": sp.pi, "E": sp.E, "r": sp.sqrt(x ** 2 + y ** 2),
             "theta": sp.atan2(y, x), "abs": sp.Abs}
    for f in ("sin", "cos", "tan", "exp", "log", "sqrt", "sinh", "cosh", "tanh", "atan", "atan2", "Abs"):
        names[f] = getattr(sp, f)
    s = str(s).replace("^", "**")
    if len(s) > 400:
        raise ValueError(f"expression not allowed: {s!r}")
    from pinneapple_data.expressions import ExpressionError, check
    try:                                 # sympify runs Python's eval: only formulas of these names get that far
        check(s, names=set(names) - {"abs", "Abs"}, functions={"abs", "Abs", "sin", "cos", "tan", "exp", "log", "sqrt",
                                                               "sinh", "cosh", "tanh", "atan", "atan2"})
    except ExpressionError as ex:
        if str(ex).startswith("unknown name"):
            raise ValueError(f"unknown symbols in {s!r}; use x, y" + (", t" if time else "")) from None
        raise ValueError(f"expression not allowed: {ex}") from None
    try:
        e = sp.sympify(s, locals=names)
    except (sp.SympifyError, TypeError, SyntaxError) as ex:
        raise ValueError(f"cannot read the expression {s!r}: {ex}") from ex
    free = {str(v) for v in e.free_symbols}
    allowed = {"x", "y", "t"} if time else {"x", "y"}
    if free - allowed:
        raise ValueError(f"unknown symbols {sorted(free - allowed)} in {s!r}; use x, y" + (", t" if time else ""))
    return e, (x, y, t)


def _fn(e, sym) -> Callable[..., np.ndarray]:
    import sympy as sp
    f = sp.lambdify(sym, e, "numpy")

    def call(P: np.ndarray, t: float = 0.0) -> np.ndarray:
        P = np.asarray(P, float)
        v = f(P[:, 0], P[:, 1], t)
        return np.broadcast_to(np.asarray(v, float), (len(P),)).copy()
    return call


@dataclass
class Problem:
    kind: str = "poisson"
    exact: str | None = None           # manufactured solution u(x, y[, t])
    source: str | None = None          # f, when there is no exact solution
    boundary: str | None = None        # g, when there is no exact solution
    initial: str | None = None         # u0 (heat), when there is no exact solution
    k: float = 4.0                        # Helmholtz wave number
    alpha: float = 1.0                    # heat diffusivity
    t_end: float = 0.1                    # heat: final time
    n_steps: int = 100                    # heat: Crank-Nicolson steps

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}")
        if self.exact is None and (self.source is None or self.boundary is None):
            raise ValueError("give a manufactured solution (exact) or both a source and a boundary value")
        if self.kind == "heat" and self.exact is None and self.initial is None:
            raise ValueError("a heat problem without an exact solution needs an initial condition")
        self._compile()

    @property
    def time_dependent(self) -> bool:
        return self.kind == "heat"

    @property
    def has_exact(self) -> bool:
        return self.exact is not None

    def _compile(self):
        import sympy as sp
        td = self.time_dependent
        if self.exact is not None:
            u, (x, y, t) = _expr(self.exact, td)
            lap = sp.diff(u, x, 2) + sp.diff(u, y, 2)
            if self.kind == "poisson":
                f = -lap
            elif self.kind == "helmholtz":
                f = -lap - self.k ** 2 * u
            else:
                f = sp.diff(u, t) - self.alpha * lap
            sym = (x, y, t)
            self._u, self._f, self._g = _fn(u, sym), _fn(sp.simplify(f), sym), _fn(u, sym)
            self._u0 = _fn(u.subs(t, 0), sym)
            self._latex = {"u": sp.latex(u), "f": sp.latex(sp.simplify(f))}
        else:
            f, sym = _expr(self.source, td)
            g, _ = _expr(self.boundary, td)
            self._u = None
            self._f, self._g = _fn(f, sym), _fn(g, sym)
            self._u0 = _fn(_expr(self.initial, False)[0], sym) if td else None
            self._latex = {"f": sp.latex(f), "g": sp.latex(g)}

    # values -----------------------------------------------------------------
    def f(self, P, t: float = 0.0) -> np.ndarray:
        return self._f(P, t)

    def g(self, P, t: float = 0.0) -> np.ndarray:
        return self._g(P, t)

    def u0(self, P) -> np.ndarray:
        return self._u0(P, 0.0)

    def u(self, P, t: float | None = None) -> np.ndarray:
        if self._u is None:
            raise ValueError("this problem has no exact solution")
        return self._u(P, self.t_end if (t is None and self.time_dependent) else (t or 0.0))

    @property
    def c(self) -> float:
        """Coefficient of u in the steady operator -Δu - c u (0 for Poisson, k² for Helmholtz)."""
        return self.k ** 2 if self.kind == "helmholtz" else 0.0

    def describe(self) -> dict:
        eq = {"poisson": "-Δu = f", "helmholtz": f"-Δu - k²u = f (k = {self.k:g})",
              "heat": f"u_t = αΔu + f (α = {self.alpha:g}, t ∈ [0, {self.t_end:g}], {self.n_steps} Crank-Nicolson steps)"}
        return {"kind": self.kind, "equation": eq[self.kind], "exact": self.exact, "source": self.source,
                "boundary": self.boundary, "initial": self.initial, "latex": self._latex, "k": self.k,
                "alpha": self.alpha, "t_end": self.t_end, "n_steps": self.n_steps,
                "reference": "manufactured solution" if self.has_exact else "fine reference solution"}
