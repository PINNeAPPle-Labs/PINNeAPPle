"""Maxwell relations and consistency checks of data or models against a potential family.

A potential gives equal mixed partials, ``d2Phi/dx_i dx_j = d2Phi/dx_j dx_i``. Written with the named first
derivatives these are the Maxwell relations (``(dS/dV)_T = (dp/dT)_V`` from the Helmholtz energy), the reciprocity
of an electromechanical converter (``dpsi/dtheta = dT/di`` from the co-energy) or the symmetry of a mass matrix.
Response functions add identities between second derivatives (``c_p - c_v = T V alpha^2 / kappa_T``, the speed of
sound from ``c_v`` and the pressure derivatives).

:func:`consistency_check` evaluates such identities on anything that predicts the quantities separately: a
surrogate with one output per property, a table, or measurements on a grid. Each identity is reported with its
relative violation; a potential satisfies them all to rounding, a model fitted property by property in general does
not. Satisfying the identities is necessary for the quantities to come from one potential (and, on a simply
connected domain, sufficient for the first-derivative ones).

    from pinneapple_physics.potential import consistency_check
    rep = consistency_check(model, "helmholtz_T_rho", at={"T": T, "rho": rho})
    print(rep)                    # one line per identity: median, 95th percentile and largest relative violation
"""
from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .core import Potential, tree

__all__ = ["Identity", "MaxwellRelation", "Derivatives", "ConsistencyReport", "maxwell_relations",
           "consistency_check", "FAMILIES", "identities"]

#: identity sets of known potential families: name -> "module:function" returning a list of :class:`Identity`
FAMILIES = {
    "helmholtz_T_rho": "pinneapple_physics.potential.thermo:identities_T_rho",
    "helmholtz_T_v": "pinneapple_physics.potential.thermo:identities_T_v",
}


@dataclass
class Identity:
    """An identity ``lhs(D) = rhs(D)`` between quantities and their derivatives.

    ``D(q, *vars)`` is the derivative of quantity ``q`` with respect to the listed natural variables (the others
    held constant); ``D.x(var)`` is the value of a variable. ``needs`` lists the quantities it reads. ``scale(D)``,
    when given, is the natural magnitude the violation is measured against (for identities whose two sides can both
    vanish, like ``(dc_v/dv)_T = T (d2p/dT2)_v`` for a van der Waals fluid); otherwise the typical size of the two
    sides is used."""

    name: str
    text: str
    lhs: Callable[[Derivatives], np.ndarray]
    rhs: Callable[[Derivatives], np.ndarray]
    needs: tuple[str, ...]
    scale: Callable[[Derivatives], np.ndarray] | None = None


@dataclass
class MaxwellRelation(Identity):
    """Equality of the mixed partials of a potential in ``pair``, written with its named first derivatives."""

    pair: tuple[str, str] = ("", "")


class Derivatives:
    """Derivatives of predicted quantities, by central finite differences of a callable or ``np.gradient`` on a
    grid. Results are cached per (quantity, variables)."""

    def __init__(self, values: Callable[[dict], Mapping[str, np.ndarray]] | None, point: Mapping[str, np.ndarray],
                 variables: Sequence[str], step: float = 1e-4, grid: Mapping | None = None):
        self.point = {k: np.asarray(v, float) for k, v in point.items()}
        self.variables = list(variables)
        self.values, self.step, self.grid = values, step, grid
        self._evals: dict[tuple, Mapping[str, np.ndarray]] = {}
        self._cache: dict[tuple, np.ndarray] = {}
        self.h = {}
        for k in self.variables:
            v = self.point[k]
            span = float(np.ptp(v)) if v.size > 1 else 0.0
            self.h[k] = step * np.maximum(np.abs(v), max(span, 1e-12))

    def x(self, var: str) -> np.ndarray:
        return self.point[var]

    def _at(self, shift: tuple[tuple[str, int], ...]) -> Mapping[str, np.ndarray]:
        if shift not in self._evals:
            p = dict(self.point)
            for k, m in shift:
                p[k] = p[k] + m * self.h[k]
            self._evals[shift] = {q: np.asarray(v, float) for q, v in self.values(p).items()}
        return self._evals[shift]

    def __call__(self, q: str, *vars: str) -> np.ndarray:
        key = (q,) + tuple(sorted(vars))
        if key in self._cache:
            return self._cache[key]
        if self.grid is not None:
            out = self._grid_derivative(q, vars)
        else:
            out = self._fd(q, tuple(sorted(vars)))
        self._cache[key] = out
        return out

    def _fd(self, q: str, vars: tuple[str, ...]) -> np.ndarray:
        if not vars:
            return self._at(())[q]
        if len(vars) == 1:
            (a,) = vars
            return (self._at(((a, 1),))[q] - self._at(((a, -1),))[q]) / (2 * self.h[a])
        if len(vars) == 2 and vars[0] == vars[1]:
            a = vars[0]
            return (self._at(((a, 1),))[q] - 2 * self._at(())[q] + self._at(((a, -1),))[q]) / self.h[a] ** 2
        if len(vars) == 2:
            a, b = vars
            f = lambda i, j: self._at(((a, i), (b, j)))[q]  # noqa: E731
            return (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * self.h[a] * self.h[b])
        raise NotImplementedError("finite differences up to second order")

    def _grid_derivative(self, q: str, vars: Sequence[str]) -> np.ndarray:
        axes = self.grid["axes"]
        names = list(axes)
        v = np.asarray(self.grid["values"][q], float)
        for a in vars:
            v = np.gradient(v, axes[a], axis=names.index(a), edge_order=2)
        return v


@dataclass
class ConsistencyReport:
    """Violation of each identity: relative to the typical magnitude of its two sides."""

    rows: list[dict[str, Any]]
    tol: float
    skipped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(r["ok"] for r in self.rows)

    def worst(self) -> dict[str, Any] | None:
        return max(self.rows, key=lambda r: r["rel_p95"]) if self.rows else None

    def to_dict(self) -> dict[str, Any]:
        return {"tol": self.tol, "ok": self.ok, "skipped": self.skipped,
                "rows": [{k: v for k, v in r.items() if k != "rel"} for r in self.rows]}

    def __str__(self) -> str:
        lines = [f"consistency against {len(self.rows)} identities (tolerance {self.tol:g} on the 95th percentile)"]
        for r in self.rows:
            lines.append(f"  {'PASS' if r['ok'] else 'FAIL'}  {r['name']}: {r['text']}  median {r['rel_median']:.2e}"
                         f", p95 {r['rel_p95']:.2e}, max {r['rel_max']:.2e}")
        for s in self.skipped:
            lines.append(f"  N/A   {s}")
        return "\n".join(lines)


def _partial(num: str, den: str, held: Sequence[str]) -> str:
    return f"(d{num}/d{den})" + (f"_{','.join(held)}" if held else "")


def maxwell_relations(potential: Potential) -> list[MaxwellRelation]:
    """The mixed-partial identities of ``potential`` for every pair of scalar variables, in terms of its named
    first derivatives (raw derivatives where unnamed), with their engineering names where the potential gives them.

    Each relation is an :class:`Identity` usable by :func:`consistency_check` on a model that predicts the first
    derivatives separately."""
    vs = [k for k, v in potential.variables.items() if not v.shape]
    out = []
    for i, xi in enumerate(vs):
        for xj in vs[i + 1:]:
            held = [v for v in potential.variables if v not in (xi, xj)]
            ni, nj = potential.names.get((xi,)), potential.names.get(potential.key((xj,)))
            yi, si = (ni.symbol, ni.sign) if ni else (potential.symbol_of((xi,)), 1.0)
            yj, sj = (nj.symbol, nj.sign) if nj else (potential.symbol_of((xj,)), 1.0)
            li = _partial(yi, xj, [xi] + held)
            rj = _partial(yj, xi, [xj] + held)
            if si == sj:
                text = f"{li} = {rj}"
            else:
                text = f"{li} = -{rj}"
            name = potential.relation_names.get(frozenset((xi, xj)),
                                                f"mixed partials of {potential.name} in ({xi}, {xj})")
            out.append(MaxwellRelation(
                name, text,
                lhs=(lambda D, yi=yi, si=si, xj=xj: D(yi, xj) / si),
                rhs=(lambda D, yj=yj, sj=sj, xi=xi: D(yj, xi) / sj),
                needs=(yi, yj), pair=(xi, xj)))
    return out


def identities(family: str) -> list[Identity]:
    """The identity set of a named potential family (see :data:`FAMILIES`)."""
    if family not in FAMILIES:
        raise KeyError(f"unknown family {family!r}; known: {sorted(FAMILIES)}")
    mod, fn = FAMILIES[family].split(":")
    return getattr(importlib.import_module(mod), fn)()


def _potential_values(p: Potential, depth: int = 2):
    def values(point):
        t = tree(p, depth, at={k: point[k] for k in p.variables}, symbolic=False)
        return t.quantities
    return values


def consistency_check(model, family, at: Mapping[str, Any] | None = None, *, variables: Sequence[str] | None = None,
                      tol: float = 1e-3, step: float = 1e-4) -> ConsistencyReport:
    """How far ``model`` is from being generated by one potential of ``family``.

    Parameters
    ----------
    model:
        ``model(point) -> {quantity: array}`` (``point`` maps variable names to arrays); a :class:`Potential` (its
        named quantities); or grid data ``{"axes": {var: 1-D array}, "values": {quantity: grid array}}`` (then
        ``at`` is not needed and derivatives come from ``np.gradient``, second order).
    family:
        a family name (:data:`FAMILIES`), a :class:`Potential` (its :func:`maxwell_relations`), or a list of
        :class:`Identity`.
    at:
        evaluation points ``{var: array}`` for a callable or potential model.
    tol:
        pass when the 95th percentile of the relative violation is below ``tol``.
    step:
        relative finite-difference step for callables.
    """
    if isinstance(family, str):
        ids = identities(family)
    elif isinstance(family, Potential):
        ids = maxwell_relations(family)
    else:
        ids = list(family)
    grid = None
    if isinstance(model, Mapping):
        grid = model
        mesh = np.meshgrid(*grid["axes"].values(), indexing="ij")
        at = {k: m for k, m in zip(grid["axes"], mesh, strict=False)}
        values, available = None, set(grid["values"])
        variables = list(grid["axes"])
    else:
        if at is None:
            raise ValueError("give the evaluation points: at={var: array}")
        if isinstance(model, Potential):
            values = _potential_values(model)
            variables = variables or list(model.variables)
        else:
            values = model
        probe = values({k: np.asarray(v, float) for k, v in at.items()})
        available = set(probe)
        variables = variables or list(at)
    D = Derivatives(values, at, variables, step=step, grid=grid)
    rows, skipped = [], []
    for ident in ids:
        miss = [q for q in ident.needs if q not in available]
        if miss:
            skipped.append(f"{ident.name}: needs {', '.join(miss)}")
            continue
        lhs, rhs = np.asarray(ident.lhs(D), float), np.asarray(ident.rhs(D), float)
        lhs, rhs = np.broadcast_arrays(lhs, rhs)
        nat = (np.broadcast_to(np.abs(np.asarray(ident.scale(D), float)), lhs.shape)
               if ident.scale is not None else None)
        if grid is not None:  # one-sided differences at the edges are less accurate: score the interior
            sl = tuple(slice(2, -2) if n > 4 else slice(None) for n in lhs.shape)
            lhs, rhs = lhs[sl], rhs[sl]
            nat = nat[sl] if nat is not None else None
        lhs, rhs = lhs.reshape(-1), rhs.reshape(-1)
        ok_pts = np.isfinite(lhs) & np.isfinite(rhs)
        lhs, rhs = lhs[ok_pts], rhs[ok_pts]
        if not lhs.size:
            scale = float("nan")
        elif nat is not None:
            scale = float(np.median(nat.reshape(-1)[ok_pts]))
        else:
            scale = float(np.median(0.5 * (np.abs(lhs) + np.abs(rhs))))
        rel = np.abs(lhs - rhs) / max(scale, 1e-300)
        row = {"name": ident.name, "text": ident.text, "n": int(rel.size),
               "rel_median": float(np.median(rel)) if rel.size else float("nan"),
               "rel_p95": float(np.percentile(rel, 95)) if rel.size else float("nan"),
               "rel_max": float(rel.max()) if rel.size else float("nan"),
               "scale": scale, "rel": rel}
        row["ok"] = bool(row["rel_p95"] <= tol)
        rows.append(row)
    return ConsistencyReport(rows, tol, skipped)
