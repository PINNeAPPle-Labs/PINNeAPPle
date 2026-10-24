"""Physics queries on a chip package (track 2): the methods behind :class:`pinneapple_query.Engine` for the
package of :mod:`.package`.

Quantities (units: K for temperature *rises* above ambient, W, K/W):

* ``"T"`` at ``"die@u,v"`` (the junction cell under the point) or ``"die@u,v,w"`` (height fraction ``w`` through
  the die; needs the block's interior, rebuilt on demand);
* ``"Tj_max"`` over ``"region:die"`` or ``"region:die[i0:i1,j0:j1]"``;
* ``"q"`` through ``"ubump[I,J]"``, ``"tsv[I,J]"``, ``"c4[I,J]"``, ``"interface:die/tim"`` (optionally a block range)
  or ``"lid:top"``;
* ``"R_ja"``: peak junction temperature rise over total power.

Conditions: ``{"power": (nx, ny) W per active cell}`` (default: the package's power map).

Methods, cheapest first (each offers an expected cost and error bound; the planner of :mod:`pinneapple_query`
picks):

* ``cached`` -- the conditions were solved before: **read** the stored state (no new physics).
* ``bound`` -- *rigorous* interval from monotonicity: the conduction matrix is a nonsingular M-matrix, so
  ``K^-1 >= 0`` entrywise and ``p_min <= p <= p_max`` cell-wise brackets every temperature between the solutions for
  the uniform maps ``p_min`` and ``p_max``. For a uniform package those are one block with adiabatic sides, so the
  bound costs one block solve, ever. A **prediction** with a guaranteed interval (often wide).
* ``superposition`` -- conduction is linear in the power: if the new map is (nearly) a combination of solved maps,
  combine their states; the residual map is bounded by monotonicity as above. A **prediction**, no new solve.
* ``condensed`` -- a new solve with the memoised hierarchy (cost of one upward/downward pass at ``m`` and at
  ``2 m``); the ``2 m`` value is returned with ``|T_m - T_2m|`` as its uncertainty (an estimate, not a bound). A
  new state, stored for later reads.
* ``reference`` -- the fine finite-volume solve (last resort; small packages).
"""
from __future__ import annotations

import time

import numpy as np

from pinneapple_query.core import Answer, Engine, Method, Offer, Planner, Query, condition_key
from pinneapple_query.refs import parse_ref

from . import fv
from .hierarchy import Hierarchy
from .package import BASE, Package
from .substructure import FLUXES, block_model, reference_observables

UNITS = {"T": "K", "Tj_max": "K", "q": "W", "R_ja": "K/W"}
#: targets when a query states none (temperatures to 0.1 K; heat and resistance: the most accurate available)
DEFAULT_ACCURACY = {"T": 0.1, "Tj_max": 0.1}
FEATURE_FLUX = {"ubump": "q_ubump", "tsv": "q_tsv", "c4": "q_c4"}


class State:
    """A solved configuration: per-block observables, its source, its own uncertainty (K), and what is needed to
    read interior points (the hierarchy and its solution, or the full reference field)."""

    def __init__(self, power, obs, method, uncertainty, seconds, solver=None, sol=None, field=None):
        self.power, self.obs, self.method = power, obs, method
        self.uncertainty, self.seconds = uncertainty, seconds
        self.solver, self.sol, self.field = solver, sol, field

    @property
    def has_interior(self) -> bool:
        return self.field is not None or (self.solver is not None and self.sol is not None)


class PackageModel:
    """Shared machinery of the methods: the package, the stored states, the hierarchies, extraction of a quantity
    from a state."""

    def __init__(self, pkg: Package, m: int = 8, reference_cells: int = 2_000_000):
        self.pkg = pkg
        self.m = m
        self.reference_cells = reference_cells
        self.states: dict[str, State] = {}
        self._h: dict[int, Hierarchy] = {}
        self.timings: dict[str, float] = {}
        self._unit = None
        self._keys: dict[int, tuple] = {}

    # -- conditions ------------------------------------------------------------------------------------------------
    def power(self, q: Query) -> np.ndarray:
        p = q.conditions.get("power")
        return self.pkg.power if p is None else np.asarray(p, float)

    def key(self, q: Query) -> str:
        """Hash of the conditions (memoised by array identity: hashing a large map on every query would dominate
        the latency of a read)."""
        p = self.power(q)
        hit = self._keys.get(id(p))
        if hit is not None and hit[0] is p:
            return hit[1]
        k = condition_key({"power": p})
        self._keys[id(p)] = (p, k)
        return k

    def store(self, state: State) -> str:
        k = condition_key({"power": state.power})
        self._keys[id(state.power)] = (state.power, k)
        self.states[k] = state
        return k

    def hierarchy(self, m: int) -> Hierarchy:
        if m not in self._h:
            t0 = time.perf_counter()
            self._h[m] = Hierarchy(self.pkg, m).prepare()
            self.timings[f"build_m{m}"] = time.perf_counter() - t0
        return self._h[m]

    # -- extraction -----------------------------------------------------------------------------------------------
    def extract(self, q: Query, state: State) -> float:
        obs, power = state.obs, state.power
        st = self.pkg.stack
        n = st.n
        nbx, nby = self.pkg.shape
        ref = q.where
        if q.quantity == "T":
            if ref.component != "die" or ref.local is None:
                raise ValueError("T needs a point on the die: die@u,v[,w]")
            u, v = ref.local[:2]
            ix = min(int(u * nbx * n), nbx * n - 1)
            iy = min(int(v * nby * n), nby * n - 1)
            I, a, J, b = ix // n, ix % n, iy // n, iy % n
            if len(ref.local) == 2:
                return float(obs["Tj"][I, J, a * n + b])
            die = st.layer_slices()["die"]
            k = die.start + min(int(ref.local[2] * (die.stop - die.start)), die.stop - die.start - 1)
            if state.field is not None:
                return float(state.field[ix, iy, k])
            if state.solver is None or state.sol is None:
                raise LookupError("an interior point needs the solved hierarchy or the field of this state")
            return float(state.solver.recover(state.sol, I, J)[a, b, k])
        if q.quantity in ("Tj_max", "R_ja"):
            tj = obs["Tj"]
            if ref.span is not None:
                (i0, i1), (j0, j1) = ref.span
                tj = tj[i0:i1, j0:j1]
            val = float(tj.max())
            if q.quantity == "R_ja":
                val /= float((self.pkg.power if power is None else power).sum())
            return val
        if q.quantity == "q":
            if ref.kind == "feature":
                return float(obs[FEATURE_FLUX[ref.component]][ref.index])
            if ref.kind == "interface" and ref.component == "die/tim":
                a = obs["q_tim"]
                if ref.span is not None:
                    (i0, i1), (j0, j1) = ref.span
                    a = a[i0:i1, j0:j1]
                return float(a.sum())
            if ref.component == "lid" and ref.kind == "face:top":
                return float(obs["q_top"].sum())
        raise ValueError(f"cannot answer {q.quantity} at {ref}")

    # -- the uniform-power bound ----------------------------------------------------------------------------------
    def unit_response(self) -> dict:
        """Observables of one block, sides adiabatic, 1 W on every active cell: by symmetry the whole package's
        response to a uniform map (identical, mirror-symmetric blocks; adiabatic package sides)."""
        if self._unit is None:
            if len(self.pkg.variants) != 1 or self.pkg.variants[0] != BASE:
                raise ValueError("the uniform-power bound needs identical blocks")
            bm = block_model(self.pkg.stack, BASE)
            n2 = self.pkg.stack.n ** 2
            # ports free: S t = W 1  ->  t = S^-1 W 1
            p = np.ones(n2)
            t = np.linalg.solve(bm.S, bm.W @ p)
            vals = bm.O_t @ t + bm.O_p @ p
            cells = bm.recover(t, p)
            self._unit = {"Tj": vals[:n2], "fluxes": dict(zip(FLUXES, vals[n2:])), "cells": cells}
        return self._unit


def _answer(model, q, value, unc, kind, method, t0, validity, provenance, rigorous=False, interval=None):
    return Answer(value=value, unit=UNITS.get(q.quantity, ""), uncertainty=unc, kind=kind, method=method,
                  validity=validity, provenance=provenance, seconds=time.perf_counter() - t0, rigorous=rigorous,
                  interval=interval)


class Cached(Method):
    name = "cached"

    def __init__(self, model: PackageModel):
        self.model = model

    def offer(self, q):
        s = self.model.states.get(self.model.key(q))
        if s is None:
            return None
        if q.quantity == "T" and q.where.local is not None and len(q.where.local) == 3 and not s.has_interior:
            return None
        return Offer(self.name, 1e-5 if not (q.where.local and len(q.where.local) == 3) else 5e-3,
                     s.uncertainty, why=f"state {self.model.key(q)} solved by {s.method}")

    def answer(self, q):
        t0 = time.perf_counter()
        s = self.model.states[self.model.key(q)]
        v = self.model.extract(q, s)
        return _answer(self.model, q, v, s.uncertainty, "read", self.name, t0,
                       {"conditions": "exactly the solved ones"},
                       {"state": self.model.key(q), "solved_by": s.method, "solve_s": s.seconds})


class Bound(Method):
    """Rigorous interval from the M-matrix monotonicity (see the module docstring)."""
    name = "bound"

    def __init__(self, model: PackageModel):
        self.model = model

    def _interval(self, q):
        p = self.model.power(q)
        u = self.model.unit_response()
        lo, hi = float(p.min()), float(p.max())
        if q.quantity in ("T", "Tj_max", "R_ja"):
            if q.quantity == "T" and q.where.local is not None and len(q.where.local) == 3:
                st = self.model.pkg.stack
                die = st.layer_slices()["die"]
                k = die.start + min(int(q.where.local[2] * (die.stop - die.start)), die.stop - die.start - 1)
                n = st.n
                ix = min(int(q.where.local[0] * self.model.pkg.shape[0] * n), self.model.pkg.shape[0] * n - 1) % n
                iy = min(int(q.where.local[1] * self.model.pkg.shape[1] * n), self.model.pkg.shape[1] * n - 1) % n
                base = float(u["cells"][ix, iy, k])
            elif q.quantity == "T":
                n = self.model.pkg.stack.n
                ix = min(int(q.where.local[0] * self.model.pkg.shape[0] * n), self.model.pkg.shape[0] * n - 1) % n
                iy = min(int(q.where.local[1] * self.model.pkg.shape[1] * n), self.model.pkg.shape[1] * n - 1) % n
                base = float(u["Tj"][ix * n + iy])
            else:
                base_hi, base_lo = float(u["Tj"].max()), float(u["Tj"].max())
                a, b = lo * base_lo, hi * base_hi
                if q.quantity == "R_ja":
                    a, b = a / p.sum(), b / p.sum()
                return a, b
            return lo * base, hi * base
        return None

    def offer(self, q):
        try:
            iv = self._interval(q)
        except ValueError:
            return None
        if iv is None:
            return None
        return Offer(self.name, 1e-4, 0.5 * (iv[1] - iv[0]), rigorous=True, why="M-matrix monotonicity")

    def answer(self, q):
        t0 = time.perf_counter()
        a, b = self._interval(q)
        return _answer(self.model, q, 0.5 * (a + b), 0.5 * (b - a), "predicted", self.name, t0,
                       {"requires": "non-negative power on the active row, identical blocks, linear conduction"},
                       {"from": "one block, adiabatic sides, uniform unit power"}, rigorous=True, interval=(a, b))


class Superposition(Method):
    """Linear combination of solved states (least squares on the power maps); the residual map bounded by
    monotonicity, plus the combined uncertainty of the states."""
    name = "superposition"

    def __init__(self, model: PackageModel, max_states: int = 64):
        self.model = model
        self.max_states = max_states
        self._fit = {}

    def _coef(self, q):
        key = self.model.key(q)
        if key in self._fit:
            return self._fit[key]
        states = [s for s in self.model.states.values()][: self.max_states]
        if len(states) < 1:
            return None
        P = np.stack([s.power.ravel() for s in states], 1)
        p = self.model.power(q).ravel()
        c, *_ = np.linalg.lstsq(P, p, rcond=None)
        r = p - P @ c
        u = self.model.unit_response()
        # |T(r)| <= max|r| * T(uniform unit map), cell-wise (K^-1 >= 0)
        res_T = float(np.abs(r).max() * u["Tj"].max())
        res_q = float(np.abs(r).sum())          # net heat of the residual map through a complete cut
        su = float(sum(abs(ci) * s.uncertainty for ci, s in zip(c, states)))
        exact = all(s.uncertainty == 0.0 for s in states)
        self._fit[key] = (states, c, res_T + su, res_q, exact, float(np.abs(r).max() / max(np.abs(p).max(), 1e-300)))
        return self._fit[key]

    def offer(self, q):
        if q.quantity == "T" and q.where.local is not None and len(q.where.local) == 3:
            return None
        try:
            f = self._coef(q)
        except ValueError:
            return None
        if f is None:
            return None
        states, c, unc_T, res_q, exact, rel = f
        flux = q.quantity == "q"
        unc = res_q if flux else unc_T
        return Offer(self.name, 2e-3 * len(states), unc, rigorous=exact and self._rigorous(q),
                     why=f"{len(states)} states, residual map {rel:.1e} relative")

    @staticmethod
    def _rigorous(q):
        """Monotonicity bounds temperatures; for heat, only the net flow through a complete cut (the whole die/TIM
        interface or the lid top) is bounded by the residual heat."""
        if q.quantity != "q":
            return True
        return q.where.span is None and (q.where.kind == "interface" or q.where.kind == "face:top")

    def answer(self, q):
        t0 = time.perf_counter()
        states, c, unc_T, res_q, exact, rel = self._coef(q)
        unc = res_q if q.quantity == "q" else unc_T
        if q.quantity in ("Tj_max", "R_ja"):
            obs = {"Tj": sum(ci * s.obs["Tj"] for ci, s in zip(c, states))}
        else:
            obs = {k: sum(ci * s.obs[k] for ci, s in zip(c, states)) for k in states[0].obs}
        v = self.model.extract(q, State(self.model.power(q), obs, self.name, unc, 0.0))
        if q.quantity == "R_ja":
            unc /= float(self.model.power(q).sum())
        return _answer(self.model, q, v, unc, "predicted", self.name, t0,
                       {"requires": "linear conduction, same geometry, materials and cooling as the states"},
                       {"states": [condition_key({"power": s.power}) for s in states],
                        "coefficients": [float(x) for x in c], "residual_map_relative": rel},
                       rigorous=exact and self._rigorous(q))


class Condensed(Method):
    """A new solve with the memoised hierarchy at ``m``; uncertainty |T_m - T_2m| (the 2m value is returned)."""
    name = "condensed"

    def __init__(self, model: PackageModel, expected_s: float = 1.0):
        self.model = model
        self.expected_s = expected_s

    def offer(self, q):
        rel = getattr(self, "_rel_unc", None)
        if rel is None:
            return Offer(self.name, self.expected_s, np.inf, why="not calibrated yet (no solve so far)")
        scale = float(self.model.power(q).max())
        bound = rel * scale if q.quantity != "q" else np.inf
        return Offer(self.name, self.expected_s, bound,
                     why="hierarchical condensation; bound scaled from the last |T_m - T_2m| per unit peak power")

    def solve(self, q) -> State:
        t0 = time.perf_counter()
        p = self.model.power(q)
        h1, h2 = self.model.hierarchy(self.model.m), self.model.hierarchy(2 * self.model.m)
        s1, s2 = h1.solve(p), h2.solve(p)
        o1, o2 = h1.observables(s1), h2.observables(s2)
        unc = float(np.abs(o1["Tj"] - o2["Tj"]).max())
        self._rel_unc = unc / max(float(p.max()), 1e-300)
        st = State(p, o2, self.name, unc, time.perf_counter() - t0, solver=h2, sol=s2)
        self.model.store(st)
        self.expected_s = st.seconds
        return st

    def answer(self, q):
        t0 = time.perf_counter()
        st = self.solve(q)
        v = self.model.extract(q, st)
        return _answer(self.model, q, v, st.uncertainty, "solved", self.name, t0,
                       {"model": f"reduced interfaces m={2 * self.model.m}", "relative_to": "the fine FV model"},
                       {"state": self.model.key(q), "m": [self.model.m, 2 * self.model.m]})


class Reference(Method):
    """The fine finite-volume solve: the last resort."""
    name = "reference"

    def __init__(self, model: PackageModel):
        self.model = model

    def offer(self, q):
        ok = self.model.pkg.n_cells <= self.model.reference_cells
        return Offer(self.name, 1e-5 * self.model.pkg.n_cells, 0.0, valid=ok,
                     why="fine FV" if ok else "too many cells for the reference in this container")

    def answer(self, q):
        t0 = time.perf_counter()
        p = self.model.power(q)
        T, info = fv.solve_package(self.model.pkg, p)
        st = State(p, reference_observables(self.model.pkg, T), self.name, 0.0, info.seconds, field=T)
        self.model.store(st)
        v = self.model.extract(q, st)
        return _answer(self.model, q, v, 0.0, "solved", self.name, t0, {"model": "fine FV (the reference)"},
                       {"solver": info.method, "iterations": info.iterations})


def package_engine(pkg: Package, m: int = 12, reference_cells: int = 2_000_000,
                   calibrate: bool = True) -> tuple[Engine, PackageModel]:
    """A query engine over a package with the methods of this module. ``calibrate``: solve the package's own power
    map once with the condensed method (part of the build cost), which gives the condensed method its uncertainty
    prior and the store its first state."""
    model = PackageModel(pkg, m, reference_cells)
    methods = [Cached(model), Bound(model), Superposition(model), Condensed(model), Reference(model)]
    if calibrate:
        t0 = time.perf_counter()
        methods[3].solve(Query("Tj_max", parse_ref("region:die"), conditions={"power": pkg.power}))
        model.timings["calibrate"] = time.perf_counter() - t0
    return Engine(Planner(methods), UNITS, dict(DEFAULT_ACCURACY)), model
