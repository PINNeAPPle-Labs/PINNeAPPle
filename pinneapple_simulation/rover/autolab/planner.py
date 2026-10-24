"""Risk-aware route planning on a site from what the rover believes about its soils.

For every soil unit the planner draws soils from the rover's belief (:class:`~.estimation.SoilBelief`), computes for
each the quasi-static operating point on every grade (:func:`~.wheel_models.grade_table`) and so knows, per unit and
grade, the expected drive energy per metre and the *risk*: the share of belief samples in which the rover cannot
climb that grade at a slip at or below the mission's slip limit. Routes are shortest paths (Dijkstra, 8-connected
grid) with edge cost ``length * (energy per metre + risk_weight * risk)`` (``objective="distance"``: length times
``1 + risk_weight / 200 * risk``); edges whose risk exceeds ``risk_max`` are removed. The grade of an edge is the
height difference along it. Risk: the soil is sampled at both ends, the midpoint and ``half_width`` (half the track
plus a margin) to either side, the worst counting. Slip and energy: the mean of the left and right wheels
(``half_track`` to either side). Cross-slope is not part of the model.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .wheel_models import GRADES, grade_table


@dataclass
class UnitTables:
    grades: np.ndarray
    slip: np.ndarray            # (n_samples, n_grades), nan = cannot climb
    energy: np.ndarray          # (n_samples, n_grades) [J/m]
    mean_slip: np.ndarray       # operating slip of the belief mean
    mean_energy: np.ndarray


@dataclass
class Route:
    waypoints: np.ndarray       # (k, 2) simplified polyline the rover follows
    cells: np.ndarray           # (m, 2) grid path
    length_m: float
    energy_J: float             # predicted drive energy (belief mean)
    max_risk: float             # largest edge risk along the path
    mean_risk: float            # length-weighted mean risk
    max_slip: float             # largest predicted slip (belief mean)
    profile: dict = field(default_factory=dict)   # per path point: s, grade, unit, slip, risk, energy_per_m
    belief_label: str = ""


class RiskPlanner:
    """Plans on ``site`` with ``belief``; ``mass`` [kg], ``wheel``, ``n_wheels`` and ``g`` describe the rover."""

    def __init__(self, site, belief, wheel, mass: float, slip_limit: float = 0.4, g: float = 1.62,
                 n_wheels: int = 4, cell: float = 0.5, n_samples: int = 24, risk_max: float = 0.05,
                 risk_weight: float = 4000.0, grades=GRADES, seed: int = 0, objective: str = "energy",
                 half_width: float = 0.9, half_track: float = 0.75):
        if objective not in ("energy", "distance"):
            raise ValueError("objective must be 'energy' or 'distance'")
        self.objective = objective
        self.half_width = float(half_width)          # half the track plus a margin [m]: wheels on both sides count
        self.half_track = float(half_track)          # half the track [m]: where the wheels predicting slip and energy run
        self.site, self.belief, self.wheel = site, belief, wheel
        self.mass, self.g, self.n_wheels = float(mass), float(g), int(n_wheels)
        self.slip_limit, self.cell = float(slip_limit), float(cell)
        self.n_samples, self.risk_max, self.risk_weight = int(n_samples), float(risk_max), float(risk_weight)
        self.grades, self.seed = np.asarray(grades, float), seed
        self.tables = [self._unit_tables(k) for k in range(len(site.units))]

    def _unit_tables(self, unit: int) -> UnitTables:
        soils = self.belief.samples(unit, self.n_samples, seed=self.seed)
        tabs = [grade_table(s, self.wheel, self.mass, self.g, self.n_wheels, self.grades) for s in soils]
        slip = np.array([t.slip for t in tabs])
        energy = np.array([t.energy_per_m for t in tabs])
        return UnitTables(self.grades, slip, energy, slip[0], energy[0])

    def edge_metrics(self, unit, grade_deg):
        """Risk, mean-belief energy per metre and slip for arrays of units and grades."""
        unit, grade = np.broadcast_arrays(np.asarray(unit, int), np.asarray(grade_deg, float))
        risk = np.zeros(unit.shape)
        energy = np.zeros(unit.shape)
        slip = np.zeros(unit.shape)
        g = np.clip(grade, self.grades[0], self.grades[-1])
        for k, T in enumerate(self.tables):
            m = unit == k
            if not m.any():
                continue
            gi = g[m]
            fail = np.zeros((T.slip.shape[0], gi.size))
            for j in range(T.slip.shape[0]):
                s = np.interp(gi, self.grades, np.nan_to_num(T.slip[j], nan=9.0))
                fail[j] = s > self.slip_limit
            risk[m] = fail.mean(0)
            energy[m] = np.interp(gi, self.grades, np.nan_to_num(T.mean_energy, nan=np.nanmax(T.mean_energy) * 3))
            slip[m] = np.interp(gi, self.grades, np.nan_to_num(T.mean_slip, nan=1.0))
        return risk, energy, slip

    # ------------------------------------------------------------------ graph
    def _grid(self):
        x0, x1 = self.site.x_range
        y0, y1 = self.site.y_range
        xs = np.arange(x0, x1 + 1e-9, self.cell)
        ys = np.arange(y0, y1 + 1e-9, self.cell)
        X, Y = np.meshgrid(xs, ys)
        return xs, ys, X, Y

    def cost_graph(self):
        from scipy.sparse import coo_matrix
        xs, ys, X, Y = self._grid()
        ny, nx = X.shape
        H = self.site.height(X, Y)
        idx = np.arange(nx * ny).reshape(ny, nx)
        rows, cols, w = [], [], []
        for di, dj in ((0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)):
            a = idx[max(0, -di):ny - max(0, di), max(0, -dj):nx - max(0, dj)]
            b = idx[max(0, di):ny - max(0, -di) or None, max(0, dj):nx - max(0, -dj) or None]
            b = b[:a.shape[0], :a.shape[1]]
            L = self.cell * math.hypot(di, dj)
            ha, hb = H.ravel()[a.ravel()], H.ravel()[b.ravel()]
            grade = np.degrees(np.arctan((hb - ha) / L))
            # the soil under both ends, the midpoint and the wheels on either side (half-track offsets): the worst counts
            nrm = np.array([-di, dj], float) / math.hypot(di, dj)            # (dx, dy) normal of the edge
            risk, energy = np.zeros(a.size), np.zeros(a.size)
            ax_, ay_, bx_, by_ = X.ravel()[a.ravel()], Y.ravel()[a.ravel()], X.ravel()[b.ravel()], Y.ravel()[b.ravel()]
            for f, (px, py) in enumerate(((ax_, ay_), (bx_, by_), (0.5 * (ax_ + bx_), 0.5 * (ay_ + by_)))):
                for off in (0.0, -self.half_width, self.half_width):
                    r, _, _ = self.edge_metrics(self.site.unit_at(px + off * nrm[0], py + off * nrm[1]), grade)
                    risk = np.maximum(risk, r)
                if f < 2:                                        # energy: left and right wheels at both ends
                    for off in (-self.half_track, self.half_track):
                        _, e, _ = self.edge_metrics(self.site.unit_at(px + off * nrm[0], py + off * nrm[1]), grade)
                        energy += 0.25 * e
            ok = risk <= self.risk_max
            rows.append(a.ravel()[ok])
            cols.append(b.ravel()[ok])
            base = energy[ok] if self.objective == "energy" else np.full(int(ok.sum()), 1.0)
            scale = 1.0 if self.objective == "energy" else 1.0 / 200.0      # risk weight in metres for "distance"
            w.append(L * (base + scale * self.risk_weight * risk[ok]) + 1e-6)
        G = coo_matrix((np.concatenate(w), (np.concatenate(rows), np.concatenate(cols))), shape=(nx * ny, nx * ny))
        return G.tocsr(), (xs, ys, X, Y)

    def plan(self, start, goal, label: str = "") -> Route:
        from scipy.sparse.csgraph import dijkstra
        G, (xs, ys, X, Y) = self.cost_graph()
        nx = len(xs)

        def node(p):
            i = int(np.clip(round((p[0] - xs[0]) / self.cell), 0, nx - 1))
            j = int(np.clip(round((p[1] - ys[0]) / self.cell), 0, len(ys) - 1))
            return j * nx + i
        s, t = node(start), node(goal)
        dist, pred = dijkstra(G, indices=s, return_predecessors=True)
        if not np.isfinite(dist[t]):
            raise RuntimeError(f"no route with risk <= {self.risk_max} from {start} to {goal}")
        path = [t]
        while path[-1] != s:
            path.append(pred[path[-1]])
        path = path[::-1]
        cells = np.array([[xs[k % nx], ys[k // nx]] for k in path])
        cells = np.vstack([np.asarray(start, float)[None], cells, np.asarray(goal, float)[None]])
        ok = lambda a, b: self.evaluate(np.array([a, b]), step=0.1).max_risk <= self.risk_max  # noqa: E731
        wps = simplify(cells, tol=0.35, accept=ok)
        return self.evaluate(wps, cells=cells, label=label)

    def evaluate(self, waypoints, cells=None, label: str = "", step: float = 0.1) -> Route:
        """Predicted slip, risk and energy along a polyline (sampled every ``step`` metres)."""
        P = densify(np.asarray(waypoints, float), step)
        d = np.diff(P, axis=0)
        L = np.linalg.norm(d, axis=1)
        mid = 0.5 * (P[1:] + P[:-1])
        grade = np.degrees(np.arctan((self.site.height(P[1:, 0], P[1:, 1]) - self.site.height(P[:-1, 0], P[:-1, 1]))
                                     / np.maximum(L, 1e-9)))
        unit = self.site.unit_at(mid[:, 0], mid[:, 1])
        risk, _, _ = self.edge_metrics(unit, grade)
        nrm = np.column_stack([-d[:, 1], d[:, 0]]) / np.maximum(L, 1e-9)[:, None]
        for off in (-self.half_width, self.half_width):            # risk: the worst of centre and either side
            Q = mid + off * nrm
            r, _, _ = self.edge_metrics(self.site.unit_at(Q[:, 0], Q[:, 1]), grade)
            risk = np.maximum(risk, r)
        epm, slip = np.zeros(len(mid)), np.zeros(len(mid))
        for off in (-self.half_track, self.half_track):            # slip and energy: mean of the left and right wheels
            Q = mid + off * nrm
            _, e, sl = self.edge_metrics(self.site.unit_at(Q[:, 0], Q[:, 1]), grade)
            epm, slip = epm + 0.5 * e, slip + 0.5 * sl
        s = np.r_[0.0, np.cumsum(L)]
        prof = {"s": s[1:], "x": mid[:, 0], "y": mid[:, 1], "grade_deg": grade, "unit": unit, "slip": slip,
                "risk": risk, "energy_per_m": epm}
        return Route(np.asarray(waypoints, float), cells if cells is not None else np.asarray(waypoints, float),
                     float(L.sum()), float((epm * L).sum()), float(risk.max()), float((risk * L).sum() / L.sum()),
                     float(slip.max()), prof, label)

    def risk_map(self, cell: float | None = None):
        """Worst-heading risk per grid point (the largest risk over 8 headings) for maps."""
        c = cell or self.cell
        x0, x1 = self.site.x_range
        y0, y1 = self.site.y_range
        X, Y = np.meshgrid(np.arange(x0, x1 + 1e-9, c), np.arange(y0, y1 + 1e-9, c))
        U = self.site.unit_at(X, Y)
        worst = np.zeros(X.shape)
        for a in np.radians(np.arange(0, 360, 45)):
            r, _, _ = self.edge_metrics(U, self.site.grade_deg(X, Y, a))
            worst = np.maximum(worst, r)
        return X, Y, worst


def densify(P, step):
    out = [P[0]]
    for a, b in zip(P[:-1], P[1:], strict=True):
        n = max(1, int(math.ceil(np.linalg.norm(b - a) / step)))
        out += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(out)


def simplify(P, tol: float, accept=None):
    """Douglas-Peucker simplification of a polyline (tolerance ``tol`` metres); ``accept(a, b)`` can veto a
    shortcut (e.g. one that crosses risky ground), which is then split further."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return P
    a, b = P[0], P[-1]
    d = b - a
    n = np.linalg.norm(d)
    dist = (np.abs(d[0] * (P[:, 1] - a[1]) - d[1] * (P[:, 0] - a[0])) / n if n > 0
            else np.linalg.norm(P - a, axis=1))
    k = int(np.argmax(dist))
    if dist[k] <= tol and (accept is None or accept(a, b)):
        return np.array([a, b])
    k = max(1, min(k, len(P) - 2)) if dist[k] > tol else len(P) // 2
    return np.vstack([simplify(P[:k + 1], tol, accept)[:-1], simplify(P[k:], tol, accept)])


__all__ = ["RiskPlanner", "Route", "UnitTables", "simplify", "densify"]
