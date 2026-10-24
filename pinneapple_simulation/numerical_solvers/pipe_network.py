"""Steady flow in a looped pipe network (water distribution) by the global gradient algorithm.

Unknowns: the flow in every pipe and the head at every junction; fixed-head nodes (reservoirs, tanks) close the
system. Energy along each pipe ``h_from - h_to = r(Q) Q |Q|^(n-1)`` and continuity at each junction
``sum(inflow) - sum(outflow) = demand`` are solved together by Newton's method on the Schur complement in the
heads (Todini & Pilati 1988, "A gradient algorithm for the analysis of pipe networks"; the method of EPANET,
Rossman 2000). Head loss:

* Hazen-Williams (SI, EPANET constant): ``h = 10.667 C^-1.852 D^-4.871 L Q^1.852``;
* Darcy-Weisbach: ``h = 8 f L Q^2 / (g pi^2 D^5)``, ``f`` from :func:`pinneapple_physics.closed_form.pipe_flow.
  friction_factor` (64/Re laminar, Colebrook turbulent), updated each iteration (friction fixed within a Newton
  step, as in EPANET).

Also: a genetic algorithm for least-cost discrete pipe sizing under a minimum-pressure constraint
(:func:`optimize_diameters`), the classic design problem of the two-loop network (Alperovits & Shamir 1977).
"""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from pinneapple_physics.closed_form.pipe_flow import friction_factor

__all__ = ["Network", "solve_network", "optimize_diameters", "two_loop_network", "TWO_LOOP_COSTS"]

G = 9.80665
NU_WATER = 1.004e-6          # m2/s at 20 C (Incropera Table A.6)


@dataclass
class Network:
    """Nodes by name: ``elevation`` (m), ``demand`` (m3/s, positive = drawn off) for junctions, ``head`` (m) for
    fixed-head nodes. Pipes: (name, from, to, length m, diameter m, roughness) with roughness = Hazen-Williams C or
    Darcy-Weisbach absolute roughness in m, per ``headloss`` ("HW" or "DW")."""
    junctions: dict[str, tuple[float, float]]                 # name -> (elevation, demand)
    reservoirs: dict[str, float]                              # name -> head
    pipes: list[tuple[str, str, str, float, float, float]]
    headloss: str = "HW"
    coords: dict[str, tuple[float, float]] = field(default_factory=dict)

    def with_diameters(self, d: Sequence[float]) -> Network:
        pipes = [(n, a, b, L, float(di), r) for (n, a, b, L, _, r), di in zip(self.pipes, d, strict=False)]
        return Network(dict(self.junctions), dict(self.reservoirs), pipes, self.headloss, dict(self.coords))

    def with_roughness(self, rough: dict[str, float]) -> Network:
        pipes = [(n, a, b, L, D, rough.get(n, r)) for (n, a, b, L, D, r) in self.pipes]
        return Network(dict(self.junctions), dict(self.reservoirs), pipes, self.headloss, dict(self.coords))


def _loss_coeff(net: Network, Q: np.ndarray) -> tuple[np.ndarray, float]:
    L = np.array([p[3] for p in net.pipes])
    D = np.array([p[4] for p in net.pipes])
    R = np.array([p[5] for p in net.pipes])
    if net.headloss == "HW":
        return 10.667 * R ** -1.852 * D ** -4.871 * L, 1.852
    if net.headloss != "DW":
        raise ValueError("headloss is 'HW' or 'DW'")
    A = math.pi * D ** 2 / 4
    Re = np.maximum(np.abs(Q) / A * D / NU_WATER, 1.0)
    f = np.array([friction_factor(float(re), float(r / d)) for re, r, d in zip(Re, R, D, strict=False)])
    return 8 * f * L / (G * math.pi ** 2 * D ** 5), 2.0


def solve_network(net: Network, tol: float = 1e-10, max_iter: int = 100) -> dict:
    """Heads, pressures (head - elevation), flows, velocities and head losses; ``iterations``, the continuity
    residual per junction and the energy residual per pipe (both checks of the converged solution)."""
    jn = list(net.junctions)
    rn = list(net.reservoirs)
    ji = {n: i for i, n in enumerate(jn)}
    ri = {n: i for i, n in enumerate(rn)}
    m = len(net.pipes)
    A12 = np.zeros((m, len(jn)))
    A10 = np.zeros((m, len(rn)))
    for k, (_, a, b, *_r) in enumerate(net.pipes):
        for node, s in ((a, -1.0), (b, 1.0)):
            if node in ji:
                A12[k, ji[node]] = s
            elif node in ri:
                A10[k, ri[node]] = s
            else:
                raise KeyError(f"pipe {net.pipes[k][0]}: unknown node {node!r}")
    H0 = np.array([net.reservoirs[n] for n in rn])
    q = np.array([net.junctions[n][1] for n in jn])
    D = np.array([p[4] for p in net.pipes])
    Q = np.full(m, 0.3 * math.pi * D.mean() ** 2 / 4)            # 0.3 m/s start
    H = np.full(len(jn), H0.max())
    for it in range(1, max_iter + 1):  # noqa: B007 - the count is returned
        r, n = _loss_coeff(net, Q)
        aq = np.maximum(np.abs(Q), 1e-9)
        hl = r * Q * aq ** (n - 1)
        F1 = hl + A12 @ H + A10 @ H0                              # energy: h_loss - (H_from - H_to) = 0
        F2 = A12.T @ Q - q                                        # continuity: inflow - outflow - demand = 0
        Dg = n * r * aq ** (n - 1)
        Di = 1.0 / Dg
        S = A12.T @ (Di[:, None] * A12)
        dH = np.linalg.solve(S, F2 - A12.T @ (Di * F1))
        dQ = -Di * (F1 + A12 @ dH)
        H, Q = H + dH, Q + dQ
        if np.abs(dQ).sum() / np.abs(Q).sum() < tol and np.abs(dH).max() < 1e-8:
            break
    r, n = _loss_coeff(net, Q)
    hl = r * Q * np.abs(Q) ** (n - 1)
    elev = np.array([net.junctions[x][0] for x in jn])
    A = math.pi * D ** 2 / 4
    return {"junctions": jn, "head": dict(zip(jn, H, strict=False)), "pressure_head": dict(zip(jn, H - elev, strict=False)),
            "pipes": [p[0] for p in net.pipes], "flow": dict(zip((p[0] for p in net.pipes), Q, strict=False)),
            "velocity": dict(zip((p[0] for p in net.pipes), Q / A, strict=False)),
            "headloss": dict(zip((p[0] for p in net.pipes), hl, strict=False)), "iterations": it,
            "continuity_residual": float(np.abs(A12.T @ Q - q).max()),
            "energy_residual": float(np.abs(hl + A12 @ H + A10 @ H0).max()),
            "supply": float(-(A10.T @ Q).sum())}


def two_loop_network(diameters_in: Sequence[float] = (18, 10, 16, 4, 16, 10, 10, 1), headloss: str = "HW",
                     roughness: float | None = None) -> Network:
    """The two-loop network of Alperovits & Shamir (1977), "Design of optimal water distribution systems", Water
    Resources Research 13(6): one reservoir at 210 m, six junctions, eight 1000 m pipes, Hazen-Williams C = 130.
    Default diameters (inches): the least-cost design reported by Savic & Walters (1997) and others ($419,000)."""
    j = {"2": (150.0, 100.0), "3": (160.0, 100.0), "4": (155.0, 120.0), "5": (150.0, 270.0), "6": (165.0, 330.0),
         "7": (160.0, 200.0)}
    junctions = {k: (z, d / 3600.0) for k, (z, d) in j.items()}
    rough = roughness if roughness is not None else (130.0 if headloss == "HW" else 0.26e-3)
    links = [("1", "1", "2"), ("2", "2", "3"), ("3", "2", "4"), ("4", "4", "5"), ("5", "4", "6"), ("6", "6", "7"),
             ("7", "3", "5"), ("8", "7", "5")]
    pipes = [(n, a, b, 1000.0, d * 0.0254, rough) for (n, a, b), d in zip(links, diameters_in, strict=False)]
    coords = {"1": (2000.0, 3000.0), "2": (2000.0, 2000.0), "3": (1000.0, 2000.0), "4": (2000.0, 1000.0),
              "5": (1000.0, 1000.0), "6": (2000.0, 0.0), "7": (1000.0, 0.0)}
    return Network(junctions, {"1": 210.0}, pipes, headloss, coords)


# Alperovits & Shamir (1977): commercial diameters (inch) and unit cost ($/m)
TWO_LOOP_COSTS = {1: 2, 2: 5, 3: 8, 4: 11, 6: 16, 8: 23, 10: 32, 12: 50, 14: 60, 16: 90, 18: 130, 20: 170, 22: 300,
                  24: 550}


def optimize_diameters(net: Network, sizes_m: Sequence[float], unit_cost: Sequence[float], p_min: float, *,
                       pop: int = 100, generations: int = 400, seed: int = 0, penalty: float = 1e6,
                       elite: int = 4) -> dict:
    """Least-cost discrete diameters with every junction pressure >= ``p_min`` (m): a genetic algorithm (integer
    genes, tournament selection, uniform crossover, creep mutation, elitism) with a penalty per metre of pressure
    deficit; every candidate is evaluated with :func:`solve_network`. Returns the best design, its cost, minimum
    pressure, number of hydraulic solves and the convergence history."""
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes_m, float)
    cost = np.asarray(unit_cost, float)
    L = np.array([p[3] for p in net.pipes])
    m, k = len(net.pipes), len(sizes)
    cache: dict[tuple, tuple[float, float, float]] = {}

    def fitness(g):
        key = tuple(int(x) for x in g)
        if key not in cache:
            c = float((cost[list(key)] * L).sum())
            try:
                r = solve_network(net.with_diameters(sizes[list(key)]))
                pmin = min(r["pressure_head"].values())
            except np.linalg.LinAlgError:
                pmin = -1e3
            cache[key] = (c + penalty * max(0.0, p_min - pmin), c, pmin)
        return cache[key]

    P = rng.integers(0, k, (pop, m))
    hist = []
    for _ in range(generations):
        F = np.array([fitness(g)[0] for g in P])
        order = np.argsort(F)
        hist.append(float(F[order[0]]))
        new = [P[i].copy() for i in order[:elite]]
        while len(new) < pop:
            a, b = (min(rng.integers(0, pop, 3), key=lambda i: F[i]) for _ in range(2))
            mask = rng.random(m) < 0.5
            child = np.where(mask, P[a], P[b])
            mut = rng.random(m) < 1.0 / m
            child[mut] = np.clip(child[mut] + rng.choice([-2, -1, 1, 2], mut.sum()), 0, k - 1)
            new.append(child)
        P = np.array(new)
    best = min(cache.items(), key=lambda kv: kv[1][0])
    g, (_, c, pmin) = best
    return {"genes": list(g), "diameters_m": sizes[list(g)].tolist(), "cost": c, "min_pressure": pmin,
            "feasible": pmin >= p_min - 1e-9, "evaluations": len(cache), "history": hist}
