"""Stage 1: the design space and how it is sampled.

A design space is two lists of ranges — geometry and operating point — and a rule for picking points inside them. The
rule matters more than it looks: a uniform random draw leaves holes and clumps, and every hole is a region the model
will be confident about and wrong in. So the default is a **low-discrepancy** draw (Sobol, or Latin hypercube when the
caller wants one sample per stratum), and the campaign can then ask where to look again.

``sample`` draws from a family's own declared ranges, so the space is always the one the family says it covers.
``refine`` is the active half: given the confidence the model reported on what already exists, it proposes the next
bodies where that confidence is weakest.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["METHODS", "Proposal", "refine", "sample"]

#: sampling rules by name. ``uniform`` stays because it is what the first eleven design spaces used, and a change of
#: sampler has to be comparable with them.
METHODS = ("sobol", "lhs", "uniform")


def unit(n: int, dims: int, method: str, seed: int) -> np.ndarray:
    """``n`` points in [0, 1]^dims by the named rule, deterministic given the seed."""
    if method not in METHODS:
        raise ValueError(f"unknown sampling method {method!r}; known: {', '.join(METHODS)}")
    if dims <= 0:
        return np.zeros((n, 0))
    if method == "uniform":
        return np.random.default_rng(seed).random((n, dims))
    from scipy.stats import qmc

    engine = qmc.Sobol(dims, scramble=True, seed=seed) if method == "sobol" else qmc.LatinHypercube(dims, seed=seed)
    return engine.random(n)


def _from_unit(keys: list[str], unit_rows: np.ndarray, ranges: dict[str, tuple[float, float]]) -> list[dict]:
    return [{k: float(ranges[k][0] + (ranges[k][1] - ranges[k][0]) * u)
             for k, u in zip(keys, row, strict=True)} for row in unit_rows]


def sample(family: Any, n_geometries: int, conditions_per_geometry: int = 1, *, method: str = "sobol",
           seed: int = 0, attempts: int = 60) -> list[Any]:
    """``n_geometries`` bodies from the family's declared ranges, filtered by the family's own ``validate``.

    Returns :class:`~pinneapple_neural.lpm_pipeline.DesignSpec` objects, which is what the rest of the pipeline takes.
    A geometry the family refuses is replaced by another draw, so a rejection shortens nothing; a geometry refused
    ``attempts`` times raises with the family's reason, because that means the declared ranges and the reference
    disagree and no amount of drawing fixes it.
    """
    from ..lpm_pipeline import DesignSpec

    geo_ranges = dict(family.geometry_params)
    op_ranges = dict(family.operating_params)
    geo_keys, op_keys = list(geo_ranges), list(op_ranges)

    geo_rows = unit(n_geometries, len(geo_keys), method, seed)
    op_rows = unit(n_geometries * conditions_per_geometry, len(op_keys), method, seed + 1)

    specs: list[Any] = []
    for gid in range(n_geometries):
        reason = "no attempt made"
        for attempt in range(attempts):
            row = geo_rows[gid] if attempt == 0 else unit(1, len(geo_keys), method, seed + 7919 * (gid + 1) +
                                                          attempt)[0]
            geometry = family.geometry_from_unit(_from_unit(geo_keys, row[None, :], geo_ranges)[0])
            accepted = []
            for c in range(conditions_per_geometry):
                slot = gid * conditions_per_geometry + c
                operating = _from_unit(op_keys, op_rows[slot % len(op_rows)][None, :], op_ranges)[0]
                spec = DesignSpec(geometry=geometry, operating=operating, geometry_id=gid,
                                  label=f"{family.name}:g{gid:05d}:c{c:02d}")
                reason = family.validate(spec)
                if reason:
                    accepted = []
                    break
                accepted.append(spec)
            if accepted:
                specs.extend(accepted)
                break
        else:
            raise RuntimeError(f"{family.name}: geometry {gid} was refused {attempts} times by the family's own "
                               f"validate ({reason}); the declared ranges and the reference disagree")
    return specs


@dataclass
class Proposal:
    """One candidate design: where it sits, and why the sampler proposed it."""

    geometry: dict[str, float]
    operating: dict[str, float]
    reason: str = ""
    source: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def refine(scores: list[float], specs: list[Any], n_new: int, *, geometry_params: dict[str, tuple[float, float]] | None = None,
           spread: float = 0.15, threshold: float = 0.5, seed: int = 0) -> list[Proposal]:
    """Where to simulate next: around the designs whose confidence sits nearest the decision threshold.

    ``scores`` are confidences in [0, 1] (see :func:`~.trust.confidence`), one per spec. The worst-informed design is
    not the most useful one — a body the model has already flagged as far outside teaches it less per simulation than a
    body it almost understands — so the proposals cluster at the boundary the next simulation actually moves. Each
    proposal is a small perturbation of its source design, clipped to the family's declared ranges when they are given.

    This is a heuristic and it says so: it proposes, the caller decides, and the reason travels with the proposal.
    """
    if len(scores) != len(specs):
        raise ValueError("one confidence per spec, or the proposal is guessing")
    if n_new <= 0 or not specs:
        return []
    order = sorted(range(len(specs)), key=lambda i: (abs(scores[i] - threshold), i))
    rng = np.random.default_rng(seed)
    out: list[Proposal] = []
    for k in range(n_new):
        i = order[k % len(order)]
        base = specs[i]
        geometry = {}
        for key, value in base.geometry.items():
            value = float(value)
            step = spread * abs(value) if value else spread
            lo, hi = (geometry_params or {}).get(key, (value - 3 * step, value + 3 * step))
            geometry[key] = float(np.clip(value + rng.normal(0.0, step), lo, hi))
        out.append(Proposal(geometry=geometry, operating={k2: float(v2) for k2, v2 in base.operating.items()},
                            reason=f"confidence {scores[i]:.2f} sits at the threshold {threshold:.2f}",
                            source=getattr(base, "label", "")))
    return out
