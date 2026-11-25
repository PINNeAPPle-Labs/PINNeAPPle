"""The design space: bounds, constraints and a validity gate over one morpher, so a sampler can propose a design and
the study gets either a mesh or the reason it cannot have one.

Everything in ``pinneapple_design.geometry.morph`` maps a parameter vector to vertices. A study needs one more thing
around that map: the band each parameter may take, the regions that may not move, and the check that decides whether
the result is still a body. That is this object, and it is deliberately small — ``.bounds``, ``.sample``, ``.build``,
``.check``, ``.sample_valid`` — so a sampler (#209), an explorer (#222) or the simulation factory (#152) can hold it
without knowing which morpher is inside.

Rejection is an outcome, not an exception: ``sample_valid`` reports how many draws it took, and counts the reasons,
so a design space whose bounds produce 80 % self-intersecting car bodies is a measured fact in the study instead of a
silent slowdown.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import numpy as np

from .surface import Surface
from .validity import Reference, ValidityReport, check_surface

__all__ = ["DesignSpace", "DesignSpaceReport"]


@dataclass
class DesignSpaceReport:
    """What ``sample_valid`` did: the designs it kept, the ones it refused, and why."""
    parameters: np.ndarray
    reports: list[ValidityReport] = field(default_factory=list)
    draws: int = 0
    rejected: dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def accepted(self) -> int:
        return len(self.parameters)

    @property
    def acceptance_rate(self) -> float:
        return float(self.accepted / self.draws) if self.draws else 0.0

    def summary(self) -> dict[str, Any]:
        return {"draws": self.draws, "accepted": self.accepted, "rejected": sum(self.rejected.values()),
                "acceptance_rate": self.acceptance_rate, "reasons": dict(self.rejected),
                "seconds": self.seconds, "seconds_per_draw": self.seconds / max(self.draws, 1)}


class DesignSpace:
    """A morpher plus everything that decides which of its outputs a solver may see.

    Parameters
    ----------
    morph: ``FFD``, ``RBFMorph`` or ``ShapeModes`` — anything with ``deform(x)``, ``n_parameters`` and
        ``bounds(scale)``.
    faces: the topology the morpher was fitted on; carried along untouched.
    reference: the reference vertices; every check compares against them.
    lo, hi: parameter bounds; default ``morph.bounds()``.
    keep_out: ``(lo, hi, tol)`` boxes: vertices that start inside a box may not move further than ``tol`` from their
        reference position. That is how a mounting face, a symmetry wall or a region graded for another purpose stays
        put while the rest of the body is free.
    gate: extra keyword arguments for :func:`check_surface` (``max_volume_change``, ``allow_flipped``, ...).
    """

    def __init__(self, morph: Any, faces: Any, reference: Any, *, lo: Any = None, hi: Any = None,
                 keep_out: Sequence[Sequence[Any]] = (), gate: dict[str, Any] | None = None,
                 names: Sequence[str] | None = None):
        self.morph = morph
        self.reference = np.asarray(reference, dtype=np.float64).reshape(-1, 3)
        self.faces = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
        dim = int(getattr(morph, "n_parameters"))
        if lo is None or hi is None:
            blo, bhi = morph.bounds()
        else:
            blo, bhi = lo, hi
        self.lo = np.asarray(blo, dtype=float).reshape(-1)
        self.hi = np.asarray(bhi, dtype=float).reshape(-1)
        if self.lo.size != dim or self.hi.size != dim:
            raise ValueError(f"DesignSpace: {dim} parameters but {self.lo.size}/{self.hi.size} bounds")
        self.keep_out = [tuple(b) for b in keep_out]
        self.gate = dict(gate or {})
        self.names = list(names) if names is not None else list(getattr(morph, "names", []) or
                                                                 [f"x{i}" for i in range(dim)])
        self._keep_out_masks = [self._mask(blo_, bhi_) for blo_, bhi_, _ in self.keep_out]
        # the reference numbers do not change between samples: measure them once
        self._reference = Reference.build(self.reference, self.faces,
                                          direction=self.gate.get("direction", (1.0, 0.0, 0.0)))

    # ------------------------------------------------------------------------------------- shape
    def _mask(self, lo: Any, hi: Any) -> np.ndarray:
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        return np.all((self.reference >= lo) & (self.reference <= hi), axis=1)

    @property
    def dim(self) -> int:
        """Number of design variables."""
        return self.lo.size

    @property
    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lo, self.hi

    def __repr__(self) -> str:
        return f"DesignSpace({self.dim} variables, {len(self.faces)} faces, {len(self.keep_out)} keep-out boxes)"

    # ------------------------------------------------------------------------------------- designs
    def build(self, x: Any) -> Surface:
        """The surface for a design; no check, no rejection (use :meth:`check` for that)."""
        x = self._vector(x)
        return Surface(self.morph.deform(x), self.faces, reference=self.reference)

    def check(self, x: Any) -> ValidityReport:
        """Deform and run the gate: ``ok`` with the measured numbers, or the reasons it is refused.

        A design outside the declared box is refused before anything is deformed: the bounds are part of the space,
        not a hint to the sampler, and a point outside them is not a member."""
        x = self._vector(x)
        outside = np.nonzero((x < self.lo) | (x > self.hi))[0]
        if len(outside):
            i = int(outside[0])
            return ValidityReport(ok=False, reasons=[
                f"{len(outside)} variables outside the declared bounds (variable {i} = {x[i]:.6g}, "
                f"bounds [{self.lo[i]:.6g}, {self.hi[i]:.6g}])"], metrics={"outside_bounds": len(outside)})
        return self._check(self.morph.deform(x))

    def _check(self, V: np.ndarray) -> ValidityReport:
        report = check_surface(V, self.faces, reference=self._reference, **self.gate)
        for (blo, bhi, tol), mask in zip(self.keep_out, self._keep_out_masks, strict=True):
            if not mask.any():
                continue
            moved = float(np.abs(V[mask] - self.reference[mask]).max())
            report.metrics.setdefault("keep_out_moved", []).append(moved)
            if moved > tol:
                report.ok = False
                report.reasons.append(
                    f"{int(mask.sum())} vertices of the keep-out box {list(map(float, blo))}.."
                    f"{list(map(float, bhi))} moved {moved:.3g} (limit {tol:g})")
        return report

    def _vector(self, x: Any) -> np.ndarray:
        v = np.asarray(x, dtype=float).reshape(-1)
        if v.size != self.dim:
            raise ValueError(f"DesignSpace: expected {self.dim} variables, got {v.size}")
        return v

    # ------------------------------------------------------------------------------------- sampling
    def sample(self, n: int, *, seed: int = 0, method: str = "lhs", centre: bool = False) -> np.ndarray:
        """``n`` designs inside the bounds: a Latin hypercube by default, ``"uniform"`` for independent draws.

        ``centre=True`` returns the middle of the box (the reference design for a symmetric band), which is what an
        optimization starts from."""
        n = int(n)
        if n <= 0:
            raise ValueError("DesignSpace: n must be positive")
        if centre:
            return 0.5 * (self.lo + self.hi)
        rng = np.random.default_rng(seed)
        if method == "uniform":
            return rng.uniform(self.lo, self.hi, size=(n, self.dim))
        if method != "lhs":
            raise ValueError(f"DesignSpace: unknown sampling method {method!r} (lhs, uniform)")
        u = np.empty((n, self.dim))
        for d in range(self.dim):
            u[:, d] = (rng.permutation(n) + rng.random(n)) / n
        return self.lo + u * (self.hi - self.lo)

    def sample_valid(self, n: int, *, seed: int = 0, method: str = "lhs", max_draws: int | None = None,
                     on_accept: Callable[[np.ndarray, Surface, ValidityReport], None] | None = None,
                     ) -> DesignSpaceReport:
        """Draw until ``n`` designs pass the gate, recording what each refusal was.

        ``max_draws`` bounds the work (default ``20 n``); reaching it returns fewer designs rather than raising, and
        the acceptance rate in the report says how far the bounds are from a usable design space."""
        import time

        t0 = time.perf_counter()
        limit = int(max_draws if max_draws is not None else 20 * max(int(n), 1))
        kept: list[np.ndarray] = []
        reports: list[ValidityReport] = []
        rejected: dict[str, int] = {}
        draws = 0
        while len(kept) < n and draws < limit:
            block = self.sample(min(max(n - len(kept), 1) * 2, 64), seed=seed + draws, method=method)
            draws += len(block)
            for x in block:
                V = self.morph.deform(x)
                report = self._check(V)
                if report.ok:
                    kept.append(x)
                    reports.append(report)
                    if on_accept is not None:
                        on_accept(x, Surface(V, self.faces, reference=self.reference), report)
                    if len(kept) >= n:
                        break
                else:
                    for r in report.reasons:
                        key = r.split(" (")[0]
                        rejected[key] = rejected.get(key, 0) + 1
        return DesignSpaceReport(parameters=np.asarray(kept).reshape(-1, self.dim), reports=reports,
                                 draws=draws, rejected=rejected, seconds=time.perf_counter() - t0)
