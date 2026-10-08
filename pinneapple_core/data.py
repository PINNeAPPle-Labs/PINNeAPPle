"""Physics-native data API: ``PhysicsDataset``, ``DataLoader`` and the samplers.

One loader serves a PINN, a neural operator and an inverse problem. A loader takes *sources*, each either a
:class:`PhysicsDataset` (labelled arrays, shuffled and batched) or a :class:`Sampler` (points drawn on demand),
and yields a :class:`Batch` with one group per source::

    loader = DataLoader({"col": CollocationSampler(domain), "bc": BoundarySampler(geometry),
                         "obs": PhysicsDataset(x_obs, u_obs)},
                        batch_size={"col": 1024, "bc": 128, "obs": 16}, steps=2000)
    for batch in loader:
        x = batch["col"]["x"]                  # requires_grad already set: collocation points are differentiated
        ...

Samplers: :class:`CollocationSampler` (interior, uniform / lhs / sobol), :class:`BoundarySampler` (boundary points with
outward normals, whole boundary or a named part), :class:`MeshSampler` (nodes, cell centres or random points in cells),
:class:`TrajectorySampler` (windows of time series), :class:`AdaptiveSampler` (density follows the PDE residual) and
:class:`ActiveSampler` (picks the highest-scoring candidates by an acquisition function).
``physics_aware=True`` (the default) makes the loader set ``requires_grad`` on the points of sources that need
derivatives, convert to the model dtype and device, and attach the coordinate names to ``batch.meta``.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import Tensor

from .domain import Domain
from .geometry import Geometry
from .mesh import Mesh

__all__ = ["Batch", "PhysicsDataset", "DataLoader", "Sampler", "CollocationSampler", "BoundarySampler", "MeshSampler",
           "TrajectorySampler", "AdaptiveSampler", "ActiveSampler"]


# -- batch ---------------------------------------------------------------------
@dataclass
class Batch:
    """One training step's data: ``batch["col"]["x"]``. With a single source, ``batch.x`` and ``batch.y`` shortcut it."""
    groups: Dict[str, Dict[str, Tensor]]
    step: int = 0
    meta: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Dict[str, Tensor]:
        return self.groups[name]

    def __contains__(self, name: str) -> bool:
        return name in self.groups

    def _only(self) -> Dict[str, Tensor]:
        if len(self.groups) != 1:
            raise AttributeError(f"batch has several groups {list(self.groups)}; use batch['name']")
        return next(iter(self.groups.values()))

    @property
    def x(self) -> Tensor:
        return self._only()["x"]

    @property
    def y(self) -> Tensor:
        return self._only()["y"]

    def to(self, device=None, dtype=None) -> "Batch":
        def move(t: Tensor) -> Tensor:
            t = t.to(device=device) if device is not None else t
            return t.to(dtype=dtype) if dtype is not None and t.is_floating_point() else t
        return Batch({g: {k: move(v) for k, v in d.items()} for g, d in self.groups.items()}, self.step, dict(self.meta))


# -- dataset ---------------------------------------------------------------------
class PhysicsDataset(torch.utils.data.Dataset):
    """Labelled samples ``inputs[i] -> targets[i]`` plus the physics context: coordinate and field names, domain.

    ``inputs`` / ``targets`` are arrays or tensors with the sample index first. For a PINN's observations ``inputs`` are
    the coordinates; for an operator they are the input functions on a grid.
    """

    needs_grad = False

    def __init__(self, inputs, targets=None, *, coords: Optional[Sequence[str]] = None, fields: Optional[Sequence[str]] = None,
                 domain: Optional[Domain] = None, name: str = "", provenance: Optional[Sequence[Mapping[str, Any]]] = None) -> None:
        self.inputs = torch.as_tensor(np.asarray(inputs) if not isinstance(inputs, Tensor) else inputs)
        self.targets = None if targets is None else torch.as_tensor(np.asarray(targets) if not isinstance(targets, Tensor) else targets)
        if self.targets is not None and self.targets.shape[0] != self.inputs.shape[0]:
            raise ValueError(f"inputs have {self.inputs.shape[0]} samples but targets have {self.targets.shape[0]}")
        self.coords = tuple(coords) if coords else None
        self.fields = tuple(fields) if fields else None
        self.domain = domain
        self.name = name
        self.provenance: List[Dict[str, Any]] = [dict(r) for r in (provenance or [])]   # transforms applied, in order

    def __len__(self) -> int:
        return int(self.inputs.shape[0])

    def __getitem__(self, i):
        return (self.inputs[i],) if self.targets is None else (self.inputs[i], self.targets[i])

    def take(self, idx: Tensor) -> Dict[str, Tensor]:
        out = {"x": self.inputs[idx]}
        if self.targets is not None:
            out["y"] = self.targets[idx]
        return out

    def split(self, fractions: Sequence[float], seed: int = 0) -> List["PhysicsDataset"]:
        """Random split, e.g. ``split([0.8, 0.2])``; fractions must sum to 1."""
        if abs(sum(fractions) - 1.0) > 1e-9:
            raise ValueError("fractions must sum to 1")
        perm = torch.randperm(len(self), generator=torch.Generator().manual_seed(seed))
        cuts = np.round(np.cumsum(fractions) * len(self)).astype(int)
        parts, lo = [], 0
        for hi in cuts:
            idx = perm[lo:hi]
            t = None if self.targets is None else self.targets[idx]
            parts.append(PhysicsDataset(self.inputs[idx], t, coords=self.coords, fields=self.fields, domain=self.domain,
                                        name=self.name, provenance=self.provenance))
            lo = hi
        return parts

    def __repr__(self) -> str:
        return f"PhysicsDataset(n={len(self)}, inputs={tuple(self.inputs.shape[1:])}, targets={None if self.targets is None else tuple(self.targets.shape[1:])})"


# -- samplers ----------------------------------------------------------------------
class Sampler:
    """Draws points on demand. Subclasses implement :meth:`sample` and return a dict of tensors with key ``"x"``."""

    needs_grad = False
    coords: Optional[Tuple[str, ...]] = None

    def sample(self, n: int) -> Dict[str, Tensor]:
        raise NotImplementedError

    def update(self, model: Optional[Callable] = None) -> None:
        """Hook called by ``DataLoader.update(model)`` for samplers that adapt to the model."""

    def _rng(self, seed: int, counter: int) -> np.random.Generator:
        return np.random.default_rng([seed, counter])


def _as_domain(domain) -> Tuple[Domain, Optional[Tuple[str, ...]], Optional[Geometry]]:
    if isinstance(domain, Geometry):
        return domain.domain, None, domain
    if isinstance(domain, Domain):
        return domain, None, None
    if isinstance(domain, Mapping):
        coords = tuple(domain)
        lo = [domain[c][0] for c in coords]
        hi = [domain[c][1] for c in coords]
        return Domain.box(lo, hi), coords, None
    raise TypeError("domain must be a Domain, a Geometry or a {coord: (lo, hi)} dict")


class CollocationSampler(Sampler):
    """Interior points for the PDE residual. ``domain``: a ``Domain``, a ``Geometry`` or ``{"x": (0, 1), "t": (0, 1)}``.
    ``strategy``: ``uniform``, ``lhs`` (Latin hypercube) or ``sobol``. Every call draws fresh points, reproducibly from ``seed``."""

    needs_grad = True

    def __init__(self, domain, strategy: str = "uniform", seed: int = 0) -> None:
        if strategy not in ("uniform", "lhs", "sobol"):
            raise ValueError("strategy must be 'uniform', 'lhs' or 'sobol'")
        self.domain, self.coords, _ = _as_domain(domain)
        self.strategy, self.seed, self._calls = strategy, seed, 0

    def _box(self, n: int, rng: np.random.Generator) -> np.ndarray:
        d, lo, span = self.domain.dim, self.domain.lo, self.domain.hi - self.domain.lo
        if self.strategy == "uniform":
            u = rng.random((n, d))
        elif self.strategy == "lhs":
            from scipy.stats import qmc
            u = qmc.LatinHypercube(d=d, seed=rng).random(n)
        else:
            from scipy.stats import qmc
            u = qmc.Sobol(d=d, scramble=True, seed=rng).random(n)
        return lo + u * span

    def sample(self, n: int) -> Dict[str, Tensor]:
        rng = self._rng(self.seed, self._calls)
        self._calls += 1
        if self.domain.is_box:
            x = self._box(n, rng)
        else:                                            # rejection from the bounding box keeps the strategy's spread
            kept: List[np.ndarray] = []
            while sum(len(k) for k in kept) < n:
                cand = self._box(max(2 * n, 64), rng)
                kept.append(cand[self.domain.contains(cand)])
            x = np.concatenate(kept)[:n]
        return {"x": torch.as_tensor(x, dtype=torch.get_default_dtype())}


class BoundarySampler(Sampler):
    """Boundary points with outward unit normals (``"x"``, ``"normal"``). With a ``Geometry``, ``part="x_min"`` samples
    one named boundary. Normals are needed by Neumann conditions, so the points are differentiable by default."""

    needs_grad = True

    def __init__(self, domain, part: Optional[str] = None, seed: int = 0) -> None:
        self.domain, self.coords, self.geometry = _as_domain(domain)
        if part is not None:
            if self.geometry is None or part not in self.geometry.boundary_names:
                raise KeyError(f"part {part!r} needs a Geometry that names it; have "
                               f"{None if self.geometry is None else self.geometry.boundary_names}")
        self.part, self.seed, self._calls = part, seed, 0

    def sample(self, n: int) -> Dict[str, Tensor]:
        seed = int(self._rng(self.seed, self._calls).integers(2**31 - 1))
        self._calls += 1
        if self.part is None:
            x, nrm = self.domain.sample_boundary(n, seed=seed, return_normals=True)
        else:
            pred = self.geometry._boundaries[self.part]
            xs, ns, tries = [], [], 0
            while sum(len(a) for a in xs) < n:
                cx, cn = self.domain.sample_boundary(max(8 * n, 1024), seed=seed + tries, return_normals=True)
                m = np.asarray(pred(cx), dtype=bool)
                xs.append(cx[m]); ns.append(cn[m])
                tries += 1
                if tries > 200:
                    raise RuntimeError(f"could not sample boundary part {self.part!r}")
            x, nrm = np.concatenate(xs)[:n], np.concatenate(ns)[:n]
        dt = torch.get_default_dtype()
        return {"x": torch.as_tensor(x, dtype=dt), "normal": torch.as_tensor(nrm, dtype=dt)}


class MeshSampler(Sampler):
    """Points tied to a mesh: ``mode="nodes"`` (random nodes, or all when ``n >= N``), ``"cells"`` (cell centres) or
    ``"random"`` (uniform points inside cells, by volume). Returns ``x`` and the node or cell index."""

    def __init__(self, mesh: Mesh, mode: str = "nodes", seed: int = 0) -> None:
        if mode not in ("nodes", "cells", "random"):
            raise ValueError("mode must be 'nodes', 'cells' or 'random'")
        self.mesh, self.mode, self.seed, self._calls = mesh, mode, seed, 0

    def sample(self, n: int) -> Dict[str, Tensor]:
        rng = self._rng(self.seed, self._calls)
        self._calls += 1
        dt = torch.get_default_dtype()
        m = self.mesh
        if self.mode == "nodes":
            idx = np.arange(m.n_points) if n >= m.n_points else rng.choice(m.n_points, n, replace=False)
            return {"x": torch.as_tensor(m.points[idx], dtype=dt), "index": torch.as_tensor(idx)}
        if self.mode == "cells":
            idx = np.arange(m.n_cells) if n >= m.n_cells else rng.choice(m.n_cells, n, replace=False)
            return {"x": torch.as_tensor(m.cell_centroids[idx], dtype=dt), "index": torch.as_tensor(idx)}
        cells = rng.choice(m.n_cells, n, p=m.cell_volumes / m.cell_volumes.sum())
        bary = rng.dirichlet(np.ones(m.dim + 1), size=n)                  # uniform on the simplex
        x = np.einsum("ni,nid->nd", bary, m.cell_points[cells])
        return {"x": torch.as_tensor(x, dtype=dt), "index": torch.as_tensor(cells)}


class TrajectorySampler(Sampler):
    """Windows of trajectories ``states (B, T, d)``: ``x`` is ``n_in`` consecutive states, ``y`` the next ``n_out``.
    For learning dynamics, one-step maps and rollouts. Returns ``x (n, n_in, d)``, ``y (n, n_out, d)``, ``t0`` (start index)."""

    def __init__(self, states, n_in: int = 1, n_out: int = 1, stride: int = 1, seed: int = 0) -> None:
        self.states = torch.as_tensor(np.asarray(states) if not isinstance(states, Tensor) else states)
        if self.states.ndim != 3:
            raise ValueError("states must have shape (trajectories, time, state_dim)")
        self.n_in, self.n_out, self.stride = n_in, n_out, stride
        if self.states.shape[1] < n_in + n_out:
            raise ValueError("trajectories are shorter than n_in + n_out")
        self.seed, self._calls = seed, 0

    def sample(self, n: int) -> Dict[str, Tensor]:
        rng = self._rng(self.seed, self._calls)
        self._calls += 1
        b = rng.integers(0, self.states.shape[0], n)
        last = self.states.shape[1] - self.n_in - self.n_out
        t0 = rng.integers(0, last // self.stride + 1, n) * self.stride
        ix = torch.as_tensor(t0)[:, None] + torch.arange(self.n_in)[None]
        iy = torch.as_tensor(t0)[:, None] + self.n_in + torch.arange(self.n_out)[None]
        bt = torch.as_tensor(b)[:, None]
        return {"x": self.states[bt, ix], "y": self.states[bt, iy], "t0": torch.as_tensor(t0)}


class AdaptiveSampler(Sampler):
    """Residual-based adaptive sampling: a pool of candidates from ``base``; after ``update(model)`` the points are drawn
    with probability proportional to ``residual ** exponent`` (mixed with ``uniform_fraction`` of uniform draws so no
    region is abandoned). ``residual_fn(model, x) -> (P,)`` gives non-negative magnitudes. Before the first update it is
    uniform over the pool."""

    needs_grad = True

    def __init__(self, base: Sampler, residual_fn: Callable, pool_size: int = 20000, exponent: float = 1.0,
                 uniform_fraction: float = 0.2, seed: int = 0) -> None:
        self.base, self.residual_fn = base, residual_fn
        self.pool = base.sample(pool_size)["x"].detach()
        self.exponent, self.uniform_fraction, self.seed, self._calls = exponent, uniform_fraction, seed, 0
        self.coords = getattr(base, "coords", None)
        self.residual: Optional[Tensor] = None

    def update(self, model: Optional[Callable] = None) -> None:
        if model is None:
            raise ValueError("AdaptiveSampler.update needs the model")
        x = self.pool.clone().requires_grad_(True)
        r = self.residual_fn(model, x).detach().abs().reshape(-1)
        self.residual = r.clamp_min(0.0)

    def probabilities(self) -> np.ndarray:
        P = self.pool.shape[0]
        if self.residual is None or float(self.residual.sum()) <= 0:
            return np.full(P, 1.0 / P)
        w = self.residual.numpy().astype(np.float64) ** self.exponent
        return (1 - self.uniform_fraction) * w / w.sum() + self.uniform_fraction / P

    def sample(self, n: int) -> Dict[str, Tensor]:
        rng = self._rng(self.seed, self._calls)
        self._calls += 1
        p = self.probabilities()
        idx = rng.choice(self.pool.shape[0], size=min(n, self.pool.shape[0]), replace=False, p=p)
        return {"x": self.pool[idx].clone(), "index": torch.as_tensor(idx)}


class ActiveSampler(Sampler):
    """Acquisition-driven selection: draw ``n_candidates`` from ``base`` and return the ``n`` with the highest
    ``score_fn(model, x)`` (uncertainty, ensemble variance, expected information, ...). Use it to choose where to run
    the next expensive simulation or experiment. Differs from :class:`AdaptiveSampler`, which samples *proportionally*."""

    def __init__(self, base: Sampler, score_fn: Callable, n_candidates: int = 2000) -> None:
        self.base, self.score_fn, self.n_candidates = base, score_fn, n_candidates
        self.coords = getattr(base, "coords", None)
        self.model: Optional[Callable] = None

    def update(self, model: Optional[Callable] = None) -> None:
        self.model = model

    def sample(self, n: int) -> Dict[str, Tensor]:
        cand = self.base.sample(max(self.n_candidates, n))["x"]
        if self.model is None:
            return {"x": cand[:n], "score": torch.zeros(min(n, len(cand)))}
        score = self.score_fn(self.model, cand).detach().reshape(-1)
        top = torch.argsort(score, descending=True)[:n]
        return {"x": cand[top], "score": score[top]}


# -- loader ----------------------------------------------------------------------------
class DataLoader:
    """Batches from datasets and samplers behind one iteration API.

    Args:
        sources: a :class:`PhysicsDataset`, a :class:`Sampler`, or ``{name: source}``.
        batch_size: an int for every source, or ``{name: int}``.
        steps: number of batches. Optional when there is a dataset: then one pass over the *first* dataset
            (``ceil(len / batch_size)`` batches) is one epoch. Required when every source is a sampler.
        shuffle: reshuffle datasets each epoch. ``drop_last``: drop the last short dataset batch.
        physics_aware: set ``requires_grad`` on the points of sources that are differentiated (collocation, boundary,
            adaptive), convert to ``dtype`` / ``device``, add ``meta["coords"]``.
    """

    def __init__(self, sources, batch_size: Union[int, Mapping[str, int]] = 256, steps: Optional[int] = None,
                 shuffle: bool = True, drop_last: bool = False, seed: int = 0, physics_aware: bool = True,
                 device=None, dtype=None) -> None:
        if isinstance(sources, (PhysicsDataset, Sampler)):
            sources = {"data" if isinstance(sources, PhysicsDataset) else "col": sources}
        if not sources:
            raise ValueError("DataLoader needs at least one source")
        for k, v in sources.items():
            if not isinstance(v, (PhysicsDataset, Sampler)):
                raise TypeError(f"source {k!r} must be a PhysicsDataset or a Sampler, got {type(v).__name__}")
        self.sources: Dict[str, Any] = dict(sources)
        if isinstance(batch_size, int):
            self.batch_size = {k: batch_size for k in self.sources}
        else:
            missing = set(self.sources) - set(batch_size)
            if missing:
                raise KeyError(f"batch_size dict is missing {sorted(missing)}")
            self.batch_size = {k: int(batch_size[k]) for k in self.sources}
        self.primary = next((k for k, v in self.sources.items() if isinstance(v, PhysicsDataset)), None)
        if steps is None and self.primary is None:
            raise ValueError("steps= is required when every source is a sampler")
        self.steps, self.shuffle, self.drop_last, self.seed = steps, shuffle, drop_last, seed
        self.physics_aware, self.device, self.dtype = physics_aware, device, dtype
        self._epoch = 0

    def __len__(self) -> int:
        if self.steps is not None:
            return self.steps
        n, b = len(self.sources[self.primary]), self.batch_size[self.primary]
        return n // b if self.drop_last else math.ceil(n / b)

    def update(self, model: Optional[Callable] = None) -> None:
        """Tell adaptive and active samplers about the current model (call every few steps)."""
        for s in self.sources.values():
            if isinstance(s, Sampler):
                s.update(model)

    def _perm(self, name: str, epoch: int, n: int) -> Tensor:
        if not self.shuffle:
            return torch.arange(n)
        return torch.randperm(n, generator=torch.Generator().manual_seed(zlib.crc32(f"{self.seed}/{epoch}/{name}".encode())))

    def __iter__(self) -> Iterator[Batch]:
        epoch = self._epoch
        self._epoch += 1
        cursors: Dict[str, Tuple[Tensor, int, int]] = {}      # dataset name -> (permutation, position, epoch)
        coords = {k: getattr(v, "coords", None) for k, v in self.sources.items()}
        for step in range(len(self)):
            groups: Dict[str, Dict[str, Tensor]] = {}
            for name, src in self.sources.items():
                bs = self.batch_size[name]
                if isinstance(src, Sampler):
                    groups[name] = src.sample(bs)
                    continue
                n = len(src)
                perm, pos, ep = cursors.get(name) or (self._perm(name, epoch, n), 0, epoch)
                if pos + bs > n:
                    if name == self.primary and self.steps is None:   # epoch mode: the short tail is the last batch
                        idx = perm[pos:]
                        groups[name] = src.take(idx)
                        cursors[name] = (perm, n, ep)
                        continue
                    ep += 1
                    perm, pos = self._perm(name, ep, n), 0
                idx = perm[pos:pos + bs]
                groups[name] = src.take(idx)
                cursors[name] = (perm, pos + bs, ep)
            batch = Batch(groups, step=step, meta={"epoch": epoch, "coords": coords})
            yield self._prepare(batch)

    def _prepare(self, batch: Batch) -> Batch:
        if self.device is not None or self.dtype is not None:
            batch = batch.to(self.device, self.dtype)
        if self.physics_aware:
            for name, src in self.sources.items():
                if getattr(src, "needs_grad", False) and batch[name]["x"].is_floating_point():
                    batch[name]["x"] = batch[name]["x"].detach().requires_grad_(True)
        return batch
