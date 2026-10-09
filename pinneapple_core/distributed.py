"""``pp.distributed``: run physics workloads in parallel, not only training.

* **Experiments and ensembles**: :func:`map`, :func:`sweep`, :func:`ensemble` run a function over many inputs in worker processes
  (or threads) and return the results in input order, with errors collected per task instead of lost.
* **Meshes**: :func:`partition_mesh` splits a mesh into parts with one layer of halo cells, so a per-part computation of a nodal
  quantity (gradients, integrals) equals the global one; :func:`map_mesh` runs it on all parts and gathers.
* **Domains**: :func:`decompose_box` and :func:`train_decomposed` train a PINN by additive Schwarz iteration: the box is cut into
  overlapping subdomains, each trained in its own worker against the interface values of its neighbours from the previous
  iteration, so the subdomains of one iteration run concurrently.

>>> import pinneapple_core.distributed as dist
>>> dist.map(lambda k: k * k, range(5), workers=1)
[0, 1, 4, 9, 16]

``backend="process"`` uses joblib's loky pool when joblib is installed (closures and lambdas are fine) and
``concurrent.futures.ProcessPoolExecutor`` otherwise (functions must then be importable). ``"thread"`` suits numpy/torch code that
releases the GIL, ``"serial"`` runs in the caller (useful to debug). Data-parallel gradient averaging across devices
(``torch.distributed``) is not part of this module.
"""
from __future__ import annotations

import itertools
import os
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

__all__ = ["map", "sweep", "ensemble", "TaskError", "SweepResult", "cpu_count", "MeshPart", "partition_mesh", "map_mesh",
           "Subdomain", "decompose_box", "train_decomposed", "DecomposedResult"]

_builtin_map = map


def cpu_count() -> int:
    return os.cpu_count() or 1


@dataclass
class TaskError:
    """A failed task, returned in place of its result when ``errors="collect"``."""
    index: int
    item: Any
    exception: str
    message: str
    traceback: str

    def __repr__(self) -> str:
        return f"TaskError(index={self.index}, {self.exception}: {self.message})"


def _run_one(fn: Callable, index: int, item: Any, collect: bool, star: bool) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    try:
        out = fn(**item) if star else fn(item)
    except BaseException as exc:  # noqa: BLE001 - reported to the caller with its index
        if not collect:
            raise
        out = TaskError(index, item, type(exc).__name__, str(exc), traceback.format_exc())
    return out, time.perf_counter() - t0


def _dispatch(fn: Callable, items: Sequence[Any], workers: int, backend: str, collect: bool, star: bool,
              chunksize: int) -> List[Tuple[Any, float]]:
    if backend not in ("process", "thread", "serial"):
        raise ValueError("backend must be 'process', 'thread' or 'serial'")
    if workers <= 1 or len(items) <= 1 or backend == "serial":
        return [_run_one(fn, i, it, collect, star) for i, it in enumerate(items)]
    if backend == "thread":
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(lambda p: _run_one(fn, p[0], p[1], collect, star), enumerate(items)))
    try:
        from joblib import Parallel, delayed

        return Parallel(n_jobs=workers, backend="loky", batch_size=max(chunksize, 1))(
            delayed(_run_one)(fn, i, it, collect, star) for i, it in enumerate(items))
    except ImportError:
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(_run_one, fn, i, it, collect, star) for i, it in enumerate(items)]
            return [f.result() for f in futs]


def map(fn: Callable[[Any], Any], items: Iterable[Any], workers: Optional[int] = None, *, backend: str = "process",
        errors: str = "raise", chunksize: int = 1) -> List[Any]:
    """``[fn(x) for x in items]`` on ``workers`` workers (default: the CPU count), in input order.

    ``errors="raise"`` re-raises the first failure; ``errors="collect"`` returns a :class:`TaskError` in the failed slots."""
    if errors not in ("raise", "collect"):
        raise ValueError("errors must be 'raise' or 'collect'")
    items = list(items)
    out = _dispatch(fn, items, min(workers or cpu_count(), max(len(items), 1)), backend, errors == "collect", False, chunksize)
    return [r for r, _ in out]


@dataclass
class SweepResult:
    """Records of a parameter sweep: one per configuration, in grid order."""
    records: List[Dict[str, Any]]
    wall_time_s: float = 0.0
    workers: int = 1

    def __len__(self) -> int:
        return len(self.records)

    @property
    def ok(self) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["error"] is None]

    @property
    def failed(self) -> List[Dict[str, Any]]:
        return [r for r in self.records if r["error"] is not None]

    def results(self) -> List[Any]:
        return [r["result"] for r in self.records]

    def best(self, key: Callable[[Any], float] = lambda r: r, minimize: bool = True) -> Dict[str, Any]:
        """The successful record whose ``key(result)`` is smallest (largest with ``minimize=False``)."""
        if not self.ok:
            raise ValueError("no configuration succeeded")
        return (min if minimize else max)(self.ok, key=lambda r: key(r["result"]))

    def table(self) -> List[Dict[str, Any]]:
        """Flat rows ``{**params, "result": ..., "error": ..., "time_s": ...}``."""
        return [{**r["params"], "result": r["result"], "error": r["error"], "time_s": r["time_s"]} for r in self.records]


def _grid(grid: Union[Mapping[str, Sequence[Any]], Sequence[Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    if isinstance(grid, Mapping):
        names = list(grid)
        return [dict(zip(names, vals)) for vals in itertools.product(*[list(grid[n]) for n in names])]
    return [dict(g) for g in grid]


def sweep(fn: Callable[..., Any], grid: Union[Mapping[str, Sequence[Any]], Sequence[Mapping[str, Any]]],
          workers: Optional[int] = None, *, backend: str = "process", chunksize: int = 1) -> SweepResult:
    """Run ``fn(**params)`` for every configuration: the Cartesian product of a ``{name: values}`` grid, or an explicit list of
    dicts. Failures are recorded (``record["error"]``), never raised, so one bad configuration does not lose the sweep."""
    configs = _grid(grid)
    if not configs:
        raise ValueError("the grid is empty")
    w = min(workers or cpu_count(), len(configs))
    t0 = time.perf_counter()
    out = _dispatch(fn, configs, w, backend, True, True, chunksize)
    records = []
    for cfg, (res, dt) in zip(configs, out):
        failed = isinstance(res, TaskError)
        records.append({"params": cfg, "result": None if failed else res, "error": res if failed else None, "time_s": dt})
    return SweepResult(records, time.perf_counter() - t0, w)


def ensemble(fn: Callable[..., Any], n: int, workers: Optional[int] = None, *, seed: int = 0, backend: str = "process") -> List[Any]:
    """``n`` independent runs ``fn(seed=s_i)`` with statistically independent seeds spawned from ``seed``
    (``numpy.random.SeedSequence``), reproducible whatever the number of workers."""
    seeds = [int(s.generate_state(1)[0]) for s in np.random.SeedSequence(seed).spawn(n)]
    return map(lambda s: fn(seed=s), seeds, workers, backend=backend)


# -- mesh parallelism ---------------------------------------------------------------------
@dataclass
class MeshPart:
    """One part of a partitioned mesh: the local mesh (owned cells plus a halo layer), the global ids of its nodes and cells,
    and which local nodes this part owns (each global node is owned by exactly one part)."""
    index: int
    mesh: Any
    node_ids: np.ndarray            # local node -> global node
    cell_ids: np.ndarray            # local cell -> global cell
    owned_nodes: np.ndarray         # local indices of the nodes this part owns
    owned_cells: np.ndarray         # local indices of the cells this part owns


def partition_mesh(mesh, n_parts: int) -> List[MeshPart]:
    """Split ``mesh`` into ``n_parts`` by recursive coordinate bisection of the cell centres, adding every cell that shares a node
    with an owned cell (one halo layer). A nodal operator that needs the cells around a node (gradient recovery, an integral
    assembled per node) is then exact on the owned nodes of each part."""
    from .mesh import Mesh

    if n_parts < 1 or n_parts > mesh.n_cells:
        raise ValueError("n_parts must be between 1 and the number of cells")
    cent = mesh.cell_centroids
    groups: List[np.ndarray] = [np.arange(mesh.n_cells)]
    while len(groups) < n_parts:
        groups.sort(key=len, reverse=True)
        big = groups.pop(0)
        pts = cent[big]
        axis = int(np.argmax(pts.max(axis=0) - pts.min(axis=0)))
        order = big[np.argsort(pts[:, axis], kind="stable")]
        half = len(order) // 2
        groups += [order[:half], order[half:]]
    groups.sort(key=lambda g: int(g.min()))
    owner_cell = np.empty(mesh.n_cells, dtype=int)
    for k, g in enumerate(groups):
        owner_cell[g] = k
    # a node is owned by the smallest part among the cells that touch it
    owner_node = np.full(mesh.n_points, n_parts, dtype=int)
    for i in range(mesh.dim + 1):
        np.minimum.at(owner_node, mesh.cells[:, i], owner_cell)
    parts: List[MeshPart] = []
    for k, g in enumerate(groups):
        touched = np.unique(mesh.cells[g])
        halo = np.flatnonzero(np.isin(mesh.cells, touched).any(axis=1))
        nodes = np.unique(mesh.cells[halo])
        remap = -np.ones(mesh.n_points, dtype=int)
        remap[nodes] = np.arange(nodes.size)
        local = Mesh(mesh.points[nodes], remap[mesh.cells[halo]])
        owned_nodes = np.flatnonzero(owner_node[nodes] == k)
        owned_cells = np.flatnonzero(owner_cell[halo] == k)
        parts.append(MeshPart(k, local, nodes, halo, owned_nodes, owned_cells))
    return parts


def map_mesh(fn: Callable[[Any, np.ndarray], np.ndarray], mesh, values: np.ndarray, n_parts: int,
             workers: Optional[int] = None, *, backend: str = "process") -> np.ndarray:
    """Apply a nodal operator in parallel. ``fn(local_mesh, local_values) -> (n_local_nodes, ...)`` runs on every part (with its
    halo); the rows of the owned nodes are gathered into one global array of shape ``(N, ...)``."""
    values = np.asarray(values)
    parts = partition_mesh(mesh, n_parts)
    results = map(lambda p: np.asarray(fn(p.mesh, values[p.node_ids])), parts, workers, backend=backend)
    first = results[0]
    out = np.empty((mesh.n_points,) + first.shape[1:], dtype=first.dtype)
    for p, r in zip(parts, results):
        out[p.node_ids[p.owned_nodes]] = r[p.owned_nodes]
    return out


# -- domain decomposition: additive Schwarz PINNs -------------------------------------------------
@dataclass
class Subdomain:
    """A box ``[lo, hi]`` that overlaps its neighbours; ``core`` is the part it owns when predicting."""
    index: int
    lo: np.ndarray
    hi: np.ndarray
    core_lo: np.ndarray
    core_hi: np.ndarray
    left: Optional[int] = None       # neighbour index across the low face of the cut axis
    right: Optional[int] = None


def decompose_box(lo: Sequence[float], hi: Sequence[float], parts: int, overlap: float = 0.1, axis: int = 0) -> List[Subdomain]:
    """Cut the box into ``parts`` strips along ``axis``, each extended by ``overlap`` (a fraction of the strip width) into its
    neighbours. Overlap is what lets additive Schwarz converge."""
    lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
    if parts < 1 or not 0 <= overlap < 1:
        raise ValueError("parts must be >= 1 and 0 <= overlap < 1")
    edges = np.linspace(lo[axis], hi[axis], parts + 1)
    width = edges[1] - edges[0]
    subs = []
    for k in range(parts):
        slo, shi = lo.copy(), hi.copy()
        clo, chi = lo.copy(), hi.copy()
        clo[axis], chi[axis] = edges[k], edges[k + 1]
        slo[axis] = max(edges[k] - (overlap * width if k > 0 else 0.0), lo[axis])
        shi[axis] = min(edges[k + 1] + (overlap * width if k < parts - 1 else 0.0), hi[axis])
        subs.append(Subdomain(k, slo, shi, clo, chi, k - 1 if k > 0 else None, k + 1 if k < parts - 1 else None))
    return subs


@dataclass
class DecomposedResult:
    subdomains: List[Subdomain]
    states: List[Dict[str, Any]]
    make_model: Callable[[], Any]
    axis: int
    history: List[Dict[str, float]]
    wall_time_s: float

    def _model(self, k: int):
        m = self.make_model()
        m.load_state_dict(self.states[k])
        return m.eval()

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Each point is evaluated by the subdomain that owns it (its core)."""
        import torch

        x = np.asarray(x, dtype=np.float64)
        out: Optional[np.ndarray] = None
        for k, s in enumerate(self.subdomains):
            last = k == len(self.subdomains) - 1
            m = (x[:, self.axis] >= s.core_lo[self.axis]) & ((x[:, self.axis] <= s.core_hi[self.axis]) if last else (x[:, self.axis] < s.core_hi[self.axis]))
            if not m.any():
                continue
            with torch.no_grad():
                y = self._model(k)(torch.as_tensor(x[m], dtype=torch.get_default_dtype())).numpy()
            if out is None:
                out = np.zeros((len(x), y.shape[1]))
            out[m] = y
        return out


def _train_subdomain(job: Dict[str, Any]) -> Dict[str, Any]:
    """Train one subdomain network for ``steps`` Adam steps (runs in a worker)."""
    import torch

    torch.manual_seed(job["seed"])
    torch.set_num_threads(1)
    model = job["make_model"]()
    if job["state"] is not None:
        model.load_state_dict(job["state"])
    sub, residual, boundary = job["sub"], job["residual"], job["boundary"]
    neighbour_fns = job["neighbour_fns"]       # {"left": callable or None, "right": ...}: previous-iteration neighbour solutions
    dt = torch.get_default_dtype()
    rng = np.random.default_rng(job["seed"])
    opt = torch.optim.Adam(model.parameters(), lr=job["lr"])
    axis, d = job["axis"], len(sub.lo)

    def uniform(n):
        return torch.as_tensor(sub.lo + rng.random((n, d)) * (sub.hi - sub.lo), dtype=dt)

    def face(n, value, ax):
        p = sub.lo + rng.random((n, d)) * (sub.hi - sub.lo)
        p[:, ax] = value
        return torch.as_tensor(p, dtype=dt)

    # fixed boundary sets for this iteration: true boundary faces use the BC, interface faces use the neighbour's solution
    sets = []
    for ax in range(d):
        for side, val in ((0, sub.lo[ax]), (1, sub.hi[ax])):
            if ax == axis and ((side == 0 and sub.left is not None) or (side == 1 and sub.right is not None)):
                fn = neighbour_fns["left" if side == 0 else "right"]
                pts = face(job["n_bc"], val, ax)
                sets.append((pts, None if fn is None else torch.as_tensor(fn(pts.numpy()), dtype=dt)))
            else:
                pts = face(job["n_bc"], val, ax)
                sets.append((pts, torch.as_tensor(boundary(pts.numpy()), dtype=dt).reshape(-1, 1)))
    for _ in range(job["steps"]):
        x = uniform(job["n_col"]).requires_grad_(True)
        r = residual(model, x)
        loss = r.pow(2).mean()
        for pts, tgt in sets:
            if tgt is not None:
                loss = loss + job["w_bc"] * (model(pts) - tgt).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def train_decomposed(residual: Callable, boundary: Callable[[np.ndarray], np.ndarray], make_model: Callable[[], Any],
                     lo: Sequence[float], hi: Sequence[float], parts: int = 4, overlap: float = 0.25, iterations: int = 6,
                     steps: int = 400, workers: Optional[int] = None, *, n_col: int = 256, n_bc: int = 32, lr: float = 3e-3,
                     w_bc: float = 20.0, axis: int = 0, seed: int = 0, backend: str = "process",
                     reference: Optional[Callable[[np.ndarray], np.ndarray]] = None) -> DecomposedResult:
    """Train a PINN on a box by additive Schwarz iteration.

    Each iteration trains every subdomain network for ``steps`` Adam steps, all subdomains concurrently, against the PDE residual
    inside it, the boundary condition ``boundary(X) -> values`` on the faces that lie on the true boundary, and the neighbours'
    previous-iteration predictions on the interface faces. ``residual(model, x) -> tensor`` must differentiate ``model(x)`` with
    respect to ``x`` (``x.requires_grad`` is set). With ``reference`` the history records the relative L2 error after every iteration."""
    t0 = time.perf_counter()
    subs = decompose_box(lo, hi, parts, overlap, axis)
    states: List[Optional[Dict[str, Any]]] = [None] * parts
    history: List[Dict[str, float]] = []
    probe = np.random.default_rng(1).uniform(np.asarray(lo, dtype=float), np.asarray(hi, dtype=float), size=(2000, len(lo)))
    for it in range(iterations):
        prev = DecomposedResult(subs, [s for s in states], make_model, axis, [], 0.0) if it > 0 else None

        def neighbour(k: int) -> Dict[str, Optional[Callable]]:
            out: Dict[str, Optional[Callable]] = {"left": None, "right": None}
            if prev is None:
                return out
            for name, j in (("left", subs[k].left), ("right", subs[k].right)):
                if j is not None:
                    out[name] = _make_predictor(make_model, states[j])
            return out

        jobs = [{"sub": subs[k], "residual": residual, "boundary": boundary, "make_model": make_model, "state": states[k],
                 "neighbour_fns": neighbour(k), "steps": steps, "n_col": n_col, "n_bc": n_bc, "lr": lr, "w_bc": w_bc,
                 "axis": axis, "seed": seed + 1000 * it + k} for k in range(parts)]
        states = map(_train_subdomain, jobs, workers or parts, backend=backend)
        rec: Dict[str, float] = {"iteration": it + 1}
        if reference is not None:
            cur = DecomposedResult(subs, states, make_model, axis, [], 0.0)
            ref = np.asarray(reference(probe)).reshape(len(probe), -1)
            rec["rel_l2"] = float(np.linalg.norm(cur.predict(probe) - ref) / np.linalg.norm(ref))
        history.append(rec)
    return DecomposedResult(subs, states, make_model, axis, history, time.perf_counter() - t0)


def _make_predictor(make_model: Callable[[], Any], state: Dict[str, Any]) -> Callable[[np.ndarray], np.ndarray]:
    def predict(x: np.ndarray) -> np.ndarray:
        import torch

        m = make_model()
        m.load_state_dict(state)
        with torch.no_grad():
            return m.eval()(torch.as_tensor(x, dtype=torch.get_default_dtype())).numpy()
    return predict
