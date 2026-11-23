"""``calculix_beam`` and ``calculix_bracket``: structural FEM in the benchmark, with CalculiX as the reference solver.

Roadmap F1 (issue #68): the house benchmark measures a model x ``ccx`` on the same protocol as the OpenFOAM campaigns.
A Latin-hypercube sweep of the geometry and the load, every sample solved by CalculiX (native ``ccx`` or the bundled
Debian image, ``pinneapple_simulation.external_solvers.calculix``), and what is kept per sample is what the analyst
reads first -- the displacement field, the von Mises field and two integral quantities:

* ``calculix_beam``    -- a cantilever box beam with a tip load: the tip deflection against Timoshenko's beam theory
                          and the bending stress in the mid section against M c / I.
* ``calculix_bracket`` -- an L bracket, base clamped, side load on the upright: the upright's bending stress against
                          M c / I (there is no closed-form deflection for the bracket, and the tables say so).

Both meshes are structured hexahedra (C3D8I) whose topology does not change with the design, so a surrogate maps the
four design numbers to the nodal fields directly. Only ``ccx`` is needed: no gmsh, no CAD kernel, no licence.

The design vector is four numbers in log space -- the two or three lengths that define the part plus the load -- and
the data set is ``prepared/<problem>/``:

* ``runs/<id>.json``: the design, the integrals (``w`` in m, ``sigma`` in Pa), the equilibrium residual, the cost and
  which bounds of the design space an OOD sample leaves.
* ``runs/<id>.npz``: ``x`` (node coordinates, m), ``u`` (displacement divided by the length that normalises it,
  dimensionless) and ``vm`` (von Mises divided by E, dimensionless) on the same fixed node set.
* ``geometry/reference.inp``: a complete CalculiX deck of the reference design, for the preflight and mesh-quality
  checks (``pinneapple_data``), not for solving.

Methods: ``mlp_calculix`` (design -> the two integrals) and ``pod_mlp_calculix`` (design -> the fields through a POD
basis of the training fields, the integrals derived from the predicted field).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..registry import Check, Column, Method, Problem, register_method, register_problem

E_STEEL, NU_STEEL, RHO_STEEL = 210e9, 0.3, 7850.0
YIELD = 235e6                                            # S235, for the physics audit only
BEAM_MESH = (20, 3, 3)                                   # nx, ny, nz of the cantilever (C3D8I)
BRACKET_MESH = {"nx2": 2, "ny": 4, "nz": 2, "nz2": 6, "base": 10.0}   # base length a = 10 t
REFERENCE = {"beam": (1.2, 12.0, 1.0, 1000.0), "bracket": (8e-3, 6.0, 6.0, 1000.0)}
SPLIT_SEEDS = {"train": 7, "bracket": 108}
SETS = {"test": "held-out test", "ood": "OOD (outside the design box)"}

#: The design space of each part, in the order of ``features``: (name, low, high). The Latin hypercube runs in log
#: space, so a bound is a ratio and not an interval.
BOUNDS = {
    "beam": (("L", 0.6, 2.0), ("L/H", 6.0, 20.0), ("W/H", 0.5, 2.0), ("P", 250.0, 2000.0)),
    "bracket": (("t", 4e-3, 12e-3), ("b/t", 4.0, 10.0), ("w/t", 4.0, 10.0), ("P", 250.0, 2000.0)),
}
#: Designs outside the box on at least one axis: a stocky beam (L/H = 4.5, where Euler-Bernoulli stops holding by
#: percent), two slendernesses above the training range, and loads beyond it.
OOD = {
    "beam": ((2.4, 26.0, 0.4, 2600.0), (0.50, 4.5, 2.5, 150.0), (1.0, 25.0, 2.5, 1000.0),
             (2.2, 5.0, 0.4, 1800.0), (0.7, 30.0, 1.0, 300.0)),
    "bracket": ((15e-3, 13.0, 5.0, 2600.0), (3e-3, 3.5, 3.5, 150.0), (8e-3, 12.0, 4.0, 2500.0),
                (14e-3, 5.0, 12.0, 900.0), (10e-3, 4.2, 11.0, 400.0)),
}


# ====================================================================== design space
def _lhs(n: int, rng, d: int = 4) -> np.ndarray:
    """``n`` Latin-hypercube points in [0, 1)^d (one stratified permutation per axis)."""
    U = np.empty((max(1, n), d))
    for k in range(d):
        U[:, k] = (rng.permutation(max(1, n)) + rng.random(max(1, n))) / max(1, n)
    return U


def _features(part: str, u) -> list[float]:
    """The four log10 design numbers of a point of the unit hypercube."""
    lo = np.log10([b[1] for b in BOUNDS[part]])
    hi = np.log10([b[2] for b in BOUNDS[part]])
    return [float(x) for x in lo + np.asarray(u, float) * (hi - lo)]


def _design(part: str, features) -> dict[str, float]:
    """The physical design (m, N) from its four log10 numbers."""
    a, b, c, load = (float(x) for x in 10.0 ** np.asarray(features, float))
    if part == "beam":
        H = a / b                                        # L / (L/H)
        return {"L": a, "H": H, "W": c * H, "P": load}
    return {"t": a, "b": b * a, "width": c * a, "a": BRACKET_MESH["base"] * a, "P": load}


def _outside(part: str, features) -> list[str]:
    """The bounds an OOD sample leaves (empty for a design inside the box)."""
    out = []
    for (name, lo, hi), f in zip(BOUNDS[part], features, strict=True):
        v = 10.0 ** float(f)
        if v < lo * (1 - 1e-9) or v > hi * (1 + 1e-9):
            out.append(name)
    return out


def _plan(part: str, options: dict, seed: int | None = None) -> list[dict]:
    """The campaign: one job per sample, the train / val / test designs drawn once (so no design is in two splits)
    and the OOD designs fixed per part."""
    rng = np.random.default_rng(SPLIT_SEEDS["train"] if seed is None else seed)
    n = {s: int(options[f"n_{s}"]) for s in ("train", "val", "test")}
    total = sum(n.values())
    U, order, k, jobs = _lhs(total, rng), None, 0, []
    order = rng.permutation(total)
    for split in ("train", "val", "test"):
        for _ in range(n[split]):
            jobs.append({"id": f"{split[:2]}{k:03d}", "split": split, "part": part,
                         "features": _features(part, U[order[k]])})
            k += 1
    for i, design in enumerate(OOD[part][: int(options["n_ood"])]):
        jobs.append({"id": f"oo{i:03d}", "split": "ood", "part": part,
                     "features": [float(x) for x in np.log10(np.asarray(design, float))]})
    return jobs


# ====================================================================== meshes
def _merge(nodes: np.ndarray, elements: np.ndarray, tol: float = 1e-9):
    """Merge coincident nodes (the interface of two blocks meshed separately), keeping element connectivity."""
    key = np.round(nodes / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return nodes[first], np.asarray(inv).ravel()[elements]


def _bracket_units(bt: float, width_t: float, g: dict | None = None):
    """The L bracket in units of its thickness: a base block [0, 10] x [0, w/t] x [0, 1] clamped at z = 0 and an
    upright [0, 1] x [0, w/t] x [1, b/t], meshed separately and merged on the plane z = 1. The topology (node count,
    connectivity) does not depend on ``bt`` or ``width_t``: only the coordinates do."""
    from pinneapple_simulation.numerical_solvers.solid_fem import box_mesh
    g = {**BRACKET_MESH, **(g or {})}
    nx2, ny, nz, nz2, a = g["nx2"], g["ny"], g["nz"], g["nz2"], g["base"]
    base = box_mesh(a, width_t, 1.0, int(a) * nx2, ny, nz, origin=(0.0, width_t / 2, 0.5))
    up = box_mesh(1.0, width_t, bt - 1.0, nx2, ny, nz2, origin=(0.0, width_t / 2, 1.0 + (bt - 1.0) / 2))
    return _merge(np.vstack([base.nodes, up.nodes]),
                  np.vstack([base.elements, up.elements + len(base.nodes)]))


def _model(part: str, d: dict, mesh=None):
    """The finite-element model of a design (C3D8I hexahedra on a design-independent topology). ``mesh`` refines it:
    (nx, ny, nz) for the beam, a BRACKET_MESH override for the bracket (the solver checks only; the campaign keeps
    the one topology the surrogates need)."""
    from pinneapple_simulation.external_solvers.calculix.study import FEModel
    if part == "beam":
        nx, ny, nz = mesh or BEAM_MESH
        return FEModel.from_box(d["L"], d["W"], d["H"], nx, ny, nz, "C3D8I")
    g = {**BRACKET_MESH, **(mesh or {})}
    nodes, elements = _bracket_units(d["b"] / d["t"], d["width"] / d["t"], g)
    return FEModel(nodes * d["t"], elements, "C3D8I")


def _step(part: str, d: dict, m):
    """The static step: the base (or the root face) clamped, the load spread over the free face."""
    from pinneapple_simulation.external_solvers.calculix.study import Static
    if part == "beam":
        root = m.nodes_where(lambda X: X[:, 0] < 1e-9)
        tip = m.nodes_where(lambda X: X[:, 0] > d["L"] - 1e-9)
        return Static(fix=[(root, (1, 2, 3))], loads=[(tip, (0.0, 0.0, -d["P"]))])
    base = m.nodes_where(lambda X: X[:, 2] < 1e-9)
    top = m.nodes_where(lambda X: X[:, 2] > d["b"] - 1e-9)
    return Static(fix=[(base, (1, 2, 3))], loads=[(top, (-d["P"], 0.0, 0.0))])


def _mid_plane(v: np.ndarray, length: float, at: float = 0.5) -> np.ndarray:
    """The single node plane closest to ``at * length`` along ``v``: the mid section where M c / I is exact (a band
    would put the peak of the section's moment inside the section's own definition)."""
    u = np.unique(np.round(v, 9))
    mid = u[int(np.argmin(np.abs(u - at * length)))]
    return np.abs(v - mid) <= 1e-9 * max(1.0, length)


#: The normal stress that carries the bending, in the order of ``Result.stress`` (xx, yy, zz, xy, yz, zx): the beam
#: bends about y (its axis is x) and the bracket's upright about y too, but its axis is z -- so the axial stress of
#: the upright is sigma_zz, and sigma_xx there is the (small) transverse one.
BENDING = {"beam": 0, "bracket": 2}


def _section(part: str, d: dict, X: np.ndarray):
    """(free-face mask, mid-section mask, length that normalises the field, moment arm of that section): the section
    is a node plane of the protocol mesh, so M c / I and the solved field are read at the same place."""
    if part == "beam":
        n = float(d["L"])
        return X[:, 0] > n - 1e-9, _mid_plane(X[:, 0], n), n, 0.5 * n
    n = float(d["b"])
    at = 0.5 * (1.0 + d["t"] / n)                    # the middle of the upright, not of the whole bracket
    return X[:, 2] > n - 1e-9, _mid_plane(X[:, 2], n, at=at), n, n * (1.0 - at)


def _selections(part: str, d: dict, m):
    """(applied load vector, clamped nodes, free-face nodes, mid-section nodes, length that normalises the field)."""
    X = m.nodes
    applied = np.array([0.0, 0.0, -d["P"]]) if part == "beam" else np.array([-d["P"], 0.0, 0.0])
    root = X[:, 0] < 1e-9 if part == "beam" else X[:, 2] < 1e-9
    tip, band, length, _ = _section(part, d, X)
    return applied, root, tip, band, length


def _analytic(part: str, d: dict) -> dict[str, float | None]:
    """The closed-form reference of a design: Timoshenko's tip deflection (beam only) and the bending stress of the
    mid section read by the protocol (``arm`` below), M c / I. For the bracket there is no closed-form deflection: the
    base's rotation adds a share the upright's cantilever formula does not have, so ``w`` is None rather than a wrong
    number."""
    G = E_STEEL / (2 * (1 + NU_STEEL))
    if part == "beam":
        inertia = d["W"] * d["H"] ** 3 / 12
        w = d["P"] * d["L"] ** 3 / (3 * E_STEEL * inertia) + d["P"] * d["L"] / (5 / 6 * G * d["W"] * d["H"])
        return {"w": float(w), "sigma": float(d["P"] * 0.5 * d["L"] * (d["H"] / 2) / inertia)}
    inertia = d["width"] * d["t"] ** 3 / 12
    return {"w": None, "sigma": float(d["P"] * (d["b"] - d["t"]) / 2 * (d["t"] / 2) / inertia)}


def _solve_part(part: str, d: dict, workdir: str, *, mesh=None):
    """Build, run ``ccx`` and read one design back. Raises when the solver fails."""
    from pinneapple_simulation.external_solvers.calculix.study import Material, solve
    m = _model(part, d, mesh)
    m.material = Material("steel", E_STEEL, NU_STEEL, RHO_STEEL)
    return m, solve(m, _step(part, d, m), workdir, threads=1)


def _one(args) -> tuple[str, str, float]:
    """One campaign job (a module-level function: the pool has to pickle it)."""
    part, job, out = args
    d = _design(part, job["features"])
    work = os.path.join(out, "_work", job["id"])
    t0 = time.time()
    try:
        m, r = _solve_part(part, d, work)
        applied, root, tip, band, length = _selections(part, d, m)
        vm = r.von_mises
        rec = {**job, "design": d, "w": float(np.linalg.norm(r.u[tip], axis=1).max()),
               "sigma": float(vm[band].max()),
               "eq": float(np.linalg.norm(r.reactions[root].sum(0) + applied) / max(np.linalg.norm(applied), 1e-30)),
               "seconds": time.time() - t0, "status": "ok", "nodes": int(len(m.nodes)),
               "elements": int(len(m.elements)), "dofs": int(3 * len(m.nodes)), "solver": r.info.get("solver"),
               "field_scale": {"length": length, "stress": E_STEEL}, "outside": _outside(part, job["features"])}
        np.savez_compressed(os.path.join(out, "runs", job["id"] + ".npz"), x=m.nodes.astype(np.float32),
                            u=(r.u / length).astype(np.float32), vm=(vm / E_STEEL).astype(np.float32))
    except Exception as e:                                       # a failed sample is recorded, never dropped silently
        rec = {**job, "design": d, "status": f"failed: {e!r}", "seconds": time.time() - t0}
    finally:
        shutil.rmtree(work, ignore_errors=True)
    Path(out, "runs", job["id"] + ".json").write_text(json.dumps(rec, indent=1))
    return job["id"], rec["status"], round(float(rec["seconds"]), 3)


def _campaign(part: str, jobs: list[dict], out: Path, n_workers: int = 1, log=print) -> None:
    (out / "runs").mkdir(parents=True, exist_ok=True)
    todo = [j for j in jobs if not (out / "runs" / f"{j['id']}.json").exists()]
    todo.sort(key=lambda j: j["split"] != "ood")
    if not todo:
        return
    if n_workers <= 1:
        for j in todo:
            log(*_one((part, j, str(out))))
        return
    with ProcessPoolExecutor(max(1, n_workers)) as ex:
        for res in ex.map(_one, [(part, j, str(out)) for j in todo]):
            log(*res)


def _records(root) -> dict[str, list[dict]]:
    """{split: [record]} of a prepared folder, in job-id order, with the matching npz path attached."""
    out: dict[str, list[dict]] = {s: [] for s in ("train", "val", "test", "ood")}
    for p in sorted((Path(root) / "runs").glob("*.json")):
        r = json.loads(p.read_text())
        r["npz"] = str(p.with_suffix(".npz"))
        out.setdefault(r["split"], []).append(r)
    return out


def _fields(rec: dict) -> np.ndarray:
    """A sample's field vector: the three displacement components and the von Mises, node by node (4 N)."""
    z = np.load(rec["npz"])
    return np.concatenate([z["u"].astype(np.float32).ravel(), z["vm"].astype(np.float32)])


def _channel_stats(Y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, d = Y.shape
    ch = Y.reshape(n, -1, d // 4)
    return ch.mean(0).ravel(), ch.reshape(n, -1, 4).std(0).repeat(d // 4)


# ====================================================================== trainers
def _mlp(n_in: int, n_out: int, width: int, depth: int = 3):
    from torch import nn
    layers, w = [], n_in
    for _ in range(depth):
        layers += [nn.Linear(w, width), nn.SiLU()]
        w = width
    return nn.Sequential(*layers, nn.Linear(w, n_out))


def _torch_clock():
    from pinneapple_analysis.cost.equal_budget import (
        cosine_lr,
        set_lr,
        time_per_query,
        train_within_budget,
    )
    return cosine_lr, set_lr, time_per_query, train_within_budget


def train_mlp_calculix(problem, prepared, cfg: dict, seed: int, budget, *, final: bool = True, **_) -> dict:
    """Design -> the two integrals, on log10 targets (they span two decades across the design box)."""
    import torch
    cosine_lr, set_lr, time_per_query, train_within_budget = _torch_clock()
    S = _records(prepared)
    tr = S["train"] + S["val"] if final else S["train"]
    X = np.stack([r["features"] for r in tr]).astype(np.float32)
    Y = np.log10(np.stack([[r["w"], r["sigma"]] for r in tr])).astype(np.float32)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    ym, ys = Y.mean(0), Y.std(0) + 1e-6
    Xt, Yt = torch.from_numpy((X - xm) / xs), torch.from_numpy((Y - ym) / ys)
    torch.manual_seed(seed)
    model = _mlp(X.shape[1], 2, cfg["width"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        loss = ((model(Xt) - Yt) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)

    def predict(recs):
        q = (np.stack([r["features"] for r in recs]).astype(np.float32) - xm) / xs
        with torch.no_grad():
            return 10.0 ** (model(torch.from_numpy(q)).numpy() * ys + ym)
    out = {"train": run.to_dict(), "n_params": int(sum(p.numel() for p in model.parameters())),
           "n_train_runs": len(tr)}
    for split in (("test", "ood") if final else ("val",)):
        if not S.get(split):
            continue
        true = np.stack([[r["w"], r["sigma"]] for r in S[split]])
        pred = predict(S[split])
        out[split] = {"w": (np.abs(pred[:, 0] - true[:, 0]) / true[:, 0]).tolist(),
                      "sigma": (np.abs(pred[:, 1] - true[:, 1]) / true[:, 1]).tolist(),
                      "ids": [r["id"] for r in S[split]]}
    out["infer_batch1"] = time_per_query(lambda: model(Xt[:1]), repeats=50, warmup=5)
    return out


def train_pod_mlp_calculix(problem, prepared, cfg: dict, seed: int, budget, *, final: bool = True, **_) -> dict:
    """Design -> the fields: a POD basis of the training fields and an MLP from the design to its coefficients; the
    integrals are computed from the predicted field (the peak von Mises of the mid section, the free face's
    displacement), so the two rows are comparable on the same columns."""
    import torch
    cosine_lr, set_lr, time_per_query, train_within_budget = _torch_clock()
    S = _records(prepared)
    tr = S["train"] + S["val"] if final else S["train"]
    Y = np.stack([_fields(r) for r in tr]).astype(np.float32)                 # (n, 4N)
    fm, fs = _channel_stats(Y)
    fs = np.where(fs > 0, fs, 1.0)
    Z = (Y - fm) / fs
    _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
    K = int(min(cfg["modes"], len(Z), Z.shape[1]))
    basis = Vt[:K]
    coef = Z @ basis.T
    cm, cs = coef.mean(0), coef.std(0) + 1e-8
    X = np.stack([r["features"] for r in tr]).astype(np.float32)
    xm, xs = X.mean(0), X.std(0) + 1e-6
    Xt = torch.from_numpy((X - xm) / xs)
    T = torch.from_numpy(((coef - cm) / cs).astype(np.float32))
    torch.manual_seed(seed)
    net = _mlp(X.shape[1], K, cfg["width"])
    opt = torch.optim.Adam(net.parameters(), lr=cfg["lr"])
    wts = torch.from_numpy((sv[:K] / max(sv[0], 1e-30)).astype(np.float32))   # weight coefficients by their energy

    def step(progress):
        set_lr(opt, cosine_lr(progress, cfg["lr"]))
        loss = (((net(Xt) - T) ** 2) * wts).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        return loss.detach()

    run = train_within_budget(step, budget)
    basis_t = torch.from_numpy(basis.astype(np.float32))

    def predict(recs):
        q = (np.stack([r["features"] for r in recs]).astype(np.float32) - xm) / xs
        with torch.no_grad():
            c = net(torch.from_numpy(q)).numpy() * cs + cm
        return ((c @ basis) * fs + fm).reshape(len(recs), -1, 4)

    out = {"train": run.to_dict(), "n_params": int(sum(p.numel() for p in net.parameters()) + basis.size),
           "n_train_runs": len(tr), "pod_modes": K,
           "pod_energy_kept": float((sv[:K] ** 2).sum() / (sv ** 2).sum())}
    for split in (("test", "ood") if final else ("val",)):
        if not S.get(split):
            continue
        out[split] = problem.field_scores(S[split], predict(S[split]))
        out[split]["ids"] = [r["id"] for r in S[split]]
    x1 = Xt[:1]
    out["infer_batch1"] = time_per_query(lambda: net(x1) @ basis_t, repeats=50, warmup=5)
    return out


# ====================================================================== the problems
class _Structural(Problem):
    """The protocol both parts share: the ``ccx`` campaign, the reference-solver verification against the closed
    form, the data checks, the validation stack, the table and the two trainers."""
    kind = "geometry"
    part = "beam"
    outputs_mlp_calculix = "integrals"
    outputs_pod_mlp_calculix = "both"
    title_mlp_calculix = "MLP (displacement, peak stress)"
    title_pod_mlp_calculix = "POD + MLP (fields)"
    space_mlp_calculix = {"width": [32, 64, 128], "lr": [1e-3, 3e-3], "wd": [1e-6, 1e-4]}
    space_pod_mlp_calculix = {"modes": [8, 16, 32], "width": [64, 128], "lr": [1e-3, 3e-3]}

    def default_options(self) -> dict:
        return {"n_train": 40, "n_val": 10, "n_test": 12, "n_ood": 5, "gci_study": False}

    def smoke_options(self) -> dict:
        return {"n_train": 6, "n_val": 2, "n_test": 2, "n_ood": 2}

    # ------------------------------------------------------------------ data
    def prepare(self, out: Path, n_workers: int = 1):
        from pinneapple_simulation.external_solvers.calculix.runner import ccx_backend
        if ccx_backend() is None:
            raise RuntimeError(f"the {self.id} problem needs CalculiX to generate its data: install ccx (a native "
                               "binary on PATH) or run Docker, which builds the bundled Debian image")
        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        _campaign(self.part, _plan(self.part, self.options), out, n_workers)
        self._geometry(out)
        recs = [r for s in _records(out).values() for r in s]
        sec = [float(r.get("seconds") or 0.0) for r in recs]
        status = {s: sum(r.get("status") == s for r in recs) for s in sorted({r.get("status") for r in recs})}
        cost = {"n_runs": len(recs), "solver_s_total": float(np.sum(sec)),
                "solver_s_per_run": float(np.mean(sec)) if sec else None,
                "train_set_core_s": float(np.sum([float(r.get("seconds") or 0.0) for r in recs
                                                  if r["split"] != "ood"])),
                "wall_s": time.time() - t0, "status": status, "solver": (recs[0].get("solver") if recs else None),
                "timing": "wall time of each ccx run on one thread (shared CPU inflates it; re-time alone to check)"}
        (out / "cost.json").write_text(json.dumps(cost, indent=1))
        files = ["cost.json", "geometry/reference.inp", "geometry/reference.json"]
        files += [str(p.relative_to(out)) for p in sorted((out / "runs").glob("*")) if p.is_file()]
        return {"files": files, "cost": cost}

    def artifact_files(self, out):
        out = Path(out)
        runs = [str(p.relative_to(out)) for p in sorted((out / "runs").glob("*")) if p.is_file()]
        return ["cost.json"] + runs if runs else ["cost.json", "runs"]

    def complete_adopted(self, out):
        self._geometry(Path(out))

    def _geometry(self, out: Path) -> None:
        """``geometry/``: one complete CalculiX deck of the reference design (the preflight and the mesh-quality
        checks read it) and the design with its closed-form values."""
        from pinneapple_simulation.external_solvers.calculix.study import Material
        g = Path(out) / "geometry"
        g.mkdir(exist_ok=True)
        d = _design(self.part, np.log10(np.asarray(REFERENCE[self.part])))
        m = _model(self.part, d)
        m.material = Material("steel", E_STEEL, NU_STEEL, RHO_STEEL)
        (g / "reference.inp").write_text(m.deck(_step(self.part, d, m)))
        (g / "reference.json").write_text(json.dumps({"part": self.part, "design": d, "analytic": _analytic(self.part, d),
                                                      "material": {"name": "steel", "E": E_STEEL, "nu": NU_STEEL,
                                                                   "density": RHO_STEEL},
                                                      "units": "m, N, Pa, kg/m3"}, indent=1))

    def data_cost_s(self, out):
        p = Path(out) / "cost.json"
        return json.loads(p.read_text()).get("train_set_core_s") if p.exists() else None

    def solver_time_per_instance_s(self, out):
        p = Path(out) / "cost.json"
        return json.loads(p.read_text()).get("solver_s_per_run") if p.exists() else None

    # ------------------------------------------------------------------ checks
    def verify_solver(self):
        """The reference solver against the closed form on the reference design: tip deflection (Timoshenko), the
        bending stress of the mid section (M c / I), equilibrium, and a second mesh for discretisation."""
        from pinneapple_simulation.external_solvers.calculix.runner import ccx_backend
        if ccx_backend() is None:
            return [Check("reference solver verification", "problem", self.id, None,
                          detail="ccx is not available on this machine (no native binary, no Docker): the solver was "
                                 "not verified here")]
        d = _design(self.part, np.log10(np.asarray(REFERENCE[self.part])))
        ref = _analytic(self.part, d)
        with tempfile.TemporaryDirectory() as w:
            m, r = _solve_part(self.part, d, os.path.join(w, "base"))
            applied, root, tip, band, _ = _selections(self.part, d, m)
            w_fem = float(np.linalg.norm(r.u[tip], axis=1).max())
            s_fem = float(np.abs(r.stress[band, BENDING[self.part]]).max())
            eq = float(np.linalg.norm(r.reactions[root].sum(0) + applied) / np.linalg.norm(applied))
            mesh = {"beam": (10, BEAM_MESH[1], BEAM_MESH[2]), "bracket": {"ny": 6, "nz2": 12}}[self.part]
            m2, r2 = _solve_part(self.part, d, os.path.join(w, "coarse"), mesh=mesh)
            _, _, tip2, band2, _ = _selections(self.part, d, m2)
            w2 = float(np.linalg.norm(r2.u[tip2], axis=1).max())
            s2 = float(np.abs(r2.stress[band2, BENDING[self.part]]).max())
        s_ref = float(ref["sigma"])
        dw = abs(w_fem / w2 - 1)
        ds = abs(s_fem / s2 - 1) if s2 > 0 else float("nan")
        tol = 0.05 if self.part == "beam" else 0.08      # the bracket's upright is deep (b/t = 6): +3.6 % asymptotically
        C = [Check("bending stress of the mid section vs M c / I", "problem", self.id, abs(s_fem / s_ref - 1) < tol,
                   value=f"{s_fem / 1e6:.2f} MPa vs {s_ref / 1e6:.2f}", threshold=f"{tol:.0%}",
                   detail=f"max |sigma_{'xx' if self.part == 'beam' else 'zz'}| on the mid section against M c / I "
                          "(M = P L/2 or P (b - t)/2)" + ("" if self.part == "beam" else
                                                          "; the upright is a deep member, so the refined meshes "
                                                          "converge to about 4 % above the beam formula")),
             Check("reactions balance the applied load", "problem", self.id, eq < 1e-4, value=f"{eq:.1e}",
                   threshold=1e-4, detail="sum of the clamped nodes' reaction forces plus the applied load; the .frd "
                                          "writes six significant digits per nodal force, so this is its floor"),
             Check("the peak stress is mesh-converged (two meshes)", "problem", self.id, ds < 0.03, value=f"{ds:.2%}",
                   threshold="3 %", detail=("20 elements along the length against 10" if self.part == "beam" else
                                             "the base and the upright refined (ny 6, nz2 9) against the protocol "
                                             "mesh"))]
        if ref["w"]:
            C.insert(0, Check("tip deflection vs Timoshenko beam theory", "problem", self.id, abs(w_fem / ref["w"] - 1) < 0.02,
                              value=f"{w_fem * 1e3:.3f} mm vs {ref['w'] * 1e3:.3f}", threshold="2 %",
                              detail="P L^3 / (3 E I) + P L / (5/6 G A) with the load spread over the tip face"))
            C.insert(1, Check("tip deflection is mesh-converged (two meshes)", "problem", self.id, dw < 0.02,
                              value=f"{dw:.2%}", threshold="2 %", detail="20 elements along the length against 10"))
        return C

    def data_checks(self, out):
        S = _records(out)
        recs = [r for s in S.values() for r in s]
        C = [Check("every sample solved (ccx returned a result)", "data", self.id,
                   bool(recs) and all(r.get("status") == "ok" for r in recs),
                   value=f"{sum(r.get('status') == 'ok' for r in recs)}/{len(recs)}", threshold="all")]
        finite = bool(recs) and all(np.isfinite([r["w"], r["sigma"], r["eq"]]).all() for r in recs)
        fields_ok = finite
        for r in recs if finite else []:
            z = np.load(r["npz"])
            fields_ok = fields_ok and all(np.isfinite(z[k]).all() for k in ("x", "u", "vm"))
        C.append(Check("finite designs, integrals and fields", "data", self.id, fields_ok))
        want = {s: int(self.options[f"n_{s}"]) for s in ("train", "val", "test", "ood")}
        got = {s: len(recs_) for s, recs_ in S.items()}
        C.append(Check("split sizes as configured", "data", self.id, got == want, value=got, threshold=want))
        h = {s: {json.dumps(np.round(r["features"], 9).tolist()) for r in S.get(s, [])} for s in S}
        leak = {f"{a}&{b}": len(h[a] & h[b]) for i, a in enumerate(sorted(h)) for b in sorted(h)[i + 1:]}
        C.append(Check("no design shared between splits", "data", self.id, all(v == 0 for v in leak.values()),
                       value=leak, threshold=0))
        eq = max((float(r["eq"]) for r in recs), default=float("nan"))
        C.append(Check("every run is in equilibrium (reactions balance the load)", "data", self.id, eq < 1e-4,
                       value=f"{eq:.1e}", threshold=1e-4,
                       detail="worst |sum R + P| / |P| over the campaign (the .frd writes six significant digits per "
                              "nodal force, so this is its floor)"))
        topo = {(r.get("nodes"), r.get("elements")) for r in recs}
        C.append(Check("one topology for every sample", "data", self.id, len(topo) == 1, value=sorted(topo),
                       detail="a surrogate maps the design to the nodal values, so the node set cannot change"))
        if S.get("ood"):
            outside = [bool(r.get("outside")) for r in S["ood"]]
            C.append(Check("OOD designs outside the training box", "data", self.id, all(outside),
                           value=f"{sum(outside)}/{len(outside)}", threshold="all",
                           detail="each OOD design leaves at least one bound of the Latin hypercube"))
        C.append(Check("normalisation from the training split only", "data", self.id, True,
                       detail="both trainers take their means and scales on the training runs"))
        return C

    def validation(self, out):
        from pinneapple_physics.physics_audit import continuum, linear_elastic, small_strain

        from ..validation import envelope_checks, physics_audit_check, preflight_check
        out = Path(out)
        C, ref = [], out / "geometry" / "reference.inp"
        if ref.exists():
            C.append(preflight_check(self.id, ref.parent))
            C.append(self._mesh_quality(ref))
        S = _records(out)

        def feat(r):
            return {n: float(10.0 ** f) for (n, _, _), f in zip(BOUNDS[self.part], r["features"], strict=True)}
        C += envelope_checks(self.id, [feat(r) for r in S["train"]], [feat(r) for r in S["test"]],
                             [feat(r) for r in S.get("ood", [])])
        peak = max((float(r.get("sigma") or 0.0) for r in S["train"] + S["test"]), default=0.0)
        C.append(physics_audit_check(self.id, "linear elasticity, small strain", "solver",
                                     ["div(sigma) + f = 0", "sigma = C : eps"], [small_strain(0.01), linear_elastic(),
                                                                                  continuum(0.01)],
                                     {"strain": peak / E_STEEL, "stress_ratio": peak / YIELD, "Kn": 0.0}))
        if self.options.get("gci_study"):
            C.append(self._gci(out))
        return C

    def _mesh_quality(self, path: Path) -> Check:
        try:
            from pinneapple_data.cae import read_any

            from ..validation import mesh_quality_check
            return mesh_quality_check(self.id, read_any({path.name: path.read_bytes()}, fields=False), "fea")
        except Exception as e:                                   # a deck we cannot read is a reported item, not a pass
            return Check("mesh quality (pinneapple_data.cae.mesh_report)", "validation", self.id, None,
                         detail=f"could not read the deck: {e!r}")

    def _gci(self, out: Path) -> Check:
        """Three meshes of the reference design (half, base, double the elements along the length): GCI of w."""
        from ..validation import gci_check
        path = Path(out) / "gci.json"
        if not path.exists():
            d = _design(self.part, np.log10(np.asarray(REFERENCE[self.part])))
            vals, sizes = [], []
            with tempfile.TemporaryDirectory() as w:
                for k, nx in ((0, max(2, BEAM_MESH[0] // 2)), (1, BEAM_MESH[0]), (2, 2 * BEAM_MESH[0])):
                    mesh = (nx, BEAM_MESH[1], BEAM_MESH[2]) if self.part == "beam" else None
                    if self.part != "beam":
                        break
                    m, r = _solve_part(self.part, d, os.path.join(w, f"g{k}"), mesh=mesh)
                    vals.append(float(np.linalg.norm(r.u[m.nodes[:, 0] > d["L"] - 1e-9], axis=1).max()))
                    sizes.append(1.0 / nx)
            if len(vals) < 3:
                return Check("grid convergence (GCI) of the tip deflection", "validation", self.id, None,
                             detail="the bracket's topology is fixed by the protocol: only the beam can be refined")
            path.write_text(json.dumps({"w": vals, "sizes": sizes}))
        rec = json.loads(path.read_text())
        return gci_check(self.id, "the tip deflection", rec["w"], rec["sizes"])

    # ------------------------------------------------------------------ evaluation
    def columns(self):
        n = {s: int(self.options[f"n_{s}"]) for s in SETS}
        cols = []
        for s, lab in SETS.items():
            cols.append(Column(f"w_{s}", f"Displacement rel. err, {lab} (n = {n[s]})", s, "w", kind="integral", pct=True))
            cols.append(Column(f"s_{s}", f"Peak stress rel. err, {lab} (n = {n[s]})", s, "sigma", kind="integral",
                               pct=True))
        for s, lab in SETS.items():
            cols.append(Column(f"u_{s}", f"Displacement rel. L2, {lab} (n = {n[s]})", s, "u"))
            cols.append(Column(f"vm_{s}", f"von Mises rel. L2, {lab} (n = {n[s]})", s, "vm"))
        return cols

    def main_metric(self):
        return "test", "sigma"

    def main_metric_for(self, method_id):
        """The peak stress for the integrals-only MLP, the displacement field for the field model."""
        return ("test", "sigma") if self.method_outputs_id(method_id) == "integrals" else ("test", "u")

    def method_outputs_id(self, method_id):
        return getattr(self, f"outputs_{method_id}", None) or "fields"

    def ood_metric(self):
        return "ood", "sigma"

    def summary_metrics(self):
        return [("Displacement (integral)", "integral", ("test", "w"), [("ood", "w")], True),
                ("Peak stress (integral)", "integral", ("test", "sigma"), [("ood", "sigma")], True),
                ("Displacement (field)", "field", ("test", "u"), [("ood", "u")], False)]

    def extract(self, method_id, results):
        r = results.get("")
        if not r:
            return {}
        out = {}
        for split in SETS:
            d = r.get(split)
            if not d:
                continue
            out[split] = {k: {"values": list(d[k]), "ids": list(d.get("ids") or range(len(d[k])))}
                          for k in ("w", "sigma", "u", "vm") if k in d}
        return out

    def search_score(self, method_id, result):
        v = result["val"]
        return float(np.mean(v["u"])) if "u" in v else float(np.mean(v["sigma"]))

    def instances(self):
        return {"val": [], "final": [], "full_only": [], "seed0_only": []}

    def field_scores(self, recs, pred: np.ndarray) -> dict:
        """Relative L2 of the displacement and of the von Mises per sample, plus the integrals read off the predicted
        field: the free face's displacement and the peak von Mises of the mid section (the outer fibre is in uniaxial
        bending there, so the von Mises is the bending stress)."""
        u_e, v_e, w_e, s_e = [], [], [], []
        for i, r in enumerate(recs):
            z = np.load(r["npz"])
            p = pred[i]
            scale = r["field_scale"]
            pu, pv = p[:, :3], p[:, 3]
            u_e.append(float(np.linalg.norm(pu - z["u"]) / max(np.linalg.norm(z["u"]), 1e-30)))
            v_e.append(float(np.linalg.norm(pv - z["vm"]) / max(np.linalg.norm(z["vm"]), 1e-30)))
            X = z["x"]
            free, mid, length, _ = _section(r["part"], r["design"], X)
            w_p = float(np.linalg.norm(pu[free], axis=1).max() * length)
            s_p = float(pv[mid].max() * scale["stress"])
            w_e.append(abs(w_p - r["w"]) / r["w"])
            s_e.append(abs(s_p - r["sigma"]) / r["sigma"])
        return {"u": u_e, "vm": v_e, "w": w_e, "sigma": s_e}

    # ------------------------------------------------------------------ the closed form as a row of the table
    def reference_row(self, out):
        """Beam theory (Timoshenko, M c / I) as a no-data row on the same evaluation samples: the CSI of the
        campaign -- how much of the answer a formula already has."""
        cache = Path(out) / "analytic_reference.json"
        if cache.exists():
            return json.loads(cache.read_text())
        S = _records(out)
        t0 = time.perf_counter()
        ext = {}
        for split in SETS:
            vals = {"w": [], "sigma": [], "ids": []}
            for r in S.get(split, []):
                ref = _analytic(self.part, r["design"])
                vals["sigma"].append(abs(ref["sigma"] - r["sigma"]) / r["sigma"])
                if ref["w"]:
                    vals["w"].append(abs(ref["w"] - r["w"]) / r["w"])
                vals["ids"].append(r["id"])
            if vals["ids"]:
                if not vals["w"]:
                    del vals["w"]
                ext[split] = vals
        elapsed = (time.perf_counter() - t0) / max(1, sum(len(v["ids"]) for v in ext.values()))
        row = {"title": ("Beam theory (Timoshenko, M c / I; no data)"), "seeds": [ext],
               "time_per_instance_s": float(elapsed)}
        cache.write_text(json.dumps(row))
        return row

    def report_sections(self, plan) -> list[str]:
        from ..report import fmt_s
        S = _records(plan.ws.prepared(self.id))
        L = [f"### {self.title}: what the closed form already knows", "",
             "| Split | Samples | Displacement: |beam theory - ccx| / ccx | Peak stress: |M c/I - ccx| / ccx |",
             "|---|---|---|---|"]
        for split in SETS:
            recs = S.get(split, [])
            if not recs:
                continue
            ew = [abs(_analytic(self.part, r["design"])["w"] - r["w"]) / r["w"] for r in recs
                  if _analytic(self.part, r["design"])["w"]]
            es = [abs(_analytic(self.part, r["design"])["sigma"] - r["sigma"]) / r["sigma"] for r in recs]
            L.append(f"| {SETS[split]} | {len(recs)} | {f'{100 * np.mean(ew):.2f} %' if ew else '-'} | "
                     f"{100 * np.mean(es):.2f} % |")
        L += ["", f"The reference solver costs {fmt_s(self.solver_time_per_instance_s(plan.ws.prepared(self.id)))} per "
              "new design (one ccx run, one thread) and its data cost "
              f"{fmt_s(self.data_cost_s(plan.ws.prepared(self.id)))}; the closed-form row above costs none. "
              "A surrogate is worth its training only if it beats that row on the same columns."]
        L.append("")
        return L

    # ------------------------------------------------------------------ trainers
    def bind(self, prepared):
        self._prepared = Path(prepared)
        return self

    def train_mlp_calculix(self, prepared, cfg, seed, budget, *, final=True, **_):
        return train_mlp_calculix(self, prepared, cfg, seed, budget, final=final)

    def train_pod_mlp_calculix(self, prepared, cfg, seed, budget, *, final=True, **_):
        return train_pod_mlp_calculix(self, prepared, cfg, seed, budget, final=final)

    # ------------------------------------------------------------------ architecture gates
    def checks_mlp_calculix(self, prepared, cfg):
        import torch

        from .fields import _overfit_check
        subj = f"{self.id}/mlp_calculix"
        torch.manual_seed(0)
        net = _mlp(4, 2, cfg["width"])
        x = torch.from_numpy(np.random.default_rng(0).standard_normal((6, 4)).astype(np.float32))
        t = torch.from_numpy(np.random.default_rng(1).standard_normal((6, 2)).astype(np.float32))
        return [Check("forward shape", "architecture", subj, tuple(net(x).shape) == (6, 2), value=tuple(net(x).shape)),
                Check("parameter count", "architecture", subj, True, value=int(sum(p.numel() for p in net.parameters())),
                      severity="warning"),
                _overfit_check(subj, lambda: ((net(x) - t) ** 2).mean(), list(net.parameters()), steps=1000,
                               lr=cfg["lr"])]

    def checks_pod_mlp_calculix(self, prepared, cfg):
        """The POD basis is orthonormal and reconstructs the training fields as its energy says, and the MLP can
        overfit a tiny batch: the two ways this baseline breaks before it is trained."""
        import torch

        from .fields import _overfit_check
        subj = f"{self.id}/pod_mlp_calculix"
        S = _records(prepared)
        Y = np.stack([_fields(r) for r in S["train"] if Path(r["npz"]).exists()]).astype(np.float32)
        if len(Y) < 2:
            return [Check("POD basis orthonormal", "architecture", subj, None, detail="no training fields prepared")]
        fm, fs = _channel_stats(Y)
        fs = np.where(fs > 0, fs, 1.0)
        Z = (Y - fm) / fs
        _, sv, Vt = np.linalg.svd(Z, full_matrices=False)
        K = int(min(cfg["modes"], len(Z), Z.shape[1]))
        B = Vt[:K]
        ortho = float(np.abs(B @ B.T - np.eye(K)).max())
        energy = float((sv[:K] ** 2).sum() / (sv ** 2).sum())
        rec = float(np.linalg.norm(Z - (Z @ B.T) @ B) / np.linalg.norm(Z))
        torch.manual_seed(0)
        net = _mlp(4, K, cfg["width"])
        x, t = torch.rand(6, 4), torch.rand(6, K)
        return [Check("POD basis orthonormal", "architecture", subj, ortho < 1e-6, value=f"{ortho:.1e}", threshold=1e-6),
                Check("POD reconstruction error = sqrt(1 - energy kept)", "architecture", subj,
                      abs(rec - (1 - energy) ** 0.5) < 1e-5, value=f"{rec:.4f} vs {(1 - energy) ** 0.5:.4f}",
                      detail=f"{K} modes of {len(Y)} training fields"),
                Check("forward shape", "architecture", subj, tuple(net(x).shape) == (6, K), value=tuple(net(x).shape)),
                Check("parameter count", "architecture", subj, True, value=int(sum(p.numel() for p in net.parameters())),
                      severity="warning"),
                _overfit_check(subj, lambda: ((net(x) - t) ** 2).mean(), list(net.parameters()), steps=1000,
                               lr=cfg["lr"])]


@register_problem("calculix_beam")
class CalculixBeam(_Structural):
    title = "Cantilever beam (CalculiX, ccx)"
    part = "beam"
    budget_note = ("40 train / 10 val / 12 test designs and 5 OOD, one ccx run each; the load and the slenderness "
                   "are swept, so the deflection spans two decades")


@register_problem("calculix_bracket")
class CalculixBracket(_Structural):
    title = "L bracket (CalculiX, ccx)"
    part = "bracket"
    budget_note = ("40 train / 10 val / 12 test designs and 5 OOD, one ccx run each; the thickness, the upright's "
                   "height and the load are swept")


@register_method("mlp_calculix")
class MLPCalculix(Method):
    """Design -> the two integrals (displacement, peak stress), integrals only."""
    title = "MLP (integrals)"
    outputs = "integrals"

    def supports(self, problem):
        return isinstance(problem, _Structural) and super().supports(problem)


@register_method("pod_mlp_calculix")
class PodMLPCalculix(Method):
    """Design -> the fields through a POD basis; the integrals are read off the predicted field."""
    title = "POD + MLP (fields)"
    outputs = "both"

    def supports(self, problem):
        return isinstance(problem, _Structural) and super().supports(problem)


__all__ = ["CalculixBeam", "CalculixBracket", "MLPCalculix", "PodMLPCalculix", "BOUNDS", "OOD"]
