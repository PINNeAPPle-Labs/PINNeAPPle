"""Data nodes: observations sampled from a solution, CSV ingest, parametric sweeps and operator-learning sets."""
from __future__ import annotations

import csv
import io
import time

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..registry import Param, Port, node
from ..runtime import load_artifact
from ._common import bounds_of, grid_points, load_solution, need_problem, system_of


@node("data.observe", "Sensor observations", category="Data",
      inputs=[Port("solution", "solution"), Port("problem", "problem", required=False,
                                                 help="to record the true coefficients of synthetic data")],
      outputs=[Port("data", "dataset")],
      params=[Param("n_points", "int", 300, minimum=1, maximum=1_000_000),
              Param("noise", "float", 0.0, minimum=0.0, maximum=1.0, help="Gaussian noise, fraction of the "
                                                                          "field's standard deviation"),
              Param("seed", "int", 0, minimum=0)],
      calls="random nodes of a solution grid + Gaussian noise (numpy)")
def observe(ctx, inputs, params):
    """Point observations from a solution (synthetic sensors), with optional noise; keeps the true coefficients of
    the problem that produced them so an inverse node can report its error."""
    sol = load_solution(inputs["solution"])
    X = grid_points(sol["axes"], sol["coords"])
    V = sol["values"].reshape(-1, sol["values"].shape[-1])
    rng = np.random.default_rng(int(params["seed"]))
    n = min(int(params["n_points"]), len(X))
    idx = rng.choice(len(X), n, replace=False)
    Y = V[idx].copy()
    sigma = float(params["noise"]) * float(V.std())
    if sigma > 0:
        Y += rng.normal(0.0, sigma, Y.shape)
    extra = {}
    if inputs.get("problem"):
        eq = need_problem(inputs["problem"])["equations"]
        coeffs = eq.get("coefficients") or {}
        extra = {"truth_names": np.array(list(coeffs), dtype=str), "truth_values": np.array(list(coeffs.values()),
                                                                                             float)}
    ctx.metric("n_points", n)
    ctx.metric("noise_sigma", sigma)
    a = ctx.save_npz("observations", X=X[idx], Y=Y, sigma=np.array(max(sigma, 1e-12)), **extra)
    a.update({"coords": sol["coords"], "fields": sol["fields"]})
    return {"data": a}


def _read_csv(text: str) -> tuple[list[str], np.ndarray]:
    rows = list(csv.reader(io.StringIO(text.strip())))
    if len(rows) < 2:
        raise StepFailed("a CSV needs a header and at least one row")
    head = [h.strip() for h in rows[0]]
    try:
        arr = np.array([[float(v) for v in r] for r in rows[1:] if r], float)
    except ValueError as exc:
        raise StepFailed(f"non-numeric CSV value: {exc}") from None
    if arr.shape[1] != len(head):
        raise StepFailed("CSV rows and header have different lengths")
    return head, arr


@node("data.ingest", "Data ingest (CSV)", category="Data",
      outputs=[Port("data", "dataset"), Port("series", "series")],
      params=[Param("source", "enum", "inline", choices=["inline", "file"]),
              Param("csv", "text", "", max_length=2_000_000, help="CSV text with a header (source = inline)"),
              Param("file", "str", "", help="file name inside the data folder (source = file)"),
              Param("inputs", "list", ["x", "t"], help="input columns (coordinates or parameters)"),
              Param("outputs", "list", ["u"], help="output columns (observed fields)")],
      calls="csv -> arrays; files only from PINNEAPPLE_FLOW_DATA (pinneapple_flow.config.safe_data_path)")
def ingest(ctx, inputs, params):
    """Load measurements from CSV (pasted, or a file in the data folder) as a dataset and a series."""
    if params["source"] == "file":
        from ..config import safe_data_path
        text = safe_data_path(params["file"]).read_text()
    else:
        text = params["csv"]
    head, arr = _read_csv(text)
    miss = [c for c in params["inputs"] + params["outputs"] if c not in head]
    if miss:
        raise StepFailed(f"columns {miss} not in the CSV header {head}")
    X = arr[:, [head.index(c) for c in params["inputs"]]]
    Y = arr[:, [head.index(c) for c in params["outputs"]]]
    ctx.metric("rows", len(arr))
    ctx.table("first rows", head, arr[:8].tolist())
    a = ctx.save_npz("data", X=X, Y=Y)
    a.update({"coords": params["inputs"], "fields": params["outputs"]})
    s = ctx.save_npz("series", t=X[:, 0], values=Y)
    s.update({"channels": params["outputs"]})
    return {"data": a, "series": s}


@node("data.parametric", "Parametric sweep (solver runs)", category="Data",
      inputs=[Port("problem", "problem")], outputs=[Port("data", "dataset")],
      params=[Param("vary", "dict", {"alpha": [0.05, 0.5]}, help="{coefficient: [low, high]}; coefficients may "
                                                                  "appear in PDEs, conditions or initial data"),
              Param("n_samples", "int", 64, minimum=4, maximum=5000),
              Param("n", "int", 41, minimum=5, maximum=512, help="nodes per spatial axis"),
              Param("output", "enum", "final", choices=["final", "space_time"],
                    help="the field at the final time, or every saved time"),
              Param("n_t", "int", 11, minimum=2, maximum=401),
              Param("seed", "int", 0, minimum=0)],
      calls="pinneapple_simulation.numerical_solvers.symbolic_fd.solve_fd per Latin-hypercube sample")
def parametric(ctx, inputs, params):
    """A training set for surrogates: one finite-difference solve per sample of the varied coefficients."""
    from pinneapple_simulation.numerical_solvers.symbolic_fd import FDError, solve_fd
    problem = need_problem(inputs["problem"])
    sys_ = system_of(problem)
    vary = {k: tuple(map(float, v)) for k, v in params["vary"].items()}
    if not vary:
        raise StepFailed("nothing to vary")
    unknown = [k for k in vary if k not in sys_.used_coefficients()]
    if unknown:
        raise StepFailed(f"{unknown} do not appear in the equations")
    rng = np.random.default_rng(int(params["seed"]))
    n_s, names = int(params["n_samples"]), list(vary)
    # Latin hypercube in [0, 1]^d, then scaled
    u = (np.argsort(rng.random((len(names), n_s)), axis=1).T + rng.random((n_s, len(names)))) / n_s
    lo = np.array([vary[k][0] for k in names])
    hi = np.array([vary[k][1] for k in names])
    A = lo + (hi - lo) * u
    bnd = bounds_of(problem)
    n = {c: int(params["n"]) for c in sys_.ctx.spatial}
    Ys, t0 = [], time.perf_counter()
    for i, a in enumerate(A):
        ctx.check_stop()
        try:
            res = solve_fd(sys_, bnd, n=n, n_save=int(params["n_t"]), coefficients=dict(zip(names, a, strict=True)))
        except FDError as exc:
            raise StepFailed(f"sample {i} ({dict(zip(names, a, strict=True))}): {exc}") from None
        V = np.stack([res.fields[f] for f in sys_.fields], -1)
        if res.t is not None and params["output"] == "final":
            V = V[-1]
        elif res.t is not None:
            V = np.moveaxis(V, 0, len(sys_.ctx.spatial))
        Ys.append(V.reshape(-1))
        if i % max(1, n_s // 10) == 0:
            ctx.log(f"sample {i + 1}/{n_s}")
    axes = dict(res.axes)
    coords = list(sys_.ctx.spatial)
    if res.t is not None and params["output"] == "space_time":
        axes["t"] = res.t
        coords = coords + ["t"]
    grid = grid_points(axes, coords)
    secs = time.perf_counter() - t0
    ctx.metric("samples", n_s)
    ctx.metric("outputs_per_sample", int(Ys[0].size))
    ctx.metric("generation_seconds", round(secs, 2))
    a = ctx.save_npz("dataset", A=A, Y=np.array(Ys), grid=grid, grid_shape=np.array([len(axes[c]) for c in coords]),
                     lo=lo, hi=hi)
    a.update({"param_names": names, "coords": coords, "fields": sys_.fields, "ranges": vary})
    return {"data": a}


@node("data.grf_burgers", "Operator set: random initial data -> Burgers", category="Data",
      outputs=[Port("data", "dataset")],
      params=[Param("n_samples", "int", 256, minimum=8, maximum=10000),
              Param("n", "int", 128, minimum=16, maximum=1024, help="grid points on [0, 1) (periodic)"),
              Param("nu", "float", 0.01, minimum=1e-4, maximum=1.0), Param("t_end", "float", 0.5, minimum=0.01),
              Param("alpha", "float", 2.5, advanced=True), Param("tau", "float", 7.0, advanced=True),
              Param("dt", "float", 2e-4, minimum=1e-6, maximum=1e-2, advanced=True), Param("seed", "int", 0)],
      calls="pinneapple_simulation.numerical_solvers.operator_benchmarks.periodic_grf + burgers_1d")
def grf_burgers(ctx, inputs, params):
    """The classic operator-learning benchmark: Gaussian-random-field initial conditions mapped to the viscous
    Burgers solution at ``t_end`` (pseudo-spectral reference solver)."""
    from pinneapple_simulation.numerical_solvers.operator_benchmarks import burgers_1d, periodic_grf
    n_s, n = int(params["n_samples"]), int(params["n"])
    u0 = periodic_grf(n_s, (n,), alpha=float(params["alpha"]), tau=float(params["tau"]), seed=int(params["seed"]))
    t0 = time.perf_counter()
    out = []
    for i in range(0, n_s, 32):
        ctx.check_stop()
        _, u = burgers_1d(u0[i:i + 32], float(params["nu"]), float(params["t_end"]), dt=float(params["dt"]))
        out.append(u[:, -1])
        ctx.log(f"{min(i + 32, n_s)}/{n_s} solved")
    Y = np.concatenate(out)
    x = np.arange(n) / n
    ctx.metric("samples", n_s)
    ctx.metric("generation_seconds", round(time.perf_counter() - t0, 2))
    a = ctx.save_npz("dataset", A=u0, Y=Y, grid=x[:, None], grid_shape=np.array([n]))
    a.update({"param_names": [f"u0(x_{j})" for j in range(n)], "coords": ["x"], "fields": ["u"],
              "input_kind": "function"})
    return {"data": a}


def dataset_of(v):
    d = load_artifact(v)
    if "A" not in d:
        raise StepFailed("expected an operator dataset (A, Y, grid): use a parametric sweep or an operator set")
    return d
