"""The MCP tools as plain functions: JSON-like arguments in, JSON-like dictionaries out.

Short calls (catalog, presets, trust card of a given field) run in the calling process. Training and solving run as
jobs (``jobs.submit``) in a separate process with the limits of ``config.Config``; their results are written to the
job directory and returned with the paths of the files.
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_core.jsonable import to_jsonable

from . import jobs
from .config import Config, get_config
from .expressions import compile_expression
from .expressions import value as _value

NEURAL = {"pinn"}
_SCALARS = (str, int, float, bool, type(None))


# ---------------------------------------------------------------------- helpers
def _clean(obj: Any) -> Any:
    """JSON-safe copy: NumPy scalars to numbers, NaN/Inf to None, unknown objects to their repr."""
    return to_jsonable(obj, unknown="repr")


def _options(options: Mapping[str, Any] | None, cfg: Config) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    for k, v in dict(options or {}).items():
        if not isinstance(v, _SCALARS + (list,)):
            raise ValueError(f"option '{k}' must be a number, string, boolean or list")
        opts[str(k)] = v
    if "epochs" in opts:
        opts["epochs"] = int(min(int(opts["epochs"]), cfg.max_epochs))
    if opts.get("device", "auto") == "auto":
        import torch
        opts["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return opts


def _method_options(method: str, opts: dict[str, Any], built) -> dict[str, Any]:
    import pinneapple_physics.solving as S
    fn = S._METHODS.get(method)
    params = set(inspect.signature(fn).parameters) if fn else set()
    out = dict(opts)
    if "device" in out and "device" not in params:
        out.pop("device")
    if built.ctx and "ctx" in params:
        out.setdefault("ctx", dict(built.ctx))
    if method == "exact":
        if built.exact is None:
            raise ValueError("method 'exact' needs an 'exact' expression in the problem")
        out["fn"] = lambda X, _f=built.exact: _f(X)[:, None]
    return out


def _grid(prob, n_per_axis: int, cfg: Config):
    coords = list(prob.coords)
    if set(prob.domain_bounds) != set(coords):
        raise ValueError("the problem has no box bounds for every coordinate; a grid cannot be built")
    n = int(max(5, min(n_per_axis, int(cfg.max_points ** (1.0 / len(coords))))))
    axes = {c: np.linspace(*map(float, prob.domain_bounds[c]), n) for c in coords}
    mesh = np.meshgrid(*axes.values(), indexing="ij")
    return axes, np.stack([m.ravel() for m in mesh], axis=1), tuple(len(a) for a in axes.values())


def _card(built, axes, values, reference, *, name: str, train_bounds=None, is_ml=True):
    from pinneapple_analysis.trust.field_card import trust_card
    prob = built.problem
    kw: dict[str, Any] = {"pde": built.check_pde, "dirichlet": built.dirichlet or None, "reference": reference,
                          "train_bounds": train_bounds}
    if built.check_pde == "heat":
        kw["alpha"] = built.alpha
    if built.source is not None:
        kw["source"] = built.source
    rng = prob.field_ranges.get(prob.fields[0]) if getattr(prob, "field_ranges", None) else None
    if rng is not None and built.spec.get("check_value_range"):
        kw["value_range"] = tuple(rng)
    return trust_card(axes, values, name=name, is_ml=is_ml, **kw)


def _plot(axes: dict[str, np.ndarray], fields: dict[str, np.ndarray], path: Path, title: str) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return None
    names = list(axes)
    if len(names) not in (1, 2):
        return None
    fig, axs = plt.subplots(1, len(fields), figsize=(4.2 * len(fields), 3.4), squeeze=False)
    for ax, (lab, v) in zip(axs[0], fields.items(), strict=True):
        if len(names) == 1:
            ax.plot(axes[names[0]], v)
            ax.set_xlabel(names[0])
        else:
            im = ax.pcolormesh(axes[names[0]], axes[names[1]], v.T, shading="auto", cmap="viridis")
            fig.colorbar(im, ax=ax)
            ax.set_xlabel(names[0])
            ax.set_ylabel(names[1])
        ax.set_title(lab)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return str(path)


# ---------------------------------------------------------------------- catalog
def capabilities() -> dict[str, Any]:
    """What this installation offers: version, optional APIs present, device, limits and the working directory."""
    import pinneapple as pp
    cfg = get_config()
    opt = {}
    for name, mod in (("benchmark", "pinneapple_analysis.benchmark"), ("zoo", "pinneapple_hub.zoo"),
                      ("datasets", "pinneapple_hub.zoo.datasets"), ("geometries", "pinneapple_geometries")):
        try:
            opt[name] = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            opt[name] = False
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        device = "unavailable"
    return _clean({"pinneapple_version": pp.__version__, "device": device, "optional_apis": opt,
                   "limits": cfg.to_dict(), "jobs_running": jobs.running(),
                   "tools": ["capabilities", "list_methods", "describe_method", "list_presets", "solve",
                             "trust_card", "benchmark", "design", "job_status", "list_jobs", "cancel_job"]
                   + (["zoo_list", "zoo_predict", "zoo_validate"] if opt.get("zoo") else [])})


def list_methods(category: str = "all", validated_only: bool = False) -> dict[str, Any]:
    """Solve methods usable by ``solve`` (grouped by kind) and the method catalog (solvers, training methods,
    equations, problems) with the validation status of each item."""
    import pinneapple_physics.solving as S
    from pinneapple_catalog import methods as M
    cats = {"all": None, "solver": "solver", "training": "training", "equation": "equation", "problem": "problem"}
    if category not in cats:
        raise ValueError(f"category must be one of {list(cats)}")
    status = M.method_status()
    items = M.validated_methods(cats[category]) if validated_only else M.list_methods(cats[category])
    rows = [{"id": m.id, "name": m.name, "category": m.category,
             "status": status.get(m.id, {}).get("status", "unknown"),
             "reference_tests": len(status.get(m.id, {}).get("reference_tests", []))} for m in items]
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"solve_methods": S.list_kinds(), "catalog": rows, "catalog_counts": counts,
            "status_rule": "validated = at least one test compares with an independent reference "
                           "(closed form, manufactured solution or published value)"}


def describe_method(method: str) -> dict[str, Any]:
    """A solve method (e.g. ``pinn``, ``fem``) with its options, or a catalog item (e.g. ``S2``) with its code,
    references, validation status and the tests that give it."""
    import pinneapple_physics.solving as S
    from pinneapple_catalog import methods as M
    if method in S._METHODS:
        fn = S._METHODS[method]
        sig = inspect.signature(fn)
        opts = {n: (None if p.default is inspect.Parameter.empty else _clean(p.default))
                for n, p in sig.parameters.items() if n not in ("problem", "_") and p.kind != p.VAR_KEYWORD}
        return {"method": method, "type": "solve_method", "kind": S._KINDS.get(method),
                "doc": inspect.getdoc(fn) or "", "options": opts}
    try:
        m = M.get_method(method)
    except KeyError:
        raise KeyError(f"unknown method '{method}': not a solve method {S.list_methods()} nor a catalog id "
                       "(see list_methods)") from None
    st = M.method_status().get(m.id, {})
    return {"method": m.id, "type": "catalog_item", "name": m.name, "category": m.category, "code": list(m.code),
            "references": list(m.references), "equations": list(m.equations),
            "status": st.get("status", "unknown"), "reference_tests": st.get("reference_tests", []),
            "other_tests": st.get("other_tests", [])[:20]}


def list_presets(contains: str | None = None) -> dict[str, Any]:
    """Problem presets usable as ``{"preset": name}`` in ``solve`` and ``benchmark``."""
    import pinneapple as pp
    names = sorted(pp.list_presets())
    if contains:
        names = [n for n in names if contains.lower() in n.lower()]
    return {"presets": names, "count": len(names)}


# ---------------------------------------------------------------------- trust card of a given field
def _load_field(path: Path, field: str | None):
    suf = path.suffix.lower()
    if suf == ".npz":
        z = np.load(path, allow_pickle=False)
        keys = list(z.files)
        if field is None:
            cand = [k for k in keys if z[k].ndim >= 1 and not k.startswith(("axis_", "grid_", "ref_", "reference"))
                    and z[k].ndim > 1]
            if len(cand) != 1:
                raise ValueError(f"give 'field': the file holds {keys}")
            field = cand[0]
        axes = {k[len("axis_"):]: z[k] for k in keys if k.startswith("axis_")}
        ref = z[f"ref_{field}"] if f"ref_{field}" in keys else None
        return z[field], axes, ref, field
    if suf == ".npy":
        return np.load(path, allow_pickle=False), {}, None, field or path.stem
    if suf in (".csv", ".txt"):
        return np.loadtxt(path, delimiter="," if suf == ".csv" else None, ndmin=2), {}, None, field or path.stem
    raise ValueError("supported files: .npz (field arrays plus axis_<coord> arrays), .npy, .csv")


def trust_card(file: str | None = None, field: str | None = None,
               grid: Mapping[str, Sequence[float]] | None = None, values: Any | None = None,
               pde: str | None = None, source: str | None = None, alpha: float = 1.0,
               dirichlet: Mapping[str, Any] | None = None, reference: str | None = None,
               reference_file: str | None = None, value_range: Sequence[float] | None = None,
               train_bounds: Mapping[str, Sequence[float]] | None = None,
               tolerances: Mapping[str, float] | None = None, name: str = "field") -> dict[str, Any]:
    """Physics checks of a field on a grid and its trust card (decision APPROVED / REVIEW / REJECT).

    The field comes from ``file`` (``.npz`` with the field and ``axis_<coord>`` arrays, ``.npy``, ``.csv``) or
    inline ``values`` (nested lists). ``grid`` gives the axes ``{coord: [...]}`` in array order (optional for an
    ``.npz`` that holds them). ``source``, ``dirichlet`` values and ``reference`` are expressions of the
    coordinates, e.g. ``"sin(pi*x)"``.
    """
    cfg = get_config().ensure()
    ref_arr = None
    if file:
        arr, axes, ref_arr, field = _load_field(cfg.readable(file), field)
        if grid:
            axes = {str(k): np.asarray(v, dtype=float) for k, v in grid.items()}
    else:
        if values is None or not grid:
            raise ValueError("give either 'file' or both 'grid' and 'values'")
        arr = np.asarray(values, dtype=float)
        axes = {str(k): np.asarray(v, dtype=float) for k, v in grid.items()}
    if not axes:
        raise ValueError("the grid axes are missing: pass 'grid' {coord: [...]} or store axis_<coord> in the .npz")
    if int(np.prod([len(a) for a in axes.values()])) > cfg.max_points:
        raise ValueError(f"the grid has more than {cfg.max_points} points (PINNEAPPLE_MCP_MAX_POINTS)")
    coords = list(axes)
    kw: dict[str, Any] = {"pde": pde, "alpha": float(alpha), "tolerances": tolerances}
    if source not in (None, ""):
        kw["source"] = compile_expression(str(source), coords)
    if dirichlet:
        kw["dirichlet"] = {str(f): _value(v, coords) for f, v in dirichlet.items()}
    if reference_file:
        r, _, _, _ = _load_field(cfg.readable(reference_file), field)
        kw["reference"] = r
    elif reference not in (None, ""):
        kw["reference"] = compile_expression(str(reference), coords)
    elif ref_arr is not None:
        kw["reference"] = ref_arr
    if value_range is not None:
        kw["value_range"] = (float(value_range[0]), float(value_range[1]))
    if train_bounds:
        kw["train_bounds"] = {str(k): (float(v[0]), float(v[1])) for k, v in train_bounds.items()}
    from pinneapple_analysis.trust.field_card import trust_card as _tc
    card = _tc(axes, arr, name=name, **kw)
    out_dir = cfg.workdir / "cards" / f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trust_card.md").write_text(card.to_markdown())
    d = _clean(card.to_dict())
    (out_dir / "trust_card.json").write_text(json.dumps(d, indent=1))
    d["files"] = {"markdown": str(out_dir / "trust_card.md"), "json": str(out_dir / "trust_card.json")}
    return d


# ---------------------------------------------------------------------- solve (job)
def _reference_solution(built, reference: str | None, opts_by_method: Mapping[str, Mapping[str, Any]], cfg):
    """(callable X -> (N, F) or None, label, note)."""
    import pinneapple as pp
    if reference in (None, "", "auto"):
        if built.exact is not None:
            return (lambda X, _f=built.exact: _f(X)[:, None]), "exact", ""
        try:
            sol = pp.solve(built.problem, "analytic")
            return sol.predict, "analytic", ""
        except Exception as exc:  # noqa: BLE001
            return None, None, f"no reference: no 'exact' expression and {type(exc).__name__}: {exc}"
    if reference == "exact":
        if built.exact is None:
            raise ValueError("reference 'exact' needs an 'exact' expression in the problem")
        return (lambda X, _f=built.exact: _f(X)[:, None]), "exact", ""
    opts = _method_options(reference, _options(opts_by_method.get(reference), cfg), built)
    sol = pp.solve(built.problem, reference, **opts)
    return sol.predict, reference, ""


def _run_solve(args: dict[str, Any], jd: Path) -> dict[str, Any]:
    import pinneapple as pp
    from pinneapple_physics import metrics as _metrics

    from .problems import build_problem

    cfg = get_config()
    built = build_problem(args["problem"])
    prob = built.problem
    method = args.get("method", "pinn")
    opts = _method_options(method, _options(args.get("options"), cfg), built)
    t0 = time.perf_counter()
    sol = pp.solve(prob, method, **opts)
    wall = time.perf_counter() - t0
    axes, pts, shape = _grid(prob, int(args.get("grid_points_per_axis", 41)), cfg)
    if sol.grid is not None and int(np.prod([len(sol.grid[c]) for c in prob.coords])) <= cfg.max_points:
        # a grid solution is checked on its own nodes: finite differences across the kinks of an interpolated
        # piecewise-linear field would measure the interpolation, not the solution
        axes = {c: np.asarray(sol.grid[c], dtype=float) for c in prob.coords}
        shape = tuple(len(a) for a in axes.values())
        pts = np.stack([m.ravel() for m in np.meshgrid(*axes.values(), indexing="ij")], axis=1)
    pred = sol.predict(pts)
    ref_name = args.get("reference")
    ref_fn, ref_label, note = _reference_solution(built, ref_name, {str(ref_name): args.get("reference_options") or {}},
                                                  cfg)
    truth = ref_fn(pts) if ref_fn is not None else None
    saved = {f"axis_{c}": a for c, a in axes.items()}
    saved.update({f: pred[:, i].reshape(shape) for i, f in enumerate(prob.fields)})
    if truth is not None:
        saved.update({f"ref_{f}": truth[:, i].reshape(shape) for i, f in enumerate(prob.fields)})
    np.savez(jd / "prediction.npz", **saved)
    cards = {}
    for i, f in enumerate(prob.fields[:1] if built.check_pde else prob.fields):
        card = _card(built, axes, pred[:, i].reshape(shape), None if truth is None else truth[:, i].reshape(shape),
                     name=f"{prob.name}:{f} by {method}", is_ml=method in NEURAL)
        (jd / f"trust_card_{f}.md").write_text(card.to_markdown())
        cards[f] = _clean(card.to_dict())
    figs = {}
    f0 = prob.fields[0]
    show = {f"{f0} ({method})": pred[:, 0].reshape(shape)}
    if truth is not None:
        show[f"|error| vs {ref_label}"] = np.abs(pred[:, 0] - truth[:, 0]).reshape(shape)
    p = _plot(axes, show, jd / "field.png", f"{prob.name}: {method}")
    if p:
        figs["field"] = p
    out = {"problem": prob.name, "method": method, "metadata": _clean(sol.metadata()), "wall_time_s": wall,
           "grid": {c: [float(a[0]), float(a[-1]), len(a)] for c, a in axes.items()},
           "reference": ref_label, "reference_note": note,
           "metrics": _clean(_metrics.summary(pred, truth, prob.fields)) if truth is not None else None,
           "trust_cards": cards,
           "decision": _worst([c["decision"] for c in cards.values()]),
           "files": {"prediction": str(jd / "prediction.npz"), "figures": figs,
                     "trust_cards": {f: str(jd / f"trust_card_{f}.md") for f in cards}}}
    hist = getattr(sol, "history", None) or {}
    for k in ("loss", "total", "total_loss"):
        if isinstance(hist, Mapping) and k in hist and len(hist[k]):
            out["final_loss"] = _clean(float(np.asarray(hist[k])[-1]))
            break
    return out


def _worst(decisions: Sequence[str]) -> str:
    order = {"REJECT": 0, "REVIEW": 1, "APPROVED": 2}
    return min(decisions, key=lambda d: order.get(d, 1)) if decisions else "REVIEW"


def solve(problem: Mapping[str, Any], method: str = "pinn", options: Mapping[str, Any] | None = None,
          reference: str | None = None, reference_options: Mapping[str, Any] | None = None,
          grid_points_per_axis: int = 41, max_seconds: float | None = None,
          wait_seconds: float = 0.0) -> dict[str, Any]:
    """Solve ``problem`` with ``method`` in a job; the result carries metrics against the reference, the physics
    checks and the trust card, and the paths of the saved prediction and figures. Returns the job status
    (``wait_seconds`` > 0 waits for the result up to that long; otherwise poll ``job_status``)."""
    import pinneapple_physics.solving as S

    from .problems import build_problem
    if method not in S._METHODS:
        raise KeyError(f"unknown method '{method}'; available: {S.list_methods()}")
    build_problem(problem)                       # fail fast on a bad spec, before starting a process
    cfg = get_config()
    args = {"problem": dict(problem), "method": method, "options": dict(options or {}), "reference": reference,
            "reference_options": dict(reference_options or {}), "grid_points_per_axis": int(grid_points_per_axis)}
    jid = jobs.submit("solve", args, cfg, max_seconds)
    return jobs.status(jid, cfg, wait_seconds)


# ---------------------------------------------------------------------- benchmark (job)
def _run_benchmark(args: dict[str, Any], jd: Path) -> dict[str, Any]:
    import pinneapple as pp
    from pinneapple_physics import metrics as _metrics

    from .problems import build_problem

    cfg = get_config()
    built = build_problem(args["problem"])
    prob = built.problem
    budget = dict(args.get("budget") or {})
    epochs = int(min(int(budget.get("epochs", 1000)), cfg.max_epochs))
    n_col = int(budget.get("n_collocation", 2048))
    seeds = list(range(int(args.get("seeds", 1))))
    axes, pts, shape = _grid(prob, int(args.get("grid_points_per_axis", 41)), cfg)
    ref = args.get("reference") or ("exact" if built.exact is not None else None)
    if ref is None:
        ref = "fem" if built.check_pde in ("laplace", "poisson") else "analytic"
    ref_fn, ref_label, _ = _reference_solution(built, ref, dict(args.get("options") or {}), cfg)
    truth = ref_fn(pts)
    rows: list[dict[str, Any]] = []
    for m in args["methods"]:
        per_method = dict((args.get("options") or {}).get(m) or {})
        runs = []
        for s in (seeds if m in NEURAL else [0]):
            o = dict(per_method)
            if m in NEURAL:
                o.update({"epochs": epochs, "n_collocation": n_col, "seed": s})
            try:
                opts = _method_options(m, _options(o, cfg), built)
                t0 = time.perf_counter()
                sol = pp.solve(prob, m, **opts)
                wall = time.perf_counter() - t0
                t1 = time.perf_counter()
                pred = sol.predict(pts)
                infer = (time.perf_counter() - t1) / len(pts) * 1e6
                rel = _metrics.summary(pred, truth, prob.fields)["relative_l2"]
                c_axes, c_pred, c_truth = axes, pred, truth
                if sol.grid is not None and int(np.prod([len(sol.grid[c]) for c in prob.coords])) <= cfg.max_points:
                    c_axes = {c: np.asarray(sol.grid[c], dtype=float) for c in prob.coords}   # its own nodes
                    c_pts = np.stack([g.ravel() for g in np.meshgrid(*c_axes.values(), indexing="ij")], axis=1)
                    c_pred, c_truth = sol.predict(c_pts), ref_fn(c_pts)
                c_shape = tuple(len(a) for a in c_axes.values())
                card = _card(built, c_axes, c_pred[:, 0].reshape(c_shape), c_truth[:, 0].reshape(c_shape),
                             name=f"{prob.name} by {m} (seed {s})", is_ml=m in NEURAL)
                runs.append({"seed": s, "ok": True, "wall_time_s": wall, "infer_us_per_point": infer,
                             "relative_l2": rel, "decision": card.decision,
                             "failed_checks": [c.name for c in card.failed]})
            except Exception as exc:  # noqa: BLE001 - one failing method must not hide the others
                runs.append({"seed": s, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
        ok = [r for r in runs if r["ok"]]
        mean_rel = {f: float(np.mean([r["relative_l2"][f] for r in ok])) for f in prob.fields} if ok else None
        std_rel = {f: float(np.std([r["relative_l2"][f] for r in ok])) for f in prob.fields} if ok else None
        rows.append({"method": m, "runs": runs, "n_ok": len(ok), "relative_l2_mean": mean_rel,
                     "relative_l2_std": std_rel,
                     "wall_time_s_mean": float(np.mean([r["wall_time_s"] for r in ok])) if ok else None,
                     "decision": _worst([r["decision"] for r in ok]) if ok else "REJECT"})
    f0 = prob.fields[0]
    lines = [f"# Equal-budget benchmark: {prob.name}", "",
             f"Reference: {ref_label}. Budget per neural run: {epochs} epochs, {n_col} collocation points, "
             f"seeds {seeds}. Errors on a {'x'.join(map(str, shape))} grid.", "",
             f"| method | runs ok | rel. L2 ({f0}) mean | std | train [s] | trust card |", "|---|---|---|---|---|---|"]
    for r in rows:
        if r["n_ok"]:
            lines.append(f"| {r['method']} | {r['n_ok']}/{len(r['runs'])} | {r['relative_l2_mean'][f0]:.3e} | "
                         f"{r['relative_l2_std'][f0]:.1e} | {r['wall_time_s_mean']:.3g} | {r['decision']} |")
        else:
            lines.append(f"| {r['method']} | 0/{len(r['runs'])} | failed: {r['runs'][0]['error'][:80]} | | | |")
    (jd / "table.md").write_text("\n".join(lines) + "\n")
    ok_rows = [r for r in rows if r["n_ok"]]
    best = min(ok_rows, key=lambda r: r["relative_l2_mean"][f0])["method"] if ok_rows else None
    return _clean({"problem": prob.name, "reference": ref_label,
                   "budget": {"epochs": epochs, "n_collocation": n_col, "seeds": seeds}, "rows": rows,
                   "best": best, "engine": "pp.solve with an equal training budget",
                   "files": {"table": str(jd / "table.md")}})


def benchmark(problem: Mapping[str, Any], methods: Sequence[str] = ("pinn", "fem"),
              budget: Mapping[str, Any] | None = None, seeds: int = 1, reference: str | None = None,
              options: Mapping[str, Mapping[str, Any]] | None = None, grid_points_per_axis: int = 41,
              max_seconds: float | None = None, wait_seconds: float = 0.0) -> dict[str, Any]:
    """Run several methods on one problem with the same training budget (epochs, collocation points, seeds for
    every neural method) and score them on the same grid against one reference; each run gets a trust card.
    Returns the job status (see ``solve``)."""
    import pinneapple_physics.solving as S

    from .problems import build_problem
    methods = [str(m) for m in methods]
    unknown = [m for m in methods if m not in S._METHODS]
    if unknown or not methods:
        raise KeyError(f"unknown methods {unknown}; available: {S.list_methods()}")
    if not 1 <= int(seeds) <= 10:
        raise ValueError("seeds must be between 1 and 10")
    build_problem(problem)
    cfg = get_config()
    args = {"problem": dict(problem), "methods": methods, "budget": dict(budget or {}), "seeds": int(seeds),
            "reference": reference, "options": {k: dict(v) for k, v in dict(options or {}).items()},
            "grid_points_per_axis": int(grid_points_per_axis)}
    jid = jobs.submit("benchmark", args, cfg, max_seconds)
    return jobs.status(jid, cfg, wait_seconds)


# ---------------------------------------------------------------------- model zoo and datasets
def _zoo():
    try:
        from pinneapple_hub import zoo
    except ImportError as exc:
        raise RuntimeError("the model zoo (pp.zoo) is not in this PINNeAPPle installation; upgrade it") from exc
    return zoo


def _offline() -> bool:
    return not get_config().allow_network


def zoo_list(task: str | None = None) -> dict[str, Any]:
    """Pre-trained, verified models of the zoo (with what they predict, their validity range and held-out errors)
    and the datasets of the dataset hub."""
    zoo = _zoo()
    models = []
    for name in zoo.list(task):
        c = zoo.info(name, offline=_offline())
        models.append({"name": name, "task": c.task, "inputs": c.inputs, "outputs": c.outputs,
                       "validity_range": c.validity_range, "validation_metrics": c.validation_metrics,
                       "reference_source": c.reference_source, "license": c.license, "version": c.version})
    from pinneapple_hub.zoo import datasets
    data = []
    for name in datasets.list():
        i = datasets.info(name)
        data.append({"name": name, "title": i.title, "splits": i.splits, "columns": i.columns, "license": i.license})
    return _clean({"models": models, "datasets": data})


def zoo_predict(name: str, inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Predictions of zoo model ``name`` for ``inputs`` ({input name: value or list}), with the mask of rows inside
    the model's validity range (rows outside it are reported, never silently extrapolated)."""
    import warnings
    zoo = _zoo()
    m = zoo.load(name, offline=_offline())
    unknown = sorted(set(inputs) - set(m.input_names))
    missing = sorted(set(m.input_names) - set(inputs))
    if unknown or missing:
        raise ValueError(f"model '{name}' takes inputs {m.input_names}; unknown {unknown}, missing {missing}")
    cols = {k: np.atleast_1d(np.asarray(v, dtype=float)) for k, v in inputs.items()}
    if max(len(v) for v in cols.values()) > get_config().max_points:
        raise ValueError(f"more than {get_config().max_points} rows (PINNEAPPLE_MCP_MAX_POINTS)")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X = m.as_array(cols)
        out = m.predict(X, warn=False)
        inside = m.in_range(X)
    return _clean({"model": name, "outputs": out, "inside_validity_range": inside,
                   "n_outside": int((~inside).sum()), "validity_range": m.card.validity_range})


def zoo_validate(name: str, inputs: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Trust report of zoo model ``name``: its card metrics reproduced on the stored reference, the zoo acceptance
    limits, the model's physics checks and, with ``inputs``, whether they are inside its validity range."""
    import warnings
    zoo = _zoo()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rep = zoo.validate(name, None if not inputs else {k: np.atleast_1d(np.asarray(v, dtype=float))
                                                          for k, v in inputs.items()}, offline=_offline())
    return _clean({"model": name, "decision": rep.decisao, "score": rep.score, "coverage": rep.cobertura,
                   "checks": [{"name": c.name, "status": c.status, "critical": c.critical, "value": c.value,
                               "limit": c.limit, "source": c.source} for c in rep.checks],
                   "details": rep.details})


# ---------------------------------------------------------------------- jobs
def job_status(job_id: str, wait_seconds: float = 0.0) -> dict[str, Any]:
    """Status of a job and, when it is done, its result with the paths of its files."""
    return _clean(jobs.status(job_id, get_config(), min(float(wait_seconds), 3600.0)))


def list_jobs(limit: int = 20) -> dict[str, Any]:
    """The most recent jobs with their status."""
    return {"jobs": _clean(jobs.list_jobs(get_config(), int(limit)))}


def cancel_job(job_id: str) -> dict[str, Any]:
    """Stop a running job (its process is killed)."""
    return _clean(jobs.cancel(job_id, get_config()))


# ---------------------------------------------------------------------- design from intent
def _answers(answers: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not answers:
        return None
    return {k: tuple(v) if isinstance(v, (list, tuple)) else v for k, v in answers.items()}


def design(text: str, answers: Mapping[str, Any] | None = None, verify: bool = True, seed: int = 0,
           max_seconds: float | None = None, wait_seconds: float = 0.0) -> dict[str, Any]:
    """Design from a plain-language goal. Out-of-scope requests and missing values are answered at once (status
    ``out_of_scope`` with what is needed, or ``needs_input`` with the questions); an in-scope request starts a job
    whose result is the design package (verdict, design, margins, verification, files)."""
    from pinneapple_design.intent import OutOfScope, formulate
    spec = formulate(text, answers=_answers(answers))
    if isinstance(spec, OutOfScope):
        return _clean(spec.to_dict())
    if spec.questions or spec.issues:
        from pinneapple_design.intent import NeedsInput
        return _clean(NeedsInput(spec).to_dict())
    cfg = get_config()
    args = {"text": text, "answers": dict(answers or {}), "verify": bool(verify), "seed": int(seed)}
    jid = jobs.submit("design", args, cfg, max_seconds)
    return jobs.status(jid, cfg, wait_seconds)


def _run_design(args: dict[str, Any], jd: Path) -> dict[str, Any]:
    from pinneapple_design.intent import design as run_design
    pkg = run_design(args["text"], answers=_answers(args.get("answers")), verify=bool(args.get("verify", True)),
                     seed=int(args.get("seed", 0)), out_dir=str(jd / "design"))
    d = pkg.to_dict()
    return _clean({k: d.get(k) for k in ("status", "verdict", "design", "assessment", "verification",
                                          "operating_window", "bill_of_materials", "limits", "savings", "files")})


JOB_KINDS = {"solve": _run_solve, "benchmark": _run_benchmark, "design": _run_design}
