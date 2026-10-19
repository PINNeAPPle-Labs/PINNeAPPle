"""Checks of a model: safe to load, method status, inputs inside the training range, physics of its predictions at
sampled points (residual, divergence, global balances on a grid over its domain), boundary and initial conditions,
bounds, and regression against reference points."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import _reuse, balances
from ..config import ModelTarget
from ..models import LoadedModel, ModelLoadError, UnsafeCheckpoint, load_model, scan
from ..numerics import ModelD
from ..physics import DISP, VEL, equations, parse_parameters
from ..result import Location
from .common import Ctx, check_units, check_values, fmt, rel_l2
from .results import check_balance, evaluate_expr, report_equations

__all__ = ["check_model", "read_table"]

MAX_ROWS = 20          # per-row violations reported for queries (the rest are counted in the summary)


def read_table(path: Path) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Columns of a CSV (header) or arrays of an NPZ, plus the line of each row in a CSV (for annotations)."""
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as z:
            return {k: np.asarray(z[k], float).ravel() for k in z.files if z[k].dtype.kind in "fiu"}, {}
    import csv
    lines = path.read_text(encoding="utf-8").splitlines()
    rows, line_of = [], {}
    header = None
    for ln, s in enumerate(lines, 1):
        if not s.strip() or s.lstrip().startswith("#"):
            continue
        r = next(csv.reader([s]))
        if header is None:
            header = [_split(h) for h in r]
            continue
        line_of[len(rows)] = ln
        rows.append(r)
    if header is None:
        raise ValueError(f"{path.name} is empty")
    cols = {}
    for j, h in enumerate(header):
        try:
            cols[h] = np.array([float(r[j]) for r in rows])
        except (ValueError, IndexError):
            pass
    return cols, line_of


def _split(h: str) -> str:
    h = h.strip()
    return h.split("[", 1)[0].strip() if "[" in h else h


def check_model(ctx: Ctx, t: ModelTarget) -> LoadedModel | None:
    p = Path(str(t.path))
    if not t.hub and not t.zoo and p.suffix.lower() in (".pt", ".pth", ".ckpt", ".bin") and p.is_file():
        rep = scan(p)
        ctx.emit("PC803", "checkpoint", rep.safe,
                 f"{p.name}: pickle imports {', '.join(sorted(rep.dangerous)) or 'only allow-listed callables'}"
                 + (f"; errors: {'; '.join(rep.errors)}" if rep.errors else ""), ctx.default_location())
    try:
        model = load_model(t)
    except UnsafeCheckpoint:
        ctx.note("not loaded: the checkpoint is unsafe (PC803)")
        return None
    except (ModelLoadError, Exception) as e:
        ctx.emit("PC001", "load", False, f"cannot load model {t.name}: {e}")
        return None
    ctx.emit("PC001", "load", True, f"loaded model {t.name}")
    t.inputs, t.outputs = t.inputs or model.inputs, t.outputs or model.outputs
    _method(ctx, t)
    rng_box = {**model.train_range, **t.train_range}
    domain = {**rng_box, **t.domain}
    params, problems = parse_parameters(t.physics or "none", t.parameters)
    for k, msg in problems:
        ctx.emit("PC504", k, False, f"parameter {k}: {msg}")
    for k, v in t.parameters.items():
        if isinstance(v, str) and k not in dict(problems):
            ctx.emit("PC504", k, True, f"parameter {k} = {v}")
    if t.queries is not None:
        _ood(ctx, t, model, rng_box)
    missing = [a for a in t.inputs if a not in domain]
    X = None
    if not missing:
        X = _sample(domain, t.inputs, t.n_points, t.seed)
    elif t.queries is not None:                       # no box to sample: check the predictions at the queries
        try:
            cols, _ = read_table(t.queries)
            if all(a in cols for a in t.inputs):
                X = np.column_stack([cols[a] for a in t.inputs])
        except Exception:                             # already reported by the range check
            X = None
    if X is None:
        ctx.note(f"no domain for input(s) {', '.join(missing)}: give `domain` or `train-range` to check the "
                 "predictions")
    else:
        try:
            P = model(X)
        except Exception as e:
            ctx.emit("PC001", "predict", False, f"the model failed on {len(X)} inputs: {e}")
            return model
        where = lambda i: ", ".join(f"{a}={X[i, k]:.4g}" for k, a in enumerate(t.inputs))   # noqa: E731
        loc = ctx.default_location()
        for name in t.outputs:
            for n, v in _named(P, name).items():
                check_values(ctx, n, v, t.units.get(n), loc, where)
                if n in t.units:
                    check_units(ctx, n, None, t.units[n], v, loc, False)
        if missing:
            ctx.note(f"physics, boundary and initial conditions need a `domain` for {', '.join(missing)}; "
                     "predictions checked at the queries only")
        else:
            if t.physics and t.physics != "none":
                _physics(ctx, t, model, domain, params, loc)
            if t.bc:
                _bc(ctx, t, model, domain, loc)
            if t.ic:
                _ic(ctx, t, model, domain, loc)
            if t.relations:
                _relations(ctx, t, model, domain, loc)
    if t.reference is not None:
        _reference(ctx, t, model)
    substantive = ctx.ran - {"PC001", "PC403", "PC502", "PC504", "PC803"}
    ctx.emit("PC002", "checks", bool(substantive),
             f"checked by {', '.join(sorted(substantive))}" if substantive else
             f"no physics, range, boundary condition or reference check applies to {t.name}")
    return model


def _relations(ctx: Ctx, t: ModelTarget, model: LoadedModel, domain, loc) -> None:
    """PC303: the declared metamorphic relations, on samples of the domain box (random search)."""
    mm = _reuse.metamorphic()
    box = {a: tuple(domain[a]) for a in t.inputs}

    def f(x: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        return model(np.column_stack([np.asarray(x[a], float) for a in t.inputs]))
    try:
        rels = mm.relations_from_spec(t.relations)
        rep = mm.check_relations(f, rels, mm.box_sampler(box), n=min(t.n_points, 256), seed=t.seed)
    except Exception as e:
        ctx.emit("PC303", "relations", False, f"the relations could not be evaluated: {e}", loc)
        return
    for r in rep.results:
        ce = r.counterexample if not r.passed else None
        at = ""
        if ce:
            at = "; counterexample " + ", ".join(f"{k}={v[0]:.4g}" for k, v in ce["input"].items()
                                                 if isinstance(v, list) and v and isinstance(v[0], float))
        ctx.emit("PC303", r.name, r.passed, f"{r.kind} relation {r.name}: worst violation {fmt(r.worst)} "
                 f"({'<=' if r.passed else '>'} {fmt(r.tol)}, {r.n_tested} samples){at}", loc,
                 value=r.worst, threshold=r.tol)


def _named(P: dict[str, np.ndarray], name: str) -> dict[str, np.ndarray]:
    if name in P:
        return {name: P[name]}
    return {k: v for k, v in P.items() if k.startswith(name)}


def _sample(domain: dict[str, list[float]], inputs: list[str], n: int, seed: int, margin: float = 0.0) -> np.ndarray:
    g = np.random.default_rng(seed)
    X = np.empty((n, len(inputs)))
    for k, a in enumerate(inputs):
        lo, hi = domain[a]
        m = margin * (hi - lo)
        X[:, k] = g.uniform(lo + m, hi - m, n)
    return X


def _method(ctx: Ctx, t: ModelTarget) -> None:
    if not t.method:
        return
    try:
        cat = _reuse.method_catalog()
        m = cat.get_method(t.method)
        st = cat.method_status().get(m.id, {}).get("status", "untested")
    except KeyError:
        ctx.emit("PC901", t.method, False, f"method {t.method!r} is not in the PINNeAPPle method catalog")
        return
    except Exception as e:                                            # the catalog is optional at run time
        ctx.note(f"method catalog unavailable: {e}")
        return
    ctx.emit("PC901", t.method, st == "validated", f"method {m.id} ({m.name}) is {st} in the method catalog")


# ---------------------------------------------------------------------------------------------- OOD
def _ood(ctx: Ctx, t: ModelTarget, model: LoadedModel, box: dict[str, list[float]]) -> None:
    qloc = ctx.cfg.rel(t.queries)
    try:
        cols, line_of = read_table(t.queries)
    except Exception as e:
        ctx.emit("PC001", "queries", False, f"cannot read the queries {qloc}: {e}")
        return
    feats = [a for a in t.inputs if a in cols]
    if not feats:
        ctx.emit("PC001", "queries", False, f"{qloc} has none of the model inputs {t.inputs} as columns")
        return
    if not box:
        ctx.note("queries given but no training range (train-range, checkpoint metadata or model card): "
                 "PC601 cannot run")
        return
    Env = _reuse.envelope_cls()
    samples = None
    if t.train_samples is not None:
        try:
            sc, _ = read_table(t.train_samples)
            sf = [a for a in feats if a in sc]
            samples = (np.column_stack([sc[a] for a in sf]), tuple(sf))
        except Exception as e:
            ctx.emit("PC001", "train-samples", False, f"cannot read {ctx.cfg.rel(t.train_samples)}: {e}")
    env = Env(ranges={a: tuple(box[a]) for a in feats if a in box},
              samples=samples[0] if samples else None, sample_features=samples[1] if samples else ())
    n = len(cols[feats[0]])
    bad, gaps = [], []
    for i in range(n):
        r = env.check({a: float(cols[a][i]) for a in feats})
        if not r["inside"]:
            bad.append((i, r))
        elif r.get("knn_distance", 0.0) > 3.0:
            gaps.append((i, r))
    def floc(i: int) -> Location:
        return Location(qloc, line_of.get(i, 1)) if line_of else ctx.default_location()

    for i, r in bad[:MAX_ROWS]:
        outside = ", ".join(f"{a}={fmt(float(cols[a][i]))} not in [{fmt(box[a][0])}, {fmt(box[a][1])}]"
                            for a in r["outside"])
        ctx.emit("PC601", f"query {i + 1}", False, f"query {i + 1} of {qloc} is outside the training range of "
                 f"{t.name}: {outside}", floc(i), value={a: float(cols[a][i]) for a in r["outside"]},
                 threshold={a: box[a] for a in r["outside"]})
    if len(bad) > MAX_ROWS:
        ctx.emit("PC601", "queries (more)", False, f"{len(bad) - MAX_ROWS} more queries of {qloc} are outside the "
                 f"training range ({len(bad)} of {n} in total)", floc(bad[MAX_ROWS][0]))
    if not bad:
        ctx.emit("PC601", "queries", True, f"all {n} queries of {qloc} inside the training range "
                 + ", ".join(f"{a} [{fmt(box[a][0])}, {fmt(box[a][1])}]" for a in feats if a in box))
    for i, r in gaps[:MAX_ROWS]:
        ctx.emit("PC602", f"query {i + 1}", False, f"query {i + 1} of {qloc} is {r['knn_distance']:.1f} typical "
                 f"spacings from the nearest training sample", floc(i), value=r["knn_distance"], threshold=3.0)
    if samples is not None and not gaps:
        ctx.emit("PC602", "queries", True, f"every query of {qloc} is near training samples")


# ---------------------------------------------------------------------------------------------- physics
def _physics(ctx: Ctx, t: ModelTarget, model: LoadedModel, domain, params, loc) -> None:
    axes = [a for a in t.inputs if a in ("x", "y", "z", "t")]
    h = {a: 5e-3 * (domain[a][1] - domain[a][0]) for a in axes}
    Xi = _sample(domain, t.inputs, t.n_points, t.seed + 1, margin=0.02)
    try:
        D = ModelD(model, Xi, t.inputs, h, spans={a: domain[a][1] - domain[a][0] for a in axes})
        eqs, why = equations(t.physics, D, params, None)
    except Exception as e:
        ctx.emit("PC001", "derivatives", False, f"evaluating the model for its residual failed: {e}")
        return
    where = lambda i: ", ".join(f"{a}={Xi[i, k]:.4g}" for k, a in enumerate(t.inputs))   # noqa: E731
    report_equations(ctx, eqs, why, where, "PC102", "PC302", t.tol("model-divergence"), t.tol("model-residual"),
                     loc)
    _grid_balances(ctx, t, model, domain, params, loc)


def _grid_balances(ctx: Ctx, t: ModelTarget, model: LoadedModel, domain, params, loc) -> None:
    sp = [a for a in t.inputs if a in ("x", "y", "z")]
    if not sp:
        return
    m = {1: 513, 2: 97, 3: 25}.get(len(sp), 25)
    coords = [np.linspace(*domain[a], m) for a in sp]
    mg = np.meshgrid(*coords, indexing="ij")
    times = [None]
    if "t" in t.inputs:
        lo, hi = domain["t"]
        times = [lo, 0.5 * (lo + hi), hi]
    worst = None
    for tv in times:
        X = np.empty((mg[0].size, len(t.inputs)))
        for k, a in enumerate(t.inputs):
            X[:, k] = mg[sp.index(a)].ravel() if a in sp else (tv if a == "t" else 0.5 * sum(domain[a]))
        P = {k: v.reshape(mg[0].shape) for k, v in model(X).items()}
        label = f" at t={tv:.4g}" if tv is not None else ""
        if t.physics == "incompressible-flow" and all(n in P for n in VEL[:len(sp)]):
            b = balances.grid_mass(coords, sp, [P[n] for n in VEL[:len(sp)]])
            key, what, tol = "PC101", "volume flux through the domain boundary", t.tol("mass")
            rel_closed = b["uref"] * b["area"]
        elif t.physics == "heat" and "T" in P and tv is None:
            b = balances.grid_heat(coords, sp, P["T"], params["k"], params.get("q") or 0.0)
            key, what, tol, rel_closed = "PC103", "heat flow through the domain boundary", t.tol("energy"), None
        elif t.physics == "elasticity" and all(n in P for n in DISP[:len(sp)]) and params.get("E") is not None:
            f = np.atleast_1d(np.asarray(params.get("f") if params.get("f") is not None else 0.0, float))
            f = np.zeros(len(sp)) if f.size == 1 and f[0] == 0 else f
            if f.size != len(sp):
                return
            b = balances.grid_force(coords, sp, [P[n] for n in DISP[:len(sp)]], params["E"], params["nu"], f)
            key, what, tol, rel_closed = "PC104", "resultant force on the body", t.tol("momentum"), None
        else:
            return
        rel = abs(b["net"]) / b["total"] if b["total"] > 0 else 0.0
        if worst is None or rel > worst[0]:
            worst = (rel, label, b, key, what, tol, rel_closed)
    if worst:
        _, label, b, key, what, tol, rel_closed = worst
        check_balance(ctx, key, what + label, b, tol, loc, closed_scale=rel_closed,
                      subject={"PC101": "mass", "PC103": "energy", "PC104": "momentum"}[key])


# ---------------------------------------------------------------------------------------------- BC / IC
def _face_points(t: ModelTarget, domain, key: str, n: int = 256):
    k = key.replace(" ", "")
    if "=" not in k:
        return None
    a, v = k.split("=", 1)
    if a not in t.inputs:
        return None
    lo, hi = domain[a]
    val = lo if v == "min" else hi if v == "max" else float(v)
    X = _sample(domain, t.inputs, n, t.seed + 2)
    X[:, t.inputs.index(a)] = val
    return X


def _bc(ctx: Ctx, t: ModelTarget, model: LoadedModel, domain, loc) -> None:
    tol = t.tol("bc")
    for key, vals in t.bc.items():
        X = _face_points(t, domain, key)
        for field, val in vals.items():
            if X is None:
                ctx.emit("PC201", f"{field}@{key}", False, f"boundary '{key}' is not `<input>=<value|min|max>` of "
                                                          f"the model inputs {t.inputs}", loc)
                continue
            P = model(X)
            names = list(_named(P, field)) if field not in P else [field]
            if not names:
                ctx.emit("PC201", f"{field}@{key}", False, f"the model has no output {field}", loc)
                continue
            a = np.column_stack([P[n] for n in names]) if len(names) > 1 else P[names[0]]
            dv = evaluate_expr(val, {n: X[:, k] for k, n in enumerate(t.inputs)}, len(X))
            err = float(np.max(np.abs(a - dv)))
            scale = max(float(np.max(np.abs(a))), float(np.max(np.abs(dv))), 1e-300)
            ok = err / scale <= tol
            ctx.emit("PC201", f"{field}@{key}", ok, f"{field} on {key}: max deviation from the declared value is "
                     f"{fmt(err)} ({fmt(err / scale)} of the scale {'<=' if ok else '>'} {fmt(tol)})", loc,
                     value=err / scale, threshold=tol)


def _ic(ctx: Ctx, t: ModelTarget, model: LoadedModel, domain, loc) -> None:
    tol = t.tol("ic")
    if "t" not in t.inputs:
        for field in t.ic:
            ctx.emit("PC202", field, False, "an initial condition needs a time input `t`", loc)
        return
    X = _face_points(t, domain, "t=min", 512)
    P = model(X)
    for field, val in t.ic.items():
        if field not in P:
            ctx.emit("PC202", field, False, f"the model has no output {field}", loc)
            continue
        dv = evaluate_expr(val, {n: X[:, k] for k, n in enumerate(t.inputs)}, len(X))
        err = float(np.max(np.abs(P[field] - dv)))
        scale = max(float(np.max(np.abs(P[field]))), float(np.max(np.abs(dv))), 1e-300)
        ok = err / scale <= tol
        ctx.emit("PC202", field, ok, f"{field} at t={domain['t'][0]:.4g}: max deviation from the declared initial "
                 f"value is {fmt(err / scale)} of the scale ({'<=' if ok else '>'} {fmt(tol)})", loc,
                 value=err / scale, threshold=tol)


# ---------------------------------------------------------------------------------------------- reference
def _reference(ctx: Ctx, t: ModelTarget, model: LoadedModel) -> None:
    ref = t.reference
    loc = ctx.default_location()
    try:
        cols, _ = read_table(ref.path)
    except Exception as e:
        ctx.emit("PC703", "reference", False, f"cannot read the reference {ctx.cfg.rel(ref.path)}: {e}", loc)
        return
    miss = [a for a in t.inputs if a not in cols]
    if miss:
        ctx.emit("PC703", "reference", False, f"the reference {ctx.cfg.rel(ref.path)} lacks input column(s) "
                                              f"{', '.join(miss)}", loc)
        return
    X = np.column_stack([cols[a] for a in t.inputs])
    ctx.emit("PC703", "reference", True, f"reference {ctx.cfg.rel(ref.path)} read ({len(X)} points)", loc)
    P = model(X)
    tol = ref.rel_l2 if ref.rel_l2 is not None else t.tol("rel-l2")
    names = ref.fields or [o for o in t.outputs if o in cols]
    if not names:
        ctx.emit("PC703", "reference", False, f"the reference has none of the outputs {t.outputs}", loc)
        return
    for n in names:
        if n not in cols or n not in P:
            ctx.emit("PC703", n, False, f"{n} is missing from the {'reference' if n not in cols else 'model'}", loc)
            continue
        e = rel_l2(P[n], cols[n])
        ok = e <= tol
        ctx.emit("PC701", n, ok, f"{n}: relative L2 error {fmt(e)} against {ctx.cfg.rel(ref.path)} "
                 f"({len(X)} points) {'<=' if ok else '>'} {fmt(tol)}", loc, value=e, threshold=tol)
