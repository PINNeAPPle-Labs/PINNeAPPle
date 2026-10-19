"""Shared pieces of the built-in nodes: problem documents, solution artifacts, exact solutions, plots."""
from __future__ import annotations

from typing import Any

import numpy as np

from pinneapple_core.workflows import StepFailed

from ..runtime import NodeContext, load_artifact


def coords_of(geometry: dict[str, Any]) -> list[str]:
    return list(geometry["coords"])


def bounds_of(problem: dict[str, Any]) -> dict[str, tuple[float, float]]:
    g = problem["geometry"]
    b = {c: tuple(map(float, v)) for c, v in g["bounds"].items()}
    if g.get("time"):
        b["t"] = tuple(map(float, g["time"]))
    return b


def system_of(problem: dict[str, Any], *, coefficients: dict | None = None):
    """The parsed :class:`~pinneapple_physics.symbolic_pde.text.EquationSystem` of a problem document."""
    from pinneapple_physics.symbolic_pde.text import EquationSyntaxError, parse_system
    eq = problem["equations"]
    try:
        return parse_system(eq["pdes"], eq.get("conditions") or [], coords=coords_of(problem["geometry"]),
                            fields=eq.get("fields") or ["u"], coefficients={**eq.get("coefficients", {}),
                                                                            **(coefficients or {})},
                            unknowns=eq.get("unknowns") or [])
    except EquationSyntaxError as exc:
        raise StepFailed(f"equations: {exc}") from None


def need_problem(v: Any) -> dict[str, Any]:
    v = load_artifact(v)
    if not isinstance(v, dict) or "geometry" not in v:
        raise StepFailed("expected a problem (geometry + equations)")
    if "equations" not in v:
        raise StepFailed("the problem has no equations: connect an equations node")
    return v


def grid_points(axes: dict[str, np.ndarray], coords: list[str]) -> np.ndarray:
    mesh = np.meshgrid(*[axes[c] for c in coords], indexing="ij")
    return np.stack([m.ravel() for m in mesh], 1)


def save_solution(ctx: NodeContext, name: str, axes: dict[str, np.ndarray], values: np.ndarray, coords: list[str],
                  fields: list[str], method: str, label: str = "", extra: dict | None = None) -> dict[str, Any]:
    """``values``: shape ``(*[len(axes[c]) for c in coords], n_fields)``."""
    values = np.asarray(values, float)
    shape = tuple(len(axes[c]) for c in coords)
    if values.shape != (*shape, len(fields)):
        values = values.reshape(*shape, len(fields))
    a = ctx.save_npz(name, values=values, **{f"axis_{c}": np.asarray(axes[c], float) for c in coords})
    a.update({"coords": list(coords), "fields": list(fields), "method": method, "label": label or method,
              **(extra or {})})
    return a


def load_solution(v: Any) -> dict[str, Any]:
    if not (isinstance(v, dict) and v.get("kind") == "npz" and "coords" in v):
        raise StepFailed("expected a solution")
    z = load_artifact(v)
    axes = {c: z[f"axis_{c}"] for c in v["coords"]}
    return {"axes": axes, "values": z["values"], "coords": v["coords"], "fields": v["fields"],
            "method": v.get("method", ""), "label": v.get("label", v.get("method", "")),
            "own_checks": list(v.get("own_checks") or [])}


def exact_function(problem: dict[str, Any]):
    """``X (N, d) -> (N, n_fields)`` for the problem's exact solution, or None."""
    eq = problem["equations"]
    ex = eq.get("exact")
    if not ex:
        return None
    coords = coords_of(problem["geometry"])
    coeffs = dict(eq.get("coefficients") or {})
    if ex == "cole_hopf":
        from pinneapple_analysis.validation.pinn import cole_hopf
        nu = float(coeffs.get("nu", 0.01 / np.pi))
        ix, it = coords.index("x"), coords.index("t")
        return lambda X: cole_hopf(X[:, ix], X[:, it], nu)[:, None]
    from pinneapple_physics.symbolic_pde.text import (
        Context,
        EquationSyntaxError,
        compile_expr,
        parse_expression,
    )
    try:
        expr, used = parse_expression(ex, Context(coords, eq.get("fields") or ["u"]))
    except EquationSyntaxError as exc:
        raise StepFailed(f"exact solution: {exc}") from None
    missing = used - set(coeffs)
    if missing:
        raise StepFailed(f"exact solution uses coefficients without a value: {sorted(missing)}")
    import sympy as sp
    expr = expr.xreplace({sp.Symbol(k, real=True): sp.Float(v) for k, v in coeffs.items()})
    expr = expr.xreplace({s: sp.Symbol(s.name) for s in expr.free_symbols})
    f = compile_expr(expr, "numpy")

    def fn(X):
        env = {c: X[:, i] for i, c in enumerate(coords)}
        return np.broadcast_to(np.asarray(f(env), float), (len(X),)).reshape(-1, 1)
    return fn


def interpolate(sol: dict[str, Any], X: np.ndarray) -> np.ndarray:
    """Values of a grid solution at points ``X`` (coordinates in ``sol['coords']`` order), linear."""
    from scipy.interpolate import RegularGridInterpolator
    axes = [sol["axes"][c] for c in sol["coords"]]
    out = []
    for j in range(sol["values"].shape[-1]):
        f = RegularGridInterpolator(axes, sol["values"][..., j], bounds_error=False, fill_value=None)
        out.append(f(X))
    return np.stack(out, 1)


def eval_axes(problem: dict[str, Any], n: int | None = None, n_t: int | None = None) -> dict[str, np.ndarray]:
    """Evaluation grid of a problem: ``n`` (and ``n_t``) points per axis when given, else the attached mesh, else a
    default per dimension."""
    mesh = problem.get("mesh") or {}
    g = problem["geometry"]
    d = len(g["bounds"])
    nn = mesh.get("n") or {}
    default = {1: 201, 2: 65, 3: 25}[d]
    axes = {c: np.linspace(*map(float, b), int(n or nn.get(c, default))) for c, b in g["bounds"].items()}
    if g.get("time"):
        axes["t"] = np.linspace(*map(float, g["time"]), int(n_t or mesh.get("n_t") or 41))
    return {c: axes[c] for c in coords_of(g)}


def plot_solution(sol: dict[str, Any], title: str = "", field: int = 0):
    """A figure of a solution: curves over time (1-D + t), a map (2-D, last time), mid-plane slices (3-D)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    coords, V = sol["coords"], sol["values"][..., field]
    spatial = [c for c in coords if c != "t"]
    has_t = "t" in coords
    if has_t:
        V = np.moveaxis(V, coords.index("t"), 0)
        times = sol["axes"]["t"]
    if len(spatial) == 1:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        x = sol["axes"][spatial[0]]
        if has_t:
            for k in np.linspace(0, len(times) - 1, 5).astype(int):
                ax.plot(x, V[k], label=f"t = {times[k]:.3g}")
            ax.legend(fontsize=8)
        else:
            ax.plot(x, V)
        ax.set_xlabel(spatial[0])
        ax.set_ylabel(sol["fields"][field])
    elif len(spatial) == 2:
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        F = V[-1] if has_t else V
        x, y = (sol["axes"][c] for c in spatial)
        im = ax.pcolormesh(x, y, F.T, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label=sol["fields"][field])
        ax.set_xlabel(spatial[0])
        ax.set_ylabel(spatial[1])
        ax.set_aspect("equal")
        if has_t:
            title = f"{title} (t = {times[-1]:.3g})"
    else:
        F = V[-1] if has_t else V
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
        for i, (ax, c) in enumerate(zip(axs, spatial, strict=True)):
            sl = [slice(None)] * 3
            sl[i] = F.shape[i] // 2
            others = [s for s in spatial if s != c]
            a, b = (sol["axes"][s] for s in others)
            im = ax.pcolormesh(a, b, F[tuple(sl)].T, shading="auto", cmap="viridis")
            ax.set_title(f"{c} = {sol['axes'][c][F.shape[i] // 2]:.2f}", fontsize=9)
            ax.set_xlabel(others[0])
            ax.set_ylabel(others[1])
            ax.set_aspect("equal")
        fig.colorbar(im, ax=list(axs), label=sol["fields"][field])
    fig.suptitle(title or sol["label"], fontsize=10)
    return fig
