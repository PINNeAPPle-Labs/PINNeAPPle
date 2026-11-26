"""The official benchmark cases and the protocol that makes their table reproducible (#30, roadmap item B3).

A benchmark case is only a benchmark if the same protocol can be applied to the next case without an argument about
what was measured. That is what lives here, in three parts:

* **the protocol** -- ``PROTOCOL``: the seeds, the training budget, the metrics, how the interval is computed, the
  evaluation points, and the tolerance a reproduction is allowed. It is data, so the page, the runner and the tests
  read the same numbers instead of three copies of them.
* **the cases** -- ``CASES``: three transient problems whose exact solution is a closed form, so the reference is not
  another numerical method that can be wrong in the same direction. Each one trains inside a *window* of the domain
  and is scored inside it (in-domain) and outside it (extrapolation): the second number is the one a surrogate user
  actually needs and the one a leaderboard usually omits.
* **the run** -- ``run_case``: one case, every seed of the protocol, the same budget for all of them, returning the
  per-seed metrics, the bootstrap interval and the cost.

Nothing here trains a competitor or claims a ranking; it measures the library against a closed form and publishes
the number it got, including the ones that are bad (all three extrapolate poorly, and the table says so).

Command line: ``python scripts/benchmark_cases.py --run`` writes ``benchmarks/cases/results.json`` and the page;
``--check`` verifies the declaration and the page without training; ``--reproduce`` runs the cases again and
compares them with what is published.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

RESULTS = Path("benchmarks") / "cases" / "results.json"
PAGE = Path("docs") / "benchmark_cases.md"
SCHEMA = "pinneapple.benchmark_cases/1"

# what a case has to declare to be a case at all: an id, something to solve, a reference that is not the method under
# test, and the window the training stops at
REQUIRED_CASE_FIELDS = ("id", "name", "pde", "reference", "train", "evaluation", "split_coord", "split_at")
# and what the protocol has to fix, so that "the same protocol" is a claim a test can check
REQUIRED_PROTOCOL_FIELDS = ("seeds", "budget", "metrics", "ci", "tolerance")

PROTOCOL: Dict[str, Any] = {
    "seeds": (0, 1, 2),
    "budget": {"epochs": 300, "n_collocation": 512, "hidden_dim": 32, "n_layers": 3},
    "metrics": ("relative_l2", "rmse", "max_abs"),
    "ci": {"method": "bootstrap over the evaluation points", "level": 0.95, "resamples": 2000, "seed": 12345},
    # a reproduction is allowed this much relative drift on a metric; cost is published, never compared, because it
    # is a property of the machine and not of the method
    "tolerance": {"relative_l2": 0.25, "rmse": 0.25, "max_abs": 0.25},
    "compared_columns": ("in_domain", "extrapolation"),
}


@dataclass(frozen=True)
class Case:
    """One official case: what it solves, how the reference is computed, and where training stops."""
    id: str
    name: str
    pde: str
    reference: str
    train: str
    evaluation: str
    split_coord: str                       # the coordinate the training window cuts
    split_at: float                        # training uses split_coord <= split_at
    build: Callable[[], Tuple[Any, np.ndarray, np.ndarray]]   # -> (problem on the training window, X_eval, exact)

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "name": self.name, "pde": self.pde, "reference": self.reference,
                "train": self.train, "evaluation": self.evaluation, "split_coord": self.split_coord,
                "split_at": self.split_at}


def _builder(domain: Mapping[str, Tuple[float, float]], pde: str, params: Mapping[str, float], field: str,
             ic: Callable[[np.ndarray], np.ndarray], walls: Sequence[str]):
    """The problem on the training window, with the IC and a zero Dirichlet on every wall named in ``walls``."""
    from pinneapple_physics.pde_environment.builder import ProblemBuilder

    b = ProblemBuilder(pde).domain(**domain).fields(field).pde(pde, **dict(params)).ic(field=field, value=ic)
    for wall in walls:
        b = b.bc("dirichlet", field=field, value=0.0, on=wall)
    return b.build()


ALPHA = 0.05
NU = 0.01


def _heat_1d():
    import pinneapple as pp

    spec = _builder({"x": (0.0, 1.0), "t": (0.0, 0.5)}, "heat_equation", {"alpha": ALPHA}, "T",
                    lambda X: np.sin(np.pi * X[:, 0]), ("x_min", "x_max"))
    x = np.linspace(0.0, 1.0, 41)
    t = np.linspace(0.0, 1.0, 41)
    X, T = np.meshgrid(x, t, indexing="ij")
    exact = np.exp(-ALPHA * np.pi ** 2 * T) * np.sin(np.pi * X)
    return pp.PhysicalProblem.from_pde_spec(spec), np.stack([X.ravel(), T.ravel()], -1), exact.reshape(-1, 1)


def _heat_2d():
    import pinneapple as pp

    spec = _builder({"x": (0.0, 1.0), "y": (0.0, 1.0), "t": (0.0, 0.4)}, "heat_equation", {"alpha": ALPHA}, "T",
                    lambda X: np.sin(np.pi * X[:, 0]) * np.sin(np.pi * X[:, 1]),
                    ("x_min", "x_max", "y_min", "y_max"))
    g = np.linspace(0.0, 1.0, 21)
    X, Y, T = np.meshgrid(g, g, g, indexing="ij")
    exact = np.exp(-2.0 * ALPHA * np.pi ** 2 * T) * np.sin(np.pi * X) * np.sin(np.pi * Y)
    return (pp.PhysicalProblem.from_pde_spec(spec),
            np.stack([X.ravel(), Y.ravel(), T.ravel()], -1), exact.reshape(-1, 1))


def _burgers_1d():
    import pinneapple as pp
    from pinneapple_physics.closed_form.burgers import burgers_sine_exact

    full = pp.PhysicalProblem.from_preset("burgers_1d")
    nu = float(full.pde.params.get("nu", NU))
    train = dataclasses.replace(full, domain_bounds={**full.domain_bounds, "t": (0.0, 0.5)})
    x = np.linspace(-1.0, 1.0, 129)
    t = np.linspace(0.0, 1.0, 41)
    X, T = np.meshgrid(x, t, indexing="ij")
    exact = burgers_sine_exact(X, T, nu=nu)
    return train, np.stack([X.ravel(), T.ravel()], -1), np.asarray(exact).reshape(-1, 1)


CASES: Tuple[Case, ...] = (
    Case("01", "heat_1d", "alpha T_xx = T_t, alpha = 0.05, T(x, 0) = sin(pi x), T = 0 at x = 0 and x = 1",
         "closed form: exp(-alpha pi^2 t) sin(pi x)", "t in [0, 0.5]", "41 x 41 grid on x, t in [0, 1]", "t", 0.5,
         _heat_1d),
    Case("02", "heat_2d", "alpha (T_xx + T_yy) = T_t, alpha = 0.05, T = sin(pi x) sin(pi y) at t = 0, T = 0 on the walls",
         "closed form: exp(-2 alpha pi^2 t) sin(pi x) sin(pi y)", "t in [0, 0.4]", "21^3 grid on x, y, t in [0, 1]",
         "t", 0.4, _heat_2d),
    Case("03", "burgers_1d", "u_t + u u_x = nu u_xx, nu = 0.01, u(x, 0) = -sin(pi x), u = 0 at x = -1 and x = 1",
         "Cole-Hopf by Gauss-Hermite quadrature (pinneapple_physics.closed_form.burgers)",
         "t in [0, 0.5]", "129 x 41 grid on x, t in [0, 1]", "t", 0.5, _burgers_1d),
)

BY_ID: Dict[str, Case] = {c.id: c for c in CASES}


def get_case(case_id: str) -> Case:
    if case_id not in BY_ID:
        raise KeyError(f"unknown benchmark case {case_id!r}; declared: {sorted(BY_ID)}")
    return BY_ID[case_id]


# ------------------------------------------------------------------ metrics
def _rel_l2(err: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(err) / max(np.linalg.norm(ref), 1e-300))


def bootstrap_ci(err: np.ndarray, ref: np.ndarray, *, level: float = 0.95, resamples: int = 2000,
                 seed: int = 12345) -> Tuple[float, float]:
    """Interval of the relative L2 over the evaluation points, by resampling them.

    The interval says how much the metric depends on *which* points were scored, not on which seed trained the
    network -- the spread across seeds is reported separately, because mixing the two would hide the bigger one.
    """
    err = np.asarray(err, dtype=np.float64).reshape(-1)
    ref = np.asarray(ref, dtype=np.float64).reshape(-1)
    if err.shape != ref.shape or err.size == 0:
        raise ValueError(f"err and ref must be the same non-empty shape, got {err.shape} and {ref.shape}")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, err.size, size=(resamples, err.size))
    stats = np.linalg.norm(err[idx], axis=1) / np.maximum(np.linalg.norm(ref[idx], axis=1), 1e-300)
    lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
    return float(np.quantile(stats, lo)), float(np.quantile(stats, hi))


def _score(pred: np.ndarray, exact: np.ndarray, in_domain: np.ndarray) -> Dict[str, Any]:
    err = pred - exact
    out: Dict[str, Any] = {}
    for where, mask in (("in_domain", in_domain), ("extrapolation", ~in_domain)):
        e, r = err[mask], exact[mask]
        lo, hi = bootstrap_ci(e, r, level=PROTOCOL["ci"]["level"], resamples=PROTOCOL["ci"]["resamples"],
                              seed=PROTOCOL["ci"]["seed"])
        out[where] = {"relative_l2": _rel_l2(e, r), "relative_l2_ci": [lo, hi],
                      "rmse": float(np.sqrt(np.mean(e ** 2))), "max_abs": float(np.max(np.abs(e)))}
    return out


# ------------------------------------------------------------------ one case, all seeds
def run_case(case_id: str, *, seeds: Optional[Iterable[int]] = None, budget: Optional[Mapping[str, Any]] = None,
             solve: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    """Every seed of the protocol on one case, with the same budget for all of them."""
    import time

    import pinneapple as pp

    case = get_case(case_id)
    seeds = tuple(PROTOCOL["seeds"] if seeds is None else seeds)
    budget = dict(PROTOCOL["budget"], **dict(budget or {}))
    solve = solve or pp.solve
    problem, X, exact = case.build()
    in_domain = X[:, list(problem.coords).index(case.split_coord)] <= case.split_at
    runs = []
    for seed in seeds:
        t0 = time.perf_counter()
        solution = solve(problem, "pinn", seed=seed, **budget)
        wall = time.perf_counter() - t0
        pred = np.asarray(solution.predict(X), dtype=np.float64).reshape(exact.shape)
        runs.append({"seed": int(seed), "wall_time_s": wall, "metrics": _score(pred, exact, in_domain)})
    return {"case": case.as_dict(), "budget": budget, "seeds": list(seeds), "runs": runs,
            "summary": _summary(runs)}


def _summary(runs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for where in PROTOCOL["compared_columns"]:
        block: Dict[str, Any] = {}
        for metric in PROTOCOL["metrics"]:
            values = [r["metrics"][where][metric] for r in runs]
            block[metric] = {"mean": float(np.mean(values)), "min": float(np.min(values)),
                             "max": float(np.max(values))}
        out[where] = block
    out["wall_time_s"] = {"mean": float(np.mean([r["wall_time_s"] for r in runs]))}
    return out


def run_all(*, seeds: Optional[Iterable[int]] = None, budget: Optional[Mapping[str, Any]] = None,
            only: Optional[Sequence[str]] = None, solve: Optional[Callable[..., Any]] = None) -> Dict[str, Any]:
    ids = [c.id for c in CASES] if not only else [get_case(i).id for i in only]
    return {"schema": SCHEMA, "protocol": _protocol_json(), "cases": {
        cid: run_case(cid, seeds=seeds, budget=budget, solve=solve) for cid in ids}}


def _protocol_json() -> Dict[str, Any]:
    p = dict(PROTOCOL)
    p["seeds"] = list(PROTOCOL["seeds"])
    p["metrics"] = list(PROTOCOL["metrics"])
    p["compared_columns"] = list(PROTOCOL["compared_columns"])
    return p


# ------------------------------------------------------------------ the table and the page
def _fmt(x: float) -> str:
    return f"{x:.2e}" if (x and abs(x) < 1e-2) or abs(x) >= 1e3 else f"{x:.3f}"


def table(results: Mapping[str, Any]) -> str:
    """The published table: one row per case, mean [min, max] across seeds, and the cost."""
    head = ("| case | in-domain relative L2 | extrapolation relative L2 | rmse (in-domain) | max abs (in-domain) "
            "| wall time per seed |")
    lines = [head, "|" + "---|" * 6]
    for cid, block in results["cases"].items():
        s = block["summary"]
        row = [f"{cid} {block['case']['name']}"]
        for where in PROTOCOL["compared_columns"]:
            m = s[where]["relative_l2"]
            row.append(f"{_fmt(m['mean'])} [{_fmt(m['min'])}, {_fmt(m['max'])}]")
        row.append(_fmt(s["in_domain"]["rmse"]["mean"]))
        row.append(_fmt(s["in_domain"]["max_abs"]["mean"]))
        row.append(f"{s['wall_time_s']['mean']:.1f} s")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def page(results: Mapping[str, Any]) -> str:
    """The documentation page, generated from a results file so the published numbers cannot drift from the run."""
    b = results["protocol"]
    parts = [
        "# Official benchmark cases",
        "",
        "Three problems with a closed-form solution, scored by one protocol: the same seeds, the same training budget,",
        "the same evaluation points and the same metrics. The table below is generated from",
        f"`{RESULTS.as_posix()}` by `python scripts/benchmark_cases.py --docs` -- do not edit it by hand.",
        "",
        "## The protocol",
        "",
        f"* **seeds**: {', '.join(str(s) for s in b['seeds'])} -- the spread across seeds is reported as `[min, max]`, "
        "never mixed with the point interval;",
        f"* **budget**: {b['budget']['epochs']} epochs, {b['budget']['n_collocation']} collocation points, "
        f"network {b['budget']['hidden_dim']} x {b['budget']['n_layers']} -- identical for every seed and every case;",
        f"* **metrics**: {', '.join(b['metrics'])}, computed on the fixed evaluation grid of each case;",
        f"* **interval**: {b['ci']['method']}, {int(b['ci']['level'] * 100)}%, {b['ci']['resamples']} resamples "
        f"(seed {b['ci']['seed']}) -- it says how much the metric depends on the points scored, not on the seed;",
        "* **in-domain** is the window the network was trained on; **extrapolation** is the rest of the domain. "
        "A method that only publishes the first number is hiding the second;",
        "* **cost** is the wall time of one run on one CPU and is informational: it is a property of the machine, so a",
        "  reproduction compares the metrics and never the time.",
        "",
        "## The published table",
        "",
        table(results),
        "",
        "## The cases",
        "",
    ]
    for cid, block in results["cases"].items():
        c = block["case"]
        parts += [f"### Case {cid} -- {c['name']}", "",
                  f"* equation: `{c['pde']}`",
                  f"* reference: {c['reference']}",
                  f"* train: {c['train']} ({c['split_coord']} <= {c['split_at']})",
                  f"* evaluation: {c['evaluation']}",
                  f"* budget: {block['budget']['epochs']} epochs, {block['budget']['n_collocation']} collocation points",
                  ""]
    parts += ["## Reproducing it", "",
              "```bash",
              "python scripts/benchmark_cases.py --run      # trains every case and rewrites the results and this page",
              "python scripts/benchmark_cases.py --reproduce  # trains again and compares against the table above",
              "python scripts/benchmark_cases.py --check      # verifies the declaration and this page, no training",
              "```",
              "",
              f"A reproduction matches when every metric is within "
              f"{int(max(b['tolerance'].values()) * 100)}% of the published value; the runner says which cell did not.",
              ""]
    return "\n".join(parts)


# ------------------------------------------------------------------ gates
def check(root: Optional[Path] = None) -> List[str]:
    """Every problem with the declaration and the published page. Empty list means the gate passes."""
    root = Path(root or ".")
    problems: List[str] = []
    for case in CASES:
        d = case.as_dict()
        for key in REQUIRED_CASE_FIELDS:
            if key not in d or d[key] in (None, "", ()):
                problems.append(f"case {case.id}: '{key}' is missing from the declaration")
    for key in REQUIRED_PROTOCOL_FIELDS:
        if key not in PROTOCOL:
            problems.append(f"protocol: '{key}' is missing")
    if len(seeds := tuple(PROTOCOL.get("seeds") or ())) < 2:
        problems.append(f"protocol: at least two seeds are needed to report a spread, got {seeds}")
    if not set(PROTOCOL["metrics"]) >= {"relative_l2"}:
        problems.append("protocol: relative_l2 is the metric every case must report")
    results_path, page_path = root / RESULTS, root / PAGE
    if not results_path.is_file():
        problems.append(f"{RESULTS.as_posix()} is missing: run scripts/benchmark_cases.py --run")
        return problems
    results = json.loads(results_path.read_text(encoding="utf-8"))
    if results.get("schema") != SCHEMA:
        problems.append(f"{RESULTS.as_posix()} has schema {results.get('schema')!r}, expected {SCHEMA!r}")
    if set(results["cases"]) != set(BY_ID):
        problems.append(f"{RESULTS.as_posix()} covers {sorted(results['cases'])}, the declaration has {sorted(BY_ID)}")
    if not page_path.is_file():
        problems.append(f"{PAGE.as_posix()} is missing: run scripts/benchmark_cases.py --docs")
    elif page_path.read_text(encoding="utf-8") != page(results):
        problems.append(f"{PAGE.as_posix()} is not what the results file generates: run --docs")
    return problems


def reproduces(published: Mapping[str, Any], fresh: Mapping[str, Any]) -> List[str]:
    """The cells of ``fresh`` that differ from ``published`` by more than the protocol's tolerance."""
    problems: List[str] = []
    tol = PROTOCOL["tolerance"]
    for cid, block in published["cases"].items():
        new = (fresh.get("cases") or {}).get(cid)
        if new is None:
            problems.append(f"case {cid}: missing from the reproduction")
            continue
        for where in PROTOCOL["compared_columns"]:
            for metric in PROTOCOL["metrics"]:
                want = block["summary"][where][metric]["mean"]
                got = new["summary"][where][metric]["mean"]
                allowed = tol.get(metric, 0.25)
                if want == 0:
                    continue
                drift = abs(got - want) / abs(want)
                if drift > allowed:
                    problems.append(f"case {cid} {where} {metric}: published {_fmt(want)}, reproduced {_fmt(got)} "
                                    f"({drift:.0%} drift, allowed {allowed:.0%})")
    return problems


def write(results: Mapping[str, Any], *, root: Optional[Path] = None) -> Tuple[Path, Path]:
    """Write the results file and the page. Returns both paths."""
    root = Path(root or ".")
    results_path, page_path = root / RESULTS, root / PAGE
    results_path.parent.mkdir(parents=True, exist_ok=True)
    page_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    page_path.write_text(page(results), encoding="utf-8")
    return results_path, page_path
