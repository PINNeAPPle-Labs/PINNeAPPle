"""Run the verification layer over the controlled corpus and record every outcome.

For each case of :mod:`corpus` this writes the prediction, the reference and a
``pinneapple-check.toml``, runs ``pinneapple check`` on that directory, and stores every outcome
(rule, subject, passed, measured value, threshold, severity). It also computes a tool-independent
statistic, the second-order central-difference PDE residual on the case grid, normalised by the
residual of the exact reference on the same grid, so that the paper can separate what the tool
reports from what the underlying physics says.

Usage::

    python research/arxiv_paper/experiments/verification/run.py [--out results/verification.json]
"""
from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import corpus  # noqa: E402

#: Rules the cases are checked with. They are the ones that apply to a field on a grid.
RULES = ("PC201", "PC202", "PC301", "PC401", "PC403", "PC701", "PC704")


# --------------------------------------------------------------------------- tool-independent residual
def _crop(a: np.ndarray, m: int = 2) -> np.ndarray:
    return np.asarray(a, float)[(slice(m, -m),) * np.ndim(a)]


def fd_residual(problem: str, field: np.ndarray, grid: corpus.Grid) -> np.ndarray:
    """PDE residual of ``field`` by second-order finite differences on the case grid (tool-independent).

    The array is cropped by two points on every axis and the derivatives use ``np.gradient`` with
    ``edge_order=2``, so the operator is the same for a prediction and for the reference it is
    compared against. The grid step is the case's own spacing, which is reported with each case.
    """
    axis = [float(v[1] - v[0]) for v in grid.values]
    u = _crop(field)

    def d1(k: int) -> np.ndarray:
        return np.gradient(u, axis[k], axis=k, edge_order=2)

    def d2(k: int) -> np.ndarray:
        return np.gradient(d1(k), axis[k], axis=k, edge_order=2)

    X = tuple(_crop(m) for m in grid.mesh())
    p = corpus.BY_NAME[problem]()
    if problem == "poisson":
        return d2(0) + d2(1) - p.source(X)
    if problem == "steady_heat":
        return d2(0) + d2(1) + 1.0
    if problem == "heat2d":
        return d1(2) - (d2(0) + d2(1))
    if problem == "helmholtz":
        return d2(0) + d2(1) + p.k ** 2 * u - p.source(X)
    if problem == "burgers":
        return d1(1) + u * d1(0) - p.nu * d2(0)
    if problem == "allen_cahn":
        return d1(1) - p.d * d2(0) - 5.0 * u + 5.0 * u ** 3
    raise KeyError(problem)


def residual_scale(problem: str, ref: np.ndarray, grid: corpus.Grid) -> float:
    """RMS of the equation's own terms at the reference, i.e. the size a residual is measured against.

    This is the same normalisation idea the checker uses (its ``Equation.scale``): the residual is
    divided by the magnitude of the terms that must balance, not by the reference's residual, which
    is at machine precision for a manufactured solution and would make every ratio meaningless.
    """
    axis = [float(v[1] - v[0]) for v in grid.values]
    u = _crop(ref)

    def d1(k):
        return np.gradient(u, axis[k], axis=k, edge_order=2)

    def d2(k):
        return np.gradient(d1(k), axis[k], axis=k, edge_order=2)

    p = corpus.BY_NAME[problem]()
    if problem == "poisson":
        terms = d2(0) + d2(1)
    elif problem == "steady_heat":
        terms = d2(0) + d2(1)
    elif problem == "heat2d":
        terms = d2(0) + d2(1)
    elif problem == "helmholtz":
        terms = d2(0) + d2(1) + p.k ** 2 * u
    elif problem == "burgers":
        terms = u * d1(0) - p.nu * d2(0)
    elif problem == "allen_cahn":
        terms = p.d * d2(0) + 5.0 * u - 5.0 * u ** 3
    else:
        raise KeyError(problem)
    return float(np.sqrt(np.nanmean(np.asarray(terms, float) ** 2)))


def residual_stat(problem: str, field: np.ndarray, ref: np.ndarray, grid: corpus.Grid) -> dict[str, float]:
    """Relative residual of a prediction, its reference floor, and the grid step it was taken at."""
    r = np.asarray(fd_residual(problem, field, grid), float)
    r0 = np.asarray(fd_residual(problem, ref, grid), float)
    scale = residual_scale(problem, ref, grid)
    rms = float(np.sqrt(np.nanmean(r ** 2))) if np.isfinite(r).any() else float("nan")
    floor = float(np.sqrt(np.nanmean(r0 ** 2))) if np.isfinite(r0).any() else float("nan")
    return {
        "residual_rms": rms,
        "reference_residual_rms": floor,
        "residual_scale": scale,
        "residual_relative": (rms / scale) if scale > 0 else float("nan"),
        "reference_residual_relative": (floor / scale) if scale > 0 else float("nan"),
        "residual_grid_step": float(min(v[1] - v[0] for v in grid.values)),
    }


# --------------------------------------------------------------------------- the run
def run_case(check, problem: corpus.Problem, case: corpus.Case, root: Path) -> dict:
    d = corpus.write_case(problem, case, root)
    t0 = time.perf_counter()
    report = check.run([str(d)], config=str(d / "pinneapple-check.toml"), select=list(RULES),
                       use_baseline=False)
    elapsed = time.perf_counter() - t0
    outcomes = [{
        "rule": o.rule, "name": o.title, "subject": o.subject, "passed": bool(o.passed),
        "severity": o.severity, "message": o.message,
        "value": _num(o.value), "threshold": _num(o.threshold),
    } for o in report.outcomes]
    return {
        "case_id": case.case_id, "problem": case.problem, "family": case.family, "defect": case.defect,
        "expressible": bool(problem.expressible),
        "target_rel_l2": case.target_rel_l2,
        "measured_rel_l2": _num(case.extra["reference_rel_l2"]),
        "nans": case.nans,
        "check_seconds": elapsed,
        "outcomes": outcomes,
        "flagged_errors": sorted({o["rule"] for o in outcomes if o["rule"] in RULES
                                  and not o["passed"] and o["severity"] == "error"}),
        "flagged_warnings": sorted({o["rule"] for o in outcomes if o["rule"] in RULES
                                    and not o["passed"] and o["severity"] == "warning"}),
    }


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results" / "verification.json")
    ap.add_argument("--work", type=Path, default=HERE / "_work")
    ap.add_argument("--only", nargs="*", default=None, help="problem names (default: all)")
    args = ap.parse_args()

    import pinneapple_check as check

    args.work.mkdir(parents=True, exist_ok=True)
    names = args.only or list(corpus.BY_NAME)
    records, t0 = [], time.perf_counter()
    for name in names:
        problem = corpus.BY_NAME[name]()
        ref = np.asarray(problem.exact(problem.grid.mesh()), float)
        for case in corpus.build_cases(problem):
            rec = run_case(check, problem, case, args.work)
            rec["expressible"] = bool(problem.expressible)
            rec["residual"] = residual_stat(name, case.field, ref, problem.grid)
            records.append(rec)
            print(f"{rec['case_id']:34s} rel_l2={rec['measured_rel_l2'] or 0:9.3g} "
                  f"err={','.join(rec['flagged_errors']) or '-':22s} "
                  f"warn={','.join(rec['flagged_warnings']) or '-':10s} "
                  f"rho={rec['residual']['residual_relative']:9.3g} "
                  f"(floor {rec['residual']['reference_residual_relative']:.2g})", flush=True)

    payload = {
        "schema": "pinneapple-arxiv-verification/v1",
        "generated_by": "research/arxiv_paper/experiments/verification/run.py",
        "elapsed_s": time.perf_counter() - t0,
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "pinneapple": _version("pinneapple"), "numpy": np.__version__,
        },
        "rules": list(RULES),
        "cases": records,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"\n{len(records)} cases -> {args.out}")
    return 0


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
