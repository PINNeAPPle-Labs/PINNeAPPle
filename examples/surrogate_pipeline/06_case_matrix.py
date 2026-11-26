"""06 — The case matrix: one pipeline, every kind of problem and every kind of model it covers.

What this demonstrates
----------------------
=============================  ==============================  ==========================================
case                           what varies                     model
=============================  ==============================  ==========================================
flow · physics only            nothing: the equations          PINN, scored against Cole-Hopf
flow · dataset generated       18 reference solves of the       study over GP / MLP
                               Burgers solution in the viscosity
flow · FNO                      the same campaign, as fields    Fourier neural operator (64 points)
flow · DeepONet                 the same campaign, as fields    DeepONet
structural · physics only      Young's modulus of the material PINN on a bar under distributed load,
                               (steel, from the catalogue)     scored against the closed form
geometry · generated           a cylinder built by the run      a parametric surrogate over the same
                                                               dataset; the surface passes the geometry gate
geometry · imported            a hull from data/geometries     the same, on an STL nobody in this repo built
dataset · imported             a dataset.npz from case 2       the same surrogate, no solve at all
=============================  ==============================  ==========================================

Every run writes its figures under ``_out/matrix/<name>/figures`` (training loss, parity, error map over a
held-out field, the geometry itself). Nothing here is cached: the numbers printed come from the run above them.

    python examples/surrogate_pipeline/06_case_matrix.py                 # the whole matrix (~3 min)
    python examples/surrogate_pipeline/06_case_matrix.py --only fno      # one case
    python examples/surrogate_pipeline/06_case_matrix.py --fast          # smallest budgets

The last three cases read ``_out/matrix/burgers_in_viscosity/data/dataset.npz``, the dataset of case 2: run with
``--only`` after it has run once, or give them a path of their own.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import pinneapple as pp  # noqa: E402

OUT = Path(__file__).resolve().parent / "_out" / "matrix"
HULL = REPO / "data" / "geometries" / "wigley_hull" / "wigley_hull.light.stl"

#: ``probes`` on a lattice of 64 points and one output name make the dataset a *field* dataset, which is what the
#: neural operators take (``fno`` and ``deeponet`` map a parameter to a whole field, not to a handful of scalars).
FIELD_DATA = {"outputs": ["u"], "probes": {"x": 64}, "probe_at": {"t": 1.0}}


def run_block(*, out_dir: Path = OUT) -> dict:
    return {"out_dir": str(out_dir), "resume": False}


# --------------------------------------------------------------------------------------------------------- cases
def case_flow_pinn(fast: bool):
    return pp.SurrogatePipeline.from_preset(
        "burgers_physics_only", execution=run_block(),
        training={"epochs": 400 if fast else 1200, "n_collocation": 1024 if fast else 2048},
        export={"enabled": False, "formats": []}, verification={"fail_policy": "report"})


def case_flow_dataset(fast: bool):
    return pp.SurrogatePipeline.from_preset(
        "burgers_in_viscosity", execution=run_block(),
        sampling={"design_sampling": {"method": "lhs", "n_samples": 12 if fast else 18},
                  "validation_sampling": {"test_fraction": 0.2, "val_fraction": 0.15, "shift": "auto"}},
        training={"budget_seconds": 1.0 if fast else 3.0}, verification={"fail_policy": "report"})


def case_flow_fno(fast: bool, methods=("fno",)):
    name = "flow_" + "+".join(methods)
    return pp.SurrogatePipeline.from_preset(
        "burgers_in_viscosity", name=name, execution=run_block(), data=FIELD_DATA,
        sampling={"design_sampling": {"method": "lhs", "n_samples": 16},
                  "validation_sampling": {"test_fraction": 0.2, "val_fraction": 0.15, "shift": "auto"}},
        architecture={"family": "auto", "candidates": list(methods)},
        training={"budget_seconds": 2.0 if fast else 5.0}, verification={"fail_policy": "report"})


def case_structural(fast: bool):
    """A 0.5 m steel bar, fixed at the root, carrying 100 kN/m; E comes from the material catalogue."""
    length, area, load = 0.5, 1.0e-4, 1.0e5
    return pp.SurrogatePipeline.from_config({
        "name": "structural_bar",
        "description": "axial bar under a distributed load, E from the material catalogue",
        "problem": {"source": "symbolic", "coords": ["x"], "fields": ["u"],
                    "domain_bounds": {"x": [0.0, length]},
                    "equations": ["lap(u) + q / (youngs_modulus * A)"],
                    "constants": {"q": load, "youngs_modulus": 1.0, "A": area, "L": length},
                    "exact_expression": "q / (2*youngs_modulus*A) * (2*L*x - x**2)",
                    "conditions": [{"kind": "dirichlet", "on": "x_min", "equation": "u = 0"},
                                   {"kind": "neumann", "on": "x_max", "equation": "d(u, x) = 0"}]},
        "materials": {"materials": {"bar": "steel_1018"}},
        "data": {"source": "physics_only"},
        "training": {"epochs": 300 if fast else 900, "lbfgs_steps": 50 if fast else 200,
                     "n_collocation": 512, "lr": 2e-3},
        "loss": {"weights": {"w_pde": 1.0, "w_bc": 20.0}},
        "verification": {"mode": "balanced", "fail_policy": "report"},
        "export": {"enabled": False, "formats": []},
        "execution": run_block(),
    })


def case_geometry(fast: bool, *, generated: bool):
    """The same surrogate as case 2 (dataset imported), with a surface the geometry gate has to accept."""
    geometry = ({"source": "builder", "builder": "cylinder", "units": "m",
                 "parameters": {"radius": 0.05, "length": 0.3, "axis": "y", "n": 36}}
                if generated else {"source": "file", "path": str(HULL), "units": "m"})
    return pp.SurrogatePipeline.from_config({
        "name": "geometry_generated" if generated else "geometry_imported",
        "description": ("a cylinder built by the geometry block (100 mm x 300 mm)"
                        if generated else f"a hull from an STL nobody here built: {HULL.name}"),
        "problem": {"source": "preset", "preset": "burgers_1d"},
        "parameters": {"parameters": {"nu": [0.005, 0.06]}},
        "geometry": geometry,
        "data": {"source": "file",
                 "path": str(OUT / "burgers_in_viscosity" / "data" / "dataset.npz")},
        "architecture": {"family": "ridge", "candidates": []},
        "training": {"budget_seconds": 0.5},
        "verification": {"mode": "balanced", "fail_policy": "report"},
        "execution": run_block(),
    })


def case_dataset_imported(fast: bool):
    return pp.SurrogatePipeline.from_config({
        "name": "dataset_imported",
        "description": "the dataset of case 2, read back from disk: no reference solve at all",
        "problem": {"source": "preset", "preset": "burgers_1d"},
        "parameters": {"parameters": {"nu": [0.005, 0.06]}},
        "data": {"source": "file",
                 "path": str(OUT / "burgers_in_viscosity" / "data" / "dataset.npz")},
        "architecture": {"family": "ridge", "candidates": []},
        "training": {"budget_seconds": 0.5},
        "verification": {"mode": "balanced", "fail_policy": "report"},
        "execution": run_block(),
    })


CASES = {
    "flow-pinn": ("flow mechanics, physics only: a PINN against Cole-Hopf", case_flow_pinn),
    "flow-dataset": ("flow mechanics, dataset generated: a study over GP and MLP", case_flow_dataset),
    "fno": ("flow mechanics, neural operator: FNO on a 64-point field", case_flow_fno),
    "deeponet": ("flow mechanics, neural operator: DeepONet on the same field",
                 lambda fast: case_flow_fno(fast, methods=("deeponet",))),
    "structural": ("structural mechanics, physics only: a steel bar with an exact solution", case_structural),
    "geometry-generated": ("geometry built here: a cylinder through the geometry gate",
                           lambda fast: case_geometry(fast, generated=True)),
    "geometry-imported": ("geometry imported: a hull from data/geometries",
                          lambda fast: case_geometry(fast, generated=False)),
    "dataset-imported": ("dataset imported: the campaign of case 2, no solve", case_dataset_imported),
}


def headline(result) -> str:
    """The one number that says whether the model is worth anything, wherever the run recorded it."""
    metrics = result.metrics or {}
    model_gate = result.gate("model")
    if model_gate is not None and model_gate.metrics.get("relative_l2") is not None:
        value = float(model_gate.metrics["relative_l2"])
        if metrics.get("relative_l2") is not None:
            return f"relative L2 {value:.4g} against {metrics.get('reference', 'analytic')}"
        return f"relative L2 {value:.4g} against the exact solution"
    if metrics.get("ranking"):
        best = metrics["ranking"][0]
        return f"{best['method']} held-out nrmse {best['error']:.4g}"
    if metrics.get("optimization"):
        return f"best of {metrics['optimization']['n_trials']} trials: {metrics['optimization']['best']}"
    return "no held-out error recorded"


def main() -> int:
    parser = argparse.ArgumentParser(description="the case matrix")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--only", default="all", choices=["all", *CASES])
    args = parser.parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    for key, (description, build) in CASES.items():
        if args.only not in ("all", key):
            continue
        print(f"\n\033[1m{key}\033[0m — {description}")
        print("─" * (len(key) + len(description) + 4))
        pipeline = build(args.fast)
        try:
            result = pipeline.run(verbose=True)
        except Exception as exc:                          # one failing case must not stop the matrix
            print(f"  case failed: {type(exc).__name__}: {exc}")
            continue
        print(f"  -> {result.summary()}")
        print(f"  -> {headline(result)}")
        gates = {g.gate: g.status for g in result.verification if g.status not in ("NOT_APPLICABLE",)}
        print(f"  -> gates {gates}")
        figures = (result.metrics.get("figures") or {}).get("figures") or []
        for figure in figures:
            print(f"  -> figure {Path(figure).relative_to(REPO) if str(figure).startswith(str(REPO)) else figure}")
    print(f"\ndone in {time.perf_counter() - started:.1f} s; runs and figures under {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
