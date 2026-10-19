"""Case files for a planned simulation campaign: one directory per case, ready for the solver.

=====================================  ==============================================================================
``write_case_tables``                  ``cases.csv`` / ``.parquet`` / ``.json`` with the parameter space as sidecar
``write_python_runner``                ``run_cases.py``: loops over the cases, calls your ``simulate``, writes results
``write_openfoam_airfoil``             OpenFOAM simpleFoam cases of CST airfoils (``pinneapple_design.aero.case``)
``write_openfoam_from_base``           copies of your base case with dictionary edits per case
                                       (``openfoam.campaign.edit_file``) and the ``plan.json`` of
                                       ``python -m pinneapple_simulation.external_solvers.openfoam.campaign``
``write_calculix_cantilever``          CalculiX decks of a tip-loaded cantilever (``calculix.study.FEModel``)
``write_calculix_from_template``       your ``.inp`` with ``{parameter}`` placeholders filled per case
``write_campaign``                     all of the above that apply, plus the contracts (``space.json``,
                                       ``study.json``, ``dataset.skeleton.json``) and a README
=====================================  ==============================================================================

Presets used by the Campaign Planner app: :func:`airfoil_space` (+ :func:`airfoil_feasible`,
:func:`airfoil_existing`: the 436 OpenFOAM runs of the zoo dataset ``airfoil_rans_cst_re4e6``) and
:func:`cantilever_space`.
"""
from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_data.contracts import Parameter, ParameterSpace, write_table

# Typical room-temperature values (ASM Handbook vol. 2 / MatWeb datasheets): Young's modulus, Poisson ratio, density.
MATERIALS = {
    "steel_S355": {"E": 210e9, "nu": 0.30, "density": 7850.0},
    "aluminium_6061_T6": {"E": 68.9e9, "nu": 0.33, "density": 2700.0},
    "titanium_Ti6Al4V": {"E": 113.8e9, "nu": 0.342, "density": 4430.0},
}


def _params(row: Mapping[str, Any], names: Sequence[str]) -> dict[str, Any]:
    out = {}
    for n in names:
        v = row[n]
        out[n] = v.item() if isinstance(v, np.generic) else v
    return out


# ------------------------------------------------------------------------------------------------- tables / Python
def write_case_tables(cases, out_dir: str | Path, space: ParameterSpace | None = None, stem: str = "cases",
                      formats: Sequence[str] = ("csv", "parquet", "json")) -> list[Path]:
    out_dir = Path(out_dir)
    written = []
    for fmt in formats:
        if fmt == "parquet":
            try:
                import pyarrow  # noqa: F401
            except ImportError:
                continue
        written.append(write_table(cases, out_dir / f"{stem}.{fmt}", space))
    return written


PYTHON_RUNNER = '''"""Run the planned cases with your own simulation code.

Fill in simulate(): it gets one case (a dict with case_id and the parameters) and returns the outputs as a dict.
Results go to results.csv, one row per case, with the status and the error of failed cases; cases already in
results.csv with status "done" are skipped, so the script can be re-run after an interruption.
"""
import csv
import os
import time
import traceback

OUTPUTS = {outputs!r}
PARAMETERS = {parameters!r}


def simulate(case):
    raise NotImplementedError("call your solver here and return {{output name: value}}")


def main():
    with open("cases.csv", newline="") as f:
        cases = list(csv.DictReader(f))
    done = set()
    if os.path.exists("results.csv"):
        with open("results.csv", newline="") as f:
            done = {{r["case_id"] for r in csv.DictReader(f) if r.get("status") == "done"}}
    new = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case_id", *PARAMETERS, *OUTPUTS, "status", "seconds", "error"])
        if new:
            w.writeheader()
        for case in sorted(cases, key=lambda c: int(c.get("priority") or 0)):
            if case["case_id"] in done:
                continue
            t0, row = time.time(), {{k: case[k] for k in ["case_id", *PARAMETERS]}}
            try:
                row.update(simulate(case), status="done")
            except Exception as exc:
                row.update(status="failed", error=f"{{type(exc).__name__}}: {{exc}}")
                traceback.print_exc()
            row["seconds"] = round(time.time() - t0, 3)
            w.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
'''


def write_python_runner(out_dir: str | Path, parameters: Sequence[str], outputs: Sequence[str] = ()) -> Path:
    p = Path(out_dir) / "run_cases.py"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(PYTHON_RUNNER.format(outputs=list(outputs), parameters=list(parameters)))
    return p


# ------------------------------------------------------------------------------------------------- airfoil preset
AIRFOIL_REYNOLDS = 4e6


def airfoil_space(budget: int = 40) -> ParameterSpace:
    """The parameter space of the airfoil campaign: 6 CST weights (thickness T0..T2, camber M0..M2, bounds of
    ``pinneapple_design.aero.geometry.BOUNDS``) and the angle of attack, at Re = 4e6."""
    from pinneapple_design.aero.geometry import BOUNDS, LABELS, PARAMS
    params = [Parameter(name=n, low=float(lo), high=float(hi), group="geometry", description=LABELS[n])
              for n, (lo, hi) in zip(PARAMS, BOUNDS, strict=True)]
    params.append(Parameter(name="alpha_deg", low=-2.0, high=14.0, unit="deg", group="operating",
                            description="angle of attack"))
    params.append(Parameter(name="reynolds", kind="constant", value=AIRFOIL_REYNOLDS, group="operating",
                            description="chord Reynolds number"))
    return ParameterSpace(name="airfoil_cst_re4e6", parameters=params, budget=budget,
                          description="CST airfoil at Re = 4e6, k-omega SST (simpleFoam)")


def airfoil_feasible(cases) -> np.ndarray:
    """Shapes whose surfaces do not cross and stay smooth (``pinneapple_design.aero.geometry.valid``)."""
    from pinneapple_design.aero.geometry import PARAMS, valid
    return np.array([valid(row)[0] for row in np.asarray(cases[list(PARAMS)], float)], dtype=bool)


def airfoil_existing():
    """The 436 OpenFOAM runs already in the zoo dataset ``airfoil_rans_cst_re4e6`` (340 train, 60 test, 36 NACA
    reference polars), as one DataFrame with ``run_id``, the 7 inputs, ``reynolds``, Cl, Cd, Cm and the split."""
    import pandas as pd

    from pinneapple_hub.zoo.datasets import load
    frames = []
    for split in ("train", "test", "naca_reference"):
        df = load("airfoil_rans_cst_re4e6", split=split).to_pandas()
        df["split"] = split
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    df["reynolds"] = AIRFOIL_REYNOLDS
    return df


def write_openfoam_airfoil(cases, out_dir: str | Path, reynolds: float = AIRFOIL_REYNOLDS, iterations: int = 1200,
                           limit: int | None = None) -> list[Path]:
    """One simpleFoam case per row (``cases/<case_id>/``) with ``params.json``, ``coefficients.py`` and ``Allrun``.
    ``limit`` writes only the first cases (each is ~4 MB with its mesh)."""
    from pinneapple_design.aero.case import ALLRUN, COEF_SCRIPT, write_case
    from pinneapple_design.aero.geometry import PARAMS
    out = []
    rows = cases.to_dict("records")[: limit if limit is not None else None]
    for row in rows:
        case = Path(out_dir) / "cases" / str(row["case_id"])
        shape = np.array([row[p] for p in PARAMS], float)
        write_case(str(case), shape, float(row["alpha_deg"]), reynolds=float(row.get("reynolds", reynolds)),
                   iterations=iterations)
        (case / "params.json").write_text(json.dumps(_params(row, ["case_id", *PARAMS, "alpha_deg"]), indent=1))
        (case / "coefficients.py").write_text(COEF_SCRIPT)
        (case / "Allrun").write_text(ALLRUN)
        os.chmod(case / "Allrun", 0o755)
        out.append(case)
    return out


# ------------------------------------------------------------------------------------------------- OpenFOAM (base)
def write_openfoam_from_base(base: str | Path, cases, out_dir: str | Path, edits: Sequence[Mapping[str, str]],
                             derived: Mapping[str, str] | None = None, solver: str = "simpleFoam", procs: int = 1,
                             pre: Sequence[str] = ("checkMesh",), limit: int | None = None) -> dict[str, Any]:
    """Copies of ``base`` (a case directory) with the ``edits`` applied per case, and ``plan.json`` for the
    OpenFOAM campaign runner. An edit is ``{"file": "0/U", "key": "boundaryField/inlet/value",
    "value": "uniform ({U} 0 0)"}``; values are templates of the parameters and ``derived`` expressions."""
    from .openfoam.campaign import (
        Campaign,
        CampaignRun,
        edit_file,
        evaluate,
        format_template,
        resolve,
    )
    base, out_dir = Path(base), Path(out_dir)
    if not (base / "system" / "controlDict").is_file():
        raise ValueError(f"{base} is not an OpenFOAM case (no system/controlDict)")
    names = [c for c in cases.columns if c not in _META_COLUMNS]
    runs, written = [], []
    for k, row in enumerate(cases.to_dict("records")):
        params = {n: row[n] for n in names if isinstance(row[n], (int, float, np.number))}
        params = {n: float(v) for n, v in params.items()}
        runs.append(CampaignRun(id=str(row["case_id"]), params=params))
        if limit is not None and k >= limit:
            continue
        values = resolve(params, dict(derived or {}))
        case = out_dir / "cases" / str(row["case_id"])
        shutil.copytree(base, case, dirs_exist_ok=True)
        for e in edits:
            value = format_template(e["value"], {n: v for n, v in values.items()})
            if e.get("evaluate"):
                value = str(evaluate(value, values))
            edit_file(case / e["file"], e["key"], value, root=case)
        (case / "params.json").write_text(json.dumps({"case_id": row["case_id"], **values}, indent=1, default=str))
        written.append(case)
    plan = Campaign(base="base_case", runs=runs, name=out_dir.name, solver=solver, procs=procs, pre=tuple(pre),
                    derived=dict(derived or {}), edits=[dict(e) for e in edits])
    shutil.copytree(base, out_dir / "base_case", dirs_exist_ok=True)
    plan.to_json(out_dir / "plan.json")
    return {"plan": out_dir / "plan.json", "cases": written}


# ------------------------------------------------------------------------------------------------- CalculiX
def cantilever_space(budget: int = 30) -> ParameterSpace:
    """A tip-loaded rectangular cantilever: length, width, height, material, tip load; slender (L >= 8 h) so that
    beam theory remains a sanity check of the results."""
    return ParameterSpace(name="cantilever_static", budget=budget, parameters=[
        Parameter(name="length", low=0.2, high=1.0, unit="m", group="geometry"),
        Parameter(name="width", low=0.02, high=0.08, unit="m", group="geometry"),
        Parameter(name="height", low=0.02, high=0.08, unit="m", group="geometry"),
        Parameter(name="material", kind="categorical", values=list(MATERIALS), group="material"),
        Parameter(name="tip_load", low=100.0, high=5000.0, unit="N", log=True, group="boundary",
                  description="total downward force on the free end face"),
    ], constraints=["length >= 8 * height"], description="Linear static cantilever, CalculiX C3D8I")


def write_calculix_cantilever(cases, out_dir: str | Path, elements_per_height: int = 2) -> list[Path]:
    """``cases/<case_id>/job.inp`` per row: structured C3D8I hexahedra, root fixed, tip load shared by the
    end-face nodes, plus ``params.json`` with the Euler-Bernoulli tip deflection F L^3 / (3 E I) for checking."""
    from .calculix.study import FEModel, Material, Static
    out = []
    for row in cases.to_dict("records"):
        L, b, h = float(row["length"]), float(row["width"]), float(row["height"])
        mat = MATERIALS[str(row["material"])]
        nz = max(1, int(elements_per_height))
        ny = max(1, round(nz * b / h))
        nx = max(4, round(nz * L / h))
        m = FEModel.from_box(L, b, h, nx, ny, nz, element="C3D8I")
        m.material = Material(str(row["material"]), E=mat["E"], nu=mat["nu"], density=mat["density"])
        m.title = str(row["case_id"])
        root = m.nodes_where(lambda X: X[:, 0] < 1e-9)
        tip = m.nodes_where(lambda X, L=L: X[:, 0] > L - 1e-9)
        deck = m.deck(Static(fix={tuple(root): (1, 2, 3)}, loads=[(tip, (0.0, 0.0, -float(row["tip_load"])))]))
        case = Path(out_dir) / "cases" / str(row["case_id"])
        case.mkdir(parents=True, exist_ok=True)
        (case / "job.inp").write_text(deck)
        inertia = b * h ** 3 / 12.0
        info = {**_params(row, ["case_id", "length", "width", "height", "material", "tip_load"]),
                "E": mat["E"], "nu": mat["nu"], "mesh": [nx, ny, nz],
                "beam_theory_tip_deflection_m": float(row["tip_load"]) * L ** 3 / (3 * mat["E"] * inertia)}
        (case / "params.json").write_text(json.dumps(info, indent=1))
        out.append(case)
    return out


def write_calculix_from_template(template: str, cases, out_dir: str | Path, name: str = "job.inp") -> list[Path]:
    """Fill ``{parameter}`` placeholders of a CalculiX deck per case (``{{`` / ``}}`` for literal braces)."""
    import string

    from .openfoam.campaign import format_template
    fields = {f for _, f, _, _ in string.Formatter().parse(template) if f}
    missing = fields - set(cases.columns)
    if missing:
        raise KeyError(f"the template uses {sorted(missing)}, which are not case columns")
    out = []
    for row in cases.to_dict("records"):
        case = Path(out_dir) / "cases" / str(row["case_id"])
        case.mkdir(parents=True, exist_ok=True)
        (case / name).write_text(format_template(template, row))
        out.append(case)
    return out


# ------------------------------------------------------------------------------------------------- everything
_META_COLUMNS = {"case_id", "priority", "information", "nearest_existing", "distance_to_existing",
                 "repeats_existing", "repeats_planned"}


def write_campaign(plan, out_dir: str | Path, *, outputs: Sequence[str] = (), units: Mapping[str, str] | None = None,
                   openfoam: str | None = None, calculix: str | None = None, openfoam_base: str | Path | None = None,
                   openfoam_edits: Sequence[Mapping[str, str]] = (), openfoam_derived: Mapping[str, str] | None = None,
                   calculix_template: str | None = None, case_limit: int | None = None,
                   app: str = "campaign") -> dict[str, Any]:
    """Write a :class:`pinneapple_data.doe.CampaignPlan` to ``out_dir``: tables, the Python runner, the contracts and
    the solver cases. ``openfoam``: ``"airfoil"`` or ``"base"`` (with ``openfoam_base`` + ``openfoam_edits``);
    ``calculix``: ``"cantilever"`` or ``"template"`` (with ``calculix_template``). ``case_limit`` caps the number of
    case directories (the tables and plan always hold every case)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    space = plan.space
    files: dict[str, Any] = {"tables": [str(p) for p in write_case_tables(plan.cases, out, space)]}
    files["python"] = str(write_python_runner(out, space.names, outputs))
    space.save(out / "space.json")
    plan.study_result(app=app).save(out / "study.json")
    plan.dataset_skeleton(outputs=outputs, units=dict(units or {}), app=app).save(out / "dataset.skeleton.json")
    if len(plan.dropped):
        write_table(plan.dropped, out / "dropped_repeats.csv")
    cases = plan.cases
    if openfoam == "airfoil":
        files["openfoam"] = [str(p) for p in write_openfoam_airfoil(cases, out, limit=case_limit)]
    elif openfoam == "base":
        if openfoam_base is None:
            raise ValueError("openfoam='base' needs openfoam_base")
        r = write_openfoam_from_base(openfoam_base, cases, out, openfoam_edits, openfoam_derived, limit=case_limit)
        files["openfoam"] = [str(p) for p in r["cases"]]
        files["openfoam_plan"] = str(r["plan"])
    if calculix == "cantilever":
        files["calculix"] = [str(p) for p in write_calculix_cantilever(cases.head(case_limit) if case_limit else cases,
                                                                       out)]
    elif calculix == "template":
        if not calculix_template:
            raise ValueError("calculix='template' needs calculix_template")
        sub = cases.head(case_limit) if case_limit else cases
        files["calculix"] = [str(p) for p in write_calculix_from_template(calculix_template, sub, out)]
    n_dirs = max(len(files.get("openfoam", [])), len(files.get("calculix", [])))
    (out / "README.txt").write_text(_readme(plan, n_dirs, outputs))
    return files


def _readme(plan, n_dirs: int, outputs: Sequence[str]) -> str:
    s = plan.summary()
    lines = [f"Campaign {plan.name} ({plan.id})", "",
             f"{s['n_cases']} cases, design {s['method']}, in priority order (cases.csv, column 'priority').",
             f"{s['n_dropped_as_repeats']} generated cases were dropped as repeats of existing runs "
             "(dropped_repeats.csv)." if s["n_dropped_as_repeats"] else "No generated case repeated an existing run.",
             f"Case directories written: {n_dirs} (cases/<case_id>/)." if n_dirs else "",
             "", "Files:",
             "  cases.csv / cases.parquet / cases.json   the cases (+ .contract.json: the parameter space)",
             "  run_cases.py                             loop over the cases with your own simulate()",
             "  space.json                               ParameterSpace",
             "  study.json                               StudyResult: coverage, checks, limits",
             "  dataset.skeleton.json                    DatasetSpec of the runs to come"
             + (f" (outputs: {', '.join(outputs)})" if outputs else ""),
             "  plan.json                                OpenFOAM campaign runner plan (base-case mode only)",
             "", "Run in priority order: any prefix of the list covers the space as evenly as possible."]
    return "\n".join(x for x in lines if x is not None) + "\n"
