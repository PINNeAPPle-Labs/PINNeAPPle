"""``pp.new`` / ``pinneapple new``: a project a person can grow, outside this repository.

The repository's own scaffold (``python -m pinneapple_lab scaffold``) creates an *example* of the
library, wired into the Lab here. A user needs the opposite: a folder of their own, with no
dependency on this checkout, that runs on a clean install and survives being copied to a colleague.

Three templates, by how much Python the user wants to write:

``plain``
    The user edits ``case.yaml`` and nothing else. ``run.py`` resolves the problem (from a preset
    or from a sentence), solves it with the cheapest method that works, saves the field, a figure,
    and a one-page report. No Python is required to go from "my problem" to "a number and a plot".
``research``
    ``study.py`` solves, compares against the classical answer where one exists, records metrics
    and checks, and writes an experiment manifest the Lab can grade, next to a notebook.
``cfd``
    A campaign runner: one preset, a parameter swept over a list, the reference solver per case,
    a CSV of quantities of interest and a convergence figure, with the geometry hook written down.

Every generated file runs as written on a clean ``pip install pinneapple``; the folders the user
will fill are created; and no path depends on the directory the script was launched from.

>>> import pinneapple as pp
>>> path = pp.new("my_study", template="plain")      # writes ./my_study and prints the next step
"""
from __future__ import annotations

import re
from pathlib import Path
from string import Template
from typing import Any, Dict, Optional

__all__ = ["TEMPLATES", "new_project", "plan"]

TEMPLATES: Dict[str, str] = {
    "plain": "edit case.yaml only, never Python",
    "research": "solve, compare, record metrics and checks, keep a notebook",
    "cfd": "sweep a parameter through the reference solver into a CSV and a figure",
}


# ---------------------------------------------------------------------------
# plain
# ---------------------------------------------------------------------------
_PLAIN_RUN = Template('''"""(project) — run this case.

Edit case.yaml, not this file. Then:

    python run.py

It reads case.yaml, finds the problem (a preset, or a sentence it resolves for you), solves it with
the cheapest method that works on this machine, and writes to results/:

    solution.npz   the field on a grid, plus the coordinates
    field.png      the field as a figure
    report.md      what was solved, by which method, how long it took, and the checks it passed
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

import pinneapple as pp

HERE = Path(__file__).resolve().parent          # every path is relative to THIS folder
RESULTS = HERE / "results"
CASE_FILE = HERE / "case.yaml"


def load_case() -> dict:
    if not CASE_FILE.exists():
        raise SystemExit(f"case.yaml not found next to this script ({CASE_FILE})")
    case = yaml.safe_load(CASE_FILE.read_text(encoding="utf-8")) or {}
    if not case.get("preset") and not case.get("problem"):
        raise SystemExit("case.yaml needs either 'preset: <name>' or 'problem: <a sentence>'")
    return case


def resolve_preset(case: dict) -> str:
    """A preset name from case.yaml, or the best one for the sentence the user wrote."""
    preset = case.get("preset")
    if preset:
        return str(preset)
    rec = pp.recommend(case["problem"])
    print(rec.explain())
    if not rec.preset:
        raise SystemExit("\\nCould not find a preset for that description. "
                         "Set 'preset' in case.yaml to one of pp.list_presets().")
    return rec.preset


def sample_grid(problem, n: int = 60):
    """A regular grid over the problem's domain: (n, n) on the first two coordinates, and the
    remaining coordinates fixed at the middle of their range."""
    coords = list(problem.coords)
    bounds = problem.domain_bounds or {}
    axes = []
    for i, c in enumerate(coords):
        lo, hi = bounds.get(c, (0.0, 1.0))
        if i < 2:
            axes.append(np.linspace(lo, hi, n))
        else:
            axes.append(np.array([(lo + hi) / 2.0]))
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    return X, axes


def plot(axes, values, coords, fields, preset, out):
    """The first field as a figure: a line in 1-D, a map in 2-D. Returns the path, or None."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not fields or values.size == 0:
        return None
    field, space = fields[0], [c for c in coords if c != "t"]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    if len(axes) >= 2 and axes[1].size > 1:
        data = values[:, 0].reshape(axes[0].size, axes[1].size).T
        im = ax.pcolormesh(axes[1], axes[0], data, shading="auto", cmap="viridis")
        fig.colorbar(im, ax=ax, label=field)
        if len(space) >= 2:
            ax.set_xlabel(space[1])
            ax.set_ylabel(space[0])
    else:
        ax.plot(axes[0], values[:, 0])
        ax.set_xlabel(coords[0] if coords else "x")
        ax.set_ylabel(field)
    ax.set_title(f"{preset} - {field}")
    path = out / "field.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> int:
    case = load_case()
    preset = resolve_preset(case)
    card = pp.card(preset)
    method = case.get("method") or "auto"
    if method == "auto":
        method = card.method
    params = case.get("params") or {}
    n = int(case.get("grid_points", 60))

    RESULTS.mkdir(parents=True, exist_ok=True)
    print(f"\\nsolving {preset} with '{method}'" + (f" (parameters: {params})" if params else "") + " ...")
    t0 = time.time()
    try:
        sol = pp.solve(preset, method=method, preset_kwargs=params or None)
    except Exception as first_error:
        # The classical method can refuse a problem outside its coverage (a field it does not
        # compute, a solver this machine does not have, geometry tags it cannot name). Train
        # instead; if that fails too, say what to install or supply instead of a traceback.
        print(f"  '{method}' cannot solve this problem here ({type(first_error).__name__}).")
        print("  training a PINN instead; this takes minutes ...")
        try:
            method = "pinn"
            sol = pp.solve(preset, method="pinn", preset_kwargs=params or None, epochs=2000)
        except Exception as second_error:
            raise SystemExit(
                f"\\nThis preset cannot run on this machine:\\n"
                f"  {type(first_error).__name__}: {first_error}\\n"
                f"  {type(second_error).__name__}: {second_error}\\n\\n"
                f"what it needs:\\n"
                f"  python -c \\"import pinneapple as pp; print(pp.card('{preset}'))\\"\\n"
                f"a preset that runs here now:\\n"
                f"  python -c \\"import pinneapple as pp; pp.recommend('{preset}')\\"\\n$fallback_hint"
            ) from second_error
    seconds = time.time() - t0
    print(f"done in {seconds:.1f} s")

    X, axes = sample_grid(sol.problem, n)
    values = np.asarray(sol.predict(X), dtype=float)
    np.savez_compressed(RESULTS / "solution.npz", X=X, values=values,
                        coords=np.array(list(sol.problem.coords)), fields=np.array(list(sol.problem.fields)))
    coords = list(sol.problem.coords)
    figure = plot(axes, values, coords, list(sol.problem.fields), preset, RESULTS)

    report = [
        f"# {preset}",
        "",
        f"- problem: {card.description}",
        f"- method: `{method}`",
        f"- wall time: {seconds:.2f} s",
        f"- grid points: {len(X)}",
        f"- fields: {', '.join(sol.problem.fields)}",
        f"- coordinates: {', '.join(sol.problem.coords)}",
        "",
        "## What it means",
        "",
        f"This is {card.description[0].lower() + card.description[1:] if card.description else 'the field'} "
        f"computed on a regular grid. `solution.npz` holds the points (X) and the values in the field "
        f"order above: `values[:, i]` is field `{ ', '.join(sol.problem.fields) }` at row `i` of `X`.",
        "",
        "## Next",
        "",
        "- change `params` in case.yaml and run again: same code, new physics",
        "- `pp.card('%s')` for what this preset takes and gives back" % preset,
        "- `pp.recommend('...')` if you want a different preset for a different problem",
    ]
    if figure:
        report.append(f"- the figure: `{figure.name}`")
    (RESULTS / "report.md").write_text("\\n".join(report) + "\\n", encoding="utf-8")
    (RESULTS / "metadata.json").write_text(json.dumps(sol.metadata(), indent=2, default=str), encoding="utf-8")

    print(f"wrote {RESULTS}/solution.npz, report.md, metadata.json" + (", field.png" if figure else ""))
    print(f"  first few values: {np.array2string(values[:3, 0], precision=4)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
''')

_PLAIN_CASE = Template('''# What to solve. Edit this file; you never need to touch run.py.
#
# Either a preset (exact, reproducible):
preset: $preset
#
# ...or describe the problem in a sentence and let pinneapple pick the preset:
# preset:
# problem: "temperature in a heat sink cooled by a fan"
#
# Or start from an industrial case instead of a preset:
# industrial_case: plate_fin_heatsink_forced

# How to solve it. "auto" uses the cheapest method that works: a closed form if the library knows
# one, else the classical solver, else a trained network.
method: auto

# Parameters of the preset (see `python -c "import pinneapple as pp; pp.card('$preset')"`).
params: {}

# Grid resolution for the saved field.
grid_points: 60
''')

# ---------------------------------------------------------------------------
# research
# ---------------------------------------------------------------------------
_RESEARCH_STUDY = Template('''"""(project) — the study.

    python study.py                 # solve, compare, record
    python study.py --epochs 500    # same, training a PINN instead of the classical answer

Every path is relative to this folder, so it runs from anywhere. What it writes:

    results/<method>.npz            the field, on the grid it was solved on
    results/metrics.json            every number, with the unit and how it was measured
    results/field.png               the field
    experiment.json                 the manifest the PINNNeAPPle Lab grades A-F (when it is installed)
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml

import pinneapple as pp

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def load_case() -> dict:
    case = yaml.safe_load((HERE / "case.yaml").read_text(encoding="utf-8")) or {}
    if not case.get("preset"):
        rec = pp.recommend(case.get("problem", ""))
        if not rec.preset:
            raise SystemExit("set 'preset' in case.yaml; pp.list_presets() lists them")
        print(rec.explain())
        case["preset"] = rec.preset
    return case


def grid_for(problem, n: int):
    axes = []
    bounds = problem.domain_bounds or {}
    for i, c in enumerate(problem.coords):
        lo, hi = bounds.get(c, (0.0, 1.0))
        axes.append(np.linspace(lo, hi, n) if i < 2 else np.array([(lo + hi) / 2.0]))
    mesh = np.meshgrid(*axes, indexing="ij")
    return np.stack([m.ravel() for m in mesh], axis=1), axes


def main() -> int:
    ap = argparse.ArgumentParser(description="solve and record one case")
    ap.add_argument("--epochs", type=int, default=0, help="train a PINN for this many epochs instead "
                                                          "of using the classical method")
    ap.add_argument("--grid", type=int, default=64)
    args = ap.parse_args()

    case = load_case()
    preset = case["preset"]
    card = pp.card(preset)
    method = "pinn" if args.epochs else card.method
    RESULTS.mkdir(parents=True, exist_ok=True)

    print(f"{preset}: {card.description}")
    t0 = time.time()
    options = {"epochs": args.epochs} if args.epochs else {}
    try:
        sol = pp.solve(preset, method=method, preset_kwargs=case.get("params") or None, **options)
    except Exception as first_error:
        print(f"'{method}' cannot solve this problem here ({type(first_error).__name__}); "
              f"training a PINN instead (minutes).")
        try:
            method = "pinn"
            sol = pp.solve(preset, method="pinn", preset_kwargs=case.get("params") or None,
                           epochs=args.epochs or 2000)
        except Exception as second_error:
            raise SystemExit(
                f"this preset cannot run on this machine:\\n"
                f"  {type(first_error).__name__}: {first_error}\\n"
                f"  {type(second_error).__name__}: {second_error}\\n"
                f"what it needs: python -c \\"import pinneapple as pp; print(pp.card('{preset}'))\\"\\n"
                "a preset that runs here now:\\n"
                f"  python -c \\"import pinneapple as pp; pp.recommend('{preset}')\\"\\n$fallback_hint"
            ) from second_error
    wall = time.time() - t0

    X, axes = grid_for(sol.problem, args.grid)
    values = np.asarray(sol.predict(X), dtype=float)
    np.savez_compressed(RESULTS / f"{method}.npz", X=X, values=values,
                        coords=np.array(list(sol.problem.coords)),
                        fields=np.array(list(sol.problem.fields)))

    metrics = [{"name": "wall_time_s", "value": round(wall, 3), "unit": "s",
                "how": f"wall clock of pp.solve(method={method!r})"},
               {"name": "grid_points", "value": int(len(X)), "unit": "", "how": "regular grid nodes predicted on"}]
    for i, f in enumerate(sol.problem.fields):
        metrics.append({"name": f"{f}_mean", "value": float(np.mean(values[:, i])), "unit": "",
                        "how": f"mean of {f} over the saved grid"})

    # Score the trained model against the classical answer when that answer exists.
    reference = None
    if method != "reference" and card.solver:
        try:
            ref = pp.solve(preset, method="reference", preset_kwargs=case.get("params") or None)
            diff = np.abs(values - np.asarray(ref.predict(X), dtype=float))
            reference = {"method": "reference", "max_abs_error": float(np.max(diff)),
                         "relative": float(np.max(diff) / (np.max(np.abs(ref.predict(X))) + 1e-12))}
            metrics.append({"name": "max_abs_error_vs_reference", "value": reference["max_abs_error"],
                            "unit": "", "how": "max |prediction - reference| on the saved grid"})
        except Exception as e:  # no reference solver output for this preset: say so, do not fake a metric
            reference = {"method": "reference", "error": str(e)}

    payload = {"preset": preset, "method": method, "wall_time_s": round(wall, 3),
               "params": case.get("params") or {}, "metrics": metrics, "against_reference": reference,
               "metadata": sol.metadata()}
    (RESULTS / "metrics.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        data = values[:, 0].reshape(axes[0].size, axes[1].size).T if axes[1].size > 1 else None
        if data is not None:
            im = ax.pcolormesh(axes[1], axes[0], data, shading="auto", cmap="viridis")
            fig.colorbar(im, ax=ax, label=sol.problem.fields[0])
        else:
            ax.plot(axes[0], values[:, 0])
        ax.set_title(f"{preset} — {sol.problem.fields[0]} ({method})")
        fig.savefig(RESULTS / "field.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"(figure skipped: {e})")

    manifest = {"schema": "pinneapple.experiment/1", "id": "$name", "title": "$name",
                "description": card.description, "domain": card.domain,
                "methods": [method], "modules": ["pinneapple_physics.solving"],
                "params": case.get("params") or {}, "scripts": ["study.py"],
                "metrics": metrics, "checks": [], "files": {"figures": ["results/field.png"],
                "data": ["results/metrics.json"]}, "links": {"script": "study.py"}, "references": [],
                "status": "not curated",
                "provenance": {"source": "pp.new research template", "created": time.strftime("%Y-%m-%dT%H:%M:%S")}}
    (HERE / "experiment.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"done in {wall:.1f} s -> {RESULTS}/")
    for m in metrics:
        unit = f" {m['unit']}" if m["unit"] else ""
        print(f"  {m['name']:<28} {m['value']}{unit}")
    if reference and "max_abs_error" in reference:
        print(f"  vs the classical solver: max abs error {reference['max_abs_error']:.3e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')

_RESEARCH_NOTEBOOK = Template('''{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": ["# $name\\n", "\\n", "$description\\n", "\\n", "`study.py` is the script version of this notebook: same calls, same numbers, runnable in CI.\\n"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["import pinneapple as pp\\n", "case = pp.card('$preset')\\n", "print(case.name, '-', case.description)\\n", "print(case.method)\\n"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["sol = pp.solve('$preset', method='$method')\\n", "print(sol.metadata())\\n"]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": ["import numpy as np\\n", "X = np.random.default_rng(0).random((200, len(sol.problem.coords)))\\n", "print(sol.predict(X)[:5])\\n"]
  }
 ],
 "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
 "nbformat": 4,
 "nbformat_minor": 5
}
''')

# ---------------------------------------------------------------------------
# cfd
# ---------------------------------------------------------------------------
_CFD_SWEEP = Template('''"""(project) — a campaign: one problem, a parameter swept, one row per case.

    python sweep.py                    # every value in case.yaml: values

What it writes:

    results/cases.csv     one row per parameter value: the inputs, the quantities of interest, the time
    results/qoi.png       the quantities of interest against the parameter
    results/report.md     what was run, and the checks each case passed

The parameter is a preset argument (``Re``, ``q_cpu``, ``Umax``...). See what this preset takes:

    python -c "import pinneapple as pp; pp.card('$preset')"
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import yaml

import pinneapple as pp

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def main() -> int:
    case = yaml.safe_load((HERE / "case.yaml").read_text(encoding="utf-8")) or {}
    preset = case.get("preset", "$preset")
    parameter = case.get("parameter", "$parameter")
    values = case.get("values") or [1.0, 2.0, 4.0]
    fixed = case.get("params") or {}
    grid = int(case.get("grid_points", 48))

    card = pp.card(preset)
    print(f"{preset}: {card.description}")
    print(f"sweeping {parameter} over {values}\\n")

    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for v in values:
        params = dict(fixed)
        params[parameter] = v
        t0 = time.time()
        try:
            sol = pp.solve(preset, method=card.method, preset_kwargs=params)
        except Exception as e:
            rows.append({"value": v, "error": str(e), "seconds": round(time.time() - t0, 3)})
            print(f"  {parameter}={v}: failed ({e})")
            continue
        bounds = sol.problem.domain_bounds or {}
        axes = [np.linspace(*bounds.get(c, (0.0, 1.0)), grid) if i < 2 else
                np.array([sum(bounds.get(c, (0.0, 1.0))) / 2.0]) for i, c in enumerate(sol.problem.coords)]
        mesh = np.meshgrid(*axes, indexing="ij")
        X = np.stack([m.ravel() for m in mesh], axis=1)
        values_out = np.asarray(sol.predict(X), dtype=float)
        row = {"value": v, "seconds": round(time.time() - t0, 3)}
        for i, f in enumerate(sol.problem.fields):
            row[f"{f}_mean"] = float(np.mean(values_out[:, i]))
            row[f"{f}_max"] = float(np.max(values_out[:, i]))
            row[f"{f}_min"] = float(np.min(values_out[:, i]))
        rows.append(row)
        print(f"  {parameter}={v}: " + ", ".join(f"{k}={row[k]:.4g}" for k in row
                                                 if k.endswith("_mean") or k.endswith("_max")))

    ok = [r for r in rows if "error" not in r]
    if not ok:
        print("\\nEvery case failed. The message per case is in results/cases.csv.")
        return 1
    keys = sorted({k for r in rows for k in r})
    with open(RESULTS / "cases.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["value"] + [k for k in keys if k != "value"])
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5))
        for k in ok[0]:
            if k.endswith("_mean"):
                ax.plot([r["value"] for r in ok], [r[k] for r in ok], marker="o", label=k)
        ax.set_xlabel(parameter)
        ax.set_ylabel("mean over the grid")
        ax.set_title(f"{preset}: {parameter} sweep")
        ax.legend(fontsize=8)
        fig.savefig(RESULTS / "qoi.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    except Exception as e:
        print(f"(figure skipped: {e})")

    report = [f"# {preset}: {parameter} sweep", "", f"- method: `{card.method}`",
              f"- cases: {len(ok)} ok, {len(rows) - len(ok)} failed",
              f"- values: {', '.join(str(r['value']) for r in rows)}", "",
              "## Cases", "", "| " + parameter + " | " + " | ".join(
                  sorted({k for r in ok for k in r if k.endswith("_mean")})) + " | seconds |",
              "|" + "---|" * (len([k for k in ok[0] if k.endswith('_mean')]) + 2)]
    means = sorted({k for r in ok for k in r if k.endswith("_mean")})
    for r in rows:
        cells = [f"{r['value']}"] + [f"{r.get(m, float('nan')):.4g}" for m in means] + [f"{r['seconds']}"]
        report.append("| " + " | ".join(cells) + " |")
    report += ["", "## Before you use these numbers", "",
               "A reference solve is not a validation of the model that will replace it: the quantity of "
               "interest here comes from the classical solver named in the preset. When you compare a "
               "trained surrogate against these rows, use the same grid and the same conditions, and "
               "report the error on rows the surrogate never saw."]
    (RESULTS / "report.md").write_text("\\n".join(report) + "\\n", encoding="utf-8")
    print(f"\\nwrote {RESULTS}/cases.csv, qoi.png, report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
''')

_CFD_CASE = Template('''# A campaign: one preset, one parameter swept over a list of values.
preset: $preset

# The preset argument to sweep (Re, Umax, q_cpu, ...). What the preset accepts:
#   python -c "import pinneapple as pp; pp.card('$preset')"
parameter: $parameter
values: $values

# Parameters held fixed across every case.
params: {}

grid_points: 48

# Geometry. The reference solve above runs on the preset's own domain. To run on your own shape,
# install the geometry and meshing extras and use the fit-for-purpose mesher:
#
#   pip install "pinneapple[fitmesh,geometry]"
#   from pinneapple_design.fitmesh import <...>      # see docs/guides/geometry_library.md
#   from pinneapple_design.geometry.csg import CSGUnion, CSGCircle, CSGDifference
#
# Then pass the resulting mesh/case to the OpenFOAM campaign runner in
# pinneapple_simulation.numerical_solvers (docs/guides/freecad_openfoam_pipelines.md).
geometry: preset-domain
''')


_PLAIN_README = Template("""# $name

$description

Template: **plain**. You edit `case.yaml` and nothing else.

## Run it

```bash
pip install "pinneapple$extras" pyyaml
python run.py
```

## What you change

Open `case.yaml`. It has three parts:

```yaml
preset: $preset        # what to solve (or delete it and write `problem: "<a sentence>"`)
method: auto           # auto | reference | analytic | pinn
params: {}             # the physics: viscosity, heat load, Reynolds number...
```

Change `params` and run again. `method: auto` picks the cheapest method that works on this machine:
a closed form when the library knows one, else the classical solver. `method: pinn` trains a
network instead (slower, and it is the case where you need `pp.trust_card` before believing it).

## What you get

| file | what |
|---|---|
| `results/solution.npz` | the grid points (`X`) and the field values (`values[:, i]` is field `i`) |
| `results/field.png` | the field as a figure |
| `results/report.md` | what was solved, by which method, how long, and what it means |
| `results/metadata.json` | the same facts as data |

## When you need more

- **a different problem**: delete `preset` and write `problem: "<what you are trying to solve>"`;
  pinneapple picks the closest preset and prints why.
- **your own PDE**: `docs/getting_started/first_pinn.md` describes a problem from its equation.
- **your own geometry**: `pip install "pinneapple[fitmesh,geometry]"`, then
  `docs/guides/geometry_library.md`.
- **a paper or a report**: start again with `pp.new("$name", template="research")`, which records
  every metric with its unit and how it was measured.
""")

_RESEARCH_README = Template("""# $name

$description

Template: **research**. Solve, compare, record.

## Run it

```bash
pip install "pinneapple$extras"
python study.py                       # the classical answer, seconds
python study.py --epochs 500          # train a PINN and score it against that answer
```

`notebook.ipynb` is the same study as a notebook; `study.py` is the version that runs in CI.

## What it writes

| file | what |
|---|---|
| `results/metrics.json` | every number with its unit and how it was measured, plus the error against the classical solver |
| `results/<method>.npz` | the field on the grid it was predicted on |
| `results/field.png` | the field |
| `experiment.json` | the manifest the Lab grades A-F; `python -m pinneapple_lab seal --explain experiment.json` says what it lacks |

## The rule this template follows

A number without its check is not a result. `study.py` always reports the error against the
reference solver when one exists, and says so explicitly when it does not, instead of leaving the
column out. Add your own checks with `ex.check(...)` when you move to
`pinneapple_lab.experiment`.

## Next

- `pp.validate(model, problem, level="quick")` before you quote an accuracy.
- `pp.benchmark` / `pp.compare` to put two methods on the same budget.
- `docs/guides/choosing_a_method.md` for why one family over another.
""")

_CFD_README = Template("""# $name

$description

Template: **cfd**. A campaign: one problem, one parameter swept, one row per case.

## Run it

```bash
pip install "pinneapple$extras" pyyaml
python sweep.py
```

`case.yaml` holds the preset, the parameter to sweep and the values. Start from `$preset`.

## What it writes

| file | what |
|---|---|
| `results/cases.csv` | one row per parameter value: inputs, mean/min/max of every field, seconds |
| `results/qoi.png` | the quantities of interest against the parameter |
| `results/report.md` | what ran, what failed and why |

## Before you use these numbers

The quantity of interest comes from the classical solver the preset names. That makes these rows a
reference, not a validation of whatever will replace them. When you compare a surrogate, use the
same grid and the same conditions, and report the error on rows it never saw.

## Your own geometry

`case.yaml` documents where the geometry hook goes. The short version:

```bash
pip install "pinneapple[fitmesh,geometry]"
```

then `pinneapple_design.fitmesh` (CAD to a fit-for-purpose mesh) and the OpenFOAM campaign runner in
`pinneapple_simulation.numerical_solvers`; guides in `docs/guides/geometry_library.md` and
`docs/guides/freecad_openfoam_pipelines.md`.
""")


def _profile_for(template: str) -> str:
    return {"plain": "start", "research": "research", "cfd": "cfd"}[template]


def _extras_for(template: str) -> str:
    return {"plain": "", "research": "", "cfd": ""}[template]


def _sweep_parameter(preset: str) -> tuple[str, str]:
    """A parameter the preset actually takes, and three values around its default.

    A campaign that opens on a parameter the preset does not accept fails on its first case, so the
    default names a real one instead of assuming every problem has a Reynolds number.
    """
    try:
        from .catalog import _spec

        spec = _spec(preset)
        params = dict(getattr(getattr(spec, "pde", None), "params", {}) or {})
        numeric = [(k, v) for k, v in params.items()
                   if isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0]
        if numeric:
            name, default = numeric[0]
            values = ", ".join(f"{default * f:g}" for f in (0.5, 1.0, 2.0))
            return str(name), f"[{values}]"
    except Exception:
        pass
    return "Re", "[50.0, 100.0, 200.0]"


def plan(name: str, template: str = "plain", preset: str = "burgers_1d") -> Dict[str, Any]:
    """What ``new_project`` would write, without writing it. Useful for a dry run and for tests."""
    if template not in TEMPLATES:
        raise ValueError(f"unknown template {template!r}; choose one of: {', '.join(TEMPLATES)}")
    files: Dict[str, str] = {}
    card = None
    try:
        from .catalog import card as _card

        card = _card(preset)
    except Exception:
        pass
    description = card.description if card else ""
    method = card.method if card else "reference"
    parameter, values = _sweep_parameter(preset)
    fallback_hint = ""
    if card is not None and not card.available and card.fallback:
        # a fragment of generated SOURCE, so its newlines are escaped, not literal
        fallback_hint = "the same physics, described so it runs here:\\n  python fallback.py"
    substitutions = dict(name=name, preset=preset, description=description, template=template,
                         parameter=parameter, values=values, method=method, fallback_hint=fallback_hint,
                         domain=card.domain if card else "general",
                         extras=_extras_for(template))
    if template == "plain":
        files["run.py"] = _PLAIN_RUN.substitute(**substitutions)
        files["case.yaml"] = _PLAIN_CASE.substitute(**substitutions)
        files["README.md"] = _PLAIN_README.substitute(**substitutions)
    elif template == "research":
        files["study.py"] = _RESEARCH_STUDY.substitute(name=name, preset=preset, description=description,
                                                       method=method, fallback_hint=fallback_hint)
        files["case.yaml"] = _PLAIN_CASE.substitute(**substitutions)
        files["notebook.ipynb"] = _RESEARCH_NOTEBOOK.substitute(name=name, preset=preset,
                                                                description=description, method=method)
        files["README.md"] = _RESEARCH_README.substitute(**substitutions)
    else:
        files["sweep.py"] = _CFD_SWEEP.substitute(name=name, preset=preset, parameter="Re")
        files["case.yaml"] = _CFD_CASE.substitute(**substitutions)
        files["README.md"] = _CFD_README.substitute(**substitutions)
    if fallback_hint:
        files["fallback.py"] = (
            f'"""(project) - the same physics as {preset}, described so it runs on any machine.\n\n'
            f'The preset itself needs: {card.requires or "its own geometry"}\n'
            f'Running this file trains a network on the same equation on a box instead.\n"""\n'
            + card.fallback + "\n")
    files["requirements.txt"] = f"pinneapple{_extras_for(template)}\npyyaml\n"
    files[".gitignore"] = "results/\n*.npz\n*.png\n__pycache__/\n.ipynb_checkpoints/\n"
    files["pinneapple.json"] = (
        "{\n"
        f'  "project": "{name}",\n'
        f'  "template": "{template}",\n'
        f'  "preset": "{preset}",\n'
        f'  "doctor_profile": "{_profile_for(template)}",\n'
        '  "generated_by": "pp.new"\n'
        "}\n"
    )
    return {"folder": name, "template": template, "preset": preset, "files": files,
            "doctor_profile": _profile_for(template)}


def new_project(name: str, *, dir: Optional[str] = None, template: str = "plain",
                preset: str = "burgers_1d", verbose: bool = True) -> str:
    """Write a standalone project and return its folder.

    ``name``: the folder name, lowercase letters, digits and underscores. ``dir``: the parent
    folder, the current directory by default. ``template``: ``plain``, ``research`` or ``cfd``
    (see :data:`TEMPLATES`). ``preset``: the problem to start from.
    """
    if not re.fullmatch(r"[a-z][a-z0-9_]*", str(name)):
        raise ValueError("name: lowercase letters, digits and underscores, starting with a letter")
    root = Path(dir).expanduser().resolve() if dir else Path.cwd()
    target = root / name
    if target.exists():
        raise FileExistsError(f"{target} already exists; pick another name or remove it")

    spec = plan(name, template=template, preset=preset)
    for rel, content in spec["files"].items():
        path = target / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (target / "results").mkdir(parents=True, exist_ok=True)
    (target / "results" / ".gitkeep").write_text("", encoding="utf-8")

    entry = {"plain": "run.py", "research": "study.py", "cfd": "sweep.py"}[template]
    if verbose:
        print(f"created {target}")
        print(f"  template : {template} ({TEMPLATES[template]})")
        print(f"  preset   : {preset}")
        print("  files    : " + ", ".join(sorted(spec["files"])) + ", results/")
        print("")
        print("  next:")
        print(f"    cd {target}")
        print(f"    python {entry}")
        print(f"    pinneapple doctor {_profile_for(template)}    # check this machine for this work")
    return str(target)
