"""Parameter sweeps / designs of experiments over the inputs of a :class:`.pipeline.PipelineSpec`, and a surrogate
of the results measured on held-out cases.

* :func:`doe`: Latin hypercube, Sobol or full-factorial designs within the input bounds (``scipy.stats.qmc``).
* :class:`Sweep`: one run folder per case (``cases/<id>``), each run by the automation
  :class:`~pinneapple_tools.automation.Runner` in its **own process** (``python -m
  pinneapple_simulation.pipelines.freecad_openfoam run``), ``parallel`` cases at a time. A rerun resumes: a case
  whose run passed or was rejected by its criteria is not run again, a failed or interrupted one resumes after its
  last passed step. ``table()`` collects inputs, state and every numeric output of the post step.
* :func:`fit_surrogate`: :func:`pinneapple_analysis.surrogate.run_study` on the table (every applicable method at
  equal budget, the best chosen on validation data, errors measured on held-out cases it never saw).
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .pipeline import PipelineSpec

__all__ = ["doe", "Sweep", "fit_surrogate", "case_id"]


def doe(bounds: dict[str, tuple[float, float]], n: int, method: str = "lhs", seed: int = 0,
        fixed: dict[str, float] | None = None) -> list[dict[str, float]]:
    """``n`` points in the box ``bounds`` (name -> (low, high)): ``lhs`` (Latin hypercube), ``sobol`` or ``grid``
    (``n`` levels per input, n**d points); ``fixed`` values are added to every point."""
    from scipy.stats import qmc
    names = list(bounds)
    lo = np.array([bounds[k][0] for k in names], float)
    hi = np.array([bounds[k][1] for k in names], float)
    d = len(names)
    if method == "lhs":
        U = qmc.LatinHypercube(d=d, seed=seed).random(n)
    elif method == "sobol":
        U = qmc.Sobol(d=d, scramble=True, seed=seed).random(n)
    elif method == "grid":
        g = np.meshgrid(*[np.linspace(0, 1, n)] * d, indexing="ij")
        U = np.stack([x.ravel() for x in g], 1)
    else:
        raise ValueError(f"method {method!r} is not lhs, sobol or grid")
    X = lo + U * (hi - lo)
    return [{**{k: float(round(v, 6)) for k, v in zip(names, row, strict=True)}, **(fixed or {})} for row in X]


def case_id(i: int, inputs: dict[str, float]) -> str:
    return f"c{i:03d}"


@dataclass
class Sweep:
    """``spec`` run at every point of ``design`` (list of input dicts) under ``folder``."""
    spec: PipelineSpec
    design: list[dict[str, float]]
    folder: str | Path
    ids: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.folder = Path(self.folder)
        if not self.ids:
            self.ids = [case_id(i, d) for i, d in enumerate(self.design)]
        if len(self.ids) != len(self.design) or len(set(self.ids)) != len(self.ids):
            raise ValueError("one unique id per design point")
        for d in self.design:
            self.spec.to_workflow().resolve_inputs(d)         # bounds checked before anything runs

    def save(self) -> Path:
        self.folder.mkdir(parents=True, exist_ok=True)
        self.spec.save(self.folder / "pipeline.json")
        p = self.folder / "design.json"
        p.write_text(json.dumps([{"id": i, "inputs": d} for i, d in zip(self.ids, self.design, strict=True)], indent=1))
        return p

    @classmethod
    def load(cls, folder: str | Path) -> Sweep:
        folder = Path(folder)
        spec = PipelineSpec.load(folder / "pipeline.json")
        rows = json.loads((folder / "design.json").read_text())
        return cls(spec, [r["inputs"] for r in rows], folder, [r["id"] for r in rows])

    def state(self, cid: str) -> str:
        p = self.folder / "cases" / cid / "run.json"
        return json.loads(p.read_text()).get("state", "pending") if p.is_file() else "pending"

    def pending(self) -> list[str]:
        return [c for c in self.ids if self.state(c) not in ("passed", "rejected")]

    def _run_one(self, cid: str, inputs: dict[str, float], log) -> tuple[str, str, float]:
        out = self.folder / "cases" / cid
        out.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, "-m", "pinneapple_simulation.pipelines.freecad_openfoam", "run",
               str(self.folder / "pipeline.json"), str(out), "--inputs", json.dumps(inputs)]
        t0 = time.time()
        with open(out / "log.pipeline", "a") as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
        st = self.state(cid)
        log(f"{cid} {json.dumps(inputs)}: {st} in {time.time() - t0:.0f} s")
        return cid, st, time.time() - t0

    def run(self, parallel: int = 1, log=print) -> list[dict[str, Any]]:
        """Run every pending case (``parallel`` processes at a time) and return :meth:`table`."""
        self.save()
        todo = [(c, d) for c, d in zip(self.ids, self.design, strict=True) if c in set(self.pending())]
        log(f"{len(todo)} of {len(self.ids)} cases to run ({len(self.ids) - len(todo)} already done)")
        with ThreadPoolExecutor(max_workers=max(1, int(parallel))) as ex:
            list(ex.map(lambda a: self._run_one(*a, log), todo))
        rows = self.table()
        (self.folder / "sweep.json").write_text(json.dumps(rows, indent=1))
        _csv(rows, self.folder / "sweep.csv")
        return rows

    def table(self) -> list[dict[str, Any]]:
        """One row per case: id, state, inputs and the numeric outputs of the post step."""
        rows = []
        for cid, d in zip(self.ids, self.design, strict=True):
            row: dict[str, Any] = {"id": cid, "state": self.state(cid), **d}
            p = self.folder / "cases" / cid / "post" / "measured.json"
            if p.is_file():
                m = json.loads(p.read_text())["measured"]
                row.update({k: v for k, v in m.items() if isinstance(v, (int, float)) and k not in d})
            rows.append(row)
        return rows


def _csv(rows: list[dict[str, Any]], path: Path) -> None:
    cols = list(dict.fromkeys(k for r in rows for k in r))
    lines = [",".join(cols)] + [",".join("" if r.get(c) is None else str(r.get(c)) for c in cols) for r in rows]
    path.write_text("\n".join(lines) + "\n")


def fit_surrogate(rows: list[dict[str, Any]], inputs: list[str], outputs: list[str], *, test_frac: float = 0.25,
                  val_frac: float = 0.15, seed: int = 0, budget: float = 5.0, methods: Any = "auto",
                  states: tuple[str, ...] = ("passed", "rejected"), source: str = "FreeCAD -> OpenFOAM sweep"):
    """A surrogate of ``outputs`` from ``inputs`` on the finished cases: :func:`pinneapple_analysis.surrogate.
    run_study` with a random held-out test set (no distribution-shift split: sweeps are small). Returns the study;
    ``study.ranking("test")`` holds the held-out errors, ``study.model.predict`` the chosen model."""
    import pandas as pd

    from pinneapple_analysis import surrogate
    use = [r for r in rows if r.get("state") in states and all(r.get(k) is not None for k in inputs + outputs)]
    df = pd.DataFrame(use)
    data = surrogate.from_table(df, inputs, outputs, meta={"source": source})
    return surrogate.run_study(data, methods=methods, budget=budget, seeds=[seed],
                               split_options={"test_frac": test_frac, "val_frac": val_frac, "shift": None})
