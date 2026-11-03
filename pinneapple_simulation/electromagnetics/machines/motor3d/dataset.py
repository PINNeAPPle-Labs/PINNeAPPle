"""The training set of the motor surrogate: a design of experiments over :data:`~.design.VARIABLES`, labelled by the
reference model and stored with its provenance.

The set is what makes the surrogate possible, and it is also the largest cost of the study: one reference design is
one electromagnetic operating point (a nonlinear magnetostatic solve over several rotor angles) plus an iron-loss
period, about a second on one core. :func:`build_dataset` therefore runs the cases in parallel processes and stores
the result, so a surrogate is trained from a file rather than from a fresh campaign.

Invalid designs (polygons that cross, a magnet that leaves its pole) are part of the record: they are marked in
``valid``, and the surrogate is trained only on the valid ones. They are also what makes the *rejection rate* of the
design space visible: at the corners of the box a large fraction of the designs cannot be built, and an optimiser
that does not know it wastes its budget.

:class:`MotorDataset` keeps the design matrix ``X`` (columns in the order of :data:`~.design.NAMES`), the outputs
``Y`` (scalars of :data:`~.model.SCALARS` and profiles of :data:`~.model.PROFILES`, flattened to ``(n, n_segments)``
columns named ``T_winding_profile_C[k]``), and a ``meta`` dictionary with the model configuration, the DOE method, the
seed, the cost and the library version. ``MotorDataset.load`` refuses a file whose design names do not match
:data:`~.design.NAMES`: a surrogate trained on a different ordering of the inputs would otherwise predict silently
wrong numbers.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .design import BOUNDS, NAMES, VARIABLES, Motor3DDesign, from_matrix
from .model import OUTPUTS, PROFILES, SCALARS, Motor3DModel

__all__ = ["parameter_space", "MotorDataset", "build_dataset", "evaluate_chunk"]


def parameter_space():
    """The study as a :class:`~pinneapple_data.contracts.ParameterSpace`, so the same box can be sampled by the
    library's design-of-experiments tools (LHS, maximin, Sobol, infill) and its coverage reported."""
    from pinneapple_data.contracts import Parameter, ParameterSpace

    return ParameterSpace(name="motor3d", description="3-D multiphysics interior-PM motor: rotor cross-section, stack, "
                                                      "skew, operating point and coolant flow",
                          parameters=[Parameter(name=v.name, low=v.low, high=v.high, unit=v.unit,
                                                group=v.doe_group, description=v.description) for v in VARIABLES])


@dataclass
class MotorDataset:
    """Designs, their outputs and where they came from."""
    X: np.ndarray                                 # (n, d) in the order of NAMES
    Y: dict[str, np.ndarray]                      # scalars (n,) and profiles (n, n_segments)
    valid: np.ndarray                             # (n,) the reference model could build and solve the design
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.X)

    @property
    def names(self) -> tuple[str, ...]:
        return NAMES

    def column(self, name: str) -> np.ndarray:
        """One output as a flat column: a profile is flattened in segment order."""
        y = np.asarray(self.Y[name], float)
        return y if y.ndim == 1 else y.reshape(len(y), -1)

    def frame(self):
        """A pandas DataFrame of designs and outputs (profiles flattened to ``name[k]``), valid rows only."""
        import pandas as pd

        data = {n: self.X[self.valid, i] for i, n in enumerate(NAMES)}
        for k in SCALARS:
            data[k] = np.asarray(self.Y[k], float)[self.valid]
        for k in PROFILES:
            y = np.asarray(self.Y[k], float)
            for j in range(y.shape[1]):
                data[f"{k}[{j}]"] = y[self.valid, j]
        return pd.DataFrame(data)

    # -------------------------------------------------------------------------------------------- persistence
    def save(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {"X": self.X, "valid": self.valid.astype(np.int8)}
        for k in SCALARS:
            arrays[k] = np.asarray(self.Y[k], float)
        for k in PROFILES:
            arrays[k] = np.asarray(self.Y[k], float)
        path = folder / "dataset.npz"
        np.savez_compressed(path, **arrays)
        (folder / "dataset.json").write_text(json.dumps({"names": list(NAMES), **self.meta}, indent=2))
        return path

    @classmethod
    def load(cls, folder: str | Path) -> MotorDataset:
        folder = Path(folder)
        meta = json.loads((folder / "dataset.json").read_text())
        stored = tuple(meta.pop("names", ()))
        if stored and stored != tuple(NAMES):
            raise ValueError(f"the dataset was built for design variables {list(stored)}, this version uses "
                             f"{list(NAMES)}: rebuild it instead of reusing the numbers")
        z = np.load(folder / "dataset.npz")
        Y = {k: z[k] for k in SCALARS}
        Y.update({k: z[k] for k in PROFILES if k in z})
        return cls(X=z["X"], Y=Y, valid=z["valid"].astype(bool), meta=meta)


def evaluate_chunk(args) -> dict[str, Any]:
    """Evaluate a block of design vectors with a fresh reference model (the entry point of a worker process)."""
    X, cfg = args
    model = Motor3DModel(**cfg)
    t0 = time.perf_counter()
    out = model.evaluate(np.asarray(X, float))
    out["_seconds"] = time.perf_counter() - t0
    out["_n_fea"] = model.count
    return out


def build_dataset(n: int = 256, *, seed: int = 0, method: str = "lhs", model: Motor3DModel | None = None,
                  workers: int = 1, chunk: int | None = None, extra: Sequence[Motor3DDesign] = (),
                  progress: Callable[[int, int], None] | None = None) -> MotorDataset:
    """``n`` designs of the box, labelled by the reference model.

    The designs are drawn with the library's DOE (:func:`pinneapple_data.doe.design`) and filtered by the geometry
    test, so no case is spent on a cross-section the mesher would refuse. ``extra`` designs (a baseline, a candidate
    from an earlier study) are evaluated first and count towards nothing but the record. ``workers`` > 1 splits the
    cases over processes; every process builds its own model, so no state is shared.
    """
    from pinneapple_data.doe.designs import design

    model = model or Motor3DModel()
    cfg = {"n_gap": model.n_gap, "n_positions": model.n_positions,
           "n_segments": model.n_segments, "magnet_T_guess": model.magnet_T_guess,
           "fea_iterations": model.fea_iterations, "coolant_flow_ref": model.coolant_flow_ref,
           "Hcj20": model.Hcj20, "v_dc": model.v_dc, "loss_steps": model.loss_steps}
    space = parameter_space()

    def feasible(frame) -> np.ndarray:
        return np.array([not Motor3DDesign.from_mapping(row).sector_errors() for row in frame.to_dict("records")])

    doE = design(space, n, method=method, seed=seed, feasible=feasible)
    X = np.array([[row[c] for c in NAMES] for row in doE.cases.to_dict("records")], float)
    if len(extra):
        X = np.vstack([np.array([d.as_vector() for d in extra], float), X])

    t0 = time.perf_counter()
    if workers <= 1 or len(X) <= 1:
        blocks = [evaluate_chunk((X, cfg))]
        done = len(X)
        if progress:
            progress(done, len(X))
    else:
        from concurrent.futures import ProcessPoolExecutor

        step = chunk or max(1, len(X) // (workers * 3))
        parts = [(X[i:i + step], cfg) for i in range(0, len(X), step)]
        blocks, done = [], 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for out in pool.map(evaluate_chunk, parts):
                blocks.append(out)
                done += len(out["_valid"])
                if progress:
                    progress(done, len(X))
    elapsed = time.perf_counter() - t0

    Y: dict[str, np.ndarray] = {k: np.concatenate([b[k] for b in blocks]) for k in SCALARS}
    for k in PROFILES:
        Y[k] = np.concatenate([b[k] for b in blocks])
    valid = np.concatenate([b["_valid"] for b in blocks]).astype(bool)
    meta = {"created": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": int(seed), "method": method,
            "n_requested": int(n), "n_cases": int(len(X)), "n_valid": int(valid.sum()),
            "rejection_rate": float(1.0 - valid.mean()) if len(valid) else float("nan"),
            "model": cfg, "model_description": model.description,
            "seconds": round(elapsed, 1), "seconds_per_design": round(elapsed / max(len(X), 1), 3),
            "fea_solves": int(sum(b["_n_fea"] for b in blocks)),
            "bounds": {v.name: [v.low, v.high] for v in VARIABLES},
            "units": {v.name: v.unit for v in VARIABLES},
            "outputs": {o.name: {"unit": o.unit, "larger_is_better": o.larger_is_better} for o in OUTPUTS}}
    return MotorDataset(X=X, Y=Y, valid=valid, meta=meta)
