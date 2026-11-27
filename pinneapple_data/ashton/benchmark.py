"""The benchmark table: force coefficients and surface pressure, measured where the data is, declared where it
is published.

Two kinds of number go in one table, and the difference is the point of the module:

* a **declared** number is one somebody else published, quoted with its paper, its table and -- where the two
  datasets scale their fields differently -- its scale. :data:`BASELINES` holds them and nothing in this module
  can compute or overwrite one;
* a **measured** number is computed here from a case that is on disk: the drag integrated from the case's own
  surface pressure (:meth:`pinneapple_data.ashton.loader.AshtonData.measured_forces`), and, when a model is
  passed in, the surface-pressure metrics of that model's prediction against the case's field.

The harness is therefore usable at three levels, and says which one it is in the row's ``status``: ``measured``
(the data was read), ``declared`` (the number is the published one; the data is not here), and
``awaiting_data`` (a row exists, the columns that need the terabytes are empty, with the size and the call that
would fill them).

What is published, and what is not
----------------------------------
The AB-UPT paper (Alkin et al., arXiv:2502.09692, v4) is the one reference that treats these data as a
benchmark, and it reports **surface-pressure prediction error on DrivAerML and AhmedML** -- Table 1 and Table 2,
MSE / L2 / MAE: 0.0388 L2 for Transolver and 0.0434 for GP-UPT on DrivAerML, 0.0272 for GP-UPT on AhmedML.

Two things are worth stating plainly rather than rounding off:

* **WindsorML is not in that paper.** The version read here (v4, March 2025) mentions it zero times. The
  WindsorML number this table declares is the dataset's own CFD validation (Cd 0.31 / 0.34 / 0.33 on three
  grids against 0.33 from experiment, Ashton et al. 2024, Table 1), which is a different kind of number -- a
  solver against a wind tunnel, not a surrogate against a solver -- and is labelled as such;
* **the two pressure columns are not on the same scale.** AhmedML's boundary file carries Cp, so its MSE is
  ~1e-3; DrivAerML's carries pressure, so its MSE is ~1e3. The L2 column is scale-free and comparable; the MSE
  and MAE columns are not comparable across the two datasets, and the table records the scale it assumed.

    >>> from pinneapple_data.ashton.benchmark import benchmark_table
    >>> table = benchmark_table({"ahmedml": "data/ashton/ahmedml"})     # no data: declared rows only
    >>> print(table.to_markdown())
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .catalog import DATASETS
from .catalog import dataset as _dataset
from .loader import AshtonData, open_dataset

__all__ = ["relative_l2", "mse", "mae", "surface_metrics", "Declared", "BASELINES", "declared",
           "BenchmarkRow", "BenchmarkTable", "benchmark_table", "constant_field_baseline", "COLLAPSE_NOTE"]

#: What the ``status`` of a row means.
MEASURED, DECLARED, AWAITING = "measured", "declared", "awaiting_data"

#: Said once, in the table's footer, because a reader who compares the two pressure columns without it will
#: conclude that a model is a thousand times better on one dataset than on the other.
COLLAPSE_NOTE = ("MSE and MAE are in the units of the field the dataset publishes (Cp for AhmedML, Pa for "
                 "DrivAerML); only L2 is scale-free and comparable between the two.")


@dataclass(frozen=True)
class Declared:
    """One published number: what it is, whose it is, and where in the paper it is printed."""

    dataset: str
    quantity: str                     # surface_pressure | drag_coefficient
    metric: str                       # L2 | MSE | MAE | Cd | Cl
    value: float
    model: str
    source: str
    scale: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "quantity": self.quantity, "metric": self.metric, "value": self.value,
                "model": self.model, "source": self.source, "scale": self.scale, "note": self.note}


_ABUPT = "Alkin et al., AB-UPT, arXiv:2502.09692 (v4), Table {}"
_AHMED_PAPER = "Ashton et al., AhmedML, arXiv:2407.20801, Table 1"
_DRIVAER_PAPER = "Ashton et al., DrivAerML, arXiv:2408.11969, Table 1"
_WINDSOR_PAPER = "Ashton et al., WindsorML, arXiv:2407.19320, Table 1"

#: The published baselines, quoted. ``Cp`` and ``Pa`` mark what the field's own scale is; see the module
#: docstring for why the two pressure columns cannot be compared.
BASELINES: tuple[Declared, ...] = (
    # -- surface pressure, surrogate against CFD (AB-UPT) -------------------------------------------
    Declared("ahmedml", "surface_pressure", "L2", 0.0272, "GP-UPT (42M)", _ABUPT.format(2), "Cp"),
    Declared("ahmedml", "surface_pressure", "MSE", 0.00078, "GP-UPT (42M)", _ABUPT.format(2), "Cp"),
    Declared("ahmedml", "surface_pressure", "MAE", 0.0085, "GP-UPT (42M)", _ABUPT.format(2), "Cp"),
    Declared("ahmedml", "surface_pressure", "L2", 0.0418, "GINO (52M)", _ABUPT.format(2), "Cp"),
    Declared("ahmedml", "surface_pressure", "MSE", 0.00168, "GINO (52M)", _ABUPT.format(2), "Cp"),
    Declared("ahmedml", "surface_pressure", "MAE", 0.0128, "GINO (52M)", _ABUPT.format(2), "Cp"),

    Declared("drivaerml", "surface_pressure", "L2", 0.0388, "Transolver (3.9M)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "MSE", 190.58, "Transolver (3.9M)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "MAE", 8.32, "Transolver (3.9M)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "L2", 0.0434, "GP-UPT (3.6M, point decoder)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "MSE", 239.63, "GP-UPT (3.6M, point decoder)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "MAE", 9.35, "GP-UPT (3.6M, point decoder)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "L2", 0.0497, "GP-UPT (4.7M, field decoder)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "L2", 0.0914, "UPT (4.0M)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "L2", 0.1043, "PointNet (3.5M)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "L2", 0.1378, "GINO (15.7M)", _ABUPT.format(1), "Pa"),
    Declared("drivaerml", "surface_pressure", "L2", 0.1615, "RegDGCNN (1.4M)", _ABUPT.format(1), "Pa"),

    # -- force coefficients, solver against wind tunnel (the dataset papers) -------------------------
    Declared("ahmedml", "drag_coefficient", "Cd", 0.285, "OpenFOAM hybrid RANS-LES, fine mesh (the dataset)",
             _AHMED_PAPER, "1", "the reference Ahmed body at a 25 deg slant; the 500 variants range around it"),
    Declared("ahmedml", "drag_coefficient", "Cl", 0.343, "OpenFOAM hybrid RANS-LES, fine mesh (the dataset)",
             _AHMED_PAPER, "1"),
    Declared("ahmedml", "drag_coefficient", "Cd", 0.299, "experiment (Meile et al.)", _AHMED_PAPER, "1"),
    Declared("drivaerml", "drag_coefficient", "Cd", 0.274, "CFD scale-resolving, reference DrivAer (case 2a)",
             _DRIVAER_PAPER, "1"),
    Declared("drivaerml", "drag_coefficient", "Cd", 0.255, "experiment (Hupertz et al.)", _DRIVAER_PAPER, "1"),
    Declared("windsorml", "drag_coefficient", "Cd", 0.33, "GPU WMLES, finest grid (G3), the reference Windsor body",
             _WINDSOR_PAPER, "1"),
    Declared("windsorml", "drag_coefficient", "Cd", 0.33, "experiment", _WINDSOR_PAPER, "1"),

    # -- WindsorML has no surrogate baseline in AB-UPT; said so, rather than left blank --------------
    Declared("windsorml", "surface_pressure", "L2", float("nan"), "none published in AB-UPT",
             _ABUPT.format(1), "",
             "the AB-UPT paper of Alkin et al. reports DrivAerML and AhmedML only (v4); WindsorML's published "
             "numbers are the solver-vs-experiment coefficients above. This row exists so the missing column "
             "is visible in the table rather than absent from it"),
)


def declared(dataset: str | None = None, quantity: str | None = None) -> list[Declared]:
    """The published numbers, optionally for one dataset and/or one quantity."""
    rows = [row for row in BASELINES if (dataset is None or row.dataset == dataset)
            and (quantity is None or row.quantity == quantity)]
    return rows


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def relative_l2(predicted: Any, true: Any) -> float:
    """``||p - t||_2 / ||t||_2`` -- the scale-free column, and the one the published numbers use."""
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    if p.shape != t.shape:
        raise ValueError(f"prediction and truth differ in shape: {p.shape} against {t.shape}")
    norm = float(np.linalg.norm(t))
    return float(np.linalg.norm(p - t) / norm) if norm > 0 else float("nan")


def mse(predicted: Any, true: Any) -> float:
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.mean((p - t) ** 2)) if p.size else float("nan")


def mae(predicted: Any, true: Any) -> float:
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    return float(np.mean(np.abs(p - t))) if p.size else float("nan")


def surface_metrics(predicted: Any, true: Any, *, weights: Any = None) -> dict[str, float]:
    """The three metrics of the published tables on one case's surface field.

    ``weights`` is the area of each face (or the quadrature weight of each point): with it the MSE and MAE are
    area-averaged -- which is what a surface field's error should be, so that a 250-vertex nose does not weigh
    as much as half the underbody. The L2 is scale-free and is not affected by a uniform weight.
    """
    p = np.asarray(predicted, dtype=np.float64).reshape(-1)
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != t.shape:
            raise ValueError(f"weights are {w.shape} and the field is {t.shape}")
        w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / max(w.size, 1))
        return {"L2": relative_l2(p, t), "MSE": float(np.sum(w * (p - t) ** 2)),
                "MAE": float(np.sum(w * np.abs(p - t)))}
    return {"L2": relative_l2(p, t), "MSE": mse(p, t), "MAE": mae(p, t)}


# ---------------------------------------------------------------------------
# a model that runs offline
# ---------------------------------------------------------------------------

def constant_field_baseline(cases: Iterable[Any], *, field: str | None = None) -> Callable[[Any], np.ndarray]:
    """The area-weighted mean of the training cases' surface field, predicted for every case.

    It is the honest first row of a field benchmark: a surrogate that predicts nothing has to be beaten by every
    surrogate that predicts something, and until it is in the table a 0.03 L2 has no scale. It needs no training,
    no GPU and no network -- which is why it is the baseline this harness can actually run here.
    """
    sums, weights = 0.0, 0.0
    for case in cases:
        name, values, location = case.field_of(field or "cp")
        del name
        area = _areas_of(case, values.size, location)
        sums += float(np.sum(values * area))
        weights += float(np.sum(area))
    if weights <= 0:
        raise ValueError("no area could be computed for the training cases, so there is no constant to take")
    mean = sums / weights

    def predict(case: Any) -> np.ndarray:
        _name, values, _location = case.field_of(field or "cp")
        return np.full(values.size, mean, dtype=np.float64)

    predict.mean_cp = mean                                     # type: ignore[attr-defined]
    return predict


def _areas_of(case: Any, size: int, location: str) -> np.ndarray:
    """The area of each face or point of a case: the dataset's own quadrature when it is there, else geometry."""
    from .vtk_xml import face_area_normals, node_area_normals

    surface = case.surface(arrays=())
    if "areas" in case.files:
        published = np.asarray(np.load(case.files["areas"]).reshape(-1), dtype=np.float64)
        if published.size == size:
            return published
    if location == "cell":
        areas, _normals = face_area_normals(surface)
        return areas
    node_area, _normals = node_area_normals(surface)
    return node_area


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkRow:
    """One row: a dataset and a quantity, with whatever could be measured and whatever is declared."""

    dataset: str
    quantity: str
    status: str
    measured: dict[str, float] = field(default_factory=dict)
    declared: list[Declared] = field(default_factory=list)
    cases: int = 0
    model: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"dataset": self.dataset, "quantity": self.quantity, "status": self.status,
                "measured": self.measured, "cases": self.cases, "model": self.model, "note": self.note,
                "declared": [row.to_dict() for row in self.declared]}


@dataclass
class BenchmarkTable:
    """The table: rows, the note that says which columns are comparable, and the markdown of both."""

    rows: list[BenchmarkRow] = field(default_factory=list)
    note: str = COLLAPSE_NOTE

    def add(self, row: BenchmarkRow) -> None:
        self.rows.append(row)

    def measured_rows(self) -> list[BenchmarkRow]:
        return [row for row in self.rows if row.status == MEASURED]

    def to_dict(self) -> dict[str, Any]:
        return {"rows": [row.to_dict() for row in self.rows], "note": self.note}

    def to_markdown(self) -> str:
        lines = ["| dataset | quantity | status | cases | measured | declared (published) | source |",
                 "|---|---|---|---|---|---|---|"]
        for row in self.rows:
            measured = ", ".join(f"{name} {value:.4g}" for name, value in row.measured.items()) or "--"
            best = _best(row.declared)
            if best is None:
                published = "--"
                source = row.declared[0].source if row.declared else "--"
                if row.declared:
                    published = f"-- ({row.declared[0].note[:60]}...)" if row.declared[0].note else "--"
            else:
                published = f"{best.metric} {best.value:.4g} ({best.model}; {len(row.declared)} published rows)"
                source = best.source
            lines.append(f"| {row.dataset} | {row.quantity} | {row.status} | {row.cases or '--'} | {measured} | "
                         f"{published} | {source} |")
        lines.append("")
        lines.append(self.note)
        return "\n".join(lines)


def _best(rows: Iterable[Declared]) -> Declared | None:
    """The row to print for a quantity: the smallest value of the most comparable metric, CFD before experiment.

    ``L2`` before ``MSE``/``MAE`` (it is the only scale-free one and the one the published tables lead with), and
    a solver's own number before a wind-tunnel one when both are declared -- the column is about what a model has
    to beat on this data, so the data's own CFD result is the right representative of it.
    """
    order = {"L2": 0, "Cd": 0, "MSE": 1, "MAE": 2, "Cl": 3}
    paintable = [row for row in rows if not np.isnan(row.value)]
    if not paintable:
        return None
    metric = min(order.get(row.metric, 9) for row in paintable)
    candidates = [row for row in paintable if order.get(row.metric, 9) == metric]
    cfd = [row for row in candidates if "experiment" not in row.model.lower()]
    return min(cfd or candidates, key=lambda row: row.value)


def benchmark_table(sources: Mapping[str, str | Path | AshtonData] | None = None, *,
                    datasets: Iterable[str] | None = None, limit: int | None = 8,
                    model: Callable[[Any], np.ndarray] | None = None, model_name: str = "constant baseline",
                    train_cases: Iterable[Any] | None = None) -> BenchmarkTable:
    """Build the table: measured where a case is on disk, declared where a number was published.

    ``sources`` maps a dataset id to the root it is stored under (or to an already opened
    :class:`~pinneapple_data.ashton.loader.AshtonData`); a dataset that is not on disk still gets its declared
    rows, so the table can be produced on a machine that will never hold 31 TB. ``limit`` caps how many cases
    are read per dataset (each case is a 500 MB surface file in the real thing; in the fixture it is 40 KB).
    ``model`` is what fills the surface-pressure column: with no model the column is empty and the row says so,
    because a surface-pressure error is a property of a model, not of the data. With no ``model`` but with
    ``train_cases``, the constant baseline of :func:`constant_field_baseline` is used.
    """
    table = BenchmarkTable()
    for name in (datasets if datasets is not None else DATASETS):
        data = _dataset(name)
        root: str | Path | AshtonData = (sources or {}).get(data.id, Path("data/ashton") / data.id)
        opened = root if isinstance(root, AshtonData) else open_dataset(data, root=root)
        runs = opened.runs()[: int(limit) if limit else None]
        if model is None and train_cases is not None:
            model = constant_field_baseline(train_cases)

        # -- force coefficients: the surface integral against the dataset's own CSV -----------------
        if runs:
            measured = [opened.measured_forces(run) for run in runs]
            errors = [row.cd_error for row in measured if row.cd_error is not None]
            force_row = BenchmarkRow(
                dataset=data.id, quantity="drag_coefficient", status=MEASURED, cases=len(measured),
                measured={"cd_mean": float(np.mean([row.cd for row in measured])),
                          "cd_csv_mean": float(np.mean([row.cd_published for row in measured
                                                        if row.cd_published is not None])) if errors else float("nan"),
                          "max_abs_error": float(np.max(np.abs(errors))) if errors else float("nan")},
                declared=declared(data.id, "drag_coefficient"),
                note="measured is -oint Cp n_x dA / A_ref on the case's own surface; the published column is the "
                     "dataset's force_mom CSV and the paper's validation")
            table.add(force_row)
        else:
            table.add(BenchmarkRow(dataset=data.id, quantity="drag_coefficient", status=AWAITING,
                                   declared=declared(data.id, "drag_coefficient"),
                                   note=f"no case under {opened.root}; {data.surface} is {data.size}"))

        # -- surface pressure: a model's error, so empty unless a model is given -------------------
        if runs and model is not None:
            scores: dict[str, list[float]] = {"L2": [], "MSE": [], "MAE": []}
            for run in runs:
                case = opened.case(run)
                _name, truth, location = case.field_of("cp")
                prediction = np.asarray(model(case), dtype=np.float64).reshape(-1)
                metrics = surface_metrics(prediction, truth, weights=_areas_of(case, truth.size, location))
                for key, value in metrics.items():
                    scores[key].append(value)
            table.add(BenchmarkRow(
                dataset=data.id, quantity="surface_pressure", status=MEASURED, cases=len(runs), model=model_name,
                measured={key: float(np.mean(values)) for key, values in scores.items()},
                declared=declared(data.id, "surface_pressure")))
        else:
            why = ("no case under " + str(opened.root)) if not runs else (
                "the data is there but no model was passed: a surface-pressure error belongs to a model, and this "
                "harness will not borrow somebody else's")
            table.add(BenchmarkRow(dataset=data.id, quantity="surface_pressure", status=AWAITING,
                                   declared=declared(data.id, "surface_pressure"), note=why))
    return table
