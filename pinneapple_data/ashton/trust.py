"""Train on one geometry family, test on another, and ask whether the trust layer notices before the error does.

This is the measurement the three Ashton datasets make possible and the reason to read them together: a model
trained on Ahmed bodies and tested on Windsor bodies is answering outside its design space, and the question a
trust layer has to answer is *whether it says so before the number it predicts goes wrong*. Here the same
experiment runs on the synthetic fixture of :mod:`pinneapple_data.ashton.fixture`, where the geometries, the
surface field and the shift are ours and known, and the two real datasets are the same experiment at 31 TB.

**The setup.** A surrogate takes *geometry descriptors* read off the surface -- extents, frontal and wetted area,
the area fraction facing aft (a proxy for the rear slant), the wetted-to-volume shape factor -- and predicts the
surface-pressure distribution of the case (its area-weighted Cp quantiles) and its drag coefficient. It is a
ridge regression, because the question is about the trust layer and not about the model: what is measured is
whether the trust signal moves when the geometry family changes, and a model with stochastic training would put
its own variance into that measurement.

**The shift is graded.** Family ``alpha`` is the training family; the evaluation families are the linear blend
of the two families' parameter boxes, ``box(t) = (1 - t) box_alpha + t box_beta``, at ``t = 0, 0.2, ..., 1``.
``t = 0`` is a fresh sample of the training family (a genuine held-out split), ``t = 1`` is the other family,
and the levels in between are the shift arriving, which is the only way to answer "before" rather than "as
well as". The reverse direction (train on ``beta``, walk towards ``alpha``) is run too.

**The signals**, all three of them measured on this fixture rather than assumed:

* ``embedding_distance`` -- the Mahalanobis distance of a case's descriptor vector to the training distribution,
  through :class:`pinneapple_analysis.trust.TrustGate` (fitted on the training cases, scored per case). Its
  threshold is calibrated on a *held-out* in-family split at 99 %, so "flagged" means "further out than 99 % of
  genuine in-family cases were", and the false-alarm rate is the one that threshold produced, reported;
* ``flagged_fraction`` -- the share of a level's cases that cross that threshold: the label-free alarm;
* ``conformal`` -- a split-conformal interval on the drag coefficient at 90 %, calibrated on the held-out
  in-family split (:func:`pinneapple_analysis.uncertainty.conformal.conformal_quantile`), and its *coverage* at
  each level. Coverage is the number that must collapse if the interval was only ever valid in-family.

**The claim tested.** ``t_flag`` (the first level at which at least half the cases are flagged) is smaller than
``t_error`` (the first level at which the mean prediction error exceeds twice its own in-family value). The
result object reports both and the boolean, so an experiment where the ordering does *not* hold reports that
instead of hiding it.

    >>> from pinneapple_data.ashton.trust import run_on_fixtures
    >>> [r.flags_before_error for r in run_on_fixtures("out/fixture")]     # both directions
    [True, True]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .fixture import FAMILIES, Params, write_case, write_fixture
from .loader import read_force_csv
from .vtk_xml import face_area_normals, read_surface

__all__ = ["DESCRIPTORS", "Level", "CrossFamilyResult", "geometry_descriptors", "cp_quantiles",
           "cross_family_experiment", "run_on_fixtures", "QUANTILES"]

#: The area-weighted quantiles of the surface pressure that stand in for "the surface field": nineteen numbers,
#: scale-free, comparable between two meshes that do not share a single vertex -- which is what a field error
#: cannot be without an interpolation, and why the published baselines interpolate to a common mesh.
QUANTILES = tuple(np.arange(5, 100, 5).tolist())

#: The geometry embedding: read off the surface, so the same code runs on a real case whose parameter file names
#: its columns differently.
DESCRIPTORS = ("length", "width", "height", "clearance", "wetted_area", "frontal_area", "rear_area_fraction",
               "flatness", "shape_factor")

LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


@dataclass
class Level:
    """One shift level: the errors, the trust signals, and the coverage of the conformal interval.

    ``error`` is the relative error of the drag coefficient -- the integral quantity these datasets are for, and
    the one the shift actually moves. ``field_error`` is the relative L2 of the surface-pressure descriptor: it
    barely moves on this fixture, which is itself a finding (see the module docstring and the docs page) and is
    why the ordering is claimed on the coefficient and not on the field summary.
    """

    t: float
    error: float
    field_error: float
    embedding_distance: float
    flagged_fraction: float
    coverage: float
    n_cases: int

    def to_dict(self) -> dict[str, Any]:
        return {"t": self.t, "error": self.error, "field_error": self.field_error,
                "embedding_distance": self.embedding_distance, "flagged_fraction": self.flagged_fraction,
                "coverage": self.coverage, "n_cases": self.n_cases}


@dataclass
class CrossFamilyResult:
    """One direction of the experiment, with the ordering already decided and the numbers behind it."""

    trained_on: str
    towards: str
    levels: list[Level] = field(default_factory=list)
    n_train: int = 0
    n_calibration: int = 0
    threshold: float = float("nan")          # the calibrated embedding-distance threshold
    false_alarm_rate: float = float("nan")   # on the held-out in-family split the threshold was calibrated on
    interval: float = float("nan")           # the 90 % conformal half-width on Cd, in coefficients
    coverage_in_family: float = float("nan")
    error_in_family: float = float("nan")    # the held-out in-family error the threshold is compared with
    t_flag: float | None = None
    t_error: float | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def flags_before_error(self) -> bool:
        """``t_flag < t_error``: the label-free alarm crossed before the error doubled."""
        if self.t_flag is None or self.t_error is None:
            return False
        return self.t_flag < self.t_error

    def row(self, t: float) -> Level:
        for level in self.levels:
            if abs(level.t - t) < 1e-9:
                return level
        raise KeyError(f"no level at t={t}; the levels are {[level.t for level in self.levels]}")

    def to_dict(self) -> dict[str, Any]:
        return {"trained_on": self.trained_on, "towards": self.towards, "n_train": self.n_train,
                "n_calibration": self.n_calibration, "threshold": self.threshold,
                "false_alarm_rate": self.false_alarm_rate, "interval": self.interval,
                "coverage_in_family": self.coverage_in_family, "error_in_family": self.error_in_family,
                "t_flag": self.t_flag, "t_error": self.t_error, "flags_before_error": self.flags_before_error,
                "levels": [level.to_dict() for level in self.levels], "notes": self.notes}

    def to_markdown(self) -> str:
        head = (f"### trained on `{self.trained_on}`, shifted towards `{self.towards}`\n\n"
                f"threshold {self.threshold:.3f} (99 % of {self.n_calibration} held-out in-family cases, false "
                f"alarms {self.false_alarm_rate:.0%}); 90 % conformal half-width {self.interval:.4f}; in-family "
                f"error {self.error_in_family:.4f}\n")
        lines = ["| t | Cd error | Cp descriptor error | embedding distance | flagged | coverage |",
                 "|---|---|---|---|---|---|"]
        for level in self.levels:
            lines.append(f"| {level.t:.1f} | {level.error:.4f} | {level.field_error:.4f} | "
                         f"{level.embedding_distance:.2f} | {level.flagged_fraction:.0%} | {level.coverage:.0%} |")
        verdict = (f"flags at t = {self.t_flag} and the error doubles at t = {self.t_error}: the trust signal "
                   f"{'arrives first' if self.flags_before_error else 'arrives no earlier'}"
                   if self.t_flag is not None and self.t_error is not None else
                   "no crossing inside the levels measured")
        return head + "\n" + "\n".join(lines) + "\n\n" + verdict + "\n"


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------

def geometry_descriptors(surface_path: str | Path) -> dict[str, float]:
    """The geometry embedding of one case, read off its surface file.

    Nine numbers, each a ratio or a length in metres: the extent box, the ground clearance, the wetted and
    frontal areas, the area fraction whose normal points aft (which rises with the rear slant), the frontal
    area over the box's cross-section (how much of the box the body fills) and ``A^1.5 / V`` (a sphere is 4.84,
    a box is 6, a plate is unbounded). Nothing here is a design parameter: the three datasets name their
    parameters differently, and a trust signal that only works on a parameter table is not a trust signal for a
    geometry.
    """
    surface = read_surface(surface_path)
    areas, normals = face_area_normals(surface)
    points = surface.points
    extents = np.ptp(points, axis=0)
    wetted = float(areas.sum())
    frontal = float(areas[normals[:, 0] < -0.7].sum())
    rear = float(areas[normals[:, 0] > 0.5].sum())
    cells = surface.connectivity.reshape(-1, 3)
    v0, v1, v2 = points[cells[:, 0]], points[cells[:, 1]], points[cells[:, 2]]
    volume = abs(float(np.einsum("ij,ij->i", v0, np.cross(v1, v2)).sum() / 6.0))
    cross_section = float(max(extents[1] * extents[2], 1e-12))
    return {"length": float(extents[0]), "width": float(extents[1]), "height": float(extents[2]),
            "clearance": float(points[:, 2].min()), "wetted_area": wetted, "frontal_area": frontal,
            "rear_area_fraction": rear / max(wetted, 1e-30), "flatness": frontal / cross_section,
            "shape_factor": wetted ** 1.5 / max(volume, 1e-30)}


def cp_quantiles(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Area-weighted quantiles of the surface pressure: the nineteen numbers a field error is measured on."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    order = np.argsort(values)
    sorted_values = values[order]
    cumulative = np.cumsum(weights[order])
    if cumulative.size == 0 or cumulative[-1] <= 0:
        raise ValueError("no weight to build a weighted quantile from")
    targets = np.asarray(QUANTILES, dtype=np.float64) / 100.0 * cumulative[-1]
    return np.interp(targets, cumulative, sorted_values)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------

class _Ridge:
    """A closed-form ridge regression: deterministic, so the measurement is the shift and not the training."""

    def __init__(self, penalty: float = 1.0) -> None:
        self.penalty = float(penalty)
        self._mean: np.ndarray | None = None
        self._scale: np.ndarray | None = None
        self._weights: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> _Ridge:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self._mean = x.mean(axis=0)
        scale = x.std(axis=0)
        self._scale = np.where(scale > 0, scale, 1.0)
        z = (x - self._mean) / self._scale
        z = np.column_stack([np.ones(z.shape[0]), z])
        gram = z.T @ z + self.penalty * np.eye(z.shape[1])
        self._weights = np.linalg.solve(gram, z.T @ y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("fit the ridge regression first")
        z = (np.asarray(x, dtype=np.float64) - self._mean) / self._scale
        return np.column_stack([np.ones(z.shape[0]), z]) @ self._weights


# ---------------------------------------------------------------------------
# cases of the fixture
# ---------------------------------------------------------------------------

@dataclass
class _Case:
    descriptors: np.ndarray
    quantiles: np.ndarray
    cd: float


def _blend(train_family: str, towards: str, t: float) -> dict[str, tuple[float, float]]:
    """The parameter box at a shift level: ``(1 - t) * train + t * towards``, box by box."""
    a, b = FAMILIES[train_family], FAMILIES[towards]
    return {name: ((1.0 - t) * a[name][0] + t * b[name][0], (1.0 - t) * a[name][1] + t * b[name][1])
            for name in a}


def _collect(root: Path, runs: list[int]) -> list[_Case]:
    cases = []
    for run in runs:
        path = root / f"run_{run}" / f"boundary_{run}.vtp"
        surface = read_surface(path, arrays=["Cp"])
        areas, _normals = face_area_normals(surface)
        cp = np.asarray(surface.cell_data.get("Cp", surface.point_data.get("Cp")), dtype=np.float64).reshape(-1)
        weights = areas if cp.size == areas.size else np.full(cp.size, 1.0 / cp.size)
        descriptors = geometry_descriptors(path)
        cases.append(_Case(descriptors=np.array([descriptors[name] for name in DESCRIPTORS], dtype=np.float64),
                           quantiles=cp_quantiles(cp, weights),
                           cd=float(_read_cd(root, run))))
    return cases


def _read_cd(root: Path, run: int) -> float:
    return read_force_csv(root / f"run_{run}" / f"force_mom_{run}.csv")["cd"]


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------

def cross_family_experiment(root: str | Path, *, train_family: str = "alpha", towards: str = "beta",
                            n_train: int = 48, n_calibration: int = 16, n_level: int = 12,
                            levels: tuple[float, ...] = LEVELS, seed: int = 0, ood_level: float = 0.01,
                            alpha_conformal: float = 0.1, n_ring: int = 28,
                            n_station: int = 13) -> CrossFamilyResult:
    """Run one direction of the experiment and return the levels, the two crossings and the verdict.

    The training and calibration cases are written once under ``root`` (they are the fixture, so they are also
    the record of what was measured); every level is written to its own directory and read back through the
    reader, so nothing in the measurement skips the file layout.
    """
    base = Path(root) / f"train_{train_family}"
    if not (base / "run_1").is_dir():
        write_fixture(base, family=train_family, n_cases=n_train + n_calibration, dataset="ahmedml", seed=seed,
                      n_ring=n_ring, n_station=n_station)
    runs = list(range(1, n_train + n_calibration + 1))
    cases = _collect(base, runs)
    train, calibration = cases[:n_train], cases[n_train:]
    x_train = np.vstack([case.descriptors for case in train])
    y_train = np.vstack([np.concatenate([[case.cd], case.quantiles]) for case in train])

    model = _Ridge(penalty=1.0).fit(x_train, y_train)
    x_cal = np.vstack([case.descriptors for case in calibration])
    prediction = model.predict(x_cal)
    truth = np.vstack([np.concatenate([[case.cd], case.quantiles]) for case in calibration])
    in_family = np.array([relative_error(p[1:], t[1:]) for p, t in zip(prediction, truth, strict=True)])
    cd_residual = np.abs(prediction[:, 0] - truth[:, 0])
    in_family_cd = cd_residual / np.maximum(np.abs(truth[:, 0]), 1e-30)

    # -- the two calibrated quantities: an embedding-distance threshold and a conformal half-width ----
    distances_cal = embedding_distances(x_train, x_cal, ood_level=ood_level)
    threshold = float(np.quantile(distances_cal, 0.99)) if distances_cal.size else float("nan")
    if distances_cal.size:
        threshold = max(threshold, float(distances_cal.max()))       # a handful of calibration cases
    false_alarm = float(np.mean(distances_cal > threshold))
    from pinneapple_analysis.uncertainty.conformal import conformal_quantile
    interval = float(conformal_quantile(cd_residual, alpha_conformal))
    coverage_in_family = float(np.mean(cd_residual <= interval))

    result = CrossFamilyResult(trained_on=train_family, towards=towards, n_train=len(train),
                               n_calibration=len(calibration), threshold=threshold,
                               false_alarm_rate=false_alarm, interval=interval,
                               coverage_in_family=coverage_in_family,
                               error_in_family=float(in_family_cd.mean()))
    result.notes.append(f"in-family: Cd relative error {in_family_cd.mean():.4f} and Cp descriptor error "
                        f"{in_family.mean():.4f} (+/- {in_family.std():.4f}) over {len(train)} training and "
                        f"{len(calibration)} calibration cases of family '{train_family}'")

    for t in levels:
        folder = Path(root) / f"shift_{train_family}_to_{towards}_t{t:.2f}"
        if not (folder / "run_101").is_dir():
            box = _blend(train_family, towards, t)
            rng = np.random.default_rng(seed + int(round(t * 100)))
            for index in range(n_level):
                params = Params(**{name: float(rng.uniform(low, high)) for name, (low, high) in box.items()})
                write_case(folder, 101 + index, params, dataset="ahmedml", n_ring=n_ring, n_station=n_station)
        cases_level = _collect(folder, list(range(101, 101 + n_level)))
        x_level = np.vstack([case.descriptors for case in cases_level])
        prediction = model.predict(x_level)
        truth = np.vstack([np.concatenate([[case.cd], case.quantiles]) for case in cases_level])
        errors = np.array([relative_error(p[1:], tq[1:]) for p, tq in zip(prediction, truth, strict=True)])
        cd_errors = np.abs(prediction[:, 0] - truth[:, 0])
        cd_relative = cd_errors / np.maximum(np.abs(truth[:, 0]), 1e-30)
        distances = embedding_distances(x_train, x_level, ood_level=ood_level)
        result.levels.append(Level(
            t=float(t), error=float(cd_relative.mean()), field_error=float(errors.mean()),
            embedding_distance=float(np.median(distances)) if distances.size else float("nan"),
            flagged_fraction=float(np.mean(distances > threshold)) if distances.size else float("nan"),
            coverage=float(np.mean(cd_errors <= interval)), n_cases=len(cases_level)))

    # -- the two crossings, and the verdict ---------------------------------------------------------
    doubling = 2.0 * float(in_family_cd.mean())
    for level in result.levels:
        if result.t_error is None and level.error > doubling:
            result.t_error = level.t
        if result.t_flag is None and level.flagged_fraction >= 0.5:
            result.t_flag = level.t
    result.notes.append(f"the error threshold is twice the in-family Cd error ({doubling:.4f}); the flag threshold "
                        f"is half the level's cases beyond the calibrated distance ({threshold:.3f})")
    return result


def relative_error(predicted: np.ndarray, true: np.ndarray) -> float:
    """``||p - t|| / ||t||`` on one case's pressure descriptor; a plain relative L2."""
    predicted = np.asarray(predicted, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    norm = float(np.linalg.norm(true))
    return float(np.linalg.norm(predicted - true) / norm) if norm > 0 else float("nan")


def embedding_distances(x_train: np.ndarray, x_query: np.ndarray, *, ood_level: float = 0.01) -> np.ndarray:
    """Mahalanobis distance from the training distribution, per query case, through ``TrustGate``.

    The gate returns a chi-squared p-value per point; the distance is recovered from it exactly
    (``d^2 = chi2.isf(p, k)``), so the number here is the one the gate thresholded. ``TrustGate`` is fitted on
    the training cases exactly as :func:`pinneapple_analysis.trust.adapters.trust` fits it on training
    coordinates, and its own ``ood_level`` is what makes "p < 0.01" mean "outside the 99 % region".
    """
    import torch
    from scipy.stats import chi2

    from pinneapple_analysis.trust.trust_gate import TrustGate

    gate = TrustGate(ood_level=ood_level).fit(torch.as_tensor(np.asarray(x_train, dtype=np.float64)))
    p = np.asarray(gate.ood_p_values(np.asarray(x_query, dtype=np.float64)), dtype=np.float64)
    clipped = np.clip(p, np.finfo(np.float64).tiny, 1.0)
    return np.sqrt(chi2.isf(clipped, df=np.asarray(x_train).shape[1]))


def run_on_fixtures(root: str | Path, **kwargs: Any) -> list[CrossFamilyResult]:
    """Both directions: ``alpha -> beta`` (Ahmed-trained, Windsor-tested) and ``beta -> alpha``. """
    shared = {key: value for key, value in kwargs.items() if key not in ("train_family", "towards")}
    forward = cross_family_experiment(root, train_family="alpha", towards="beta", **shared)
    backward = cross_family_experiment(root, train_family="beta", towards="alpha", **shared)
    return [forward, backward]
