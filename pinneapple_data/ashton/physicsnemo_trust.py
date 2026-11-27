"""PhysicsNeMo's DoMINO inside the trust layer: does a trust signal fire before the error doubles on an unseen body?

:mod:`pinneapple_data.ashton.trust` runs this experiment with a ridge regression, because there the question is
whether the *signals* move with the geometry family and a stochastic model would put its own variance into the
measurement. Here the model is the thing under test: PhysicsNeMo's DoMINO, trained on one family of the fixture,
scored on the graded blend towards the other, and wrapped as a member of
:class:`pinneapple_physics.ensemble.PhysicsEnsemble` next to the constant-field baseline -- the member that
predicts nothing, which is the honest second expert for an ensemble whose lead should not be trusted out of family.

**The three signals, all measured on the same cases:**

* ``embedding`` -- the Mahalanobis distance of the case's *own* DoMINO embedding (the geometry encoding inside the
  model, pooled; see :meth:`~pinneapple_data.ashton.physicsnemo_models.DoMINOMember.embedding`) to the training
  distribution, through :class:`pinneapple_analysis.trust.trust_gate.TrustGate`, with its threshold calibrated at
  99 % on held-out in-family cases. Its share of a level's cases over that threshold is the label-free alarm;
* ``residual`` -- :class:`~pinneapple_data.ashton.physicsnemo_models.SurfaceResidual`, the Bernoulli bound, the
  zero-side-force condition on a symmetric body and non-negative drag: a physics check that needs no truth. It is
  calibrated the same way the embedding is -- its threshold is the 99 % quantile of the held-out in-family
  residuals, so "flagged" means "less admissible than every genuine in-family case was" -- and reported next to its
  own in-family median;
* ``conformal coverage`` -- the share of cases whose true Cp falls inside the ensemble's interval
  (:class:`~pinneapple_physics.ensemble.PhysicsEnsemble` scales the spread between the experts by an online
  conformal factor).

**What the fixture can and cannot answer.** The residual has no truth-free *field* prior to compare against here:
the only cheap pressure model the fixture offers is the law that generated its field, so a residual built on it
would be the error under another name. What is measured instead is the residual an operator actually has --
physical admissibility -- and the finding is reported as it came out: it stays at its in-family floor on the
unseen family (the failure is an error, not a violation) while it does fire on a model that is plainly unphysical
(untrained weights, measured in the same run as the control).

    >>> from pinneapple_data.ashton.physicsnemo_trust import run_on_fixtures
    >>> forward, backward = run_on_fixtures("out/pnemo", steps=240)
    >>> forward.flags_before_error, forward.residual_flags_before_error
    (True, False)            # on the fixture: the embedding catches the shift, the residual does not (see the guide)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .benchmark import relative_l2
from .fixture import Params, write_case, write_fixture
from .loader import open_dataset
from .physicsnemo_models import DoMINOMember, SurfaceResidual, availability
from .trust import LEVELS, _blend, embedding_distances

__all__ = ["MemberLevel", "CrossFamilyMemberResult", "cross_family_with_domino", "run_on_fixtures", "MEMBERS"]

#: The ensemble's members, as a row names them: the PhysicsNeMo model and the baseline that predicts nothing.
MEMBERS = ("domino", "constant field")


@dataclass
class MemberLevel:
    """One shift level: the error, the three trust signals, and what the ensemble did with them."""

    t: float
    error: float                  # area-weighted relative L2 of the DoMINO Cp field, mean over the level's cases
    cd_error: float               # |Cd(prediction) - Cd(truth)| / Cd(truth), the quantity these datasets exist for
    embedding_distance: float     # median Mahalanobis distance of the level's embeddings to the training set
    flagged_fraction: float       # share of the level's cases past the calibrated embedding threshold
    residual: float               # median admissibility residual of the DoMINO prediction
    residual_ratio: float         # that median over the in-family median: > 1 means the physics check moved
    residual_flagged_fraction: float   # share of the level's cases past the calibrated residual threshold
    coverage: float               # share of surface points inside the ensemble's interval
    ensemble_error: float         # the ensemble's own relative L2 against the reference
    active: str                   # the expert the ensemble led on, on the level's last case
    embedding: list[float]        # the level's median embedding, for a reader who wants the vector
    n_cases: int

    def to_dict(self) -> dict[str, Any]:
        payload = {name: getattr(self, name) for name in (
            "t", "error", "cd_error", "embedding_distance", "flagged_fraction", "residual", "residual_ratio",
            "residual_flagged_fraction", "coverage", "ensemble_error", "active", "n_cases")}
        payload["embedding"] = list(self.embedding)
        return payload


@dataclass
class CrossFamilyMemberResult:
    """One direction of the experiment: the crossings, the verdict on each signal, and the numbers behind it."""

    trained_on: str
    towards: str
    member: str
    levels: list[MemberLevel] = field(default_factory=list)
    n_train: int = 0
    n_calibration: int = 0
    steps: int = 0
    embedding_threshold: float = float("nan")
    false_alarm_rate: float = float("nan")
    residual_threshold: float = float("nan")     # 99 % of the held-out in-family residuals, like the embedding's
    residual_false_alarm_rate: float = float("nan")
    residual_in_family: float = float("nan")
    residual_untrained: float = float("nan")     # the control: the same residual on a member with no training
    error_in_family: float = float("nan")
    coverage_in_family: float = float("nan")
    t_flag: float | None = None                  # first level at which half the cases are flagged by the embedding
    t_residual: float | None = None              # first level at which half the cases are flagged by the residual
    t_error: float | None = None                 # first level at which the error exceeds twice its in-family value
    notes: list[str] = field(default_factory=list)

    @property
    def flags_before_error(self) -> bool:
        """``t_flag < t_error``: the embedding alarm crossed before the error doubled."""
        if self.t_flag is None or self.t_error is None:
            return False
        return self.t_flag < self.t_error

    @property
    def residual_flags_before_error(self) -> bool:
        """``t_residual < t_error``: the physics residual crossed before the error doubled."""
        if self.t_residual is None or self.t_error is None:
            return False
        return self.t_residual < self.t_error

    def row(self, t: float) -> MemberLevel:
        for level in self.levels:
            if abs(level.t - t) < 1e-9:
                return level
        raise KeyError(f"no level at t={t}; the levels are {[level.t for level in self.levels]}")

    def to_dict(self) -> dict[str, Any]:
        return {"trained_on": self.trained_on, "towards": self.towards, "member": self.member,
                "n_train": self.n_train, "n_calibration": self.n_calibration, "steps": self.steps,
                "embedding_threshold": self.embedding_threshold, "false_alarm_rate": self.false_alarm_rate,
                "residual_threshold": self.residual_threshold,
                "residual_false_alarm_rate": self.residual_false_alarm_rate,
                "residual_in_family": self.residual_in_family, "residual_untrained": self.residual_untrained,
                "error_in_family": self.error_in_family, "coverage_in_family": self.coverage_in_family,
                "t_flag": self.t_flag, "t_residual": self.t_residual, "t_error": self.t_error,
                "flags_before_error": self.flags_before_error,
                "residual_flags_before_error": self.residual_flags_before_error,
                "levels": [level.to_dict() for level in self.levels], "notes": list(self.notes)}

    def to_markdown(self) -> str:
        head = (f"### `{self.member}`, trained on `{self.trained_on}`, shifted towards `{self.towards}`\n\n"
                f"{self.n_train} training cases, {self.steps} training steps; embedding threshold "
                f"{self.embedding_threshold:.2f} (99 % of {self.n_calibration} held-out in-family cases, false "
                f"alarms {self.false_alarm_rate:.0%}); residual threshold {self.residual_threshold:.4f} on the same "
                f"cases (false alarms {self.residual_false_alarm_rate:.0%}); in-family Cp L2 "
                f"{self.error_in_family:.4f}, median admissibility residual {self.residual_in_family:.4f} (an "
                f"untrained member scores {self.residual_untrained:.3f})\n")
        lines = ["| t | Cp L2 | Cd error | embedding distance | flagged | residual | residual x floor | "
                 "flagged by residual | coverage | ensemble L2 | leads on |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for level in self.levels:
            lines.append(f"| {level.t:.1f} | {level.error:.4f} | {level.cd_error:.4f} | "
                         f"{level.embedding_distance:.2f} | {level.flagged_fraction:.0%} | {level.residual:.4f} | "
                         f"{level.residual_ratio:.2f} | {level.residual_flagged_fraction:.0%} | "
                         f"{level.coverage:.0%} | {level.ensemble_error:.4f} | {level.active} |")
        def _verdict(signal: str, crossing: float | None) -> str:
            if crossing is None or self.t_error is None:
                return f"{signal}: no crossing inside the levels measured (the error doubles at t = {self.t_error})"
            return (f"{signal} crosses at t = {crossing} and the error doubles at t = {self.t_error}: the alarm "
                    f"{'arrives first' if crossing < self.t_error else 'arrives no earlier'}")

        verdicts = [_verdict("the embedding alarm", self.t_flag),
                    _verdict("the admissibility residual", self.t_residual)]
        return head + "\n" + "\n".join(lines) + "\n\n" + "\n".join(f"* {line}" for line in verdicts) + "\n"


# ---------------------------------------------------------------------------
# the experiment
# ---------------------------------------------------------------------------

def _collect(data_root: Path, runs: list[int]) -> list[Any]:
    data = open_dataset("ahmedml", root=data_root)
    return [data.case(run) for run in runs]


def _field_error(member: Any, case: Any, prediction: Any = None) -> tuple[float, float]:
    """The Cp L2 of one case's prediction and the relative error of the drag that prediction integrates to."""
    from .vtk_xml import face_area_normals

    _name, truth, location = case.field_of("cp")
    prediction = member.predict(case) if prediction is None else prediction
    error = float(relative_l2(prediction, truth))
    if location != "cell":
        return error, float("nan")
    surface = case.surface(arrays=())
    areas, normals = face_area_normals(surface)
    a_ref = float(case.dataset.a_ref)
    cd_prediction = float((-(np.asarray(prediction) * areas) @ normals)[0] / a_ref)
    cd_truth = float((-(truth * areas) @ normals)[0] / a_ref)
    return error, abs(cd_prediction - cd_truth) / max(abs(cd_truth), 1e-30)


def cross_family_with_domino(root: str | Path, *, train_family: str = "alpha", towards: str = "beta",
                             n_train: int = 16, n_calibration: int = 8, n_level: int = 8, steps: int = 240,
                             levels: tuple[float, ...] = LEVELS, seed: int = 0, lr: float = 3e-3,
                             n_ring: int = 28, n_station: int = 13, member: Any = None) -> CrossFamilyMemberResult:
    """Train a PhysicsNeMo member on one family, walk towards the other, and measure what the signals did.

    ``member`` is any object with ``fit(cases, steps=...)``, ``predict(case)`` and ``embedding(case)`` -- a
    :class:`~pinneapple_data.ashton.physicsnemo_models.DoMINOMember` by default, and the place an X-MeshGraphNet
    wrapper would be passed once its wheels are installed. The fixture is written once under ``root`` and read back
    through the reader, so every number is measured on a file.
    """
    from pinneapple_physics.ensemble import PhysicsEnsemble, Expert

    base = Path(root) / f"pnemo_train_{train_family}"
    if not (base / "run_1").is_dir():
        write_fixture(base, family=train_family, n_cases=n_train + n_calibration, dataset="ahmedml", seed=seed,
                      n_ring=n_ring, n_station=n_station)
    runs = list(range(1, n_train + n_calibration + 1))
    cases = _collect(base, runs)
    train, calibration = cases[:n_train], cases[n_train:]

    member = DoMINOMember(seed=seed) if member is None else member
    member.fit(train, steps=steps, lr=lr)
    if hasattr(member, "trained_on"):
        member.trained_on = f"family {train_family} of the ashton fixture, {n_train} cases"

    # -- the second expert: a surrogate that predicts nothing, which the ensemble has to beat ------------
    from .benchmark import constant_field_baseline

    constant = constant_field_baseline(train)
    experts = [member.expert("domino"),
               Expert("constant field", lambda case: constant(case), kind="baseline",
                      provenance={"source": "area-weighted mean Cp of the training cases",
                                  "licence": {"spdx": "CC-BY-SA-4.0"}})]
    residual_of = SurfaceResidual()
    ensemble = PhysicsEnsemble(experts, mode="select", interval_level=0.9, explore_every=1)

    # -- the in-family calibration: error, residual floor, embedding threshold ---------------------------
    in_family, in_residual, in_cd, coverage_in = [], [], [], []
    for case in calibration:
        prediction = member.predict(case)
        _name, truth, _location = case.field_of("cp")
        error, cd_error = _field_error(member, case, prediction)
        in_family.append(error)
        in_residual.append(residual_of(prediction, case))
        if np.isfinite(cd_error):
            in_cd.append(cd_error)
        # the ensemble's own interval, on the same case -- counted only once the interval is calibrated
        out = ensemble.predict(case)
        ensemble.update(case, truth)
        if np.all(np.isfinite(out["lower"])) and np.all(np.isfinite(out["upper"])):
            coverage_in.append(float(np.mean((truth >= out["lower"]) & (truth <= out["upper"]))))

    distances_cal = embedding_distances(np.vstack([member.embedding(case) for case in train]),
                                        np.vstack([member.embedding(case) for case in calibration]),
                                        ood_level=0.01)
    threshold = float(max(np.quantile(distances_cal, 0.99), distances_cal.max())) if distances_cal.size else \
        float("nan")
    false_alarm = float(np.mean(distances_cal > threshold)) if distances_cal.size else float("nan")
    residual_cal = np.asarray(in_residual, dtype=np.float64)
    residual_threshold = float(max(np.quantile(residual_cal, 0.99), residual_cal.max())) if residual_cal.size else \
        float("nan")
    residual_false_alarm = float(np.mean(residual_cal > residual_threshold)) if residual_cal.size else float("nan")
    floor = float(np.median(residual_cal)) if residual_cal.size else float("nan")
    error_in_family = float(np.mean(in_family))

    # -- the control: what the same residual says about a member with no training at all -----------------
    untrained = DoMINOMember(seed=seed + 1)
    control = [residual_of(untrained.predict(case), case) for case in calibration]

    result = CrossFamilyMemberResult(
        trained_on=train_family, towards=towards, member=member.model_name if hasattr(member, "model_name") else
        type(member).__name__, n_train=len(train), n_calibration=len(calibration), steps=int(getattr(member, "steps",
                                                                                                  steps)),
        embedding_threshold=threshold, false_alarm_rate=false_alarm, residual_threshold=residual_threshold,
        residual_false_alarm_rate=residual_false_alarm, residual_in_family=floor,
        residual_untrained=float(np.median(control)) if control else float("nan"),
        error_in_family=error_in_family,
        coverage_in_family=float(np.mean(coverage_in)) if coverage_in else float("nan"))
    result.notes.append(f"in-family (held-out, {len(calibration)} cases): Cp L2 {error_in_family:.4f}, Cd error "
                        f"{np.mean(in_cd) if in_cd else float('nan'):.4f}, admissibility residual {floor:.4f} "
                        f"against {result.residual_untrained:.3f} for the untrained control")
    result.notes.append(f"the error threshold is twice the in-family Cp L2 ({2.0 * error_in_family:.4f}); both "
                        f"alarms are half a level's cases past the 99 % calibration of their own signal "
                        f"({threshold:.2f} for the embedding, {residual_threshold:.4f} for the residual)")

    # -- the graded shift ------------------------------------------------------------------------------
    for t in levels:
        folder = Path(root) / f"pnemo_shift_{train_family}_to_{towards}_t{t:.2f}"
        if not (folder / "run_101").is_dir():
            box = _blend(train_family, towards, t)
            rng = np.random.default_rng(seed + int(round(t * 100)))
            for index in range(n_level):
                params = Params(**{name: float(rng.uniform(low, high)) for name, (low, high) in box.items()})
                write_case(folder, 101 + index, params, dataset="ahmedml", n_ring=n_ring, n_station=n_station)
        level_cases = _collect(folder, list(range(101, 101 + n_level)))
        errors, cd_errors, residuals, coverages, ensemble_errors, embeddings = [], [], [], [], [], []
        active = ""
        for case in level_cases:
            prediction = member.predict(case)
            _name, truth, _location = case.field_of("cp")
            error, cd_error = _field_error(member, case, prediction)
            errors.append(error)
            residuals.append(residual_of(prediction, case))
            embeddings.append(member.embedding(case))
            if np.isfinite(cd_error):
                cd_errors.append(cd_error)
            out = ensemble.predict(case)
            info = ensemble.update(case, truth)
            active = str(out["active"])
            if np.all(np.isfinite(out["lower"])) and np.all(np.isfinite(out["upper"])):
                coverages.append(float(np.mean((truth >= out["lower"]) & (truth <= out["upper"]))))
            ensemble_errors.append(float(info.get("ensemble_error", np.nan)))
        distances = embedding_distances(np.vstack([member.embedding(case) for case in train]),
                                        np.vstack(embeddings), ood_level=0.01)
        result.levels.append(MemberLevel(
            t=float(t), error=float(np.mean(errors)),
            cd_error=float(np.mean(cd_errors)) if cd_errors else float("nan"),
            embedding_distance=float(np.median(distances)) if distances.size else float("nan"),
            flagged_fraction=float(np.mean(distances > threshold)) if distances.size else float("nan"),
            residual=float(np.median(residuals)),
            residual_ratio=float(np.median(residuals) / floor) if floor and np.isfinite(floor) else float("nan"),
            residual_flagged_fraction=float(np.mean(np.asarray(residuals) > residual_threshold))
            if np.isfinite(residual_threshold) else float("nan"),
            coverage=float(np.mean(coverages)) if coverages else float("nan"),
            ensemble_error=float(np.nanmean(ensemble_errors)) if ensemble_errors else float("nan"),
            active=active, embedding=[float(v) for v in np.median(np.vstack(embeddings), axis=0)],
            n_cases=len(level_cases)))

    # -- the crossings, and the verdict on each signal --------------------------------------------------
    for level in result.levels:
        if result.t_error is None and level.error > 2.0 * error_in_family:
            result.t_error = level.t
        if result.t_flag is None and level.flagged_fraction >= 0.5:
            result.t_flag = level.t
        if result.t_residual is None and level.residual_flagged_fraction >= 0.5:
            result.t_residual = level.t
    return result


def run_on_fixtures(root: str | Path, **kwargs: Any) -> list[CrossFamilyMemberResult]:
    """Both directions: ``alpha -> beta`` (trained on the rounded family) and ``beta -> alpha``."""
    shared = {key: value for key, value in kwargs.items() if key not in ("train_family", "towards")}
    return [cross_family_with_domino(root, train_family="alpha", towards="beta", **shared),
            cross_family_with_domino(root, train_family="beta", towards="alpha", **shared)]


def member_availability() -> dict[str, Any]:
    """Which members this machine can offer the experiment, with the reason for the ones it cannot."""
    facts = availability()
    return {"domino": bool(facts["domino"]), "x_meshgraphnet": bool(facts["x_meshgraphnet"]),
            "reasons": dict(facts["reasons"])}
