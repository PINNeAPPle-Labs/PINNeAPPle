"""Design loop of a printed end-fire antenna: DOE, multi-objective optimisation, active learning with solver
verification, multi-fidelity bias correction and a trust card per prediction.

* :func:`doe` -- Latin hypercube or Sobol samples of :data:`AVA_RANGES` (configurable), mapped by
  :func:`ava_transform` (director and slots switched on/off, the director placed as a fraction of the length) and
  filtered by :func:`.ava.layout` (unbuildable designs are skipped and counted);
* :func:`optimize` -- NSGA-II (:mod:`pinneapple_design.design_optimizer.methods`) on surrogates: maximise the
  minimum realized gain over the band, minimise the board area, subject to ``max S11 <= -10 dB`` over the band;
* :func:`active_learning` -- rounds of: fit the GP, pick candidates by predictive variance
  (:class:`pinneapple_data.active_learning.VarianceBasedAL`), simulate them, record the error of the prediction made
  *before* the simulation (prospective error, never in-sample), add them;
* :func:`multi_fidelity` -- coarse simulations (``res`` 0.25 wavelength) for many designs, fine ones (``res`` 0.2)
  for a few: the coarse-to-fine bias per frequency is learnt on the fine designs and validated leaving one fine
  design out;
* :func:`prediction_card` -- a :class:`~pinneapple_analysis.trust.TrustReport` for one surrogate prediction: inside
  the training domain, geometry buildable, uncertainty below a threshold, below the physical limit, surrogate
  out-of-sample error known.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pinneapple_analysis.trust.trust_report import (
    CONTRADICTS,
    NOT_RUN,
    SUPPORTS,
    Check,
    TrustReport,
)

from . import bounds
from .ava import AVAParams, InvalidGeometry, layout
from .runner import ResultTable
from .surrogates import GPModel, _Model

__all__ = ["AVA_RANGES", "INTEGER", "ava_transform", "is_buildable", "training_box", "doe", "optimize", "OptimizationOutcome",
           "active_learning", "ALRound", "multi_fidelity", "prediction_card", "band_metrics"]

#: default sampling ranges (mm; ``R`` 1/mm; ``dir_x`` is a fraction of ``Lb``; ``dir_a < 4`` means no director)
AVA_RANGES: dict[str, tuple[float, float]] = {
    "Lb": (150.0, 260.0), "W": (80.0, 140.0), "Wa": (40.0, 110.0), "R": (0.01, 0.06), "xs": (30.0, 60.0),
    "Wg": (25.0, 50.0), "Lo": (10.0, 25.0), "nslot": (0, 6), "Ls": (5.0, 15.0), "ws": (2.0, 4.0), "ps": (6.0, 10.0),
    "dir_a": (0.0, 20.0), "dir_b": (3.0, 10.0), "dir_x": (0.7, 0.95),
}
INTEGER = ("nslot",)


def ava_transform(sample: Mapping[str, float], *, fixed: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """A raw sample of :data:`AVA_RANGES` to AVA parameters: ``dir_x`` from a fraction of ``Lb`` to mm, director off
    when ``dir_a < 4`` mm, slot fields ignored when ``nslot = 0``; ``fixed`` overrides (e.g. ``res``)."""
    d = {k: (int(round(v)) if k in INTEGER else float(v)) for k, v in sample.items()}
    if "dir_x" in d and "Lb" in d and d["dir_x"] <= 1.0:
        d["dir_x"] = d["dir_x"] * d["Lb"]
    if d.get("dir_a", 0.0) < 4.0:
        d["dir_a"] = d["dir_b"] = d["dir_x"] = 0.0
    d.update(fixed or {})
    return d


def is_buildable(params: Mapping[str, Any]) -> bool:
    known = set(AVAParams.__dataclass_fields__)
    try:
        layout(AVAParams.from_dict({k: v for k, v in params.items() if k in known}))
    except InvalidGeometry:
        return False
    return True


def training_box(designs: Sequence[Mapping[str, Any]], keys: Sequence[str] = ("Lb", "W", "Wa"),
                 ranges: Mapping[str, tuple[float, float]] | None = None) -> dict[str, tuple[float, float]]:
    """``ranges`` with ``keys`` narrowed to the span of ``designs``: optimise where the surrogate was trained."""
    out = dict(ranges or AVA_RANGES)
    for k in keys:
        v = [float(d[k]) for d in designs if k in d]
        if len(v) > 1 and max(v) > min(v):
            out[k] = (min(v), max(v))
    return out


def doe(n: int, ranges: Mapping[str, tuple[float, float]] | None = None, *, method: str = "lhs", seed: int = 0,
        transform: Callable[[Mapping[str, float]], dict[str, Any]] | None = ava_transform,
        valid: Callable[[Mapping[str, Any]], bool] | None = is_buildable,
        max_factor: int = 30) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """``n`` valid designs from a space-filling sample; returns ``(designs, {"drawn": .., "invalid": ..})``.

    Draws in batches (``pinneapple_data.parameter_sampling.sample_parameters``) until ``n`` designs pass ``valid``
    or ``max_factor * n`` have been drawn."""
    from pinneapple_data.parameter_sampling import sample_parameters

    ranges = dict(ranges or AVA_RANGES)
    out: list[dict[str, Any]] = []
    drawn = invalid = 0
    batch = max(2 * n, 8)
    k = 0
    while len(out) < n and drawn < max_factor * n:
        for s in sample_parameters(ranges, batch, method=method, seed=seed + k):
            drawn += 1
            d = transform(s) if transform else dict(s)
            if valid is None or valid(d):
                out.append(d)
                if len(out) == n:
                    break
            else:
                invalid += 1
        k += 1
    return out, {"drawn": drawn, "invalid": invalid}


def band_metrics(models: Mapping[str, _Model], params: Sequence[Mapping[str, Any]], freqs: Sequence[float],
                 *, gain: str = "Greal_bore_dBi", s11: str = "S11_dB") -> dict[str, np.ndarray]:
    """Predicted minimum gain, its std at the arg-min, and maximum S11 over ``freqs`` for each design."""
    G = np.stack([models[gain].predict(params, f)[0] for f in freqs], 1)
    Gs = np.stack([models[gain].predict(params, f)[1] for f in freqs], 1)
    S = np.stack([models[s11].predict(params, f)[0] for f in freqs], 1) if s11 in models else np.full_like(G, -99)
    j = G.argmin(1)
    return {"Gmin": G.min(1), "Gmin_std": Gs[np.arange(len(G)), j], "S11max": S.max(1)}


@dataclass
class OptimizationOutcome:
    designs: list[dict[str, Any]]            # Pareto designs (transformed), best gain first
    Gmin: np.ndarray
    S11max: np.ndarray
    area_cm2: np.ndarray
    n_evals: int
    wall_s: float
    method: str
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"method": self.method, "n_evals": self.n_evals, "wall_s": round(self.wall_s, 2),
                "pareto": [d | {"Gmin_dBi": float(g), "S11max_dB": float(s), "area_cm2": float(a)}
                           for d, g, s, a in zip(self.designs, self.Gmin, self.S11max, self.area_cm2, strict=True)],
                "notes": self.notes}


def optimize(models: Mapping[str, _Model], freqs: Sequence[float], *,
             ranges: Mapping[str, tuple[float, float]] | None = None, fixed: Mapping[str, Any] | None = None,
             budget: int = 600, seed: int = 0, method: str = "nsga2", s11_max: float = -10.0,
             gain: str = "Greal_bore_dBi", s11: str = "S11_dB") -> OptimizationOutcome:
    """Multi-objective search on the surrogates ``models[gain]`` and ``models[s11]`` (fit beforehand).

    Objectives: maximise the minimum predicted gain over ``freqs``, minimise ``Lb * W``; constraint: the maximum
    predicted S11 over ``freqs`` at most ``s11_max``. Unbuildable designs are marked invalid (never on the front)."""
    from pinneapple_design.design_optimizer.methods import run
    from pinneapple_design.design_optimizer.study import (
        VALID,
        Constraint,
        DesignProblem,
        DesignSpace,
        FunctionEvaluator,
        Objective,
        Variable,
    )

    ranges = dict(ranges or AVA_RANGES)
    space = DesignSpace([Variable(k, float(lo), float(hi), "integer" if k in INTEGER else "continuous")
                         for k, (lo, hi) in ranges.items()])

    def fn(X):
        ds = [ava_transform(r, fixed=fixed) for r in space.as_dicts(X)]
        ok = np.array([is_buildable(d) for d in ds])
        bm = band_metrics(models, ds, freqs, gain=gain, s11=s11)
        area = np.array([d["Lb"] * d["W"] / 100.0 for d in ds])
        return {"Gmin": bm["Gmin"], "S11max": bm["S11max"], "area": area, VALID: ok}

    ev = FunctionEvaluator(fn, ["Gmin", "S11max", "area", VALID], name="AVA surrogates", fidelity="surrogate")
    prob = DesignProblem("ava_band", space, [Objective("Gmin", "max", "dBi"), Objective("area", "min", "cm2")],
                         [Constraint("S11max", upper=s11_max, unit="dB")], ev)
    res = run(prob, method, budget=budget, seed=seed)
    E = res.evaluations
    idx = list(res.pareto())
    if E is None or not idx:
        return OptimizationOutcome([], np.array([]), np.array([]), np.array([]), res.n_evals, res.wall_s, method,
                                   [res.message or "no feasible design found"] + res.notes)
    idx.sort(key=lambda i: -E.Y["Gmin"][i])
    designs = [ava_transform(space.as_dicts(E.X[i])[0], fixed=fixed) for i in idx]
    return OptimizationOutcome(designs, E.Y["Gmin"][idx], E.Y["S11max"][idx], E.Y["area"][idx], res.n_evals,
                               res.wall_s, method, res.notes)


# ---------------------------------------------------------------------------------------------- active learning
@dataclass
class ALRound:
    round: int
    n_train_designs: int
    new_tags: list[str]
    rmse_prospective: float          # error of the predictions made before simulating the new designs
    mean_std: float                  # mean predicted std of those predictions
    coverage95: float


def active_learning(simulate: Callable[[list[dict[str, Any]], int], ResultTable], table: ResultTable,
                    freqs: Sequence[float], *, target: str = "Greal_bore_dBi", rounds: int = 2, per_round: int = 2,
                    ranges: Mapping[str, tuple[float, float]] | None = None, fixed: Mapping[str, Any] | None = None,
                    n_candidates: int = 400, seed: int = 0,
                    model: Callable[[], _Model] = GPModel) -> tuple[ResultTable, list[ALRound]]:
    """Variance-driven active learning with verification by the simulator.

    ``simulate(designs, round) -> ResultTable`` runs the solver (e.g. :class:`.runner.CLISimulator`). Each round
    fits ``model`` on ``table``, selects ``per_round`` buildable candidates with
    :class:`pinneapple_data.active_learning.VarianceBasedAL` (probability proportional to the mean predictive
    variance over ``freqs``), predicts them, simulates them, records the prospective error and appends them."""
    from pinneapple_data.active_learning import ActiveLearningConfig, VarianceBasedAL

    ranges = dict(ranges or AVA_RANGES)
    names = list(ranges)
    al = VarianceBasedAL(ActiveLearningConfig(n_candidates=n_candidates, n_select=per_round, seed=seed),
                         {k: ranges[k] for k in names})
    history: list[ALRound] = []
    for r in range(rounds):
        m = model().fit(table, target)

        def to_designs(C):
            return [ava_transform(dict(zip(names, c, strict=True)), fixed=fixed) for c in np.asarray(C, dtype=float)]

        def variance(C, m=m):
            ds = to_designs(C)
            ok = np.array([is_buildable(d) for d in ds])
            v = np.mean([m.predict(ds, f)[1] ** 2 for f in freqs], 0)
            return np.where(ok, v, 0.0)

        picked: list[dict[str, Any]] = []
        for _ in range(10):
            for d in to_designs(al.select(variance, per_round)):
                if is_buildable(d) and d not in picked:
                    picked.append(d)
            if len(picked) >= per_round:
                break
        picked = picked[:per_round]
        if not picked:
            break
        pred = {i: [m.predict([d], f) for f in freqs] for i, d in enumerate(picked)}
        new = simulate(picked, r)
        errs, stds, inside = [], [], []
        for row in new.rows:
            y = float(row.get(target, np.nan))
            i = next((i for i, d in enumerate(picked) if all(np.isclose(float(row.get(k, np.nan)), float(v))
                                                              for k, v in d.items() if k in row)), None)
            if i is None or not np.isfinite(y):
                continue
            j = int(np.argmin([abs(f - float(row["f_GHz"])) for f in freqs]))
            mu, sd = float(pred[i][j][0][0]), float(pred[i][j][1][0])
            errs.append(mu - y)
            stds.append(sd)
            inside.append(abs(mu - y) <= 1.96 * sd)
        history.append(ALRound(r, len(table.tags), new.tags,
                               float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan"),
                               float(np.mean(stds)) if stds else float("nan"),
                               float(np.mean(inside)) if inside else float("nan")))
        table = table + new
    return table, history


# ---------------------------------------------------------------------------------------------- multi-fidelity
def multi_fidelity(coarse: ResultTable, fine: ResultTable, target: str) -> dict[str, Any]:
    """Coarse-to-fine bias of ``target`` per frequency, validated leaving one fine design out.

    Pairs (tag, f) present in both tables; returns the mean bias per frequency (fine - coarse), the RMSE of the
    raw coarse values against the fine ones, and the RMSE of coarse + bias learnt without the held-out design."""
    key = lambda r: (str(r.get("tag")), round(float(r["f_GHz"]), 6))  # noqa: E731
    c = {key(r): float(r.get(target, np.nan)) for r in coarse.rows}
    pairs = [(key(r), c[key(r)], float(r.get(target, np.nan))) for r in fine.rows if key(r) in c]
    pairs = [p for p in pairs if np.isfinite(p[1]) and np.isfinite(p[2])]
    if not pairs:
        return {"n_pairs": 0, "note": "no (design, frequency) simulated at both fidelities"}
    tags = sorted({p[0][0] for p in pairs})
    freqs = sorted({p[0][1] for p in pairs})
    bias = {f: float(np.mean([p[2] - p[1] for p in pairs if p[0][1] == f])) for f in freqs}
    raw = np.array([p[1] - p[2] for p in pairs])
    corr = []
    for t in tags:
        rest = [p for p in pairs if p[0][0] != t]
        for p in (p for p in pairs if p[0][0] == t):
            b = [q[2] - q[1] for q in rest if q[0][1] == p[0][1]]
            if b:
                corr.append(p[1] + float(np.mean(b)) - p[2])
    return {"n_pairs": len(pairs), "fine_designs": tags, "bias_per_freq": bias,
            "rmse_coarse_vs_fine": float(np.sqrt(np.mean(raw ** 2))),
            "rmse_corrected_lofo": float(np.sqrt(np.mean(np.square(corr)))) if corr else float("nan")}


# ---------------------------------------------------------------------------------------------- trust card
def prediction_card(model: _Model, params: Mapping[str, Any], f_ghz: float, train: ResultTable, *,
                    target: str = "Greal_bore_dBi", std_max_db: float = 1.0, lodo_rmse: float | None = None,
                    rmse_max_db: float = 1.5, margin_db: float = 0.5) -> TrustReport:
    """Trust card of one surrogate prediction (mean, std in ``details``).

    Checks: ``in_domain`` (every physical feature of :func:`.surrogates.physical_features` inside the range of the
    training rows; critical), ``buildable``
    (critical), ``uncertainty`` (std <= ``std_max_db``), ``below_physical_limit`` (gain or directivity below the
    Hansen-Woodyard limit + ``margin_db``), ``surrogate_error_known`` (out-of-sample RMSE given and below
    ``rmse_max_db``)."""
    from .surrogates import FEATURES, physical_features

    mu, sd = (float(v[0]) for v in model.predict([params], f_ghz))
    # the surrogate sees the design only through its physical features: the domain is where those were trained
    rows = [r for r in train.rows if np.isfinite(float(r.get("f_GHz", np.nan)))]
    F = physical_features(rows, [float(r["f_GHz"]) for r in rows])
    q = physical_features([params], f_ghz)[0]
    outside = [f"{n}={v:.3g} not in [{F[:, i].min():.3g}, {F[:, i].max():.3g}]"
               for i, (n, v) in enumerate(zip(FEATURES, q, strict=True)) if not F[:, i].min() - 1e-9 <= v <= F[:, i].max() + 1e-9]
    names = FEATURES
    checks = [Check("in_domain", CONTRADICTS if outside else SUPPORTS, critical=True,
                    reason="; ".join(outside) or f"{len(names)} features inside the training range"),
              Check("buildable", SUPPORTS if is_buildable(params) else CONTRADICTS, critical=True),
              Check("uncertainty", SUPPORTS if sd <= std_max_db else CONTRADICTS, value=sd, limit=std_max_db,
                    reason=f"predictive std {sd:.2f} dB")]
    if "dBi" in target and "Lb" in params:
        lim = float(bounds.endfire_directivity_dbi(float(params["Lb"]) * 1e-3, f_ghz * 1e9)) + margin_db
        checks.append(Check("below_physical_limit", SUPPORTS if mu <= lim else CONTRADICTS, value=mu, limit=lim,
                            source="Hansen-Woodyard end-fire limit"))
    else:
        checks.append(Check("below_physical_limit", NOT_RUN, reason=f"no physical limit for {target}"))
    if lodo_rmse is None:
        checks.append(Check("surrogate_error_known", NOT_RUN, reason="no out-of-sample (LODO) error given"))
    else:
        checks.append(Check("surrogate_error_known", SUPPORTS if lodo_rmse <= rmse_max_db else CONTRADICTS,
                            value=float(lodo_rmse), limit=rmse_max_db, reason="leave-one-design-out RMSE"))
    return TrustReport(checks, model=getattr(model, "name", "surrogate"),
                       details={"target": target, "f_GHz": f_ghz, "mean": mu, "std": sd})
