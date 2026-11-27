"""pinneapple_analysis.trust.producers — the same record, emitted by three kinds of prediction (#345).

The issue's "done when" is one sentence: the weather, ensemble and one CFD surrogate example all emit the same
``Trust`` JSON. "The same" is a property of the code, not of three look-alikes, so all three producers here end on
the same call — :func:`pinneapple_analysis.trust.adapters.from_ensemble` — with ``kind`` saying which of the three a
prediction is. Nothing else differs: the keys, the gates, the sources and the JSON schema are the adapters', and the
adapters are one module.

=================  ==============================================  ==================  =============================
producer           the prediction                                  ``kind``            what measures it
=================  ==============================================  ==================  =============================
``weather_trust``  a testbed forecast on the ERA5 32 x 64 grid      ``"weather"``       ``weather.evaluate.horizons``
(``forecast_                         (:mod:`pinneapple_analysis.trust.                   (ACC 0.6, climatology RMSE)
skill``)                             forecast_skill``)
``ensemble_trust``  two experts combined online                    ``"ensemble"``      ``PhysicsEnsemble``'s residual
                   (here)
``cfd_surrogate_                   a POD surrogate over steady      ``"surrogate"``     the momentum residual of the
trust``            Navier-Stokes (Kovasznay), asked at a           field the surrogate emits
                   parameter it was fitted for and at one it
                   was not
=================  ==============================================  ==================  =============================

Why the CFD surrogate is a POD surrogate over an analytic case
--------------------------------------------------------------
The repository's CFD surrogate is :class:`pinneapple_simulation.cfd_validation.warm_start.OperatorSurrogate` — a
proper orthogonal decomposition of solved fields with the coefficients interpolated in the parameter. Its snapshots
come from OpenFOAM runs, which a test and a docs build cannot make. The case used here, Kovasznay flow
(:func:`pinneapple_data.datasets.physics._load_kovasznay_ns`, registered as ``kovasznay_ns``), is the steady
incompressible Navier-Stokes flow with an exact solution that the same module already ships for the CFD benchmarks,
so the surrogate is fitted from **solver fields that are right by construction** and the floor of every residual below
is the reference solution's own, measured on the same grid with the same finite differences — not a tolerance anyone
declared.

Two numbers are reported and they are different numbers:

* the residual of the surrogate's field — the momentum equation, made dimensionless by the size of the terms that
  have to balance (advection, pressure gradient, diffusion), evaluated in the interior;
* the in-domain level it is judged against — the median of the same residual over the parameters the surrogate was
  fitted for (:func:`pinneapple_physics.ensemble.residual_baseline`). A residual with no such level has no pass mark
  and :meth:`Trust.gates` says so.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from pinneapple_core.field import Field
from pinneapple_physics.ensemble import Expert, PhysicsEnsemble, from_callable, residual_baseline

from .adapters import from_ensemble
from .record import Trust
from .trust_gate import TrustGate

__all__ = ["GRID", "KOVASZNAY_FIT", "kovasznay_case", "navier_stokes_residual", "cfd_surrogate_trust",
           "ensemble_trust", "three_records", "emit"]

#: The grid every field of the CFD surrogate lives on: 48 x 48 over ``[-0.5, 1] x [-0.5, 1.5]``.
GRID = 48
#: The Reynolds numbers the surrogate is fitted for. The case is asked about one it has not seen (70, between two
#: fitted values) and about one outside the envelope (200), which is the pair a trust record has to tell apart.
KOVASZNAY_FIT: tuple[float, ...] = (20.0, 40.0, 60.0, 80.0, 100.0, 120.0)


# ---------------------------------------------------------------------------------------- the Navier-Stokes case
def kovasznay_case(reynolds: float, n: int = GRID) -> tuple[tuple[np.ndarray, np.ndarray], dict[str, np.ndarray]]:
    """The exact steady incompressible Navier-Stokes flow of Kovasznay (1948) on an ``n x n`` grid.

    Returns the axes and ``{"U": (n, n, 2), "p": (n, n)}``. The field is not solved: it is the closed form the
    library already registers as the ``kovasznay_ns`` dataset, so it costs nothing and it is exact — which is what
    makes the residual floor below measurable instead of assumed.
    """
    import pinneapple_data.datasets.physics  # noqa: F401  (registers the datasets)
    from pinneapple_data.datasets.registry import DatasetRegistry

    d = DatasetRegistry.load("kovasznay_ns", N=int(n), Re=float(reynolds))
    return (np.asarray(d["x"], float), np.asarray(d["y"], float)), {
        "U": np.stack([np.asarray(d["u"], float), np.asarray(d["v"], float)], axis=-1),
        "p": np.asarray(d["p"], float),
    }


def navier_stokes_residual(axes: tuple[np.ndarray, np.ndarray], fields: Mapping[str, np.ndarray], reynolds: float,
                           *, pad: int = 2) -> dict[str, float]:
    """The two laws of the steady incompressible Navier-Stokes on the field, as relative residuals.

    ``momentum``: ``||u u_x + v u_y + p_x - lap(u)/Re, u v_x + v v_y + p_y - lap(v)/Re||`` divided by the norm of
    the terms that have to balance (advection, pressure gradient, diffusion) — the same normalisation
    :func:`pinneapple_analysis.trust.field_card.field_checks` uses for its PDE residual, so a residual is a fraction
    of the terms instead of a number whose size depends on the units.

    ``mass``: ``||u_x + v_y||`` divided by ``||u_x, v_y||``.

    Both are taken in the interior (``pad`` points from each edge): the derivatives come from
    :class:`pinneapple_core.field.Field`, whose grid gradient is second-order accurate with one-sided stencils at the
    boundary, and a residual that mixes the two stencils is not what the interior of the field is doing. The
    discretisation floor this leaves is visible: on the exact solution the momentum residual is 1.1e-2 on the 48 x 48
    grid (:func:`kovasznay_case`), and it falls as the square of the spacing.
    """
    x, y = axes
    U = np.asarray(fields["U"], float)
    u, v, p = U[..., 0], U[..., 1], np.asarray(fields["p"], float)
    shape = u.shape

    def d1(f: np.ndarray) -> np.ndarray:
        """``(d f/dx, d f/dy)`` of a scalar field on the grid, as two arrays of the grid's shape."""
        return Field.on_grid((x, y), f).gradient().values.reshape(*shape, 2)

    gu, gv, gp = d1(u), d1(v), d1(p)
    ux, uy, vx, vy = gu[..., 0], gu[..., 1], gv[..., 0], gv[..., 1]
    lap_u = d1(ux)[..., 0] + d1(uy)[..., 1]
    lap_v = d1(vx)[..., 0] + d1(vy)[..., 1]
    adv_u, adv_v = u * ux + v * uy, u * vx + v * vy
    res_u = adv_u + gp[..., 0] - lap_u / float(reynolds)
    res_v = adv_v + gp[..., 1] - lap_v / float(reynolds)
    mass = ux + vy

    sl = (slice(int(pad), -int(pad)), slice(int(pad), -int(pad)))
    terms = (adv_u[sl], adv_v[sl], gp[..., 0][sl], gp[..., 1][sl], (lap_u / reynolds)[sl], (lap_v / reynolds)[sl])
    momentum_scale = float(np.sqrt(np.mean(sum(t ** 2 for t in terms)))) or 1.0
    mass_scale = float(np.sqrt(np.mean(ux[sl] ** 2 + vy[sl] ** 2))) or 1.0
    return {
        "momentum": float(np.sqrt(np.mean(res_u[sl] ** 2 + res_v[sl] ** 2))) / momentum_scale,
        "mass": float(np.sqrt(np.mean(mass[sl] ** 2))) / mass_scale,
        "momentum_scale": momentum_scale,
        "mass_scale": mass_scale,
    }


# ---------------------------------------------------------------------------------------- the CFD surrogate
def _flat(fields: Mapping[str, np.ndarray]) -> np.ndarray:
    """The surrogate's field as one vector: ``u``, then ``v``, then ``p``, each in C order."""
    U = np.asarray(fields["U"], float)
    return np.concatenate([U[..., 0].ravel(), U[..., 1].ravel(), np.asarray(fields["p"], float).ravel()])


def _unflat(vector: np.ndarray, shape: tuple[int, int]) -> dict[str, np.ndarray]:
    n = int(np.prod(shape))
    a = np.asarray(vector, float).ravel()
    return {"U": np.stack([a[:n].reshape(shape), a[n:2 * n].reshape(shape)], axis=-1),
            "p": a[2 * n:].reshape(shape)}


def _pod_expert(name: str, surrogate: Any, cost: float) -> Expert:
    """One POD truncation of the CFD surrogate as an ensemble expert: the whole field of a case, as a vector."""
    def predict(query: Mapping[str, Any]) -> np.ndarray:
        return _flat(surrogate.predict(float(query["re"])))

    return from_callable(name, predict, cost=cost, kind="surrogate")


def cfd_surrogate_trust(reynolds: float = 70.0, *, n: int = GRID, modes: Sequence[int] = (3, 6),
                        fit: Sequence[float] | None = None) -> Trust:
    """The :class:`Trust` of one CFD surrogate prediction — the third producer of the issue's "done when".

    The surrogate is :class:`~pinneapple_simulation.cfd_validation.warm_start.OperatorSurrogate` fitted on the exact
    Kovasznay solutions at the Reynolds numbers in ``fit``; two truncations of it (``modes``) are the ensemble's
    experts, so the interval is their disagreement and the residual is the momentum equation on the field the
    combination emits. Both are the ensemble's own adapters: this function fits, runs and asks
    :func:`~pinneapple_analysis.trust.adapters.from_ensemble` for the record.

    The ensemble first runs over the cases it was fitted for, with their references — that is the ensemble's normal
    loop and the only place its conformal interval is calibrated — and then the case asked about is predicted and
    scored. ``reynolds = 70`` is inside the fitted envelope and between two fitted values; ``reynolds = 200`` is
    outside it, which the domain gate reports as no skill.
    """
    from pinneapple_simulation.cfd_validation.warm_start import OperatorSurrogate

    fit = tuple(float(r) for r in (KOVASZNAY_FIT if fit is None else fit))
    modes = tuple(int(m) for m in modes)
    shape = (int(n), int(n))
    axes, _ = kovasznay_case(fit[0], n)
    snapshots = {r: kovasznay_case(r, n)[1] for r in fit}

    experts: list[Expert] = []
    for m in modes:
        surrogate = OperatorSurrogate(modes=m).fit(snapshots)
        experts.append(_pod_expert(f"pod-{m}", surrogate, cost=float(m) / max(modes)))

    def score(prediction: np.ndarray, query: Mapping[str, Any]) -> float:
        """The label-free residual of the prediction: the momentum law of the flow it claims to solve."""
        return navier_stokes_residual(axes, _unflat(prediction, shape), float(query["re"]))["momentum"]

    train = [{"re": r, "n": int(n)} for r in fit]
    baseline = residual_baseline(experts, {e.name: train for e in experts}, score)
    ensemble = PhysicsEnsemble(experts, mode="combine", residual_fn=score, residual_lookahead=1.0,
                               residual_baseline=baseline, cost_weight=0.0, min_weight=0.0, interval_level=0.9)
    ensemble.run(train, [_flat(snapshots[r]) for r in fit])     # its own loop: the interval is calibrated here

    query = {"re": float(reynolds), "n": int(n)}
    _axes, truth = kovasznay_case(reynolds, n)
    gate = TrustGate().fit_from_bounds({"re": (min(fit), max(fit))})
    return from_ensemble(ensemble, query, reference=_flat(truth), gate=gate,
                         domain_points=np.array([[float(reynolds)]]), kind="surrogate")


# ---------------------------------------------------------------------------------------- the ensemble
def ensemble_trust(*, in_domain: bool = True) -> Trust:
    """The :class:`Trust` of an online ensemble of two experts — the second producer.

    The case is the one ``docs/guides/trust_as_one_object.md`` shows: two experts over one input ``u``, one of them
    exact and one 0.1 % high, combined by :class:`~pinneapple_physics.ensemble.PhysicsEnsemble` with the residual
    lookahead and an in-domain level for each expert. What it adds to the guide's snippet is the calibration: the
    ensemble runs over its own cases first, which is where its conformal interval learns, and the case asked about is
    then scored against its reference.

    ``in_domain = False`` asks the same ensemble about an input outside the box it was fitted for: the domain gate
    reports no skill, which is the second half of what a record is for.
    """
    def exact(q: Mapping[str, Any]) -> np.ndarray:
        return np.full(8, float(q["u"]))

    def calibrated_near_one(q: Mapping[str, Any]) -> np.ndarray:
        """The same field, off by a quadratic in the distance from ``u = 1``: a model trained around a nominal
        condition is exact there and loses accuracy away from it (2 % at u = 0.25, 0.5 % at u = 0.5)."""
        u = float(q["u"])
        return np.full(8, u * (1.0 + 0.02 * (u - 1.0) ** 2))

    def residual(prediction: np.ndarray, query: Mapping[str, Any]) -> float:
        """Relative deviation of the prediction from the case's own value: 0 for the exact expert."""
        u = float(query["u"])
        return float(abs(float(np.mean(prediction)) - u) / abs(u))

    experts = [from_callable("fno", exact, cost=1.0, kind="operator"),
               from_callable("gnn", calibrated_near_one, cost=0.2, kind="graph")]
    train = [{"u": u} for u in (0.25, 0.5, 1.5, 2.0)]
    baseline = residual_baseline(experts, {e.name: train for e in experts}, residual)
    ensemble = PhysicsEnsemble(experts, mode="combine", residual_fn=residual, residual_lookahead=1.0,
                               residual_baseline=baseline, cost_weight=0.0, min_weight=0.0, interval_level=0.9)
    ensemble.run(train, [exact(q) for q in train])

    query = {"u": 1.0 if in_domain else 4.0}
    gate = TrustGate().fit_from_bounds({"u": (0.25, 2.0)})
    return from_ensemble(ensemble, query, reference=exact(query), gate=gate, domain_points=np.array([[query["u"]]]),
                         kind="ensemble")


# ---------------------------------------------------------------------------------------- the three together
def three_records(*, reynolds: float | None = None) -> dict[str, Trust]:
    """The three records of the issue's "done when", produced by the three functions above.

    ``{"weather": ..., "ensemble": ..., "surrogate": ...}``. The weather record takes the minutes its testbed needs
    (twelve initialisations, twenty leads): it is the only one that measures a horizon, because it is the only one
    that forecasts in time.
    """
    from .forecast_skill import weather_trust

    return {"weather": weather_trust(), "ensemble": ensemble_trust(),
            "surrogate": cfd_surrogate_trust(70.0 if reynolds is None else float(reynolds))}


def emit(folder: Any) -> dict[str, Any]:
    """Write the three records as JSON to ``folder`` and return ``{kind: path}``.

    The files are the measurement: ``results/trust_three_kinds/`` in the repository holds them, written by
    ``scripts/trust_three_kinds.py``, and the guide prints them. Nothing here post-processes the JSON — it is
    :meth:`pinneapple_analysis.trust.record.Trust.to_json` and nothing else, which is what makes the three files
    comparable key for key.
    """
    out = Path(folder)
    out.mkdir(parents=True, exist_ok=True)
    written = {}
    for kind, record in three_records().items():
        written[kind] = record.write(out / f"{kind}.json")
    return written
