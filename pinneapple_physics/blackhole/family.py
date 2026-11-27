"""The model family of the black-hole weather forecast, and its ensemble (issue #399, plan item 4).

Item 4 asks for the CNN/U-Net of the paper, a Fourier/spherical neural operator and the SphereUNet-style backbone of
:mod:`pinneapple_physics.weather`, combined in :class:`pinneapple_physics.ensemble.PhysicsEnsemble`, and for the
measured answer to whether that ensemble beats the best single model over the horizon.

Members
-------
Every member obeys the forecast contract of :mod:`pinneapple_physics.blackhole.forecast`: a batch of ``k = 5``
consecutive normalised log-density frames in, the next ``k`` frames out, same shape, same normalisation. That is what
makes them comparable and what lets all of them go through ``rollout`` and ``skill.evaluate_skill`` unchanged.

``duarte-unet``        :class:`~pinneapple_physics.blackhole.forecast.DuarteUNet`, the paper's U-Net, residual
                       (``residual=True``: it predicts the change from the last input frame).
``fourier-operator``   :class:`FourierForecaster`: the repo's ``FNO2d`` (Fourier neural operator, four spectral
                       layers on the low modes of the r-theta plane), also predicting the change from the last frame.
``sphere-backbone``    :class:`SphereBackbone`: the ``SphereUNet`` of :mod:`pinneapple_physics.weather`, with its
                       residual group-norm blocks, pixel-unshuffle patch embedding and full-resolution refinement.

Two adaptations are named rather than hidden. The weather backbone's padding (:func:`sphere_pad`) is periodic in
longitude and wraps across the poles; the accretion grid is not a sphere, it is one meridional half-plane, so
:func:`grid_pad` pads r with zeros (open boundary) and theta by replication (the flow is reflecting at the poles).
The backbone keeps ``n_state = k``: the block of frames is the state, and the previous state it is differenced
against is the same block shifted by one frame. Neither changes a layer, and neither is a wrapper around something
that is not that model.

Ensemble
--------
* ``equal``: the arithmetic mean of the members' fields, no learning. The plain answer to "combine them".
* ``adaptive``: :class:`~pinneapple_physics.ensemble.PhysicsEnsemble` in ``combine`` mode over the same members,
  weights from the fixed-share / AdaHedge grid, learned **prequentially** (case k is predicted with the weights the
  labels of cases before k produced), with the mass rate of the rollout as the label-free residual tilt
  (``residual_lookahead``, the same idea as the ``MassEnvelope`` trust check).

The aggregation of ``equal`` is the convex combination ``sum_i w_i P_i`` with ``w_i = 1/n``; for identical
predictions it is the identity (``combine([P, P], [w, 1-w]) == P`` exactly), which the tests check.
"""
from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pinneapple_analysis.verification.forecast_skill import SkillResult, skill_vs_lead
from pinneapple_physics.ensemble import Expert, PhysicsEnsemble, residual_baseline
from pinneapple_physics.weather.model import SphereUNet

from .forecast import DensityCodec, MassEnvelope, duarte_loss, rollout, window_mass
from .skill import evaluate_skill
from .skill import window_mass as normalised_mass

__all__ = ["MEMBERS", "DEFAULT_LR", "DEFAULT_EPOCHS", "FourierForecaster", "SphereBackbone", "grid_pad",
           "make_member", "member_names", "EqualWeightEnsemble", "combine", "mass_rate_residual",
           "adaptive_ensemble", "forecast_queries", "per_lead_error", "skill_against", "train_member",
           "evaluate_family"]

K = 5                                     # frames in, frames out (the contract, as in the paper)
MEMBERS = ("duarte-unet", "fourier-operator", "sphere-backbone")
#: Default learning rate per member: the deep U-Net of the paper diverges at the rate the shallower members take (its
#: own training config in ``forecast.TrainConfig`` uses 2e-4 for the same reason), so the family does not handicap it.
DEFAULT_LR = {"duarte-unet": 2e-4, "fourier-operator": 1e-3, "sphere-backbone": 1e-3}
#: Default epochs: an epoch of the U-Net costs about fifty times an epoch of the other two on the same blocks, so the
#: members get the training their size can use in the same wall-clock budget -- not the same number of passes.
DEFAULT_EPOCHS = {"duarte-unet": 12, "fourier-operator": 30, "sphere-backbone": 30}


# ---------------------------------------------------------------------- the members
def grid_pad(x: torch.Tensor, p: int) -> torch.Tensor:
    """Padding of a stencil on the accretion grid ``(B, C, r, theta)``: zeros in r (open boundary), replication in
    theta (the meridional plane ends at the poles, where the flow is reflecting and nothing crosses).

    The counterpart of :func:`pinneapple_physics.weather.model.sphere_pad`, which is the right padding for a
    latitude-longitude grid and the wrong one here: the accretion grid has no periodic axis."""
    if p == 0:
        return x
    x = F.pad(x, (p, p, 0, 0), mode="replicate")               # theta (last axis)
    return F.pad(x, (0, 0, p, p), mode="constant", value=0.0)  # r


class FourierForecaster(nn.Module):
    """The repo's ``FNO2d`` on the forecast contract: 5 frame channels in, the change of the same 5 out.

    The Fourier neural operator of Li et al. (2021) learns a global convolution in the low modes of the plane, which
    is the natural counterpart of the local kernel of the U-Net: the accretion flow is dominated by axisymmetric
    structure plus spiral waves, and a handful of modes carries the former while the local path of the FNO carries
    the latter. ``padding`` keeps the two open edges of the window from interacting through the periodic transform."""

    def __init__(self, frames: int = K, width: int = 32, modes=(12, 8), layers: int = 4, padding=(2, 2)):
        super().__init__()
        from pinneapple_neural.architectures.neural_operators.fno import FNO2d
        self.fno = FNO2d(frames, frames, width=width, modes1=int(modes[0]), modes2=int(modes[1]), layers=layers,
                         use_grid=True, padding=tuple(int(p) for p in padding))

    def forward(self, x):
        return x[:, -1:] + self.fno(x).y


class SphereBackbone(nn.Module):
    """The ``SphereUNet`` of :mod:`pinneapple_physics.weather` on the forecast contract.

    The weather model maps the state at t and t - dt to the change; here the state is the block of ``frames``
    snapshots and the previous state is the same block shifted by one frame, so the difference channel carries the
    frame-to-frame tendency and the output is the block of the next ``frames`` snapshots. The two normalised
    coordinates (r, theta) go in as extra channels, the way the weather model is given latitude, longitude and the
    forcing. Its output layer is initialised to zero, so an untrained backbone is exactly persistence."""

    def __init__(self, frames: int = K, widths=(16, 24, 32, 48), blocks: int = 1, patch: int = 1, refine: int = 16,
                 n_extra: int = 2):
        super().__init__()
        self.frame_count, self.n_extra = frames, n_extra
        self.net = SphereUNet(frames, n_extra, tuple(int(w) for w in widths), blocks=blocks, patch=patch,
                              refine=refine, pad=grid_pad)
        self._coords: torch.Tensor | None = None

    def coords(self, x: torch.Tensor) -> torch.Tensor:
        """``(B, n_extra, r, theta)`` normalised coordinates of the window (constant, so they are built once)."""
        if self._coords is None or self._coords.shape[-2:] != x.shape[-2:]:
            nr, nt = x.shape[-2], x.shape[-1]
            r = torch.linspace(0.0, 1.0, nr)
            t = torch.linspace(0.0, 1.0, nt)
            R, T = torch.meshgrid(r, t, indexing="ij")
            self._coords = torch.stack([R, T])[: self.n_extra]
        return self._coords[None].expand(x.shape[0], -1, -1, -1)

    def forward(self, x):
        prev = torch.cat([x[:, :1], x[:, :-1]], 1)              # the same block, shifted by one frame
        return self.net(x, prev, self.coords(x))


def make_member(name: str, **kwargs) -> nn.Module:
    """Build one member of the family by name, with its defaults (see :data:`MEMBERS`)."""
    if name == "duarte-unet":
        from .forecast import DuarteUNet
        kw = {"filters": 16, "frames": K, "residual": True}
        kw.update(kwargs)
        return DuarteUNet(**kw)
    if name == "fourier-operator":
        return FourierForecaster(**{"frames": K, **kwargs})
    if name == "sphere-backbone":
        return SphereBackbone(**{"frames": K, **kwargs})
    raise ValueError(f"unknown member {name!r}; the family is {MEMBERS}")


def member_names(members: Mapping[str, Any]) -> list[str]:
    return list(members)


# ---------------------------------------------------------------------- the aggregation
def combine(predictions: Sequence[np.ndarray], weights: Sequence[float]) -> np.ndarray:
    """Convex combination of the members' fields: ``sum_i w_i P_i`` with ``w`` normalised to sum 1.

    This is the aggregation of both ensembles here (equal weights for the ``equal`` one, learned ones for the
    ``adaptive`` one). It is affine in the inputs, so identical predictions combine back to themselves exactly."""
    w = np.asarray(weights, float)
    if w.shape != (len(predictions),):
        raise ValueError(f"{len(predictions)} predictions need {len(predictions)} weights, got {w.shape}")
    if w.sum() <= 0:
        raise ValueError("weights must sum to a positive number")
    w = w / w.sum()
    return np.tensordot(w, np.stack([np.asarray(p, float) for p in predictions]), axes=1)


class EqualWeightEnsemble(nn.Module):
    """The ``equal`` aggregation as a module, so that it is scored by the same ``evaluate_skill`` as the members."""

    def __init__(self, members: Mapping[str, nn.Module]):
        super().__init__()
        if len(members) < 2:
            raise ValueError("an ensemble needs at least two members")
        self.members = nn.ModuleDict(dict(members))
        self.names = list(members)

    def forward(self, x):
        return torch.stack([self.members[n](x) for n in self.names], 0).mean(0)


# ---------------------------------------------------------------------- label-free residual
def mass_rate_residual(frames: np.ndarray, codec: DensityCodec, cell_volume: np.ndarray, frame_dt: float) -> float:
    """Mean |d ln M / dt| of the window mass over a forecast, in units of the training bound.

    1.0 means the forecast moves the gas in and out of the window as fast as the training simulations ever did; much
    more than that is the artificial mass injection the original authors identify as the cause of drift. It needs no
    ground truth, which is what ``PhysicsEnsemble(residual_lookahead=...)`` asks of its tilt."""
    m = window_mass(codec.decode(frames), cell_volume)
    if len(m) < 2 or not np.all(np.isfinite(m)) or np.any(m <= 0):
        return float("inf")
    return float(np.mean(np.abs(np.diff(np.log(m))) / frame_dt))


# ---------------------------------------------------------------------- the adaptive ensemble
def forecast_queries(x: np.ndarray, blocks: int, starts: Sequence[int] | None = None, start_every: int = 5) -> list[dict]:
    """One query per start time of a trajectory ``x`` (T, H, W): the ``K`` frames the member is given and the number
    of blocks it has to forecast. The same start times ``skill.evaluate_skill`` uses, so the numbers are pairable."""
    n_lead = blocks * K
    starts = list(range(K - 1, len(x) - n_lead, start_every)) if starts is None else [int(s) for s in starts]
    return [{"x0": x[s - K + 1:s + 1], "start": int(s), "blocks": int(blocks)} for s in starts]


def adaptive_ensemble(members: Mapping[str, nn.Module], codec: DensityCodec, cell_volume: np.ndarray,
                      frame_dt: float, envelope: MassEnvelope, *, residual_lookahead: float = 1.0,
                      baseline_queries: Mapping[str, Sequence[dict]] | None = None,
                      min_weight: float = 0.0, log=None) -> PhysicsEnsemble:
    """``PhysicsEnsemble`` over the family: each member forecasts the whole rollout, the combination is the
    weighted average of the fields, and the tilt before each case is the mass-rate residual of that member's own
    forecast (label-free, ``lookahead_reference="self"``).

    ``baseline_queries`` ({member: queries}): the in-domain residual level of each member, measured on the
    validation trajectory, so that the lookahead compares a case with a level that case did not contribute to."""
    rate_max = envelope.rate_max

    def res(pred: np.ndarray, query: dict) -> float:
        return mass_rate_residual(np.asarray(pred), codec, cell_volume, frame_dt) / max(rate_max, 1e-300)

    experts: list[Expert] = []
    for name, model in members.items():
        experts.append(Expert(name, _member_predictor(model), cost=None, kind="neural"))
    base = None
    if baseline_queries:
        base = residual_baseline(experts, baseline_queries, res)
    return PhysicsEnsemble(experts, mode="combine", residual_fn=res, residual_lookahead=residual_lookahead,
                           lookahead_reference="self", residual_baseline=base, min_weight=min_weight,
                           interval_level=0.9)


def _member_predictor(model: nn.Module):
    def predict(q: dict) -> np.ndarray:
        return rollout(model, q["x0"], int(q["blocks"]))
    return predict


def per_lead_error(predictions: Sequence[np.ndarray], references: Sequence[np.ndarray]) -> np.ndarray:
    """MAE per lead of every forecast: ``(n_cases, n_leads)``, the error array ``skill_vs_lead`` pairs."""
    return np.array([np.abs(np.asarray(p, float) - np.asarray(r, float)).mean((1, 2))
                     for p, r in zip(predictions, references, strict=True)])


def skill_against(err_model: np.ndarray, err_reference: np.ndarray, leads: np.ndarray, block: int,
                  seed: int = 0) -> SkillResult:
    """``skill_vs_lead`` with the moving-block bootstrap over start times -- the one comparison the whole module uses,
    so that every "A beats B" here carries the same 95 % interval."""
    return skill_vs_lead(err_model, err_reference, leads, block, seed=seed)


# ---------------------------------------------------------------------- training
@dataclass
class MemberTrainConfig:
    """Small-CPU budget: one block of loss per sample, the paper's loss, Adam with gradient clipping."""
    epochs: int = 4
    batch_size: int = 8
    lr: float = 1e-3
    alpha: float = 8.0
    clip: float = 1.0
    seed: int = 0


@torch.no_grad()
def _val_loss(model, X, Y, alpha: float, batch: int = 16) -> float:
    model.eval()
    tot = 0.0
    for i in range(0, len(X), batch):
        pred = model(X[i:i + batch])
        tot += float(duarte_loss(pred, Y[i:i + batch], alpha)) * len(pred)
    return tot / max(len(X), 1)


def train_member(model: nn.Module, X: np.ndarray, Y: np.ndarray, Xv: np.ndarray, Yv: np.ndarray,
                 cfg: MemberTrainConfig | None = None, *, log=print) -> dict:
    """Train one member on blocks ``X -> Y`` with the paper's loss; the weights with the best validation loss are
    restored at the end (and returned in ``state``). ``X``: (n, K, H, W) normalised log density."""
    cfg = cfg or MemberTrainConfig()
    torch.manual_seed(cfg.seed)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    Xt, Yt = torch.from_numpy(np.asarray(X, np.float32)), torch.from_numpy(np.asarray(Y, np.float32))
    Xv, Yv = torch.from_numpy(np.asarray(Xv, np.float32)), torch.from_numpy(np.asarray(Yv, np.float32))
    rng = np.random.default_rng(cfg.seed)
    hist = {"train": [], "val": [], "seconds": []}
    best, best_state, t0 = float("inf"), None, time.time()
    for ep in range(cfg.epochs):
        model.train()
        perm = rng.permutation(len(Xt))
        tot = 0.0
        for i in range(0, len(perm), cfg.batch_size):
            b = perm[i:i + cfg.batch_size]
            loss = duarte_loss(model(Xt[b]), Yt[b], cfg.alpha)
            opt.zero_grad()
            loss.backward()
            if cfg.clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.clip)
            opt.step()
            tot += loss.item() * len(b)
        val = _val_loss(model, Xv, Yv, cfg.alpha)
        hist["train"].append(tot / max(len(perm), 1))
        hist["val"].append(val)
        hist["seconds"].append(time.time() - t0)
        if val < best:
            best, best_state = val, {k: v.detach().clone() for k, v in model.state_dict().items()}
        if log is not None:
            log(f"  epoch {ep + 1}/{cfg.epochs}  train {hist['train'][-1]:.4f}  val {val:.4f}  "
                f"({hist['seconds'][-1]:.1f} s)")
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    hist["best_val"] = best
    return hist


# ---------------------------------------------------------------------- the measurement
def evaluate_family(members: Mapping[str, nn.Module], codec: DensityCodec, train, val, test, *, frame_dt: float,
                    cell_volume: np.ndarray, blocks: int = 10, start_every: int = 5, block_M: float = 1000.0,
                    dmd_ranks: Sequence[int] = (5, 10, 20, 40, 80), histories: Mapping[str, dict] | None = None,
                    log=print) -> dict:
    """Every member, both aggregations and the baselines, over the same start times and with the same intervals.

    * ``skill``: :func:`pinneapple_physics.blackhole.skill.evaluate_skill` over the members and the ``equal``
      ensemble -- so their trust horizons, their intervals against persistence, DMD and the time-mean flow, and their
      mass checks are exactly the numbers the rest of this module's docs report, from the same code path.
    * ``adaptive``: the prequential ensemble, scored on the same starts (its weights before case k come only from
      the labels of cases before k).
    * ``paired``: the headline question -- the ensemble against the best single model, per lead, with a moving-block
      bootstrap over start times (``skill_vs_lead``), for both aggregations. "Best single" is chosen on the
      validation trajectory, never on the test one.
    """
    names = list(members)
    codec0 = codec
    train_x = [((np.asarray(t, np.float32) - codec0.log_min) / (codec0.log_max - codec0.log_min)) for t in train]
    val_x = [((np.asarray(t, np.float32) - codec0.log_min) / (codec0.log_max - codec0.log_min)) for t in val]
    test_x = [((np.asarray(t, np.float32) - codec0.log_min) / (codec0.log_max - codec0.log_min)) for t in test]

    equal_members = {n: members[n] for n in names}
    models = {**{n: (members[n], codec0) for n in names},
              "ensemble-equal": (EqualWeightEnsemble(equal_members), codec0)}
    t0 = time.time()
    skill = evaluate_skill(models, train, val, test, frame_dt=frame_dt, cell_volume=cell_volume, blocks=blocks,
                           start_every=start_every, block_M=block_M, dmd_ranks=dmd_ranks, log=log)
    log(f"evaluate_skill done in {time.time() - t0:.0f} s")

    # the same start times, for the adaptive ensemble and for the paired comparisons
    n_lead = blocks * K
    leads = frame_dt * (1 + np.arange(n_lead))
    starts = {i: list(range(K - 1, len(x) - n_lead, start_every)) for i, x in enumerate(test_x)}
    queries = [q for i, x in enumerate(test_x) for q in forecast_queries(x, blocks, starts[i])]
    refs = [x[q["start"] + 1:q["start"] + 1 + n_lead] for i, x in enumerate(test_x)
            for q in forecast_queries(x, blocks, starts[i])]
    last = [x[q["start"]] for i, x in enumerate(test_x) for q in forecast_queries(x, blocks, starts[i])]

    alone = {n: [rollout(members[n], q["x0"], blocks) for q in queries] for n in names}
    equal = [combine([alone[n][k] for n in names], np.full(len(names), 1.0 / len(names)))
             for k in range(len(queries))]

    env = MassEnvelope.fit([normalised_mass(x, codec0, cell_volume) for x in train_x], frame_dt)
    base_q = {n: (forecast_queries(val_x[0], blocks) if val_x else []) for n in names}
    ens = adaptive_ensemble(members, codec0, cell_volume, frame_dt, env, baseline_queries=base_q, log=log)
    run = ens.run(queries, refs, keep_predictions=True)
    adapt = [p["prediction"] for p in run.predictions]

    errs = {n: per_lead_error(alone[n], refs) for n in names}
    errs["ensemble-equal"] = per_lead_error(equal, refs)
    errs["ensemble-adaptive"] = per_lead_error(adapt, refs)
    errs["persistence"] = per_lead_error([np.repeat(x[None], n_lead, 0) for x in last], refs)
    block = max(1, int(round(block_M / (start_every * frame_dt))))

    # the best single model, chosen on the validation trajectory
    def val_error(n: str) -> float:
        vq = forecast_queries(val_x[0], blocks)
        if not vq:
            return float("nan")
        return float(np.mean(per_lead_error([rollout(members[n], q["x0"], blocks) for q in vq],
                                            [val_x[0][q["start"] + 1:q["start"] + 1 + n_lead] for q in vq])))
    val_errors = {n: val_error(n) for n in names}
    best = min(val_errors, key=val_errors.get)
    log(f"validation error by member {val_errors} -> best single {best}")

    paired = {}
    for label in ("ensemble-equal", "ensemble-adaptive"):
        res = skill_against(errs[label], errs[best], leads, block, seed=7)
        paired[label] = _paired_record(res, errs[label], errs[best], leads)
    paired["ensemble-adaptive_vs_equal"] = _paired_record(
        skill_against(errs["ensemble-adaptive"], errs["ensemble-equal"], leads, block, seed=11),
        errs["ensemble-adaptive"], errs["ensemble-equal"], leads)

    summary = {
        "protocol": {"frame_dt": frame_dt, "blocks": blocks, "n_lead": n_lead, "start_every": start_every,
                     "starts": int(len(queries)), "block_starts_bootstrap": block, "block_M": block_M,
                     "trajectories": int(len(test_x))},
        "members": {n: {"parameters": int(sum(p.numel() for p in members[n].parameters())),
                        "val_mae": float(val_errors[n]),
                        "history": (histories or {}).get(n, {})} for n in names},
        "skill": skill,
        "adaptive": {"weights_final": {k: float(v) for k, v in ens.weights().items()},
                     "weights_by_case": run.weights.tolist(), "active": run.active,
                     "residual_baseline": dict(ens.residual_baseline),
                     "mean_cost_per_case": run.summary()["mean_cost_per_case"],
                     "mae_by_lead": errs["ensemble-adaptive"].mean(0).tolist(),
                     "skill_vs_persistence": _paired_record(
                         skill_against(errs["ensemble-adaptive"], errs["persistence"], leads, block, seed=13),
                         errs["ensemble-adaptive"], errs["persistence"], leads)},
        "paired": paired,
        "verdict": _verdict(paired, errs, best),
        "mean_mae": {n: e.mean(0).tolist() for n, e in errs.items()},
        "leads_M": leads.tolist(),
    }
    return summary


def _paired_record(res: SkillResult, a: np.ndarray, b: np.ndarray, leads: np.ndarray) -> dict:
    ss = res.skill_score
    better = res.diff[:, 2] < 0
    return {"horizon_M": res.horizon, "skill_score": ss.tolist(), "diff": res.diff.tolist(),
            "leads_where_better": int(better.sum()), "leads": int(len(leads)),
            "mean_error_a": float(a.mean()), "mean_error_b": float(b.mean()),
            "skill_at": {f"{leads[j]:.0f}M": [float(ss[j, 0]), float(ss[j, 1]), float(ss[j, 2])] for j in
                         np.unique(np.linspace(0, len(leads) - 1, min(5, len(leads))).round().astype(int))}}


def _verdict(paired: dict, errs: dict, best: str) -> dict:
    """The measured answer, in one dict: does each aggregation beat the best single member, and where."""
    out = {"best_single": best, "mean_mae": {n: float(e.mean()) for n, e in errs.items()}}
    for label, rec in paired.items():
        if label != "ensemble-adaptive_vs_equal":
            out[label] = {"beats_best_single": bool(rec["mean_error_a"] < rec["mean_error_b"]),
                          "horizon_M": rec["horizon_M"],
                          "leads_where_better": f"{rec['leads_where_better']}/{rec['leads']}"}
    good = [lab for lab in ("ensemble-equal", "ensemble-adaptive") if out[lab]["beats_best_single"]]
    out["answer"] = (f"{' and '.join(good)} beat {best} over the horizon" if good
                     else f"no aggregation beats {best} over the horizon")
    return out
