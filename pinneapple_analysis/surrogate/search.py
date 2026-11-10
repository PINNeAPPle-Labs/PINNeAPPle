"""Leakage-aware hybrid DeepONet search on field datasets.

Every trial sees only train and validation. Held-out and shift metrics are computed
once, after the validation winner is selected. Nothing is auto-approved for serving.
"""
from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .data import SurrogateData
from .limits import OODGate
from .methods import TrainContext, fit_candidate
from .model import SurrogateModel
from .splits import Splits, make_splits
from .study import evaluate


def real_data_splits(data: SurrogateData, *, seed: int = 0, test_frac: float = 0.2,
                     val_frac: float = 0.15, shift_frac: float = 0.15) -> Splits:
    """Disjoint sample splits, fully group-disjoint if case/geometry IDs exist.

    Splitting individual measurements from one experiment across partitions
    inflates apparent generalization and must be avoided when groups are provided.
    """
    if not 0 < test_frac < 1 or not 0 < val_frac < 1 or not 0 <= shift_frac < 1:
        raise ValueError("invalid split fractions")
    if test_frac + val_frac + shift_frac >= 1:
        raise ValueError("split fractions leave no training data")
    if data.groups is None:
        return make_splits(data, seed=seed, test_frac=test_frac, val_frac=val_frac,
                           shift="edge" if shift_frac else None, shift_frac=shift_frac)
    groups = np.unique(data.groups)
    if len(groups) < 8:
        raise ValueError("grouped real-world data need at least eight independent groups")
    rng = np.random.default_rng(seed)
    g = rng.permutation(groups)
    counts = [max(1, int(round(len(g) * f))) if f else 0 for f in (shift_frac, test_frac, val_frac)]
    if sum(counts) >= len(g):
        raise ValueError("too few independent groups for the requested splits")
    sh, te, va = counts
    partitions = (g[sh + te + va:], g[sh + te:sh + te + va], g[sh:sh + te], g[:sh])
    indices = [np.flatnonzero(np.isin(data.groups, part)) for part in partitions]
    if len(indices[0]) < 4 or not all(len(x) for x in indices[1:3]):
        raise ValueError("insufficient samples in grouped train, validation or test")
    return Splits(*indices, shift_kind="group" if sh else None,
                  shift_description="completely unseen experiment/geometry groups" if sh else "", seed=seed,
                  info={"leakage_control": "group-disjoint"})


@dataclass
class HybridSearchResult:
    model: SurrogateModel
    trials: list[dict[str, Any]]
    winner: dict[str, Any]
    holdout: dict[str, Any]
    split: Splits

    def to_dict(self) -> dict[str, Any]:
        return {"trials": self.trials, "winner": self.winner, "holdout": self.holdout,
                "splits": self.split.to_dict(), "model_card": self.model.card}


def hybrid_search(data: SurrogateData, *, branches: Sequence[str] = ("mlp", "rbf_kan", "fourier"),
                  trunks: Sequence[str] = ("mlp", "rbf_kan", "fourier"),
                  hidden: Sequence[int] = (32, 64), modes: Sequence[int] = (16, 32),
                  budget_s: float = 5.0, max_trials: int = 12, seed: int = 0,
                  split: Splits | None = None, base_options: Mapping[str, Any] | None = None) -> HybridSearchResult:
    """Search architecture and width with common splits, seeds and time budgets.

    The input dataset must be a *field* with fixed coordinates per sample.
    This does not claim to learn on changing meshes; that needs alignment first.
    """
    from .hybrid import ENCODERS

    if data.kind != "field":
        raise ValueError("DeepONet requires a field dataset with coordinates")
    if budget_s <= 0 or max_trials < 1:
        raise ValueError("budget_s and max_trials must be positive")
    for choice in (*branches, *trunks):
        if choice not in ENCODERS:
            raise ValueError(f"unsupported encoder {choice!r}")
    from pinneapple_analysis.cost.equal_budget import ComputeBudget

    sp = split if split is not None else real_data_splits(data, seed=seed)
    if min(len(sp.train), len(sp.val), len(sp.test)) < 1:
        raise ValueError("train/validation/test partitions must be nonempty")
    parts = [np.asarray(getattr(sp, k), dtype=int) for k in ("train", "val", "test", "shift")]
    if any(np.any((p < 0) | (p >= data.n)) for p in parts):
        raise ValueError("split indices outside dataset")
    if len(np.unique(np.concatenate(parts))) != sum(map(len, parts)):
        raise ValueError("splits overlap")
    if data.groups is not None:
        collections = [set(data.groups[p].tolist()) for p in parts]
        if any(collections[i] & collections[j] for i in range(4) for j in range(i + 1, 4)):
            raise ValueError("group leakage between training and evaluation splits")
    candidates = list(itertools.product(branches, trunks, hidden, modes))
    if not candidates:
        raise ValueError("empty hyperparameter search space")
    if any(int(h) < 1 or int(m) < 1 for _, _, h, m in candidates):
        raise ValueError("hidden and modes must be positive")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(candidates))[:max_trials]
    shape = data.Y.shape[1:]
    flat = lambda idx: data.Y[idx].reshape(len(idx), -1)
    Xtr, Ytr, Xva, Yva = data.X[sp.train], flat(sp.train), data.X[sp.val], flat(sp.val)
    scale = Ytr.std(0) + 1e-12
    trials, winner, best_fitted = [], None, None
    for ix in order:
        branch, trunk, width, n_modes = candidates[int(ix)]
        params = dict(base_options or {})
        params.update(branch=branch, trunk=trunk, hidden=int(width), modes=int(n_modes))
        ctx = TrainContext(Xtr, Ytr, Xva, Yva, ComputeBudget(wall_s=budget_s), seed=seed,
                           grid_shape=data.grid_shape, coords=data.coords, options=params)
        try:
            fitted, elapsed = fit_candidate("deeponet_hybrid", ctx)
            validation = evaluate("field", fitted.predict(Xva), Yva, scale, data.output_names)
            val_error = float(validation["error"])
            if not np.isfinite(val_error):
                raise ValueError("non-finite validation score")
            row = {"options": params, "val_error": val_error, "fit_wall_s": elapsed, "status": "ok"}
            if winner is None or val_error < winner["val_error"]:
                winner, best_fitted = row, fitted
        except Exception as exc:
            row = {"options": params, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        trials.append(row)
    if best_fitted is None or winner is None:
        raise RuntimeError("no hybrid candidate trained successfully; inspect search options and trial errors")
    holdout = {"test": evaluate("field", best_fitted.predict(data.X[sp.test]), flat(sp.test), scale,
                                 data.output_names)}
    if len(sp.shift):
        holdout["shift"] = evaluate("field", best_fitted.predict(data.X[sp.shift]), flat(sp.shift), scale,
                                    data.output_names)
    gate = OODGate(Xtr, data.input_names)
    card = {"dataset_sha256": data.sha256(), "source": data.meta.get("source"),
            "hpo": {"selected_by": "validation_only", "winner": winner["options"],
                    "n_trials": len(trials), "seed": seed, "splits": sp.to_dict()},
            "metrics": holdout, "trust": {"decision": "not_verified",
                                          "reason": "independent physical validation and approval required"}}
    model = SurrogateModel(best_fitted, gate, input_names=data.input_names, output_names=data.output_names,
                           grid_shape=shape, coords=data.coords, coord_names=data.coord_names, card=card)
    return HybridSearchResult(model, trials, winner, holdout, sp)
