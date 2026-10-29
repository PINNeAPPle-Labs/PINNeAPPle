"""Pass outcome: physics-based pitch control against data-only classifiers at equal data.

Train on one Metrica match, test on the other (and the swap). Every model sees the same passes:

* ``pc_default`` -- Spearman (2018) pitch control with the parameters common in open implementations (no fit);
* ``pc_reaction`` / ``pc_accel`` -- the same with the reaction-time or the acceleration-limited time to intercept,
  parameters fitted by maximum likelihood on the training passes (5 numbers);
* ``logistic`` / ``gbm`` -- logistic regression and gradient-boosted trees on geometric features
  (:func:`pitch_control.geometric_features`), no physics;
* ``hybrid`` -- the trees with the pitch-control probability and the intercept-time margin as extra features.

:func:`data_efficiency` repeats the comparison with 25 to all training passes (random subsets, several seeds).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from . import metrics as M
from . import pitch_control as PC

__all__ = ["physics_features", "fit_models", "predict_models", "compare", "data_efficiency"]

FIT_NAMES = {"reaction": ("sigma", "lam", "kappa_def", "v_max", "reaction"),
             "accel": ("sigma", "lam", "kappa_def", "v_max", "a_max")}


def physics_features(ps: PC.PassSet, p: PC.PCParams) -> np.ndarray:
    """Pitch-control probability, earliest attacker and defender intercept times minus the ball travel time."""
    q = PC.control_probability(ps.att_pos, ps.att_vel, ps.def_pos, ps.def_vel, ps.ball, ps.target, p,
                               ps.att_gk, ps.def_gk)
    tgt = ps.target[:, None, :]
    ta = PC.time_to_intercept(ps.att_pos, np.nan_to_num(ps.att_vel), tgt, p)
    td = PC.time_to_intercept(ps.def_pos, np.nan_to_num(ps.def_vel), tgt, p)
    ta = np.where(np.isfinite(ps.att_pos[..., 0]), ta, 99)
    td = np.where(np.isfinite(ps.def_pos[..., 0]), td, 99)
    tb = PC._ball_time(np.linalg.norm(ps.target - ps.ball, axis=1), p)
    # the passer is the attacker nearest the ball: exclude him
    db = np.linalg.norm(ps.att_pos - ps.ball[:, None, :], axis=2)
    ta[np.arange(len(ta)), np.nanargmin(np.where(np.isfinite(db), db, 999), 1)] = 99
    return np.column_stack([q, ta.min(1) - tb, td.min(1) - tb, td.min(1) - ta.min(1)])


def fit_models(train: PC.PassSet, *, seed: int = 0, which=("pc_default", "pc_reaction", "pc_accel", "logistic",
                                                            "gbm", "hybrid")) -> dict[str, Any]:
    """Fitted models (parameters or estimators) on ``train``."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    out: dict[str, Any] = {}
    X, _ = PC.geometric_features(train)
    if "pc_default" in which:
        out["pc_default"] = PC.PCParams()
    if "pc_reaction" in which:
        out["pc_reaction"], _ = PC.fit_params(train, PC.PCParams(tti="reaction"), FIT_NAMES["reaction"])
    if "pc_accel" in which:
        out["pc_accel"], _ = PC.fit_params(train, PC.PCParams(tti="accel"), FIT_NAMES["accel"])
    if "logistic" in which:
        out["logistic"] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(X, train.y)
    gbm = dict(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=max(5, len(train) // 50),
               l2_regularization=1.0, random_state=seed)
    if "gbm" in which:
        out["gbm"] = HistGradientBoostingClassifier(**gbm).fit(X, train.y)
    if "hybrid" in which:
        p = out.get("pc_reaction") or PC.DEFAULT
        Xh = np.column_stack([X, physics_features(train, p)])
        out["hybrid"] = (p, HistGradientBoostingClassifier(**gbm).fit(Xh, train.y))
    return out


def predict_models(models: dict[str, Any], ps: PC.PassSet) -> dict[str, np.ndarray]:
    X, _ = PC.geometric_features(ps)
    out = {}
    for k, m in models.items():
        if isinstance(m, PC.PCParams):
            out[k] = PC.control_probability(ps.att_pos, ps.att_vel, ps.def_pos, ps.def_vel, ps.ball, ps.target, m,
                                            ps.att_gk, ps.def_gk)
        elif k == "hybrid":
            p, est = m
            out[k] = est.predict_proba(np.column_stack([X, physics_features(ps, p)]))[:, 1]
        else:
            out[k] = m.predict_proba(X)[:, 1]
    return {k: np.clip(v, 1e-4, 1 - 1e-4) for k, v in out.items()}


def compare(train: PC.PassSet, test: PC.PassSet, *, seed: int = 0) -> dict[str, Any]:
    """Scores of every model on ``test`` after fitting on ``train``, paired bootstrap of log-loss differences
    against the best data-only model, and the fitted physical parameters."""
    models = fit_models(train, seed=seed)
    P = predict_models(models, test)
    scores = {k: M.score(v, test.y) for k, v in P.items()}
    best_data = min(("logistic", "gbm"), key=lambda k: scores[k]["log_loss"])
    paired = {k: M.paired_bootstrap(P[k], P[best_data], test.y) for k in P if k != best_data}
    params = {k: {n: getattr(v, n) for n in ("sigma", "lam", "kappa_def", "v_max", "a_max", "reaction")}
              for k, v in models.items() if isinstance(v, PC.PCParams)}
    rel = {k: {kk: vv.tolist() for kk, vv in M.reliability(v, test.y, bins=8).items()} for k, v in P.items()}
    return {"scores": scores, "paired_vs_best_data_only": paired, "best_data_only": best_data, "params": params,
            "reliability": rel, "n_train": len(train), "n_test": len(test), "pred": {k: v.tolist() for k, v in P.items()}}


def data_efficiency(train: PC.PassSet, test: PC.PassSet, sizes=(25, 50, 100, 200, 400, None), seeds=(0, 1, 2),
                    which=("pc_default", "pc_reaction", "logistic", "gbm", "hybrid")) -> dict[str, Any]:
    """Test log-loss of each model trained on random subsets of ``train`` (``None`` = all passes)."""
    rows = []
    rng0 = np.random.default_rng(0)
    for n in sizes:
        for s in seeds if n is not None else seeds[:1]:
            rng = np.random.default_rng(int(rng0.integers(1 << 30)) + s)
            if n is None:
                idx = np.arange(len(train))
            else:
                # stratified: keep both classes
                pos, neg = np.where(train.y == 1)[0], np.where(train.y == 0)[0]
                k0 = max(2, int(round(n * len(neg) / len(train.y))))
                idx = np.concatenate([rng.choice(pos, n - k0, replace=False), rng.choice(neg, k0, replace=False)])
            sub = train.subset(idx)
            models = fit_models(sub, seed=s, which=which)
            P = predict_models(models, test)
            for k, v in P.items():
                rows.append({"n_train": len(idx), "seed": s, "model": k, "log_loss": M.log_loss(v, test.y),
                             "brier": M.brier(v, test.y), "auc": M.auc(v, test.y)})
    return {"rows": rows}


def with_params(p: PC.PCParams, **kw) -> PC.PCParams:
    return replace(p, **kw)
