"""Expected goals with and without physics: a standard shot-geometry baseline against intercept-time features
computed from the StatsBomb 360 freeze frame.

Standard features (what most public xG models use): distance and angle to goal, header, body part, first time,
under pressure, free kick, play pattern.

Physics features (:func:`physics_features`, 2-D, ball height ignored): the ball travels in a straight line from
the shot to each of ``n_mouth`` points of the goal mouth at speed ``v_shot``; a player at distance ``d`` from a
point of that path can intercept it if ``reaction + max(d - reach, 0) / v_player <= t_ball``, with a larger
``reach`` and a lower ``reaction`` for the goalkeeper (dive). From that:

* ``open_fraction`` -- share of the goal mouth no opponent can intercept (the physics analogue of the visible
  goal angle);
* ``gk_margin`` -- the keeper's best time margin over the mouth (negative: he cannot get there in time);
* ``blockers`` -- outfield opponents able to intercept the path to the goal centre.

Units: StatsBomb coordinates are yards (120 x 80, goal posts at y = 36 and 44 on x = 120); converted to metres.
:func:`select_params` chooses ``v_shot`` and the keeper's speed from a small grid on the training shots only.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from . import metrics as M

__all__ = ["XGParams", "standard_features", "physics_features", "select_params", "compare", "data_efficiency",
           "shots_arrays", "freeze_geometry_features"]

YD = 0.9144
GOAL_X = 120 * YD
POSTS = (36 * YD, 44 * YD)


@dataclass(frozen=True)
class XGParams:
    v_shot: float = 20.0          # m/s, straight-line average ball speed of a shot (grid-selected)
    v_gk: float = 5.0             # m/s, keeper's effective lateral speed (grid-selected)
    reach_gk: float = 1.5         # m, keeper reach (arm span plus dive)
    reaction_gk: float = 0.25     # s
    v_def: float = 5.0            # m/s
    reach_def: float = 0.5        # m, a blocking leg or body
    reaction_def: float = 0.3     # s
    n_mouth: int = 25
    n_path: int = 20


def shots_arrays(shots: list[dict]) -> dict[str, Any]:
    """Shot table to arrays: positions in metres, label, freeze frame padded to ``(n, 22, 4)`` (x, y, mate, gk)."""
    n = len(shots)
    F = np.full((n, 22, 4), np.nan)
    for i, s in enumerate(shots):
        for j, (x, y, mate, gk) in enumerate(s["freeze"][:22]):
            F[i, j] = (x * YD, y * YD, float(mate), float(gk))
    return {"xy": np.array([[s["x"] * YD, s["y"] * YD] for s in shots]), "y": np.array([s["goal"] for s in shots]),
            "freeze": F, "match": np.array([s["match_id"] for s in shots]),
            "season": np.array([s["season"] for s in shots]), "sb_xg": np.array([s["statsbomb_xg"] for s in shots]),
            "shots": shots}


def standard_features(A: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    xy = A["xy"]
    c = np.array([GOAL_X, 40 * YD])
    d = np.linalg.norm(xy - c, axis=1)
    a1 = np.arctan2(POSTS[0] - xy[:, 1], GOAL_X - xy[:, 0])
    a2 = np.arctan2(POSTS[1] - xy[:, 1], GOAL_X - xy[:, 0])
    ang = np.abs(a2 - a1)
    S = A["shots"]
    cols = [d, ang, np.log1p(d), d * ang,
            [s["body_part"] == "Head" for s in S], [s["body_part"] == "Other" for s in S],
            [s["first_time"] for s in S], [s["under_pressure"] for s in S],
            [s["shot_type"] == "Free Kick" for s in S], [s["play_pattern"] in ("From Corner", "From Free Kick")
                                                           for s in S],
            [s["play_pattern"] == "From Counter" for s in S]]
    names = ["distance", "angle", "log_distance", "dist_x_angle", "header", "other_body", "first_time",
             "under_pressure", "free_kick", "set_piece_pattern", "counter"]
    return np.column_stack([np.asarray(c, float) for c in cols]), names


def physics_features(A: dict[str, Any], p: XGParams | None = None, *, chunk: int = 200) -> tuple[np.ndarray, list[str]]:
    p = p or XGParams()
    names = ["open_fraction", "gk_margin", "blockers", "sqrt_open_fraction"]
    parts = [_physics_chunk(A["xy"][i:i + chunk], A["freeze"][i:i + chunk], p) for i in range(0, len(A["xy"]), chunk)]
    return np.concatenate(parts) if parts else np.empty((0, len(names))), names


def _physics_chunk(xy, F, p: XGParams) -> np.ndarray:
    mouth = np.column_stack([np.full(p.n_mouth, GOAL_X), np.linspace(POSTS[0], POSTS[1], p.n_mouth)])
    s = np.linspace(0, 1, p.n_path)[1:]                                   # fractions along the path
    # path points (n, mouth, path, 2) and ball times
    P = xy[:, None, None, :] + (mouth[None, :, None, :] - xy[:, None, None, :]) * s[None, None, :, None]
    L = np.linalg.norm(mouth[None] - xy[:, None], axis=2)                 # (n, mouth)
    tb = L[:, :, None] * s[None, None, :] / p.v_shot
    opp = F[:, :, 2] == 0
    gk = (F[:, :, 3] == 1) & opp
    d = np.linalg.norm(P[:, None] - F[:, :, None, None, :2], axis=-1)     # (n, players, mouth, path)
    reach = np.where(gk, p.reach_gk, p.reach_def)[:, :, None, None]
    react = np.where(gk, p.reaction_gk, p.reaction_def)[:, :, None, None]
    v = np.where(gk, p.v_gk, p.v_def)[:, :, None, None]
    t_need = react + np.maximum(d - reach, 0) / v
    margin = tb[:, None] - t_need                                          # >= 0: can intercept there
    margin = np.where((opp & np.isfinite(F[:, :, 0]))[:, :, None, None], margin, -np.inf)
    best_pp = margin.max(axis=3)                                           # (n, players, mouth)
    covered = (best_pp >= 0).any(axis=1)                                   # (n, mouth)
    open_fraction = 1 - covered.mean(axis=1)
    gk_best = np.where(gk[:, :, None], best_pp, -np.inf).max(axis=(1, 2))
    gk_margin = np.where(np.isfinite(gk_best), gk_best, -1.0)
    centre = p.n_mouth // 2
    blockers = ((best_pp[:, :, centre] >= 0) & ~gk).sum(axis=1)
    return np.column_stack([open_fraction, np.clip(gk_margin, -1, 1), blockers, np.sqrt(open_fraction)])


def freeze_geometry_features(A: dict[str, Any]) -> tuple[np.ndarray, list[str]]:
    """Data-only freeze-frame features (no physics): opponents inside the shot triangle, keeper distance to the
    goal centre and his lateral offset from the shot line, nearest opponent distance to the shooter."""
    xy, F = A["xy"], A["freeze"]
    opp = (F[:, :, 2] == 0) & np.isfinite(F[:, :, 0])
    gk = opp & (F[:, :, 3] == 1)
    a, b = np.array([GOAL_X, POSTS[0]]), np.array([GOAL_X, POSTS[1]])
    P = F[:, :, :2]

    def side(p1, p2, q):
        return (p2[..., 0] - p1[..., 0]) * (q[..., 1] - p1[..., 1]) - (p2[..., 1] - p1[..., 1]) * (q[..., 0] - p1[..., 0])

    s0 = xy[:, None, :]
    d1, d2, d3 = side(s0, a, P), side(a, b, P), side(b, s0, P)
    inside = ((d1 >= 0) & (d2 >= 0) & (d3 >= 0)) | ((d1 <= 0) & (d2 <= 0) & (d3 <= 0))
    in_tri = (inside & opp & ~gk).sum(1)
    c = np.array([GOAL_X, 40 * YD])
    gk_pos = np.where(gk[..., None], P, np.nan)
    has = gk.any(1)
    gpos = np.nansum(np.where(gk[..., None], gk_pos, 0.0), axis=1) / np.maximum(gk.sum(1), 1)[:, None]
    gpos = np.where(has[:, None], gpos, c)
    gk_goal = np.linalg.norm(gpos - c, axis=1)
    u = (c - xy) / np.maximum(np.linalg.norm(c - xy, axis=1), 1e-6)[:, None]
    rel = gpos - xy
    gk_off = np.abs(rel[:, 0] * u[:, 1] - rel[:, 1] * u[:, 0])
    dn = np.where(opp, np.linalg.norm(P - s0, axis=2), 99).min(1)
    return np.column_stack([in_tri, gk_goal, gk_off, np.minimum(dn, 30), has]), [
        "opp_in_triangle", "gk_to_goal", "gk_off_line", "nearest_opp", "gk_visible"]


def _fit(kind: str, X, y, seed: int = 0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if kind == "logistic":
        return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)).fit(X, y)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=max(10, len(y) // 50), l2_regularization=1.0,
                                          random_state=seed).fit(X, y)


def select_params(A_train: dict[str, Any], grid_v_shot=(12.0, 16.0, 20.0, 25.0, 30.0),
                  grid_v_gk=(3.0, 5.0, 7.0), folds: int = 5) -> tuple[XGParams, dict]:
    """Choose ``v_shot`` and ``v_gk`` by grouped cross-validated log-loss of a logistic model on the standard plus
    physics features, on the training shots only."""
    Xs, _ = standard_features(A_train)
    y = A_train["y"]
    groups = A_train["match"]
    ug = np.unique(groups)
    rng = np.random.default_rng(0)
    fold_of = dict(zip(ug, rng.integers(0, folds, len(ug)), strict=True))
    fid = np.array([fold_of[g] for g in groups])
    table = {}
    for vs in grid_v_shot:
        for vg in grid_v_gk:
            p = replace(XGParams(), v_shot=vs, v_gk=vg)
            X = np.column_stack([Xs, physics_features(A_train, p)[0]])
            pr = np.empty(len(y))
            for f in range(folds):
                tr, te = fid != f, fid == f
                pr[te] = _fit("logistic", X[tr], y[tr]).predict_proba(X[te])[:, 1]
            table[(vs, vg)] = M.log_loss(pr, y)
    best = min(table, key=table.get)
    return replace(XGParams(), v_shot=best[0], v_gk=best[1]), {f"{k[0]}_{k[1]}": v for k, v in table.items()}


def compare(A_train: dict[str, Any], A_test: dict[str, Any], *, params: XGParams | None = None,
            seed: int = 0) -> dict[str, Any]:
    """Test scores of: standard logistic / GBM, physics-only logistic, standard+physics logistic / GBM, standard plus
    data-only freeze-frame geometry (with and without physics); StatsBomb's
    own xG on the same shots as an external reference (a proprietary model trained on far more shots)."""
    if params is None:
        params, grid = select_params(A_train)
    else:
        grid = {}
    Xs_tr, _ = standard_features(A_train)
    Xs_te, _ = standard_features(A_test)
    Xp_tr, _ = physics_features(A_train, params)
    Xp_te, _ = physics_features(A_test, params)
    Xg_tr, _ = freeze_geometry_features(A_train)
    Xg_te, _ = freeze_geometry_features(A_test)
    ytr, yte = A_train["y"], A_test["y"]
    specs = {"standard_logistic": ("logistic", Xs_tr, Xs_te), "standard_gbm": ("gbm", Xs_tr, Xs_te),
             "physics_logistic": ("logistic", np.column_stack([Xs_tr[:, :2], Xp_tr]),
                                  np.column_stack([Xs_te[:, :2], Xp_te])),
             "standard+physics_logistic": ("logistic", np.column_stack([Xs_tr, Xp_tr]), np.column_stack([Xs_te, Xp_te])),
             "standard+physics_gbm": ("gbm", np.column_stack([Xs_tr, Xp_tr]), np.column_stack([Xs_te, Xp_te])),
             "standard+freeze_geometry_logistic": ("logistic", np.column_stack([Xs_tr, Xg_tr]),
                                                   np.column_stack([Xs_te, Xg_te])),
             "standard+freeze+physics_logistic": ("logistic", np.column_stack([Xs_tr, Xg_tr, Xp_tr]),
                                                  np.column_stack([Xs_te, Xg_te, Xp_te]))}
    P = {k: np.clip(_fit(kind, a, ytr, seed).predict_proba(b)[:, 1], 1e-4, 1 - 1e-4) for k, (kind, a, b) in specs.items()}
    P["statsbomb_xg_reference"] = np.clip(A_test["sb_xg"], 1e-4, 1 - 1e-4)
    scores = {k: M.score(v, yte) for k, v in P.items()}
    paired = {k: M.paired_bootstrap(P[k], P["standard_logistic"], yte) for k in P if k != "standard_logistic"}
    rel = {k: {kk: vv.tolist() for kk, vv in M.reliability(v, yte, bins=8).items()} for k, v in P.items()}
    return {"params": params.__dict__, "grid_logloss": grid, "scores": scores, "paired_vs_standard_logistic": paired,
            "reliability": rel, "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "pred": {k: v.tolist() for k, v in P.items()}}


def data_efficiency(A_train, A_test, params: XGParams, sizes=(100, 200, 400, 800, None), seeds=(0, 1, 2, 3, 4)):
    """Test log-loss of the standard and standard+physics logistic models trained on random subsets."""
    Xs_tr, _ = standard_features(A_train)
    Xs_te, _ = standard_features(A_test)
    Xp_tr, _ = physics_features(A_train, params)
    Xp_te, _ = physics_features(A_test, params)
    rows = []
    for n in sizes:
        for s in seeds if n is not None else seeds[:1]:
            rng = np.random.default_rng(1000 + s)
            idx = np.arange(len(A_train["y"])) if n is None else rng.choice(len(A_train["y"]), n, replace=False)
            for name, a, b in (("standard_logistic", Xs_tr, Xs_te),
                               ("standard+physics_logistic", np.column_stack([Xs_tr, Xp_tr]),
                                np.column_stack([Xs_te, Xp_te])),
                               ("physics_logistic", np.column_stack([Xs_tr[:, :2], Xp_tr]),
                                np.column_stack([Xs_te[:, :2], Xp_te]))):
                if A_train["y"][idx].sum() < 2:
                    continue
                pr = _fit("logistic", a[idx], A_train["y"][idx]).predict_proba(b)[:, 1]
                rows.append({"n_train": int(len(idx)), "seed": s, "model": name, "log_loss": M.log_loss(pr, A_test["y"]),
                             "auc": M.auc(pr, A_test["y"])})
    return {"rows": rows}
