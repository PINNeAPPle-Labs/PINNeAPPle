"""Geometry-conditioned surrogate: the field error on a body the model never saw (#83, roadmap item G11).

The question a surrogate user actually asks is *"what happens on my geometry"*, and the only honest answer is a
number measured on a geometry that was not in the training set. This case produces that number for a family of
2-D bodies, with every step measured inside the call:

1. **Geometry -> data.** A body of the family is a closed CST profile (thickness weights ``T0..T2``, camber weights
   ``M0..M2``) from :mod:`pinneapple_design.aero.geometry`, the six-number shape the library's airfoil work and its
   OpenFOAM campaign use. Each body is labelled by the Hess-Smith panel method
   (:func:`pinneapple_design.aero.panel.panel_solve`): the inviscid reference that any potential-flow or neural
   model can at best reach, and cheap enough that the labels cost milliseconds, not hours.
2. **Geometry -> model.** The surrogate never sees the six numbers. It sees the *geometry*: the panel mid-points of
   the body, one outward normal per panel, and a geometric code pooled from them (:class:`pinneapple_neural.lpm.
   LargePhysicsModel`, a DeepSets encoder plus multi-scale neighbourhoods), and it predicts the pressure coefficient
   ``Cp`` at every panel of that body. This is what makes it geometry-conditioned instead of parameter-conditioned:
   the shape reaches the model as a point set, so the same model can answer for a body described any other way.
3. **The split.** Training shapes are a Latin hypercube inside the *inner* design box (the documented parameter
   ranges shrunk by :data:`PROTOCOL` on each side). Held out are (a) shapes inside that box the model never saw,
   (b) **new geometries**: shapes drawn in the outer band of the documented ranges, so at least one weight is
   outside everything the model was trained on, and (c) two named real airfoils (NACA 2412 and NACA 4412) fitted to
   the same CST basis -- NACA 4412 sits outside the documented bounds of the 6th weight (``M1``), so the model is
   asked about a body *and* a corner of the family it never had.
4. **The report.** Per held-out body: the relative L2 of the ``Cp`` field, the suction-peak (``min Cp``) error, the
   lift coefficient integrated from the *predicted* field with the solver's own pressure integration (so the
   coefficient and the field never disagree), the geometry code distance to the training set (the library's
   out-of-distribution score), and which parameters are outside the training box. At the decision level
   (:mod:`pinneapple_analysis.decision`): the flip rate and rank preservation over "which body carries the most
   lift", because a field error that does not reorder the designs costs nothing.

The published number is in ``benchmarks/_out/geometry_conditioned_surrogate.json``; ``--quick`` runs the same
pipeline at a test budget, and it is not a benchmark.

    python -m benchmarks.geometry_conditioned_surrogate            # the published protocol
    python -m benchmarks.geometry_conditioned_surrogate --quick     # a minute, for a laptop or a test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time

import numpy as np

from pinneapple_design.aero.geometry import BOUNDS, LABELS, PARAMS, REFERENCE, latin_hypercube, properties, valid
from pinneapple_design.aero.panel import airfoil_nodes, panel_solve
from pinneapple_neural.lpm import Design, LargePhysicsModel, fit

__all__ = ["PROTOCOL", "OUT", "bodies", "body", "design", "lift_from_cp", "run", "split"]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "benchmarks", "_out", "geometry_conditioned_surrogate.json")

#: the protocol: what is trained, on what, with which budget. ``box_shrink`` is how far the training box is pulled
#: back from the documented parameter ranges, so that a "new geometry" is outside it by construction and not by luck.
#: ``seeds`` are training seeds over one design set: the spread between them is reported next to the mean, because a
#: single seed is a point, not a distribution.
PROTOCOL = {
    "train_shapes": 24,
    "unseen_shapes": 4,             # inside the training box, never trained on (interpolation)
    "new_geometries": 4,            # in the outer band of the documented ranges (a body outside the training set)
    "named_airfoils": ("NACA 2412", "NACA 4412"),
    "box_shrink": 0.2,
    "panels_per_side": 50,
    "epochs": 1500,
    "lr": 2e-3,
    "width": 96,
    "heads": 3,
    "code": 32,
    "alpha_deg": 0.0,
    "seeds": (7, 8, 9),
}

QUICK = {"train_shapes": 10, "unseen_shapes": 2, "new_geometries": 2, "named_airfoils": ("NACA 2412",),
         "panels_per_side": 24, "epochs": 400, "width": 32, "heads": 2, "seeds": (1,)}


# --------------------------------------------------------------------------------------------- the geometry
def _train_box(shrink: float) -> np.ndarray:
    """The training box: the documented ranges pulled back by ``shrink`` of their span at each end."""
    lo = BOUNDS[:, 0] + shrink * (BOUNDS[:, 1] - BOUNDS[:, 0])
    hi = BOUNDS[:, 1] - shrink * (BOUNDS[:, 1] - BOUNDS[:, 0])
    return np.stack([lo, hi], 1)


def _outside(shape: np.ndarray, box: np.ndarray) -> list[str]:
    """The parameters of ``shape`` that fall outside ``box`` (empty: the shape is inside the training box)."""
    return [n for i, n in enumerate(PARAMS) if not (box[i, 0] <= shape[i] <= box[i, 1])]


def split(protocol: dict | None = None, rng: np.random.Generator | None = None) -> dict[str, list[tuple[str, np.ndarray]]]:
    """The design set: ``{"train", "unseen", "new"}`` lists of ``(name, CST weights)``.

    ``train`` and ``unseen`` are Latin hypercubes inside the training box; ``new`` is drawn over the full documented
    ranges and kept only when at least one weight is outside the training box, so every one of them is a geometry
    the model was not trained on in any parameter. The named real airfoils are added to ``new``.
    """
    p = dict(PROTOCOL, **(protocol or {}))
    rng = rng or np.random.default_rng(p["seeds"][0])    # the design set is one; only training is repeated per seed
    box = _train_box(p["box_shrink"])

    def draw(n: int, bounds: np.ndarray, want_outside: bool | None = None) -> list[tuple[str, np.ndarray]]:
        out: list[tuple[str, np.ndarray]] = []
        tries = 0
        while len(out) < n and tries < 200:
            tries += 1
            for shape in latin_hypercube(max(n, 4), bounds, rng):
                ok, _ = valid(shape)
                if not ok:
                    continue
                outside = bool(_outside(shape, box))
                if want_outside is not None and outside != want_outside:
                    continue
                out.append((f"cst{len(out):02d}", shape))
                if len(out) == n:
                    break
        if len(out) < n:
            raise RuntimeError(f"could not draw {n} valid shapes inside {bounds.tolist()}")
        return out

    train = draw(p["train_shapes"], box)
    unseen = draw(p["unseen_shapes"], box)
    new = draw(p["new_geometries"], BOUNDS, want_outside=True)
    for name in p["named_airfoils"]:
        new.append((name, np.asarray(REFERENCE[name], dtype=float)))
    return {"train": train, "unseen": unseen, "new": new}


def body(shape: np.ndarray, n_per_side: int = 50, alpha_deg: float = 0.0) -> dict:
    """One solved body: its geometry (panel nodes, mid-points, outward normals, lengths) and its label (``Cp``).

    The panel frame and the label come from the same call, so a predicted ``Cp`` and a solved ``Cp`` are integrated
    over the same panels and the two lift coefficients are comparable.
    """
    nodes = airfoil_nodes(np.asarray(shape, float), n_per_side, close_te=True)
    result = panel_solve(nodes=nodes, alpha_deg=alpha_deg)
    x1, y1, x2, y2 = nodes[:-1, 0], nodes[:-1, 1], nodes[1:, 0], nodes[1:, 1]
    length = np.hypot(x2 - x1, y2 - y1)
    tangent = np.stack([(x2 - x1) / length, (y2 - y1) / length], 1)
    normals = np.stack([-tangent[:, 1], tangent[:, 0]], 1)          # outward for the clockwise airfoil_nodes loop
    mid = np.stack([(x1 + x2) / 2, (y1 + y2) / 2], 1)
    return {"shape": np.asarray(shape, float), "nodes": nodes, "mid": mid, "normals": normals, "length": length,
            "cp": np.asarray(result.cp, float), "cl": float(result.cl_pressure),
            "alpha_deg": float(alpha_deg), "suction_peak": float(result.cp.min())}


def lift_from_cp(cp: np.ndarray, normals: np.ndarray, length: np.ndarray, alpha_deg: float = 0.0) -> float:
    """Lift coefficient of a pressure distribution over the body's own panels.

    The solver's own pressure integration (``panel_solve``: force per unit dynamic pressure ``-(Cp n L)``, resolved
    onto the free-stream axes), applied to a *predicted* ``Cp``, so the coefficient reported for the surrogate is the
    integral of the field it predicted and not a second model.
    """
    cp = np.asarray(cp, float).reshape(-1)
    fx = -(cp * normals[:, 0] * length).sum()
    fy = -(cp * normals[:, 1] * length).sum()
    a = np.radians(alpha_deg)
    return float(-fx * np.sin(a) + fy * np.cos(a))


def design(solved: dict) -> Design:
    """The body as a training sample for the geometry-conditioned model: surface points, normals, query, target."""
    import torch

    mid = torch.tensor(solved["mid"], dtype=torch.float32)
    normals = torch.tensor(solved["normals"], dtype=torch.float32)
    cp = torch.tensor(solved["cp"][:, None], dtype=torch.float32)
    return Design(mid, normals, mid, torch.zeros(0), cp)


def bodies(shapes: list[tuple[str, np.ndarray]], n_per_side: int, alpha_deg: float = 0.0) -> list[dict]:
    """Every body of a split, solved."""
    out = []
    for name, shape in shapes:
        b = body(shape, n_per_side, alpha_deg)
        b["name"] = name
        out.append(b)
    return out


# --------------------------------------------------------------------------------------------- the measurement
def _score(model: LargePhysicsModel, solved: dict, alpha_deg: float) -> dict:
    """Everything measured on one held-out body: the field error, the derived numbers and the geometry distance."""
    pred = model.predict(design(solved))
    cp = pred["mean"].detach().numpy().reshape(-1).astype(float)
    err = cp - solved["cp"]
    rel = float(np.linalg.norm(err) / np.linalg.norm(solved["cp"]))
    cl_pred = lift_from_cp(cp, solved["normals"], solved["length"], alpha_deg)
    props = properties(solved["shape"])
    return {"name": solved["name"], "relative_l2": rel, "rmse": float(np.sqrt(np.mean(err ** 2))),
            "max_abs": float(np.max(np.abs(err))),
            "suction_peak_solver": solved["suction_peak"], "suction_peak_surrogate": float(cp.min()),
            "cl_solver": solved["cl"], "cl_surrogate": cl_pred,
            "ood_score": float(pred["ood_score"]), "ood": bool(pred["ood"]),
            "thickness": props["thickness"], "camber": props["camber"], "le_radius": props["le_radius"],
            "shape": [float(v) for v in solved["shape"]]}


def _train_score(seed: int, p: dict, box: np.ndarray, sets: dict) -> dict:
    """One training seed over the one design set: fit the model, then score every body of it."""
    import torch

    train = bodies(sets["train"], p["panels_per_side"], p["alpha_deg"])
    unseen = bodies(sets["unseen"], p["panels_per_side"], p["alpha_deg"])
    new = bodies(sets["new"], p["panels_per_side"], p["alpha_deg"])

    torch.manual_seed(seed)
    model = LargePhysicsModel(dim=2, n_fields=1, radii=(0.1, 0.4, 1.2), width=p["width"], code=p["code"],
                              n_heads=p["heads"])
    t0 = time.perf_counter()
    history = fit(model, [design(b) for b in train], epochs=p["epochs"], lr=p["lr"], seed=seed)
    train_s = time.perf_counter() - t0

    scored = {"train": [_score(model, b, p["alpha_deg"]) for b in train],
              "unseen": [_score(model, b, p["alpha_deg"]) for b in unseen],
              "new": [_score(model, b, p["alpha_deg"]) for b in new]}
    for group, rows in scored.items():
        for row in rows:
            row["group"] = group
            row["outside_training_box"] = _outside(np.asarray(row["shape"]), box)

    # decision level: "which body carries the most lift", measured against the solver on the bodies never trained on
    from pinneapple_analysis.decision import flip_rate, rank_preservation, regret

    held = scored["unseen"] + scored["new"]
    truth = [r["cl_solver"] for r in held]
    guess = [r["cl_surrogate"] for r in held]
    flip = flip_rate(truth, guess, margin=0.0, lower_is_better=False)
    rank = rank_preservation(truth, guess, lower_is_better=False)
    reg = regret(truth, guess, lower_is_better=False)
    decision = {"n_designs": len(held), "flip_rate": float(flip.flip_rate), "choice_flip": bool(flip.choice_flip),
                "spearman": float(rank.spearman), "kendall_tau": float(rank.kendall_tau),
                "top_k_kept": float(rank.top_k_kept), "regret_relative": float(reg.relative)}
    return {"seed": int(seed), "train_s": train_s, "final_loss": float(history[-1]) if history else None,
            "rows": scored, "decision": decision}


def _merge(per_seed: list[dict]) -> dict[str, list[dict]]:
    """Average the per-seed rows of the same body: the field error of a body is quoted as the mean over seeds."""
    merged: dict[str, list[dict]] = {"train": [], "unseen": [], "new": []}
    for group, rows in merged.items():
        by_name: dict[str, list[dict]] = {}
        for run_i in per_seed:
            for row in run_i["rows"][group]:
                by_name.setdefault(row["name"], []).append(row)
        for name, variants in by_name.items():
            first = dict(variants[0])
            rel = [v["relative_l2"] for v in variants]
            first["relative_l2"] = float(np.mean(rel))
            first["relative_l2_seeds"] = [float(v) for v in rel]
            first["rmse"] = float(np.mean([v["rmse"] for v in variants]))
            first["max_abs"] = float(np.mean([v["max_abs"] for v in variants]))
            first["suction_peak_surrogate"] = float(np.mean([v["suction_peak_surrogate"] for v in variants]))
            first["cl_surrogate"] = float(np.mean([v["cl_surrogate"] for v in variants]))
            first["ood_score"] = float(np.mean([v["ood_score"] for v in variants]))
            rows.append(first)
    return merged


def run(protocol: dict | None = None, *, write: bool = False, out_path: str | None = None) -> dict:
    """Train the geometry-conditioned surrogate on the family and measure it on bodies it never saw.

    ``protocol`` overrides :data:`PROTOCOL` (``QUICK`` is the test budget); ``write`` stores the result as JSON next
    to the other benchmark outputs and the labelled designs as a CSV. Every seed of ``protocol["seeds"]`` is trained
    on the same design set, so the spread between seeds is a property of the training and not of the geometries.
    """
    p = dict(PROTOCOL, **(protocol or {}))
    box = _train_box(p["box_shrink"])
    t0 = time.perf_counter()
    sets = split(p)
    label_s = time.perf_counter() - t0
    per_seed = [_train_score(seed, p, box, sets) for seed in p["seeds"]]
    merged = _merge(per_seed)

    def group_block(group: str) -> dict:
        rel = [row["relative_l2"] for row in merged[group]]
        seed_means = [float(np.mean([r["relative_l2"] for r in run_i["rows"][group]])) for run_i in per_seed]
        return {"n": len(merged[group]), "relative_l2_mean": float(np.mean(seed_means)),
                "relative_l2_seed_range": [float(np.min(seed_means)), float(np.max(seed_means))],
                "relative_l2_max": float(np.max(rel)), "rows": merged[group]}

    new_rows = merged["new"]
    result = {
        "protocol": {k: (list(v) if isinstance(v, tuple) else v) for k, v in p.items()},
        "training_box": {"shrink": p["box_shrink"],
                         "bounds": {n: [float(box[i, 0]), float(box[i, 1])] for i, n in enumerate(PARAMS)},
                         "labels": dict(LABELS)},
        "cost": {"labels_s": label_s, "train_s": float(np.mean([r["train_s"] for r in per_seed])),
                 "final_loss": float(np.mean([r["final_loss"] for r in per_seed]))},
        "train": group_block("train"),
        "unseen": group_block("unseen"),
        # the number the issue asks for: bodies outside the training set, and the error on each of them
        "new_geometry": group_block("new"),
        "decision": {"objective": "lift coefficient at alpha = 0 (higher is better)",
                     "n_designs": per_seed[0]["decision"]["n_designs"],
                     **{k: float(np.mean([r["decision"][k] for r in per_seed]))
                        for k in ("flip_rate", "spearman", "kendall_tau", "top_k_kept", "regret_relative")},
                     "choice_flip_seeds": [bool(r["decision"]["choice_flip"]) for r in per_seed]},
    }
    if write:
        path = out_path or OUT
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=1)
        result["files"] = {"json": path, "designs": _write_designs(path, merged["train"] + merged["unseen"]
                                                                   + new_rows)}
    return result


def _write_designs(result_path: str, rows: list[dict]) -> str:
    """The labelled design table (one row per body): CST weights, geometry descriptors, solver and surrogate."""
    path = os.path.splitext(result_path)[0] + "_designs.csv"
    fields = ["name", "group", *PARAMS, "thickness", "camber", "le_radius", "suction_peak_solver", "cl_solver",
              "cl_surrogate", "relative_l2", "ood_score"]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({**row, **{n: row["shape"][i] for i, n in enumerate(PARAMS)}})
    return path


def table(result: dict) -> str:
    """The rows of the report: one per held-out body, with what makes it new."""
    lines = ["| body | group | outside the training box | relative L2 | Cp min (solver -> surrogate) | Cl (solver -> "
             "surrogate) | OOD |", "|---|---|---|---|---|---|---|"]
    for row in result["unseen"]["rows"] + result["new_geometry"]["rows"]:
        outside = ", ".join(row["outside_training_box"]) or "-"
        lines.append(f"| {row['name']} | {row['group']} | {outside} | {row['relative_l2']:.4f} | "
                     f"{row['suction_peak_solver']:.3f} -> {row['suction_peak_surrogate']:.3f} | "
                     f"{row['cl_solver']:.3f} -> {row['cl_surrogate']:.3f} | {row['ood_score']:.2f} |")
    return "\n".join(lines)


def _fmt_block(block: dict) -> str:
    lo, hi = block["relative_l2_seed_range"]
    return f"mean {block['relative_l2_mean']:.4f} [{lo:.4f}, {hi:.4f}], worst body {block['relative_l2_max']:.4f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Geometry-conditioned surrogate: the error on a body never trained on.")
    ap.add_argument("--quick", action="store_true", help="the test budget (a minute); not a benchmark")
    ap.add_argument("--no-write", action="store_true", help="print the report without writing the results file")
    args = ap.parse_args(argv)
    result = run(QUICK if args.quick else None, write=not args.no_write, out_path=None if not args.quick else
                 os.path.join(os.path.dirname(OUT), "geometry_conditioned_surrogate_quick.json"))
    print(table(result))
    print()
    seeds = ", ".join(str(s) for s in result["protocol"]["seeds"])
    print(f"trained on {result['train']['n']} bodies, seeds {seeds}: train relative L2 "
          f"{_fmt_block(result['train'])}")
    print(f"unseen bodies  : {_fmt_block(result['unseen'])}")
    print(f"new geometries : {_fmt_block(result['new_geometry'])}")
    d = result["decision"]
    print(f"decision       : flip rate {d['flip_rate']:.3f}, Spearman {d['spearman']:.3f}, regret "
          f"{d['regret_relative']:.3f} of the spread over {d['n_designs']} bodies")
    print(f"cost           : labels {result['cost']['labels_s']:.2f} s, training {result['cost']['train_s']:.1f} s "
          f"per seed")
    if "files" in result:
        print(f"wrote {result['files']['json']} and {result['files']['designs']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
