"""Stage 4 of :mod:`pinneapple_neural.lpm_pipeline`: the fit — split **by geometry**, train, and report what the split does and does not say.

Two operating points of one body are the same body, so the split is by geometry and never by design: a model scored
on a body it trained on at another incidence reads as generalisation without being it. The split is fixed by the seed
and written into the card, and a dataset's **refined** geometries (stage 1's second round) are pinned into training —
a body was simulated *because* the model was unsure about it, so it is evidence about the sampler, not about the model.

What the fit measures, and what makes the number usable: the held-out error against the **constant field** baseline,
the ensemble spread, the calibrated confidence, and the out-of-distribution report. :func:`train` also writes
``model/card.json`` (schema ``pinneapple.lpm.family_card/1``) with all of it, and ``model/view.json``, which is what
the panels read.

The registry of design spaces lives in :mod:`pinneapple_neural.lpm_pipeline`, which re-exports this module, so the
import back happens inside :func:`_family`.
"""
from __future__ import annotations

import json
import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from ..lpm import Design, LargePhysicsModel, fit
from .dataset import _progress, load_dataset

if TYPE_CHECKING:                                       # the annotations are strings; this is for the readers of them
    from ..lpm_pipeline import FieldFamily

CARD_SCHEMA = "pinneapple.lpm.family_card/1"
VIEW_SCHEMA = "pinneapple.lpm.view/1"
TRUST = "not_verified_until_independent_review"

__all__ = ["CARD_SCHEMA", "TRUST", "VIEW_SCHEMA", "evaluate", "load_model", "train"]


# ---------------------------------------------------------------------------------------------------------- stage 4


def _split_by_geometry(manifest: dict[str, Any], holdout_fraction: float,
                       seed: int) -> tuple[list[int], list[int]]:
    """Train/hold-out indices **by geometry**: two operating points of the same body never straddle the split.

    Splitting by design instead would let the model see a body during training and be scored on the same body at
    another operating point, which reads as generalisation without being it.
    """
    ids = sorted({record["geometry_id"] for record in manifest["designs"]})
    if len(ids) < 2:
        raise ValueError("a design space needs at least two geometries to hold one out")
    # a refined body was simulated because the model was unsure about it (stage 1), so it is evidence about the
    # sampler, not about the model: it trains and is never held out. The drawn geometries decide the split.
    refined = set(manifest.get("refined_geometries") or ())
    candidates = [gid for gid in ids if gid not in refined]
    if len(candidates) < 2:
        raise ValueError(f"a design space needs at least two drawn geometries to hold one out; this dataset has "
                         f"{len(candidates)} of {len(ids)} ({len(refined)} refined)")
    order = np.random.default_rng(seed).permutation(len(candidates))
    n_hold = min(max(1, round(len(candidates) * holdout_fraction)), len(candidates) - 1)
    held = {candidates[i] for i in order[:n_hold]}
    train = [i for i, r in enumerate(manifest["designs"]) if r["geometry_id"] not in held]
    hold = [i for i, r in enumerate(manifest["designs"]) if r["geometry_id"] in held]
    return train, hold


def _relative_l2(pred: torch.Tensor, target: torch.Tensor) -> float:
    error = float(torch.linalg.vector_norm(pred - target))
    norm = float(torch.linalg.vector_norm(target))
    return error / norm if norm > 0 else float("nan")


def _warm_start(model: LargePhysicsModel, source: str | Path, expected: dict[str, Any]) -> None:
    """Copy a fitted run's weights into ``model``, refusing a source built for a different architecture.

    A checkpoint is a state dict plus the config it was built from; loading it into a model of another width either
    raises from ``load_state_dict`` with a shape nobody can read, or — worse — succeeds for the layers that happen to
    match. Comparing the config first turns that into one sentence.
    """
    path = Path(source) / "model" / "model.pt"
    if not path.is_file():
        raise FileNotFoundError(f"init_from: {path} is not a fitted run (it has no model/model.pt)")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = payload.get("config") or {}
    theirs = {"dim": int(config.get("dim", expected["dim"])),
              "n_params": int(config.get("n_params", expected["n_params"])),
              "radii": list(config.get("radii") or expected["radii"])}
    mismatched = [k for k in ("dim", "n_params", "radii") if expected[k] != theirs[k]]
    if mismatched:
        detail = ", ".join(f"{k}: {expected[k]} != {theirs[k]}" for k in mismatched)
        raise ValueError(f"init_from {path} was fitted for another design space ({detail}); warm-starting across a "
                         f"different conditioning width or point dimension mixes two models rather than transferring "
                         f"one")
    state = payload["state_dict"]
    if "train_codes" in state:
        # the codes travel with the weights and the fit replaces them with this dataset's, so the buffer is only
        # sized here to make the load succeed
        model.train_codes = state["train_codes"].clone()
    model.load_state_dict(state)


def load_model(out_dir: str | Path, device: str | None = None) -> LargePhysicsModel:
    """Load a model written by :func:`train`, so a fitted design space can be used without the pipeline.

    The training geometry codes travel in the checkpoint, so the out-of-distribution rule still measures against the
    bodies the model was actually fitted on.
    """
    payload = torch.load(Path(out_dir) / "model" / "model.pt", map_location="cpu", weights_only=False)
    config = payload["config"]
    model = LargePhysicsModel(
        dim=int(config["dim"]), n_fields=int(config["n_fields"]), n_params=int(config["n_params"]),
        radii=tuple(config["radii"]), n_freq=int(config["n_freq"]), width=int(config["width"]),
        code=int(config["code"]), n_heads=int(config["n_heads"]), backbone=str(config["backbone"]),
        attn_heads=int(config["attn_heads"]), attn_layers=int(config["attn_layers"]),
    )
    state = payload["state_dict"]
    if "train_codes" in state:
        # the remembered codes are a buffer whose size is the number of distinct training bodies, so it has to be
        # sized from the checkpoint before the state is copied into it
        model.train_codes = state["train_codes"].clone()
    model.load_state_dict(state)
    model = model.to(torch.device(device) if device is not None else torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    return model


def evaluate(model: LargePhysicsModel, family: FieldFamily, designs: list[Design],
             train_designs: list[Design] | None = None) -> dict[str, Any]:
    """Held-out error, ensemble spread, constant-field baseline and the OOD report, over a list of designs.

    Everything here is in the family's physical units: the model fits the scaled field, and the scale is undone
    before any error is computed, so a reported number can be compared with the reference's own units.
    """
    scales = torch.tensor(family.scales, dtype=torch.float32)
    rows, squared_error, squared_normal, base_error, base_normal = [], 0.0, 0.0, 0.0, 0.0
    spreads: list[float] = []
    baseline = None
    if train_designs:
        stacked = torch.cat([d.target * scales for d in train_designs])
        baseline = stacked.mean(0)  # the constant field: the mean target vector of the training designs
    for design in designs:
        report = model.predict(design)
        pred = report["mean"].detach().cpu() * scales
        target = design.target * scales
        error = float(torch.linalg.vector_norm(pred - target))
        norm = float(torch.linalg.vector_norm(target))
        squared_error += error ** 2
        squared_normal += norm ** 2
        if baseline is not None:
            base_error += float(torch.linalg.vector_norm(baseline.expand_as(target) - target)) ** 2
            base_normal += norm ** 2
        std = report["std"].detach().reshape(-1)
        spreads += [float(v) for v in std]
        rows.append({"relative_l2": error / norm if norm > 0 else None,
                     "max_abs_error": float((pred - target).abs().max()),
                     "mean_std": float(std.mean()), "ood": bool(report["ood"]),
                     "ood_score": float(report["ood_score"])})
    errors = sorted(r["relative_l2"] for r in rows if r["relative_l2"] is not None)
    std_t = torch.tensor(spreads) if spreads else torch.zeros(1)
    return {
        "n_designs": len(designs),
        "n_points": sum(int(d.query.shape[0]) for d in designs),
        "relative_l2": math.sqrt(squared_error / squared_normal) if squared_normal > 0 else None,
        "relative_l2_p90": errors[min(int(0.9 * (len(errors) - 1)), len(errors) - 1)] if errors else None,
        "max_abs_error": max(r["max_abs_error"] for r in rows) if rows else None,
        "constant_field_baseline_relative_l2": (math.sqrt(base_error / base_normal)
                                                if base_normal > 0 else None),
        "spread": {"mean_std": float(std_t.mean()), "p95_std": float(torch.quantile(std_t, 0.95)),
                   "max_std": float(std_t.max()),
                   "note": "standard deviation between the bagged ensemble heads, not a confidence level"},
        "per_design": rows,
    }


def _ood_report(model: LargePhysicsModel, family: FieldFamily,
                holdout: list[Design]) -> dict[str, Any]:
    """The OOD flag on the family's own outside-the-family probes and on the held-out designs."""
    probes = []
    from ..lpm_pipeline import build

    for name, spec in family.ood_probes():
        report = model.ood(build(family, spec))
        probes.append({"name": name, "ood": bool(report["ood"]), "ood_score": float(report["ood_score"]),
                       "reasons": [str(r) for r in report["reasons"]]})
    inside = []
    for i, design in enumerate(holdout):
        report = model.ood(design)
        inside.append({"name": f"held-out design {i}", "ood": bool(report["ood"]),
                       "ood_score": float(report["ood_score"])})
    scores = sorted(r["ood_score"] for r in inside)
    return {
        "probes": probes,
        "held_out": inside,
        "probes_flagged": sum(1 for p in probes if p["ood"]),
        "probes_total": len(probes),
        "held_out_flagged": sum(1 for r in inside if r["ood"]),
        "held_out_ood_score_max": scores[-1] if scores else None,
        "held_out_ood_score_median": scores[len(scores) // 2] if scores else None,
        "threshold": "the library's own: geometry-code distance over the 95th percentile of the training "
                     "nearest-neighbour distances, above 1.5x",
        "note": "reported as measured: a probe the model does not flag is not thereby inside its validity domain",
    }


def _stride(n: int, keep: int) -> list[int]:
    """Every ``n/keep``-th index: the shape of a curve without the file becoming the dataset again."""
    if n <= keep:
        return list(range(n))
    step = n / float(keep)
    return [min(n - 1, int(i * step)) for i in range(keep)]


def _view(*, family_info: dict[str, Any], manifest: dict[str, Any], designs: list[Design], train_idx: list[int],
          hold_idx: list[int], history: list[float], held: dict[str, Any], model: LargePhysicsModel,
          scales: torch.Tensor, tolerance: float) -> dict[str, Any]:
    """What a screen draws: the loss curve, the design space and its split, the bodies, and predictions.

    Written next to the card so a figure can be rebuilt without refitting anything. Everything is in the family's
    physical units, and every sample carries the design it came from — a picture of one held-out body is a picture of
    that body, never a claim about a part nobody ran.
    """
    records = manifest["designs"]
    first = records[0] if records else {"geometry": {}, "operating": {}}
    names = [*first.get("geometry", {}), *first.get("operating", {})]
    held_pos = {int(i): k for k, i in enumerate(hold_idx)}

    rows = []
    for i, record in enumerate(records):
        values = {**record.get("geometry", {}), **record.get("operating", {})}
        k = held_pos.get(i)
        rows.append({"label": record.get("label", str(i)), "geometry_id": record.get("geometry_id"),
                     "values": {n: float(values[n]) for n in names if isinstance(values.get(n), (int, float))},
                     "held_out": k is not None,
                     "relative_l2": (held["per_design"][k]["relative_l2"] if k is not None else None)})

    def spread(name: str) -> float:
        vals = [r["values"][name] for r in rows if name in r["values"]]
        if not vals:
            return 0.0
        return (max(vals) - min(vals)) / (abs(max(vals)) + abs(min(vals)) + 1e-12)

    # the two parameters the design space actually varies most, so the scatter is the design space and not two
    # nearly-constant axes with everything piled in a line
    axes = sorted([n for n in names if rows and n in rows[0]["values"]], key=spread, reverse=True)[:2]
    idx = _stride(len(history), 240)

    picks: list[int] = []
    scored = sorted((k for k in range(len(hold_idx)) if held["per_design"][k]["relative_l2"] is not None),
                    key=lambda k: held["per_design"][k]["relative_l2"])
    if scored:                                          # the best, the median and the worst held-out body
        for k in (scored[0], scored[len(scored) // 2], scored[-1]):
            if k not in picks:
                picks.append(k)
    field_names = list(family_info["field_names"])
    chosen = [(f"held-out {k}", hold_idx[k], True) for k in picks]
    if train_idx:
        chosen.append(("training", train_idx[0], False))
    samples = []
    for tag, position, is_held in chosen:
        design = designs[position]
        report = model.predict(design)
        pred = (report["mean"].detach().cpu() * scales).numpy()
        target = (design.target * scales).numpy()
        q = design.query.detach().cpu().numpy()
        pts = design.points.detach().cpu().numpy()
        keep_q, keep_p = _stride(len(q), 240), _stride(len(pts), 240)
        same = bool(q.shape == pts.shape and np.allclose(q, pts))
        samples.append({
            "label": tag, "design": records[position].get("label", str(position)), "held_out": is_held,
            "relative_l2": (held["per_design"][held_pos[position]]["relative_l2"] if is_held else None),
            "ood": bool(report["ood"]),
            # whether the field is evaluated on the body's own points: the 3-D viewer colours the skin only when the
            # values and the vertices are one set, and draws the bare geometry when they are not
            "at_points": same,
            "points": pts[keep_p].tolist(), "query": q[keep_q].tolist(),
            "target": {f: target[keep_q, j].tolist() for j, f in enumerate(field_names)},
            "pred": {f: pred[keep_q, j].tolist() for j, f in enumerate(field_names)},
        })
    return {
        "schema": VIEW_SCHEMA,
        "family": family_info["name"],
        "title": family_info.get("title", family_info["name"]),
        "dim": int(family_info["dim"]),
        "fields": field_names,
        "units": list(family_info.get("field_units", [])),
        "reference": family_info.get("reference", {}),
        "tolerance": float(tolerance),
        "n_designs": len(rows),
        "split": {"n_train": len(train_idx), "n_held_out": len(hold_idx)},
        "params": names,
        "axes": axes,
        "designs": rows,
        "loss": {"epoch": [i + 1 for i in idx], "loss": [float(history[i]) for i in idx]},
        "samples": samples,
        "note": ("every field here came from the family's own reference (see the card); the prediction is the model's "
                 "mean over its bagged heads, and a sample is one body, not a statement about the design space"),
    }


def _family(name: str) -> Any:
    """The registered design space by name.

    The registry lives in :mod:`pinneapple_neural.lpm_pipeline`, which re-exports this module, so the import happens
    here rather than at the top: a module that needs its own importer at import time is a cycle.
    """
    from ..lpm_pipeline import _family_by_name
    return _family_by_name(name)


def train(dataset_dir: str | Path, out_dir: str | Path, *, epochs: int = 400, lr: float = 2e-3,
          holdout_fraction: float = 0.25, seed: int = 0, width: int = 96, n_heads: int = 4,
          code: int = 32, n_freq: int = 6, backbone: str = "attention", attn_layers: int = 2,
          attn_heads: int = 4, batch_designs: int = 32, radii: tuple[float, float, float] = (0.1, 0.4, 1.2),
          device: str | None = None, verbose: bool = False, progress_every_s: float = 0.0,
          resume: bool = False, init_from: str | Path | None = None) -> dict[str, Any]:
    """Fit an LPM on a dataset and write the model and its card.

    The split is by geometry (see :func:`_split_by_geometry`); the reported error is therefore on bodies the model
    never saw, at operating points it never saw. Nothing here decides that the model is good: it measures it and
    says how.

    ``progress_every_s`` writes ``model/progress.json`` at that interval while the fit runs — epoch of how many, the
    loss so far and the time still needed — so a four-hour fit is not one line of output at the end with nothing
    between. ``0``, the default, writes nothing.

    ``resume`` keeps ``model/fit.pt`` while the fit runs (see :func:`~.lpm.fit`), so a run that is killed continues
    from its last checkpoint instead of from epoch one; the file is gone once the fit finishes.

    ``init_from`` warm-starts from a run's ``model/model.pt`` — the fine-tune between design spaces. The weights are
    only meaningful if the architecture and the conditioning width match, so a mismatch is refused with the two
    numbers named rather than loaded into a model whose first layer is a different size.
    """
    designs, manifest = load_dataset(dataset_dir)
    # the dataset caches the neighbourhood features for the radii it was written with; fitting with other radii would
    # make that cache unusable and pay the distance matrix again on every step, so the manifest wins.
    cached = tuple((manifest.get("local") or {}).get("radii") or ())
    if cached and tuple(radii) != cached:
        radii = cached
    family_info = manifest["family"]
    published = _family(family_info["name"]).published_split(manifest["designs"])
    train_idx, hold_idx = published if published is not None else _split_by_geometry(
        manifest, holdout_fraction, seed)
    train_designs = [designs[i] for i in train_idx]
    hold_designs = [designs[i] for i in hold_idx]
    dev = torch.device(device) if device is not None else torch.device(
        "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = LargePhysicsModel(
        dim=int(family_info["dim"]), n_fields=len(family_info["field_names"]),
        n_params=len(family_info["operating_params"]) + len(family_info.get("geometry_features", {})),
        radii=radii, n_freq=n_freq, width=width, code=code,
        n_heads=n_heads, backbone=backbone, attn_heads=attn_heads, attn_layers=attn_layers,
    ).to(dev)
    if init_from is not None:
        _warm_start(model, Path(init_from), expected={"dim": int(family_info["dim"]),
                                                      "n_params": len(family_info["operating_params"])
                                                      + len(family_info.get("geometry_features", {})),
                                                      "radii": list(radii)})
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat(timespec="seconds")
    progress_path = Path(out_dir) / "model" / "progress.json"
    first_loss: list[float] = []
    resumed_from = [epochs]                       # lowered to the first epoch this run actually ran
    last_progress = [started]

    def _on_epoch(epoch: int, loss: float) -> None:
        if not first_loss:
            first_loss.append(loss)
        resumed_from[0] = min(resumed_from[0], epoch - 1)    # a resumed fit reports the rate of this run, not of both
        now = time.monotonic()
        if now - last_progress[0] >= progress_every_s or epoch == epochs:
            _progress(progress_path, kind="fit", family=family_info["name"], started_at=started_at, started=started,
                      done=epoch, total=epochs, loss=round(loss, 8), loss_first=round(first_loss[0], 8),
                      resumed_from=resumed_from[0],
                      n_train_designs=len(train_designs), n_holdout_designs=len(hold_designs), device=str(dev))
            last_progress[0] = now

    if progress_every_s > 0:
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.unlink(missing_ok=True)           # a previous run's record is not this run's progress
        on_epoch = _on_epoch
    else:
        on_epoch = None
    if resume:
        # the first checkpoint is written at the end of epoch one, before anything below creates the model directory:
        # a fresh run directory plus ``resume`` and no progress writer failed there, which is how this was found
        (Path(out_dir) / "model").mkdir(parents=True, exist_ok=True)
    history = fit(model, train_designs, epochs=epochs, lr=lr, seed=seed,
                  batch_designs=min(batch_designs, len(train_designs)), device=dev, on_epoch=on_epoch,
                  checkpoint_dir=(Path(out_dir) / "model") if resume else None)
    seconds = time.monotonic() - started
    if progress_every_s > 0:
        _progress(progress_path, kind="fit", family=family_info["name"], started_at=started_at, started=started,
                  done=epochs, total=epochs, finished=True,
                  loss=round(history[-1], 8) if history else None,
                  loss_first=round(history[0], 8) if history else None,
                  n_train_designs=len(train_designs), n_holdout_designs=len(hold_designs), device=str(dev))
    model.eval()
    held = evaluate(model, _family(family_info["name"]), hold_designs, train_designs)
    ood = _ood_report(model, _family(family_info["name"]), hold_designs)
    tolerance = float(family_info.get("tolerance", 0.05))
    within = held["relative_l2"] is not None and held["relative_l2"] <= tolerance

    out = Path(out_dir)
    (out / "model").mkdir(parents=True, exist_ok=True)
    view = _view(family_info=family_info, manifest=manifest, designs=designs, train_idx=train_idx,
                 hold_idx=hold_idx, history=history, held=held, model=model,
                 scales=torch.tensor(_family(family_info["name"]).scales, dtype=torch.float32),
                 tolerance=tolerance)
    (out / "model" / "view.json").write_text(json.dumps(view), encoding="utf-8")
    config = {"dim": int(family_info["dim"]), "n_fields": len(family_info["field_names"]),
              "n_params": (len(family_info["operating_params"])
                           + len(family_info.get("geometry_features", {}))), "radii": list(radii), "n_freq": n_freq,
              "width": width, "code": code, "n_heads": n_heads, "backbone": backbone,
              "attn_heads": attn_heads, "attn_layers": attn_layers}
    torch.save({"schema": CARD_SCHEMA, "config": config, "state_dict": model.state_dict()},
               out / "model" / "model.pt")
    card = {
        "schema": CARD_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "family": family_info,
        "dataset": {"schema": manifest["schema"], "seed": manifest["seed"],
                    "n_designs": manifest["n_designs"], "n_geometries": manifest["n_geometries"],
                    "conditions_per_geometry": manifest["conditions_per_geometry"],
                    # stage 1's second round, when there was one: which bodies were simulated because the model was
                    # unsure, and that they train only. A card without this key was fitted on the drawn space alone.
                    **({"refinement": manifest["refinement"],
                        "refined_geometries": manifest["refined_geometries"]}
                       if manifest.get("refinement") else {})},
        "split": {"policy": ("the published split of the dataset: a held-out body was held out before training"
                             if published is not None else
                             "random by geometry, fixed by the seed: a held-out body is never a training body"),
                  "seed": int(seed), "holdout_fraction": float(holdout_fraction),
                  "n_train_designs": len(train_designs), "n_holdout_designs": len(hold_designs),
                  "among_refined": sorted({manifest["designs"][i]["geometry_id"] for i in train_idx}
                                          & set(manifest.get("refined_geometries") or ())),
                  "n_train_geometries": len({manifest["designs"][i]["geometry_id"] for i in train_idx}),
                  "n_holdout_geometries": len({manifest["designs"][i]["geometry_id"] for i in hold_idx})},
        "model": {"class": "LargePhysicsModel", "config": config,
                  "parameters": sum(p.numel() for p in model.parameters())},
        "training": {"epochs": epochs, "lr": lr, "seed": seed, "batch_designs": min(batch_designs, len(train_designs)),
                     "loss_first": history[0] if history else None, "loss_last": history[-1] if history else None,
                     "seconds": round(seconds, 3), "device": str(dev)},
        "held_out": {**{k: held[k] for k in ("relative_l2", "relative_l2_p90", "max_abs_error",
                                             "constant_field_baseline_relative_l2", "n_designs", "n_points")},
                     "per_design": held["per_design"], "tolerance": tolerance, "within_tolerance": within},
        "ensemble_spread": held["spread"],
        "ood": ood,
        "trust": {"decision": TRUST,
                  "measured_here": [
                      f"held-out relative L2 {held['relative_l2']:.4g} on {held['n_designs']} designs of "
                      f"{held['n_points']} points, tolerance {tolerance:g}" if held["relative_l2"] is not None
                      else "held-out error not computed",
                      f"constant-field baseline on the same designs: "
                      f"{held['constant_field_baseline_relative_l2']:.4g}"
                      if held["constant_field_baseline_relative_l2"] is not None else "baseline not computed",
                      f"OOD: {ood['probes_flagged']}/{ood['probes_total']} probes outside the family flagged, "
                      f"{ood['held_out_flagged']}/{len(ood['held_out'])} held-out designs flagged",
                  ],
                  "not_measured": [
                      f"the reference itself: {family_info['reference']['implements']} — "
                      f"{family_info['reference']['name']}, with the assumptions it lists",
                      "agreement with a solver or with measured data for any geometry of this family",
                      "whether the ensemble spread is calibrated: it is the spread between bootstrap heads",
                      "behaviour outside the declared geometry and operating bounds of the family",
                  ]},
    }
    (out / "model" / "card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    summary = {
        "family": family_info["name"], "field_names": family_info["field_names"],
        "held_out_relative_l2": held["relative_l2"], "held_out_relative_l2_p90": held["relative_l2_p90"],
        "constant_field_baseline_relative_l2": held["constant_field_baseline_relative_l2"],
        "ensemble_std_mean": held["spread"]["mean_std"],
        "ood_probes_flagged": f"{ood['probes_flagged']}/{ood['probes_total']}",
        "within_tolerance": within, "seconds": round(seconds, 1), "device": str(dev),
        "model": str(out / "model" / "model.pt"), "card": str(out / "model" / "card.json"),
    }
    if verbose:
        print(json.dumps(summary, indent=2))
    return summary
