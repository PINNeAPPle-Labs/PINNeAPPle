"""Do the LPMs work? One table, the same gates, for every model that can actually answer on this corpus.

The campaign's cards already say what each model scored on its own held-out split, and a held-out split of our own
data is the weakest evidence there is: the bodies, the reference and the split all came from here. This harness puts
two questions in one table instead.

*On our data* — the model against the constant field, on the split the view recorded, so the number is the one the
card reports, recomputed from the files rather than quoted from the card.

*On somebody else's data* — the same model asked about geometry it never saw, from a corpus solved and published by
its authors (Luminary's SHIFT datasets, through :mod:`pinneapple_neural.lpm_stages.shift`), under the same gates.

    python -m benchmarks.lpm_compare --run ~/lpm-campaign/pp_wing3d
    python -m benchmarks.lpm_compare --run ~/lpm-campaign/pp_wing3d \\
        --shift ~/data/luminary-shift/Wing-sample --limit 2 --condition alpha_deg=alpha_deg

Four gates per row, and the two that matter are the first two: **accuracy** against the family's own tolerance, and
**baseline** against the constant field fitted on the training designs. The other three (conservation, boundary,
residual) need a structured bundle with quadrature weights and are reported as not applicable with that reason rather
than quietly dropped — a point cloud on a body's surface cannot carry them.

What this harness will **not** do is print a number for a model it could not run. PhysicsNeMo's DoMINO, GeoTransolver
and X-MeshGraphNet are available in this environment (see ``pinneapple_data.ashton.physicsnemo_models``), but their
inputs are a signed-distance grid and a mesh bundle built from an Ashton case, not a point-cloud design, and the
ashton package already scores them there against the same metric. Their row here says so and names the module that
runs them; inventing a wrapper to make the table look fuller would be the opposite of what the table is for.
"""
from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch

__all__ = ["Candidate", "ConstantField", "LPMMember", "compare", "table"]


class Candidate(Protocol):
    """Anything that can answer on a :class:`~pinneapple_neural.lpm.Design` in **physical** units."""

    name: str
    budget: str

    def predict(self, design: Any) -> np.ndarray:  # (Q, f)
        ...


@dataclass
class LPMMember:
    """A fitted :class:`~pinneapple_neural.lpm.LargePhysicsModel` as a candidate.

    ``predict`` undoes the field scale exactly as :func:`pinneapple_neural.lpm_pipeline.evaluate` does, so a number in
    this table is comparable with the one in the card.
    """

    model: Any
    family: Any
    name: str = "lpm"
    budget: str = "a few hundred epochs on one machine"
    reports: list[dict[str, Any]] = field(default_factory=list)

    def predict(self, design: Any) -> np.ndarray:
        scales = torch.tensor(self.family.scales, dtype=torch.float32)
        with torch.no_grad():
            report = self.model.predict(design)
        self.reports.append({"ood": bool(report["ood"]), "ood_score": float(report["ood_score"]),
                             "std": float(report["std"].mean())})
        return (report["mean"].detach().cpu() * scales).numpy()


@dataclass
class ConstantField:
    """The mean target vector of the training designs: what any model has to beat to be worth its cost."""

    value: np.ndarray
    name: str = "constant field"
    budget: str = "no training"

    def predict(self, design: Any) -> np.ndarray:
        n = int(design.target.shape[0])
        return np.repeat(self.value.reshape(1, -1), n, axis=0)


@dataclass
class TrainingMember:
    """A stage-5 backend (DoMINO today) as a candidate: it answers in the target's units, this puts it in physical."""

    member: Any
    family: Any
    name: str = "DoMINO"
    budget: str = ""

    def predict(self, design: Any) -> np.ndarray:
        return np.asarray(self.member.predict(design), dtype=np.float64) * np.asarray(self.family.scales).ravel()

    def describe(self) -> dict[str, Any]:
        return self.member.describe()


@dataclass
class Unavailable:
    """A model that exists and cannot answer on this corpus: the row says which, and why, instead of a number."""

    name: str
    reason: str
    budget: str = "-"

    def predict(self, design: Any) -> np.ndarray:
        raise RuntimeError(f"{self.name} cannot answer on this corpus: {self.reason}")


# ---------------------------------------------------------------------------------------------- the scoring
def _relative_l2(pred: np.ndarray, truth: np.ndarray) -> float:
    denom = float(np.linalg.norm(truth))
    return float(np.linalg.norm(pred - truth)) / denom if denom > 0 else float("nan")


@dataclass
class Row:
    """One model's line: the error, the baseline beside it, and the gates that ran."""

    model: str
    relative_l2: float | None = None
    p90: float | None = None
    worst: float | None = None
    baseline: float | None = None
    n_designs: int = 0
    designs_lost_to_baseline: int = 0
    tolerance: float | None = None
    budget: str = ""
    note: str = ""

    @property
    def within_tolerance(self) -> bool:
        return self.relative_l2 is not None and self.tolerance is not None and self.relative_l2 <= self.tolerance

    @property
    def beats_baseline(self) -> bool:
        return self.relative_l2 is not None and self.baseline is not None and self.relative_l2 < self.baseline

    def gates(self) -> dict[str, str]:
        """The five checks: two that ran, three that this corpus cannot carry."""
        if self.relative_l2 is None:
            return {"accuracy": "not run", "baseline": "not run"}
        out = {"accuracy": "passed" if self.within_tolerance else "FAILED",
               "baseline": "passed" if self.beats_baseline else "FAILED"}
        out["conservation"] = out["boundary"] = out["residual"] = "not applicable (point cloud, no quadrature)"
        return out

    def as_dict(self) -> dict[str, Any]:
        return {"model": self.model, "relative_l2": self.relative_l2, "p90": self.p90, "worst": self.worst,
                "baseline": self.baseline, "n_designs": self.n_designs,
                "designs_lost_to_baseline": self.designs_lost_to_baseline, "tolerance": self.tolerance,
                "budget": self.budget, "note": self.note, "gates": self.gates()}


def compare(candidates: Sequence[Candidate], validation: Sequence[Any], *, train_designs: Sequence[Any],
            tolerance: float, scales: Any = None) -> list[Row]:
    """Every candidate on every design, with the constant field computed once from the training targets.

    Everything is compared in the family's **physical** units: a dataset stores its target divided by
    ``family.scales`` and the model answers in those same divided units, so the scale is undone on both sides here —
    and ``predict`` must answer physical too (see :class:`LPMMember`), otherwise a model in the wrong units is
    visibly wrong rather than quietly rescaled into agreement.
    """
    if not validation:
        raise ValueError("no validation designs: there is nothing to score")
    scale = np.asarray(1.0 if scales is None else scales, dtype=np.float64).ravel()
    stacked = torch.cat([d.target for d in train_designs if d.target is not None])
    constant = ConstantField(stacked.mean(0).numpy() * scale)

    truths = [d.target.numpy() * scale for d in validation]
    baselines = [_relative_l2(constant.predict(d), np.asarray(t, dtype=np.float64))
                 for d, t in zip(validation, truths, strict=True)]
    squared_norm = sum(float(np.sum(t ** 2)) for t in truths)
    squared_baseline = sum(float(np.sum((constant.predict(d) - t) ** 2))
                           for d, t in zip(validation, truths, strict=True))
    rows: list[Row] = []
    for candidate in candidates:
        errors, squared_error = [], 0.0
        for design, truth in zip(validation, truths, strict=True):
            pred = np.asarray(candidate.predict(design), dtype=np.float64)
            truth = np.asarray(truth, dtype=np.float64)
            errors.append(_relative_l2(pred, truth))
            squared_error += float(np.sum((pred - truth) ** 2))
        finite = [e for e in errors if math.isfinite(e)]
        rows.append(Row(
            model=candidate.name,
            # the global number is what the cards report and what the tolerance is about; the per-design quantiles
            # are the spread beside it, and on a family whose designs differ in target norm the two are far apart
            relative_l2=math.sqrt(squared_error / squared_norm) if squared_norm > 0 else None,
            p90=float(np.quantile(finite, 0.9)) if finite else None,
            worst=float(np.max(finite)) if finite else None,
            baseline=math.sqrt(squared_baseline / squared_norm) if squared_norm > 0 else None,
            n_designs=len(validation),
            designs_lost_to_baseline=sum(1 for e, b in zip(errors, baselines, strict=True) if e >= b),
            tolerance=tolerance,
            budget=candidate.budget,
            note=getattr(candidate, "reason", ""),
        ))
    return rows


def table(rows: Sequence[Row], *, title: str = "", corpus: str = "") -> str:
    """The table a person reads: the error, the baseline, and the verdict on each gate."""
    head = f"{'model':28s} {'held-out':>9s} {'p90':>7s} {'worst':>7s} {'baseline':>9s} {'beaten':>7s}  gates"
    lines = [f"corpus: {corpus}"] if corpus else []
    if title:
        lines.append(title)
    lines += [head, "-" * len(head)]
    for row in rows:
        if row.relative_l2 is None:
            lines.append(f"{row.model:28s} {'-':>9s} {'-':>7s} {'-':>7s} {'-':>9s} {'-':>7s}  {row.note[:60]}")
            continue
        gates = row.gates()
        verdict = f"acc {'ok' if gates['accuracy'] == 'passed' else 'FAIL'} / base " \
                  f"{'ok' if gates['baseline'] == 'passed' else 'FAIL'}"
        lines.append(f"{row.model:28s} {row.relative_l2:9.3f} {row.p90 or 0:7.3f} {row.worst or 0:7.3f} "
                     f"{row.baseline or 0:9.3f} {row.designs_lost_to_baseline:7d}  {verdict}")
    lines.append(f"tolerance: {rows[0].tolerance:g} (the family's own)" if rows and rows[0].tolerance else "")
    lines.append("conservation/boundary/residual: not applicable on a surface point cloud (no quadrature weights, "
                 "no declared field equation) — see pinneapple_veriphysics.surrogate_gates for the structured path")
    return "\n".join(line for line in lines if line != "")


# ---------------------------------------------------------------------------------------------- the two corpora
def family_by_name(name: str) -> Any:
    """A family by name, 2-D and 3-D alike."""
    from pinneapple_neural.lpm_families_3d import families_3d
    from pinneapple_neural.lpm_pipeline import families

    known = {**families(), **families_3d()}
    if name not in known:
        raise ValueError(f"unknown family {name!r}; known: {', '.join(sorted(known))}")
    return known[name]


def _split_from_view(view_path: Path, designs: list[Any], manifest: dict[str, Any]) -> tuple[list[Any], list[Any]]:
    """Train and held-out designs on the split the training run recorded, not on a freshly drawn one."""
    view = json.loads(view_path.read_text(encoding="utf-8"))
    held = {d["label"] for d in view.get("designs", []) if d.get("held_out")}
    if not held:
        raise ValueError(f"{view_path}: no held-out designs recorded; score the run that produced it instead")
    train, hold = [], []
    for design, record in zip(designs, manifest["designs"], strict=True):
        (hold if record.get("label") in held else train).append(design)
    if not hold or not train:
        raise ValueError(f"{view_path}: the recorded split does not match the dataset (train {len(train)}, "
                         f"held out {len(hold)})")
    return train, hold


def campaign_corpus(run_dir: str | Path) -> dict[str, Any]:
    """A fitted run directory: the model, the family, the designs on the split its own view recorded."""
    from pinneapple_neural.lpm_pipeline import load_dataset, load_model

    root = Path(run_dir)
    dataset_dir = root / "dataset" if (root / "dataset").is_dir() else root
    designs, manifest = load_dataset(dataset_dir)
    family = family_by_name(manifest["family"]["name"])
    train, hold = _split_from_view(root / "model" / "view.json", designs, manifest)
    return {"family": family, "model": load_model(root), "train": train, "validation": hold,
            "corpus": f"{manifest['family']['name']} held-out split ({len(hold)} designs, split by geometry)",
            "tolerance": float(manifest["family"].get("tolerance", family.tolerance))}


def shift_corpus(shift_root: str | Path, family: Any, model: Any, condition_map: dict[str, str], *,
                 train_designs: Sequence[Any], dataset_name: str | None, limit: int | None = None,
                 max_points: int = 2048, field_name: str | None = None) -> dict[str, Any]:
    """A SHIFT tree as the validation set: never trained on, and scored against the constant field of our own data."""
    from pinneapple_neural.lpm_stages import shift

    validation = shift.designs(shift_root, family, condition_map, dataset_name=dataset_name, field=field_name,
                               max_points=max_points, limit=limit)
    if not validation:
        raise ValueError(f"{shift_root}: no case directory holds {shift.CASE_SURFACE}; nothing to validate on")
    source = validation[0].source
    return {"family": family, "model": model, "train": list(train_designs), "validation": validation,
            "corpus": f"{source['dataset']} at {Path(shift_root).name} — an outside corpus, {len(validation)} case(s), "
                      f"{source['licence']}, {source['usage']} only",
            "tolerance": float(family.tolerance)}


def physicsnemo_rows(compared: Sequence[str] = ()) -> list[Unavailable]:
    """The PhysicsNeMo baselines this run did *not* score, with why. A model already in the table is not listed."""
    for prefix in ("DoMINO", "GeoTransolver"):
        if any(name.startswith(prefix) for name in compared):
            compared = (*compared, prefix)
    try:
        from pinneapple_data.ashton.physicsnemo_models import availability
    except ImportError:                       # the ashton reader is not on this checkout
        return [Unavailable(name, "pinneapple_data.ashton is not on this checkout")
                for name in ("DoMINO", "GeoTransolver", "X-MeshGraphNet") if name not in compared]
    state = availability()
    point_cloud = ("its inputs are a signed-distance grid and a mesh bundle built from a case file, not a point-cloud "
                   "design; pinneapple_data.ashton.physicsnemo_models scores it on the Ashton cases against the same "
                   "metric")
    rows = [Unavailable("DoMINO", point_cloud) if state.get("domino") else
            Unavailable("DoMINO", "not importable here: physicsnemo is not installed"),
            Unavailable("GeoTransolver",
                        "not importable here: physicsnemo is not installed") if not state.get("geotransolver") else
            Unavailable("GeoTransolver", point_cloud)]
    rows.append(Unavailable("X-MeshGraphNet", point_cloud) if state.get("x_meshgraphnet") else
                Unavailable("X-MeshGraphNet", state.get("reasons", {}).get("x_meshgraphnet", "not importable here")))
    return [row for row in rows if row.name not in compared]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, help="a fitted run directory (the one holding model/ and dataset/)")
    parser.add_argument("--shift", default="", help="a SHIFT tree on disk; scored as validation, never trained on")
    parser.add_argument("--dataset-name", default="", help="the SHIFT repository name (inferred from the path)")
    parser.add_argument("--condition", action="append", default=[],
                        help="ours=theirs, mapping a family parameter onto a key of params.json")
    parser.add_argument("--field", default=None, help="the surface array to score against (default: first match)")
    parser.add_argument("--limit", type=int, default=None, help="cases to read from the SHIFT tree")
    parser.add_argument("--max-points", type=int, default=2048, help="points per case (the neighbourhood is O(PQ))")
    parser.add_argument("--designs", type=int, default=None, help="cap the held-out designs (a quick look)")
    parser.add_argument("--train-designs", type=int, default=120,
                        help="cap the designs a PhysicsNeMo baseline is fitted on (default 120: a full wing family "
                             "is 900+ bodies, and its inputs are a grid per design, not a point cloud)")
    parser.add_argument("--domino-steps", type=int, default=0,
                        help="also train DoMINO on the training designs for this many steps (slow; 0 = skip)")
    parser.add_argument("--geotransolver-steps", type=int, default=0,
                        help="also train GeoTransolver on the training designs for this many steps (0 = skip)")
    parser.add_argument("--json", default="", help="write the rows here as JSON")
    args = parser.parse_args(argv)

    corpus = campaign_corpus(args.run)
    family, model = corpus["family"], corpus["model"]
    if args.designs and args.designs < len(corpus["validation"]):
        # evenly spaced, not the first N: a design file is ordered by geometry, so a prefix is a biased sample and
        # would report a different number from the card without anything being wrong
        pick = np.unique(np.linspace(0, len(corpus["validation"]) - 1, args.designs).round().astype(int))
        corpus["validation"] = [corpus["validation"][i] for i in pick]
    candidates: list[Candidate] = [LPMMember(model, family)]
    if args.shift:
        condition_map = dict(pair.split("=", 1) for pair in args.condition)
        unmapped = [name for name in family.conditioning_params if name not in condition_map]
        if unmapped:
            print(f"{family.name} conditions on {len(family.conditioning_params)} parameters and the map names "
                  f"{len(condition_map)} of them; {unmapped} would have to be read off the SHIFT case, and a design "
                  f"with half its inputs invented is not a validation of anything. Map them with --condition "
                  f"ours=theirs, or score a family whose inputs the corpus states.")
            return 2
        try:
            outside = shift_corpus(args.shift, family, model, condition_map, train_designs=corpus["train"],
                                   dataset_name=args.dataset_name or None, limit=args.limit,
                                   max_points=args.max_points, field_name=args.field)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            print(f"the SHIFT corpus is not readable as {family.name}: {exc.args[0] if exc.args else exc}")
            return 2
        corpus["validation"] = outside["validation"]
        corpus["corpus"] = outside["corpus"]

    if args.domino_steps or args.geotransolver_steps:
        from pinneapple_neural.lpm_stages import training

        training_designs = corpus["train"]
        cap = int(args.train_designs or 0)
        if cap and len(training_designs) > cap:
            # evenly spaced, like the held-out cap: the design files are ordered by geometry, so a prefix is a
            # biased slice of the space and the baseline would be measured on the easy end of it
            pick = np.unique(np.linspace(0, len(training_designs) - 1, cap).round().astype(int))
            training_designs = [corpus["train"][i] for i in pick]
        first = training_designs[0]
        shape = {"n_params": int(first.params.numel()), "n_fields": int(first.target.shape[-1])}
        for steps, cls in ((args.domino_steps, training.DoMINOMember),
                           (args.geotransolver_steps, training.GeoTransolverMember)):
            if not steps:
                continue
            member = cls(**shape)
            member.fit(training_designs, steps=steps)
            member.notes.append(f"fitted on {len(training_designs)} of the {len(corpus['train'])} training designs "
                                f"(the cap is --train-designs {cap})")
            candidates.append(TrainingMember(member, family, name=member.name, budget=member.budget))
            for note in member.describe()["notes"]:
                print(f"note: {note}")

    rows = compare(candidates, corpus["validation"], train_designs=corpus["train"],
                   tolerance=corpus["tolerance"], scales=family.scales)
    print(table(rows, corpus=corpus["corpus"]))
    ood = [r for member in candidates if isinstance(member, LPMMember) for r in member.reports]
    if ood:
        flagged = sum(1 for r in ood if r["ood"])
        print(f"ood flag: {flagged}/{len(ood)} design(s) flagged outside the trained family "
              f"(mean raw score {np.mean([r['ood_score'] for r in ood]):.2f}); on an outside corpus that flag firing "
              f"is the model saying the answer is an extrapolation")
    unavailable = physicsnemo_rows([getattr(candidate, "name", "") for candidate in candidates])
    if unavailable:
        print("\nnot compared here:")
        for row in unavailable:
            print(f"  {row.name:22s} {row.reason}")
    if args.json:
        Path(args.json).write_text(json.dumps({"corpus": corpus["corpus"], "rows": [r.as_dict() for r in rows],
                                               "not_compared": [{"model": u.name, "reason": u.reason}
                                                                for u in unavailable]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
