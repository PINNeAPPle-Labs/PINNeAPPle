"""The LPM campaign in one command: one dataset and one model per design space.

The pipeline (``pinneapple_neural.lpm_pipeline``) knows how to turn one family into a dataset and a model. This script
knows *which* family, how big, in what order, and how to keep going when one of them fails — which is what a campaign
needs at 3 a.m.

    python -m benchmarks.lpm_campaign --plan                  # what would run, and roughly how long
    python -m benchmarks.lpm_campaign --scale smoke           # the whole wiring, in about two minutes
    python -m benchmarks.lpm_campaign --stage dataset         # the data only
    python -m benchmarks.lpm_campaign --stage train           # the fits only
    python -m benchmarks.lpm_campaign --max-hours 6           # a night: stop before the next family
    python -m benchmarks.lpm_campaign --detach campaign.log   # survives the shell (and the editor session)
    python -m benchmarks.lpm_monitor --out benchmarks/_out/lpm_campaign   # the screen that follows it

What it guarantees, because a campaign that has to be watched is not a campaign:

* **one at a time.** The families run in sequence in this process: one design space, one fit, one device. Nothing is
  parallelised, because the machine that runs a campaign is usually the machine somebody is working on. A lock in
  the output directory makes that true across processes too: a second campaign pointed at the same ``--out`` refuses
  to start instead of refitting what the first is fitting.
* **it resumes.** A dataset whose manifest exists is not regenerated, a model whose card exists is not refitted, and
  ``--force`` is the only way to change that. A run that dies at family six restarts at family six.
* **it stops when you said.** ``--max-hours`` finishes the family in flight and stops before starting the next, so a
  night is a night; re-running the same command picks up where it stopped.
* **it reports while it runs.** ``runner.json`` says which family is in flight; ``progress.json`` in each family's
  dataset and model directories says how far that stage got and how long it still needs —
  ``python -m benchmarks.lpm_monitor --out ...`` is the screen that reads them.
* **it records its own failures.** A family that raises is written to ``failures.json`` with the traceback and the run
  continues; the exit code is 1 at the end so a scheduler sees it.
* **it says what it will cost.** ``--plan`` prints the designs, the wall-clock estimate from the measurements in
  ``FIELD_SECONDS`` and ``TRAIN_SECONDS_PER_DESIGN_EPOCH``, and the output path of every family.

Where the numbers come from: the per-design times are the measured cost of one reference evaluation on this machine
(the 3-D heat sink and the 3-D bracket are solver-backed and are the two that cost seconds), and training is fitted
through one constant from the measured 3-D wing run. They are estimates for planning, not promises.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks import _daemon

#: the recipes live in ``benchmarks/lpm_recipes.json``: the campaign's sizes are data, so changing how big a design
#: space is does not mean editing this file
RECIPES_FILE = Path(__file__).with_name("lpm_recipes.json")


def _recipes() -> dict[str, Any]:
    return json.loads(RECIPES_FILE.read_text(encoding="utf-8"))


_RECIPES = _recipes()
#: measured seconds per design for one reference evaluation (see the module docstring)
FIELD_SECONDS: dict[str, float] = {k: float(v) for k, v in _RECIPES["field_seconds"].items()}
#: seconds of training per design-epoch at width 128, from the measured 3-D wing run (72 designs, 400 epochs, 518 s)
TRAIN_SECONDS_PER_DESIGN_EPOCH = float(_RECIPES["train_seconds_per_design_epoch"])
#: the campaign's own recipes. ``full`` is what a dataset worth training on looks like; ``smoke`` is the wiring.
RECIPES: dict[str, dict[str, dict[str, Any]]] = _RECIPES["families"]
#: the order they run in: the cheap families first, so a night that gets cut short still leaves usable models
ORDER: tuple[str, ...] = tuple(_RECIPES["order"])


def override(recipe: dict[str, Any], args: Any) -> dict[str, Any]:
    """Apply the command-line size overrides, so one family can be run bigger without touching the recipes file."""
    out = dict(recipe)
    for key in ("geometries", "conditions", "epochs", "width"):
        value = getattr(args, key, None)
        if value is not None and out.get(key) is not None:
            out[key] = value
    return out


def catalogue(selected: list[str], scale: str, out: Path) -> None:
    """Every design space, what it predicts and how big the recipe is: the menu, before anything runs."""
    fams = families_by_name()
    print(f"{'family':<18}{'dim':>4} {'fields':<32}{'reference':<10}{'bodies':>10}{'conds':>7}{'epochs':>8}",
          flush=True)
    for name in selected:
        fam, recipe = fams[name], RECIPES[name][scale]
        fields = ", ".join(fam.field_names)
        bodies = "published" if recipe["geometries"] is None else str(recipe["geometries"])
        conditions = "-" if recipe["conditions"] is None else str(recipe["conditions"])
        print(f"{name:<18}{fam.dim:>4} {fields[:31]:<32}{fam.reference.implements:<10}"
              f"{bodies:>10}{conditions:>7}{recipe['epochs']:>8}", flush=True)


def status(selected: list[str], out: Path) -> int:
    """What already exists under ``--out``: the designs written and the held-out number each model reached."""
    done = 0
    print(f"{'family':<18}{'designs':>8}{'bodies':>8}{'held-out':>10}{'baseline':>10}{'OOD probes':>12}"
          f"{'epochs':>8}  written", flush=True)
    for name in selected:
        manifest = out / name / "dataset" / "manifest.json"
        card = out / name / "model" / "card.json"
        designs = bodies = held = baseline = probes = epochs = "-"
        created = ""
        if manifest.is_file():
            m = json.loads(manifest.read_text(encoding="utf-8"))
            designs, bodies, created = str(m.get("n_designs", "-")), str(m.get("n_geometries", "-")),                 str(m.get("created_at", ""))[:16]
        if card.is_file():
            c = json.loads(card.read_text(encoding="utf-8"))
            held_out = c.get("held_out", {})
            held = _fmt(held_out.get("relative_l2"))
            baseline = _fmt(held_out.get("constant_field_baseline_relative_l2"))
            ood = c.get("ood", {})
            probes = f"{ood.get('probes_flagged', '-')}/{ood.get('probes_total', '-')}"
            epochs = str(c.get("training", {}).get("epochs", "-"))
            created = str(c.get("created_at", created))[:16]
            done += 1
        print(f"{name:<18}{designs:>8}{bodies:>8}{held:>10}{baseline:>10}{probes:>12}{epochs:>8}  {created}",
              flush=True)
    print(f"\n{done} of {len(selected)} fitted. held-out is the relative L2 on bodies held out by geometry; baseline is "
          f"predicting the training-average field, and a model at or above it has not learnt the design space.",
          flush=True)
    return 0


def families_by_name() -> dict[str, Any]:
    """Every design space the pipeline ships, 2-D and 3-D, by name."""
    from pinneapple_neural.lpm_families_3d import families_3d
    from pinneapple_neural.lpm_pipeline import families

    return {**families(), **families_3d()}


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def _acquire_lock(out: Path) -> Path:
    """One campaign per output directory. A second one would refit what the first is fitting, on the same machine."""
    lock = Path(out) / ".campaign.lock"
    if lock.is_file():
        try:
            pid = int(lock.read_text(encoding="utf-8").split()[0])
        except (OSError, ValueError, IndexError):
            pid = 0
        if pid and _alive(pid):
            raise SystemExit(f"another campaign is already writing to {out} (pid {pid}). "
                             f"Stop it, or run this one with a different --out.")
    lock.write_text(f"{os.getpid()}\n", encoding="utf-8")
    return lock


def limit_threads(threads: int) -> int:
    """Cap the torch/BLAS pools so a fit does not take the whole machine.

    The campaigns run on the machine its owner is working on: a 34-hour campaign that makes the laptop unusable is a
    campaign nobody starts. ``0`` leaves the pools alone; anything else is also exported for the linear solvers of the
    3-D references, which read ``OMP_NUM_THREADS`` and friends rather than torch.
    """
    if threads <= 0:
        return 0
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    try:
        import torch  # noqa: PLC0415 - only when a cap was asked for
        torch.set_num_threads(threads)
    except Exception:                                    # noqa: BLE001 - a missing torch is not this function's problem
        pass
    return threads


class Reporter:
    """``runner.json``: which design space is in flight, and what the ones before it produced.

    The monitor reads this file; so does a person coming back to a campaign they left running, which is why it is
    written before each stage rather than only at the end. Written atomically — a reader never sees half a record.
    """

    def __init__(self, out: Path, *, scale: str, stage: str, seed: int, order: list[str], estimate_s: float,
                 max_hours: float | None, threads: int) -> None:
        self.path = Path(out) / "runner.json"
        self.started = time.monotonic()
        self.payload: dict[str, Any] = {
            "schema": "pinneapple.lpm.runner/1", "pid": os.getpid(), "scale": scale, "stage": stage, "seed": seed,
            "families": list(order), "started_at": _now(), "current": None, "current_stage": None,
            "estimate_s": round(estimate_s, 1), "max_hours": max_hours, "threads": threads or None,
            "one_at_a_time": True, "stopped": False, "done": [],
        }
        self._write()

    def _write(self) -> None:
        self.payload["elapsed_s"] = round(time.monotonic() - self.started, 1)
        self.payload["updated_at"] = _now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_name(self.path.name + ".tmp")
        tmp.write_text(json.dumps(self.payload, indent=2), encoding="utf-8")
        tmp.replace(self.path)

    def on(self, family: str, stage: str) -> None:
        self.payload["current"], self.payload["current_stage"] = family, stage
        self._write()

    def finished(self, family: str, result: dict[str, Any]) -> None:
        self.payload["done"].append(result)
        self.payload["current"], self.payload["current_stage"] = None, None
        self._write()

    def stop(self, reason: str) -> None:
        self.payload["stopped"] = True
        self.payload["stop_reason"] = reason
        self._write()


def estimate(name: str, recipe: dict[str, Any], stage: str) -> dict[str, Any]:
    """Designs, and a wall-clock estimate for the requested stages."""
    geometries = recipe["geometries"]
    published = geometries is None
    if published:
        # the campaign's own run count, read from the family without downloading anything it does not have
        try:
            designs = len(families_by_name()[name].specs() or [])
        except Exception:                              # noqa: BLE001 - no cache: report what we can
            designs = 0
        geometries, conditions = 100, 3
    else:
        conditions = recipe["conditions"]
        designs = geometries * conditions
    field_s = FIELD_SECONDS.get(name, 0.05) * designs
    train_s = (designs * recipe["epochs"] * (recipe["width"] / 128.0) ** 2 * TRAIN_SECONDS_PER_DESIGN_EPOCH)
    total = (field_s if stage in ("dataset", "both") else 0.0) + (train_s if stage in ("train", "both") else 0.0)
    return {"designs": designs, "geometries": geometries, "conditions": conditions, "published": published,
            "dataset_s": field_s, "train_s": train_s, "total_s": total}


def plan(selected: list[str], stage: str, out: Path, force: bool, scale: str,
         args: Any | None = None) -> list[dict[str, Any]]:
    rows = []
    for name in selected:
        recipe = RECIPES[name][scale] if args is None else override(RECIPES[name][scale], args)
        e = estimate(name, recipe, stage)
        dataset, model = out / name / "dataset", out / name / "model"
        e.update(name=name, recipe=recipe,
                 dataset_done=(dataset / "manifest.json").is_file() and not force,
                 model_done=(model / "card.json").is_file() and not force,
                 out=str(out / name))
        rows.append(e)
    return rows


def print_plan(rows: list[dict[str, Any]], stage: str) -> None:
    print(f"{'family':<18}{'designs':>8}{'dataset':>10}{'fit':>10}{'total':>10}  state", flush=True)
    total = 0.0
    for r in rows:
        state = []
        if r["dataset_done"] and stage in ("dataset", "both"):
            state.append("dataset exists")
        if r["model_done"] and stage in ("train", "both"):
            state.append("model exists")
        parts = {"dataset": r["dataset_s"], "train": r["train_s"], "both": r["total_s"]}[stage]
        print(f"{r['name']:<18}{r['designs']:>8}"
              f"{_hms(r['dataset_s']) if stage in ('dataset', 'both') else '-':>10}"
              f"{_hms(r['train_s']) if stage in ('train', 'both') else '-':>10}"
              f"{_hms(parts):>10}  {', '.join(state) or 'to do'}", flush=True)
        total += parts
    print(f"{'':<18}{'':>8}{'':>10}{'':>10}{_hms(total):>10}  total estimate", flush=True)


def _hms(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def run_family(name: str, recipe: dict[str, Any], out: Path, stage: str, seed: int, force: bool,
               verbose: bool, progress_every_s: float = 5.0, reporter: Reporter | None = None,
               refine: int = 0) -> dict[str, Any]:
    """Generate one family's dataset and/or fit its model, skipping what is already there."""
    from pinneapple_neural.lpm_pipeline import generate_dataset, train

    family = families_by_name()[name]
    root, dataset = out / name, out / name / "dataset"
    result: dict[str, Any] = {"family": name}
    if stage in ("dataset", "both") and (force or not (dataset / "manifest.json").is_file()):
        if reporter is not None:
            reporter.on(name, "dataset")
        _log(f"{name}: drawing {recipe['geometries'] or 'the published'} geometries and building the field")
        started = time.monotonic()
        manifest = generate_dataset(family, dataset, n_geometries=recipe["geometries"] or 100,
                                    conditions_per_geometry=recipe["conditions"] or 3, seed=seed,
                                    verbose=verbose, progress_every_s=progress_every_s)
        result["dataset"] = {"designs": manifest["n_designs"], "geometries": manifest["n_geometries"],
                             "seconds": round(time.monotonic() - started, 1)}
        _log(f"{name}: dataset {manifest['n_designs']} designs, {manifest['n_geometries']} bodies, "
             f"{result['dataset']['seconds']}s")
    elif stage in ("dataset", "both"):
        result["dataset"] = {"skipped": "already there"}
        _log(f"{name}: dataset already there, skipped")
    if stage in ("train", "both") and (force or not (root / "model" / "card.json").is_file()):
        if reporter is not None:
            reporter.on(name, "fit")
        _log(f"{name}: fitting {recipe['epochs']} epochs at width {recipe['width']}")
        started = time.monotonic()
        summary = train(dataset, root, epochs=recipe["epochs"], width=recipe["width"], seed=seed,
                        batch_designs=min(24, max(4, recipe["geometries"] or 24)),
                        progress_every_s=progress_every_s, resume=True)
        result["model"] = {k: summary[k] for k in ("held_out_relative_l2", "constant_field_baseline_relative_l2",
                                                   "ood_probes_flagged", "within_tolerance")}
        result["model"]["seconds"] = round(time.monotonic() - started, 1)
        _log(f"{name}: held-out {_fmt(summary['held_out_relative_l2'])} against baseline "
             f"{_fmt(summary['constant_field_baseline_relative_l2'])}, OOD probes {summary['ood_probes_flagged']}, "
             f"{result['model']['seconds']}s")
    elif stage in ("train", "both"):
        result["model"] = {"skipped": "already there"}
        _log(f"{name}: model already there, skipped")
    if refine and (root / "model" / "card.json").is_file() and \
            (force or not (root / "model" / "refinement.json").is_file()):
        _log(f"{name}: refining — {refine} bodies where the model's confidence is weakest, then a warm refit")
        started = time.monotonic()
        round_out = refine_family(name, family, dataset, root, refine, seed=seed, epochs=recipe["epochs"],
                                  progress_every_s=progress_every_s, verbose=verbose)
        round_out["seconds"] = round(time.monotonic() - started, 1)
        result["refinement"] = round_out
        _log(f"{name}: refined {round_out.get('added', 0)} bodies, held-out "
             f"{_fmt(round_out.get('held_out_relative_l2'))} against baseline "
             f"{_fmt(round_out.get('constant_field_baseline_relative_l2'))} ({round_out['seconds']}s)")
    return result


def refine_family(name: str, family: Any, dataset: Path, root: Path, n_new: int, *, seed: int,
                  epochs: int, threshold: float = 0.5, progress_every_s: float = 0.0,
                  verbose: bool = False) -> dict[str, Any]:
    """Stage 1's second round: simulate where the model is least sure, then refit warm from where it was.

    The scores are the model's own confidence on the **training** designs (:mod:`pinneapple_neural.lpm_stages.trust`),
    the proposals come from :func:`~pinneapple_neural.lpm_stages.space.refine`, and the new bodies go into the same
    dataset marked as refined — which is what keeps them out of the held-out set, because a body chosen *for* the
    model's uncertainty cannot also be the evidence that it generalises.

    The refit warm-starts from the run it is refining (:func:`~.lpm_pipeline.train`'s ``init_from``), so the second
    round spends its epochs on the new bodies instead of relearning the first round's field. The parent checkpoint is
    copied aside first: the fit writes into the same directory it is reading from.
    """
    import shutil

    from pinneapple_neural.lpm_pipeline import (
        _split_by_geometry,
        extend_dataset,
        load_dataset,
        load_model,
        train,
    )
    from pinneapple_neural.lpm_stages import space, trust

    designs, manifest = load_dataset(dataset)
    if family.published_split(manifest["designs"]) is not None:
        return {"round": 1, "added": 0,
                "note": "the dataset carries a published split: a refined body would train on a split somebody else "
                        "fixed, and the two rounds would stop being comparable"}
    model = load_model(root)
    # the same split the fit used, so the confidence being read is on bodies the model was trained on
    train_idx, _ = _split_by_geometry(manifest, 0.25, seed)
    training = [designs[i] for i in train_idx]
    calibration = trust.calibrate(model, training)
    scores = [trust.confidence(model, d, calibration)["confidence"] for d in training]
    records = [manifest["designs"][i] for i in train_idx]

    def _spec(record: Any) -> Any:
        from pinneapple_neural.lpm_pipeline import DesignSpec
        return DesignSpec(geometry=dict(record["geometry"]), operating=dict(record["operating"]),
                          geometry_id=int(record["geometry_id"]), label=record["label"])

    proposals = space.refine(scores, [_spec(r) for r in records], n_new, geometry_params=dict(family.geometry_params),
                             threshold=threshold, seed=seed + 1)
    first_id = max(int(r["geometry_id"]) for r in manifest["designs"]) + 1
    specs = space.as_specs(proposals, family, first_geometry_id=first_id)
    if not specs:
        return {"round": 1, "proposed": len(proposals), "added": 0,
                "note": "every proposal fell outside the family's declared ranges"}
    round_info = {"round": 1, "proposed": len(proposals), "threshold": float(threshold),
                  "scores": {"mean": float(sum(scores) / len(scores)), "min": min(scores), "max": max(scores)},
                  "seed": int(seed), "policy": "cluster at the confidence threshold on the training designs"}
    extend_dataset(family, dataset, specs, refinement=round_info)
    if verbose:
        print(f"  {name}: {len(specs)} refined bodies added, {len(proposals) - len(specs)} refused by validate")

    parent = root / "refinement" / "parent"
    (parent / "model").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "model" / "model.pt", parent / "model" / "model.pt")
    # the second round is the same model as the first or it cannot be warm-started from it, so the architecture comes
    # off the parent's card rather than out of this function's arguments
    card = json.loads((root / "model" / "card.json").read_text(encoding="utf-8"))
    config = card["model"]["config"]
    summary = train(dataset, root, epochs=epochs, seed=seed, init_from=parent, resume=True,
                    progress_every_s=progress_every_s,
                    batch_designs=min(24, max(4, manifest["n_geometries"] + len(specs))),
                    **{key: config[key] for key in ("width", "code", "n_heads", "n_freq", "backbone",
                                                    "attn_layers", "attn_heads") if key in config})
    out = {**round_info, "added": len(specs), "geometry_ids": [s.geometry_id for s in specs],
           "held_out_relative_l2": summary["held_out_relative_l2"],
           "constant_field_baseline_relative_l2": summary["constant_field_baseline_relative_l2"],
           "within_tolerance": summary["within_tolerance"],
           "parent": str(parent.relative_to(root))}
    (root / "model" / "refinement.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    return out


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def detach(log_path: Path) -> None:
    """Re-run this command in its own session, so it outlives the shell and the editor session that started it.

    ``nohup`` only ignores SIGHUP; what kills a long fit is the process group going away, and ``os.setsid`` is what
    puts it beyond reach of that. Shared with the panel, which has the same problem (see :mod:`benchmarks._daemon`).
    """
    _daemon.detach("benchmarks.lpm_campaign", log_path, marker="LPM_CAMPAIGN_DETACHED")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LPM campaign: one dataset and one model per design space.")
    parser.add_argument("--families", nargs="+", default=list(ORDER), choices=list(ORDER),
                        help="which design spaces to run (default: all of them, cheapest first)")
    parser.add_argument("--stage", choices=("dataset", "train", "both"), default="both")
    parser.add_argument("--scale", choices=("smoke", "full"), default="full",
                        help="smoke: two bodies and five epochs, to prove the wiring; full: the campaign recipe")
    parser.add_argument("--out", type=Path, default=Path("benchmarks/_out/lpm_campaign"),
                        help="where the datasets and models go; put it outside the repository for a long campaign")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plan", action="store_true", help="print what would run and exit")
    parser.add_argument("--list", action="store_true", help="what each design space predicts and how big it is")
    parser.add_argument("--status", action="store_true", help="what already exists under --out, and what it scored")
    parser.add_argument("--geometries", type=int, help="override the recipe's body count for the selected families")
    parser.add_argument("--conditions", type=int, help="override the recipe's operating points per body")
    parser.add_argument("--epochs", type=int, help="override the recipe's epochs")
    parser.add_argument("--width", type=int, help="override the recipe's model width")
    parser.add_argument("--refine", type=int, default=0, metavar="N",
                        help="after the fit, simulate N more bodies where the model's confidence is closest to the "
                             "decision threshold and refit warm from the round it refined (stage 1)")
    parser.add_argument("--force", action="store_true", help="regenerate and refit even what is already there")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--threads", type=int, default=max(1, (os.cpu_count() or 4) // 2),
                        help="cap on the torch/BLAS pools, so the machine stays usable; 0 leaves them alone "
                             "(default: half the cores)")
    parser.add_argument("--max-hours", type=float, default=None,
                        help="stop before starting a family once this many hours have passed; re-run to resume")
    parser.add_argument("--progress-every", type=float, default=5.0, metavar="SECONDS",
                        help="write progress.json this often while a dataset or a fit runs (0 disables it)")
    parser.add_argument("--detach", type=Path, metavar="LOG",
                        help="run in its own session, writing to LOG, and return immediately")
    a = parser.parse_args(argv)

    def _selected() -> list[str]:
        return [n for n in ORDER if n in set(a.families)]

    # the two read-only views answer and exit before anything is detached or drawn
    if a.list:
        catalogue(_selected(), a.scale, a.out)
        return 0
    if a.status:
        return status(_selected(), a.out)
    if a.detach:
        detach(a.detach)

    # a smoke run must never leave a dataset where the real campaign looks for one, or the campaign would skip the
    # family it was supposed to build
    if a.scale == "smoke" and a.out == Path("benchmarks/_out/lpm_campaign"):
        a.out = a.out.with_name(a.out.name + "_smoke")
    selected = _selected()
    rows = plan(selected, a.stage, a.out, a.force, a.scale, a)
    print(f"scale {a.scale}, stage {a.stage}, out {a.out}", flush=True)
    print_plan(rows, a.stage)
    if a.plan:
        return 0

    started = time.monotonic()
    capped = limit_threads(a.threads)
    a.out.mkdir(parents=True, exist_ok=True)
    lock = _acquire_lock(a.out)
    reporter = Reporter(a.out, scale=a.scale, stage=a.stage, seed=a.seed, order=selected,
                        estimate_s=sum(r["total_s"] for r in rows), max_hours=a.max_hours, threads=capped)
    print(f"one design space at a time: {len(selected)} families in sequence, nothing in parallel"
          f"{f', {capped} threads per process' if capped else ''}", flush=True)
    print(f"follow it with: python -m benchmarks.lpm_monitor --out {a.out}", flush=True)
    results, failures, stopped = [], {}, None
    try:
        for r in rows:
            if a.max_hours is not None and (time.monotonic() - started) / 3600.0 >= a.max_hours:
                stopped = f"--max-hours {a.max_hours:g} reached"
                _log(f"{stopped}; stopping before {r['name']}. Re-run the same command to resume.")
                break
            recipe = r["recipe"]
            try:
                result = run_family(r["name"], recipe, a.out, a.stage, a.seed, a.force, a.verbose,
                                    progress_every_s=a.progress_every, reporter=reporter, refine=a.refine)
                results.append(result)
                reporter.finished(r["name"], result)
            except Exception as exc:                    # noqa: BLE001 - one family dying must not stop the night
                failures[r["name"]] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                reporter.finished(r["name"], {"family": r["name"], "error": f"{type(exc).__name__}: {exc}"})
                _log(f"{r['name']}: FAILED - {type(exc).__name__}: {exc}")
                continue
    finally:
        lock.unlink(missing_ok=True)                    # a killed run leaves the lock, and the next one says so
        if stopped:
            reporter.stop(stopped)
    (a.out / "campaign.json").parent.mkdir(parents=True, exist_ok=True)
    (a.out / "campaign.json").write_text(json.dumps(
        {"scale": a.scale, "stage": a.stage, "seed": a.seed, "started_at": reporter.payload["started_at"],
         "seconds": round(time.monotonic() - started, 1), "order": selected,
         "estimates": {r["name"]: round(r["total_s"], 1) for r in rows},
         "stopped": stopped, "families": results, "failures": sorted(failures)}, indent=2), encoding="utf-8")
    if failures:
        (a.out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        _log(f"{len(failures)} family/families failed: {', '.join(sorted(failures))} (see {a.out / 'failures.json'})")
    _log(f"done in {_hms(time.monotonic() - started)}; {a.out / 'campaign.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
