"""The LPM campaign in one command: one dataset and one model per design space.

The pipeline (``pinneapple_neural.lpm_pipeline``) knows how to turn one family into a dataset and a model. This script
knows *which* family, how big, in what order, and how to keep going when one of them fails — which is what a campaign
needs at 3 a.m.

    python -m benchmarks.lpm_campaign --plan                  # what would run, and roughly how long
    python -m benchmarks.lpm_campaign --scale smoke           # the whole wiring, in about two minutes
    python -m benchmarks.lpm_campaign --stage dataset         # the data only
    python -m benchmarks.lpm_campaign --stage train           # the fits only
    python -m benchmarks.lpm_campaign --detach campaign.log   # survives the shell (and the editor session)

What it guarantees, because a campaign that has to be watched is not a campaign:

* **it resumes.** A dataset whose manifest exists is not regenerated, a model whose card exists is not refitted, and
  ``--force`` is the only way to change that. A run that dies at family six restarts at family six.
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
import sys
import time
import traceback
from pathlib import Path
from typing import Any

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
               verbose: bool) -> dict[str, Any]:
    """Generate one family's dataset and/or fit its model, skipping what is already there."""
    from pinneapple_neural.lpm_pipeline import generate_dataset, train

    family = families_by_name()[name]
    root, dataset = out / name, out / name / "dataset"
    result: dict[str, Any] = {"family": name}
    if stage in ("dataset", "both") and (force or not (dataset / "manifest.json").is_file()):
        _log(f"{name}: drawing {recipe['geometries'] or 'the published'} geometries and building the field")
        started = time.monotonic()
        manifest = generate_dataset(family, dataset, n_geometries=recipe["geometries"] or 100,
                                    conditions_per_geometry=recipe["conditions"] or 3, seed=seed,
                                    verbose=verbose)
        result["dataset"] = {"designs": manifest["n_designs"], "geometries": manifest["n_geometries"],
                             "seconds": round(time.monotonic() - started, 1)}
        _log(f"{name}: dataset {manifest['n_designs']} designs, {manifest['n_geometries']} bodies, "
             f"{result['dataset']['seconds']}s")
    elif stage in ("dataset", "both"):
        result["dataset"] = {"skipped": "already there"}
        _log(f"{name}: dataset already there, skipped")
    if stage in ("train", "both") and (force or not (root / "model" / "card.json").is_file()):
        _log(f"{name}: fitting {recipe['epochs']} epochs at width {recipe['width']}")
        started = time.monotonic()
        summary = train(dataset, root, epochs=recipe["epochs"], width=recipe["width"], seed=seed,
                        batch_designs=min(24, max(4, recipe["geometries"] or 24)))
        result["model"] = {k: summary[k] for k in ("held_out_relative_l2", "constant_field_baseline_relative_l2",
                                                   "ood_probes_flagged", "within_tolerance")}
        result["model"]["seconds"] = round(time.monotonic() - started, 1)
        _log(f"{name}: held-out {_fmt(summary['held_out_relative_l2'])} against baseline "
             f"{_fmt(summary['constant_field_baseline_relative_l2'])}, OOD probes {summary['ood_probes_flagged']}, "
             f"{result['model']['seconds']}s")
    elif stage in ("train", "both"):
        result["model"] = {"skipped": "already there"}
        _log(f"{name}: model already there, skipped")
    return result


def _fmt(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def detach(log_path: Path) -> None:
    """Re-run this command in its own session, so it outlives the shell and the editor session that started it.

    ``nohup`` only ignores SIGHUP; what kills a long fit is the process group going away, and ``os.setsid`` is what
    puts it beyond reach of that.
    """
    if os.environ.get("LPM_CAMPAIGN_DETACHED"):
        return
    argv = [sys.executable, "-m", "benchmarks.lpm_campaign"] + [a for a in sys.argv[1:]
                                                               if a not in ("--detach", str(log_path))]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    pid = os.fork()
    if pid:
        print(f"detached as pid {pid}; follow it with: tail -f {log_path}")
        print(f"stop it with: kill {pid}")
        sys.exit(0)
    os.setsid()
    env = {**os.environ, "LPM_CAMPAIGN_DETACHED": "1"}
    with open(log_path, "w", encoding="utf-8") as log:
        os.dup2(log.fileno(), 1)
        os.dup2(log.fileno(), 2)
        os.execve(sys.executable, argv, env)


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
    parser.add_argument("--force", action="store_true", help="regenerate and refit even what is already there")
    parser.add_argument("--verbose", action="store_true")
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
    results, failures = [], {}
    for r in rows:
        recipe = r["recipe"]
        try:
            results.append(run_family(r["name"], recipe, a.out, a.stage, a.seed, a.force, a.verbose))
        except Exception as exc:                        # noqa: BLE001 - one family dying must not stop the night
            failures[r["name"]] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
            _log(f"{r['name']}: FAILED - {type(exc).__name__}: {exc}")
            continue
    (a.out / "campaign.json").parent.mkdir(parents=True, exist_ok=True)
    (a.out / "campaign.json").write_text(json.dumps(
        {"scale": a.scale, "stage": a.stage, "seed": a.seed, "seconds": round(time.monotonic() - started, 1),
         "families": results, "failures": sorted(failures)}, indent=2), encoding="utf-8")
    if failures:
        (a.out / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
        _log(f"{len(failures)} family/families failed: {', '.join(sorted(failures))} (see {a.out / 'failures.json'})")
    _log(f"done in {_hms(time.monotonic() - started)}; {a.out / 'campaign.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
