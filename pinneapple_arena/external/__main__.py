#!/usr/bin/env python3
"""
Command line for the external model catalog (``python -m pinneapple_arena.external``).

This is what the
external-catalog API points you at when it can't run a model itself in this
process (heavy GPU dependency, or a git-only research repo) — install the
dependency yourself, then run the same demo the "Run inference" / "Fine-tune"
actions would run, from your own machine.

Usage
-----
    python -m pinneapple_arena.external --list
    python -m pinneapple_arena.external --list --category pinn
    python -m pinneapple_arena.external --list --domain fluids

    python -m pinneapple_arena.external --model deepxde --action infer
    python -m pinneapple_arena.external --model deepxde --action finetune --epochs 300 --lr 0.001
    python -m pinneapple_arena.external --model mace --action infer --include-heavy

    python -m pinneapple_arena.external --category pinn --action infer --json report.json
    python -m pinneapple_arena.external --action infer --json report.json --csv report.csv   # whole catalog
"""

from __future__ import annotations

import argparse
import csv
import json
import sys

from pinneapple_arena.external.benchmark import select_slugs
from pinneapple_arena.external.finetune import run_finetune
from pinneapple_arena.external.models_data import MODELS, MODELS_BY_SLUG
from pinneapple_arena.external.result import STATUS_OK
from pinneapple_arena.external.runners.registry import call_infer

STATUS_ICON = {
    "ok": "✓", "failed": "✗", "not_installed": "·",
    "manual_install": "△", "skipped_heavy": "…", "finetune_not_supported": "△",
}


def cmd_list(args) -> None:
    items = MODELS
    if args.category:
        items = [m for m in items if m.arch == args.category]
    if args.domain:
        items = [m for m in items if args.domain in m.domains]
    for m in items:
        pkg = m.pip_package or ("manual install" if m.manual_only else "n/a")
        print(f"[{m.arch}] {m.slug:<28} {m.name:<45} {pkg:<24} {m.repo}")
    print(f"\n{len(items)} of {len(MODELS)} models.")


def print_result(r) -> None:
    icon = STATUS_ICON.get(r.status, "?")
    print(f"{icon} {r.name} [{r.status}]")
    if r.note:
        print(f"   note: {r.note}")
    if r.error:
        print(f"   error: {r.error}")
    if r.output_shape:
        print(f"   output_shape: {r.output_shape}")
    if r.output_summary:
        print(f"   output_summary: {r.output_summary}")
    if r.metrics:
        print(f"   metrics: {r.metrics}")
    if r.local_command:
        print(f"   run locally instead: {r.local_command}")


def cmd_run(args) -> None:
    if args.model:
        if args.model not in MODELS_BY_SLUG:
            sys.exit(f"Unknown model slug '{args.model}'. Use --list to see valid slugs.")
        slugs = [args.model]
    else:
        try:
            slugs = select_slugs(None, args.category, args.domain)
        except KeyError as exc:
            sys.exit(str(exc))

    results = []
    for slug in slugs:
        if args.action == "infer":
            r = call_infer(slug, include_heavy=args.include_heavy)
        else:
            params = {"epochs": args.epochs, "lr": args.lr}
            r = run_finetune(slug, {k: v for k, v in params.items() if v is not None})
        results.append(r)
        print_result(r)
        print()

    n_ok = sum(1 for r in results if r.status == STATUS_OK)
    print(f"{n_ok}/{len(results)} models produced a real result.")

    if args.json:
        with open(args.json, "w") as f:
            json.dump([r.to_dict() for r in results], f, indent=2)
        print(f"JSON report written to {args.json}")

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["slug", "name", "status", "output_shape", "note", "error"])
            for r in results:
                writer.writerow([r.slug, r.name, r.status, r.output_shape, r.note, r.error])
        print(f"CSV report written to {args.csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", type=str, default=None, help="run a single model by slug (see --list)")
    parser.add_argument("--action", choices=["infer", "finetune"], default="infer")
    parser.add_argument("--category", type=str, default=None, help="pinn, operator, diffsim, rigidbody, graph, domain")
    parser.add_argument("--domain", type=str, default=None, help="e.g. fluids, materials, weather...")
    parser.add_argument("--include-heavy", action="store_true")
    parser.add_argument("--epochs", type=int, default=None, help="finetune only")
    parser.add_argument("--lr", type=float, default=None, help="finetune only")
    parser.add_argument("--list", action="store_true", help="print matching models and exit, run nothing")
    parser.add_argument("--json", type=str, default=None)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    if args.list:
        cmd_list(args)
        return
    cmd_run(args)


if __name__ == "__main__":
    main()
