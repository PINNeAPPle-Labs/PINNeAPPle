"""``python -m pinneapple.pipeline``: the pipeline on the command line, for a checked-in YAML file.

    python -m pinneapple.pipeline validate experiment.yaml     # configuration + problem, nothing runs
    python -m pinneapple.pipeline plan experiment.yaml         # the stages and the cost
    python -m pinneapple.pipeline run experiment.yaml          # the whole thing, then the report
    python -m pinneapple.pipeline resume experiment.yaml       # continue where the last run stopped
    python -m pinneapple.pipeline report experiment.yaml       # the report of the last run
    python -m pinneapple.pipeline view runs/demo/burgers        # the run in a browser, with the model to sweep

Exit codes: 0 the run passed, 1 a stage failed, 2 the configuration is wrong, 3 the run was rejected by a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

__all__ = ["main", "build_parser"]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m pinneapple.pipeline",
                                description="Run a surrogate-model pipeline from a YAML configuration.")
    p.add_argument("command", choices=("validate", "plan", "run", "resume", "report", "presets", "view"))
    p.add_argument("config", nargs="?", help="the pipeline YAML (or JSON) file")
    p.add_argument("--out", default=None, help="override execution.out_dir")
    p.add_argument("--until", default=None, help="run up to and including this stage")
    p.add_argument("--json", action="store_true", help="print machine-readable output")
    p.add_argument("--verbose", action="store_true", help="print each stage as it finishes")
    p.add_argument("--no-open", action="store_true", help="with view: write the page without opening a browser")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from .configs import ConfigError
    from .pipeline import PipelineFailed, PipelineRejected, SurrogatePipeline
    from .presets import PRESETS

    if args.command == "presets":
        print("\n".join(f"{name}: {cfg['description']}" for name, cfg in PRESETS.items()))
        return 0
    if not args.config:
        print("give a configuration file (see `presets`)", file=sys.stderr)
        return 2
    if args.command == "view":
        from .viewer import build_viewer

        target = Path(args.config)
        if target.is_dir():
            run_dir = target
        else:
            config = SurrogatePipeline.from_yaml(args.config).config
            run_dir = Path(config.execution.out_dir) / config.name
        if not (run_dir / "report.json").is_file():
            print(f"no finished run at {run_dir}: run the pipeline first", file=sys.stderr)
            return 1
        page = build_viewer(run_dir, open_browser=not args.no_open)
        print(page)
        return 0
    overrides = {"execution": {"out_dir": args.out}} if args.out else {}
    try:
        pipeline = SurrogatePipeline.from_yaml(args.config, **overrides) if overrides else \
            SurrogatePipeline.from_yaml(args.config)
    except ConfigError as exc:
        print(f"configuration error: {exc}", file=sys.stderr)
        return 2

    if args.command == "validate":
        report = pipeline.dry_run()
        print(json.dumps(report, indent=1) if args.json else _dry_run_text(report))
        return 0 if report["ok"] else 2
    if args.command == "plan":
        plan = pipeline.plan()
        if args.json:
            print(json.dumps(plan, indent=1))
        else:
            print(pipeline.describe())
            print("\nstages:")
            for step in plan["steps"]:
                state = "runs" if step["runs"] else f"skipped ({step['skipped_because']})"
                print(f"  {step['stage']:<13} {state:<40} {step['does']}")
            print(f"\ncost: {plan['cost']}")
        return 0
    if args.command == "report":
        out = Path(pipeline.config.execution.out_dir) / pipeline.config.name / "report.md"
        if not out.is_file():
            print(f"no report at {out}: run the pipeline first", file=sys.stderr)
            return 1
        print(out.read_text(encoding="utf-8"))
        return 0
    try:
        result = pipeline.run(until=args.until, verbose=args.verbose) if args.command == "run" else \
            pipeline.resume(verbose=args.verbose)
    except PipelineFailed as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    except PipelineRejected as exc:
        print(exc.result.to_markdown())
        print(f"\n{exc}", file=sys.stderr)
        return 3
    print(json.dumps(result.to_dict(), indent=1) if args.json else result.to_markdown())
    return 0


def _dry_run_text(report: dict) -> str:
    lines = [f"{report['name']}: {'ready' if report['ok'] else 'not ready'}"]
    for check in report["checks"]:
        lines.append(f"  [{'ok' if check['ok'] else 'NO'}] {check['check']}: {check['detail']}")
    return "\n".join(lines)


if __name__ == "__main__":                                        # pragma: no cover
    raise SystemExit(main())
