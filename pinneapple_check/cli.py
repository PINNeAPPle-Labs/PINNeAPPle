"""``pinneapple check [paths]``: the command line.

Exit codes (like ruff and pytest): 0 no violation (or only warnings with ``--fail-on error``), 1 violations,
2 a usage or configuration error.

The ``pinneapple`` command dispatches sub-commands: ``check`` here, and ``benchmark`` to
:mod:`pinneapple_analysis.benchmark.cli` when that package is installed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .baseline import write_baseline
from .config import ConfigError, load_config
from .engine import run
from .report import FORMATS, render
from .rules import RULES, get_rule

__all__ = ["main", "check_main"]


def _csv(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="pinneapple check",
        description="Physics linter for simulation results and Physics-AI models: conservation, units, boundary "
                    "conditions, PDE residual, bounds, out-of-range use, regression and artefact integrity.",
        epilog="Configuration: pinneapple-check.toml or [tool.pinneapple.check] in pyproject.toml. "
               "Docs: https://pinneapple-labs.github.io/PINNeAPPle/guides/physics_checks_in_ci/")
    ap.add_argument("paths", nargs="*", help="files or folders to check (default: every target of the configuration, "
                                             "or results found under the current folder)")
    ap.add_argument("--config", help="configuration file (default: searched upwards from the first path)")
    ap.add_argument("--select", type=_csv, help="comma-separated rule ids or prefixes to enable (e.g. PC1,PC201)")
    ap.add_argument("--ignore", type=_csv, help="comma-separated rule ids or prefixes to disable")
    ap.add_argument("--extend-select", type=_csv, help="add rules to the selection of the configuration")
    ap.add_argument("--format", choices=FORMATS, default=None,
                    help="output on stdout (default text; github when GITHUB_ACTIONS is set and stdout is not a tty)")
    ap.add_argument("--output", "-o", help="write the main output to this file instead of stdout")
    ap.add_argument("--sarif", metavar="FILE", help="also write SARIF 2.1.0 (GitHub code scanning) to FILE")
    ap.add_argument("--junit", metavar="FILE", help="also write JUnit XML to FILE")
    ap.add_argument("--json", metavar="FILE", dest="json_out", help="also write the JSON report (with the trust "
                                                                    "card) to FILE")
    ap.add_argument("--summary", metavar="FILE", help="also append a Markdown summary to FILE "
                                                      "(e.g. $GITHUB_STEP_SUMMARY)")
    ap.add_argument("--baseline", metavar="FILE", help="baseline file (default: `baseline` of the configuration)")
    ap.add_argument("--write-baseline", action="store_true", help="record today's violations in the baseline and "
                                                                  "exit 0")
    ap.add_argument("--no-baseline", action="store_true", help="ignore the baseline")
    ap.add_argument("--fail-on", choices=("error", "warning", "never"), help="what makes the exit code 1 "
                                                                              "(default error)")
    ap.add_argument("--exit-zero", action="store_true", help="always exit 0 on violations (still 2 on errors of use)")
    ap.add_argument("--show-passed", action="store_true", help="list the checks that passed too (text format)")
    ap.add_argument("--no-fixes", action="store_true", help="do not print fix hints (text format)")
    ap.add_argument("--color", choices=("auto", "always", "never"), default="auto")
    ap.add_argument("--list-rules", action="store_true", help="list every rule and exit")
    ap.add_argument("--explain", metavar="RULE", help="explain a rule (e.g. PC101) and exit")
    ap.add_argument("--arena-cases", nargs="*", metavar="FILE", default=None,
                    help="replay the stored escapes and false alarms of the validation arena (default: the cases "
                         "shipped with pinneapple_analysis.validation.arena) and exit 1 on a regression")
    ap.add_argument("--version", action="version", version=f"pinneapple check {__version__}")
    return ap


def _list_rules() -> str:
    lines = [f"{'id':6} {'severity':8} {'stage':13} name", "-" * 72]
    for r in RULES.values():
        lines.append(f"{r.id:6} {r.severity:8} {r.stage:13} {r.name}: {r.title}")
    return "\n".join(lines) + "\n"


def _explain(rule_id: str) -> str:
    r = get_rule(rule_id)
    return (f"{r.id} {r.name} ({r.severity}, trust-card stage: {r.stage})\n\n{r.title}.\n\n{r.explanation}\n\n"
            f"How to fix: {r.fix}\n\nDocs: {r.help_uri}\n")


def _arena_cases(files: list[str]) -> int:
    try:
        from pinneapple_analysis.validation.arena import builtin_cases, replay
    except ImportError as e:                                         # pragma: no cover - minimal installs
        print(f"error: the validation arena is not available ({e})", file=sys.stderr)
        return 2
    paths = [Path(f) for f in files] or builtin_cases()
    if not paths:
        print("error: no arena cases files given or shipped", file=sys.stderr)
        return 2
    bad = 0
    for p in paths:
        try:
            res = replay(p)
        except (OSError, ValueError, KeyError) as e:
            print(f"error: {p}: {e}", file=sys.stderr)
            return 2
        print("\n".join(res.lines()))
        bad += len(res.regressions)
    print(f"{bad} regression(s)" if bad else "every stored arena case holds")
    return 1 if bad else 0


def check_main(argv: list[str] | None = None) -> int:
    ap = _parser()
    a = ap.parse_args(argv)
    if a.arena_cases is not None:
        return _arena_cases(a.arena_cases)
    if a.list_rules:
        sys.stdout.write(_list_rules())
        return 0
    if a.explain:
        try:
            sys.stdout.write(_explain(a.explain))
        except KeyError as e:
            print(f"error: {e.args[0]}", file=sys.stderr)
            return 2
        return 0
    try:
        paths = [Path(p) for p in a.paths]
        for p in paths:
            if not p.exists():
                raise ConfigError(f"{p} does not exist")
        start = paths[0] if paths else Path.cwd()
        cfg = load_config(Path(a.config) if a.config else None, start=start if start.is_dir() else start.parent)
        report = run(paths, config=cfg, select=a.select, ignore=a.ignore, extend_select=a.extend_select,
                     baseline=a.baseline, use_baseline=not a.no_baseline and not a.write_baseline)
    except (ConfigError, KeyError) as e:
        print(f"error: {e.args[0] if e.args else e}", file=sys.stderr)
        return 2
    if a.write_baseline:
        target = Path(a.baseline) if a.baseline else cfg.baseline or Path(".pinneapple-check-baseline.json")
        n = write_baseline(report, target)
        print(f"wrote {n} violation(s) to {target}; commit it and set `baseline = \"{target.name}\"` under [check]")
        return 0
    fmt = a.format or ("github" if os.environ.get("GITHUB_ACTIONS") == "true" and not sys.stdout.isatty() and
                       not a.output else "text")
    color = {"always": True, "never": False}.get(a.color)
    body = render(report, fmt, color=color if a.output is None else False, show_passed=a.show_passed,
                  show_fixes=not a.no_fixes)
    if fmt == "github":                                        # annotations, then a readable log for humans
        body += render(report, "text", color=False, show_passed=a.show_passed, show_fixes=not a.no_fixes)
    if a.output:
        Path(a.output).write_text(body, encoding="utf-8")
    else:
        sys.stdout.write(body)
    for path, f in ((a.sarif, "sarif"), (a.junit, "junit"), (a.json_out, "json")):
        if path:
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text(render(report, f), encoding="utf-8")
    if a.summary:
        with open(a.summary, "a", encoding="utf-8") as fh:
            fh.write(render(report, "summary"))
    if a.exit_zero:
        return 0
    return report.exit_code(a.fail_on or cfg.fail_on)


def main(argv: list[str] | None = None) -> int:
    """``pinneapple <command>``: ``check`` (this package) or ``benchmark`` (pinneapple_analysis.benchmark)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("usage: pinneapple <command> [options]\n\ncommands:\n"
              "  check      physics linter for results and models (pinneapple check --help)\n"
              "  benchmark  equal-budget benchmarks of physics-AI methods (when installed)\n"
              "  design     design from a plain-language goal (pinneapple design --help)\n")
        return 0 if argv else 2
    if argv[0] == "--version":
        print(f"pinneapple {__version__}")
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "check":
        return check_main(rest)
    if cmd == "design":
        from pinneapple_design.intent.cli import main as design
        return design(rest)
    if cmd == "benchmark":
        try:
            from pinneapple_analysis.benchmark.cli import main as bench
        except ImportError:
            print("error: `pinneapple benchmark` is not available in this installation", file=sys.stderr)
            return 2
        return bench(argv)
    print(f"error: unknown command {cmd!r}; try `pinneapple check`", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
