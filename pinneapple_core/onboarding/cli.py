"""``pinneapple doctor | recommend | card | new``: the four commands a new user needs first.

They are separate from ``pinneapple check`` (which lints results) because they answer questions
about *getting started*, not about a result. All four run offline and print plain text: no colour,
no box drawing, readable in a terminal, a log or a CI transcript.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .catalog import card as _card
from .catalog import domains, format_card
from .doctor import doctor as _doctor
from .doctor import profiles
from .newproject import TEMPLATES, new_project
from .recommend import recommend as _recommend

__all__ = ["main", "doctor_main", "recommend_main", "card_main", "new_main"]


def doctor_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="pinneapple doctor",
                                 description="Check this installation against what you intend to do.")
    ap.add_argument("profile", nargs="?", default="start",
                    help=", ".join(profiles()))
    ap.add_argument("--json", action="store_true", help="machine-readable report")
    ap.add_argument("-q", "--quiet", action="store_true", help="exit code only")
    a = ap.parse_args(argv)
    try:
        report = _doctor(a.profile, verbose=not (a.json or a.quiet))
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(report.as_dict(), indent=2))
    return 0 if report.ok else 1


def recommend_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="pinneapple recommend",
                                 description="Turn a problem described in a sentence into a preset, "
                                             "a method and a model.")
    ap.add_argument("description", nargs="*", help="the problem, in a few words (English or Portuguese)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--top", type=int, default=3, help="how many alternatives to show")
    ap.add_argument("--domains", action="store_true", help="list the presets by domain and exit")
    a = ap.parse_args(argv)
    if a.domains:
        for domain, names in domains().items():
            print(f"{domain}:")
            for n in names:
                print(f"  {n}")
        return 0
    text = " ".join(a.description).strip()
    if not text:
        print("error: describe the problem, e.g. pinneapple recommend \"heat sink cooled by a fan\"",
              file=sys.stderr)
        return 2
    rec = _recommend(text, top=a.top)
    if a.json:
        print(json.dumps(rec.as_dict(), indent=2))
    else:
        print(rec.explain())
    return 0 if rec.preset else 1


def card_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="pinneapple card",
                                 description="What a preset takes, what it gives back, and the "
                                             "smallest program that uses it.")
    ap.add_argument("preset", nargs="?", help="preset name; omit with --list")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--list", action="store_true", help="every preset, grouped by domain")
    a = ap.parse_args(argv)
    if a.list or not a.preset:
        if not a.list and not a.preset:
            ap.print_help()
            print("\npresets by domain: pinneapple card --list")
            return 2
        for domain, names in domains().items():
            print(f"{domain}:")
            for n in names:
                print(f"  {n}")
        return 0
    try:
        c = _card(a.preset)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    print(json.dumps(c.as_dict(), indent=2) if a.json else format_card(c))
    return 0


def new_main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(prog="pinneapple new",
                                 description="Create a project of your own (outside this repository).")
    ap.add_argument("name", help="folder name: lowercase letters, digits and underscores")
    ap.add_argument("--template", default="plain", choices=sorted(TEMPLATES),
                    help="; ".join(f"{k}: {v}" for k, v in TEMPLATES.items()))
    ap.add_argument("--preset", default="burgers_1d", help="the problem to start from")
    ap.add_argument("--dir", default=None, help="parent folder (default: the current directory)")
    ap.add_argument("--dry-run", action="store_true", help="show the files without writing them")
    a = ap.parse_args(argv)
    if a.dry_run:
        from .newproject import plan

        try:
            spec = plan(a.name, template=a.template, preset=a.preset)
        except Exception as e:
            print(f"error: {e}", file=sys.stderr)
            return 2
        print(f"would create ./{a.name}/ ({a.template})")
        for rel in sorted(spec["files"]):
            print(f"  {rel}")
        return 0
    try:
        new_project(a.name, dir=a.dir, template=a.template, preset=a.preset)
    except (ValueError, FileExistsError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    """``pinneapple <doctor|recommend|card|new> ...``"""
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(
            "usage: pinneapple <command> [options]\n\n"
            "getting started:\n"
            "  doctor [profile]        is this installation ready for my work?\n"
            "  recommend <sentence>    which preset, method and model for my problem?\n"
            "  card <preset>           what does a preset take and give back?\n"
            "  new <name>              create a project of my own\n\n"
            "working with results:\n"
            "  check <paths>           physics linter for results and models\n"
            "  benchmark               equal-budget benchmarks of physics-AI methods\n"
            "  design <goal>           design from a plain-language goal\n"
        )
        return 0 if argv else 2
    cmd, rest = argv[0], argv[1:]
    table = {"doctor": doctor_main, "recommend": recommend_main, "card": card_main, "new": new_main}
    if cmd in table:
        return table[cmd](rest)
    return 2 if cmd not in table else 0


if __name__ == "__main__":
    raise SystemExit(main())
