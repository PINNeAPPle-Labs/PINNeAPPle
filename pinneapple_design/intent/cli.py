"""``pinneapple design "<goal>"``: the design-from-intent pipeline from a shell.

    pinneapple design "a bracket that holds 2 kN" --answer arm_length=100mm --out bracket_design
    pinneapple design "I want a toy car that does a loop" --out loop --no-verify
    pinneapple design --templates
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from typing import Any


def _answer(s: str) -> tuple[str, Any]:
    if "=" not in s:
        raise argparse.ArgumentTypeError("answers look like slot=value[unit], e.g. arm_length=100mm")
    k, v = s.split("=", 1)
    m = re.fullmatch(r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)\s*(.*)", v)
    if not m:
        return k.strip(), v.strip()                     # a choice (material name)
    return k.strip(), (float(m.group(1)), m.group(2).strip()) if m.group(2).strip() else float(m.group(1))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="pinneapple design", description="Design from a plain-language goal.")
    ap.add_argument("text", nargs="?", help="the goal, in quotes")
    ap.add_argument("--answer", "-a", action="append", type=_answer, default=[],
                    help="answer a follow-up question: slot=value[unit] (repeatable)")
    ap.add_argument("--out", "-o", help="folder for the design package (report, geometry, check files)")
    ap.add_argument("--no-verify", action="store_true", help="skip the full-fidelity solver")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--template", help="force a template instead of selecting one from the text")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    ap.add_argument("--templates", action="store_true", help="list the installed problem templates")
    a = ap.parse_args(argv)
    from . import design, list_templates

    if a.templates:
        for name, t in list_templates().items():
            print(f"{name:14s} {t.title} ({t.domain})")
        return 0
    if not a.text:
        ap.print_usage()
        return 2
    r = design(a.text, answers=dict(a.answer) or None, seed=a.seed, out_dir=a.out, verify=not a.no_verify,
               template=a.template, progress=None if a.json else (lambda m: print(f"  {m}", file=sys.stderr)))
    if a.json:
        print(json.dumps(r.to_dict(), indent=1, default=str))
    else:
        print(r.summary() if hasattr(r, "summary") else str(r))
        if hasattr(r, "questions"):
            for q in r.questions:
                print(f"  --answer {q.slot}=<value>{q.unit.replace(' ', '')}   # {q.text}")
        if getattr(r, "files", None):
            print(f"  report: {r.files.get('html')}")
    status = getattr(r, "status", None)
    return 0 if status == "designed" else (3 if hasattr(r, "questions") else 4)


if __name__ == "__main__":
    sys.exit(main())
