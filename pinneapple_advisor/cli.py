"""``python -m pinneapple_advisor``: the same advice from a terminal, for a script or a cron job.

    python -m pinneapple_advisor --text "transient heat in a 3D bracket, I have the STEP but no mesh"
    python -m pinneapple_advisor --text "..." --answers answers.json --out advice --offline
    python -m pinneapple_advisor --questions
    python -m pinneapple_advisor --text "..." --format mermaid

``--answers`` is a JSON object of brief fields (``{"dimension": 3, "goal": "inverse"}``); ``--offline`` skips the
language model entirely. Without ``--out`` nothing is written and the requested format goes to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import __version__, advise, questions
from .schema import text_of


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pinneapple_advisor", description="Suggest a PINNeAPPle study pipeline.")
    p.add_argument("--text", "-t", default="", help="the problem in a sentence or more")
    p.add_argument("--answers", "-a", default="", help="JSON file (or inline JSON) of brief fields")
    p.add_argument("--out", "-o", default="", help="folder to write plan.json and the diagrams into")
    p.add_argument("--format", "-f", default="json", choices=["json", "mermaid", "dot", "svg", "html", "markdown",
                                                             "text"],
                   help="what to print when --out is not given")
    p.add_argument("--questions", action="store_true", help="print the predefined questions and exit")
    p.add_argument("--offline", action="store_true", help="no language model: rules and questions only")
    p.add_argument("--provider", default="ollama", help="LLM provider for the intake (default: ollama)")
    p.add_argument("--model", default="", help="model name for the provider")
    p.add_argument("--language", default="pt", choices=["pt", "en"])
    p.add_argument("--version", action="version", version=f"pinneapple_advisor {__version__}")
    return p


def _answers(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    if raw.lstrip().startswith("{"):
        return dict(json.loads(raw))
    with open(raw, encoding="utf-8") as fh:
        return dict(json.load(fh))


def _text(plan: Any, lang: str) -> str:
    lines = [f"estimativa: {plan.total_estimate().to_dict()['total_hours']} h", ""]
    for i, s in enumerate(plan.stages, 1):
        mark = {"have": "=", "generate": "+", "confirm": "?", "optional": "o", "skip": "-"}.get(s.status, "?")
        est = ""
        if s.estimate is not None:
            est = f"  ({s.estimate.total_seconds / 3600:.2f} h)" if s.estimate.total_seconds >= 3600 else (
                f"  ({s.estimate.total_seconds / 60:.0f} min)")
        lines.append(f"{i:2d}. [{mark}] {text_of(s.title, lang)}{est}")
    if plan.open_questions:
        lines += ["", "perguntas em aberto: " + ", ".join(q["id"] for q in plan.open_questions)]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.questions:
        print(json.dumps(questions(), ensure_ascii=False, indent=2))
        return 0
    if not args.text:
        print("give the problem with --text, or list the questions with --questions", file=sys.stderr)
        return 2
    plan = advise(args.text, answers=_answers(args.answers), llm=False if args.offline else None,
                  provider=args.provider, model=args.model, language=args.language,
                  out_dir=args.out or None)
    if args.out:
        print(json.dumps({"written": plan.write(args.out), "stages": [s.id for s in plan.stages]}, indent=2))
        return 0
    if args.format == "json":
        print(plan.json())
    elif args.format == "text":
        print(_text(plan, args.language))
    else:
        print(plan.diagram(args.format))
    return 0


if __name__ == "__main__":      # pragma: no cover - the entry point
    raise SystemExit(main())
