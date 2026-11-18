"""``python -m pinneapple_intent``: compile a request, run it, replay a run, list the families."""

from __future__ import annotations

import argparse
import json
import sys

from .compile import compile_study
from .contract import NEEDS_CLARIFICATION, StudyContract
from .execute import run_study
from .family import families
from .replay import replay


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pinneapple_intent",
                                     description="From a sentence to a verified, replayable physics study.")
    sub = parser.add_subparsers(dest="command", required=True)

    compile_parser = sub.add_parser("compile", help="compile a request into a study contract and print it")
    compile_parser.add_argument("text")
    compile_parser.add_argument("--mode", default="simulation")
    compile_parser.add_argument("--family", default=None)
    compile_parser.add_argument("--answer", action="append", default=[], metavar="SLOT=VALUE")
    compile_parser.add_argument("--out", default=None, help="write contract.json here")

    run_parser = sub.add_parser("run", help="compile and run a request")
    run_parser.add_argument("text")
    run_parser.add_argument("--mode", default="simulation")
    run_parser.add_argument("--family", default=None)
    run_parser.add_argument("--answer", action="append", default=[], metavar="SLOT=VALUE")
    run_parser.add_argument("--out", default="study_runs")

    contract_parser = sub.add_parser("run-contract", help="run a contract.json that already exists")
    contract_parser.add_argument("path")
    contract_parser.add_argument("--out", default="study_runs")

    replay_parser = sub.add_parser("replay", help="rebuild a recorded run from its files")
    replay_parser.add_argument("path")
    replay_parser.add_argument("--out", default=None)
    replay_parser.add_argument("--strict", action="store_true")

    sub.add_parser("families", help="list the installed study families")

    args = parser.parse_args(argv)

    if args.command == "families":
        for family in families().values():
            print(f"{family.name:18} {family.title}")
            print(f"{'':18} reports {family.qoi} [{family.qoi_unit}] over "
                  f"{', '.join(r.name for r in family.ladder())}")
        return 0

    if args.command == "replay":
        result = replay(args.path, out_dir=args.out, strict=args.strict)
        print(result.summary())
        for difference in result.differences:
            print(f"  {difference.get('rung')}: {difference.get('quantity')} recorded "
                  f"{difference.get('recorded')} and now {difference.get('replayed')}")
        return 0 if result.matches else 1

    answers = _answers(args.answer)
    if args.command == "compile":
        contract = compile_study(args.text, answers=answers, mode=args.mode, family=args.family)
        if isinstance(contract, StudyContract):
            print(contract.to_json())
            if args.out:
                contract.write(args.out)
            return 0 if contract.status != NEEDS_CLARIFICATION else 2
        print(json.dumps(contract.to_dict(), indent=2))
        return 3

    if args.command == "run":
        contract = compile_study(args.text, answers=answers, mode=args.mode, family=args.family)
        if not isinstance(contract, StudyContract):
            print(json.dumps(contract.to_dict(), indent=2))
            return 3
        if contract.status == NEEDS_CLARIFICATION:
            print(contract.summary())
            for question in contract.questions:
                print(f"  --answer {question.slot}=<value {question.unit}>   {question.text}")
            return 2
        run = run_study(contract, out_dir=args.out)
        print(run.summary())
        for name, path in run.files.items():
            print(f"  {name}: {path}")
        return 0 if run.decision == "APPROVED" else 1

    if args.command == "run-contract":
        contract = StudyContract.read(args.path)
        if contract.status == NEEDS_CLARIFICATION:
            print(contract.summary())
            return 2
        run = run_study(contract, out_dir=args.out)
        print(run.summary())
        return 0 if run.decision == "APPROVED" else 1

    parser.error("unknown command")
    return 64


def _answers(items: list[str]) -> dict[str, str]:
    answers: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--answer expects SLOT=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        answers[key.strip()] = value.strip()
    return answers


if __name__ == "__main__":
    sys.exit(main())
