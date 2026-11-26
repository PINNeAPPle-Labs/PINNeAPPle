"""The battle scoreboard (#373): run it, publish it, or check it against the record.

    python scripts/battle.py --run     # train both models over the stream, rewrite the results file and the page
    python scripts/battle.py --docs    # regenerate the page from the results file (no training)
    python scripts/battle.py --check   # declaration, record and page agree -- the CI gate, no training

``--only`` picks cases, ``--seeds`` overrides the protocol seeds. ``--seeds`` is a quick look, not a benchmark: the
results file and the page are only written when the protocol ran as declared, so a reduced run can never replace the
published numbers by accident.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pinneapple_arena import battle   # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="The battle scoreboard: two models, one stream, who left first.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--run", action="store_true", help="train both models, then write the results and the page")
    group.add_argument("--docs", action="store_true", help="regenerate the page from the results file")
    group.add_argument("--check", action="store_true", help="verify the declaration, the record and the page")
    parser.add_argument("--only", nargs="*", metavar="ID", help="case ids, e.g. --only burgers")
    parser.add_argument("--seeds", nargs="*", type=int, help="override the protocol seeds (a quick look)")
    args = parser.parse_args(argv)

    if args.check:
        problems = battle.check(ROOT)
        for problem in problems:
            print(f"problem: {problem}")
        print(f"{len(problems)} problem(s)" if problems else "the battle scoreboard is declared, recorded and in sync")
        return 1 if problems else 0

    if args.docs:
        path = ROOT / battle.RESULTS
        if not path.is_file():
            print(f"{battle.RESULTS.as_posix()} is missing: run --run first")
            return 1
        results = json.loads(path.read_text(encoding="utf-8"))
        battle.write(results, root=ROOT)
        print(f"rewrote {battle.PAGE.as_posix()} from {battle.RESULTS.as_posix()}")
        return 0

    results = battle.run_all(seeds=args.seeds, only=args.only)
    print(battle.table(results))
    declared = tuple(args.seeds) == tuple(battle.PROTOCOL["seeds"]) if args.seeds else True
    if not declared:
        print("seeds were overridden: the results file and the page were not touched")
        return 0
    path, page_path = battle.write(results, root=ROOT)
    print(f"wrote {path.relative_to(ROOT).as_posix()} and {page_path.relative_to(ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
