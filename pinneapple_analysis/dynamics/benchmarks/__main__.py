"""``python -m pinneapple_analysis.dynamics.benchmarks <framework> [--quick] [--seed N] [--json PATH]``."""
from __future__ import annotations

import argparse
import importlib
import json
import sys

from ._case import to_markdown

FRAMEWORKS = ("first_passage", "dimensional", "fisher", "early_warning", "koopman")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pinneapple_analysis.dynamics.benchmarks")
    ap.add_argument("framework", choices=FRAMEWORKS)
    ap.add_argument("--quick", action="store_true", help="smaller samples")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", help="also write the cases to this JSON file")
    a = ap.parse_args(argv)
    mod = importlib.import_module(f".{a.framework}", __package__)
    cases = mod.run(quick=a.quick, seed=a.seed)
    print(to_markdown(cases))
    if a.json:
        with open(a.json, "w") as fh:
            json.dump([c.to_dict() for c in cases], fh, indent=2, default=str)
    return 0


if __name__ == "__main__":
    sys.exit(main())
