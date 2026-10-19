"""Red team / blue team for the validation checks on Darcy 2-D and Burgers 1-D (#707).

The red team looks for wrong outputs that pass every check (escapes) and correct ones that fail (false alarms);
the blue team calibrates one combined test with a stated false-alarm rate and adds checks until every stored escape
is caught. Run from the repository root:

    python examples/validation_arena/run_arena.py                 # full size, alpha = 1 %, ~10-20 min on a laptop
    python examples/validation_arena/run_arena.py --small         # a minute or two
    pinneapple check --arena-cases examples/validation_arena/out/cases/darcy_2d.json   # replay in CI

Everything is in pinneapple_analysis.validation.arena; this script only calls it.
"""
import sys
from pathlib import Path

from pinneapple_analysis.validation.arena.__main__ import main

if __name__ == "__main__":
    here = Path(__file__).resolve().parent / "out"
    args = sys.argv[1:]
    if "--out" not in args:
        args += ["--out", str(here)]
    if "--cases" not in args:
        args += ["--cases", str(here / "cases")]
    sys.exit(main(args))
