"""Write the three trust records of #345 to ``results/trust_three_kinds/``.

    python scripts/trust_three_kinds.py                       # the three files, with their badges
    python scripts/trust_three_kinds.py --out /tmp/trust      # somewhere else

The three records come from ``pinneapple_analysis.trust.producers`` — a weather forecast, an online ensemble and a
CFD surrogate — and they are written by ``Trust.to_json`` and nothing else, so the three files are comparable key for
key. Each one is validated against the versioned schema on the way out: a record that does not validate is a failure
of this script, not a warning.

This script is glue: the producers, the measurements and the schema live in the package.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pinneapple_analysis.trust.producers import emit  # noqa: E402
from pinneapple_analysis.trust.record import Trust  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/trust_three_kinds", help="folder for the three JSON files")
    args = ap.parse_args(argv)

    written = emit(args.out)
    bad = 0
    for kind, path in written.items():
        document = Path(path).read_text(encoding="utf-8")
        problems = Trust.validate_document(document)
        badge = Trust.from_json(document).badge()
        print(f"{kind:<10} {badge:<9} {path}   {'valid' if not problems else 'INVALID: ' + '; '.join(problems)}")
        bad += len(problems)
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
