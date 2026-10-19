"""Validation reports of the zoo models (#706).

    python scripts/zoo/validation_reports.py build [names]   # store pinneapple_hub/zoo/validation_reports/<name>.json
    python scripts/zoo/validation_reports.py check [names]   # fail (exit 1) when a model lost a check it passed

Run by .github/workflows/publish-zoo.yml before any upload: a new model version that loses a check the previous
version passed is not published.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pinneapple_hub import zoo  # noqa: E402
from pinneapple_hub.zoo import validation_reports as vr  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "check"])
    ap.add_argument("names", nargs="*")
    a = ap.parse_args(argv)
    names = a.names or zoo.list()
    bad = 0
    for n in names:
        if a.cmd == "build":
            rep = vr.build(n)
            print(f"{n}: {rep.verdict}")
            continue
        d, fresh = vr.check_release(n)
        print(d.to_markdown())
        if d.regressed:
            bad += 1
            print(f"::error::{n}: the validation regressed against the stored report")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
