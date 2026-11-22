"""CI gate: every examples/* folder has a valid experiment.json, every Lab run reads as a valid manifest, and every
``apps/*/cinema.json`` (the video products made from an app) names its bytes.

    python scripts/check_experiment_manifests.py        # exit 1 and one line per problem, else 0

The logic is pinneapple_lab.manifest.check_repo and check_cinema; fix a folder with
`python -m pinneapple_lab register <folder>`.
"""
import sys

from pinneapple_lab.manifest import check_cinema, check_repo

if __name__ == "__main__":
    problems = check_repo() + check_cinema()
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)" if problems else "every experiment has a valid manifest")
    sys.exit(1 if problems else 0)
