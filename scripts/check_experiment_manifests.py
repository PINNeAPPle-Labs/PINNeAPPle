"""CI gate: every examples/* folder has a valid experiment.json and every Lab run reads as a valid manifest.

    python scripts/check_experiment_manifests.py        # exit 1 and one line per problem, else 0

The logic is pinneapple_lab.manifest.check_repo; fix a folder with `python -m pinneapple_lab register <folder>`.
"""
import sys

from pinneapple_lab.manifest import check_repo

if __name__ == "__main__":
    problems = check_repo()
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)" if problems else "every experiment has a valid manifest")
    sys.exit(1 if problems else 0)
