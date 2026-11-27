"""Every stage of the trust study, in order.

    python examples/end_to_end/airfoil_trust/run_all.py            # everything, resuming what is on disk
    python examples/end_to_end/airfoil_trust/run_all.py --fast     # skip the two campaigns (analysis only)

The campaigns are the expensive part: stage 1 solves 60 designs on the production mesh plus the four meshes of the
grid-convergence study, and stage 5 adds 68 on the cambered family. Both are resumable -- a run already on disk is
skipped -- so the script is safe to stop and restart. Everything after them is minutes.
"""
from __future__ import annotations

import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
PY = sys.executable
STAGES = ["01_reference_campaign.py", "02_analysis.py", "03_physics_validation.py", "04_validated_domain.py",
          "05_camber_round.py", "06_revalidation.py", "07_report.py"]
CAMPAIGNS = {"01_reference_campaign.py", "05_camber_round.py"}


def run(script: str, *args: str) -> None:
    print(f"\n=== {script} {' '.join(args)}", flush=True)
    cmd = [PY, os.path.join(HERE, script), *args]
    done = subprocess.run(cmd, cwd=ROOT, check=False)
    if done.returncode != 0:
        raise SystemExit(f"{script} exited {done.returncode}")


def main() -> int:
    fast = "--fast" in sys.argv
    workers = sys.argv[sys.argv.index("--workers") + 1] if "--workers" in sys.argv else "4"
    for script in STAGES:
        if fast and script in CAMPAIGNS:
            print(f"skipping {script} (--fast)", flush=True)
            continue
        run(script, "--workers", workers) if script in CAMPAIGNS else run(script)
    print("\nall stages done: results/dossier.md has the evidence report, experiment.json the manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
