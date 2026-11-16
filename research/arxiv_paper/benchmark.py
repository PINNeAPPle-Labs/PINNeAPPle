"""Auditable 1D Poisson analytical reference, not a trained PINN benchmark."""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path


def exact(x):
    return math.sin(math.pi * x)


def forcing(x):
    """-u'' = f on [0,1], u(0)=u(1)=0."""
    return math.pi ** 2 * math.sin(math.pi * x)


def central_difference_residual(x, h):
    """Approximate -u'' - f for the analytic solution."""
    second = (exact(x - h) - 2 * exact(x) + exact(x + h)) / (h * h)
    return -second - forcing(x)


def evaluate(n=101, h=1e-3):
    if n < 3 or not (0 < h < 1 / (n - 1)):
        raise ValueError("Require n >= 3 and 0 < h < grid spacing")
    xs = [i / (n - 1) for i in range(1, n - 1)]
    residuals = [central_difference_residual(x, h) for x in xs]
    return {
        "problem": "poisson_1d_analytic_reference",
        "equation": "-u_xx = pi^2 sin(pi x), u(0)=u(1)=0",
        "grid_points": n,
        "difference_step": h,
        "max_abs_residual": max(abs(v) for v in residuals),
        "rms_residual": math.sqrt(sum(v*v for v in residuals) / len(residuals)),
        "boundary_max_abs_error": max(abs(exact(0)), abs(exact(1))),
        "status": "analytical_reference_only",
    }


def git_sha():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--n", type=int, default=101)
    parser.add_argument("--h", type=float, default=1e-3)
    args = parser.parse_args()
    result = evaluate(args.n, args.h)
    result["provenance"] = {
        "git_sha": git_sha(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
