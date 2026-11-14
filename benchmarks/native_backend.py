"""The C backend timed against the NumPy reference on the same kernels (INT-015).

Three operations of the same five-point stencil, at grid sizes where the per-element cost of an interpreter
stops being academic. Nothing here is a claim about other machines: the report records this machine's compiler,
the flags and the grid, so the number can be read for what it is.

    python -m benchmarks.native_backend                    # 129 and 513 grids, best of 3
    python -m benchmarks.native_backend --sizes 65 257 --reps 5

Writes ``benchmarks/_out/native_backend.{json,md}``. When no compiler is found the report says so and records
only the reference timings — a benchmark that silently timed the fallback as if it were the native path would be
the worst possible result of this exercise.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

from pinneapple_core.native import NativeKernels, native_status, status

OUT = Path(__file__).resolve().parent / "_out"


def _grid(n: int) -> tuple:
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    return exact, 2.0 * np.pi ** 2 * exact, 1.0 / (n - 1)


def _stable_dt(n: int, alpha: float = 1.0, fraction: float = 0.45) -> float:
    """The largest ``dt`` under the explicit scheme's limit, at ``fraction`` of it."""
    h = 1.0 / (n - 1)
    return fraction * 0.5 / (alpha * (1.0 / (h * h) + 1.0 / (h * h)))


def _times(call, kernel, reps: int) -> list:
    """Every run of ``call(kernel)`` in ms — the kernel is passed in, so no closure captures a loop variable.

    All the runs are kept, not only the best: on this machine the C side repeats to within a few percent while
    the NumPy side swings by a factor of two between runs, and a table that printed only the best of each would
    hide exactly the part a reader needs to know before quoting a speedup.
    """
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        call(kernel)
        samples.append((time.perf_counter() - start) * 1e3)
    return samples


def run(sizes=(129, 513), reps: int = 5, heat_steps: int = 50, jacobi_steps: int = 200) -> dict:
    native = NativeKernels()
    reference = NativeKernels(force_reference=True)
    rows = []
    for n in sizes:
        u, f, h = _grid(n)
        alpha, dt = 1.0, _stable_dt(n)
        cases = [
            ("laplacian_2d", lambda k, u=u, h=h: k.laplacian_2d(u, h)),
            ("heat_step_2d", lambda k, u=u, alpha=alpha, dt=dt: k.heat_step_2d(u, alpha, dt, heat_steps)),
            ("jacobi_2d", lambda k, n=n, f=f, h=h: k.jacobi_2d(
                np.zeros((n, n)), f, h=h, steps=jacobi_steps, omega=2.0 / 3.0)),
        ]
        for name, call in cases:
            reference_result = call(reference)
            row = {"grid": f"{n}x{n}", "op": name}
            if native.is_native:
                native_result = call(native)
                row["max_abs_difference"] = float(np.max(np.abs(native_result - reference_result)))
                row["native_ms"] = _times(call, native, reps)
            row["numpy_ms"] = _times(call, reference, reps)
            if "native_ms" in row:
                row["speedup"] = float(np.min(row["numpy_ms"])) / float(np.min(row["native_ms"]))
                row["native_spread"] = float(np.max(row["native_ms"])) / float(np.min(row["native_ms"]))
            row["numpy_spread"] = max(row["numpy_ms"]) / min(row["numpy_ms"])
            rows.append(row)
    return {
        "machine": {
            "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "native": native.describe(),
        "status": status(),
        "reps": reps,
        "heat_steps": heat_steps,
        "jacobi_steps": jacobi_steps,
        "rows": rows,
    }


def to_markdown(report: dict) -> str:
    lines = [
        "# Native backend vs the NumPy reference",
        "",
        f"Machine: {report['machine']['platform']} · {report['machine']['processor']} · "
        f"Python {report['machine']['python']} · NumPy {report['machine']['numpy']}",
        "",
    ]
    native = report["native"]
    if native["native"]:
        lines += [f"Library: `{native['library']}`", f"Build: `{native['build']}`", ""]
    else:
        lines += [f"**Native backend not available:** {native['reason']}", ""]
        lines += ["The table below is the reference alone; there is no speedup to report.", ""]
    header = ("| grid | operation | C best (ms) | NumPy best (ms) | speedup | C spread | NumPy spread | "
              "max abs difference |")
    lines += [header, "|---|---|---|---|---|---|---|---|"]
    for row in report["rows"]:
        numpy = row["numpy_ms"]
        lines.append(
            "| {grid} | `{op}` | {native} | {numpy} | {speedup} | {cspread} | {spread} | {difference} |".format(
                grid=row["grid"],
                op=row["op"],
                native=f"{min(row['native_ms']):.3f}" if "native_ms" in row else "—",
                numpy=f"{min(numpy):.3f}",
                speedup=f"{row['speedup']:.1f}×" if "speedup" in row else "—",
                cspread=f"{row['native_spread']:.2f}×" if "native_spread" in row else "—",
                spread=f"{row['numpy_spread']:.2f}×",
                difference=f"{row['max_abs_difference']:.1e}" if "max_abs_difference" in row else "—",
            )
        )
    lines += [
        "",
        f"Each cell is the best of {report['reps']} runs on the kernels as written — not a microbenchmark of the "
        f"loop: the heat and Jacobi rows carry {report['heat_steps']} and {report['jacobi_steps']} steps "
        "respectively, and the speedup compares those two bests. The spread columns are (slowest run / fastest "
        "run) for each backend, and they are large on a loaded machine -- for both sides, not only NumPy's -- so "
        "a single speedup number here would claim a precision the measurement does not have. Quote the C column "
        "and the order of magnitude of the speedup; rerun on an idle machine for anything tighter. The difference column is the two backends' agreement, so a fast row that "
        "disagreed would be visible rather than reported as a win.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[129, 513])
    parser.add_argument("--reps", type=int, default=5)
    args = parser.parse_args(argv)
    report = run(sizes=tuple(args.sizes), reps=args.reps)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "native_backend.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    markdown = to_markdown(report)
    (OUT / "native_backend.md").write_text(markdown + "\n", encoding="utf-8")
    print(markdown)
    if not report["status"]["available"]:
        print(f"native backend unavailable: {native_status()['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
