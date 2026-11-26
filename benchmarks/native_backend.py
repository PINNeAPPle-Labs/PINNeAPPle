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
import statistics
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


def _one(call, kernel) -> float:
    start = time.perf_counter()
    call(kernel)
    return (time.perf_counter() - start) * 1e3


def _pairs(call, native, reference, reps: int) -> list:
    """``(native_ms, numpy_ms)`` timed **back to back**, alternating which goes first.

    Timing the two backends in separate blocks was the first version, and on a machine that is also running
    other work it produced speedups that disagreed with themselves by a factor of four between runs: the C block
    and the NumPy block each got a different share of the interference. Interleaving them pairs every C
    measurement with an adjacent NumPy one, so the ratio of a pair is a comparison of the same machine state,
    and the median of those ratios is the number that survives a noisy host.
    """
    samples = []
    for rep in range(reps):
        if rep % 2 == 0:
            native_ms = _one(call, native)
            numpy_ms = _one(call, reference)
        else:
            numpy_ms = _one(call, reference)
            native_ms = _one(call, native)
        samples.append((native_ms, numpy_ms))
    return samples


def run(sizes=(129, 513), reps: int = 11, heat_steps: int = 50, jacobi_steps: int = 200) -> dict:
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
            reference_result = call(reference)          # warm-up, and the reference every difference is against
            row = {"grid": f"{n}x{n}", "op": name}
            if native.is_native:
                native_result = call(native)
                row["max_abs_difference"] = float(np.max(np.abs(native_result - reference_result)))
                pairs = _pairs(call, native, reference, reps)
                ratios = [numpy / native_ms for native_ms, numpy in pairs]
                row["pairs_ms"] = [{"native": c, "numpy": m} for c, m in pairs]
                row["native_ms"] = [c for c, _ in pairs]
                row["numpy_ms"] = [m for _, m in pairs]
                row["speedup"] = float(statistics.median(ratios))
                row["speedup_min"] = float(min(ratios))
                row["speedup_max"] = float(max(ratios))
            else:
                row["numpy_ms"] = [_one(call, reference) for _ in range(reps)]
            row["native_spread"] = (
                float(max(row["native_ms"])) / float(min(row["native_ms"])) if "native_ms" in row else None
            )
            row["numpy_spread"] = float(max(row["numpy_ms"])) / float(min(row["numpy_ms"]))
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
    header = ("| grid | operation | C median (ms) | NumPy median (ms) | speedup (median of paired ratios) | "
              "ratio range | max abs difference |")
    lines += [header, "|---|---|---|---|---|---|---|"]
    for row in report["rows"]:
        has_native = "native_ms" in row
        lines.append(
            "| {grid} | `{op}` | {native} | {numpy} | {speedup} | {rng} | {difference} |".format(
                grid=row["grid"],
                op=row["op"],
                native=f"{statistics.median(row['native_ms']):.3f}" if has_native else "—",
                numpy=f"{statistics.median(row['numpy_ms']):.3f}",
                speedup=f"{row['speedup']:.1f}×" if has_native else "—",
                rng=f"{row['speedup_min']:.1f}–{row['speedup_max']:.1f}×" if has_native else "—",
                difference=f"{row['max_abs_difference']:.1e}" if has_native else "—",
            )
        )
    lines += [
        "",
        f"{report['reps']} pairs per row, the two backends timed back to back and the order alternated, on the "
        f"kernels as written — the heat and Jacobi rows carry {report['heat_steps']} and "
        f"{report['jacobi_steps']} steps respectively. The speedup is the median of the per-pair ratios and the "
        "range is the smallest and largest of those same ratios: on a host that is also running other work the "
        "range is wide for both backends, and a paired ratio is the only comparison that survives it. The "
        "difference column is the two backends' agreement, so a fast row that disagreed would be visible rather "
        "than reported as a win.",
        "",
    ]
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="+", default=[129, 513])
    parser.add_argument("--reps", type=int, default=11)
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
