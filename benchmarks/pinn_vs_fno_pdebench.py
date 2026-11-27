"""PINN vs FNO on a real PDEBench case, with the VeriPhysics gates attached (INT-007).

Reads one trajectory of a PDEBench file -- over HTTP by default, so nothing is downloaded -- trains both families on
the same split with the same budget, scores them on the held-out tail of the trajectory, gates the winner's field
against the reference, and writes a markdown report plus the JSON behind it to ``benchmarks/_out/``.

    python -m benchmarks.pinn_vs_fno_pdebench                       # Sod6.hdf5 (4.9 MB, 1D Euler), reduced budget
    python -m benchmarks.pinn_vs_fno_pdebench --case 1D_Burgers_Sols_Nu0.001 --epochs 300
    python -m benchmarks.pinn_vs_fno_pdebench --file data/pdebench/Sod6.hdf5 --field density
    PINNEAPPLE_NETWORK_TESTS=1 python -m benchmarks.pinn_vs_fno_pdebench     # same thing, in CI

What the numbers mean: a reduced CPU budget (see ``--epochs``/``--width``), so this compares the two families on this
trajectory -- not against the published results of either. The report says so, and the gates say what was checked.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from pinneapple_arena.bundle_bench import BenchConfig, run_benchmark
from pinneapple_data.datasets import pdebench

OUT = Path(__file__).resolve().parent / "_out"


def _scorable_field(bundle) -> str:
    """The first field with variation in it: a channel that is constant (the velocity of a shock tube at t=0 is
    zero everywhere) has no relative L2 to report, and choosing it would fail after the training run."""
    import numpy as np

    for name in bundle.manifest.fields.names():
        values = np.asarray(bundle.field_array(name))
        if float(np.ptp(values)) > 0 and float(np.linalg.norm(values)) > 0:
            return name
    raise SystemExit(f"no field of {bundle.manifest.id} has any variation to score: "
                     f"{bundle.manifest.fields.names()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", default="Sod6", help=f"a curated PDEBench case: {', '.join(sorted(pdebench.DATASETS))}")
    parser.add_argument("--file", default="", help="a local HDF5 file or a URL (overrides --case)")
    parser.add_argument("--sample", type=int, default=0, help="which trajectory to read (Sod6 has one)")
    parser.add_argument("--field", default="", help="the field to reproduce (default: the case's first field)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--width", type=int, default=24)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=5e-2, help="relative L2 a model has to stay under")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args(argv)

    source = args.file or pdebench.dataset(args.case).url
    started = time.time()
    with pdebench.open_pdebench(source) as handle:
        bundle = handle.to_bundle(args.sample)
    field = args.field or _scorable_field(bundle)
    print(bundle.summary())
    print(f"field under test: {field}")

    config = BenchConfig(field=field, epochs=args.epochs, width=args.width, modes=args.modes, layers=args.layers,
                         tolerance=args.tolerance)
    report = run_benchmark(bundle, config)
    markdown = report.to_markdown()
    print("\n" + markdown)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"pinn_vs_fno_{bundle.manifest.id}".replace("/", "_")
    (out / f"{stem}.md").write_text(markdown, encoding="utf-8")
    (out / f"{stem}.json").write_text(json.dumps(report.to_dict(), indent=1), encoding="utf-8")
    print(f"wrote {out / (stem + '.md')} and .json  ({time.time() - started:.0f} s)")
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
