"""PDEBench -> the universal physics data schema -> verification gates, end to end.

What it shows
-------------
1. a PDEBench file in the archive's root-dataset layout is opened through the adapter (a local file here, so the
   example runs offline; the same call takes a DaRUS URL and range-reads it);
2. the reader reports what it found -- layout, which leading axis is samples and which is time, the fields after
   canonicalisation -- before any array is trusted;
3. the sample becomes a ``PhysicsDataBundle``: geometry, mesh axes with units, fields with node/cell location,
   physics, provenance (licence, citation, input hash) and verification;
4. the bundle is saved, reloaded with its digest checked, and put through the gates;
5. the report is what a reviewer reads: per gate, the number it was decided on.

The data here is synthetic but written to the published layout, so the numbers are a check of the machinery rather
than a statement about the Burgers equation. Set ``PINNEAPPLE_NETWORK_TESTS=1`` to run the same path against
``Sod6.hdf5`` on the real archive (4.9 MB, read over HTTP without downloading it).
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from pinneapple_data.datasets import pdebench
from pinneapple_data.schema import PhysicsDataBundle, run_gates


def write_pdebench_fixture(path: Path, *, n_samples: int = 4, nt: int = 32, nx: int = 64) -> Path:
    """A file in the layout of the archive's 1D solver cases (root datasets + coordinates + a parameter)."""
    rng = np.random.default_rng(0)
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    t = np.linspace(0.0, 1.0, nt, dtype=np.float32)
    # a decaying travelling wave, per sample, so the integral is non-zero and conservation is measurable
    waves = np.stack([
        (1.0 + 0.2 * np.sin(2 * np.pi * (x[None, :] - speed * t[:, None]))).astype(np.float32)
        for speed in np.linspace(0.1, 0.4, n_samples)
    ])
    with h5py.File(path, "w") as f:
        f.create_dataset("density", data=waves)
        f.create_dataset("Vx", data=rng.normal(0, 0.01, size=waves.shape).astype(np.float32))
        f.create_dataset("x-coordinate", data=x)
        f.create_dataset("t-coordinate", data=t)
        f.create_dataset("nu", data=np.linspace(0.001, 0.01, n_samples, dtype=np.float32))
        f.attrs["case"] = "synthetic 1D advection"
    return path


def main() -> int:
    workdir = Path(tempfile.mkdtemp(prefix="pinneapple-pdebench-"))
    source = write_pdebench_fixture(workdir / "1D_Burgers_Sols_Nu0.01.hdf5")
    print(f"fixture: {source}  ({source.stat().st_size / 1024:.0f} kB)")

    with pdebench.open_pdebench(source) as handle:
        print(f"\nlayout           {handle.layout}")
        print(f"axis flags       {handle.axis_flags}")
        print(f"samples          {handle.n_samples}")
        print(f"fields in file   {handle.fields}")
        print(f"spatial dims     {handle.spatial_dims}")
        bundle = handle.to_bundle(index=1)

    print("\n" + bundle.summary())

    saved = bundle.save(workdir / "bundle")
    reloaded = PhysicsDataBundle.load(saved)
    print(f"\nsaved to         {saved}   digest {reloaded.digest()[:12]}...  intact={reloaded.verify()['ok']}")

    report = run_gates(reloaded, conserved="density")
    print("\n" + report.summary())
    report.apply(reloaded)
    print(f"\nmanifest says    {'passed' if reloaded.manifest.verification.passed else 'not passed'} "
          f"(required: {reloaded.manifest.verification.required})")

    if os.environ.get("PINNEAPPLE_NETWORK_TESTS") == "1":
        print("\nlive archive: Sod6.hdf5 (4.9 MB) read over HTTP")
        live = pdebench.load_bundle(pdebench.darus_url(133150))
        print(live.summary())
        print(run_gates(live, conserved="density").summary())
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
