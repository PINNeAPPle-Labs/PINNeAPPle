"""30_zarr_data_pipeline.py — UPD + Zarr data pipeline.

Demonstrates:
- PhysicalSample: the current Unified Physical Data container
- UPDZarrStore: writing PhysicalSample objects to a Zarr store
- ZarrUPDIterable: lazy, sample-by-sample reading from the store
- A small CPU-only analytic heat-data example with validation

The former ZarrDatasetWriter, ZarrDatasetReader, byte-cache, prefetch, and
geometry-specific sampler APIs are no longer public APIs. This example uses the
current supported UPD and Zarr interfaces instead.

Run:

    python templates/30_zarr_data_pipeline.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from pinneapple_data.physical_sample import PhysicalSample
from pinneapple_data.zarr_iterable import ZarrUPDIterable
from pinneapple_data.zarr_store import UPDZarrStore

GRID_SIZE = 16
N_SAMPLES = 8


def analytic_temperature(
    points: np.ndarray, top_temperature: float, bottom_temperature: float
) -> np.ndarray:
    """Return the linear 2-D heat profile at ``points`` with shape (N, 2)."""
    y = points[:, 1:2]
    return (bottom_temperature + (top_temperature - bottom_temperature) * y).astype(np.float32)


def make_sample(
    sample_id: int, top_temperature: float, bottom_temperature: float
) -> PhysicalSample:
    """Create one grid-based UPD sample with analytic temperature values."""
    axis = np.linspace(0.0, 1.0, GRID_SIZE, dtype=np.float32)
    x, y = np.meshgrid(axis, axis, indexing="xy")
    points = np.column_stack((x.ravel(), y.ravel())).astype(np.float32)
    temperature = analytic_temperature(points, top_temperature, bottom_temperature)

    return PhysicalSample(
        state={"xy": points, "temperature": temperature},
        domain={"type": "grid", "shape": [GRID_SIZE, GRID_SIZE]},
        schema={
            "equation": "steady linear temperature profile",
            "units": {"xy": "m", "temperature": "K"},
        },
        provenance={
            "sample_id": str(sample_id),
            "generator": "analytic_linear_heat_profile",
            "top_temperature": top_temperature,
            "bottom_temperature": bottom_temperature,
        },
    )


def main() -> None:
    """Write, lazily read, and validate a small CPU-only Zarr dataset."""
    temporary_directory = Path(tempfile.mkdtemp(prefix="pinneapple_zarr_"))
    store_path = temporary_directory / "heat_samples.zarr"

    try:
        rng = np.random.default_rng(42)
        samples = [
            make_sample(
                sample_id=index,
                top_temperature=float(rng.uniform(50.0, 200.0)),
                bottom_temperature=float(rng.uniform(0.0, 50.0)),
            )
            for index in range(N_SAMPLES)
        ]

        UPDZarrStore.write(
            str(store_path),
            samples,
            manifest={"name": "analytic_linear_heat_samples", "version": "1"},
        )
        print(f"Wrote {N_SAMPLES} samples to: {store_path}")

        store = UPDZarrStore(str(store_path), mode="r")
        print(f"Stored fields: {store.field_names()}")
        print(f"Stored samples: {store.num_samples()}")

        dataset = ZarrUPDIterable(
            str(store_path),
            fields=["xy", "temperature"],
        )

        maximum_error = 0.0
        loaded_samples = 0

        for sample in dataset:
            points = sample.state["xy"].detach().cpu().numpy()
            temperature = sample.state["temperature"].detach().cpu().numpy()
            top_temperature = float(sample.provenance["top_temperature"])
            bottom_temperature = float(sample.provenance["bottom_temperature"])

            expected = analytic_temperature(
                points,
                top_temperature=top_temperature,
                bottom_temperature=bottom_temperature,
            )
            maximum_error = max(maximum_error, float(np.abs(temperature - expected).max()))
            loaded_samples += 1

        assert loaded_samples == N_SAMPLES
        assert maximum_error < 1e-6

        print(f"Lazy-read {loaded_samples} samples successfully.")
        print(f"Maximum analytic temperature error: {maximum_error:.6f}")
        print("Zarr data pipeline completed successfully on CPU.")

    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        print("Temporary Zarr store cleaned up.")


if __name__ == "__main__":
    main()
