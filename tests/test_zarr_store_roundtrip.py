"""UPDZarrStore write/read round trip on the installed Zarr (2 or 3), without deprecated calls (#239)."""
import warnings

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")

from pinneapple_data.physical_sample import PhysicalSample  # noqa: E402
from pinneapple_data.zarr_store import UPDZarrStore  # noqa: E402


def test_write_read_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    samples = [PhysicalSample(state={"u": rng.random((4, 3)).astype("float32"), "k": np.arange(4) + i},
                              provenance={"run": i}) for i in range(3)]
    root = str(tmp_path / "s.zarr")
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)          # create_dataset is deprecated in Zarr 3
        if hasattr(zarr, "errors") and hasattr(zarr.errors, "ZarrDeprecationWarning"):
            warnings.simplefilter("error", zarr.errors.ZarrDeprecationWarning)
        UPDZarrStore.write(root, samples, chunks={"u": (1, 4, 3)})
    store = UPDZarrStore(root)
    assert store.num_samples() == 3 and set(store.field_names()) == {"u", "k"}
    for i, s in enumerate(samples):
        r = store.read_sample(i)
        assert np.allclose(np.asarray(r.state["u"]), s.state["u"])
        assert np.array_equal(np.asarray(r.state["k"]), s.state["k"])
        assert r.provenance["run"] == i
