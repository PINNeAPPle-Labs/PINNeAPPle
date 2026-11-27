"""The PhysicsNeMo dictionary bridge (INT-014): a bundle as the ``dict`` its training loops iterate.

The contract is PhysicsNeMo's own: its benchmark datapipes yield ``{"permeability": ..., "darcy": ...}`` -- a plain
dictionary of named tensors -- and its pointwise pipelines add ``"pos"``. These tests pin that shape, the inverse
conversion and the honest failure when the package is not installed.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.schema import AxisSpec, Field, Physics, PhysicsDataBundle
from pinneapple_data.schema.physicsnemo import (
    POS,
    BundleDictDataset,
    from_dict,
    to_dict,
    to_physicsnemo_dataset,
)

NX, NT = 12, 5


def bundle() -> PhysicsDataBundle:
    x = np.linspace(0.0, 1.0, NX)
    t = np.linspace(0.0, 1.0, NT)
    u = np.array([np.sin(2 * np.pi * (x - ti)) for ti in t], dtype=np.float32)
    return PhysicsDataBundle.from_arrays(
        {"axis:t": t, "axis:x": x, "field:u": u, "field:k": (0.5 + u * 0.1).astype(np.float32)},
        fields=[Field.from_array("u", u, axes=("t", "x")),
                Field.from_array("k", (0.5 + u * 0.1).astype(np.float32), axes=("t", "x"))],
        axes=[AxisSpec("t", "s"), AxisSpec("x", "m")], physics=Physics(), source="bridge test",
        license="CC0-1.0", kind="synthetic")


def test_every_field_becomes_a_key_and_the_coordinates_are_pos():
    d = to_dict(bundle())
    assert set(d) == {"u", "k", POS}
    assert d["u"].shape == (NT, NX)                        # the trajectory is passed through, not flattened
    assert d[POS].shape == (NX, 1)                         # one point per column of the grid


def test_a_step_selects_one_time_index():
    d = to_dict(bundle(), step=1, fields=("u",))
    assert set(d) == {"u", POS}
    assert d["u"].shape == (NX,)
    assert np.allclose(d["u"], np.asarray(bundle().field_array("u"))[1])


def test_pos_can_be_left_out_for_a_pipeline_that_does_not_want_it():
    assert POS not in to_dict(bundle(), pos=False)


def test_an_unstructured_bundle_takes_its_positions_from_the_node_array():
    from pinneapple_data.cae.model import CellBlock, Mesh
    from pinneapple_data.schema.cae import bundle_from_mesh

    mesh = Mesh(points=np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
                blocks=[CellBlock(type="triangle", conn=np.array([[0, 1, 2]]))],
                point_data={"T": np.array([1.0, 2.0, 3.0])})
    d = to_dict(bundle_from_mesh(mesh, source="tri", license="CC0-1.0"))
    assert d[POS].shape == (3, 3) and np.allclose(d[POS][1], [1.0, 0.0, 0.0])
    assert np.allclose(d["T"], [1.0, 2.0, 3.0])


def test_the_dictionary_comes_back_as_a_bundle():
    original = bundle()
    back = from_dict(to_dict(original), source="round trip", license="CC0-1.0", field="u",
                     axes=("t", "x"), axis_values={"t": np.asarray(original.axis_array("t")),
                                                   "x": np.asarray(original.axis_array("x"))})
    assert set(back.manifest.fields.names()) == {"u", "k"}
    assert np.allclose(back.field_array("u"), original.field_array("u"))
    assert back.manifest.provenance.license == "CC0-1.0"


def test_a_dictionary_of_only_positions_is_refused():
    with pytest.raises(ValueError, match="no field"):
        from_dict({POS: np.zeros((3, 2))}, source="empty", license="CC0-1.0")


def test_an_axis_without_coordinates_is_refused():
    with pytest.raises(ValueError, match="no coordinates"):
        from_dict({"u": np.zeros((2, 3))}, source="bad", license="CC0-1.0", field="u", axes=("x",))


def test_the_dataset_iterates_over_bundles():
    items = list(BundleDictDataset([bundle(), bundle()], fields=("u",), tensors=False))
    assert len(items) == 2 and isinstance(items[0]["u"], np.ndarray)


def test_torch_tensors_when_torch_is_installed():
    torch = pytest.importorskip("torch")
    item = BundleDictDataset([bundle()], fields=("u",))[0]
    assert isinstance(item["u"], torch.Tensor) and item["u"].dtype == torch.float32
    assert item["u"].shape == (NT, NX)


def test_an_empty_dataset_is_refused():
    with pytest.raises(ValueError, match="no bundles"):
        BundleDictDataset([])


def test_the_datapipe_says_what_to_install_when_physicsnemo_is_absent(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("physicsnemo"):
            raise ImportError("no physicsnemo here")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(ImportError, match="nvidia-physicsnemo"):
        to_physicsnemo_dataset([bundle()])
