"""SPE10: the archive's own text layout, the cell ordering, and the licence that is the caller's.

The ordering is the part worth pinning: the file gives no index, so the reader's claim that ``i`` varies fastest is
the difference between a permeability field and noise. The test writes a grid whose value encodes its own cell index,
so a wrong reshape fails loudly instead of returning plausible numbers.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_data.datasets import spe10
from pinneapple_data.schema import run_gates

SHAPE = (2, 3, 4)          # a small grid laid out the way the archive lays out 60 x 220 x 85


def write_tables(root, shape=SHAPE):
    """Two tables in the archive's format: six values per line for the permeability, six for the porosity."""
    cells = int(np.prod(shape))
    index = np.arange(cells, dtype=np.float64)
    perm = np.stack([index, index + 0.5, index + 0.25], axis=1)
    perm_path = root / "spe_perm.dat"
    phi_path = root / "spe_phi.dat"
    np.savetxt(perm_path, perm, fmt="%.6g")
    np.savetxt(phi_path, index.reshape(-1, 1), fmt="%.6g")
    return perm_path, phi_path


def test_a_cell_lands_where_the_archive_puts_it(tmp_path):
    perm_path, phi_path = write_tables(tmp_path)
    perm = spe10.read_perm(perm_path, shape=SHAPE)
    assert perm.shape == SHAPE + (3,)
    # cell (i, j, k) holds the flat index i + 2 * (j + 3 * k): i varies fastest
    for i, j, k in ((0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 2, 3)):
        assert perm[i, j, k, 0] == pytest.approx(i + 2 * (j + 3 * k))
    assert spe10.read_perm(perm_path, shape=SHAPE)[0, 0, :, 0].tolist() == [0.0, 6.0, 12.0, 18.0]
    assert spe10.read_phi(phi_path, shape=SHAPE)[1, 2, 3] == pytest.approx(1 + 2 * (2 + 3 * 3))


def test_a_table_of_the_wrong_size_is_refused(tmp_path):
    perm_path, _ = write_tables(tmp_path)
    with pytest.raises(ValueError, match="the grid asked for"):
        spe10.read_perm(perm_path, shape=(3, 3, 4))


def test_the_bundle_carries_feet_and_cell_centres(tmp_path):
    perm_path, phi_path = write_tables(tmp_path)
    bundle = spe10.load_bundle(perm_path, phi_path, license="SPE CSP dataset terms", shape=SHAPE)
    assert bundle.axes == ["x", "y", "z"]
    assert [a.unit for a in bundle.manifest.mesh.axes] == ["ft", "ft", "ft"]
    x = bundle.axis_array("x")
    assert x[0] == pytest.approx(spe10.CELL_SIZE_FT[0] / 2)                    # cell centres, not corners
    z = bundle.axis_array("z")
    assert z[0] == pytest.approx(spe10.TOP_DEPTH_FT + spe10.CELL_SIZE_FT[2] / 2)
    assert bundle.manifest.fields.get("kx").location == "cell"
    assert bundle.manifest.fields.get("kx").unit == "mD"
    assert bundle.manifest.fields.get("porosity").unit == "1"


def test_the_bundle_passes_the_gates_on_a_well_formed_archive(tmp_path):
    perm_path, phi_path = write_tables(tmp_path)
    report = run_gates(spe10.load_bundle(perm_path, phi_path, license="SPE CSP dataset terms", shape=SHAPE))
    assert report.ok is True, report.summary()


def test_the_licence_is_a_required_argument(tmp_path):
    perm_path, _ = write_tables(tmp_path)
    with pytest.raises(TypeError):
        spe10.load_bundle(perm_path, None, shape=SHAPE)          # type: ignore[call-arg]
    with pytest.raises(ValueError, match="states no licence"):
        spe10.load_bundle(perm_path, None, license="  ", shape=SHAPE)


def test_the_three_components_can_be_named(tmp_path):
    perm_path, _ = write_tables(tmp_path)
    bundle = spe10.load_bundle(perm_path, None, license="x", shape=SHAPE, components=("kh", "kh2", "kv"))
    assert set(bundle.manifest.fields.names()) == {"kh", "kh2", "kv"}
    assert bundle.manifest.metadata["spe10"]["components"] == ["kh", "kh2", "kv"]


def test_the_ordering_and_the_component_caveat_are_recorded_in_the_bundle(tmp_path):
    perm_path, _ = write_tables(tmp_path)
    meta = spe10.load_bundle(perm_path, None, license="x", shape=SHAPE).manifest.metadata["spe10"]
    assert "i + 60 * (j + 220 * k)" in meta["ordering"]
    assert "not recoverable from the numbers" in meta["component_order_note"]


def test_the_fine_grid_is_the_default_and_its_size_is_what_the_reader_checks():
    assert spe10.SHAPE == (60, 220, 85) and int(np.prod(spe10.SHAPE)) == 1_122_000
    assert spe10.CELL_SIZE_FT == (20.0, 10.0, 2.0) and spe10.TARBERT_LAYERS == 35
