"""The Ashton-family datasets offline: the reader on a fixture in the same layout, and the measurements.

The three real datasets are 41 TB and need the network, so nothing here downloads anything. What is exercised
is everything that would be wrong silently if it were wrong: the layout resolution, the VTK XML reader (partial
reads included), the force integral against a hand-computed case, the metric definitions, the licence and the
credit travelling with the data, the declared baselines against the papers they are quoted from, and the
cross-family experiment whose ordering is the point of reading three datasets instead of one.
"""
from __future__ import annotations

import base64
import math
import zlib

import numpy as np
import pytest

from pinneapple_data.ashton import benchmark, catalog, fixture, loader, trust
from pinneapple_data.ashton.vtk_xml import read_surface, surface_quadrature

BODY = fixture.Params(length=1.02, width=0.39, height=0.29, slant=25.0, roundness=3.0, taper=0.05, clearance=0.06)


# ---------------------------------------------------------------------------
# the catalogue: licence, credit, layout
# ---------------------------------------------------------------------------

def test_the_three_datasets_carry_their_licence_and_their_credit():
    assert sorted(catalog.dataset_ids()) == ["ahmedml", "drivaerml", "windsorml"]
    assert set(catalog.licences().values()) == {"CC-BY-SA-4.0"}
    for name in catalog.dataset_ids():
        data = catalog.dataset(name)
        assert data.licence == "CC-BY-SA-4.0" and "by-sa" in catalog.LICENCE_URL
        assert "Ashton" in data.attribution
        assert data.paper in data.attribution or data.paper.removeprefix("arXiv:") in data.attribution
        assert data.repo.startswith("neashton/") and data.url.startswith("https://huggingface.co/datasets/")
    assert catalog.dataset("AHMEDML") is catalog.dataset("ahmedml")            # case-insensitive, same object
    with pytest.raises(KeyError):
        catalog.dataset("drivaernet")


def test_the_layout_is_the_one_the_cards_describe():
    ahmed, drivaer, windsor = (catalog.dataset(name) for name in catalog.dataset_ids())
    assert ahmed.file("surface", 7) == "run_7/boundary_7.vtp"
    assert ahmed.file("areas", 7) == "run_7/boundary_cell_area_7.npy"
    assert ahmed.file("forces", 7) == "run_7/force_mom_7.csv"
    assert ahmed.file("geometry", 7) == "run_7/ahmed_7.stl"
    assert windsor.file("surface", 7) == "run_7/boundary_7.vtu"                # the one container that differs
    assert windsor.file("areas", 7) == "run_7/boundary_dual_area_7.npy"
    assert windsor.quadrature == "point" and ahmed.quadrature == "cell"
    assert ahmed.n_cases == 500 and drivaer.n_cases == 484 and windsor.n_cases == 350
    assert 167 in drivaer.withdrawn and drivaer.runs[0] == 1 and windsor.runs[0] == 0
    with pytest.raises(KeyError):
        ahmed.file("nonsense", 1)
    notes = catalog.layout_notes()
    for fragment in ("boundary_<i>.vtp", "boundary_dual_area_<i>.npy", "volume_<i>.vtu", "geo_parameters_<i>.csv"):
        assert fragment in notes


def test_the_reference_quantities_are_the_ones_the_papers_print():
    ahmed = catalog.dataset("ahmedml")
    assert (ahmed.velocity, ahmed.density, ahmed.a_ref) == (1.0, 1.0, 0.112)
    drivaer = catalog.dataset("drivaerml")
    assert drivaer.velocity == 38.889 and drivaer.a_ref == 2.17 and drivaer.l_ref == 2.78618
    windsor = catalog.dataset("windsorml")
    assert (windsor.velocity, windsor.density, windsor.a_ref) == (40.0, 1.31, 0.112)


# ---------------------------------------------------------------------------
# the reader on the fixture
# ---------------------------------------------------------------------------

def test_surface_reader_reads_the_polygonal_layout(tmp_path):
    fixture.write_case(tmp_path, 1, BODY, dataset="ahmedml")
    surface = read_surface(tmp_path / "run_1" / "boundary_1.vtp")
    assert surface.grid == "polydata"
    assert surface.n_points == 366 and surface.n_cells == 728           # 28 ring points x 13 stations + 2 centres
    assert surface.cells_per_shape == {3: 728}                          # triangles only
    assert sorted(surface.point_data) == [] and sorted(surface.cell_data) == ["Cp", "p"]
    quadrature = surface_quadrature(surface)
    assert quadrature["area"] > 1.0
    assert quadrature["integrated_normal_ratio"] < 1e-9                 # a closed body: the normals cancel


def test_surface_reader_reads_the_ascii_and_the_binary_encoding_alike(tmp_path):
    fixture.write_case(tmp_path, 1, BODY, dataset="ahmedml", fmt="ascii")
    fixture.write_case(tmp_path, 2, BODY, dataset="ahmedml", fmt="binary")
    ascii_surface = read_surface(tmp_path / "run_1" / "boundary_1.vtp")
    binary_surface = read_surface(tmp_path / "run_2" / "boundary_2.vtp")
    assert np.allclose(ascii_surface.points, binary_surface.points, atol=1e-6)
    assert np.allclose(ascii_surface.cell_data["Cp"], binary_surface.cell_data["Cp"], atol=1e-6)
    full_bytes = binary_surface.bytes_decoded
    partial = read_surface(tmp_path / "run_2" / "boundary_2.vtp", arrays=["Cp"])
    assert partial.bytes_decoded < full_bytes and sorted(partial.cell_data) == ["Cp"]


_VTK_TYPE = {"Float32": "<f4", "Float64": "<f8", "Int32": "<i4", "Int64": "<i8"}


def _compressed(values, vtk_type: str, *, block_bytes: int = 24) -> str:
    """One ``DataArray`` payload the way VTK writes a compressed one.

    A ``UInt32`` header of ``[nBlocks, blockSize, lastBlockSize, compressedSize x nBlocks]``, then one zlib stream per
    block -- and the header and the data are two separate base64 streams, so a ``==`` lands *inside* the payload.
    A block compression does not shrink is written raw, exactly as VTK does.
    """
    raw = np.asarray(values, dtype=_VTK_TYPE[vtk_type]).tobytes()
    blocks = [raw[i:i + block_bytes] for i in range(0, len(raw), block_bytes)]
    stored = [zlib.compress(b) if len(zlib.compress(b)) < len(b) else b for b in blocks]
    head = np.array([len(blocks), block_bytes, len(blocks[-1]), *map(len, stored)], dtype="<u4").tobytes()
    return base64.b64encode(head).decode() + base64.b64encode(b"".join(stored)).decode()


def test_a_compressed_surface_reads_its_blocks_and_ignores_the_information_keys(tmp_path):
    """The SHIFT layout: every array compressed, a block header ahead of the blocks, XML children after them."""
    points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]], dtype=np.float32)
    cp = np.array([0.5, 0.25, 0.75, 0.125], dtype=np.float32)
    arrays = {
        "Points": _compressed(points, "Float32"),
        "connectivity": _compressed(np.array([0, 1, 2, 1, 2, 3], dtype=np.int64), "Int64"),
        "offsets": _compressed(np.array([3, 6], dtype=np.int64), "Int64"),          # end offsets, as VTK writes them
    }
    information = ("<InformationKey name='L2_NORM_RANGE' location='vtkDataArray' length='2'>"
                   "<Value index='0'>0.125</Value><Value index='1'>0.75</Value></InformationKey>")
    file = tmp_path / "compressed.vtp"
    file.write_text(
        "<?xml version='1.0'?><VTKFile type='PolyData' version='1.0' byte_order='LittleEndian' "
        "header_type='UInt32' compressor='vtkZLibDataCompressor'><PolyData>"
        "<Piece NumberOfPoints='4' NumberOfCells='2'>"
        f"<Points><DataArray type='Float32' NumberOfComponents='3' format='binary'>{arrays['Points']}"
        "</DataArray></Points>"
        f"<PointData><DataArray type='Float32' Name='Cp' format='binary'>{_compressed(cp, 'Float32')}"
        f"{information}</DataArray></PointData>"
        f"<Polys><DataArray type='Int64' Name='connectivity' format='binary'>{arrays['connectivity']}</DataArray>"
        f"<DataArray type='Int64' Name='offsets' format='binary'>{arrays['offsets']}</DataArray></Polys>"
        "</Piece></PolyData></VTKFile>", encoding="ascii")

    surface = read_surface(file)
    assert surface.n_points == 4
    assert surface.n_cells == 2 and surface.cells_per_shape == {3: 2}
    assert np.allclose(surface.points, points)                                  # the floating-point payload survives
    assert np.allclose(surface.point_data["Cp"], cp)
    assert np.array_equal(surface.cells_of(3), [[0, 1, 2], [1, 2, 3]])          # cells not shifted by the offsets
    assert surface_quadrature(surface)["area"] == pytest.approx(1.0)


def test_offsets_with_and_without_the_leading_zero_are_the_same_mesh(tmp_path):
    """VTK omits the leading zero of a start-offset array; read as start offsets it loses the last cell and hands
    every other one its neighbour's vertices. The SHIFT surface came back with 3,197,219 of its 3,197,220
    triangles that way, and with them the point areas the force integral is built on."""
    def surface_file(name: str, offsets: str):
        file = tmp_path / name
        file.write_text(
            "<?xml version='1.0'?><VTKFile type='PolyData' version='0.1' byte_order='LittleEndian' "
            "header_type='UInt64'><PolyData><Piece NumberOfPoints='4' NumberOfCells='2'>"
            "<Points><DataArray type='Float64' NumberOfComponents='3' format='ascii'>"
            "0 0 0 1 0 0 0 1 0 1 1 0</DataArray></Points>"
            "<Polys><DataArray type='Int64' Name='connectivity' format='ascii'>0 1 2 1 2 3</DataArray>"
            f"<DataArray type='Int64' Name='offsets' format='ascii'>{offsets}</DataArray></Polys>"
            "</Piece></PolyData></VTKFile>", encoding="ascii")
        return file

    for name, offsets in (("vtk.vtp", "3 6"), ("start.vtp", "0 3 6")):
        surface = read_surface(surface_file(name, offsets))
        assert surface.n_cells == 2
        assert np.array_equal(surface.cells_of(3), [[0, 1, 2], [1, 2, 3]])
        assert surface_quadrature(surface)["area"] == pytest.approx(1.0)


def test_partial_read_decodes_only_the_arrays_asked_for(tmp_path):
    fixture.write_case(tmp_path, 1, BODY, dataset="ahmedml")
    path = tmp_path / "run_1" / "boundary_1.vtp"
    everything = read_surface(path)
    one = read_surface(path, arrays=["Cp"])
    assert "p" in one.skipped and "Cp" not in one.skipped               # the field, named and not decoded
    assert one.available_cells == ("Cp", "p")                           # topology is the mesh, not cell data
    assert one.bytes_decoded < everything.bytes_decoded
    assert np.allclose(one.cell_data["Cp"], everything.cell_data["Cp"])
    with pytest.raises(KeyError, match="no array"):
        read_surface(path, arrays=["Pressure"])


def test_the_names_of_the_arrays_are_readable_without_decoding_any(tmp_path):
    fixture.write_case(tmp_path, 1, BODY, dataset="ahmedml")
    path = tmp_path / "run_1" / "boundary_1.vtp"
    assert loader.list_arrays(path) == ("Cp", "Points", "p")
    manifest = read_surface(path, arrays=())
    assert manifest.skipped == ("Cp", "TimeValue", "p")                  # named, none of them decoded
    everything = read_surface(path)
    assert manifest.bytes_decoded < 0.7 * everything.bytes_decoded       # only the points and the cells
    del everything
    assert manifest.connectivity.size > 0 and manifest.points.size > 0   # the mesh always has to be read


def test_the_unstructured_surface_and_its_point_quadrature(tmp_path):
    fixture.write_case(tmp_path / "windsor", 2, BODY, dataset="windsorml")
    path = tmp_path / "windsor" / "run_2" / "boundary_2.vtu"
    surface = read_surface(path)
    assert surface.grid == "unstructuredgrid" and surface.cell_types.size == surface.n_cells
    assert sorted(surface.point_data) == ["Cp"]                         # Cp per point, as WindsorML publishes it
    weights = np.load(tmp_path / "windsor" / "run_2" / "boundary_dual_area_2.npy")
    assert weights.shape == (surface.n_points,) and weights.sum() > 1.0


# ---------------------------------------------------------------------------
# the force integral
# ---------------------------------------------------------------------------

def test_hand_computed_drag_on_the_box_case(tmp_path):
    """A 2 x 1 x 1 box, Cp = 1 upstream, -0.4 downstream, 0 elsewhere, A_ref = 1 m^2.

    Cd = -(oint Cp n_x dA) / A_ref. The upstream face has n_x = -1 and area 1: +1.0. The downstream face has
    n_x = +1 and area 1 with Cp = -0.4: +0.4. The four other faces have n_x = 0. So Cd = 1.4 and Cl = Cs = 0,
    and the pressure is the only force -- the box has no friction field.
    """
    fixture.write_box_case(tmp_path, run=900, a_ref=1.0)
    data = loader.open_dataset("ahmedml", root=tmp_path)
    measured = data.measured_forces(900, a_ref=1.0)
    assert measured.cd == pytest.approx(1.4, abs=1e-9)
    assert measured.cs == pytest.approx(0.0, abs=1e-9) and measured.cl == pytest.approx(0.0, abs=1e-9)
    assert measured.quadrature == "published" and measured.field == "Cp" and measured.location == "cell"
    assert measured.area == pytest.approx(10.0)                          # 6 faces: two of 1 m^2, four of 2 m^2
    assert measured.integrated_normal_ratio < 1e-9
    # the same integral with the areas computed from the faces rather than read from the published array
    geometry = data.measured_forces(900, a_ref=1.0, use_published_areas=False)
    assert geometry.quadrature == "computed" and geometry.cd == pytest.approx(1.4, abs=1e-9)
    # half the reference area, twice the coefficient
    assert data.measured_forces(900, a_ref=0.5).cd == pytest.approx(2.8, abs=1e-9)


def test_measured_drag_agrees_with_the_cases_own_forces_file(tmp_path):
    fixture.write_fixture(tmp_path, family="alpha", n_cases=6, seed=0)
    data = loader.open_dataset("ahmedml", root=tmp_path)
    measured = [data.measured_forces(run) for run in data.runs()]
    assert len(measured) == 6
    assert max(abs(row.cd_error) for row in measured) < 1e-6             # two routes, one number
    assert all(0.2 < row.cd < 1.5 for row in measured)                   # a road-vehicle coefficient, plausible
    assert all(row.area == pytest.approx(row.published_area) for row in measured)


def test_the_two_quadratures_agree_on_the_same_body(tmp_path):
    """The cell quadrature against the point quadrature of the same body and field.

    They are not the same arithmetic: the point route weights a node's value by the magnitude of the node's
    summed area vector, so a node shared by two perpendicular faces is represented by one direction. On the
    fixture's coarse surface that is worth 2.2 % of the drag, and it is measured here rather than assumed to
    vanish -- WindsorML's own numbers carry the same approximation.
    """
    fixture.write_case(tmp_path / "cell", 1, BODY, dataset="ahmedml")
    fixture.write_case(tmp_path / "point", 2, BODY, dataset="windsorml")
    by_cell = loader.open_dataset("ahmedml", root=tmp_path / "cell").measured_forces(1)
    by_point = loader.open_dataset("windsorml", root=tmp_path / "point").measured_forces(2)
    assert by_cell.location == "cell" and by_point.location == "point"
    assert by_point.face_area == pytest.approx(by_cell.face_area, rel=1e-9)   # the same body, measured twice
    assert by_point.area <= by_cell.area                                 # a point weight is a dual-cell area
    assert abs(by_point.cd - by_cell.cd) / by_cell.cd < 0.05


def test_the_volume_subsample_reads_the_points_a_full_read_would_have_given(tmp_path):
    fixture.write_case(tmp_path, 1, BODY, dataset="ahmedml", volume_points=300)
    path = tmp_path / "run_1" / "volume_1.vtu"
    full = read_surface(path)
    sampled = loader.volume_subsample(path, 24, seed=4, fields=["p", "Ux"])
    index = sampled["_index"]
    assert index.size == 24 and np.all(np.diff(index) > 0) and index.max() < full.n_points
    assert np.allclose(sampled["points"], full.points[index])
    assert np.allclose(sampled["p"], full.point_data["p"][index])
    assert np.allclose(sampled["Ux"], full.point_data["Ux"][index])
    again = loader.volume_subsample(path, 24, seed=4, fields=["p"])
    assert np.array_equal(again["_index"], index)                        # seeded, so the same points
    everything = loader.volume_subsample(path, 500, seed=0, fields=["p"])
    assert everything["_index"].size == full.n_points                    # asking for more returns the file
    with pytest.raises(KeyError, match="no point array"):
        loader.volume_subsample(path, 4, fields=["Cp"])


def test_a_case_subsamples_its_own_volume(tmp_path):
    fixture.write_case(tmp_path, 3, BODY, dataset="ahmedml", volume_points=200)
    data = loader.open_dataset("ahmedml", root=tmp_path)
    sampled = data.volume_subsample(3, count=10, seed=1)
    assert sampled["points"].shape == (10, 3) and "p" in sampled


# ---------------------------------------------------------------------------
# the metrics
# ---------------------------------------------------------------------------

def test_the_metric_definitions_are_the_published_ones():
    true = np.array([1.0, 1.0])
    predicted = np.array([2.0, 1.0])
    assert benchmark.relative_l2(predicted, true) == pytest.approx(math.sqrt(1 / 2))
    assert benchmark.mse(predicted, true) == pytest.approx(0.5)
    assert benchmark.mae(predicted, true) == pytest.approx(0.5)
    weighted = benchmark.surface_metrics(predicted, true, weights=np.array([3.0, 1.0]))
    assert weighted["MSE"] == pytest.approx(0.75)                        # area-averaged, not point-averaged
    assert weighted["MAE"] == pytest.approx(0.75)
    assert weighted["L2"] == pytest.approx(math.sqrt(1 / 2))             # scale-free: the weights do not move it
    with pytest.raises(ValueError):
        benchmark.relative_l2(np.zeros(3), np.zeros(2))
    with pytest.raises(ValueError):
        benchmark.surface_metrics(predicted, true, weights=np.ones(3))


def test_a_perfect_prediction_has_no_error_and_a_constant_one_does(tmp_path):
    fixture.write_fixture(tmp_path, family="alpha", n_cases=6, seed=0)
    data = loader.open_dataset("ahmedml", root=tmp_path)
    model = benchmark.constant_field_baseline([data.case(run) for run in data.runs()[:4]])
    row = data.case(1)
    name, true, _location = row.field_of("cp")
    assert name == "Cp"
    exact = benchmark.surface_metrics(true, true)
    assert exact == {"L2": 0.0, "MSE": 0.0, "MAE": 0.0}
    constant = benchmark.surface_metrics(model(row), true)
    assert 0.2 < constant["L2"] < 0.6                                    # the baseline is visibly wrong, not noise
    assert float(model.mean_cp) == pytest.approx(float(np.mean(true)), rel=0.5)


# ---------------------------------------------------------------------------
# the declared baselines
# ---------------------------------------------------------------------------

def test_the_published_numbers_are_quoted_from_the_paper_that_printed_them():
    ahmed = {(row.metric, row.model): row.value for row in benchmark.declared("ahmedml", "surface_pressure")}
    assert ahmed[("L2", "GP-UPT (42M)")] == pytest.approx(0.0272)        # AB-UPT Table 2
    assert ahmed[("MSE", "GP-UPT (42M)")] == pytest.approx(0.00078)
    assert ahmed[("MAE", "GP-UPT (42M)")] == pytest.approx(0.0085)
    drivaer = {(row.metric, row.model): row.value
               for row in benchmark.declared("drivaerml", "surface_pressure")}
    assert drivaer[("L2", "Transolver (3.9M)")] == pytest.approx(0.0388)  # AB-UPT Table 1
    assert drivaer[("MSE", "GP-UPT (3.6M, point decoder)")] == pytest.approx(239.63)
    for row in benchmark.BASELINES:
        assert "arXiv:" in row.source and "Table" in row.source
        assert row.dataset in catalog.DATASETS


def test_windsorml_has_no_surrogate_baseline_and_the_table_says_so():
    rows = benchmark.declared("windsorml", "surface_pressure")
    assert len(rows) == 1 and math.isnan(rows[0].value)
    assert "reports DrivAerML and AhmedML only" in rows[0].note
    assert benchmark.declared("windsorml", "drag_coefficient")           # the solver against the tunnel
    table = benchmark.benchmark_table(datasets=["windsorml"], limit=1)
    markdown = table.to_markdown()
    assert "awaiting_data" in markdown and "-- (the AB-UPT paper" in markdown
    assert "CP" not in markdown and benchmark.COLLAPSE_NOTE in markdown


# ---------------------------------------------------------------------------
# the table
# ---------------------------------------------------------------------------

def test_the_table_is_measured_where_the_data_is_and_declared_where_it_is_not(tmp_path):
    fixture.write_fixture(tmp_path / "ahmed", family="alpha", n_cases=8, seed=0)
    table = benchmark.benchmark_table({"ahmedml": tmp_path / "ahmed"},
                                      datasets=["ahmedml", "drivaerml"], limit=4)
    measured = {row.quantity: row for row in table.measured_rows()}
    assert measured["drag_coefficient"].cases == 4
    assert measured["drag_coefficient"].measured["max_abs_error"] < 1e-6
    assert "surface_pressure" not in measured                            # no model, no field error
    statuses = {(row.dataset, row.quantity): row.status for row in table.rows}
    assert statuses[("ahmedml", "drag_coefficient")] == "measured"
    assert statuses[("ahmedml", "surface_pressure")] == "awaiting_data"      # the data is there, the model is not
    assert statuses[("drivaerml", "drag_coefficient")] == "awaiting_data"
    assert "31 TB" in next(row for row in table.rows if row.dataset == "drivaerml").note


def test_a_model_fills_the_surface_pressure_column(tmp_path):
    fixture.write_fixture(tmp_path / "ahmed", family="alpha", n_cases=8, seed=0)
    data = loader.open_dataset("ahmedml", root=tmp_path / "ahmed")
    model = benchmark.constant_field_baseline([data.case(run) for run in data.runs()[:6]])
    table = benchmark.benchmark_table({"ahmedml": data}, datasets=["ahmedml"], limit=4, model=model,
                                      model_name="constant baseline")
    row = next(row for row in table.rows if row.quantity == "surface_pressure")
    assert row.status == "measured" and row.model == "constant baseline" and row.cases == 4
    assert 0.2 < row.measured["L2"] < 0.6 and row.measured["MSE"] > 0
    assert "L2 0.0272" in table.to_markdown()                            # the declared column is still the paper's


# ---------------------------------------------------------------------------
# a dataset that is not there
# ---------------------------------------------------------------------------

def test_a_missing_dataset_opens_and_a_missing_case_names_the_url(tmp_path):
    data = loader.open_dataset("drivaerml", root=tmp_path / "nowhere")
    assert data.runs() == [] and data.n_cases == 484 and data.licence == "CC-BY-SA-4.0"
    with pytest.raises(loader.CaseNotPresent) as error:
        data.case(1)
    message = str(error.value)
    assert "huggingface.co/datasets/neashton/drivaerml/resolve/main/run_1/force_mom_1.csv" in message
    assert "fetch('forces', 1" in message and "31 TB" in message and "boundary_1.vtp" in message
    with pytest.raises(loader.CaseNotPresent, match="is not published"):
        data.case(167)                                                    # one of the 16 withheld runs
    with pytest.raises(loader.CaseNotPresent, match="no run 501"):
        data.case(501)


def test_the_parameter_and_force_files_are_read_in_either_shape(tmp_path):
    both_rows = tmp_path / "name_value.csv"
    both_rows.write_text("slant_angle,25\nbody_length,1044\n")
    assert loader.read_parameters(both_rows) == {"slant_angle": 25.0, "body_length": 1044.0}
    header_row = tmp_path / "header.csv"
    header_row.write_text("Angle of rear-window,Tilted surface length\n25.0,222\n")
    assert loader.read_parameters(header_row) == {"angle_of_rear_window": 25.0, "tilted_surface_length": 222.0}
    forces = tmp_path / "force_mom_1.csv"
    forces.write_text("cd, cl\n2.3848566658e-01, -9.4516081781e-02\n")
    assert loader.read_force_csv(forces) == pytest.approx({"cd": 0.23848566658, "cl": -0.094516081781})


# ---------------------------------------------------------------------------
# the bundle
# ---------------------------------------------------------------------------

def test_a_bundle_carries_the_licence_the_credit_and_the_coefficients(tmp_path):
    from pinneapple_data.schema import PhysicsDataBundle

    fixture.write_case(tmp_path, 1, BODY, dataset="ahmedml")
    bundle = loader.case_bundle(loader.open_dataset("ahmedml", root=tmp_path), 1)
    assert isinstance(bundle, PhysicsDataBundle)
    provenance = bundle.manifest.provenance
    assert provenance.license == "CC-BY-SA-4.0"
    assert "arXiv:2407.20801" in provenance.citation
    assert provenance.url.startswith("https://huggingface.co/datasets/neashton/ahmedml")
    assert {field.name for field in bundle.manifest.fields.variables} == {"Cp", "p"}
    parameters = {p.name: p.value for p in bundle.manifest.physics.parameters}
    assert parameters["slant"] == pytest.approx(BODY.slant)              # the case's own design point
    metadata = bundle.manifest.metadata["ahmedml"]
    assert metadata["licence"] == "CC-BY-SA-4.0" and metadata["forces"]["cd"] > 0.2
    assert bundle.manifest.mesh.blocks[0].type == "triangle"


# ---------------------------------------------------------------------------
# the cross-family experiment
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cross_family(tmp_path_factory):
    root = tmp_path_factory.mktemp("cross_family")
    return trust.run_on_fixtures(root, n_train=36, n_calibration=12, n_level=10, seed=0)


def test_the_shift_family_really_is_a_different_family():
    alpha = fixture.sample_family("alpha", 40, seed=0)
    beta = fixture.sample_family("beta", 40, seed=1)
    mean = lambda rows, name: float(np.mean([getattr(row, name) for row in rows]))  # noqa: E731
    assert mean(beta, "slant") > mean(alpha, "slant") + 10
    assert mean(beta, "roundness") > mean(alpha, "roundness") + 2
    assert mean(beta, "height") > mean(alpha, "height")
    blend = trust._blend("alpha", "beta", 0.5)
    assert blend["slant"] == pytest.approx((19.0, 32.0))                 # alpha (10, 22) and beta (28, 42), halfway


def test_geometry_descriptors_are_read_off_the_surface(tmp_path):
    fixture.write_case(tmp_path / "alpha", 1, BODY, dataset="ahmedml")
    fixture.write_case(tmp_path / "boxy", 2, fixture.Params(length=1.1, width=0.38, height=0.30, slant=35.0,
                                                            roundness=6.0, taper=0.20, clearance=0.05))
    smooth = trust.geometry_descriptors(tmp_path / "alpha" / "run_1" / "boundary_1.vtp")
    boxy = trust.geometry_descriptors(tmp_path / "boxy" / "run_2" / "boundary_2.vtp")
    assert sorted(smooth) == sorted(trust.DESCRIPTORS)
    assert smooth["length"] == pytest.approx(BODY.length, rel=1e-6)
    assert smooth["frontal_area"] > 0 and 0 < smooth["rear_area_fraction"] < 1
    assert boxy["shape_factor"] != pytest.approx(smooth["shape_factor"], rel=1e-3)


def test_the_pressure_descriptor_is_scale_free_and_area_weighted():
    values = np.array([-1.0, 0.0, 1.0])
    weights = np.array([1.0, 1.0, 1.0])
    evenly = trust.cp_quantiles(values, weights)
    doubled = trust.cp_quantiles(values, 7.0 * weights)                  # scale-free: the weights normalise
    assert np.allclose(evenly, doubled)
    skewed = trust.cp_quantiles(values, np.array([1.0, 1.0, 8.0]))
    assert skewed[-1] > evenly[-1] and skewed[3] > evenly[3]             # the mass moved to the high Cp
    assert skewed[0] <= evenly[0]
    assert evenly.shape == (len(trust.QUANTILES),)


def test_the_trust_signal_flags_the_shifted_family_before_the_error_doubles(cross_family):
    forward, backward = cross_family
    for result in cross_family:
        assert result.n_train == 36 and result.n_calibration == 12
        assert 3.0 < result.threshold < 5.0                             # calibrated on the held-out in-family split
        assert result.false_alarm_rate <= 0.1
        assert 0.015 < result.error_in_family < 0.04
        assert 0.02 < result.interval < 0.12                            # 90 % conformal half-width on Cd
        assert result.coverage_in_family >= 0.9
        # the in-family level is genuinely in-family, the far end is not
        assert result.row(0.0).flagged_fraction <= 0.1
        assert result.row(1.0).flagged_fraction == 1.0
        assert result.row(1.0).embedding_distance > 4 * result.row(0.0).embedding_distance
        assert result.row(1.0).error > 4 * result.error_in_family
        assert result.row(0.0).coverage >= 0.9 and result.row(1.0).coverage <= 0.1
        # the embedding distance rises monotonically with the shift
        distances = [level.embedding_distance for level in result.levels]
        assert distances == sorted(distances)
        assert result.flags_before_error and result.t_flag is not None and result.t_error is not None
        assert result.t_flag < result.t_error
    # the numbers this fixture measured (seed 0, 28 x 13 mesh, 36 training cases)
    assert (forward.t_flag, forward.t_error) == (0.4, 0.6)
    assert (backward.t_flag, backward.t_error) == (0.2, 0.6)
    assert forward.row(1.0).error == pytest.approx(0.110, abs=0.02)
    assert backward.row(1.0).error == pytest.approx(0.209, abs=0.03)
    assert 0.015 < forward.row(0.0).error < 0.025                        # the held-out in-family level
    assert forward.row(1.0).coverage == 0.0 and backward.row(1.0).coverage == 0.0


def test_the_surface_pressure_summary_barely_moves_even_when_the_drag_does(cross_family):
    """The finding that decides what the ordering is claimed on.

    The Cp *distribution* (its area-weighted quantiles) is nearly the same in the two families, because the
    pressure model is nearly the same function of the local normal; what the shift moves is the integral. So the
    experiment claims its ordering on the drag coefficient -- the quantity these datasets exist for -- and
    reports the descriptor error as the flat column it is, rather than quoting a flat number as a result.
    """
    for result in cross_family:
        descriptor = [level.field_error for level in result.levels]
        assert max(descriptor) < 0.06
        assert max(descriptor) - min(descriptor) < 0.02
        assert result.row(1.0).error > 2 * max(descriptor)


def test_the_result_writes_itself_out(cross_family):
    forward = cross_family[0]
    markdown = forward.to_markdown()
    assert "trained on `alpha`" in markdown and "flags at t = 0.4" in markdown
    payload = forward.to_dict()
    assert payload["flags_before_error"] is True and len(payload["levels"]) == 6
    assert {level["t"] for level in payload["levels"]} == {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
