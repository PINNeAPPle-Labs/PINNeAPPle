"""The structural run: one contract, several readers, and the same run read two ways agreeing.

The case is real and small: a coupon is driven into a rigid wall by the library's own explicit crash solver
(``numerical_solvers.crash_fem``, 63 nodes, 24 hexahedra, six output frames, ~0.3 s), and every reader is checked
against that one run rather than against a fixture of itself:

* ``CrashResult`` -> ``StructuralRun`` is one call and carries parts, fields, times, units, provenance and masks;
* the same run written as a legacy-VTK time series (the format ``anim_to_vtk`` writes) and read back gives the same
  object to the precision the format stores — the discrepancy is printed per field and asserted;
* the same run laid out as a staged OpenRadioss case (``frames/frame_000.vtk`` plus ``disp_XXX.txt``/``time_XXX.txt``,
  the reduction the campaign leaves on disk) gives the same displacement and the same times, and says which fields
  only exist where a full frame does;
* the same run declared in ``mm-tonne-s`` and in SI is the same object in the library's canonical units;
* the ASCII half of an LS-DYNA run (keyword deck plus ``nodout``) is read, and the run states the quantities that
  route cannot carry instead of shipping an empty array that looks like a zero field.

A note on what this file does **not** claim: no OpenRadioss and no LS-DYNA solver was run to produce it. The VTK series
and the staged-case layout are written by this library from the crash solver's own output, in the format each source
writes; the LS-DYNA files are built from the documented card and print layouts. ``docs/guides/structural_ingestion.md``
carries that as a table row per format, and it is the reason the LS-DYNA run records ``unsupported``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinneapple_data.cae import (
    CellBlock,
    Mesh,
    StructuralRun,
    build_training_set,
    read_keyword_deck,
    read_lsdyna_directory,
    read_nodout,
    read_vtk_frame,
    read_vtk_series,
)
from pinneapple_data.cae.lsdyna import DynaFormatError
from pinneapple_data.cae.structural import canonical_field
from pinneapple_data.cae.training_set import decimate, deduplicate
from pinneapple_data.cae.vtk_series import VtkFormatError, write_vtk_frame
from pinneapple_simulation.numerical_solvers.crash_fem import CrashSolver, Material, RigidWall
from pinneapple_simulation.numerical_solvers.solid_fem import box_mesh

STEEL = Material(rho=7.8e3, E=2.1e11, nu=0.3, yield_stress=6e8, hardening=4e8, erosion_strain=0.4)


@pytest.fixture(scope="module")
def crash():
    """One real impact: a 60 mm coupon with two parts, held on its back face, into a rigid wall at 30 m/s."""
    mesh = box_mesh(0.06, 0.02, 0.02, 6, 2, 2)
    parts = np.where(np.arange(len(mesh.elements)) < len(mesh.elements) // 2, 1, 2)
    solver = CrashSolver(mesh, material=STEEL, parts=parts, contact=RigidWall(x0=0.0605),
                         fixed=mesh.face_nodes("x-"), initial_velocity=(30.0, 0.0, 0.0), end_time=1.2e-4, frames=6)
    return solver.run(), mesh.face_nodes("x-")


@pytest.fixture(scope="module")
def run(crash):
    result, fixed = crash
    return StructuralRun.from_crash_result(result, part_names={1: "beam", 2: "box"}, fixed_nodes=fixed,
                                           units="SI", case="coupon_impact")


def _staged_openradioss(root: Path, run: StructuralRun) -> Path:
    """The artifact layout ``external_solvers.openradioss.runner`` leaves on disk, filled from this run.

    ``frame_000.vtk`` is the full first frame (points, cells, ``PART_ID``), and every step has its displacement dump
    and its time file — the reduction the runner does inside the container so a long run cannot fill a disk.
    """
    frames = root / "frames"
    frames.mkdir(parents=True)
    write_vtk_frame(frames / "frame_000.vtk", run.frames[0], time=float(run.times[0]),
                    part_ids=run.part_of_cell())
    disp = run.field("displacement")
    for k in range(run.n_frames):
        np.savetxt(frames / f"disp_{k:03d}.txt", disp[k], fmt="%.6e")
        (frames / f"time_{k:03d}.txt").write_text(f"{run.times[k]:.6e}\n")
    return root


def _dyna_directory(root: Path, *, times=(0.0, 1.0e-3, 2.0e-3), nsteps=3) -> Path:
    """A minimal LS-DYNA ASCII run: a keyword deck (8 nodes, 2 solids, 2 parts, one shell section) and a ``nodout``."""
    deck = """*KEYWORD
*TITLE
coupon impact, mm-tonne-s
*NODE
       1       0.0       0.0       0.0
       2      10.0       0.0       0.0
       3      10.0      10.0       0.0
       4       0.0      10.0       0.0
       5       0.0       0.0      10.0
       6      10.0       0.0      10.0
       7      10.0      10.0      10.0
       8       0.0      10.0      10.0
*ELEMENT_SOLID
       1       1       1       2       3       4       5       6       7       8
       2       2       1       2       3       4       5       6       7       8
*PART
$#                                                                         title
beam coupon
$#     pid     secid       mid     eosid      hgid      grav    adpopt      tmid
         1         1         1
*PART
box coupon
         2         2         1
*SECTION_SHELL
$#   secid    elform      shrf       nip
         1         2       1.0         2
$#      t1        t2        t3        t4
       1.5       1.5       1.5       1.5
*SECTION_SOLID
         2         1
*CONTROL_TERMINATION
     1.0
*END
"""
    (root / "crash.key").write_text(deck)
    lines = [" n o d a l   p r i n t   o u t", " n o d a l   p r i n t   o u t   f o r   s e t       1", ""]
    for k, t in enumerate(times[:nsteps]):
        lines += [f" time value =  {t:.6E}", "",
                  "  nodal point    x-disp      y-disp      z-disp    x-velocity  y-velocity  z-velocity"]
        for nid in range(1, 9):
            lines.append(f"       {nid}  {k * 0.1:.6E}  0.000000E+00  0.000000E+00  {k * 30.0:.6E}  "
                         "0.000000E+00  0.000000E+00")
        lines.append("")
    (root / "nodout").write_text("\n".join(lines))
    return root


# ---------------------------------------------------------------------------------------- the container
def test_crash_result_becomes_a_run_in_one_call(run, crash):
    result, fixed = crash
    assert run.n_frames == len(result.times) and run.n_points == 63 and run.n_cells == 24
    assert run.fields == ["displacement", "velocity", "von_mises", "plastic_strain"]
    assert [p.name for p in run.parts] == ["beam", "box"]
    assert sum(len(p) for p in run.parts) == 24
    assert np.array_equal(run.part_of_cell(), np.asarray(result.parts))
    assert np.array_equal(run.eroded, np.asarray(result.eroded))
    assert np.allclose(run.contact_force, np.asarray(result.contact_force))
    assert run.fixed_nodes.sum() == len(fixed)                       # the mask, not the index list
    assert np.allclose(run.times, result.times)
    assert run.units["von_mises"] == "Pa" and run.units["displacement"] == "m" and run.units["time"] == "s"
    for key in ("solver", "library_version", "seed", "data_hash", "commit", "mesh", "case"):
        assert key in run.provenance, key
    assert run.provenance["solver"] == "pinneapple-crash-fem"
    assert run.provenance["out_of_envelope"] is False


def test_the_undeformed_geometry_is_recoverable_frame_by_frame(run):
    undeformed = run.points
    for k in range(run.n_frames):
        assert np.allclose(run.frames[k].points, undeformed + run.field("displacement")[k])


def test_a_run_whose_frames_disagree_on_connectivity_is_refused(run):
    other = Mesh(points=run.frames[0].points, blocks=[CellBlock("tetra", np.array([[0, 1, 2, 3]]))])
    with pytest.raises(ValueError, match="cells, the first frame has"):
        StructuralRun(frames=[run.frames[0], other], times=np.array([0.0, 1.0]))


# ------------------------------------------------------------------------------ two paths, one object
def test_the_vtk_series_path_gives_the_same_run_as_the_in_memory_path(run, tmp_path, capsys):
    run.to_vtk_series(tmp_path, precision="single")
    back = StructuralRun.from_vtk_series(tmp_path, units="SI")
    print("\nthe same run through two paths (VTK series against the in-memory CrashResult):")
    for name in run.fields:
        a, b = run.field(name), back.field(name)
        peak = float(np.max(np.abs(a))) or 1.0
        worst = float(np.max(np.abs(a - b)))
        print(f"  {name:<16} max |difference| = {worst:.3e}   relative = {worst / peak:.3e}   unit = "
              f"{run.units[name]}")
        assert worst / peak < 1e-6                      # nine stored digits, so a tenth of a millionth at worst
    assert back.n_frames == run.n_frames
    assert np.allclose(back.times, run.times, rtol=0, atol=1e-12)
    assert back.units == run.units
    assert [p.id for p in back.parts] == [p.id for p in run.parts]
    assert np.array_equal(back.part_of_cell(), run.part_of_cell())
    assert back.frames[0].source["dataset"] == "UNSTRUCTURED_GRID"


def test_the_openradioss_case_layout_gives_the_same_run(run, tmp_path, capsys):
    case = _staged_openradioss(tmp_path / "case_01", run)
    back = StructuralRun.from_openradioss(case, units="SI", part_names={1: "beam", 2: "box"})
    a, b = run.field("displacement"), back.field("displacement")
    peak = float(np.max(np.abs(a))) or 1.0
    worst = float(np.max(np.abs(a - b)))
    print(f"\nOpenRadioss staged case against the in-memory run: max |difference| = {worst:.3e}, "
          f"relative = {worst / peak:.3e} over {back.n_frames} frames")
    assert worst / peak < 1e-6
    assert np.allclose(back.times, run.times, rtol=0, atol=1e-9)
    assert [p.name for p in back.parts] == ["beam", "box"]
    assert back.provenance["solver"] == "openradioss"
    # only frame 0 is a full frame here, so the elemental fields exist at that one instant and the run says so
    assert back.info["fields_partial"]["von_mises"] == 1
    assert np.isnan(back.field("von_mises")[1:]).all(), "a frame without the field is nan, never a zero field"


def test_the_two_readers_agree_with_each_other(run, tmp_path):
    run.to_vtk_series(tmp_path / "vtk", precision="double")
    via_vtk = StructuralRun.from_vtk_series(tmp_path / "vtk", units="SI")
    via_case = StructuralRun.from_openradioss(_staged_openradioss(tmp_path / "case", run), units="SI")
    a, b = via_vtk.field("displacement"), via_case.field("displacement")
    assert np.allclose(a, b, rtol=0, atol=1e-9 * max(1.0, float(np.abs(a).max())))


def test_binary_and_xml_vtk_say_what_to_do_instead(tmp_path):
    binary = tmp_path / "b.vtk"
    binary.write_text("# vtk DataFile Version 3.0\nt\nBINARY\nDATASET UNSTRUCTURED_GRID\n")
    with pytest.raises(VtkFormatError, match="BINARY"):
        read_vtk_frame(binary)
    xml = tmp_path / "c.vtu"
    xml.write_text("<VTKFile/>")
    with pytest.raises(VtkFormatError, match="meshio"):
        read_vtk_series(tmp_path)


# ---------------------------------------------------------------------------------------------- units
def test_a_run_in_mm_tonne_s_is_the_same_object_in_canonical_units(crash):
    result, fixed = crash
    si = StructuralRun.from_crash_result(result, fixed_nodes=fixed, units="SI", case="coupon")
    mm = StructuralRun.from_crash_result(result, fixed_nodes=fixed, units="mm-tonne-s", case="coupon")
    scaled = StructuralRun(
        frames=[Mesh(points=f.points * 1e3, blocks=f.blocks,
                     point_data={k: v * (1e3 if k == "velocity" else 1e3) for k, v in f.point_data.items()},
                     cell_data={k: (v * 1e-6 if k == "von_mises" else v.copy()) for k, v in f.cell_data.items()},
                     units=dict(mm.units)) for f in si.frames],
        times=si.times.copy(), parts=list(si.parts), units=dict(mm.units), provenance=dict(mm.provenance),
        eroded=si.eroded.copy(), contact_force=si.contact_force.copy(), fixed_nodes=si.fixed_nodes.copy())
    assert mm.units["displacement"] == "mm" and si.units["displacement"] == "m"
    back = scaled.to_units("SI")
    assert back.units == si.units
    assert np.allclose(back.field("displacement"), si.field("displacement"), rtol=1e-12, atol=1e-18)
    assert np.allclose(back.field("velocity"), si.field("velocity"), rtol=1e-12, atol=1e-9)
    assert np.allclose(back.field("von_mises"), si.field("von_mises"), rtol=1e-12, atol=1e-3)
    assert np.allclose(back.field("plastic_strain"), si.field("plastic_strain"), rtol=1e-12)  # dimensionless: untouched
    assert np.allclose(back.times, si.times)


# ------------------------------------------------------------------------------------------ zarr round trip
def test_the_zarr_round_trip_is_bit_identical_for_float64(run, tmp_path):
    run.to_zarr(tmp_path / "run.zarr")
    back = StructuralRun.from_zarr(tmp_path / "run.zarr")
    assert back.checksum() == run.checksum()
    assert back.info["stored_as"] == "float64"
    for name in run.fields:
        assert np.array_equal(back.field(name), run.field(name)), name
    assert np.array_equal(back.times, run.times)
    assert np.array_equal(back.points, run.points)
    assert back.units == run.units and back.provenance == run.provenance
    assert [(p.id, p.name, len(p)) for p in back.parts] == [(p.id, p.name, len(p)) for p in run.parts]
    assert np.array_equal(back.part_of_cell(), run.part_of_cell())
    assert np.array_equal(back.eroded, run.eroded) and np.array_equal(back.fixed_nodes, run.fixed_nodes)


def test_the_float32_store_is_cheaper_and_says_so(run, tmp_path):
    run.to_zarr(tmp_path / "f32.zarr", dtype="float32")
    back = StructuralRun.from_zarr(tmp_path / "f32.zarr")
    assert back.info["stored_as"] == "float32"
    a, b = run.field("displacement"), back.field("displacement")
    rel = float(np.max(np.abs(a - b))) / (float(np.max(np.abs(a))) or 1.0)
    assert 0 < rel < 1e-6, f"single precision should be close and not equal, relative {rel:.3e}"
    assert back.units == run.units and back.provenance == run.provenance


# --------------------------------------------------------------------------------------- training set
def test_deduplication_drops_a_repeated_state_and_counts_it(run):
    repeated = StructuralRun(frames=list(run.frames) + [run.frames[-1]], times=np.append(run.times, run.times[-1]),
                             parts=list(run.parts), units=dict(run.units), provenance=dict(run.provenance),
                             eroded=np.vstack([run.eroded, run.eroded[-1:]]),
                             contact_force=np.vstack([run.contact_force, run.contact_force[-1:]]))
    trimmed, log = deduplicate(repeated)
    assert trimmed.n_frames == run.n_frames
    assert log["frames"] == 1 and log["kept"][-1] == run.n_frames - 1


def test_decimation_reports_the_error_it_introduces(run):
    thinned, log = decimate(run, 40, keep_cells=False)
    assert thinned.n_points == 40 and log["dropped_points"] == run.n_points - 40
    assert set(log["error"]) == {"displacement", "velocity"}
    assert all(0.0 < v < 1.0 for v in log["error"].values()), log["error"]
    assert thinned.n_cells == 0 and thinned.fields == ["displacement", "velocity"]
    with pytest.raises(ValueError, match="point-cloud"):
        decimate(run, 4)                                  # no element can be built from four nodes


def test_decimation_keeps_a_connected_sub_mesh_and_the_parts(run):
    thinned, log = decimate(run, 40)
    assert 0 < thinned.n_cells < run.n_cells and thinned.n_points <= 40
    assert log["kept_cells"] == thinned.n_cells and log["kept_points"] <= 40
    assert np.isfinite(thinned.field("von_mises")).all()
    assert all(len(p) > 0 for p in thinned.parts), "a part lost every element"
    assert set(np.unique(thinned.part_of_cell())) == {p.id for p in thinned.parts}
    assert thinned.frames[0].blocks[0].conn.max() < thinned.n_points      # the connectivity was re-indexed


def test_the_split_holds_out_whole_geometry_families(run):
    runs = []
    for case in ("case_a", "case_b", "case_c", "case_d"):
        scaled = run.to_units("SI")
        scaled.provenance = dict(scaled.provenance, case=case)
        runs.append(scaled)
    ts = build_training_set(runs, hold_out=("case_c",))
    table = ts.table()
    assert ts.split["test"] == ["case_c"]
    sides: dict[str, set] = {}
    for family, side in zip(table["family"], table["split"], strict=True):
        sides.setdefault(str(family), set()).add(str(side))
    assert all(len(s) == 1 for s in sides.values()), f"a family spans two sides: {sides}"
    assert set(table["split"].tolist()) <= {"train", "val", "test"}
    assert ts.n_samples == sum(r.n_frames for r in runs)
    assert set(ts.scales) >= {"displacement", "von_mises", "coordinates"}
    assert ts.runs[0].units["displacement"] == "1"        # non-dimensionalised, and the scale is kept
    assert float(np.max(np.abs(ts.runs[0].field("von_mises")))) == pytest.approx(1.0, rel=1e-12)


def test_the_part_hierarchy_survives_normalisation(run):
    ts = build_training_set([run], make_dimensionless=True)
    kept = ts.runs[0]
    assert [p.name for p in kept.parts] == ["beam", "box"]
    assert np.array_equal(kept.part_of_cell(), run.part_of_cell())
    assert np.allclose(kept.field("von_mises") * ts.scales["von_mises"], run.field("von_mises"))


# -------------------------------------------------------------------------------------------- LS-DYNA
def test_the_lsdyna_ascii_run_carries_parts_and_states_what_it_cannot_carry(tmp_path):
    _dyna_directory(tmp_path)
    got = read_lsdyna_directory(tmp_path)
    assert got.deck.title.startswith("coupon impact")
    assert got.deck.part_names == {1: "beam coupon", 2: "box coupon"}
    assert got.deck.shell_thickness == {1: 1.5}
    assert got.history.displacement().shape == (3, 8, 3)

    run = StructuralRun.from_lsdyna(tmp_path, units="mm-tonne-s")
    assert run.fields == ["displacement", "velocity"]
    assert [(p.id, p.name, p.thickness) for p in run.parts] == [(1, "beam coupon", 1.5), (2, "box coupon", None)]
    assert run.n_frames == 3 and run.n_cells == 2
    assert run.field("displacement")[1, 0, 0] == pytest.approx(0.1)
    assert run.provenance["solver"] == "lsdyna-ascii"
    assert any("d3plot" in s for s in run.info["unsupported"])
    assert any("glstat" in s for s in run.info["unsupported"])
    assert run.info["nodes_outside_nodout"] == 0


def test_a_node_nodout_does_not_print_is_nan_and_not_a_zero(tmp_path):
    """A deck node without a printed displacement has no measurement — nan in the field, counted, never a zero."""
    _dyna_directory(tmp_path)
    lines = (tmp_path / "nodout").read_text().splitlines()
    kept = [ln for ln in lines if not ln.strip().startswith(("5", "6", "7", "8"))]
    (tmp_path / "nodout").write_text("\n".join(kept) + "\n")
    run = StructuralRun.from_lsdyna(tmp_path, units="mm-tonne-s")
    assert run.info["nodes_outside_nodout"] == 4
    assert np.isnan(run.field("displacement")[1:, 4:]).all()
    assert np.isfinite(run.frames[0].points).all(), "an unmeasured node stays at its undeformed position"


def test_the_lsdyna_deck_is_read_from_its_cards(tmp_path):
    _dyna_directory(tmp_path)
    deck = read_keyword_deck(tmp_path / "crash.key")
    assert len(deck.nodes) == 8 and deck.nodes[2] == (10.0, 0.0, 0.0)
    assert deck.section_kind == {1: "shell", 2: "solid"}
    assert deck.termination_time == 1.0
    assert deck.elements["solid"][0] == (1, 1, (1, 2, 3, 4, 5, 6, 7, 8))


def test_a_file_that_is_not_nodout_is_refused(tmp_path):
    (tmp_path / "nodout").write_text("nothing here\n")
    (tmp_path / "x.key").write_text("*KEYWORD\n*NODE\n       1       0.0       0.0       0.0\n*END\n")
    with pytest.raises(DynaFormatError, match="time value"):
        read_nodout(tmp_path / "nodout")
    with pytest.raises(DynaFormatError, match="nodout"):
        read_lsdyna_directory(tmp_path)


# -------------------------------------------------------------------------------------------- names
def test_the_names_a_solver_writes_land_on_the_canonical_fields():
    assert canonical_field("Displacement") == "displacement"
    assert canonical_field("von Mises") == "von_mises"
    assert canonical_field("Effective Plastic Strain") == "plastic_strain"
    assert canonical_field("something_else") == "something_else"
