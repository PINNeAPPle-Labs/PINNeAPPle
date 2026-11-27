"""CAE data layer: read meshes and results from OpenFOAM, CalculiX/Abaqus, Gmsh, VTK, STL and point tables into one
``Mesh`` type; mesh quality (checkMesh-equivalent finite-volume metrics, element shape metrics, surface checks);
repair of surface defects (holes, orientation, degenerate and duplicate faces, split vertices), every operation
measured before and after; per-cell Courant number and wall distance.

The CalculiX run goes further than the mesh: ``read_blocks``/``contact``/``coulomb_ratio`` read the result blocks —
``CONTACT`` included — and ``read_totals`` reads the reaction totals of the ``.dat``, which together are what a
verification of a structural run compares against a closed form.

An explicit-FEA *run* is one object: ``StructuralRun`` holds the frames of one mesh, the time axis, the named parts
with the elements each owns, a unit per field and a provenance record, and the masks a surrogate needs (eroded,
contact, fixed nodes). It is built from ``crash_fem.CrashResult`` in one call, from a legacy-VTK time series
(``read_vtk_series``, the ``anim_to_vtk`` output of the OpenRadioss campaign), from a staged OpenRadioss case
directory, or from the ASCII half of an LS-DYNA run (``read_lsdyna_directory``: keyword deck plus ``nodout``).
``build_training_set`` is the normalisation between a run and a training set — dedup, node decimation with the error
it introduces, unit conversion, non-dimensionalisation and a split that holds out whole geometry families.
"""
from . import repair
from .ccx import (
    FINISHED,
    GIVE_UP,
    Status,
    Totals,
    Verdict,
    finite,
    read_displacements,
    read_log,
    read_status,
    read_totals,
)
from .frd import (
    CONTACT_FIELDS,
    ResultBlock,
    contact,
    coordinates,
    coulomb_ratio,
    displacement,
    read_blocks,
    stress,
)
from .io import FormatError, kind_of, read_any, read_meshio, read_points, read_stl
from .lsdyna import DynaDeck, DynaHistory, read_keyword_deck, read_lsdyna_directory, read_nodout
from .metrics import courant_number, mesh_courant_number, mesh_wall_distance, wall_distance
from .model import CellBlock, Mesh, PolyMesh
from .quality import METRICS, element_metrics, fv_metrics, surface_checks
from .repair import (
    RepairPlan,
    RepairResult,
    close_holes,
    drop_degenerate_faces,
    drop_duplicate_faces,
    merge_vertices,
    orient_consistently,
    repair_surface,
    surface_summary,
)
from .report import build as mesh_report
from .report import view_payload
from .structural import Part, StructuralRun
from .training_set import TrainingSet, build_training_set
from .vtk_series import read_vtk_frame, read_vtk_series, write_vtk_frame, write_vtk_series

__all__ = ["FormatError", "kind_of", "read_any", "read_meshio", "read_points", "read_stl", "CellBlock", "Mesh",
           "PolyMesh", "METRICS", "element_metrics", "fv_metrics", "surface_checks", "mesh_report", "view_payload",
           "courant_number", "wall_distance", "mesh_courant_number", "mesh_wall_distance",
           "repair", "RepairResult", "RepairPlan", "surface_summary", "merge_vertices", "drop_degenerate_faces",
           "drop_duplicate_faces", "orient_consistently", "close_holes", "repair_surface",
           "CONTACT_FIELDS", "ResultBlock", "read_blocks", "contact", "coordinates", "coulomb_ratio",
           "displacement", "stress", "Totals", "read_totals", "read_displacements", "Status", "read_status",
           "Verdict", "read_log", "finite", "FINISHED", "GIVE_UP",
           "Part", "StructuralRun", "TrainingSet", "build_training_set",
           "read_vtk_frame", "read_vtk_series", "write_vtk_frame", "write_vtk_series",
           "DynaDeck", "DynaHistory", "read_keyword_deck", "read_lsdyna_directory", "read_nodout"]
