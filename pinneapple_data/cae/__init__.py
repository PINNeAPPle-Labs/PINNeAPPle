"""CAE data layer: read meshes and results from OpenFOAM, CalculiX/Abaqus, Gmsh, VTK, STL and point tables into one
``Mesh`` type; mesh quality (checkMesh-equivalent finite-volume metrics, element shape metrics, surface checks);
repair of surface defects (holes, orientation, degenerate and duplicate faces, split vertices), every operation
measured before and after; per-cell Courant number and wall distance.

The CalculiX run goes further than the mesh: ``read_blocks``/``contact``/``coulomb_ratio`` read the result blocks —
``CONTACT`` included — and ``read_totals`` reads the reaction totals of the ``.dat``, which together are what a
verification of a structural run compares against a closed form.
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

__all__ = ["FormatError", "kind_of", "read_any", "read_meshio", "read_points", "read_stl", "CellBlock", "Mesh",
           "PolyMesh", "METRICS", "element_metrics", "fv_metrics", "surface_checks", "mesh_report", "view_payload",
           "courant_number", "wall_distance", "mesh_courant_number", "mesh_wall_distance",
           "repair", "RepairResult", "RepairPlan", "surface_summary", "merge_vertices", "drop_degenerate_faces",
           "drop_duplicate_faces", "orient_consistently", "close_holes", "repair_surface",
           "CONTACT_FIELDS", "ResultBlock", "read_blocks", "contact", "coordinates", "coulomb_ratio",
           "displacement", "stress", "Totals", "read_totals", "read_displacements", "Status", "read_status",
           "Verdict", "read_log", "finite", "FINISHED", "GIVE_UP"]
