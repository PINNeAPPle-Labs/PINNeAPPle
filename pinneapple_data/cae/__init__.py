"""CAE data layer: read meshes and results from OpenFOAM, CalculiX/Abaqus, Gmsh, VTK, STL and point tables into one
``Mesh`` type; mesh quality (checkMesh-equivalent finite-volume metrics, element shape metrics, surface checks)."""
from .io import FormatError, kind_of, read_any, read_meshio, read_points, read_stl
from .model import CellBlock, Mesh, PolyMesh
from .quality import METRICS, element_metrics, fv_metrics, surface_checks
from .report import build as mesh_report, view_payload

__all__ = ["FormatError", "kind_of", "read_any", "read_meshio", "read_points", "read_stl", "CellBlock", "Mesh",
           "PolyMesh", "METRICS", "element_metrics", "fv_metrics", "surface_checks", "mesh_report", "view_payload"]
