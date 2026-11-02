"""Exact B-rep geometry from STEP files (issue #1114): import every body at its real size, with its faces,
their surface types and parameters, and the kernel's exact volume and area.

    from pinneapple_design.geometry.brep import read_step
    bodies = read_step("part.step")            # list[CadBody], SI units (metres)
    bodies[0].volume_m3, bodies[0].face_counts()

The exact kernel is SMLib (NVIDIA, Apache-2.0) through ``smlibpy`` (MIT): ``pip install "pinneapple[brep]"``.
For STEP -> mesh without an exact kernel, :func:`pinneapple_design.geometry.io.step.step_to_mesh` (gmsh) remains.
"""
from .diagnostics import (
    discrete_curvature,
    edge_continuity,
    exact_curvature,
    highlight_lines,
    refine,
    render,
    zebra,
)
from .step import (
    CadBody,
    CadFace,
    available_backends,
    read_step,
    write_step,
)
from .tessellate import QUALITY, TessMesh, tessellate

__all__ = ["CadBody", "CadFace", "available_backends", "read_step", "write_step", "QUALITY", "TessMesh", "tessellate",
           "exact_curvature", "discrete_curvature", "edge_continuity", "zebra", "highlight_lines", "refine", "render"]
