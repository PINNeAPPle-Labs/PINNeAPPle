"""OpenRadioss (open-source explicit FEA) bridge: deck editing for DoE studies,
Docker-based execution, and readers for ANIM->VTK results.

No Python dependency beyond numpy; running cases requires Docker and an
unpacked OpenRadioss release (https://github.com/OpenRadioss/OpenRadioss/releases).
"""
from .deck import read_shell_thickness, set_shell_thickness, stage_case, write_engine_anim
from .runner import OpenRadiossDockerConfig, run_openradioss_case, run_openradioss_cases, docker_available
from .vtk_reader import RadiossMesh, read_vtk_mesh, read_case_displacements

__all__ = [
    "read_shell_thickness", "set_shell_thickness", "stage_case", "write_engine_anim",
    "OpenRadiossDockerConfig", "run_openradioss_case", "run_openradioss_cases", "docker_available",
    "RadiossMesh", "read_vtk_mesh", "read_case_displacements",
]
