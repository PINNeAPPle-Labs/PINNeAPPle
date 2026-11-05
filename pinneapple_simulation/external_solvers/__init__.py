"""External tool integrations for PINNeAPPle.

Subpackages
-----------
openfoam  — case staging, execution, field sampling, and UPD packaging
matlab    — MATLAB Engine API wrapper, subprocess runner, .mat I/O
modelica  — FMU simulation (fmpy) and OpenModelica (OMPython)
fenics    — workflow layer on top of pinneapple_solvers.fenics_bridge
mujoco    — MuJoCo (google-deepmind/mujoco) trajectory collection + MJX parallel rollouts
genesis   — Genesis AI (Genesis-Embodied-AI/Genesis) multi-physics robotics simulation
newton_physics — NVIDIA Newton rigid-body solvers (XPBD, Featherstone, MuJoCo-Warp) with analytic checks
pict      — PICT (tum-pbs), a differentiable multi-block PISO solver on CUDA: cavity case, runner, Ghia checks

All subpackages use lazy / optional imports so that the base package remains
importable even when a specific external tool is not installed.
"""

# OpenFOAM (no optional deps beyond subprocess + pandas)
from .openfoam import (
    OpenFOAMCaseTemplate,
    OpenFOAMRunConfig,
    export_bundle,
    openfoam_case_to_upd,
    read_sampled_scalar_field,
    run_openfoam_case,
    run_sampling,
    stage_case_for_scenario,
    write_sample_dict_cloud,
)

# CFD/FEM open-format readers beyond OpenFOAM (CGNS, Exodus, Fluent mesh,
# Abaqus .inp/.odb bridge). See cfd_formats/__init__.py for per-format
# validation status -- these were not all checked against a real file from
# a real writer, unlike the OpenFOAM readers above.
try:
    from .cfd_formats import (
        abaqus_inp_to_upd,
        cgns_to_upd,
        exodus_to_upd,
        export_odb_fields,
        fluent_mesh_to_upd,
        load_exported_odb_npz,
        read_abaqus_inp_mesh,
        read_cgns_mesh_and_fields,
        read_exodus,
        read_fluent_mesh_nodes,
    )
except Exception:
    read_cgns_mesh_and_fields = None    # type: ignore
    cgns_to_upd = None                  # type: ignore
    read_exodus = None                  # type: ignore
    exodus_to_upd = None                # type: ignore
    read_fluent_mesh_nodes = None       # type: ignore
    fluent_mesh_to_upd = None           # type: ignore
    read_abaqus_inp_mesh = None         # type: ignore
    abaqus_inp_to_upd = None            # type: ignore
    export_odb_fields = None            # type: ignore
    load_exported_odb_npz = None        # type: ignore

# MATLAB (optional: matlab.engine or scipy)
try:
    from .matlab import (
        MATLABEngine,
        load_mat,
        mat_to_upd,
        run_matlab_function,
        run_matlab_script,
        save_mat,
    )
except Exception:
    MATLABEngine = None          # type: ignore
    run_matlab_script = None     # type: ignore
    run_matlab_function = None   # type: ignore
    load_mat = None              # type: ignore
    save_mat = None              # type: ignore
    mat_to_upd = None            # type: ignore

# Modelica / FMI (optional: fmpy, OMPython)
try:
    from .modelica import (
        FMUSimConfig,
        OMSimConfig,
        fmu_to_upd,
        modelica_result_to_upd,
        om_to_upd,
        simulate_fmu,
        simulate_openmodelica,
    )
except Exception:
    FMUSimConfig = None              # type: ignore
    simulate_fmu = None              # type: ignore
    fmu_to_upd = None                # type: ignore
    OMSimConfig = None               # type: ignore
    simulate_openmodelica = None     # type: ignore
    om_to_upd = None                 # type: ignore
    modelica_result_to_upd = None    # type: ignore

# FEniCS (optional: pinneapple_solvers.fenics_bridge + dolfinx / legacy FEniCS)
try:
    from .fenics import (
        FEniCSConfig,
        FEniCSWorkflow,
        dof_to_upd,
        solve_and_package,
    )
except Exception:
    FEniCSConfig = None       # type: ignore
    solve_and_package = None  # type: ignore
    dof_to_upd = None         # type: ignore
    FEniCSWorkflow = None     # type: ignore

# MuJoCo (optional: mujoco >= 3.0)
try:
    from .mujoco import (
        MJXParallelRunner,
        MuJoCoConfig,
        MuJoCoRunner,
        load_model,
        make_data,
        trajectories_to_upd,
        trajectory_to_upd,
    )
except Exception:
    MuJoCoConfig = None           # type: ignore
    load_model = None             # type: ignore
    make_data = None              # type: ignore
    MuJoCoRunner = None           # type: ignore
    trajectory_to_upd = None      # type: ignore
    trajectories_to_upd = None    # type: ignore
    MJXParallelRunner = None      # type: ignore

# TurboDesigner (optional: turbodesigner; falls back to built-in analytical solver)
try:
    from .turbodesigner import (
        TurboDesignerConfig,
        TurboDesignerWorkflow,
        run_turbodesigner,
        turbodesigner_to_upd,
    )
except Exception:
    TurboDesignerConfig = None       # type: ignore
    TurboDesignerWorkflow = None     # type: ignore
    run_turbodesigner = None         # type: ignore
    turbodesigner_to_upd = None      # type: ignore

# Genesis AI (optional: genesis-world)
try:
    from .genesis import (
        EntitySpec,
        GenesisConfig,
        GenesisRunner,
        build_scene,
        genesis_traj_to_upd,
        genesis_trajs_to_upd,
    )
except Exception:
    GenesisConfig = None          # type: ignore
    EntitySpec = None             # type: ignore
    build_scene = None            # type: ignore
    GenesisRunner = None          # type: ignore
    genesis_traj_to_upd = None    # type: ignore
    genesis_trajs_to_upd = None   # type: ignore

# PICT multi-block PISO on CUDA (optional, from source: PISOtorch + a CUDA device). Everything except the solver
# import runs without it, including the grid, the case validation and the Ghia et al. (1982) comparison.
# NVIDIA Newton rigid bodies (optional: newton; the scene description imports without it)
from .newton_physics import RigidScene, RigidTrajectory
from .newton_physics import simulate as newton_simulate
from .newton_physics import validate as newton_validate
from .pict import (
    GHIA,
    CavityCase,
    PICTResult,
    PICTRunConfig,
    build_cavity_domain,
    cavity_checks,
    pict_availability,
    pict_available,
    run_cavity,
    wall_refined_grid,
)

__all__ = [
    # OpenFOAM
    "OpenFOAMCaseTemplate",
    "stage_case_for_scenario",
    "OpenFOAMRunConfig",
    "run_openfoam_case",
    "write_sample_dict_cloud",
    "run_sampling",
    "read_sampled_scalar_field",
    "export_bundle",
    "openfoam_case_to_upd",
    # CFD/FEM open-format readers
    "read_cgns_mesh_and_fields", "cgns_to_upd",
    "read_exodus", "exodus_to_upd",
    "read_fluent_mesh_nodes", "fluent_mesh_to_upd",
    "read_abaqus_inp_mesh", "abaqus_inp_to_upd",
    "export_odb_fields", "load_exported_odb_npz",
    # MATLAB
    "MATLABEngine",
    "run_matlab_script",
    "run_matlab_function",
    "load_mat",
    "save_mat",
    "mat_to_upd",
    # Modelica / FMI
    "FMUSimConfig",
    "simulate_fmu",
    "fmu_to_upd",
    "OMSimConfig",
    "simulate_openmodelica",
    "om_to_upd",
    "modelica_result_to_upd",
    # FEniCS
    "FEniCSConfig",
    "solve_and_package",
    "dof_to_upd",
    "FEniCSWorkflow",
    # MuJoCo
    "MuJoCoConfig",
    "load_model",
    "make_data",
    "MuJoCoRunner",
    "trajectory_to_upd",
    "trajectories_to_upd",
    "MJXParallelRunner",
    # Genesis AI
    "GenesisConfig",
    "EntitySpec",
    "build_scene",
    "GenesisRunner",
    "genesis_traj_to_upd",
    "genesis_trajs_to_upd",
    # NVIDIA Newton
    "RigidScene",
    "RigidTrajectory",
    "pict_available",
    "pict_availability",
    "wall_refined_grid",
    "CavityCase",
    "build_cavity_domain",
    "PICTRunConfig",
    "PICTResult",
    "run_cavity",
    "cavity_checks",
    "GHIA",
    "newton_simulate",
    "newton_validate",
    # TurboDesigner
    "TurboDesignerConfig",
    "TurboDesignerWorkflow",
    "run_turbodesigner",
    "turbodesigner_to_upd",
]
