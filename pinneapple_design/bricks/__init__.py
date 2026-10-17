"""Interlocking toy bricks: geometry, physics and a self-modifying design loop.

Modules
-------
parts        nominal dimensions (8 mm stud pitch, 3.2 mm plate, 9.6 mm brick), the part catalogue, masses, colours,
             and the stud-joint model (:class:`JointModel`, clutch 0.98 N per stud, cited)
assembly     grid assemblies, stud-tube connectivity, validity rules (no overlaps, supported placement)
ldraw        .ldr / .mpd import and export (LDraw-compatible), the LDraw parts library loaded at run time
mesh         brick meshes, glTF (per-brick colours), STL and USD export
statics      equilibrium LP: contact forces, stud utilisation, joint failures, tipping explanations, load capacity
closed_form  tipping, harmonic stacking, stud cantilever, simply supported beam
dynamics     Newton rigid-body replay (welded clusters, released joints, table edges)
fea          Hex8 finite elements on the voxelised build
goals        cantilever, bridge and tower goals with explained evaluations
search       mutation / grammar edits, NSGA-II evolution and a GP-surrogate-guided search at equal budget
visuals      evolution timeline, twin3d viewer, GIF / MP4, Blender renders, building instructions

>>> from pinneapple_design.bricks import Assembly, analyse_assembly
>>> a = Assembly()
>>> a.add("brick_2x4", 0, 0, 0)
0
>>> analyse_assembly(a).stable
True
"""
from .assembly import Assembly, Brick, Connection, Ground, stack
from .goals import (
    GOALS,
    BridgeGoal,
    CantileverGoal,
    Evaluation,
    Goal,
    TowerGoal,
    reference_cantilever,
)
from .parts import (
    BRICK_MM,
    CATALOG,
    PALETTE,
    PITCH_MM,
    PLATE_MM,
    JointModel,
    PartType,
    part,
    smooth_block,
)
from .search import SearchResult, compare, evolve, mutate, surrogate_search
from .statics import (
    Load,
    StaticsResult,
    Structure,
    analyse,
    analyse_assembly,
    capacity,
    from_assembly,
    progressive_failure,
    top_load,
)

__all__ = [
    "Assembly", "Brick", "Connection", "Ground", "stack",
    "GOALS", "BridgeGoal", "CantileverGoal", "Evaluation", "Goal", "TowerGoal", "reference_cantilever",
    "BRICK_MM", "CATALOG", "PALETTE", "PITCH_MM", "PLATE_MM", "JointModel", "PartType", "part", "smooth_block",
    "SearchResult", "compare", "evolve", "mutate", "surrogate_search",
    "Load", "StaticsResult", "Structure", "analyse", "analyse_assembly", "capacity", "from_assembly",
    "progressive_failure", "top_load",
]
