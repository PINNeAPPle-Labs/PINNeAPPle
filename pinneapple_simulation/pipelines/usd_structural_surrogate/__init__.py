"""The geometry the surrogate learns on: an NVIDIA OpenUSD asset, and a parametric family anchored to it.

NVIDIA's public USD library (the ``omniverse-content-production`` buckets Isaac Sim reads from) is a robotics and
warehouse prop library -- bins, mugs, tables, screws, robot arms. It ships **no** structural-mechanics benchmark
part, so "take a ready geometry from OpenUSD" cannot mean "download a bracket". What it does ship, under
``Props/Mounts/``, is the pedestal a UR10 arm is bolted to: a round base plate with a bolt circle and a column up
to the robot flange. That is a real mounting interface, and it is what this pipeline anchors to.

    from pinneapple_simulation.pipelines.usd_structural_surrogate import (
        NVIDIA_UR10_MOUNT, fetch_asset, read_mesh, mount_interface, DesignSpace)

    path = fetch_asset(NVIDIA_UR10_MOUNT, "assets/ur10_mount.usd")
    mesh = read_mesh(path)                       # points, faces, units, sha256
    iface = mount_interface(mesh)                # plate diameter, plate thickness, radial extent
    space = DesignSpace.from_interface(iface)    # the bracket family, nominal point at the asset

Modules: :mod:`.usd_asset` (fetch, read, measure), :mod:`.design` (the family and its gmsh geometry),
:mod:`.grid` (the regular grid the operator lives on), :mod:`.campaign` (the CalculiX runs and the dataset),
:mod:`.surrogate` (the PhysicsNeMo FNO and the ``virtual_ccx`` predictor), :mod:`.checks` (the physics checks).

The asset is fetched once and cached; the numbers extracted from it go into the run manifest, so a report can say
which asset, which sha256, and which measurements the family was built from.
"""
from .campaign import build_campaign
from .design import NOMINAL, DesignSpace, bracket_geo, run_design
from .grid import GridSpec, occupancy, rasterize, sample_field
from .usd_asset import NVIDIA_UR10_MOUNT, UsdMesh, fetch_asset, mount_interface, read_mesh

__all__ = ["NVIDIA_UR10_MOUNT", "UsdMesh", "fetch_asset", "read_mesh", "mount_interface", "DesignSpace",
           "bracket_geo", "run_design", "NOMINAL", "GridSpec", "occupancy", "rasterize", "sample_field",
           "build_campaign"]