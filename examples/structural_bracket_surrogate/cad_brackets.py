"""Parametric CAD bracket families with explicit millimetre dimensions.

These are original benchmark geometries inspired by the *families* of the
published study, not exact copies of the author's undisclosed CAD.
Requires cadquery. Exports watertight B-rep STEP, not voxel geometry.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import cadquery as cq


def build(family: str, *, width: float = 50.0, length: float = 75.0,
          height: float = 75.0, thickness: float = 6.0,
          hole_diameter: float = 8.0, offset: float = 35.0,
          gusset_thickness: float = 6.0) -> cq.Workplane:
    """Return solid bracket. All dimensions in mm."""
    if min(width, length, height, thickness, hole_diameter, gusset_thickness) <= 0:
        raise ValueError("Dimensions must be positive")
    if thickness >= min(length, height) / 2:
        raise ValueError("Thickness too large")
    if hole_diameter >= min(width, height - thickness, length - thickness):
        raise ValueError("Hole too large for selected envelope")
    if family not in {"L", "G", "U", "Z"}:
        raise ValueError("family must be L, G, U or Z")

    # Plate union: x horizontal leg, z vertical; y is bracket width.
    base = cq.Workplane("XY").box(length, width, thickness).translate(
        (length/2, 0, thickness/2))
    vertical = cq.Workplane("XY").box(thickness, width, height).translate(
        (thickness/2, 0, height/2))
    if family in {"L", "G"}:
        shape = base.union(vertical)
        # Through-hole axes are normal to each mounting plate.
        base_hole = cq.Solid.makeCylinder(
            hole_diameter/2, thickness + 2,
            cq.Vector(length*0.72, 0, -1), cq.Vector(0, 0, 1))
        wall_hole = cq.Solid.makeCylinder(
            hole_diameter/2, thickness + 2,
            cq.Vector(-1, 0, height*0.72), cq.Vector(1, 0, 0))
        shape = shape.cut(base_hole).cut(wall_hole)
        if family == "G":
            # Triangular reinforcement in the x-z plane, extruded along y.
            rib = (cq.Workplane("XZ", origin=(0, gusset_thickness/2, 0))
                   .polyline([(thickness, thickness),
                              (length*0.65, thickness),
                              (thickness, height*0.65)])
                   .close().extrude(gusset_thickness))
            shape = shape.union(rib)
    elif family == "U":
        # Two vertical ears joined by a base, with coaxial pin holes.
        gap = width - 2*thickness
        if gap <= hole_diameter:
            raise ValueError("Clevis gap must exceed hole diameter")
        ear1 = vertical.translate((0, -(width-thickness)/2, 0))
        ear2 = vertical.translate((0, +(width-thickness)/2, 0))
        # Rebuild ears with thickness across y, not x.
        ear1 = cq.Workplane("XY").box(length, thickness, height).translate(
            (length/2, -(width-thickness)/2, height/2))
        ear2 = cq.Workplane("XY").box(length, thickness, height).translate(
            (length/2, +(width-thickness)/2, height/2))
        shape = base.union(ear1).union(ear2)
        pin_hole = cq.Solid.makeCylinder(
            hole_diameter/2, width+2,
            cq.Vector(length*0.7, -width/2-1, height*0.7),
            cq.Vector(0, 1, 0))
        shape = shape.cut(pin_hole)
    else:
        if offset <= thickness:
            raise ValueError("Z offset must exceed thickness")
        top = cq.Workplane("XY").box(length, width, thickness).translate(
            (length/2, 0, offset + thickness/2))
        web = cq.Workplane("XY").box(thickness, width, offset+thickness).translate(
            (thickness/2, 0, (offset+thickness)/2))
        shape = base.union(web).union(top)
        for z in (-1, offset-1):
            hole = cq.Solid.makeCylinder(
                hole_diameter/2, thickness+2,
                cq.Vector(length*0.7, 0, z), cq.Vector(0,0,1))
            shape = shape.cut(hole)
    if not shape.val().isValid() or len(shape.solids().vals()) != 1:
        raise ValueError("CAD geometry is not a single valid solid")
    return shape


def export_family(family: str, destination: Path, **params) -> dict:
    shape = build(family, **params)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cq.exporters.export(shape, str(destination))
    return {"family": family, "step": str(destination),
            "volume_mm3": shape.val().Volume(),
            "bounding_box_mm": [shape.val().BoundingBox().xlen,
                                shape.val().BoundingBox().ylen,
                                shape.val().BoundingBox().zlen]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="artifacts/cad")
    args = parser.parse_args()
    for family in ("L", "G", "U", "Z"):
        print(export_family(family, Path(args.output_dir)/f"{family}.step"))
