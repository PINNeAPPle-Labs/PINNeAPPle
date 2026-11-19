"""Explicit CFD/FEM mesh file publishing CLI for Physics Dashboard Studio.

Requires meshio installed and a supported file containing triangular surface
cells. Volumetric cells are not silently converted to surfaces.
"""
from __future__ import annotations

import argparse
import os

from .runtime_adapters import FEMResultPublisher


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Publish measured CFD/FEM surface mesh")
    parser.add_argument("mesh_path", help="meshio-readable mesh file")
    parser.add_argument("--endpoint", required=True, help="HTTPS Engineering API origin")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--channel", default="mesh")
    parser.add_argument("--field", default=None, help="named scalar point_data field")
    parser.add_argument("--coordinate-unit", default="m")
    parser.add_argument("--field-unit", default=None)
    args = parser.parse_args(argv)
    token = os.environ.get("PINNEAPPLE_DASHBOARD_API_KEY")
    if not token:
        parser.error("PINNEAPPLE_DASHBOARD_API_KEY is required")
    try:
        import meshio
    except ImportError as exc:
        raise SystemExit("Install meshio to read CFD/FEM mesh files") from exc
    mesh = meshio.read(args.mesh_path)
    result = FEMResultPublisher(args.endpoint, token, args.case_id).publish_mesh(
        mesh, field_name=args.field, channel=args.channel,
        coordinate_unit=args.coordinate_unit, field_unit=args.field_unit)
    print("Published scientific artifact:", result.get("id", "unknown"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
