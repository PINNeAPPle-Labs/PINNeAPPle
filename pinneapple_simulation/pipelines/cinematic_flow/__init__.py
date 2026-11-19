"""Cinematic flow films: a real 3-D solve of the library, rendered as light and smoke.

The pipeline is three steps, each one writing its artefact to disk so a re-render never repeats the solve:

1. :mod:`.solve` — the geometry of the case is voxelized into the lattice-Boltzmann tunnel
   (:class:`pinneapple_simulation.numerical_solvers.lbm_tunnel.WindTunnelLBM`), the flow is run, and the
   velocity field is written at a handful of instants, subsampled, as ``.npz``;
2. :mod:`.scene` — runs inside Blender: the velocity fields become an OpenVDB density grid (vorticity and
   wake deficit, not a colour map), the geometry becomes a lit surface, and a camera moves through a dark
   studio; the frames go out as PNG;
3. :mod:`.film` — the frames become an ``.mp4`` with ffmpeg.

Nothing in the picture is invented: the smoke is the simulated vorticity, the surface is the simulated body,
and every film records its resolution, Reynolds number and step count next to it.

    from pinneapple_simulation.pipelines.cinematic_flow import CASES, make_film

    make_film(CASES["carro"], out_dir="filmes/carro", cells_per_length=120, steps=2400)
"""
from .cases import CASES, Case, build
from .film import make_film

__all__ = ["CASES", "Case", "build", "make_film"]
