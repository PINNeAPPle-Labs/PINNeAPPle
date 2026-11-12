"""A film of a real flow, from the library's own parametric car.

Runs the small version of the pipeline — 60 cells across the body, a few hundred steps, six frames — so the
whole chain (lattice Boltzmann solve -> OpenVDB volume -> Blender EEVEE -> frames) is exercised in a couple of
minutes. The delivered films use `--cells 120 --steps 2200 --frames 144` and take the better part of an hour.

    PYTHONPATH=. python examples/cinematic_flow/run.py

Needs Blender on PATH and ffmpeg. Without either, the solve still runs and the NPZ fields are left on disk —
which is the half of the work that a re-render can always reuse.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneapple_simulation.pipelines.cinematic_flow import CASES, build, make_film  # noqa: E402
from pinneapple_simulation.pipelines.cinematic_flow.solve import plan_tunnel  # noqa: E402

OUT = os.path.join(_HERE, "_out", "carro")


def main() -> dict:
    case = build("carro")
    tunnel = plan_tunnel(case, cells_per_length=60, steps=700, u_lattice=0.06, re=1200)
    print(f"{CASES[case.name]}: malha {tunnel.shape} ({tunnel.cells / 1e6:.2f} M células), "
          f"dx = {tunnel.dx * 1000:.1f} mm, Re_lattice = {tunnel.re:.0f}")
    ficha = make_film(case, out_dir=OUT, cells_per_length=60, steps=700, snapshots=6, device="cpu",
                      frames=48, fps=16, res_x=960, res_y=540, samples=24, force_solve=True)
    print(f"malha   {ficha['malha']}  ({ficha['celulas'] / 1e6:.2f} M células)")
    print(f"Re_fis  {ficha['re_fisico']:.0f} a {ficha['u_inf_ms']} m/s")
    print(f"vídeo   {ficha['video']}")
    return ficha


if __name__ == "__main__":
    main()
