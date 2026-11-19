"""``python -m pinneapple_simulation.pipelines.cinematic_flow`` — solve a case and film it.

    python -m pinneapple_simulation.pipelines.cinematic_flow carro --out filmes/carro --cells 140 --steps 2600
    python -m pinneapple_simulation.pipelines.cinematic_flow asa  --out filmes/asa --still
    python -m pinneapple_simulation.pipelines.cinematic_flow list
"""
from __future__ import annotations

import argparse
import json

from .cases import CASES
from .film import make_film, render_still


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="cinematic_flow", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("caso", nargs="?", default="carro", help="carro | asa | foguete | list")
    p.add_argument("--out", default="filmes", help="pasta da entrega (padrão: filmes)")
    p.add_argument("--cells", type=int, default=140, help="células ao longo do corpo (padrão: 140)")
    p.add_argument("--steps", type=int, default=2600, help="passos de lattice (padrão: 2600)")
    p.add_argument("--snapshots", type=int, default=10, help="instantes guardados (padrão: 10)")
    p.add_argument("--frames", type=int, default=144, help="quadros do vídeo (padrão: 144)")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--res", default="1920x1080")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--device", default="mps", help="cpu | mps | cuda")
    p.add_argument("--still", action="store_true", help="só um quadro (para olhar o look antes do vídeo)")
    p.add_argument("--frame", type=int, default=1, help="quadro do --still")
    p.add_argument("--force-solve", action="store_true", help="refazer a simulação mesmo se já houver meta.json")
    a = p.parse_args(argv)

    if a.caso == "list":
        print(json.dumps(CASES, indent=2, ensure_ascii=False))
        return 0

    res_x, _, res_y = a.res.partition("x")
    width, height = int(res_x), int(res_y or res_x)
    if a.still:
        path = render_still(a.caso, out_dir=a.out, frame=a.frame, frames=max(a.frame, 2),
                            cells_per_length=a.cells, res_x=width, res_y=height, samples=a.samples)
        print(path)
        return 0

    ficha = make_film(a.caso, out_dir=a.out, cells_per_length=a.cells, steps=a.steps,
                      snapshots=a.snapshots, device=a.device, frames=a.frames, fps=a.fps,
                      res_x=width, res_y=height, samples=a.samples, force_solve=a.force_solve)
    print(json.dumps(ficha, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
