"""A film of a real flow, from the library's own parametric car.

Runs the small version of the pipeline — 60 cells across the body, a few hundred steps, six frames — so the
whole chain (lattice Boltzmann solve -> OpenVDB volume -> Blender EEVEE -> frames) is exercised in a couple of
minutes. The delivered films use `--cells 120 --steps 2200 --frames 144` and take the better part of an hour.

    PYTHONPATH=. python examples/cinematic_flow/run.py

Needs Blender on PATH and ffmpeg. Without either, the solve still runs and the NPZ fields are left on disk —
which is the half of the work that a re-render can always reuse.

Before it renders anything it checks the scalar the smoke is made of, because a picture is only worth its field:
the Q-criterion has to be positive in solid rotation, exactly zero in pure shear, and negative in pure strain.
Pure shear is the case that matters — vorticity magnitude is large there and Q is zero, and that is precisely
why the film is built on Q and not on |ω|.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pinneapple_simulation.pipelines.cinematic_flow import CASES, build, make_film  # noqa: E402
from pinneapple_simulation.pipelines.cinematic_flow.solve import plan_tunnel, q_criterion  # noqa: E402

OUT = os.path.join(_HERE, "_out", "carro")


def q_criterion_checks(n: int = 32, dx: float = 0.1) -> list[dict]:
    """Q on three fields whose answer is known without solving anything."""
    x, y, z = np.meshgrid(*(np.linspace(-1.0, 1.0, n),) * 3, indexing="ij")   # axis 0 is x, 1 is y, 2 is z
    o = np.zeros_like(x)
    inside = (x * x + y * y + z * z) < 0.36
    fields = {
        "solid rotation u=(-y, x, 0)": np.stack([-y, x, o]),
        "pure shear u=(y, 0, 0)": np.stack([y, o, o]),
        "pure strain u=(x, -y, 0)": np.stack([x, -y, o]),
    }
    means = {k: float(q_criterion(u, dx)[inside].mean()) for k, u in fields.items()}
    ref = means["solid rotation u=(-y, x, 0)"]
    out = []
    for name, want in (("solid rotation u=(-y, x, 0)", 1.0), ("pure shear u=(y, 0, 0)", 0.0),
                       ("pure strain u=(x, -y, 0)", -1.0)):
        got = means[name] / ref if ref else 0.0
        out.append({"name": f"Q-criterion: {name}", "kind": "reference", "value": round(got, 6),
                    "reference": want, "rtol": 1e-3, "passed": bool(abs(got - want) < 1e-3),
                    "detail": "Q normalised by the solid-rotation value, so the check is scale free"})
    return out


def main() -> dict:
    checks = q_criterion_checks()
    for c in checks:
        print(f"{'ok ' if c['passed'] else 'FALHA'} {c['name']}: {c['value']:+.4f} (esperado {c['reference']:+.1f})")
    if not all(c["passed"] for c in checks):
        raise SystemExit("o critério Q não passou nas verificações: não faz sentido filmar o campo")

    case = build("carro")
    tunnel = plan_tunnel(case, cells_per_length=60, steps=700, u_lattice=0.06, re=1200)
    print(f"{CASES[case.name]}: malha {tunnel.shape} ({tunnel.cells / 1e6:.2f} M células), "
          f"dx = {tunnel.dx * 1000:.1f} mm, Re_lattice = {tunnel.re:.0f}")
    ficha = make_film(case, out_dir=OUT, cells_per_length=60, steps=700, snapshots=6, device="cpu",
                      frames=48, fps=16, res_x=960, res_y=540, samples=16, force_solve=True)
    print(f"malha   {ficha['malha']}  ({ficha['celulas'] / 1e6:.2f} M células)")
    print(f"Re_fis  {ficha['re_fisico']:.0f} a {ficha['u_inf_ms']} m/s")
    print(f"vídeo   {ficha['video']}")

    summary = {
        "checks": checks,
        "malha": ficha["malha"],
        "celulas": ficha["celulas"],
        "dx_m": ficha["dx_m"],
        "re_lattice": ficha["re_lattice"],
        "re_fisico": ficha["re_fisico"],
        "passos": ficha["passos"],
        "video": ficha["video"],
    }
    os.makedirs(os.path.dirname(os.path.join(_HERE, "_out", "summary.json")), exist_ok=True)
    with open(os.path.join(_HERE, "_out", "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


if __name__ == "__main__":
    main()
