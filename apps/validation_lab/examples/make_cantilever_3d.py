"""Regenerate the 3-D example of the Physics Validation Lab (a few seconds):

    python apps/validation_lab/examples/make_cantilever_3d.py

A reduced-order model checked against the finite-element solution it replaces, on a 3-D solid:

* ``cantilever_rom.vtu``: the POD + GPR model of the Reduced Order Model Lab (rank 8; height and width in log scale;
  trained on 80 of the 100 FEM cantilever designs of ``apps/rom_lab/examples/cantilever_designs.npz``, the same random
  split, seed 0) evaluated at held-out design 6: H = 0.289 m, W = 0.173 m, load at 4.8° from -z, near the edge of
  the training box. Point data ``D`` (displacement, m) and ``von_mises`` (MPa). Design 6 is chosen on purpose: of the
  20 held-out designs it is the one where this model is worst (21 % / 28 %, against a 4.4 % mean), so the example
  shows a failing reference check.
* ``cantilever_fem.vtu``: the solid FEM of that design solved now (``pinneapple_lab.experiments.rom.cantilever``,
  30 x 3 x 6 hexahedra, clamped at x = 0, 10 kN on the face x = 2 m): the reference.

Both live on the same hexahedral mesh of the design. Every number comes from these two runs.
"""
from __future__ import annotations

import os

import meshio
import numpy as np

from pinneapple_data.snapshots import load_snapshots
from pinneapple_lab.experiments.rom import cantilever
from pinneapple_neural.architectures.rom.study import fit_reduced_model, split_snapshots

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "..", "rom_lab", "examples", "cantilever_designs.npz")
DESIGN = 6


def main():
    s = load_snapshots([("cantilever_designs.npz", open(DATA, "rb").read())])
    split = split_snapshots(s, how="random", test_fraction=0.2, seed=0)
    assert DESIGN in split.test, "the example design must be held out"
    rom = fit_reduced_model(s, split.train, "pod_gpr", 8, log_params=["H", "W"])
    H, W, angle = s.params[DESIGN]
    pred = rom.predict(params=s.params[DESIGN:DESIGN + 1])
    fem = cantilever(H, W, angle)
    cells = [("hexahedron", fem.mesh.elements)]
    meshio.write(os.path.join(HERE, "cantilever_fem.vtu"), meshio.Mesh(
        fem.mesh.nodes, cells, point_data={"D": fem.u, "von_mises": fem.von_mises / 1e6}))
    meshio.write(os.path.join(HERE, "cantilever_rom.vtu"), meshio.Mesh(
        fem.mesh.nodes, cells, point_data={"D": s.field_values("displacement", pred)[0] * 1e-3,
                                            "von_mises": s.field_values("von_mises", pred)[0]}))
    for n, a, b in (("D", s.field_values("displacement", pred)[0] * 1e-3, fem.u),
                    ("von_mises", s.field_values("von_mises", pred)[0], fem.von_mises / 1e6)):
        print(f"{n}: relative L2 of the ROM against the FEM {np.linalg.norm(a - b) / np.linalg.norm(b):.4f}")


if __name__ == "__main__":
    main()
