"""A graph network (MeshGraphNet) for linear elasticity: a steel cantilever under a tip load, on the FEM mesh.

1. Reference: ``numerical_solvers.solid_fem`` (3-D hexahedral FEM, C3D8I, checked against Timoshenko beam theory in
   ``tests/test_lab.py``). 12 load cases from 1 to 10 kN; MeshGraphNet (``graphnn.MeshGraphNet``) maps node
   features (position, clamped and loaded flags, the load broadcast to every node) to the vertical displacement on
   the same mesh.
2. ``pp.validate(predict, {"valid_range", "outputs", "relations"}, data={"test"}, query=...)`` on the quantity of
   interest, the tip deflection: validity range, error against FEM solutions the network never saw, and a relation
   from the physics (more load, more deflection).
3. Trust card and ``pinneapple check`` outputs.
4. Failure caught: a 30 kN query, three times the largest training load (``applicability.range``). The network
   still returns a number: 4.2 mm, where the FEM gives 5.6 mm (25 % low; within the range its error is 2 %).

Run: ``python examples/validation/04_meshgraphnet_elasticity.py [--full]``.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import torch

import pinneapple as pp
from pinneapple_analysis.validation.gallery import conclude
from pinneapple_neural.architectures.graphnn.base import GraphBatch
from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
from pinneapple_neural.utils import fit_supervised
from pinneapple_simulation.numerical_solvers.solid_fem import SolidFEM, box_mesh

FAMILY, PHYSICS, CATCHES = "graph", "elasticity", "applicability.range"
L, W, H, E, NU = 2.0, 0.1, 0.2, 210e9, 0.3
MESH = box_mesh(L, W, H, 10, 1, 2)
CLAMP, TIP = MESH.nodes_on(x=0.0), MESH.face_nodes("x+")
HEX_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (5, 6), (6, 7), (7, 4), (0, 4), (1, 5), (2, 6), (3, 7)]


def fem_uz(load_kn: float) -> np.ndarray:
    f = SolidFEM(MESH, E, NU)
    f.fix(CLAMP)
    f.load_face(TIP, (0.0, 0.0, -1e3 * load_kn))
    return f.solve().u[:, 2]


def edge_index() -> torch.Tensor:
    e = {tuple(sorted((int(el[a]), int(el[b])))) for el in MESH.elements for a, b in HEX_EDGES}
    e = np.array(sorted(e)).T
    return torch.tensor(np.concatenate([e, e[::-1]], 1), dtype=torch.long)


def node_features(load_kn) -> np.ndarray:
    load_kn = np.atleast_1d(np.asarray(load_kn, float))
    pos = MESH.nodes / np.array([L, W, H])
    clamp = np.zeros(len(MESH.nodes))
    clamp[CLAMP] = 1.0
    tip = np.zeros(len(MESH.nodes))
    tip[TIP] = 1.0
    base = np.column_stack([pos, clamp, tip])
    return np.stack([np.column_stack([base, np.full(len(pos), p / 10.0)]) for p in load_kn])   # (B, N, 6): the
    # load is a global parameter, broadcast to every node (the usual MeshGraphNet input for case parameters)


def run(smoke: bool = True, out_dir=None):
    loads = np.linspace(1.0, 10.0, 12)
    U = np.stack([fem_uz(p) for p in loads])
    scale = float(np.abs(U).max())
    ei = edge_index()
    torch.manual_seed(0)
    net = MeshGraphNet(6, 1, hidden_dim=64, n_message_passing=4)

    def forward(m, x):
        return m(GraphBatch(x=x, edge_index=ei)).y[..., 0]
    fit_supervised(net, node_features(loads), U / scale, steps=600 if smoke else 3000, lr=2e-3, batch=12,
                   forward=forward)

    def predict(q):                                                     # QoI: mean tip deflection, mm
        with torch.no_grad():
            uz = forward(net, torch.tensor(node_features(q["load_kn"]), dtype=torch.float32)).numpy() * scale
        return {"tip_deflection_mm": -1e3 * uz[:, TIP].mean(1)}
    test_loads = np.array([1.7, 3.3, 5.5, 8.2, 9.6])
    test = {"load_kn": test_loads, "tip_deflection_mm": np.array([-1e3 * fem_uz(p)[TIP].mean() for p in test_loads])}
    spec = {"valid_range": {"load_kn": (1.0, 10.0)}, "outputs": ["tip_deflection_mm"],
            "relations": [{"kind": "monotonic", "name": "deflection-grows-with-load", "input": "load_kn",
                           "output": "tip_deflection_mm", "direction": "increasing"}],
            "name": "steel cantilever 2 m x 0.1 m x 0.2 m, tip load"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        good = pp.validate(predict, spec, data={"test": test}, query={"load_kn": test_loads})
        bad = pp.validate(predict, spec, data={"test": test}, query={"load_kn": [30.0]})
    return conclude("04_meshgraphnet_elasticity", good, bad, expect=CATCHES, family=FAMILY, physics=PHYSICS,
                    out_dir=out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true", help="full size instead of the CI smoke size")
    print(run(smoke=not ap.parse_args().full).good)
