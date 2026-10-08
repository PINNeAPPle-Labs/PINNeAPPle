"""Surrogates for the airfoil flow: a MeshGraphNet on the CFD grid (fields + Cl, Cd, Cm) and an MLP ensemble.

The graph is a subset of the O-grid cells near the airfoil (every 2nd cell around, ~24 levels from the wall out to
~5 chords), so every design has the same nodes and edges and only the coordinates change. Edges join neighbours on
that lattice plus longer jumps (multi-scale), so a few message-passing rounds reach across the near field.

Node inputs: position, log wall distance, level, wall flag, free-stream direction, and the shape weights.
Node outputs: p, Ux, Uy, log(1 + nut/nu). A pooled readout over the wall nodes gives Cl, Cd, Cm.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from pinneapple_neural.architectures.graphnn.base import GraphBatch
from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet

from .geometry import BOUNDS
from .mesh import GridSpec, _distances, cell_centres, grid_points

FIELDS = ("p", "Ux", "Uy", "nut")
COEFS = ("Cl", "Cd", "Cm")
A_LO, A_HI = -2.0, 14.0


@dataclass
class Layout:
    grid: GridSpec = field(default_factory=GridSpec)
    i_step: int = 2
    max_dist: float = 5.0
    j_sel: np.ndarray = None
    i_sel: np.ndarray = None
    edge_index: np.ndarray = None

    def __post_init__(self):
        g = self.grid
        d = _distances(g.first, g.radius, g.nj)
        dc = 0.5 * (d[:-1] + d[1:])                               # cell-centre distance of each level
        jmax = int(np.searchsorted(dc, self.max_dist))
        # levels: dense in index near the wall (boundary layer), then every 3rd
        js = sorted(set([0, 1, 2, 4, 6, 9, 12, 15] + list(range(18, jmax + 1, 3))))
        self.j_sel = np.array([j for j in js if j < g.nj])
        self.i_sel = np.arange(0, g.ni, self.i_step)
        ni, nj = len(self.i_sel), len(self.j_sel)
        nid = lambda i, j: j * ni + (i % ni)                       # noqa: E731
        E = []
        for j in range(nj):
            for i in range(ni):
                for di, dj in ((1, 0), (0, 1), (4, 0), (0, 3)):
                    jj = j + dj
                    if 0 <= jj < nj:
                        a, b = nid(i, j), nid(i + di, jj)
                        E += [(a, b), (b, a)]
        self.edge_index = np.array(sorted(set(E))).T
        self.n_nodes = ni * nj
        self.shape2 = (nj, ni)
        self.cell_index = (self.j_sel[:, None] * g.ni + self.i_sel[None, :]).ravel()   # into the CFD cell list
        self.dist = dc[self.j_sel]

    def positions(self, shape: np.ndarray) -> np.ndarray:
        cc = cell_centres(grid_points(shape, self.grid))         # (nj, ni, 2)
        return cc[self.j_sel][:, self.i_sel].reshape(-1, 2)


def node_features(lay: Layout, shape: np.ndarray, alpha: float, pos: Optional[np.ndarray] = None) -> np.ndarray:
    pos = lay.positions(shape) if pos is None else pos
    nj, ni = lay.shape2
    a = math.radians(alpha)
    lvl = np.repeat(np.arange(nj) / (nj - 1), ni)
    ld = np.repeat(np.log10(lay.dist), ni)
    wall = (lvl == 0).astype(float)
    s = (np.asarray(shape) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0]) * 2 - 1
    r = np.linalg.norm(pos - [0.5, 0], axis=1)
    feats = np.c_[pos[:, 0] - 0.5, pos[:, 1], np.log10(r + 0.05), (ld + 3) / 3, lvl, wall,
                  np.full(len(pos), math.cos(a)), np.full(len(pos), math.sin(a)), np.full(len(pos), (alpha - 6) / 8),
                  np.tile(s, (len(pos), 1))]
    return feats.astype(np.float32)


def edge_features(lay: Layout, pos: np.ndarray) -> np.ndarray:
    s, d = lay.edge_index
    rel = pos[d] - pos[s]
    L = np.linalg.norm(rel, axis=1, keepdims=True) + 1e-12
    return np.c_[rel / L, np.log10(L) / 3 + 1].astype(np.float32)


class AeroGNN(nn.Module):
    def __init__(self, node_in: int, edge_in: int = 3, hidden: int = 48, mp: int = 8, n_fields: int = 4):
        super().__init__()
        self.gnn = MeshGraphNet(node_in, n_fields, edge_in_dim=edge_in, hidden_dim=hidden, n_layers=2,
                                n_message_passing=mp)
        self.head = nn.Sequential(nn.Linear(2 * hidden + node_in, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(),
                                  nn.Linear(128, len(COEFS)))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, wall: torch.Tensor):
        out = self.gnn(GraphBatch(x=x, edge_index=edge_index, edge_attr=edge_attr))
        h = out.extras["h"]                                       # (B, N, H)
        hw = h[:, wall]
        pooled = torch.cat([hw.mean(1), h.mean(1), x[:, 0]], -1)
        return out.y, self.head(pooled)


def coef_targets(cl, cd, cm):
    return np.stack([np.asarray(cl), np.log(np.asarray(cd)), np.asarray(cm)], -1)


def coef_from_targets(t):
    t = np.asarray(t)
    return t[..., 0], np.exp(t[..., 1]), t[..., 2]


class MLPEnsemble(nn.Module):
    """K small MLPs on (shape, alpha) -> (Cl, log Cd, Cm); the spread is the uncertainty."""

    def __init__(self, k: int = 5, width: int = 96):
        super().__init__()
        self.nets = nn.ModuleList([nn.Sequential(nn.Linear(8, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(),
                                                 nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))
                                   for _ in range(k)])

    @staticmethod
    def inputs(shape: np.ndarray, alpha: np.ndarray) -> np.ndarray:
        s = (np.atleast_2d(shape) - BOUNDS[:, 0]) / (BOUNDS[:, 1] - BOUNDS[:, 0]) * 2 - 1
        a = np.atleast_1d(alpha)
        return np.c_[s, (a - 6) / 8, np.sin(np.radians(a)) * 4].astype(np.float32)

    def forward(self, x):
        return torch.stack([n(x) for n in self.nets], 0)          # (K, B, 3)
