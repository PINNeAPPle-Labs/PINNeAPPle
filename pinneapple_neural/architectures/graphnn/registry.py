"""Registry and catalog for graph neural network model family."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .base import GraphModelBase
from .gnn import GraphNeuralNetwork
from .equivariant_gnn import EquivariantGNN
from .gnn_ode import GraphNeuralODE
from .spatiotemporal_gnn import SpatiotemporalGNN
from .graphcast import GraphCast
from .mesh_graph_net import MeshGraphNet
from .mgn_dynamics import MeshDynamicsMGN


_REGISTRY: dict[str, type[Any]] = {
    "gnn": GraphNeuralNetwork,
    "graph_neural_network": GraphNeuralNetwork,

    "equivariant_gnn": EquivariantGNN,
    "egnn": EquivariantGNN,

    "graph_neural_ode": GraphNeuralODE,
    "gnn_ode": GraphNeuralODE,

    "spatiotemporal_gnn": SpatiotemporalGNN,
    "stgnn": SpatiotemporalGNN,

    "graphcast": GraphCast,

    "mesh_graph_net": MeshGraphNet,
    "mgn": MeshGraphNet,
    "meshgraphnet": MeshGraphNet,

    # Transient dynamics on a static mesh: the input is a mesh plus a trajectory and the output is a rollout, not a
    # per-node field on a topology shared by the batch, so this one declares its own capabilities through the
    # class attribute ``MeshDynamicsMGN.capabilities`` (read by the getter below).
    "mesh_dynamics_mgn": MeshDynamicsMGN,
    "mesh_dynamics": MeshDynamicsMGN,
    "mgn_dynamics": MeshDynamicsMGN,
}

def register_into_global() -> None:
    from pinneapple_neural.architectures._registry_bridge import register_family_registry

    def capabilities(name: str, cls) -> dict:
        # A model may carry its own ``capabilities`` dict (MeshDynamicsMGN does); everything else gets the family
        # default — they all take a graph object or (x, edge_index) — see forward_batch() on
        # gnn.py/mesh_graph_net.py/etc — never a plain (N, in_dim) tensor.
        own = getattr(cls, "capabilities", None)
        if isinstance(own, dict) and own:
            return dict(own)
        return {"input_kind": "graph", "expects": ["graph"], "predicts": ["u"]}

    register_family_registry(_REGISTRY, family="graphnn", capabilities_getter=capabilities)

@dataclass
class GraphCatalog:
    registry: dict[str, type[Any]] = None

    def __post_init__(self):
        self.registry = dict(_REGISTRY)

    def list(self):
        return sorted(self.registry.keys())

    def get(self, name: str) -> type[GraphModelBase]:
        key = name.lower().strip()
        if key not in self.registry:
            raise KeyError(f"Unknown graph model '{name}'. Available: {self.list()}")
        return self.registry[key]

    def build(self, name: str, **kwargs) -> GraphModelBase:
        return self.get(name)(**kwargs)
