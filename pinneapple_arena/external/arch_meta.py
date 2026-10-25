"""Architecture-family labels (the `arch` field on ModelSpec)."""

from __future__ import annotations

from pinneapple_arena.external.schema import ArchMeta

ARCH_META: dict[str, ArchMeta] = {
    'pinn': ArchMeta(key='pinn', label='PINN Frameworks'),
    'operator': ArchMeta(key='operator', label='Neural Operators'),
    'diffsim': ArchMeta(key='diffsim', label='Differentiable Simulators'),
    'rigidbody': ArchMeta(key='rigidbody', label='Rigid-Body Engines'),
    'graph': ArchMeta(key='graph', label='Graph-Based Simulators'),
    'domain': ArchMeta(key='domain', label='Domain-Specific Models'),
}

