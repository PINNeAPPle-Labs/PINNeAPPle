"""Optional WorldParticle trajectory interoperability (upstream MIT).

No upstream model code is vendored. The adapter is importable without CUDA,
Open3D, FlashAttention or a local WorldParticle checkout.
"""
from .adapter import (
    WorldParticleState,
    WorldParticleBackend,
    load_worldparticle_npz,
    save_worldparticle_npz,
    worldparticle_to_upd,
    validate_worldparticle_rollout,
)

__all__ = [
    "WorldParticleState", "WorldParticleBackend",
    "load_worldparticle_npz", "save_worldparticle_npz",
    "worldparticle_to_upd", "validate_worldparticle_rollout",
]
