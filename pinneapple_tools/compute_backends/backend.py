"""Backend abstraction layer for PINNeAPPle.

Provides a simple global backend selector so PINN code can switch between
PyTorch (default) and JAX without modifying the calling code.
"""
from __future__ import annotations

from typing import Literal

BackendName = Literal["torch", "jax"]


class Backend:
    """Namespace for backend string constants."""
    TORCH: str = "torch"
    JAX: str = "jax"


_current_backend: str = Backend.TORCH


def set_backend(name: str) -> None:
    """Set the active computation backend.

    Parameters
    ----------
    name:
        ``"torch"``, ``"jax"`` or a name registered with
        ``pinneapple_core.backend.register_backend``. The matching
        :class:`~pinneapple_core.backend.PhysicsBackend` (array, ``grad``,
        ``solve``, ``integrate``, ... API) is ``get_physics_backend()``.

    Raises
    ------
    ValueError
        If *name* is not a recognised backend string.
    """
    global _current_backend
    from pinneapple_core.backend import list_backends  # registry: torch, jax and anything registered

    known = list_backends()
    if name not in known:
        raise ValueError(
            f"Unknown backend '{name}'. Choose one of {known} "
            "(add engines with pinneapple_core.backend.register_backend)."
        )
    _current_backend = name


def get_backend() -> str:
    """Return the name of the currently active backend."""
    return _current_backend
