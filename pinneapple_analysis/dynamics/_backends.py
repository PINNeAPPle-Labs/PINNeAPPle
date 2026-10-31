"""Which array backends this machine has, without importing them.

The dynamics toolkit computes with NumPy and SciPy. PyTorch and JAX are used only where a method offers an
autodiff path (derivatives for Fisher information, neural observables for Koopman), and only when the caller asks
for them; :func:`capabilities` reports what is installed by looking for the package, so calling it never pays the
import cost of either. :func:`to_numpy` turns a tensor or a JAX array into a NumPy array by duck typing, again
without importing the library.

Note: the ``pinneapple_analysis`` package itself imports PyTorch on load (its uncertainty and validation
sub-packages are written in it, and ``torch`` is a core dependency of ``pinneapple``). The modules of
``pinneapple_analysis.dynamics`` do not import it at module level.
"""
from __future__ import annotations

import importlib
import importlib.util
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["BackendUnavailable", "Capabilities", "capabilities", "require", "to_numpy"]

_INSTALL = {
    "torch": 'pip install torch  (a core dependency of pinneapple; see https://pytorch.org for GPU builds)',
    "jax": 'pip install "jax[cpu]"  (optional; see https://jax.readthedocs.io for GPU builds)',
    "sympy": "pip install sympy",
    "scipy": "pip install scipy",
}


class BackendUnavailable(ImportError):
    """A method was asked to use a backend that is not installed."""


@dataclass(frozen=True)
class Capabilities:
    numpy: str
    scipy: str | None
    sympy: str | None
    torch: str | None
    jax: str | None

    def has(self, backend: str) -> bool:
        return getattr(self, backend) is not None

    @property
    def autodiff(self) -> tuple[str, ...]:
        """The installed backends that provide automatic differentiation."""
        return tuple(b for b in ("torch", "jax") if self.has(b))

    def to_dict(self) -> dict[str, str | None]:
        return {"numpy": self.numpy, "scipy": self.scipy, "sympy": self.sympy, "torch": self.torch, "jax": self.jax}


def _installed_version(pkg: str) -> str | None:
    if importlib.util.find_spec(pkg) is None:
        return None
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version(pkg)
    except PackageNotFoundError:
        return "unknown"


def capabilities() -> Capabilities:
    """The installed backends and their versions (``None``: not installed). Nothing is imported."""
    return Capabilities(numpy=np.__version__, scipy=_installed_version("scipy"), sympy=_installed_version("sympy"),
                        torch=_installed_version("torch"), jax=_installed_version("jax"))


def require(backend: str, purpose: str = ""):
    """Import and return ``backend`` (``"torch"``, ``"jax"``, ``"sympy"``, ``"scipy"``), or raise
    :class:`BackendUnavailable` with the install command and what it was needed for."""
    if backend not in _INSTALL:
        raise ValueError(f"unknown backend {backend!r}; one of {sorted(_INSTALL)}")
    if importlib.util.find_spec(backend) is None:
        why = f" for {purpose}" if purpose else ""
        raise BackendUnavailable(f"{backend} is needed{why} but is not installed: {_INSTALL[backend]}")
    return importlib.import_module(backend)


def to_numpy(x: Any, dtype=np.float64) -> np.ndarray:
    """``x`` as a NumPy array: a PyTorch tensor (detached, moved to the CPU), a JAX array, or anything
    ``np.asarray`` accepts. ``dtype=None`` keeps the dtype."""
    mod = type(x).__module__.split(".")[0]
    if mod == "torch":
        x = x.detach().cpu().numpy()
    elif mod in ("jax", "jaxlib"):
        x = np.asarray(x)
    return np.asarray(x) if dtype is None else np.asarray(x, dtype=dtype)
