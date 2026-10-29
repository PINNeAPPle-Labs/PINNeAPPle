"""Track C of the cross-framework benchmark: identify the viscosity of Burgers from noisy samples.

:func:`observations` draws the 2000 sample points (uniform in ``[-1, 1] x [0, 1]``), evaluates the exact Cole-Hopf
solution there and adds Gaussian noise of standard deviation ``0.01 * std(u)`` (1 % noise, the convention of
Raissi et al. 2019). Every framework gets the same arrays.
"""
from __future__ import annotations

import numpy as np

from .problems import get_problem
from .protocol import C

__all__ = ["observations", "NU_TRUE"]

NU_TRUE = get_problem("burgers").params["nu"]


def observations(n: int = C["n_obs"], noise_rel: float = C["noise_rel"], seed: int = C["obs_seed"]) -> dict[str, np.ndarray]:
    """``{"X", "u", "u_clean", "sigma"}``: points (n, 2) as (x, t), noisy and clean values (n, 1), float32."""
    rng = np.random.default_rng(seed)
    X = np.stack([rng.uniform(-1.0, 1.0, n), rng.uniform(0.0, 1.0, n)], axis=1)
    u = get_problem("burgers").exact(X)
    sigma = noise_rel * float(np.std(u))
    noisy = u + sigma * rng.standard_normal(n)
    return {"X": X.astype(np.float32), "u": noisy[:, None].astype(np.float32),
            "u_clean": u[:, None].astype(np.float32), "sigma": np.float32(sigma)}
