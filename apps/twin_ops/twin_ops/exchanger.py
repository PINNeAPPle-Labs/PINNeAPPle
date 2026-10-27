"""A counterflow water/water heat exchanger with fouling — the plant a twin is built for.

The model is the exchanger's own effectiveness-NTU solution. What it cannot know is the
fouling resistance: it is not an input anybody measures, it builds up over months, and it is
the number that decides when the exchanger is cleaned. That is the twin's job.

The *plant* is deliberately not exactly the model: the overall coefficient drifts a few
percent with the flow, an effect the model ignores. Without that, a twin would be correct by
construction and the coverage check would be meaningless.

Physical constants are ordinary engineering values for a shell-and-tube water exchanger; the
fouling trajectory is an asymptotic (Kern-Seaton style) curve, not a lab dataset.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

CP_WATER = 4180.0          # J/(kg K)
AREA = 12.0                # m2


def _effectiveness(ntu: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Counterflow effectiveness. ``cr == 1`` uses the limiting form."""
    ntu = np.asarray(ntu, dtype=float)
    cr = np.asarray(cr, dtype=float)
    out = np.empty_like(ntu)
    cr1 = np.isclose(cr, 1.0)
    exp_term = np.exp(-ntu[~cr1] * (1.0 - cr[~cr1]))
    out[~cr1] = (1.0 - exp_term) / (1.0 - cr[~cr1] * exp_term)
    out[cr1] = ntu[cr1] / (1.0 + ntu[cr1])
    return out


def exchanger_model(X: np.ndarray, theta: np.ndarray, *, ua_clean: float) -> dict[str, np.ndarray]:
    """``model(X, theta)`` for the twin: outlet temperatures from the operating point.

    ``X`` columns, in the order the spec declares: ``position, T_hi, T_ci, m_h, m_c``.
    ``theta[0]`` is the fouling resistance in m2 K/W.
    """
    X = np.asarray(X, dtype=float)
    t_hi, t_ci, m_h, m_c = X[:, 1], X[:, 2], X[:, 3], X[:, 4]
    rf = float(np.asarray(theta, dtype=float).ravel()[0])

    inv_ua = 1.0 / ua_clean + rf / AREA
    ua = 1.0 / inv_ua

    c_h = m_h * CP_WATER
    c_c = m_c * CP_WATER
    c_min = np.minimum(c_h, c_c)
    c_max = np.maximum(c_h, c_c)
    cr = c_min / c_max
    ntu = ua / c_min

    eps = _effectiveness(ntu, cr)
    q = eps * c_min * (t_hi - t_ci)
    return {"T_ho": t_hi - q / c_h, "T_co": t_ci + q / c_c}


@dataclass(frozen=True)
class OperatingPoint:
    """Where the exchanger is being run."""

    t_hi: float = 95.0
    t_ci: float = 30.0
    m_h: float = 1.20
    m_c: float = 1.50


@dataclass(frozen=True)
class PlantConfig:
    """The as-built exchanger, including the model error the twin has to live with."""

    ua_clean: float = 10_200.0     # W/K
    flow_sensitivity: float = 0.05  # UA drifts ±5 % with flow, model ignores it
    tau_fouling_s: float = 30 * 3600.0
    rf_final: float = 0.0018        # m2 K/W after one full fouling period


def plant_outlets(op: OperatingPoint, rf: float, cfg: PlantConfig) -> tuple[float, float]:
    """The real exchanger: the model plus a flow-dependent UA the model does not know."""
    x = np.array([[0.0, op.t_hi, op.t_ci, op.m_h, op.m_c]])
    # A real exchanger's UA rises slightly with both flows; the model uses the clean rating.
    gain = 1.0 + cfg.flow_sensitivity * ((op.m_h - 1.2) / 1.2 + (op.m_c - 1.5) / 1.5)
    out = exchanger_model(x, np.array([rf]), ua_clean=cfg.ua_clean * gain)
    return float(out["T_ho"][0]), float(out["T_co"][0])


def fouling_at(t_s: float, cfg: PlantConfig) -> float:
    """Asymptotic fouling build-up: fast at first, then a plateau."""
    return cfg.rf_final * (1.0 - float(np.exp(-max(0.0, t_s) / cfg.tau_fouling_s)))


def generate_history(
    hours: float = 72.0,
    sampling_s: float = 300.0,
    *,
    seed: int = 7,
    cfg: PlantConfig | None = None,
    op: OperatingPoint | None = None,
    t_start: float = 1_767_225_600.0,
) -> list[dict[str, float]]:
    """A plant history with fouling, flow wobble and instrument noise.

    Clearly synthetic — it exists so the twin can be exercised end to end without a plant,
    and every report it produces says so.
    """
    cfg = cfg or PlantConfig()
    op = op or OperatingPoint()
    rng = np.random.default_rng(seed)

    frames: list[dict[str, float]] = []
    n = int(round(hours * 3600.0 / sampling_s))
    for i in range(n):
        t_rel = i * sampling_s
        # Slow operating wander: the plant is not held at one point for two days.
        wobble = rng.normal(0.0, 1.0)
        m_h = op.m_h * (1.0 + 0.10 * np.sin(2 * np.pi * t_rel / (18 * 3600.0)) + 0.01 * wobble)
        m_c = op.m_c * (1.0 + 0.08 * np.cos(2 * np.pi * t_rel / (26 * 3600.0)) + 0.01 * wobble)
        t_hi = op.t_hi + 1.5 * np.sin(2 * np.pi * t_rel / (30 * 3600.0)) + rng.normal(0, 0.1)
        t_ci = op.t_ci + 1.0 * np.cos(2 * np.pi * t_rel / (22 * 3600.0)) + rng.normal(0, 0.1)

        rf = fouling_at(t_rel, cfg)
        t_ho, t_co = plant_outlets(OperatingPoint(t_hi, t_ci, m_h, m_c), rf, cfg)

        frames.append(
            {
                "t": t_start + t_rel,
                "TT-101": t_hi + rng.normal(0, 0.15),
                "TT-102": t_ci + rng.normal(0, 0.15),
                "FT-101": m_h + rng.normal(0, 0.006),
                "FT-102": m_c + rng.normal(0, 0.006),
                "TT-103": t_ho + rng.normal(0, 0.15),
                "TT-104": t_co + rng.normal(0, 0.15),
                "TT-105": t_ho + rng.normal(0, 0.15),   # redundant outlet element: the holdout
            }
        )
    return frames
