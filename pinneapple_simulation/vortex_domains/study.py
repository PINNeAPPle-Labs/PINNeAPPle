"""Running a case and deciding what the run is worth: coefficients, Strouhal, physics checks, saved results.

A study is the unit the rest of PINNeAPPle consumes -- the same shape as the library's other studies: run the
physics, keep the time series, reduce it to the numbers a decision needs, and say which checks failed. The checks
here are the ones this solver can be held to:

=============================  ====================================================================================
``impermeability``             the normal velocity left at the wall (the source sheet's residual), against ``U_inf``
``no_slip``                    the tangential velocity left at the wall, against ``U_inf``
``displacement``               the largest domain step, against that domain's own core radius (the method's
                               stability criterion, ``dt <= 2 Re eps^2``)
``circulation``                the total circulation created at the walls stays bounded (nothing is destroyed)
``resolution``                 the core radius never grew beyond ``2x`` its first value because of ``max_domains``
``drag`` / ``strouhal``        against the reference the case carries, when it carries one
=============================  ====================================================================================
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .cases import Case
from .solver import VVD2D

__all__ = ["StudyResult", "run_study", "strouhal_number", "window_mean"]


@dataclass
class StudyResult:
    """What a run produced, and what can be said about it."""

    case: Case
    simulation: VVD2D
    series: dict[str, np.ndarray]
    coefficients: dict[str, float]
    checks: list[dict[str, Any]] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c["ok"] for c in self.checks)

    def failed(self) -> list[str]:
        return [c["name"] for c in self.checks if not c["ok"]]

    def summary(self) -> str:
        c = self.coefficients
        head = (f"{self.case.name}: Cl = {c.get('cl', float('nan')):+.3f}, Cd = {c.get('cd', float('nan')):+.3f}, "
                f"Cm = {c.get('cm', float('nan')):+.3f}")
        if "strouhal" in c:
            head += f", St = {c['strouhal']:.3f}"
        bad = self.failed()
        tail = "all checks passed" if not bad else "failed: " + ", ".join(bad)
        return f"{head} | {tail}"

    def to_json(self) -> dict[str, Any]:
        return {"case": self.case.name, "note": self.case.note,
                "config": {k: getattr(self.case.config, k) for k in self.case.config.__dataclass_fields__},
                "coefficients": self.coefficients, "checks": self.checks,
                "metadata": self.case.metadata, "reference": self.case.reference}

    def save(self, out_dir: str | Path) -> Path:
        """Write ``study.json``, the time series and the last field to ``out_dir``."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "study.json").write_text(json.dumps(self.to_json(), indent=2, default=str))
        np.savez(out / "series.npz", **self.series)
        snap = self.simulation.snapshot()
        np.savez(out / "final_state.npz", **{k: v for k, v in snap.items() if k != "panels"},
                 **{f"panel_{k}": v for k, v in snap["panels"].items()})
        return out


def window_mean(series: dict[str, np.ndarray], key: str, t_from: float, t_to: float) -> float:
    """Mean of ``series[key]`` over ``[t_from, t_to]``."""
    t = series["t"]
    mask = (t >= t_from) & (t <= t_to)
    if not np.any(mask):
        raise ValueError(f"no samples between t = {t_from} and {t_to}")
    return float(np.mean(series[key][mask]))


def strouhal_number(series: dict[str, np.ndarray], t_from: float | None = None,
                    amplitude_threshold: float = 1e-3) -> float | None:
    """Strouhal number ``f D / U`` of the lift signal, or ``None`` when the signal is not periodic.

    The frequency is the dominant one of a Hann-windowed periodogram above ``10`` per unit time; the amplitude
    threshold is what separates a shedding wake from a steady one.
    """
    t = series["t"]
    fy = series.get("fy")
    if fy is None or len(t) < 32:
        return None
    mask = np.ones_like(t, dtype=bool) if t_from is None else (t >= t_from)
    if mask.sum() < 32:
        return None
    tt, y = t[mask], fy[mask]
    if not np.all(np.isfinite(y)):
        return None
    span = tt[-1] - tt[0]
    if span <= 0:
        return None
    amplitude = 0.5 * (np.max(y) - np.min(y))
    if amplitude < amplitude_threshold * max(1.0, float(np.abs(y).mean())):
        return None
    dt = float(np.median(np.diff(tt)))
    if dt <= 0:
        return None
    n = int(2 ** np.floor(np.log2(len(y))))
    y = (y - y.mean())[-n:] * np.hanning(n)
    spectrum = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(n, d=dt)
    band = freqs > 10.0 * dt / max(span, dt) * 0.0 + 1e-9        # keep the DC out
    if not np.any(band):
        return None
    k = int(np.argmax(np.where(band, spectrum, 0.0)))
    if freqs[k] * span < 2.0:
        return None                    # fewer than two cycles in the window is a transient, not shedding
    return float(freqs[k])


def run_study(case: Case, t_end: float | None = None, progress: Any = None,
              average_from: float | None = None) -> StudyResult:
    """Run ``case`` and reduce it: mean coefficients over the last third of the run, Strouhal, the checks."""
    sim = case.build()
    sim.run(t_end=t_end, progress=progress)
    series = sim.series()
    t_from = average_from if average_from is not None else 0.66 * series["t"][-1]
    t_from = min(max(t_from, series["t"][0]), series["t"][-1])

    coefficients: dict[str, float] = {}
    for key, value in (("cl", series["fy"]), ("cd", series["fx"] + series["fx_friction"]),
                       ("cm", series["torque"])):
        mask = series["t"] >= t_from
        coefficients[key] = float(np.mean(value[mask])) if np.any(mask) else float("nan")
        coefficients[f"{key}_std"] = float(np.std(value[mask])) if np.any(mask) else float("nan")
    dynamic = 0.5 * case.config.rho * case.config.velocity_scale ** 2 * case.config.reference_length
    for key in ("cl", "cd", "cm"):
        coefficients[key] = coefficients[key] / dynamic / (case.config.reference_length if key == "cm" else 1.0)
        coefficients[f"{key}_std"] = coefficients[f"{key}_std"] / dynamic / (
            case.config.reference_length if key == "cm" else 1.0)
    st = strouhal_number(series, t_from=0.2 * series["t"][-1])
    if st is not None:
        coefficients["strouhal"] = st * case.config.reference_length / max(case.config.speed, 1e-30)

    result = StudyResult(case=case, simulation=sim, series=series, coefficients=coefficients)
    result.checks = _checks(result)
    return result


def _checks(result: StudyResult) -> list[dict[str, Any]]:
    s = result.series
    cfg = result.case.config
    speed = max(cfg.speed, 1e-30)
    checks: list[dict[str, Any]] = []

    def add(name: str, ok: bool, detail: str) -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    if "impermeability" in s and s["impermeability"].size:
        worst = float(np.max(s["impermeability"]))
        add("impermeability", worst <= 1e-6 * speed,
            f"largest normal velocity left on the wall: {worst:.3e} (U_inf = {speed:g})")
    if "slip" in s and s["slip"].size:
        worst = float(np.max(s["slip"]))
        first = float(s["slip"][0])
        add("no_slip", worst <= 0.75 * first,
            f"wall slip {first:.3f} -> {worst:.3f} (the wall took most of it away)")
    if "max_displacement" in s and "eps_floor" in s:
        ratio = float(np.max(s["max_displacement"] / np.maximum(s["eps_floor"], 1e-30)))
        add("displacement", ratio <= 4.0,
            f"largest domain step / core radius = {ratio:.2f} (the method's stability rule asks for order one)")
    if "circulation" in s:
        values = s["circulation"][np.isfinite(s["circulation"])]
        bounded = bool(values.size and np.max(np.abs(values)) < 1e3 * max(1.0, abs(float(values[0])) + 1.0))
        add("circulation", bounded, f"total circulation stayed between {values.min():.3f} and {values.max():.3f}")
    if "eps_floor" in s and s["eps_floor"].size:
        growth = float(s["eps_floor"][-1] / max(s["eps_floor"][0], 1e-30))
        add("resolution", growth <= 2.0,
            f"core radius floor grew {growth:.2f}x (more than 2x means max_domains was hit and the layer coarsened)")
    reference = result.case.reference
    if "drag_coefficient" in reference and "cd" in result.coefficients:
        target = float(reference["drag_coefficient"])
        cd = result.coefficients["cd"]
        rel = abs(cd - target) / abs(target)
        add("drag", rel <= 0.25,
            f"Cd = {cd:.3f} against {target:.4f} ({100 * rel:+.1f} %, {reference.get('drag_source', '')})")
    if "strouhal_law" in reference and "strouhal" in result.coefficients:
        law = reference["strouhal_law"]
        target = float(0.2731 - 1.1129 / np.sqrt(cfg.re) + 0.4821 / cfg.re) if cfg.re > 0 else float("nan")
        st = result.coefficients["strouhal"]
        rel = abs(st - target) / target
        add("strouhal", rel <= 0.20, f"St = {st:.3f} against {target:.4f} ({law})")
    return checks
