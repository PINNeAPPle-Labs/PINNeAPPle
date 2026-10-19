"""Ready-made inverse problems with known answers, and models described as JSON (for apps and agents).

Every preset's measurements are *synthetic*: the forward model at the stated true parameters plus Gaussian noise
from a fixed seed, so any estimate can be checked against the truth (``Parameter.truth``). The heat cases are the
Inverse Heat Lab's (pin fin and heat-spreader plate), plus a fin with zone-wise convection (a field), an unknown
base temperature (a boundary condition), a cooling curve and a reaction-kinetics case.

``model_from_spec`` builds a model from JSON::

    {"type": "pin_fin", "options": {"d_mm": 5, "length_mm": 50}}
    {"type": "expression", "expressions": {"T": "T_air + (T0 - T_air) * exp(-k * t)"}, "inputs": ["t"],
     "parameters": [{"name": "k", "value": 0.01, "lower": 1e-5, "upper": 1}, ...]}
    {"type": "table", "rows": [{...}, ...], "parameters": ["h"], "inputs": ["x_mm"], "output": "T"}
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .forward import ExpressionModel, ForwardModel, Observations, Parameter, TabulatedModel
from .models import ArrheniusModel, LumpedCoolingModel, PinFinModel, PlateModel, RodModel
from .studio import InverseProblem

__all__ = ["MODEL_TYPES", "model_from_spec", "Preset", "PRESETS", "get_preset", "list_presets", "synthetic"]

MODEL_TYPES: dict[str, Callable[..., ForwardModel]] = {
    "pin_fin": PinFinModel, "rod_fv": RodModel, "plate_2d": PlateModel, "lumped_cooling": LumpedCoolingModel,
    "arrhenius": ArrheniusModel,
}


def model_from_spec(spec: Mapping[str, Any]) -> ForwardModel:
    """A forward model from a JSON-able dict (see module docstring)."""
    kind = spec.get("type")
    if kind in MODEL_TYPES:
        opts = dict(spec.get("options") or {})
        allowed = {"pin_fin": {"d_mm", "length_mm"}, "rod_fv": {"d_mm", "length_mm", "n_zones", "n_cells"},
                   "plate_2d": {"nx", "ny"}, "lumped_cooling": {"area_m2"}, "arrhenius": set()}[kind]
        bad = set(opts) - allowed
        if bad:
            raise ValueError(f"unknown options for {kind}: {sorted(bad)}; allowed: {sorted(allowed)}")
        if kind == "rod_fv":
            opts["n_zones"] = int(min(max(int(opts.get("n_zones", 1)), 1), 12))
            opts["n_cells"] = int(min(max(int(opts.get("n_cells", 200)), 20), 2000))
        if kind == "plate_2d":
            opts = {k: int(min(max(int(v), 10), 200)) for k, v in opts.items()}
        return MODEL_TYPES[kind](**opts)
    if kind == "expression":
        params = [Parameter.from_dict(p) for p in spec.get("parameters", [])]
        if not params:
            raise ValueError("an expression model needs its parameters")
        return ExpressionModel(spec["expressions"], inputs=list(spec.get("inputs", [])), parameters=params,
                               name=spec.get("name", "expression"))
    if kind == "table":
        import pandas as pd

        df = pd.DataFrame(spec["rows"])
        return TabulatedModel(df, parameters=spec["parameters"], inputs=spec["inputs"], output=spec["output"],
                              name=spec.get("name", "simulated results"))
    raise ValueError(f"unknown model type '{kind}'; use one of {sorted(MODEL_TYPES) + ['expression', 'table']}")


def synthetic(model: ForwardModel, params: list[Parameter], obs: Observations, seed: int,
              rel_sigma: bool = False) -> Observations:
    """``obs`` with values = model at the parameters' truth (or value) + N(0, sigma) noise from ``seed``, recorded
    at an instrument resolution of about sigma / 10 (rounded to that many decimals)."""
    theta = {p.name: (p.truth if p.truth is not None else p.value) for p in params}
    clean = model.predict(theta, obs.inputs, obs.observable)
    rng = np.random.default_rng(seed)
    noisy = clean + rng.normal(0.0, 1.0, obs.n) * obs.sigma
    decimals = np.clip(np.ceil(-np.log10(obs.sigma / 10)), 0, 12).astype(int)
    return obs.with_values(np.array([round(float(v), int(d)) for v, d in zip(noisy, decimals, strict=True)]))


@dataclass
class Preset:
    id: str
    title: str
    summary: str
    model_spec: dict[str, Any]
    parameters: list[dict[str, Any]]
    observations: dict[str, Any]
    validation: dict[str, Any] | None = None
    candidates: dict[str, Any] | None = None
    methods: list[str] = field(default_factory=lambda: ["least_squares", "bayesian"])
    tags: list[str] = field(default_factory=list)
    seed: int = 0
    smoothness: float = 0.0
    prior_std: float = 1.0
    field_groups: list[list[str]] = field(default_factory=list)
    lesson: str = ""

    def build(self) -> dict[str, Any]:
        model = model_from_spec(self.model_spec)
        params = [Parameter.from_dict(p) for p in self.parameters]
        obs = Observations.from_dict(self.observations)
        val = Observations.from_dict(self.validation) if self.validation else None
        cand = Observations.from_dict(self.candidates) if self.candidates else None
        return {"model": model, "parameters": params, "observations": obs, "validation": val, "candidates": cand}

    def problem(self) -> InverseProblem:
        b = self.build()
        return InverseProblem(b["model"], b["observations"], b["parameters"], b["validation"], name=self.title,
                              description=self.summary, smoothness=self.smoothness, field_groups=self.field_groups,
                              prior_std=self.prior_std)

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def _p(name, value, lower, upper, unit="", truth=None, fixed=False, log=None, description=""):
    return Parameter(name, value, lower, upper, unit, log=log, fixed=fixed, truth=truth,
                     description=description).to_dict()


def _obs(model, params, inputs, sigma, observable, seed):
    ps = [Parameter.from_dict(p) for p in params]
    o = Observations(inputs, sigma, observable=np.array(observable, dtype=object))
    return synthetic(model, ps, o, seed).to_dict()


def _fin_presets() -> list[Preset]:
    spec = {"type": "pin_fin", "options": {"d_mm": 5.0, "length_mm": 50.0}}
    model = model_from_spec(spec)
    base = [_p("h", 100.0, 1.0, 2000.0, "W/m²K", truth=25.0, description="convection coefficient"),
            _p("k", 16.0, 1.0, 500.0, "W/m·K", truth=16.0, fixed=True, description="stainless steel"),
            _p("T_base", 80.0, -50.0, 1000.0, "°C", truth=80.0, fixed=True, log=False),
            _p("T_air", 25.0, -50.0, 500.0, "°C", truth=25.0, fixed=True, log=False)]
    xs = [10.0, 20.0, 30.0, 40.0, 50.0]
    obs = _obs(model, base, {"x_mm": xs}, 0.5, ["T"] * 5, 0)
    val = _obs(model, base, {"x_mm": [5.0, 15.0, 25.0, 35.0, 45.0]}, 0.5, ["T"] * 5, 1)
    fin_h = Preset("fin_h", "Pin fin: h from five thermocouples",
                   "The Inverse Heat Lab's 1D case: a 5 mm stainless pin fin, 50 mm long, on an 80 °C wall in 25 °C "
                   "air, thermocouples every 10 mm (±0.5 °C). Unknown: the convection coefficient h (true 25 W/m²K).",
                   spec, base, obs, val, methods=["least_squares", "autodiff", "bayesian", "pinn"],
                   tags=["heat", "fin", "inverse heat lab"],
                   lesson="One parameter, five readings: every method should land within a few percent of 25.")
    hk = [dict(base[0]), dict(base[1], fixed=False, value=40.0), base[2], base[3]]
    obs_hk = _obs(model, hk, {"x_mm": xs}, 0.5, ["T"] * 5, 0)
    fin_hk = Preset("fin_h_k", "Pin fin: h and k from thermocouples only",
                    "Same fin, but the conductivity k is unknown too. The temperatures depend on h and k only through "
                    "h/k, so no amount of thermocouples can separate them.",
                    spec, hk, obs_hk, val, methods=["least_squares", "bayesian"], tags=["heat", "identifiability"],
                    lesson="Structurally non-identifiable: the design suggests measuring the base heat flow.")
    q_sigma = 0.02 * 0.579                                     # a 2 % heat-flux meter on a 0.58 W fin
    obs_hkq = _obs(model, hk, {"x_mm": xs + [0.0]}, [0.5] * 5 + [q_sigma], ["T"] * 5 + ["q_base"], 0)
    fin_hkq = Preset("fin_h_k_flux", "Pin fin: h and k with a heat-flux meter at the base",
                     "Thermocouples plus the heat flow through the base (±2 %): q grows with sqrt(h k) while the "
                     "temperatures fix h/k, so both become identifiable.",
                     spec, hk, obs_hkq, val, methods=["least_squares", "autodiff", "bayesian"],
                     tags=["heat", "identifiability", "design"],
                     lesson="The extra observable suggested by the D-optimal design resolves the ambiguity.")
    return [fin_h, fin_hk, fin_hkq]


def _rod_presets() -> list[Preset]:
    spec = {"type": "rod_fv", "options": {"d_mm": 6.0, "length_mm": 120.0, "n_zones": 4, "n_cells": 240}}
    model = model_from_spec(spec)
    truth = [90.0, 45.0, 20.0, 30.0]                          # a fan blowing on the root of the fin
    params = [_p(f"h{j + 1}", 40.0, 2.0, 1000.0, "W/m²K", truth=t, description=f"h, zone {j + 1}")
              for j, t in enumerate(truth)]
    params += [_p("k", 200.0, 1.0, 500.0, "W/m·K", truth=200.0, fixed=True, description="aluminium"),
               _p("T_base", 95.0, -50.0, 1000.0, "°C", truth=95.0, fixed=True, log=False),
               _p("T_air", 22.0, -50.0, 500.0, "°C", truth=22.0, fixed=True, log=False)]
    xs = list(np.linspace(8, 120, 12).round(2))
    obs = _obs(model, params, {"x_mm": xs}, 0.15, ["T"] * len(xs), 3)
    val = _obs(model, params, {"x_mm": [20.0, 50.0, 80.0, 110.0]}, 0.15, ["T"] * 4, 4)
    zones = Preset("rod_h_field", "Fin with a fan: the h(x) field in four zones",
                   "A 6 mm aluminium pin fin, 120 mm long, cooled harder near the root by a fan: h varies along it "
                   "(true 90 / 45 / 20 / 30 W/m²K by 30 mm zone). Twelve thermocouples (±0.15 °C) and a finite-volume "
                   "model with exact sensitivities: reconstruct h(x) and T(x).",
                   spec, params, obs, val, methods=["least_squares", "adjoint", "autodiff", "bayesian"],
                   tags=["heat", "field", "adjoint"], seed=3, smoothness=0.5,
                   field_groups=[["h1", "h2", "h3", "h4"]],
                   lesson="A field: the adjoint gradient costs one extra solve whatever the number of zones; the tip "
                          "zone is the least determined.")
    spec1 = {"type": "rod_fv", "options": {"d_mm": 6.0, "length_mm": 120.0, "n_zones": 1, "n_cells": 240}}
    m1 = model_from_spec(spec1)
    pb = [_p("h", 60.0, 2.0, 1000.0, "W/m²K", truth=35.0),
          _p("k", 200.0, 1.0, 500.0, "W/m·K", truth=200.0, fixed=True),
          _p("T_base", 60.0, -50.0, 1000.0, "°C", truth=110.0, log=False, description="unknown wall temperature"),
          _p("T_air", 22.0, -50.0, 500.0, "°C", truth=22.0, fixed=True, log=False)]
    xb = [40.0, 60.0, 80.0, 100.0, 120.0]
    obs_b = _obs(m1, pb, {"x_mm": xb}, 0.3, ["T"] * 5, 5)
    val_b = _obs(m1, pb, {"x_mm": [50.0, 70.0, 90.0, 110.0]}, 0.3, ["T"] * 4, 6)
    bc = Preset("rod_base_bc", "Unknown boundary condition: the wall temperature",
                "The wall the fin is mounted on cannot be reached: estimate its temperature (true 110 °C) and h (true "
                "35 W/m²K) from five thermocouples on the outer part of the fin.",
                spec1, pb, obs_b, val_b, methods=["least_squares", "adjoint", "autodiff", "bayesian"],
                tags=["heat", "boundary condition"], seed=5,
                lesson="A boundary condition is an unknown like any other: it extrapolates the profile to the wall.")
    return [zones, bc]


def _plate_preset() -> Preset:
    spec = {"type": "plate_2d", "options": {"nx": 60, "ny": 36}}
    model = model_from_spec(spec)
    params = [_p("h", 60.0, 2.0, 300.0, "W/m²K", truth=15.0),
              _p("power", 8.0, 0.5, 50.0, "W", truth=8.0, fixed=True),
              _p("T_air", 25.0, -50.0, 200.0, "°C", truth=25.0, fixed=True, log=False)]
    s = np.array([[10, 10], [50, 10], [90, 10], [70, 30], [90, 50], [50, 50], [10, 50], [30, 55]], float)
    obs = _obs(model, params, {"x_mm": s[:, 0], "y_mm": s[:, 1]}, 0.5, ["T"] * len(s), 100)
    v = np.array([[30, 30], [20, 40], [75, 15], [60, 40]], float)
    val = _obs(model, params, {"x_mm": v[:, 0], "y_mm": v[:, 1]}, 0.5, ["T"] * len(v), 101)
    return Preset("plate_h", "Heat-spreader plate: h and the hot spot",
                  "The Inverse Heat Lab's 2D case: an 8 W device on a 100 x 60 x 1 mm aluminium plate, eight "
                  "thermocouples (±0.5 °C). Unknown: h (true 15 W/m²K); the reconstructed field gives the hot spot "
                  "nobody measured.", spec, params, obs, val,
                  methods=["least_squares", "bayesian", "pinn"], tags=["heat", "2D", "inverse heat lab"], seed=100,
                  lesson="The field is the product: the hot spot under the device comes from the calibrated model.")


def _plate_table_preset() -> Preset:
    """The plate case with the model given as a table of simulated runs (the finite-volume solver at 8 values of h,
    read at the sensor positions): what a calibration against a solver campaign looks like."""
    plate = model_from_spec({"type": "plate_2d", "options": {"nx": 60, "ny": 36}})
    s = np.array([[10, 10], [50, 10], [90, 10], [70, 30], [90, 50], [50, 50], [10, 50], [30, 55],
                  [30, 30], [20, 40], [75, 15], [60, 40]], float)
    rows = []
    for h in (4.0, 6.0, 9.0, 13.0, 19.0, 28.0, 40.0, 60.0):
        T = plate.predict({"h": h, "power": 8.0, "T_air": 25.0}, {"x_mm": s[:, 0], "y_mm": s[:, 1]})
        rows += [{"h": h, "x_mm": float(x), "y_mm": float(y), "T": round(float(t), 4)} for (x, y), t in zip(s, T, strict=True)]
    spec = {"type": "table", "rows": rows, "parameters": ["h"], "inputs": ["x_mm", "y_mm"], "output": "T",
            "name": "plate FV runs"}
    params = [_p("h", 30.0, 4.0, 60.0, "W/m²K", truth=15.0, log=False)]
    truth_model = [_p("h", 30.0, 4.0, 60.0, "W/m²K", truth=15.0), _p("power", 8.0, 0.5, 50.0, "W", fixed=True),
                   _p("T_air", 25.0, -50.0, 200.0, "°C", fixed=True, log=False)]
    obs = _obs(plate, truth_model, {"x_mm": s[:8, 0], "y_mm": s[:8, 1]}, 0.5, ["T"] * 8, 100)
    val = _obs(plate, truth_model, {"x_mm": s[8:, 0], "y_mm": s[8:, 1]}, 0.5, ["T"] * 4, 101)
    return Preset("plate_h_table", "Plate: h from a table of simulated runs",
                  "The heat-spreader plate again, but the model is a campaign of 8 solver runs (h = 4 to 60 W/m²K) "
                  "read at the sensor positions, interpolated between runs. Measurements: eight thermocouples "
                  "(±0.5 °C); held out: four more. True h: 15 W/m²K.", spec, params, obs, val,
                  methods=["least_squares", "bayesian"], tags=["heat", "2D", "simulated results"], seed=100,
                  lesson="A table of solver runs is a model, and interpolating between runs adds an error the "
                         "confidence interval does not include: compare with the plate_h preset, which runs the "
                         "solver itself.")


def _cooling_presets() -> list[Preset]:
    spec = {"type": "lumped_cooling", "options": {"area_m2": 0.012}}
    model = model_from_spec(spec)
    params = [_p("h", 30.0, 0.5, 500.0, "W/m²K", truth=11.0),
              _p("C", 450.0, 1.0, 1e5, "J/K", truth=450.0, fixed=True, description="heat capacity m·c"),
              _p("T0", 80.0, -50.0, 1000.0, "°C", truth=94.0, log=False),
              _p("T_air", 30.0, -50.0, 500.0, "°C", truth=23.0, log=False)]
    t = list(np.linspace(0, 1800, 31))
    obs = _obs(model, params, {"t_s": t}, 0.3, ["T"] * len(t), 7)
    tv = list(np.linspace(1900, 3650, 8))
    val = _obs(model, params, {"t_s": tv}, 0.3, ["T"] * len(tv), 8)
    cool = Preset("cooling_h", "Cooling curve: h, start and air temperature",
                  "A 450 J/K aluminium part (0.012 m²) cools in still air; a thermocouple logs it every minute for "
                  "30 minutes (±0.3 °C). Unknown: h (true 11 W/m²K), the start and the air temperature. Validation: "
                  "the next 30 minutes, never seen (an extrapolation test).",
                  spec, params, obs, val, methods=["least_squares", "autodiff", "bayesian"],
                  tags=["heat", "transient", "sensor log"], seed=7,
                  lesson="Three parameters, one curve: all identifiable once the curve has bent.")
    pc = [dict(params[0]), dict(params[1], fixed=False, value=900.0), params[2], params[3]]
    obs_c = _obs(model, pc, {"t_s": t}, 0.3, ["T"] * len(t), 7)
    cool_c = Preset("cooling_h_C", "Cooling curve: h and heat capacity together",
                    "Same curve with the heat capacity unknown too: the curve depends only on hA/C, so h and C trade "
                    "off exactly.", spec, pc, obs_c, val, methods=["least_squares", "bayesian"],
                    tags=["heat", "identifiability"], seed=7,
                    lesson="Structural ambiguity: only h/C is determined; weigh the part (fix C) or measure a heat "
                           "flux.")
    return [cool, cool_c]


def _arrhenius_presets() -> list[Preset]:
    spec = {"type": "arrhenius"}
    model = model_from_spec(spec)
    params = [_p("A", 1e5, 1.0, 1e14, "1/s", truth=2.0e7, description="pre-exponential factor"),
              _p("Ea", 50e3, 1e3, 3e5, "J/mol", truth=75e3, description="activation energy")]
    t = [120.0, 300.0, 600.0, 1200.0, 2400.0]
    one = _obs(model, params, {"t_s": t, "T_K": [360.0] * 5}, 0.01, ["X"] * 5, 9)
    two = _obs(model, params, {"t_s": t * 2, "T_K": [360.0] * 5 + [390.0] * 5}, 0.01, ["X"] * 10, 9)
    val = _obs(model, params, {"t_s": [600.0, 600.0], "T_K": [375.0, 345.0]}, 0.01, ["X"] * 2, 10)
    return [
        Preset("arrhenius_one_T", "Reaction kinetics at one temperature",
               "First-order conversion measured at 360 K only (±0.01). Unknown: A and Ea of the Arrhenius rate "
               "(true 2e7 1/s and 75 kJ/mol).", spec, params, one, val, methods=["least_squares", "bayesian"],
               tags=["kinetics", "identifiability"], seed=9, prior_std=5.0,
               lesson="At one temperature only the rate constant is seen: A and Ea are not separable."),
        Preset("arrhenius_two_T", "Reaction kinetics at two temperatures",
               "The same reaction measured at 360 K and 390 K: both Arrhenius parameters become identifiable "
               "(strongly correlated, as always for Arrhenius fits).", spec, params, two, val,
               methods=["least_squares", "bayesian"], tags=["kinetics", "calibration"], seed=9, prior_std=5.0,
               lesson="A second temperature is the measurement the design asks for."),
    ]


def _build() -> dict[str, Preset]:
    out: dict[str, Preset] = {}
    for p in (_fin_presets() + _rod_presets() + [_plate_preset(), _plate_table_preset()] + _cooling_presets()
              + _arrhenius_presets()):
        out[p.id] = p
    return out


_CACHE: dict[str, Preset] = {}


def _all() -> dict[str, Preset]:
    if not _CACHE:
        _CACHE.update(_build())
    return _CACHE


PRESETS = ("fin_h", "fin_h_k", "fin_h_k_flux", "rod_h_field", "rod_base_bc", "plate_h", "plate_h_table", "cooling_h",
           "cooling_h_C", "arrhenius_one_T", "arrhenius_two_T")


def get_preset(name: str) -> Preset:
    try:
        return _all()[name]
    except KeyError:
        raise KeyError(f"unknown preset '{name}'; available: {list(PRESETS)}") from None


def list_presets(tag: str | None = None) -> list[dict[str, Any]]:
    return [{"id": p.id, "title": p.title, "summary": p.summary, "tags": p.tags, "methods": p.methods,
             "lesson": p.lesson} for p in _all().values() if tag is None or tag in p.tags]
