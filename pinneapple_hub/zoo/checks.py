"""Per-query trust checks of the zoo models, and the verdict they add up to.

``pp.zoo.validate`` judges a model as a whole (its metrics, its physics checks). This module judges one *query*:
is this input inside the validity range, are the outputs physical, does the ensemble agree as well as it did on
held-out data, do the model's physics checks hold *here*? Each check has the rule ID of ``pinneapple check``
(PC601 out-of-range query, PC602 sparse region, PC402/PC403 unphysical or non-finite output, PC302 model residual).

A check is a value computed from the query and a comparison with a limit (``check_spec``). The values come from a
``torch`` module (:func:`checks_module`), the same one :func:`pinneapple_hub.zoo.export.export_onnx` puts in the
ONNX graph, so a browser evaluating the graph and Python evaluating :func:`query_checks` agree to float precision.
:func:`verdict` turns the statuses into one decision: any failing check -> ``fail`` ("outside the validity range,
an extrapolation"), any warning -> ``warn`` ("use with caution"), else ``pass``. Design constraints (``kind ==
"design"``) are reported but never enter the verdict: they say whether a design is acceptable, not whether the model
can be trusted.

    import pinneapple as pp
    r = pp.zoo.query_checks("airfoil_cst_rans_mlp", {"T0": .17, "T1": .15, "T2": .13, "M0": .02, "M1": .07,
                                                     "M2": .06, "alpha_deg": 16})[0]
    r["verdict"]["status"]       # 'fail': alpha 16 is beyond the trained range
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ._models import ZooModel

VERDICTS = {
    "pass": {"title": "Trusted for this query", "text": "Inside the validity range; every live check passes."},
    "warn": {"title": "Use with caution", "text": "Inside the range, but a check flags this query."},
    "fail": {"title": "Outside the validity range",
             "text": "This is an extrapolation: the number is not backed by the model's evidence."},
}
RANK = {"pass": 0, "warn": 1, "fail": 2}
NOISE = 2e-3                                    # 0.2 % of the output range: the rover model's noise level


def _rng(name, col, lo, hi, *, rule="PC601", group=None, label=None):
    # The checks run in float32 (the torch module and its ONNX graph), so a bound itself, e.g. 0.08, arrives as
    # 0.0799999982; the slack covers that rounding (2 float32 epsilons) as well as ZooModel.in_range's 1e-9.
    tol = max(1e-9 * max(abs(lo), abs(hi), 1.0), 2 * float(np.finfo(np.float32).eps) * max(abs(lo), abs(hi)))
    return {"id": f"range:{name}", "rule": rule, "name": label or f"{name} inside {lo:g} … {hi:g}", "op": "in",
            "limit": [lo - tol, hi + tol], "bounds": [lo, hi], "severity": "fail", "kind": "trust",
            "group": group, "detail": f"{name} = {{value:.4g}} (range {lo:g} … {hi:g})"}


def _chk(id_, rule, name, op, limit, severity="warn", detail="", kind="trust", group=None):
    return {"id": id_, "rule": rule, "name": name, "op": op, "limit": limit, "severity": severity, "kind": kind,
            "group": group, "detail": detail}


# ====================================================================== airfoil
def _airfoil_thresholds(zm: ZooModel) -> dict[str, float]:
    X, _ = zm.reference("test", offline=True)
    p = zm._forward(X)
    return {"Cl_std_p99": float(np.percentile(p["Cl_std"], 99)),
            "Cd_std_rel_p99": float(np.percentile(p["Cd_std"] / p["Cd"], 99))}


def _airfoil(zm: ZooModel):
    from pinneapple_design.aero.aircraft import Requirements
    from pinneapple_design.aero.geometry import BOUNDS, PARAMS
    th = _airfoil_thresholds(zm)
    slope_thin = 2 * math.pi * math.pi / 180
    req = Requirements()
    groups = {"cst-range": {"name": "Shape inside the CST bounds the model was trained on", "rule": "PC601"}}
    spec = [_rng(p, i, *BOUNDS[i], group="cst-range") for i, p in enumerate(PARAMS)]
    spec.append(_rng("alpha_deg", 6, -2.0, 14.0, label="Angle of attack inside −2…14°"))
    spec += [
        _chk("geometry:thickness", "PC402", "Surfaces do not cross", ">", 0.0, "fail",
             "smallest interior thickness {value:.4g}"),
        _chk("geometry:wavy", "PC402", "No wavy surface (|y''| ≤ 40)", "<=", 40.0, "fail", "max |y''| = {value:.3g}"),
        _chk("finite", "PC403", "Outputs finite, drag positive", ">", 0.0, "fail", "Cd = {value:.5f}"),
        _chk("spread:Cl", "PC602", "Ensemble agrees on Cl as well as on held-out shapes", "<=", th["Cl_std_p99"],
             "warn", "σ(Cl) {value:.2g} (held-out p99 {limit:.2g})"),
        _chk("spread:Cd", "PC602", "Ensemble agrees on Cd as well as on held-out shapes", "<=", th["Cd_std_rel_p99"],
             "warn", "σ(Cd)/Cd {value:.2%} (held-out p99 {limit:.2%})"),
        _chk("slope", "PC302", "Lift slope within 15 % of thin-airfoil 2π/rad", "<=", 0.15, "warn",
             "dCl/dα = {slope:.4f}/deg ({value:.1%} from 0.1097/deg)"),
        _chk("monotonic", "PC302", "Cl rises with α below stall (−2…10°)", ">", 0.0, "warn",
             "smallest Cl step {value:.3g}"),
        _chk("design:thickness", "", f"Thickness t/c ≥ {req.min_thickness:.0%}", ">=", req.min_thickness, "warn",
             "t/c = {value:.1%} at x/c {thickness_at:.2f}", kind="design"),
        _chk("design:moment", "", f"|Cm| ≤ {req.max_abs_cm:g}", "<=", req.max_abs_cm, "warn", "|Cm| = {value:.4f}",
             kind="design"),
    ]
    aux = ["slope", "thickness_at", "Cl_std", "Cd", "Cm"]

    def values(q, predict):
        import torch
        geo = values.geo
        m = q.shape[0]
        mean, std, _ = predict(q)
        g = geo(q[:, 0:6].double())
        a9 = torch.linspace(-2.0, 6.0, 9)
        a25 = torch.linspace(-2.0, 10.0, 25)

        def sweep(a):
            shapes = q[:, 0:6].unsqueeze(1).expand(m, a.shape[0], 6)
            rows = torch.cat([shapes, a.view(1, -1, 1).expand(m, a.shape[0], 1)], 2).reshape(-1, 7)
            return predict(rows)[0][:, 0].reshape(m, a.shape[0])
        cl9 = sweep(a9)
        ac = a9 - a9.mean()
        slope = ((cl9 - cl9.mean(1, keepdim=True)) * ac).sum(1) / (ac ** 2).sum()
        cl25 = sweep(a25)
        mono = torch.amin(cl25[:, 1:] - cl25[:, :-1], dim=1)
        cd, cl, cm = mean[:, 1], mean[:, 0], mean[:, 2]
        cols = [q[:, i] for i in range(7)] + [
            g["validity"][:, 0].float(), g["validity"][:, 1].float(), cd + 0 * cl + 0 * cm,
            std[:, 0], std[:, 1] / cd, torch.abs(slope / slope_thin - 1), mono,
            g["props"][:, 0].float(), torch.abs(cm)]
        aux_ = [slope, g["props"][:, 1].float(), std[:, 0], cd, cm]
        return torch.stack(cols, 1), torch.stack(aux_, 1)

    def init():
        from pinneapple_design.aero.geometry_onnx import torch_geometry
        values.geo = torch_geometry()
    return spec, aux, groups, values, init


# ====================================================================== heat sink
def _heatsink(zm: ZooModel):
    """Forced convection (outputs R, ΔP, R upper bound) or natural convection (R, R upper bound; no air speed): the
    validity box, the derived limits of ``ZooModel.in_range`` (fin gap, source fractions, two fins), positive
    outputs and the monotonicity the model was trained or gated on, here."""
    forced = zm.adapter == "heatsink_forced"
    vr = zm.card.validity_range
    names = zm.input_names
    box = [k for k in names if k in vr and isinstance(vr[k], (list, tuple))]
    groups = {"box": {"name": "Inputs inside the validity range", "rule": "PC601"}}
    spec = [_rng(k, names.index(k), *vr[k], group="box") for k in box]
    spec += [
        _rng("fin_gap_mm", -1, *vr["fin_gap_mm"], label=f"Fin gap inside {vr['fin_gap_mm'][0]:g}…"
             f"{vr['fin_gap_mm'][1]:g} mm"),
        _rng("source_frac_w", -1, *vr["source_frac_w"], group="source", label="source width / base width"),
        _rng("source_frac_d", -1, *vr["source_frac_d"], group="source", label="source depth / base depth"),
        _chk("fins", "PC601", "At least two fins", ">=", 2.0, "fail", "{value:.0f} fins"),
    ]
    if forced:
        spec += [
            _chk("positive", "PC403", "R > 0 and ΔP > 0, finite", ">", 0.0, "fail", "min(R, ΔP) = {value:.4g}"),
            _chk("mono:V", "PC302", "More airflow lowers R here (+5 % speed)", "<", 0.0, "warn",
                 "R {R:.4g} → {R_v:.4g} K/W"),
            _chk("mono:dP", "PC302", "Pressure drop rises with air speed here", ">", 0.0, "warn",
                 "ΔP {dP:.4g} → {dP_v:.4g} Pa"),
        ]
    else:
        spec += [_chk("positive", "PC403", "R > 0, finite", ">", 0.0, "fail", "R = {value:.4g}")]
    spec += [_chk("mono:k", "PC302", "A more conductive material lowers R here (+5 % k)", "<", 0.0, "warn",
                  "R {R:.4g} → {R_k:.4g} K/W")]
    groups["source"] = {"name": "Heat source covers 15–100 % of the base", "rule": "PC601"}
    aux = ["R", "R_v", "R_k", "dP", "dP_v"] if forced else ["R", "R_k"]
    ik = names.index("k_w_mk")
    iV = names.index("air_velocity_m_s") if forced else None

    def values(q, predict):
        import torch
        W, D, N, t, sw, sd = (q[:, names.index(k)] for k in ("base_width_mm", "base_depth_mm", "n_fins",
                                                               "fin_thickness_mm", "source_width_mm",
                                                               "source_depth_mm"))
        gap = (W - N * t) / torch.clamp(N - 1, min=1.0)
        y0 = predict(q)[0]
        fk = torch.tensor([1.05 if j == ik else 1.0 for j in range(len(names))])     # +5 % conductivity
        yk = predict(q * fk)[0]
        cols = [q[:, names.index(k)] for k in box] + [gap, sw / W, sd / D, N]
        if forced:
            fv = torch.tensor([1.05 if j == iV else 1.0 for j in range(len(names))])  # +5 % air speed
            yv = predict(q * fv)[0]
            cols += [torch.minimum(y0[:, 0], y0[:, 1]), torch.log(yv[:, 0]) - torch.log(y0[:, 0]),
                     torch.log(yv[:, 1]) - torch.log(y0[:, 1])]
            aux_ = [y0[:, 0], yv[:, 0], yk[:, 0], y0[:, 1], yv[:, 1]]
        else:
            cols += [y0[:, 0]]
            aux_ = [y0[:, 0], yk[:, 0]]
        cols += [torch.log(yk[:, 0]) - torch.log(y0[:, 0])]
        return torch.stack(cols, 1), torch.stack(aux_, 1)
    return spec, aux, groups, values, None


# ====================================================================== rover
def _dfx_sign_table(zm: ZooModel):
    """Sign (+1 / 0 / -1, beyond the noise level) of dFx/ds of the Wong-Reece reference the model was trained on,
    over +0.01 slip, on a slip x sinkage grid (slip -0.4..0.3 step 0.01, sinkage 2..60 mm step 1 mm): what the zoo
    gate compares the model with when the training physics is recorded (``config["physics"]["soil"]``)."""
    from pinneapple_simulation.terramechanics import WongReeceModel
    c = zm.config["physics"]
    span = float(zm.config["y_hi"][0] - zm.config["y_lo"][0])
    s = np.round(np.arange(-0.4, 0.3 + 1e-9, 0.01), 4)
    z = np.round(np.arange(0.002, 0.060 + 1e-9, 0.001), 4)
    S, Z = np.meshgrid(s, z, indexing="ij")
    m = WongReeceModel.from_physics_config(c)
    f0 = m.forces(S.ravel(), Z.ravel()).Fx.reshape(S.shape)
    f1 = m.forces(S.ravel() + 0.01, Z.ravel()).Fx.reshape(S.shape)
    d = (f1 - f0) / span
    return {"s0": float(s[0]), "ds": 0.01, "ns": len(s), "z0": float(z[0]), "dz": 0.001, "nz": len(z),
            "sign": np.where(d > NOISE, 1.0, np.where(d < -NOISE, -1.0, 0.0)).ravel()}


def _rover(zm: ZooModel):
    c = zm.config["physics"]
    lo, hi = np.asarray(zm.config["y_lo"]), np.asarray(zm.config["y_hi"])
    span = (hi - lo).tolist()
    groups = {"range": {"name": "Slip and sinkage inside the validity range", "rule": "PC601"}}
    spec = [_rng("slip", 0, -0.4, 0.9, group="range"), _rng("sinkage_m", 1, 0.002, 0.06, group="range")]
    table = _dfx_sign_table(zm) if "soil" in c else None
    spec += [
        # Fz near zero load: a negative value within the model's noise is not unphysical
        _chk("finite", "PC403", "Forces finite, soil load not negative (beyond 0.2 % of its range)", ">=", -NOISE,
             "fail", "Fz = {Fz:.1f} N"),
        _chk("mohr", "PC302", "Mohr–Coulomb traction limit Fx ≤ cA + Fz tan φ", "<=", NOISE, "warn",
             "Fx {Fx:.0f} N, limit {limit_N:.0f} N"),
        _chk("torque", "PC302", "Moment balance My ≥ R·Fx", "<=", NOISE, "warn", "My {My:.1f} N·m, R·Fx {RFx:.1f} N·m"),
        _chk("dFz", "PC302", "Deeper sinkage carries more load (dFz/dz ≥ 0)", ">=", -NOISE, "warn",
             "ΔFz over +1 mm: {value:.2%} of the range"),
        _chk("dFx", "PC302", "dFx/ds has the sign of the Wong–Reece reference (slip ≤ 0.3)", ">=", -NOISE, "warn",
             "ΔFx over +0.01 slip: {dFx:.2%} of the range, reference sign {ref_sign:.0f} (not tested above s = 0.3)")
        if table else
        _chk("dFx", "PC302", "More slip, more pull below s = 0.3 (dFx/ds ≥ 0)", ">=", -NOISE, "warn",
             "ΔFx over +0.01 slip: {value:.2%} of the range (not tested above s = 0.3)"),
    ]
    aux = ["Fx", "limit_N", "My", "RFx", "Fz", "dFx", "ref_sign"]

    def values(q, predict):
        import torch
        y = predict(q)[0]
        yz = predict(q + torch.tensor([0.0, 1e-3]))[0]
        ys = predict(q + torch.tensor([0.01, 0.0]))[0]
        Fx, Fz, My = y[:, 0], y[:, 1], y[:, 2]
        theta = torch.acos(torch.clamp(1 - q[:, 1] / c["R_m"], -1.0, 1.0))
        lim = c["c_Pa"] * c["b_m"] * c["R_m"] * theta + Fz * math.tan(math.radians(c["phi_deg"]))
        dfx = (ys[:, 0] - Fx) / span[0]
        if table:
            # nearest cell of the reference's sign table (step 0.01 slip, 1 mm sinkage)
            i = torch.clamp(torch.floor((q[:, 0] - table["s0"]) / table["ds"] + 0.5), 0, table["ns"] - 1)
            j = torch.clamp(torch.floor((q[:, 1] - table["z0"]) / table["dz"] + 0.5), 0, table["nz"] - 1)
            sgn = torch.index_select(values.sign, 0, (i * table["nz"] + j).long())
        else:
            sgn = torch.ones_like(dfx)
        above = torch.clamp(q[:, 0] - 0.3, min=0.0) * 1e3                  # not tested above s = 0.3
        cols = [q[:, 0], q[:, 1], Fz / span[1] + 0 * Fx + 0 * My, (Fx - lim) / span[0], (c["R_m"] * Fx - My) / span[2],
                (yz[:, 1] - Fz) / span[1], sgn * dfx + above]
        return torch.stack(cols, 1), torch.stack([Fx, lim, My, c["R_m"] * Fx, Fz, dfx, sgn], 1)

    def init():
        import torch
        values.sign = torch.tensor(table["sign"] if table else [1.0], dtype=torch.float32)
    return spec, aux, groups, values, init


_BUILDERS = {"airfoil_mlp": _airfoil, "heatsink_forced": _heatsink, "heatsink_natural": _heatsink,
             "rover_contact": _rover}


def has_query_checks(zm: ZooModel) -> bool:
    """Whether the model has per-query checks (:func:`query_checks`); otherwise only ``in_range`` applies."""
    return zm.adapter in _BUILDERS


def _build(zm: ZooModel):
    cached = getattr(zm, "_query_checks", None)
    if cached is None:
        spec, aux, groups, values, init = _BUILDERS[zm.adapter](zm)
        if init:
            init()
        cached = zm._query_checks = (spec, aux, groups, values)
    return cached


def check_spec(zm: ZooModel) -> dict[str, Any]:
    """The declarative description of a model's query checks: ``checks`` (one entry per column of the ``checks``
    output: id, rule, name, op, limit, severity, kind, group, detail template), ``aux`` (names of the ``aux``
    columns the templates use), ``groups`` (display name and rule of grouped checks) and ``verdicts``."""
    spec, aux, groups, _ = _build(zm)
    return {"checks": spec, "aux": aux, "groups": groups, "verdicts": VERDICTS,
            "templates": "Python str.format fields {value}, {limit}, and the aux names; formats .Ng .Nf .N%"}


def checks_module(zm: ZooModel, predictor=None):
    """``torch.nn.Module``: query rows (m, d) -> (values (m, C), aux (m, A)) of :func:`check_spec`."""
    import torch

    from .export import predictor_module
    pred = predictor if predictor is not None else predictor_module(zm)
    _, _, _, values = _build(zm)

    class QueryChecks(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pred = pred
            if hasattr(values, "geo"):
                self.geo = values.geo
            if hasattr(values, "sign"):
                self.register_buffer("sign", values.sign)

        def forward(self, q):
            return values(q, self.pred)
    return QueryChecks()


def check_values(zm: ZooModel, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(values, aux) of the query checks for rows ``X`` (physical inputs, card order)."""
    import torch
    with torch.no_grad():
        v, a = checks_module(zm)(torch.as_tensor(np.asarray(X, float), dtype=torch.float32))
    return v.double().numpy(), a.double().numpy()


def compare(value: float, op: str, limit) -> bool:
    """``value op limit``; NaN never passes."""
    if value is None or not np.isfinite(value):
        return False
    if op == "in":
        return limit[0] <= value <= limit[1]
    return {"<": value < limit, "<=": value <= limit, ">": value > limit, ">=": value >= limit}[op]


def _fmt(template: str, fields: Mapping[str, float]) -> str:
    try:
        return template.format(**fields)
    except (KeyError, ValueError, IndexError):
        return template


def evaluate(spec: Mapping[str, Any], values: Sequence[float], aux: Sequence[float]) -> dict[str, Any]:
    """Statuses, details and the verdict for one row of check values (the browser does the same with the graph's
    outputs)."""
    fields = dict(zip(spec["aux"], (float(a) for a in aux), strict=True))
    checks = []
    for c, v in zip(spec["checks"], values, strict=True):
        ok = compare(float(v), c["op"], c["limit"])
        lim = c["limit"][0] if c["op"] == "in" else c["limit"]
        checks.append({**c, "value": float(v), "status": "pass" if ok else c["severity"],
                       "detail_text": _fmt(c["detail"], {**fields, "value": float(v), "limit": lim})})
    return {"checks": checks, "verdict": verdict(checks)}


def verdict(checks: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    """The worst status of the trust checks (design constraints excluded) with its title and explanation."""
    worst = max((c["status"] for c in checks if c.get("kind", "trust") == "trust"), key=RANK.__getitem__,
                default="pass")
    return {"status": worst, **VERDICTS[worst]}


def query_checks(model: str | ZooModel, inputs, *, offline: bool | None = None) -> list[dict[str, Any]]:
    """The query checks and the verdict for each input row (a mapping of named columns or an (n, d) array).

    Returns one ``{"checks": [...], "verdict": {"status", "title", "text"}}`` per row; each check carries its
    ``status`` (pass / warn / fail), ``value``, ``limit``, rule ID and a filled-in ``detail_text``."""
    from . import load
    zm = load(model, offline=offline) if isinstance(model, str) else model
    X = zm.as_array(inputs)
    v, a = check_values(zm, X)
    spec = check_spec(zm)
    return [evaluate(spec, v[i], a[i]) for i in range(len(X))]
