"""The offline planner: keywords (English and Portuguese), quantities with units and explicit coefficient values
detect the physics (PDE family, dimension, steady or transient, boundary conditions) and what is asked (solve,
compare, identify a coefficient, a fast surrogate, a twin, a forecast, a take-off); each proposal is a flow built
from the node catalog. Deterministic and offline: the same text always gives the same plan."""
from __future__ import annotations

import math
import re
from typing import Any

from ..rules import RULES
from ..templates import FAMILIES, template
from .build import FlowBuilder

# (family, weight) per keyword; matched at word starts, case-insensitive
FAMILY_WORDS: dict[str, dict[str, float]] = {
    "heat": {"heat": 2, "thermal": 1.5, "temperature": 1.5, "conduction": 2, "diffusion": 1.2, "cool": 1,
             "heating": 1, "warm": 0.8, "furnace": 1, "insulat": 0.8, "calor": 2, "térmic": 1.5, "termic": 1.5, "temperatura": 1.5, "difus": 1.2,
             "condução": 2, "resfria": 1, "aquec": 1},
    "wave": {"wave": 2, "vibrat": 2, "acoustic": 1, "string": 1.5, "membrane": 1.5, "drum": 1.5, "seismic": 1.5,
             "onda": 2, "vibra": 2, "corda": 1.5, "membrana": 1.5, "acústic": 1},
    "poisson": {"poisson": 3, "electrostatic": 2, "charge": 1.5, "source term": 1.5, "heat generation": 1.5,
                "torsion": 1.5, "eletrostát": 2, "carga": 1, "fonte": 1},
    "laplace": {"laplace": 3, "potential flow": 2, "harmonic": 1.5, "steady temperature": 1.5, "potencial": 1.5},
    "helmholtz": {"helmholtz": 3, "time-harmonic": 2, "frequency domain": 2, "eigen": 1, "resonan": 1.2,
                  "frequência": 1},
    "burgers": {"burgers": 3, "shock": 2, "traffic": 1.5, "nonlinear advection": 2, "choque": 2, "tráfego": 1.5},
    "advection_diffusion": {"advection": 2.5, "convection": 2, "transport": 1.5, "pollutant": 2, "contaminant": 2,
                            "dispersion": 1.5, "advecç": 2.5, "convecç": 2, "transporte": 1.5, "poluent": 2},
    "reaction_diffusion": {"reaction": 2, "fisher": 3, "population": 1.5, "tumor": 1.5, "invasion": 1.5,
                           "epidemic": 1.5, "logistic": 1.5, "reação": 2, "população": 1.5, "epidemia": 1.5},
}
STEADY = ("steady", "equilibrium", "stationary", "long-time", "permanente", "estacionári", "regime permanente")
TRANSIENT = ("transient", "over time", "evolution", "evolves", "time-dependent", "transiente", "ao longo do tempo",
             "evolução", "cools down", "after", "seconds", "minutes")
DIM_WORDS = {1: ("1d", "1-d", "one-dimensional", "rod", "bar", "wire", "string", "pipe", "barra", "fio", "corda",
                 "unidimensional", "cable", "cabo"),
             2: ("2d", "2-d", "two-dimensional", "plate", "square", "membrane", "sheet", "slab", "placa", "chapa",
                 "quadrado", "bidimensional", "rectangle", "retângulo", "cross-section"),
             3: ("3d", "3-d", "three-dimensional", "cube", "block", "volume", "room", "box", "solid", "cubo",
                 "bloco", "sala", "tridimensional", "sólido", "chip")}
INTENTS = {
    "inverse": ("identify", "estimate", "unknown", "calibrat", "infer", "discover", "recover", "inverse",
                "from measurements", "from data", "identificar", "estimar", "desconhecid", "calibr", "descobrir",
                "inverso", "a partir de medições", "parameter discovery"),
    "surrogate": ("surrogate", "operator", "deeponet", "fno", "fourier neural", "many queries", "parametric",
                  "real-time prediction", "fast prediction", "instant", "design space", "sweep", "emulat",
                  "reduced order", "rom", "substitut", "operador", "paramétric", "rápid"),
    "hybrid": ("hybrid", "mixed", "rom+nn", "pod", "reduced order", "reduced-order", "physics+data",
               "híbrid", "mist"),
    "hpo": ("hyperparameter", "hpo", "tune", "tuning", "autotune", "optuna", "search the architecture",
            "hiperparâmetro", "ajuste fino de hiper", "busca de hiper"),
    "uq": ("uncertainty", "sensitivity", "sobol", "robust", "incerteza", "sensibilidade"),
    "twin": ("digital twin", "twin", "live data", "stream", "sensor feed", "mqtt", "monitor", "leak",
             "gêmeo digital", "gemeo digital", "tempo real", "vazamento", "monitorar"),
    "forecast": ("forecast", "time series", "predict the next", "enso", "el niño", "el nino", "sea surface",
                 "previsão", "série temporal", "serie temporal", "prever"),
    "flight": ("aircraft", "airplane", "take-off", "takeoff", "flight", "runway", "avião", "aviao", "decolagem",
               "voo"),
    "pinn": ("pinn", "physics-informed", "physics informed", "rede neural informada"),
    "fem": ("finite element", "fem", "elementos finitos"),
    "fd": ("finite difference", "fdm", "diferenças finitas"),
    "viz3d": ("3d view", "visuali", "render", "animation", "blender", "three.js", "animação"),
    "finetune": ("fine-tun", "finetun", "transfer", "ajuste fino", "new data"),
}
COEFF_NAMES = {"alpha": "alpha", "diffusivity": None, "nu": "nu", "viscosity": "nu", "c": "c", "speed": "c",
               "k": "k", "wavenumber": "k", "d": "D", "r": "r", "growth": "r"}
_ASSIGN = re.compile(r"\b(alpha|nu|kappa|c|k|D|r)\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")


def _hit(text: str, kw: str) -> bool:
    return re.search(r"(?<![a-z0-9])" + re.escape(kw), text) is not None


def detect(text: str) -> dict[str, Any]:
    """Everything the heuristics read from the text (shown to the user as 'what the planner understood')."""
    t = text.lower()
    scores = {f: sum(w for k, w in words.items() if _hit(t, k)) for f, words in FAMILY_WORDS.items()}
    family = max(scores, key=lambda f: scores[f]) if max(scores.values()) > 0 else None
    steady = any(_hit(t, k) for k in STEADY)
    transient = any(_hit(t, k) for k in TRANSIENT)
    if family == "heat" and steady and not transient:
        family = "poisson" if any(_hit(t, k) for k in ("source", "generation", "fonte", "geração", "power")) \
            else "laplace"
    dims = {d: sum(_hit(t, k) for k in words) for d, words in DIM_WORDS.items()}
    dim_change = re.search(r"(\d)\s*-?\s*d\s*(?:to|->|→|para|e depois)\s*(\d)\s*-?\s*d", t)
    dimension = None
    if dim_change:
        dimension = int(dim_change.group(1))
    elif max(dims.values()) > 0:
        dimension = max(dims, key=lambda d: (dims[d], -d))
    intents = sorted(k for k, words in INTENTS.items() if any(_hit(t, w) for w in words))
    coeffs: dict[str, float] = {}
    for name, val in _ASSIGN.findall(text):
        coeffs[name] = float(val)
    quantities = []
    length = duration = None
    try:
        from pinneapple_design.intent.parse import find_quantities
        mentions = find_quantities(text)
    except Exception:  # noqa: BLE001 - the unit parser is optional for planning
        mentions = []
    for m in mentions:
        u = (m.unit or "").replace(" ", "")
        si = _si(m.value, u)
        if si is None:
            continue
        kind, val = si
        quantities.append({"quote": m.quote, "kind": kind, "si": val})
        if kind == "length" and length is None:
            length = val
        elif kind == "diffusivity":
            key = {"burgers": "nu", "advection_diffusion": "D", "reaction_diffusion": "D"}.get(family or "", "alpha")
            coeffs.setdefault(key, val)
        elif kind == "speed":
            coeffs.setdefault("c", val)
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*(s|sec|seconds|segundos|min|minutes|minutos|h|hours|horas)\b", t):
        v = float(m.group(1)) * {"s": 1, "sec": 1, "seconds": 1, "segundos": 1, "min": 60, "minutes": 60,
                                 "minutos": 60, "h": 3600, "hours": 3600, "horas": 3600}[m.group(2)]
        duration = duration or v
        quantities.append({"quote": m.group(0), "kind": "duration", "si": v})
    bcs = []
    if any(_hit(t, k) for k in ("insulated", "adiabatic", "isolad", "no flux", "zero flux", "sem fluxo")):
        bcs.append("neumann")
    if any(_hit(t, k) for k in ("fixed", "held at", "clamped", "dirichlet", "prescribed", "fixa", "mantid")):
        bcs.append("dirichlet")
    if _hit(t, "periodic") or _hit(t, "periódic"):
        bcs.append("periodic")
    return {"family": family, "family_scores": {k: v for k, v in scores.items() if v}, "dimension": dimension,
            "dimension_change": [int(dim_change.group(1)), int(dim_change.group(2))] if dim_change else None,
            "steady": steady and not transient, "intents": intents, "coefficients": coeffs,
            "length": length, "duration": duration, "boundary": bcs, "quantities": quantities}


def _si(value: float, unit: str):
    lengths = {"m": 1.0, "cm": 1e-2, "mm": 1e-3, "km": 1e3}
    if unit in lengths:
        return "length", value * lengths[unit]
    if unit in ("m^2/s", "m2/s", "m²/s"):
        return "diffusivity", value
    if unit in ("mm^2/s", "mm2/s"):
        return "diffusivity", value * 1e-6
    if unit in ("m/s",):
        return "speed", value
    return None


def _equation_params(det: dict[str, Any], family: str, dim: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Template equations with the detected coefficients, boundary type and length applied. The exact solution is
    kept only while it still solves the edited problem."""
    t = template(family, dim)
    coeffs = dict(t["coefficients"])
    for k, v in det["coefficients"].items():
        if k in coeffs:
            coeffs[k] = v
    pdes, conds, exact = list(t["pdes"]), [dict(c) for c in t["conditions"]], t["exact"] or ""
    if "neumann" in det["boundary"] and "dirichlet" not in det["boundary"] and \
            family in ("heat", "wave", "advection_diffusion"):
        conds = [c for c in conds if c["kind"] == "initial"] + [
            {"kind": "dirichlet", "on": "x_min", "equation": "u = 0"},
            {"kind": "neumann", "on": "x_max", "equation": "u_x = 0"}] + [
            {"kind": "neumann", "on": f"{c}_{s}", "equation": f"u_{c} = 0"} for c in "yz"[:dim - 1]
            for s in ("min", "max")]
        exact = ""
    L = det.get("length")
    if L and family_length_ok(family) and abs(L - 1.0) > 1e-12:
        def scale(e: str) -> str:
            return re.sub(r"pi\*([xyz])", lambda m: f"pi*{m.group(1)}/{L:g}", e)
        pdes = [scale(p) for p in pdes]
        conds = [{**c, "equation": scale(c["equation"])} for c in conds]
        exact = ""
    return {"template": family, "follow_template": False, "fields": t["fields"], "pdes": pdes,
            "conditions": conds, "coefficients": coeffs, "unknowns": [], "exact": exact}, t


def _with_amplitude(eqp: dict[str, Any]) -> tuple[dict[str, Any], str]:
    """Give a coefficient-free problem a parameter to sweep: an amplitude ``a`` on the source or the first
    non-zero boundary value."""
    eqp = {**eqp, "coefficients": {**eqp["coefficients"], "a": 1.0}, "exact": ""}
    lhs, _, rhs = eqp["pdes"][0].partition("=")
    if rhs.strip() not in ("", "0"):
        eqp["pdes"] = [f"{lhs.strip()} = a*({rhs.strip()})"] + eqp["pdes"][1:]
        return eqp, "a"
    conds = [dict(c) for c in eqp["conditions"]]
    for c in conds:
        l2, _, r2 = c["equation"].partition("=")
        if r2.strip() not in ("", "0"):
            c["equation"] = f"{l2.strip()} = a*({r2.strip()})"
            break
    eqp["conditions"] = conds
    return eqp, "a"


def _domain_params(det, t, dim) -> dict[str, Any]:
    p = {"dimension": dim, "shape": "box", "time_dependent": bool(t["time"])}
    rng = t["bounds"]["x"]
    if det["length"] and family_length_ok(t["family"]):
        rng = [0.0, float(det["length"])]
    p["x_range"] = list(rng)
    p["y_range"] = list(t["bounds"].get("y", rng))
    p["z_range"] = list(t["bounds"].get("z", rng))
    p["t_range"] = list(t["time"] or [0.0, 1.0])
    if det["duration"] and t["time"]:
        p["t_range"] = [0.0, float(det["duration"])]
    return p


def family_length_ok(family: str) -> bool:
    return family not in ("burgers",)


def _n_for(family, dim):
    if family == "burgers":
        return 513
    return {1: 201, 2: 65, 3: 25}[dim]


def plan_heuristic(text: str, *, max_proposals: int = 3) -> dict[str, Any]:
    det = detect(text)
    intents = set(det["intents"])
    props: list[dict[str, Any]] = []
    warnings: list[str] = []
    explicit_domain = det["dimension"] is not None or any(_hit(text.lower(), k) for k in ("pde", "equation",
                                                                                          "equação"))
    if "twin" in intents and det["family"] and explicit_domain:
        props.append(_pde_twin(det, text))
    elif "twin" in intents:
        props.append(_twin(det, text))
    if "forecast" in intents:
        props.append(_forecast(det))
    if "flight" in intents:
        props.append(_flight(det))
        from ..registry import REGISTRY
        ok, why = REGISTRY["flight.takeoff"].available()
        if not ok:
            warnings.append(why)
    family = det["family"]
    # a twin / forecast / flight request mentions temperatures and speeds too: plan a PDE only when the text also
    # names a domain (a rod, a plate, 2D, ...) or an equation
    explicit_pde = det["dimension"] is not None or any(_hit(text.lower(), k) for k in ("pde", "equation", "equação"))
    pde_wanted = (family is not None and (not props or explicit_pde)) or not props
    if pde_wanted:
        if family is None:
            family = "heat"
            warnings.append("no physics recognised in the text: proposing the heat equation as a starting point; "
                            "edit the equations node")
        dim = det["dimension"] or (1 if family == "burgers" else 2)
        if family == "burgers" and dim != 1:
            warnings.append("Burgers is planned in 1-D (the template's dimension)")
            dim = 1
        target = det["dimension_change"][1] if det["dimension_change"] else None
        if "inverse" in intents:
            props.append(_inverse(det, family, dim))
        if intents & {"surrogate", "hybrid", "hpo", "uq", "finetune"}:
            props.append(_surrogate(det, family, dim))
        props.append(_solve(det, family, dim, target))
        if len(props) < 2 and "inverse" not in intents and family in ("heat", "wave", "advection_diffusion",
                                                                       "reaction_diffusion", "burgers"):
            props.append(_surrogate(det, family, dim))
    return {"planner": "heuristic", "detected": det, "proposals": props[:max_proposals], "warnings": warnings}


# ----------------------------------------------------------------------------------------------- proposals
def _problem(fb: FlowBuilder, det, family, dim, *, prefix="", unknowns=None, label=""):
    eqp, t = _equation_params(det, family, dim)
    if unknowns:
        eqp = {**eqp, "unknowns": list(unknowns), "coefficients": {k: v for k, v in eqp["coefficients"].items()
                                                                  if k not in unknowns}, "exact": ""}
    geo = fb.add(f"{prefix}geo", "geometry.domain", _domain_params(det, t, dim), label=label or "Domain")
    mesh = fb.add(f"{prefix}mesh", "geometry.mesh", {"n": _n_for(family, dim), "n_t": 41}, label="Grid")
    fb.connect(geo, "geometry", mesh, "geometry")
    eq = fb.add(f"{prefix}eq", "equations.system", eqp, label=f"{family.replace('_', '-')} equations")
    fb.connect(geo, "geometry", eq, "geometry")
    fb.connect(mesh, "mesh", eq, "mesh")
    if not prefix or prefix == "true_":
        _decide_problem(fb, det, family, dim, geo, eq, eqp)
    return eq, t


def _decide_problem(fb: FlowBuilder, det, family, dim, geo, eq, eqp):
    """The decision records of the physics: family, dimension, conditions, coefficients, exact solution."""
    scores = det.get("family_scores") or {}
    if det.get("family") and scores:
        words = ", ".join(f"{k} ({v:g})" for k, v in sorted(scores.items(), key=lambda kv: -kv[1])[:3])
        fb.decide("family.keywords", f"PDE family: {family}", node=eq, value=family,
                  rationale=f"keyword scores {words}; " + ("steady wording turned heat into "
                                                         f"{family}" if family in ("laplace", "poisson")
                                                         and "heat" in scores else "highest score wins"))
    elif det.get("family") is None:
        fb.decide("family.default", f"PDE family: {family} (default)", node=eq, value=family)
    rid = f"family.{family}" if f"family.{family}" in RULES else "family.keywords"
    fb.decide(rid, f"equations: {'; '.join(eqp['pdes'])}", node=eq, value=eqp["pdes"])
    if det.get("dimension"):
        fb.decide("dimension.keywords", f"{dim}-D", node=geo, value=dim)
    else:
        fb.decide("dimension.default", f"{dim}-D (no dimension in the text)", node=geo, value=dim)
    fb.decide("domain.unit_box", "domain extent and time interval", node=geo)
    if any(c["kind"] == "neumann" for c in eqp["conditions"]) and "neumann" in det.get("boundary", []):
        fb.decide("bc.insulated", "insulated faces: zero normal derivative", node=eq,
                  value=[c for c in eqp["conditions"] if c["kind"] == "neumann"])
    fb.decide("bc.dirichlet_template", "boundary conditions of the template", node=eq,
              value=[c for c in eqp["conditions"] if c["kind"] != "initial"])
    if any(c["kind"] == "initial" for c in eqp["conditions"]):
        fb.decide("ic.template", "initial condition", node=eq,
                  value=[c for c in eqp["conditions"] if c["kind"] == "initial"])
    for k, v in eqp["coefficients"].items():
        if k in det.get("coefficients", {}):
            fb.decide("coef.from_text", f"{k} = {v:g} (from the description)", node=eq, value={k: v})
        else:
            fb.decide("coef.template_default", f"{k} = {v:g} (template default)", node=eq, value={k: v})
    if eqp.get("exact"):
        fb.decide("exact.cole_hopf" if eqp["exact"] == "cole_hopf" else "exact.template",
                  f"exact solution: {eqp['exact']}", node=eq, value=eqp["exact"])


def _solve(det, family, dim, target_dim=None):
    intents = set(det["intents"])
    fb = FlowBuilder(f"{family} {dim}D: classical vs PINN")
    eq, t = _problem(fb, det, family, dim)
    methods = []
    fd = fb.add("fd", "solver.classical", {"method": "fd"}, label="Finite differences")
    fb.connect(eq, "problem", fd, "problem")
    methods.append(fd)
    if family in ("poisson", "laplace") and dim >= 2:
        fem = fb.add("fem", "solver.classical", {"method": "fem", "n": 25 if dim == 2 else 11}, label="Finite elements")
        fb.connect(eq, "problem", fem, "problem")
        methods.append(fem)
    steps = {1: 3000, 2: 4000, 3: 3000}[dim] if family != "burgers" else 6000
    pinn = fb.add("pinn", "train.pinn", {"adam_steps": steps, "lbfgs_steps": 300, "width": 32, "depth": 4,
                                         "n_collocation": 3000 if dim < 3 else 4000}, label="PINN")
    fb.connect(eq, "problem", pinn, "problem")
    methods.append(pinn)
    val = fb.add("validate", "validate.compare", {"reference": "auto"}, label="Validation")
    for m in methods:
        fb.connect(m, "solution", val, "candidates")
    fb.connect(eq, "problem", val, "problem")
    if not t["exact"]:
        fb.connect(fd, "solution", val, "reference")
        fb.nodes[val]["params"]["reference"] = "input"
    lab = fb.add("lab", "validate.lab", {"title": f"{family} {dim}D"}, label="Lab seal")
    fb.connect(val, "metrics", lab, "validation")
    plot = fb.add("plot", "viz.plot", {}, label="Plots")
    fb.connect(fd, "solution", plot, "solution")
    fb.connect(pinn, "solution", plot, "solution")
    if dim >= 2 or "viz3d" in intents:
        v3 = fb.add("view3d", "viz.field3d", {}, label="3-D viewer")
        fb.connect(fd, "solution", v3, "solution")
    if "viz3d" in intents:
        r = fb.add("render", "viz.render", {}, label="Render")
        fb.connect(fd, "solution", r, "solution")
    rep = fb.add("report", "export.report", {"title": f"{family} in {dim}-D"}, label="Report")
    fb.connect(val, "metrics", rep, "items")
    fb.connect(val, "figure", rep, "items")
    fb.decide("method.fd", "finite differences (method of lines) as the classical reference", node=fd)
    if "fem" in fb.nodes:
        fb.decide("method.fem", "P1 finite elements as a second classical method", node="fem")
    fb.decide("method.pinn", "a PINN on the same equations", node=pinn)
    fb.decide("method.pinn_training", f"{steps} Adam steps then 300 L-BFGS, width 32, depth 4", node=pinn,
              value={"adam_steps": steps, "lbfgs_steps": 300})
    fb.decide("architecture.default_sizes", "PINN width 32, depth 4", node=pinn, value={"width": 32, "depth": 4})
    fb.decide("validation.reference", "compare against " + ("the exact solution" if t["exact"] else
                                                            "the finite-difference solution"), node=val)
    fb.decide("validation.threshold", "pass at relative L2 <= 0.02", node=val, value=0.02)
    fb.decide("validation.lab_tiers", "Lab curation and seal of the validation", node=lab)
    fb.decide("route.alternatives", "route: classical solver + PINN, validated")
    if target_dim:
        fb.decide("dimension.lap_switch", f"switch {dim}-D -> {target_dim}-D on the domain node; equations follow "
                  "the template", node="eq", value=[dim, target_dim])
    why = (f"Solve the {t['description']} in {dim}-D with finite differences"
           + (" and finite elements" if len(methods) > 2 else "") + " and a PINN, and measure both against "
           + ("the exact solution" if t["exact"] else "the finite-difference reference")
           + "; the Lab grades the evidence and seals A/B results.")
    if target_dim:
        why += f" Switch the domain node to {target_dim}-D afterwards: the equations follow (lap() is dimension-free)."
        fb.nodes["eq"]["params"]["follow_template"] = True
    return fb.proposal("Classical solver + PINN, validated", why, family=family)


def _surrogate(det, family, dim):
    intents = set(det["intents"])
    kind = "pod_mlp" if "hybrid" in intents else "deeponet"
    fb = FlowBuilder(f"{family} {dim}D surrogate")
    eq, t = _problem(fb, det, family, min(dim, 2))
    vary_key = next(iter(t["coefficients"]), None)
    if vary_key is None:
        fb.nodes[eq]["params"], vary_key = _with_amplitude(fb.nodes[eq]["params"])
        vary = {vary_key: [0.5, 2.0]}
    else:
        v = fb.nodes[eq]["params"]["coefficients"][vary_key]
        vary = {vary_key: [round(v * 0.5, 8), round(v * 2.0, 8)]}
    data = fb.add("sweep", "data.parametric", {"vary": vary, "n_samples": 96 if dim == 1 else 64,
                                               "n": 81 if dim == 1 else 33, "output": "final"},
                  label="Parametric sweep (FD)")
    fb.connect(eq, "problem", data, "problem")
    if "hpo" in intents:
        sur = fb.add("hpo", "surrogate.hpo", {"kind": kind, "n_trials": 10, "trial_epochs": 150,
                                              "final_epochs": 600,
                                              "space": {"lr": [3e-4, 1e-2, "log"], "width": [16, 128],
                                                        "depth": [1, 4]} | ({"rank": [4, 16]} if kind == "pod_mlp"
                                                                            else {})},
                     label="HPO")
    else:
        sur = fb.add("surrogate", "surrogate.train", {"kind": kind, "epochs": 600}, label=f"{kind} surrogate")
    fb.connect(data, "data", sur, "data")
    uq = fb.add("sobol", "uq.sensitivity", {"quantity": "mean"}, label="Sensitivity")   # cheap on a surrogate
    fb.connect(sur, "model", uq, "model")
    fb.decide("sampling.lhs", f"Latin hypercube over {vary}", node=data, value=vary)
    fb.decide("method.fd", "training data from the finite-difference solver", node=data)
    fb.decide(f"surrogate.{kind}", f"{kind} surrogate", node=sur, value=kind)
    fb.decide("architecture.default_sizes", "suggested architecture sizes (see the preview)", node=sur)
    if "hpo" in intents:
        fb.decide("hpo.search", "TPE (Optuna) when installed, else random search", node=sur)
        fb.decide("hpo.ranges", "search ranges", node=sur, value=fb.nodes[sur]["params"]["space"])
    fb.decide("uq.sobol", "Sobol indices of the field mean", node=uq)
    fb.decide("route.alternatives", "route: parametric data + surrogate")
    rep = fb.add("report", "export.report", {"title": f"{family} surrogate"}, label="Report")
    fb.connect(sur, "metrics", rep, "items")
    why = (f"Generate {fb.nodes['sweep']['params']['n_samples']} finite-difference solutions over "
           f"{', '.join(vary)} and train a {'POD + MLP (ROM + NN hybrid)' if kind == 'pod_mlp' else 'DeepONet'} "
           f"surrogate{' with a hyperparameter search' if 'hpo' in intents else ''}; held-out test error and "
           f"Sobol sensitivity of the field.")
    return fb.proposal("Fast surrogate (neural operator)", why, family=family)


def _inverse(det, family, dim):
    t0 = template(family, dim)
    unknown = next(iter(det["coefficients"]), None) or next(iter(t0["coefficients"]), None)
    fb = FlowBuilder(f"identify {unknown} ({family} {dim}D)")
    eq_true, t = _problem(fb, det, family, dim, prefix="true_", label="Domain (synthetic truth)")
    fd = fb.add("fd", "solver.classical", {"method": "fd"}, label="Synthetic measurements (FD)")
    fb.connect(eq_true, "problem", fd, "problem")
    obs = fb.add("sensors", "data.observe", {"n_points": 400, "noise": 0.01}, label="Sensors (1% noise)")
    fb.connect(fd, "solution", obs, "solution")
    fb.connect(eq_true, "problem", obs, "problem")
    eq_inv, _ = _problem(fb, det, family, dim, prefix="inv_", unknowns=[unknown] if unknown else None,
                         label="Domain (inverse model)")
    disc = fb.add("discover", "inverse.discover", {"initial_guess": {unknown: 1.0} if unknown else {},
                                                   "adam_steps": 3000, "lbfgs_steps": 200},
                  label="Inverse PINN")
    fb.connect(eq_inv, "problem", disc, "problem")
    fb.connect(obs, "data", disc, "data")
    cal = fb.add("calibrate", "inverse.calibrate", {"bounds": {unknown: [1e-4, 10.0]} if unknown else {},
                                                    "n": 41 if dim == 1 else 21}, label="Calibration (FD)")
    fb.connect(eq_inv, "problem", cal, "problem")
    fb.connect(obs, "data", cal, "data")
    rep = fb.add("report", "export.report", {"title": f"identify {unknown}"}, label="Report")
    fb.connect(disc, "metrics", rep, "items")
    fb.connect(cal, "metrics", rep, "items")
    fb.decide("inverse.pinn", f"identify {unknown} with an inverse PINN", node=disc, value=unknown)
    fb.decide("inverse.calibration", f"identify {unknown} by calibration with the FD solver", node=cal)
    fb.decide("method.fd", "synthetic measurements from the FD solver", node=fd)
    fb.decide("route.alternatives", "route: inverse problem")
    why = (f"Treat {unknown} as unknown: synthetic sensor data from a finite-difference solution (replace with your "
           f"measurements in a data-ingest node), then identify it two ways: an inverse PINN and a least-squares "
           f"calibration with the classical solver.")
    return fb.proposal(f"Identify {unknown} from measurements", why, family=family)


def _twin(det, text):
    fb = FlowBuilder("digital twin")
    leak = "leak" in text.lower() or "vazamento" in text.lower()
    tw = fb.add("twin", "twin.connect", {"source": "simulated", "fault": "leak" if leak else "none"},
                label="Twin runtime")
    rep = fb.add("report", "export.report", {"title": "digital twin"}, label="Report")
    fb.connect(tw, "metrics", rep, "items")
    fb.decide("twin.design", "twin of the heated tank: plant stream, EnKF, alerts", node=tw)
    fb.decide("twin.enkf", "ensemble Kalman filter", node=tw)
    fb.decide("twin.parameter_state", "leak flow and heater efficiency as estimated states", node=tw)
    fb.decide("twin.alert_thresholds", "alert thresholds", node=tw)
    why = ("Stream sensor readings (the simulated heated tank; switch the source to MQTT or connect recorded data) "
           "into the twin runtime: an ensemble Kalman filter estimates level, temperature and hidden states (leak "
           "flow, heater efficiency) with uncertainty and raises alerts.")
    return fb.proposal("Digital twin on a sensor stream", why, family="twin")


NUMBER_WORDS = {w: i for i, ws in enumerate([(), ("one", "um", "uma"), ("two", "dois", "duas"), ("three", "três", "tres"),
                                               ("four", "quatro"), ("five", "cinco"), ("six", "seis"), ("seven", "sete"),
                                               ("eight", "oito"), ("nine", "nove"), ("ten", "dez")]) for w in ws}


def sensor_count(text: str) -> int | None:
    """``5 temperature sensors`` / ``three sensors`` / ``6 sensores`` -> the count (1-64), else None: the nearest
    number within the three words before a sensor word."""
    words = re.findall(r"[0-9]+|[a-zçãéêó-]+", text.lower())
    for k, w in enumerate(words):
        if not re.fullmatch(r"sensors?|sensores?|probes?|thermocouples?|termopares?", w):
            continue
        for prev in reversed(words[max(0, k - 3):k]):
            n = int(prev) if prev.isdigit() else NUMBER_WORDS.get(prev)
            if n is not None:
                return n if 1 <= n <= 64 else None
    return None


def sensor_layout(n: int, dim: int) -> list:
    """``n`` positions evenly inside the unit interval / square (a near-square grid in 2-D)."""
    if dim == 1:
        return [round((i + 1) / (n + 1), 4) for i in range(n)]
    nx = math.ceil(math.sqrt(n))
    ny = math.ceil(n / nx)
    pts = [[round((i + 1) / (nx + 1), 4), round((j + 1) / (ny + 1), 4)] for j in range(ny) for i in range(nx)]
    return pts[:n]


def _pde_twin(det, text):
    """A twin generated from the description: identify the equations, stream sensors, build the twin."""
    fam = det["family"] or "heat"
    fb = FlowBuilder(f"digital twin ({fam})")
    gen = fb.add("problem", "problem.generic", {"description": text, "dimension": det["dimension"] or 0},
                 label="Identified problem")
    anomaly = any(_hit(text.lower(), k) for k in ("detect", "change", "degrad", "anomal", "fault", "drift",
                                                  "detectar", "mudança", "falha"))
    t = template(fam if fam in FAMILIES else "heat", det["dimension"] or 1) if fam != "burgers" else template(fam, 1)
    coef = next(iter({**t["coefficients"], **det["coefficients"]}), None)
    tend = (t["time"] or [0.0, 1.0])[1]
    fault = {"t": round(0.5 * tend, 6), coef: round(1.5 * {**t["coefficients"], **det["coefficients"]}[coef], 8)} \
        if anomaly and coef else {}
    dim = det["dimension"] or 1
    n_sensors = sensor_count(text)
    sensors = sensor_layout(n_sensors, min(dim, 2)) if n_sensors else (
        [0.2, 0.4, 0.6, 0.8] if dim == 1 else [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75], [0.5, 0.5]])
    live = fb.add("live", "data.stream", {"source": "simulated", "sensors": sensors, "t_end": tend,
                                          "dt": round(tend / 300, 6), "fault": fault}, label="Live data")
    fb.connect(gen, "problem", live, "problem")
    tw = fb.add("twin", "twin.auto", {"estimate": [coef] if coef else []}, label="Digital twin")
    fb.connect(gen, "problem", tw, "problem")
    fb.connect(live, "stream", tw, "stream")
    v3 = fb.add("view3d", "viz.field3d", {}, label="Twin 3-D view")
    fb.connect(tw, "solution", v3, "solution")
    rep = fb.add("report", "export.report", {"title": "digital twin"}, label="Report")
    fb.connect(tw, "metrics", rep, "items")
    if n_sensors:
        fb.decide("twin.sensor_layout", f"{n_sensors} sensors, evenly spaced", node=live, value=sensors)
    fb.decide("twin.design", "twin generated from the identified equations", node=tw)
    fb.decide("twin.coarse_model", "identified PDE on a coarse grid as the twin model", node=tw)
    fb.decide("twin.enkf", "ensemble Kalman filter for the state", node=tw)
    if coef:
        fb.decide("twin.parameter_state", f"calibrate {coef} live", node=tw, value=coef)
    if fault:
        fb.decide("twin.simulated_fault", f"simulated asset changes {coef} at t = {fault['t']:g}", node=live,
                  value=fault)
    fb.decide("route.alternatives", "route: automatic digital twin")
    why = ("Identify the governing equations from the description, simulate (or connect) the sensors, and build the "
           "twin: the identified PDE on a coarse grid with an ensemble Kalman filter, live calibration of "
           f"{coef or 'the coefficients'}, alerts when the asset stops behaving like the model, 3-D view.")
    return fb.proposal("Digital twin generated from the description", why, family=fam)


def _forecast(det):
    fb = FlowBuilder("forecast")
    fc = fb.add("forecast", "forecast.lab", {"source": "elnino"}, label="Forecast lab")
    rep = fb.add("report", "export.report", {"title": "forecast"}, label="Report")
    fb.connect(fc, "metrics", rep, "items")
    fb.decide("forecast.predictability_first", "predictability report before models", node=fc)
    fb.decide("forecast.baselines", "baselines and DM tests on an untouched test period", node=fc)
    why = ("Measure predictability first (surrogate tests), then compare baselines and learned forecasters on "
           "purged walk-forward folds with an untouched test period.")
    return fb.proposal("Forecast a near-random series", why, family="forecast")


def _flight(det):
    fb = FlowBuilder("take-off")
    fl = fb.add("takeoff", "flight.takeoff", {}, label="Take-off (6-DOF)")
    rep = fb.add("report", "export.report", {"title": "take-off"}, label="Report")
    fb.connect(fl, "metrics", rep, "items")
    fb.decide("flight.raymer", "compare with Raymer's closed form", node=fl)
    return fb.proposal("Aircraft take-off simulation", "6-DOF take-off with the vortex lattice in the loop, "
                       "against Raymer's closed form.", family="flight")


__all__ = ["detect", "plan_heuristic", "FAMILIES"]
