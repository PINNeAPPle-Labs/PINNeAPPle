"""Kinds of study and the steps each one takes, so a pipeline is assembled from what the user wants to know.

A design optimisation (geometry → mesh → physics → DOE → simulation → dataset → surrogate → optimisation →
verification) is one kind of study among several. An inverse problem identifies parameters from measurements; an
uncertainty study propagates input distributions and ranks sensitivities; a twin assimilates live data; a forecast
learns from a time series; a control study designs a controller and checks the closed loop; a coupled study iterates
between physics; a data analysis fits and tests a model on experiments. Each :class:`StudyKind` lists its steps
(:data:`STUDY_ROLES`: catalog query in English concept words, checks, titles in English and Portuguese) and the words
that announce it. :func:`classify_study` ranks the kinds for a goal; the pipeline then fills each step with a tool of
the catalog, or leaves a gap with what is missing.

    classify_study("estimate the thermal conductivity of the wall from these temperature measurements")[0]
    # StudyMatch(kind='inverse', score=..., words=['estimate', 'from these ... measurements', ...])
    roles_for_kind("inverse", spatial=True, has_geometry=False, has_data=True)
    # ['geometry', 'mesh', 'dataset', 'physics', 'simulate', 'identify', 'uq']   (identify_step verifies on held-out rows)

Steps marked ``spatial`` (geometry, mesh) are added only when the physics is spatially resolved; ``dataset`` comes
from the user's file when there is one, otherwise from a generating step.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class StudyRole:
    id: str
    title: str
    title_pt: str
    query: str                                        # catalog search words (English concept words)
    checks: tuple[tuple[str, str], ...] = ()          # (code, what it checks); evaluated on the step's result
    cost_s: float = 10.0                              # heuristic estimate, never presented as measured
    tool: str = ""                                    # the library's own tool for this step, tried first


@dataclass(frozen=True)
class StudyKind:
    id: str
    title: str
    title_pt: str
    description: str
    roles: tuple[str, ...]
    words: tuple[str, ...]                            # English and Portuguese, matched at word starts
    verify_checks: tuple[tuple[str, str], ...] = ()   # what "verified" means for this kind
    spatial_roles: bool = True                        # geometry / mesh may be prepended when the physics is spatial
    verify_in: str = ""                               # the step whose own tool computes verify_checks (no verify step)


STUDY_ROLES: dict[str, StudyRole] = {r.id: r for r in [
    StudyRole("geometry", "Parametric geometry", "Geometria paramétrica", "parametric geometry generate shape",
              (("geometry.in_ranges", "dimensions inside the design-variable ranges"),
               ("geometry.positive_volume", "positive volume")), 2),
    StudyRole("mesh", "Mesh", "Malha", "mesh generation",
              (("mesh.quality", "element quality above threshold"),
               ("mesh.independence", "mesh-independence of the quantity of interest")), 30),
    StudyRole("physics", "Physical model", "Modelo físico", "model",
              (("physics.known_answer", "reproduces a known-answer case of the model"),), 1),
    StudyRole("doe", "Design of experiments (DOE)", "Plano de experimentos (DOE)",
              "design of experiments sampling latin hypercube",
              (("doe.covers_ranges", "samples cover every varied range"),), 1),
    StudyRole("simulate", "Simulation", "Simulação", "simulate solver",
              (("simulate.balance", "balance residual below tolerance"),
               ("simulate.converged", "solver converged on every sample")), 20),
    StudyRole("dataset", "Dataset", "Conjunto de dados", "dataset build table",
              (("dataset.complete", "no missing values"), ("dataset.units", "units recorded per column")), 2),
    StudyRole("surrogate", "Surrogate model", "Modelo substituto (surrogate)", "surrogate model regression train",
              (("surrogate.held_out_error", "held-out error below the accuracy target"),
               ("surrogate.no_extrapolation", "no extrapolation outside the sampled ranges")), 60),
    StudyRole("optimize", "Optimization", "Otimização", "optimization optimizer minimize",
              (("optimize.constraints", "constraints satisfied at the optimum"),
               ("optimize.in_bounds", "optimum inside the bounds")), 30),
    StudyRole("identify", "Parameter identification", "Identificação de parâmetros",
              "inverse problem identify estimate parameters calibrate from measurements",
              (("identify.fits_data", "identified model reproduces the measurements within their noise"),
               ("identify.identifiable", "parameters identifiable: finite, non-degenerate sensitivity")), 30, "pinneapple_flow.study_steps:identify_step"),
    StudyRole("sensitivity", "Sensitivity analysis", "Análise de sensibilidade",
              "sensitivity analysis sobol indices morris",
              (("sensitivity.indices_sum", "first-order indices sum to at most one (within sampling error)"),
               ("sensitivity.converged", "indices stable when the sample size is doubled")), 30, "pinneapple_flow.study_steps:sensitivity_step"),
    StudyRole("uq", "Uncertainty propagation", "Propagação de incerteza",
              "uncertainty propagation monte carlo confidence interval",
              (("uq.converged", "statistics stable when the sample size is doubled"),
               ("uq.coverage", "intervals cover held-out cases at their nominal rate")), 30, "pinneapple_flow.study_steps:uncertainty_step"),
    StudyRole("stream", "Live data connection", "Conexão a dados ao vivo", "stream sensor live data ingest",
              (("stream.units", "every channel has a unit"), ("stream.gaps", "gaps and outliers flagged")), 5),
    StudyRole("twin", "Digital twin (assimilation and monitoring)", "Gêmeo digital (assimilação e monitoramento)",
              "digital twin state estimation assimilation kalman monitor anomaly",
              (("twin.tracks", "the twin's state tracks the measurements within their noise"),
               ("twin.alarms", "anomaly alarms checked on cases with and without a fault")), 30),
    StudyRole("part_twin", "Digital twin of the part (3-D field, assimilation, alarms)",
              "Gêmeo digital da peça (campo 3D, assimilação, alarmes)",
              "digital twin part geometry thermal field assimilation coolant passage clogging",
              (("twin.tracks", "the twin's state tracks the measurements within their noise"),
               ("twin.alarms", "anomaly alarms checked on cases with and without a fault")), 120,
              "pinneapple_flow.study_steps:twin_step"),
    StudyRole("analyze", "Data analysis", "Análise dos dados", "time series analysis statistics predictability",
              (("analyze.stationarity", "stationarity and trends stated"),
               ("analyze.leakage", "no train/test leakage")), 10, "pinneapple_flow.study_steps:analyze_step"),
    StudyRole("forecast", "Forecast model", "Modelo de previsão", "forecast time series predict",
              (("forecast.horizon", "skill reported per horizon"),), 30, "pinneapple_flow.study_steps:forecast_step"),
    StudyRole("linearize", "Plant analysis", "Análise da planta",
              "linearize equilibrium stability controllability observability",
              (("linearize.stability", "open-loop stability stated"),
               ("linearize.controllable", "controllability checked")), 5),
    StudyRole("control", "Controller design", "Projeto do controlador", "controller design tuning pid lqr",
              (("control.margins", "gain and phase margins reported"),), 10),
    StudyRole("closed_loop", "Closed-loop simulation", "Simulação em malha fechada",
              "closed loop simulate controller actuator disturbance",
              (("closed_loop.tracks", "the reference is tracked within the requirement"),
               ("closed_loop.saturation", "actuator limits respected")), 20),
    StudyRole("couple", "Coupling between physics", "Acoplamento entre físicas",
              "coupled multiphysics coupling interface iteration",
              (("couple.interface_balance", "flux balance across the interface"),
               ("couple.converged", "coupling iterations converged")), 30),
    StudyRole("fit", "Model fit", "Ajuste do modelo", "fit regression model parameters least squares uncertainty",
              (("fit.residuals", "residuals unstructured and within the measurement noise"),
               ("fit.parameter_uncertainty", "parameter uncertainties reported")), 10),
    StudyRole("verify", "Verification", "Verificação", "verification validation compare error",
              (("verify.rechecked", "result re-checked with the full model"),), 20),
]}

STUDY_KINDS: dict[str, StudyKind] = {k.id: k for k in [
    StudyKind("design_optimization", "Design optimization", "Otimização de projeto",
              "find the design that minimizes or maximizes an objective within constraints",
              ("physics", "doe", "simulate", "dataset", "surrogate", "optimize", "verify"),
              ("optimi", "minimi", "maximi", "best design", "lightest", "cheapest", "lowest", "highest", "improve",
               "otimiz", "minimiz", "maximiz", "melhor projeto", "mais leve", "mais barat", "menor", "maior",
               "melhorar"),
              (("verify.resimulated", "optimum re-simulated with the full model"),
               ("verify.surrogate_agrees", "surrogate vs simulation within tolerance"))),
    StudyKind("simulation", "Simulation", "Simulação",
              "compute the response of a given system",
              ("physics", "simulate", "verify"),
              ("simulate", "compute", "solve", "calculate", "what is the", "how hot", "how much", "simul", "calcul",
               "resolver", "qual é", "quanto"),
              (("verify.independent", "result re-checked with an independent method or a refined discretisation"),)),
    StudyKind("surrogate", "Surrogate model", "Modelo substituto",
              "a fast model of a simulation over a range of inputs",
              ("physics", "doe", "simulate", "dataset", "surrogate", "verify"),
              ("surrogate", "fast model", "real-time", "emulat", "reduced order", "many queries", "substitut",
               "modelo rápido", "tempo real", "ordem reduzida"),
              (("verify.surrogate_agrees", "surrogate vs simulation within tolerance on held-out cases"),)),
    StudyKind("inverse", "Parameter identification (inverse problem)", "Identificação de parâmetros (problema inverso)",
              "estimate unknown parameters or inputs of a model from measurements",
              ("dataset", "physics", "simulate", "identify", "uq", "verify"),
              ("identify", "estimate", "unknown", "calibrat", "infer", "inverse", "from measurements", "from data",
               "fit the model", "parameter discovery", "identificar", "estimar", "desconhecid", "calibr", "inferir",
               "inverso", "medições", "medidas", "a partir dos dados", "ajustar o modelo", "measured"),
              (("verify.held_out", "identified model predicts measurements held out of the identification"),),
              verify_in="identify"),
    StudyKind("uq_sensitivity", "Uncertainty and sensitivity", "Incerteza e sensibilidade",
              "propagate input uncertainty to the outputs and rank the inputs that matter",
              ("physics", "doe", "simulate", "sensitivity", "uq", "verify"),
              ("uncertain", "sensitiv", "sobol", "morris", "robust", "tolerance stack", "probability of failure",
               "reliability", "monte carlo", "incerteza", "sensibilidade", "robustez", "confiabilidade",
               "probabilidade de falha"),
              (("verify.converged", "statistics and indices stable when the sample size is doubled"),)),
    StudyKind("twin", "Digital twin", "Gêmeo digital",
              "a model kept in step with live measurements, to monitor and predict",
              ("physics", "stream", "identify", "surrogate", "twin", "verify"),
              ("digital twin", "twin", "live data", "sensor feed", "stream", "monitor", "real-time data", "mqtt",
               "opc", "gêmeo digital", "gemeo digital", "dados ao vivo", "sensores", "monitorar", "tempo real"),
              (("verify.replay", "replayed recorded data: alarms on known faults, none on healthy data"),)),
    StudyKind("forecast", "Forecast", "Previsão",
              "predict the future of a measured time series",
              ("dataset", "analyze", "forecast", "verify"),
              ("forecast", "predict the next", "time series", "next week", "next month", "demand", "previs",
               "série temporal", "serie temporal", "prever", "próxima semana", "demanda"),
              (("verify.beats_baseline", "out-of-sample skill compared with naive baselines on a purged split"),),
              verify_in="forecast"),
    StudyKind("control", "Control design", "Projeto de controle",
              "design a controller and check the closed loop",
              ("physics", "linearize", "control", "closed_loop", "verify"),
              ("control", "controller", "pid", "lqr", "regulat", "stabiliz", "setpoint", "feedback", "actuator",
               "controle", "controlador", "regular", "estabiliz", "realimentação", "atuador"),
              (("verify.robust", "closed loop robust to the stated model uncertainty"),), spatial_roles=False),
    StudyKind("coupled", "Coupled multiphysics", "Multifísica acoplada",
              "several physics that exchange quantities and must be solved together",
              ("physics", "couple", "simulate", "verify"),
              ("coupled", "multiphysics", "fluid-structure", "thermo-mechanical", "thermomechanical", "conjugate",
               "electro-thermal", "acoplad", "multifísic", "fluido-estrutura", "termomecânic", "eletrotérmic"),
              (("verify.interface", "interface conservation and coupling convergence"),)),
    StudyKind("data_analysis", "Experimental data analysis", "Análise de dados experimentais",
              "fit and test a model against experimental data",
              ("dataset", "analyze", "fit", "verify"),
              ("experiment", "measurements", "lab data", "test data", "fit a curve", "regression", "statistic",
               "experimento", "medições", "dados de ensaio", "ensaio", "ajustar uma curva", "regressão",
               "estatístic"),
              (("verify.cross_validated", "fit cross-validated; residuals within the measurement noise"),),
              spatial_roles=False),
]}

# the more specific kind wins a tie (an inverse problem also "computes"; a twin also "monitors" a simulation)
_PRIORITY = ("twin", "inverse", "control", "coupled", "uq_sensitivity", "forecast", "data_analysis",
             "design_optimization", "surrogate", "simulation")


@dataclass(frozen=True)
class StudyMatch:
    kind: str
    score: float
    words: list[str] = field(default_factory=list)


def _hit(text: str, w: str) -> bool:
    tail = r"(?![a-z0-9à-ü])" if len(w) <= 4 else ""
    return re.search(r"(?<![a-z0-9à-ü])" + re.escape(w) + tail, text) is not None


def classify_study(text: str, *, has_data: bool = False, has_objective: bool = False) -> list[StudyMatch]:
    """The kinds of study a goal announces, best first (always at least one: ``simulation`` when nothing matches).

    ``has_data`` (the user gave measurements) and ``has_objective`` (a quantity to minimise or maximise with design
    variables) add evidence for the kinds that need them."""
    t = text.lower()
    out = []
    for k in STUDY_KINDS.values():
        words = [w for w in k.words if _hit(t, w)]
        score = float(len(words))
        if has_data and k.id in ("inverse", "forecast", "data_analysis", "twin"):
            score += 0.5
        if has_objective and k.id == "design_optimization":
            score += 1.0
        if score > 0:
            out.append(StudyMatch(k.id, score, words))
    out.sort(key=lambda m: (-m.score, _PRIORITY.index(m.kind)))
    return out or [StudyMatch("simulation", 0.0, [])]


def roles_for_kind(kind: str, *, spatial: bool = False, has_geometry: bool = False, has_data: bool = False,
                   is_step: bool = False) -> list[str]:
    """The ordered steps of a study kind for this problem: ``cad`` first for a STEP file; ``geometry`` when the physics
    is spatial and no geometry was given (always, for a design optimisation); ``mesh`` when it is spatial; ``dataset`` from the user's file (and the DOE
    and simulation that would only generate it dropped) when there is data. A digital twin of a given part is the
    one step of the library's part twin."""
    k = STUDY_KINDS[kind]
    roles: list[str] = []
    if is_step:
        roles.append("cad")
    if kind == "twin" and has_geometry:
        # the library's twin of a given part (twin_step) does the physics on the part, the data (recorded, or a
        # simulated replay), the assimilation and the replay check in one step; a twin without a part (a machine
        # watched through its sensors' data) keeps the generic twin steps
        return roles + ["part_twin"]
    if k.spatial_roles:
        # a parametric geometry when none was given and the physics is spatial; a design optimisation keeps it even
        # without spatial physics, because its design variables are usually dimensions
        if not has_geometry and (spatial or kind == "design_optimization"):
            roles.append("geometry")
        if spatial:
            roles.append("mesh")
    body = list(k.roles)
    if has_data and kind in ("design_optimization", "surrogate"):
        body = [r for r in body if r not in ("doe", "simulate")]       # the data replaces the generated campaign
        if "dataset" not in body:
            body.insert(body.index("physics") + 1, "dataset")
    if k.verify_in:                  # the verification is computed by that step's own tool, held-out data included
        body = [r for r in body if r != "verify"]
    for r in body:
        if r not in roles:
            roles.append(r)
    return roles


_VERIFIES_ITSELF = {"part_twin"}           # roles whose library tool emits the kind's verification checks


def role_checks(kind: str, role: str) -> tuple[tuple[str, str], ...]:
    """The checks of a step: the kind's own definition of verification for ``verify`` (or, for a kind with
    ``verify_in``, added to that step's), the role's otherwise."""
    k = STUDY_KINDS[kind]
    if role == "verify" and k.verify_checks:
        return k.verify_checks
    r = STUDY_ROLES.get(role)
    own = r.checks if r else ()
    return own + k.verify_checks if role == k.verify_in or role in _VERIFIES_ITSELF else own


__all__ = ["STUDY_KINDS", "STUDY_ROLES", "StudyKind", "StudyRole", "StudyMatch", "classify_study", "roles_for_kind",
           "role_checks"]
