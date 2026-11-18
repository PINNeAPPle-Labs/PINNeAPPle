"""The predefined questions, so the web app can walk a user to a plan without a language model at all.

Each question fills one field of the brief (`pinneapple_advisor.brief.FIELDS`), and every choice carries a label in
Portuguese and in English. ``next_questions`` returns what is still open, in the order physics -> geometry -> data ->
budget, and never asks something the sentence already answered.

A question exists when its answer changes the plan: whether the problem is 3-D decides if a geometry and a mesh are
on the path, whether the user already has a dataset removes the campaign, and the number of evaluations decides
between sweeping the solver and training a surrogate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .brief import ASK_ORDER, Brief


@dataclass(frozen=True)
class Question:
    id: str
    field: str
    kind: str                       # "choice" | "bool" | "int" | "number" | "list"
    pt: str
    en: str
    why: str                        # what changes in the plan because of the answer (English, for the API meta)
    choices: list[dict[str, str]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.id, "field": self.field, "kind": self.kind, "pt": self.pt, "en": self.en, "why": self.why,
                "choices": list(self.choices)}


def _choices(*pairs: dict[str, str]) -> list[dict[str, str]]:
    return list(pairs)


_FAMILY_LABELS = [
    ("heat_equation", "condução de calor"), ("laplace", "Laplace (potencial)"), ("poisson", "Poisson"),
    ("wave", "onda"), ("helmholtz", "Helmholtz"), ("linear_elasticity", "elasticidade linear"),
    ("navier_stokes", "escoamento (Navier-Stokes)"), ("darcy", "Darcy (meio poroso)"), ("burgers", "Burgers"),
    ("reaction_diffusion", "reação-difusão"), ("fisher_kpp", "Fisher-KPP"), ("sir", "epidemia SIR"),
    ("black_scholes", "Black-Scholes"), ("inviscid_burgers", "Burgers invíscido"),
    ("custom", "nenhuma delas / não sei"),
]

QUESTIONS: list[Question] = [
    Question("physics_family", "physics_family", "choice",
             "Qual é a física do problema (que equações governam)?",
             "What is the physics (which equations govern it)?",
             "Sets the equations to confirm, the solver and the estimated cost per solve.",
             _choices(*[{"value": value, "pt": label, "en": label} for value, label in _FAMILY_LABELS])),
    Question("dimension", "dimension", "choice",
             "O problema é 1-D, 2-D ou 3-D?", "Is the problem 1-D, 2-D or 3-D?",
             "3-D puts a geometry and a mesh on the path, and multiplies the cost per solve.",
             _choices({"value": "1", "pt": "1-D", "en": "1-D"}, {"value": "2", "pt": "2-D", "en": "2-D"},
                      {"value": "3", "pt": "3-D", "en": "3-D"})),
    Question("time_dependent", "time_dependent", "bool",
             "O resultado é transiente (varia no tempo)?", "Is the result transient (does it change in time)?",
             "A transient adds a time loop: many more solver steps, and a time axis in the dataset."),
    Question("nonlinear", "nonlinear", "bool",
             "A física é não linear (propriedades que dependem do campo, contato, radiação)?",
             "Is the physics nonlinear (field-dependent properties, contact, radiation)?",
             "A nonlinear solve iterates; the cost estimate multiplies by the iteration count."),
    Question("turbulent", "turbulent", "bool",
             "O escoamento é turbulento?", "Is the flow turbulent?",
             "Turbulence needs a closure, a finer near-wall mesh (y+) and roughly three times the iterations."),
    Question("geometry_present", "geometry_present", "bool",
             "Você já tem a geometria (CAD, STEP, STL)?", "Do you already have the geometry (CAD, STEP, STL)?",
             "With the geometry in hand the plan starts at the import and the mesh; without it, generation comes first."),
    Question("geometry_formats", "geometry_formats", "list",
             "Em que formato a geometria está?", "In which format is the geometry?",
             "The format decides the importer, and whether a repair step is needed before meshing.",
             _choices(*[{"value": v, "pt": v.upper(), "en": v.upper()} for v in ("cad", "stl", "step", "analytic")])),
    Question("geometry_varies", "geometry_varies", "bool",
             "A geometria varia entre os casos (família de geometrias ou otimização de forma)?",
             "Does the geometry vary between cases (a geometry family, or shape optimisation)?",
             "A varying geometry points the surrogate to a graph network and adds one mesh per sample to the campaign."),
    Question("mesh_present", "mesh_present", "bool",
             "Você já tem a malha?", "Do you already have the mesh?",
             "A ready mesh removes the meshing step; an analytic domain lets the solver build its own grid."),
    Question("dataset_present", "dataset_present", "bool",
             "Você já tem um dataset de simulações?", "Do you already have a dataset of simulations?",
             "With data the plan goes straight to training and validation; without it, a campaign comes first."),
    Question("n_simulations_present", "n_simulations_present", "int",
             "Quantas simulações o dataset tem?", "How many simulations does the dataset have?",
             "The count decides data-scarce (PINN / physics-informed) versus data-rich (operator learning)."),
    Question("parameter_varies", "parameter_varies", "bool",
             "Os parâmetros variam entre os casos (varredura, otimização)?",
             "Do parameters vary between cases (sweep, optimisation)?",
             "Varying parameters is what makes it an operator-learning problem instead of a single fit."),
    Question("n_parameters", "n_parameters", "int",
             "Quantos parâmetros de entrada variam?", "How many input parameters vary?",
             "Sets the number of samples the surrogate needs (about ten per input for a scalar target)."),
    Question("goal", "goal", "choice",
             "Qual é o objetivo do estudo?", "What is the goal of the study?",
             "An inverse problem adds the calibration stage; an optimisation adds the search over the surrogate.",
             _choices({"value": "forward", "pt": "prever o campo / a resposta", "en": "predict the field / the answer"},
                      {"value": "inverse", "pt": "descobrir um parâmetro a partir de medições",
                       "en": "recover a parameter from measurements"},
                      {"value": "optimization", "pt": "otimizar um projeto", "en": "optimise a design"},
                      {"value": "screening", "pt": "triagem de muitos casos", "en": "screen many cases"})),
    Question("evaluations_needed", "evaluations_needed", "int",
             "Quantas avaliações o estudo precisa?", "How many evaluations does the study need?",
             "Direct cost = solves x evaluations; that product is what decides solver versus surrogate."),
    Question("output_kind", "output_kind", "choice",
             "O que o modelo deve devolver?", "What should the model return?",
             "A scalar goes to a KPI regressor, a grid field to FNO/DeepONet/PINO, a field on varying meshes to a graph network.",
             _choices({"value": "scalar", "pt": "um número (KPI)", "en": "a number (a KPI)"},
                      {"value": "grid_field", "pt": "um campo numa malha estruturada", "en": "a field on a structured grid"},
                      {"value": "mesh_field", "pt": "um campo numa malha não estruturada", "en": "a field on an unstructured mesh"},
                      {"value": "time_series", "pt": "uma série no tempo", "en": "a time series"})),
    Question("accuracy_target", "accuracy_target", "number",
             "Qual erro relativo é aceitável no resultado?", "What relative error is acceptable in the result?",
             "The accuracy target becomes the first check the surrogate must pass on held-out solves."),
    Question("cores", "cores", "int",
             "Quantos núcleos de CPU estão disponíveis?", "How many CPU cores are available?",
             "Only the wall-clock estimate scales with it (sub-linearly, Amdahl)."),
    Question("gpu", "gpu", "bool",
             "Há GPU disponível?", "Is a GPU available?",
             "A GPU changes the training time estimate and which trainers are worth suggesting."),
    Question("approx_cells", "approx_cells", "int",
             "Quantas células a malha tem (ou terá)?", "How many cells does (or will) the mesh have?",
             "Cells x iterations is the cost model; a measured mesh replaces the default resolution."),
    Question("solve_seconds_measured", "solve_seconds_measured", "number",
             "Quanto tempo levou uma simulação, em segundos?", "How long did one simulation take, in seconds?",
             "A measured solve replaces the estimated cost model and makes every downstream number measured."),
    Question("budget_hours", "budget_hours", "number",
             "Qual é o orçamento de tempo (em horas)?", "What is the time budget (in hours)?",
             "A campaign that does not fit the budget is flagged, and the surrogate becomes the recommended path."),
    Question("budget_usd", "budget_usd", "number",
             "Qual é o orçamento em dinheiro (US$)?", "What is the money budget (US$)?",
             "The cost estimate is compared with it; a local machine stays at zero by default."),
]

BY_FIELD: dict[str, Question] = {q.field: q for q in QUESTIONS}
BY_ID: dict[str, Question] = {q.id: q for q in QUESTIONS}


def questions() -> list[dict[str, Any]]:
    """Every predefined question, in asking order."""
    return [BY_FIELD[f].to_dict() for f in ASK_ORDER if f in BY_FIELD]


def next_questions(brief: Brief, limit: int | None = None) -> list[dict[str, Any]]:
    """The open questions for this brief -- what the web app asks next, and nothing it already knows."""
    open_fields = brief.unknown()
    out = [BY_FIELD[f].to_dict() for f in open_fields if f in BY_FIELD]
    return out[:limit] if limit else out


def question(question_id: str) -> Question:
    try:
        return BY_ID[question_id]
    except KeyError as exc:
        raise KeyError(f"no question {question_id!r}; known: {', '.join(BY_ID)}") from exc


__all__ = ["Question", "QUESTIONS", "BY_FIELD", "BY_ID", "questions", "next_questions", "question"]
