"""One sentence in, a plan out: which preset, which method, which model, and why.

A new user arrives with a problem, not with a preset name. This module answers the three
questions they actually have, in the order they have them:

1. which preset is closest to my problem (and what do the near ones differ in),
2. which method gives an answer today without training anything,
3. which model to train once the problem description is right.

The ranking is keyword scoring over a curated catalogue (:mod:`pinneapple_core.onboarding.catalog`),
so it can be wrong, and it says how it decided: every recommendation carries the terms that matched
and the terms that did not. When nothing matches, it says so and routes to
``pinneapple_physics.pde_environment.capabilities.suggest_problem_spec`` and to ``pp.new`` instead
of inventing a preset.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .catalog import _COMPILER_KINDS, card, runnable, search, snippet

__all__ = ["Recommendation", "recommend"]

# Words that change the intended METHOD rather than the problem.
_TRAIN_WORDS = {"train", "treinar", "training", "pinn", "learn", "aprender", "neuro", "neural", "rede"}
_INVERSE_WORDS = {"inverse", "inverso", "inversa", "estimar", "estimate", "recover", "recuperar",
                  "calibrar", "calibrate", "identificar", "identify", "desconhecido", "unknown"}
_SWEEP_WORDS = {"varios", "varias", "sweep", "familia", "family", "range", "faixa", "parametrico",
                "parametric", "parametros", "parameters", "lote", "serie"}
_GEOMETRY_WORDS = {"geometria", "geometry", "cad", "stl", "step", "shape", "forma", "malha", "mesh"}


@dataclass
class Recommendation:
    """What to run, why this, and what to run next. ``explain()`` is the text a person reads."""

    query: str
    preset: Optional[str]
    method: str
    model: str
    confidence: float
    reason: str
    snippet: str
    matched: List[str] = field(default_factory=list)
    alternatives: List[Dict[str, Any]] = field(default_factory=list)
    next_steps: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    swapped_from: Optional[str] = None  # the closer preset that does not run on this machine

    def as_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query, "preset": self.preset, "method": self.method, "model": self.model,
            "confidence": self.confidence, "reason": self.reason, "snippet": self.snippet,
            "matched": list(self.matched), "alternatives": list(self.alternatives),
            "next_steps": list(self.next_steps), "notes": list(self.notes),
            "swapped_from": self.swapped_from,
        }

    def explain(self) -> str:
        bar = "-" * 78
        out = [bar, f"  recommendation for: {self.query!r}", bar]
        if not self.preset:
            out += ["  No preset is close enough to recommend.", ""]
            out += ["  " + n for n in self.notes]
            out += ["", "  next:", *["    " + s for s in self.next_steps], bar]
            return "\n".join(out)
        if self.swapped_from:
            out.append(f"  closest  : {self.swapped_from}   (a closer match, but it does not run here)")
        line = f"  preset   : {self.preset}"
        if self.matched:
            line += f"   (matched: {', '.join(self.matched)})"
        out.append(line)
        out.append(f"  method   : {self.method}")
        out.append(f"  model    : {self.model}   (only used when you train)")
        out.append(f"  sure     : {self.confidence:.2f}   how much of your description the catalogue "
                   f"recognised, not a probability")
        out.append(f"  why      : {self.reason}")
        if self.notes:
            out += ["", *["  note: " + n for n in self.notes]]
        if self.alternatives:
            out += ["", "  if it is not quite that one:"]
            for a in self.alternatives:
                out.append(f"    {a['name']:<34} {a['description']}")
        out += ["", "  run it now:", "", *["    " + line for line in self.snippet.splitlines()]]
        if self.next_steps:
            out += ["", "  next:", *["    " + s for s in self.next_steps]]
        out.append(bar)
        return "\n".join(out)

    def __str__(self) -> str:  # so ``print(pp.recommend(...))`` reads well
        return self.explain()


def _has(text: str, words: Sequence[str]) -> bool:
    from .catalog import _tokens

    return bool(set(_tokens(text)) & set(words))


def _model_for(text: str, *, has_solver: bool, method: str) -> tuple[str, str, List[str]]:
    """The model to train, with the reason and the caveats. Never a name the registry does not have."""
    notes: List[str] = []
    try:
        from pinneapple_analysis.verification.architecture_recommendation import (
            ARCHITECTURE_CATALOG,
            recommend_architecture,
        )

        rec = recommend_architecture(
            n_high_fidelity_simulations=0,
            has_analytical_solver=has_solver,
            needs_parameter_generalization=_has(text, _SWEEP_WORDS),
            geometry_varies=_has(text, _GEOMETRY_WORDS),
            is_inverse_problem=_has(text, _INVERSE_WORDS),
        )
        key = rec.recommended[0]
        candidate = ARCHITECTURE_CATALOG.get(key)
        registry_key = getattr(candidate, "registry_key", "") or "vanilla_pinn"
        # one short caveat at most: a beginner does not need the taxonomy, an expert will read the doc
        for caveat in rec.caveats:
            if len(caveat) <= 140:
                notes.append(caveat)
                break
        reason = f"{getattr(candidate, 'name', key)}: {getattr(candidate, 'when_to_use', '')}".strip()
    except Exception:
        registry_key, reason = "vanilla_pinn", "the plain PINN, the safest first model"

    try:
        from pinneapple_neural.architectures.registry import ModelRegistry

        available = set(ModelRegistry.list())
        if registry_key not in available:
            registry_key = "vanilla_pinn" if "vanilla_pinn" in available else sorted(available)[0]
    except Exception:
        pass
    if method in ("analytic", "reference"):
        notes.append("train a model only after the classical answer looks right: it is the reference "
                     "that a trained model is scored against.")
    return registry_key, reason or "the safest first model", notes


def recommend(text: str, *, top: int = 3) -> Recommendation:
    """Turn a problem description into a preset, a method and a model, with the reasons.

    Accepts English or Portuguese. Runs offline: no model is trained and nothing is downloaded.

    >>> import pinneapple as pp
    >>> r = pp.recommend("heat sink cooled by a fan")
    >>> r.preset
    'fan_cooler_cfd'
    >>> print(r.explain())
    """
    text = str(text or "").strip()
    if not text:
        raise ValueError("describe the problem in a few words, e.g. pp.recommend('heat sink cooled by a fan')")

    hits = search(text, top=max(top + 1, 4))
    if not hits:
        return _no_match(text)

    # Prefer a preset that produces a field on this machine. A closer match the user cannot run is
    # worth less than the next one down, and the swap is stated rather than hidden.
    swapped_from = None
    chosen = hits[0]
    if not runnable(chosen["name"]):
        for hit in hits[1:]:
            if runnable(hit["name"]):
                swapped_from, chosen = chosen, hit
                hits = [chosen] + [h for h in hits if h is not chosen]
                break

    best = chosen
    name = best["name"]
    whites = card(name, with_related=True)
    matched = best["matched"]

    wants_training = _has(text, _TRAIN_WORDS)
    # ``whites.method`` is already what this machine can run; the snippet and the reason follow it.
    method = whites.method
    if wants_training and whites.available:
        method = "pinn"
    # Absolute, not relative: a single weak word is a weak recommendation however clearly it leads.
    raw = float(best.get("raw_score", 0.0))
    confidence = round(min(0.95, raw / (raw + 6.0)), 2)

    reason = whites.description
    if not whites.available:
        first = (whites.error or "it does not run here").splitlines()[0][:160]
        reason += f"  This preset does not produce a field on this machine yet: {first}"
        if whites.fallback:
            reason += ("  The program below is the same physics on a box you define here, which"
                       " trains anywhere.")
            code = whites.fallback
            method = "pinn"
        else:
            code = snippet(name, whites.method)
    elif wants_training:
        reason += "  You asked to train, so the method is a PINN rather than the classical solver."
        code = snippet(name, "pinn", train=True)
    elif whites.method == "pinn":
        reason += "  This preset has no classical solver, so the method is a trained network."
        code = snippet(name, "pinn", train=True)
    else:
        reason += f"  The '{whites.method}' method answers this preset without training anything."
        code = snippet(name, whites.method)

    model, model_reason, notes = _model_for(text, has_solver=method in ("analytic", "reference"), method=method)
    short = model_reason.split(". ")[0].strip()
    notes = [f"model {model}: {short}"] + notes
    if swapped_from is not None:
        notes.insert(0, f"the closest match was {swapped_from['name']}, which cannot produce a field on "
                        f"this machine yet ({swapped_from['description']}); the recommendation below "
                        f"is the closest one that runs.")
    if not whites.available:
        runnable_names = [h["name"] for h in hits[1:] if runnable(h["name"])]
        if runnable_names:
            notes.append("presets on this machine that run now: " + ", ".join(runnable_names))
        else:
            notes.append("no close preset runs on this machine without its solver or its geometry; "
                         "pp.doctor('cfd') lists what is missing")
    if whites.tags and not (set(matched) & set(whites.tags)):
        notes.append("the match came from the name and the domain, not from the tags: check the "
                     "description before trusting it.")

    alternatives = [
        {"name": h["name"], "description": h["description"], "score": h["score"]}
        for h in hits[1:top]
    ]
    next_steps = [
        f"pp.card({name!r})   # what it takes and what it gives back",
        f"pp.solve({name!r}, method={method!r})",
        "pp.trust_card(...)   # before you believe the numbers",
        f"pp.new('my_study', preset={name!r})   # a project you can grow",
    ]
    return Recommendation(
        query=text, preset=name, method=method, model=model, confidence=confidence,
        reason=reason, snippet=code, matched=matched, alternatives=alternatives,
        next_steps=next_steps, notes=notes,
        swapped_from=swapped_from["name"] if swapped_from else None,
    )


def _no_match(text: str) -> Recommendation:
    """Nothing in the catalogue is close. Say what was understood and where to go next."""
    kind, fields, coords = "heat_equation", ["u"], ["x", "t"]
    notes: List[str] = []
    understood = False
    try:
        from pinneapple_physics.pde_environment.capabilities import suggest_problem_spec

        guess = suggest_problem_spec(text)
        suggested = str(guess.get("kind") or "custom")
        if suggested in _COMPILER_KINDS:
            kind = suggested
            fields = list(guess.get("fields") or fields)
            coords = list(guess.get("coords") or coords)
            understood = True
            notes.append(f"the PDE knowledge base reads this as a '{kind}' problem in "
                         f"{', '.join(coords)} with fields {', '.join(fields)}.")
        elif suggested != "custom":
            notes.append(f"the PDE knowledge base reads this as a '{suggested}' problem, which the "
                         f"symbolic compiler does not train yet; the example below is a heat equation "
                         f"to replace.")
    except Exception:
        pass
    notes.append("no preset is close, so start from your own description instead of bending one of ours.")
    if not understood:
        notes.append("the program below is the smallest problem that trains: replace the equation, the "
                     "domain and the conditions with yours.")
    axes = ", ".join(f"{c}=(0.0, 1.0)" for c in coords)
    field = fields[0] if fields else "u"
    code = "\n".join([
        "import numpy as np",
        "import pinneapple as pp",
        "from pinneapple_physics import ProblemBuilder",
        "",
        f"# a '{kind}' problem; replace the equation, the domain and the conditions with yours",
        "spec = (ProblemBuilder('my_problem')",
        f"        .domain({axes})",
        f"        .fields({', '.join(repr(f) for f in fields)})",
        f"        .pde({kind!r}, alpha=0.01)             # <- your equation and its coefficients",
        f"        .ic(field='{field}', fn=lambda X: np.sin(np.pi * X[:, 0:1]))   # <- your initial state",
        f"        .bc('dirichlet', field='{field}', value=0.0, on='x_boundary')  # <- your boundaries",
        "        .build())",
        "sol = pp.solve(spec, method='pinn', epochs=2000)",
        "print(sol.metadata())",
        "print(sol.predict(np.array([[0.5, 0.5]])))",
    ])
    return Recommendation(
        query=text, preset=None, method="pinn", model="vanilla_pinn", confidence=0.0,
        reason="nothing in the catalogue matched the words in the description",
        snippet=code, matched=[],
        next_steps=["pp.new('my_study', template='plain')  # a project with a case file to fill in",
                    "pp.list_presets()                     # the 72 presets, by name",
                    "docs/getting_started/first_pinn.md   # the five objects every run connects"],
        notes=notes,
    )
