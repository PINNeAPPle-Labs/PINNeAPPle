"""O relatório de decisão a partir de um ensemble: distribuição, risco por restrição, o caso que dirige o risco e a
recomendação com o seu domínio de validade (#219, roadmap S38).

Uma predição é um número; uma decisão é um número com a probabilidade de ele estar errado do lado que importa. Este
módulo transforma um ensemble -- as mesmas realizações que a camada de incerteza (#80) e o domínio de validade (#58)
já sabem produzir -- no que um engenheiro assina embaixo:

``distribution``
    a distribuição da quantidade de interesse: média, desvio, mínimo, mediana, máximo e os quantis que o cliente
    pergunta (5 %, 95 %). Nada de um número só: a largura é a resposta, não um detalhe dela.
``probabilities``
    a **probabilidade de violar cada restrição**, e ela não é uma estatística nova: é a fração de realizações em que
    :func:`pinneapple_physics.constraint_engine.check` reprovou aquele mesmo ``pp.Constraint`` (#221). Restrição dura e
    restrição branda aparecem separadas, porque uma reprova o projeto e a outra só avisa.
``drivers`` / ``combinations``
    o que dirige o risco: os casos individuais com a maior violação, e -- mais útil -- os **intervalos de entrada**
    que separam quem viola de quem não viola (``P <= 14.2 W``, ``h >= 24 W/m2K``), com a correlação que justifica
    dizer isso em vez de mostrar uma nuvem de pontos.
``recommendation``
    o ponto do envelope com a maior folga, a faixa segura por variável derivada dos violadores, e o *status de
    confiança* do relatório (:class:`pinneapple_analysis.trust.TrustReport`, S8) -- que é uma afirmação sobre a
    evidência, não sobre a física, e por isso vem separado das probabilidades.
``domain``
    o envelope em que tudo isso foi medido (:class:`pinneapple_analysis.benchmark.validity.ValidityDomain`, #58). Fora
    dele o relatório não vale, e é por isso que ele é um campo e não uma nota de rodapé.

O formato é o que se copia para uma decisão; a simulação social que inspirou a issue não interessa a um problema de
física, e por isso aqui não há cenários narrativos, só números com procedência.

Uso
---
>>> report = decision_report(
...     case="placa", qoi="T_max", domain=BOX, sample=draw, evaluate=fv_model,
...     constraints=[T_MAX], reference=fine_grid)
>>> round(report.probabilities["T_max <= 85 degC"], 3)
0.118
>>> print(report.to_markdown())          # a página que vai para a decisão
"""
from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from pinneapple_analysis.trust.trust_report import (
    APPROVED,
    CONTRADICTS,
    NOT_RUN,
    SUPPORTS,
    Check,
    TrustReport,
)
from pinneapple_physics.constraint_engine import ConstraintReport, check_all

if TYPE_CHECKING:                       # the benchmark package pulls torch in; a decision report should not
    from pinneapple_analysis.benchmark.validity import ValidityDomain  # noqa: F401  (typing only)

__all__ = ["EnsembleMember", "InputDriver", "Combination", "DecisionReport", "decision_report"]

#: a variable drives the risk when |correlation| between it and the constraint margin reaches this. It is a
#: convention (a number below which a monotone trend is not named in the recommendation), not a measured threshold.
DRIVER_CORRELATION = 0.30


@dataclass(frozen=True)
class EnsembleMember:
    """One realization: the inputs, the quantity of interest and the verdict of every constraint on it."""
    inputs: dict[str, float]
    qoi: dict[str, float]
    report: ConstraintReport

    @property
    def worst_margin(self) -> float:
        """The margin of the constraint closest to -- or furthest past -- its limit. Negative is slack."""
        return max((c.margin for c in self.report.checks), default=float("-inf"))

    def to_dict(self) -> dict[str, Any]:
        return {"inputs": dict(self.inputs), "qoi": dict(self.qoi), "worst_margin": self.worst_margin,
                "violated": [c.name for c in self.report.violated]}


@dataclass(frozen=True)
class InputDriver:
    """An input variable that separates the realizations that violate from those that do not.

    ``safe_side`` is the direction the recommendation states: ``"<"`` means "stay below ``threshold``", ``">"`` the
    other way, and the threshold is the edge of the observed violators -- so the side it names has **no** observed
    violation by construction, which is the only thing the ensemble can promise. ``correlation`` is the Pearson
    correlation between the variable and the constraint margin: a margin that rises with the variable is what makes
    the threshold a statement about risk rather than about this sample.
    """
    variable: str
    constraint: str
    correlation: float
    safe_side: str
    threshold: float
    violation_rate_risky: float
    violation_rate_safe: float


@dataclass(frozen=True)
class Combination:
    """One cell of the input space, with the fraction of its realizations that violate the constraint.

    A single variable rarely causes the violation on its own: the risk sits in a corner -- high load *and* low
    cooling. The cell is the smallest statement that names a combination, cut at the median of each variable, and the
    report gives its size so a cell of three cases cannot be read as a rule.
    """
    label: str
    constraint: str
    n: int
    violation_rate: float
    variables: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "constraint": self.constraint, "n": self.n,
                "violation_rate": self.violation_rate, "variables": dict(self.variables)}


@dataclass
class DecisionReport:
    """What a decision needs from an ensemble, in one object: the spread, the risk, its drivers and the trust."""
    case: str
    qoi: str
    n: int
    domain: ValidityDomain
    distribution: dict[str, float] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    soft_probabilities: dict[str, float] = field(default_factory=dict)
    drivers: list[EnsembleMember] = field(default_factory=list)
    inputs: list[InputDriver] = field(default_factory=list)
    combinations: list[Combination] = field(default_factory=list)
    recommendation: dict[str, Any] = field(default_factory=dict)
    trust: TrustReport | None = None
    notes: list[str] = field(default_factory=list)

    @property
    def violated(self) -> dict[str, float]:
        """The hard constraints the ensemble can violate, with their probability."""
        return {k: v for k, v in self.probabilities.items() if v > 0.0}

    @property
    def verdict(self) -> str:
        """The recommendation in one word: does this design hold over the envelope it was measured on?"""
        if any(v > 0.0 for v in self.probabilities.values()):
            return "não opera no envelope inteiro"
        if self.trust is not None and self.trust.decisao != APPROVED:
            return f"opera, com a confiança em {self.trust.decisao}"
        return "opera no envelope"

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case, "qoi": self.qoi, "n": self.n, "verdict": self.verdict,
            "domain": self.domain.to_dict(), "distribution": dict(self.distribution),
            "probabilities": dict(self.probabilities), "soft_probabilities": dict(self.soft_probabilities),
            "drivers": [d.to_dict() for d in self.drivers],
            "inputs": [vars(d) for d in self.inputs],
            "combinations": [c.to_dict() for c in self.combinations],
            "recommendation": {k: v for k, v in self.recommendation.items() if k != "member"},
            "trust": None if self.trust is None else {"decisao": self.trust.decisao, "score": self.trust.score,
                                                      "cobertura": self.trust.cobertura,
                                                      "checks": [vars(c) for c in self.trust.checks]},
            "notes": list(self.notes),
        }

    def summary(self) -> str:
        p = ", ".join(f"P({k} violada) = {v:.1%}" for k, v in self.probabilities.items()) or "sem restrição dura"
        return (f"{self.case}: {self.qoi} = {self.distribution.get('mean', float('nan')):.4g} "
                f"+- {self.distribution.get('std', float('nan')):.3g} sobre {self.n} realizações -- {p} -- "
                f"{self.verdict}")

    def to_markdown(self) -> str:
        """The page that goes to the decision: numbers first, then who drives the risk, then what is recommended."""
        d = self.distribution
        lines = [f"# Relatório de decisão -- {self.case}", "",
                 f"**{self.qoi}** sobre {self.n} realizações no envelope declarado.", "",
                 "## Distribuição", "",
                 "| média | desvio | mín | p05 | mediana | p95 | máx |", "|---|---|---|---|---|---|---|",
                 f"| {d['mean']:.4g} | {d['std']:.4g} | {d['min']:.4g} | {d['p05']:.4g} | "
                 f"{d['median']:.4g} | {d['p95']:.4g} | {d['max']:.4g} |", "",
                 "## Probabilidade de violar cada restrição", "",
                 "| restrição | família | dura | P(viola) | limite |", "|---|---|---|---|---|"]
        by_name = {c.name: c for c in self.report_checks()}
        for name, p in self.probabilities.items():
            c = by_name.get(name)
            lines.append(f"| {name} | {getattr(c, 'family', '')} | sim | {p:.2%} | {_limit(c)} |")
        for name, p in self.soft_probabilities.items():
            lines.append(f"| {name} | | não | {p:.2%} | |")
        lines += ["", "## O que dirige o risco", ""]
        if self.combinations:
            lines += ["| combinação de entrada | restrição | casos | P(viola) |", "|---|---|---|---|"]
            for comb in self.combinations:
                lines.append(f"| {comb.label} | {comb.constraint} | {comb.n} | {comb.violation_rate:.1%} |")
            lines.append("")
        if self.inputs:
            for drv in self.inputs:
                lines.append(f"- **{drv.variable} {drv.safe_side} {drv.threshold:.4g}** -- correlação "
                             f"{drv.correlation:+.2f} com a folga de {drv.constraint!r}; viola "
                             f"{drv.violation_rate_risky:.1%} do lado arriscado e {drv.violation_rate_safe:.1%} do "
                             f"lado seguro.")
        elif not self.combinations:
            lines.append("- nenhuma variável de entrada separa quem viola de quem não viola neste ensemble.")
        if self.drivers:
            lines += ["", "Piores casos:", "", "| # | entradas | " + f"{self.qoi} | margem |", "|---|---|---|---|"]
            for i, m in enumerate(self.drivers, 1):
                ent = ", ".join(f"{k}={v:.4g}" for k, v in m.inputs.items())
                lines.append(f"| {i} | {ent} | {m.qoi[self.qoi]:.4g} | {m.worst_margin:+.4g} |")
        lines += ["", "## Recomendação", "", self.recommendation.get("text", "sem recomendação"), "",
                  f"**Validade:** {self.domain.describe()}", ""]
        if self.trust is not None:
            lines += [f"**Confiança (S8):** {self.trust.decisao} -- score {self.trust.score}, cobertura "
                      f"{self.trust.cobertura:.0%}", ""]
            for c in self.trust.checks:
                lines.append(f"- {c.name}: {c.status}" + (f" -- {c.reason}" if c.reason else ""))
        if self.notes:
            lines += ["", "## Notas", ""] + [f"- {n}" for n in self.notes]
        return "\n".join(lines) + "\n"

    def report_checks(self):
        """The per-constraint checks of the worst realization, which is where the limits and families live."""
        return list(self.drivers[0].report.checks) if self.drivers else []


def _limit(c: Any) -> str:
    """The limit of a constraint as it appears in the table, or empty when the constraint has no single limit."""
    return "" if c is None or getattr(c, "limit", None) is None else f"{c.limit:g}"


def _draw(sample: Any, n: int, seed: int) -> dict[str, np.ndarray]:
    """``sample`` is either a callable ``(n, rng)`` or ``(rng, n)`` -- both are in the wild; the signature decides."""
    rng = np.random.default_rng(seed)
    try:
        params = list(inspect.signature(sample).parameters)
    except (TypeError, ValueError):
        params = []
    out = sample(rng, n) if params and params[0] == "rng" else sample(n, rng)
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def _evaluate(evaluate: Callable[[Mapping[str, Any]], Any], inputs: Mapping[str, Any]) -> dict[str, float]:
    """One realization: ``evaluate`` takes the input mapping and returns the quantity (or quantities) of interest."""
    out = evaluate(inputs)
    if isinstance(out, Mapping):
        return {k: float(np.asarray(v).ravel()[0]) for k, v in out.items()}
    return {"qoi": float(np.asarray(out).ravel()[0])}


def _distribution(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=float)
    q = np.percentile(v, [5, 50, 95])
    return {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max()), "p05": float(q[0]), "median": float(q[1]),
            "p95": float(q[2]), "n": int(v.size)}


def _correlations(members: Sequence[EnsembleMember], variables: Sequence[str],
                  constraint: str) -> dict[str, float]:
    """Pearson correlation between each input variable and the margin of ``constraint`` over the ensemble."""
    margins = np.array([m.report.by_name()[constraint].margin for m in members], dtype=float)
    out: dict[str, float] = {}
    for v in variables:
        x = np.array([m.inputs[v] for m in members], dtype=float)
        if x.std() == 0 or margins.std() == 0:
            continue
        corr = float(np.corrcoef(x, margins)[0, 1])
        if np.isfinite(corr):
            out[v] = corr
    return out


def _input_drivers(members: Sequence[EnsembleMember], variables: Sequence[str], constraint: str,
                   correlations: Mapping[str, float]) -> list[InputDriver]:
    """Which inputs separate the violating realizations from the others, and on which side.

    Only variables whose correlation with the constraint margin reaches :data:`DRIVER_CORRELATION` are named: a
    threshold read off a scatter with no trend would be a coincidence dressed as a rule. The margin rises towards
    violation (``> 0`` is a violation), so a positive correlation puts the violators at the top of the variable and
    makes ``"< threshold"`` the statement that carries no observed violation -- and the mirror image for a negative
    one. The threshold is the edge of the observed violators, so it is a statement about this ensemble, which is why
    the report also prints the two rates it separates.
    """
    violated = np.array([m.report.by_name()[constraint].violated for m in members])
    if not violated.any() or violated.all():
        return []                                   # nothing separates: every case is on the same side
    out = []
    for v, corr in correlations.items():
        if abs(corr) < DRIVER_CORRELATION:
            continue
        x = np.array([m.inputs[v] for m in members], dtype=float)
        hot = x[violated]
        side, threshold = ("<", float(hot.min())) if corr > 0 else (">", float(hot.max()))
        risky = x >= threshold if corr > 0 else x <= threshold
        out.append(InputDriver(variable=v, constraint=constraint, correlation=corr, safe_side=side,
                               threshold=threshold,
                               violation_rate_risky=float(violated[risky].mean()) if risky.any() else 0.0,
                               violation_rate_safe=float(violated[~risky].mean()) if (~risky).any() else 0.0))
    out.sort(key=lambda d: -abs(d.correlation))
    return out


def _combinations(members: Sequence[EnsembleMember], constraint: str, correlations: Mapping[str, float],
                  top: int = 2) -> list[Combination]:
    """The cells of the input space, cut at the median of the ``top`` most influential variables.

    A single variable rarely causes the violation on its own -- the risk sits in a corner, high load *and* low
    cooling -- so this is the smallest statement that names a combination. Cells are sorted by violation rate, and
    the size of each is kept: a cell of three cases is not a rule.
    """
    variables = [v for v, _ in sorted(correlations.items(), key=lambda kv: -abs(kv[1]))][:top]
    if not variables:
        return []
    violated = np.array([m.report.by_name()[constraint].violated for m in members])
    cuts = {v: float(np.median([m.inputs[v] for m in members])) for v in variables}
    cells: dict[tuple[tuple[str, str], ...], list[bool]] = {}
    for i, m in enumerate(members):
        key = tuple((v, "<=" if m.inputs[v] <= cuts[v] else ">") for v in variables)
        cells.setdefault(key, []).append(bool(violated[i]))
    out = [Combination(label=" e ".join(f"{v} {side} {cuts[v]:.4g}" for v, side in key), constraint=constraint,
                       n=len(vals), violation_rate=float(np.mean(vals)), variables=dict(key))
           for key, vals in cells.items()]
    out.sort(key=lambda c: (-c.violation_rate, -c.n))
    return out


def _recommendation(members: Sequence[EnsembleMember], inputs: Sequence[InputDriver],
                    combinations: Sequence[Combination], qoi: str, probabilities: Mapping[str, float],
                    n: int) -> dict[str, Any]:
    """The point with the most slack, the envelope that holds per constraint, and a sentence saying both.

    ``worst_margin`` is the *largest* of the margins of a realization, so the most slack is its **minimum**: minimizing
    it picks the point furthest inside every limit at once, not the one that violates the most.
    """
    best = min(members, key=lambda m: m.worst_margin)
    text = ["Ponto recomendado: " + ", ".join(f"{k} = {v:.4g}" for k, v in best.inputs.items()) +
            f" -- {qoi} = {best.qoi[qoi]:.4g}, folga mínima {best.worst_margin:+.4g}."]
    violated = {k: p for k, p in probabilities.items() if p > 0}
    if not violated:
        text.append(f"Nenhuma restrição dura é violada em {n} realizações: o envelope inteiro é operável, dentro do "
                    f"domínio de validade declarado.")
    else:
        text.append("Nem todo o envelope é operável: " +
                    "; ".join(f"P({k} violada) = {p:.1%}" for k, p in violated.items()) + ".")
        if combinations:
            worst = combinations[0]
            text.append(f"O risco se concentra em {worst.label} ({worst.n} casos, "
                        f"{worst.violation_rate:.1%} de violação em {worst.constraint!r}), contra "
                        f"{probabilities[worst.constraint]:.1%} no ensemble inteiro.")
        for d in inputs[:2]:
            text.append(f"Para {d.constraint!r}, manter {d.variable} {d.safe_side} {d.threshold:.4g} não deixa "
                        f"nenhuma violação observada (correlação {d.correlation:+.2f} entre a variável e a folga).")
        if not combinations and not inputs:
            text.append("Nenhuma variável isolada separa os violadores: a restrição é violada por combinações de "
                        "entrada que o tamanho do ensemble não resolve, e o envelope operável tem de ser reduzido "
                        "por outro critério.")
    return {"point": dict(best.inputs), "qoi": float(best.qoi[qoi]), "slack": float(best.worst_margin),
            "envelope": [vars(d) for d in inputs], "text": " ".join(text), "member": best}


def decision_report(case: str, qoi: str, domain: ValidityDomain, sample: Callable[..., Mapping[str, Any]],
                    evaluate: Callable[[Mapping[str, Any]], Any], constraints: Sequence[Any], *,
                    reference: Callable[[Mapping[str, Any]], Any] | None = None, n: int = 200, seed: int = 0,
                    tolerance: float = 0.05, top_k: int = 5, trust_check: bool = True,
                    min_members: int = 100) -> DecisionReport:
    """Run the ensemble over the declared envelope and turn it into the decision report.

    ``sample(n, rng) -> {variable: array}`` draws the uncertain inputs; ``evaluate(inputs) -> value or {qoi: value}``
    produces the quantity of interest for one of them; ``constraints`` are ``pp.Constraint`` objects (#221). When
    ``reference`` is given, the same inputs are evaluated with it on a subsample and the disagreement becomes a trust
    check (#S8) -- a probability of violation measured on a model nobody validated does not deserve the same reading
    as one that was.

    ``n`` is the number of draws, and it is also the resolution of every probability in the report: with 200 draws a
    probability of 0 is "below 0.5 %", not "impossible", and the trust checks carry that statement rather than leaving
    it to the reader.

    Every draw is also tested against ``domain``: a member outside the declared envelope is an extrapolation, and the
    probabilities are computed anyway but the report says how many were -- and the recommendation cannot claim more
    than the domain that was measured.
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if min_members < 1:
        raise ValueError("min_members must be at least 1")

    inputs = _draw(sample, n, seed)
    variables = list(inputs)
    members: list[EnsembleMember] = []
    outside = 0
    for i in range(n):
        one = {v: float(inputs[v][i]) for v in variables}
        outside += 0 if domain.contains(one) else 1
        qoi_values = _evaluate(evaluate, one)
        members.append(EnsembleMember(inputs=one, qoi=qoi_values, report=check_all(constraints, qoi_values)))

    hard = [c for c in constraints if getattr(c, "hard", True)]
    soft = [c for c in constraints if not getattr(c, "hard", True)]
    probabilities = {c.name: float(np.mean([m.report.by_name()[c.name].violated for m in members])) for c in hard}
    soft_probabilities = {c.name: float(np.mean([m.report.by_name()[c.name].violated for m in members]))
                          for c in soft}

    series = np.array([m.qoi[qoi] for m in members], dtype=float)
    order = np.argsort([-m.worst_margin for m in members])
    drivers = [members[i] for i in order[:max(1, top_k)]]

    # every hard constraint the ensemble can violate gets its own drivers: which inputs separate the violators is a
    # property of the limit, and a tighter limit can be driven by other variables than a looser one.
    variable_drivers: list[InputDriver] = []
    combinations: list[Combination] = []
    for c in sorted(hard, key=lambda c: -probabilities[c.name]):
        if probabilities[c.name] <= 0.0:
            continue
        corr = _correlations(members, variables, c.name)
        variable_drivers += _input_drivers(members, variables, c.name, corr)
        combinations += _combinations(members, c.name, corr)
    variable_drivers.sort(key=lambda d: -abs(d.correlation))
    combinations.sort(key=lambda c: (-c.violation_rate, -c.n))

    report = DecisionReport(
        case=case, qoi=qoi, n=n, domain=domain, distribution=_distribution(series),
        probabilities=probabilities, soft_probabilities=soft_probabilities, drivers=drivers,
        inputs=variable_drivers, combinations=combinations,
        recommendation=_recommendation(members, variable_drivers, combinations, qoi, probabilities, n),
    )
    if outside:
        report.notes.append(f"{outside}/{n} realizações caem fora do domínio de validade declarado: nelas os números "
                            f"acima são extrapolação, não medição.")
    if trust_check:
        report.trust = _trust(case, domain, sample, evaluate, reference, sseries=series,
                              n=n, seed=seed, tolerance=tolerance, outside=outside, min_members=min_members)
    return report


def _trust(case: str, domain: ValidityDomain, sample: Callable[..., Mapping[str, Any]],
           evaluate: Callable[[Mapping[str, Any]], Any], reference: Callable[[Mapping[str, Any]], Any] | None,
           *, sseries: np.ndarray, n: int, seed: int, tolerance: float, outside: int,
           min_members: int) -> TrustReport:
    """The evidence behind the report, in the vocabulary of #S8: a claim about the numbers, never about the physics.

    The accuracy check evaluates a subsample with the reference as well. Its absence is recorded as ``not_run`` with
    the reason, not as a pass -- a report that cannot say how far the model is from the solver has no business
    printing an APPROVED.
    """
    checks: list[Check] = []
    if reference is None:
        checks.append(Check("accuracy against the reference", NOT_RUN, True,
                            reason="no reference was supplied, so the ensemble was not checked against the solver"))
    else:
        k = min(16, n)
        idx = np.linspace(0, n - 1, k).astype(int)
        drawn = _draw(sample, n, seed)
        errors = []
        for i in idx:
            one = {v: float(drawn[v][i]) for v in drawn}
            truth = _evaluate(reference, one)
            model = _evaluate(evaluate, one)
            (name, _), = list(model.items())[:1]
            denom = abs(truth[name]) if abs(truth[name]) > 0 else 1.0
            errors.append(abs(model[name] - truth[name]) / denom)
        mean_err = float(np.mean(errors))
        checks.append(Check("accuracy against the reference", SUPPORTS if mean_err <= tolerance else CONTRADICTS,
                            True, value=mean_err, limit=tolerance,
                            reason=f"mean relative error {mean_err:.3%} on {k} realizations against the reference"))
    checks.append(Check("inside the declared validity domain", SUPPORTS if outside == 0 else CONTRADICTS,
                        True, value=float(outside), limit=0.0,
                        reason="every realization is inside the envelope" if outside == 0
                        else f"{outside}/{n} realizations outside the envelope"))
    checks.append(Check("sample size", SUPPORTS if n >= min_members else CONTRADICTS, False,
                        value=float(n), limit=float(min_members),
                        reason=f"{n} realizations; probabilities below 1/{n} are not resolved by this sample"))
    checks.append(Check("spread of the quantity of interest", SUPPORTS if np.std(sseries) > 0 else CONTRADICTS,
                        False, value=float(np.std(sseries)), limit=0.0,
                        reason="the ensemble varies the answer, so the probabilities are drawn from a distribution"
                        if np.std(sseries) > 0 else "every realization gives the same value: nothing was sampled"))
    return TrustReport(checks, model=case)
