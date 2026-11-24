"""Três tipos de exigência, checados numa ordem fixa, e o registro de quem rejeitou cada projeto (#172).

Um projeto gerado é aprovado ou recusado por exigências de naturezas diferentes, e tratá-las como uma pilha só é como
um projeto reprovado chega ao solver para depois ser jogado fora:

* **restrições duras** (espessura > 2 mm, massa < 50 kg, folga > 5 mm, T < Tmax): nunca podem ser violadas, e são
  decididas sobre a geometria — antes de qualquer corrida;
* **restrições de física** (conservação de massa e de energia, condições de contorno, simetria): só podem ser ditas
  sobre uma solução;
* **objetivos suaves** (arrasto, sustentação, massa, eficiência térmica): medidos e ordenados, nunca impostos.

A ordem é a mesma para todo projeto e é o tipo que a determina, não o chamador: ``geometry`` → ``mesh`` →
``convergence`` → ``physics`` → ``objectives``. Uma falha para o projeto naquela etapa e a etapa fica registrada, com
a exigência que falhou.

>>> from pinneapple_design.design_gate import DesignGate, Requirement
>>> gate = DesignGate([Requirement.hard("thickness_mm >= 2.0", name="espessura"),
...                    Requirement.objective("cd", sense="min")])
>>> d = gate.evaluate({"thickness_mm": 1.4, "cd": 0.011}, run=lambda design: {"cd": 0.011})
>>> d.status, d.rejected_by
('rejected', 'geometry')

``run`` (o solver) só é chamado quando a geometria passa: é o que o aceite pede literalmente — um projeto que viola
uma restrição dura é recusado antes de qualquer corrida —, e ``DesignDecision.ran`` registra se ele chegou a rodar.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["STAGES", "KINDS", "Requirement", "RequirementResult", "StageRecord", "DesignDecision", "DesignGate",
           "summarise"]

#: a ordem fixa: geometria antes de malha, malha antes de solver, solver antes de física, física antes de objetivo
STAGES: tuple[str, ...] = ("geometry", "mesh", "convergence", "physics", "objectives")
KINDS: tuple[str, ...] = ("hard", "physics", "objective")

#: o tipo decide em que etapas a exigência pode ser checada — não é o chamador que escolhe. A restrição dura é
#: decidida antes de qualquer corrida (geometria e malha); a de física só pode ser dita sobre uma solução (convergência
#: e verificação física); o objetivo é medido no fim.
_STAGES_OF_KIND = {"hard": ("geometry", "mesh"), "physics": ("convergence", "physics"),
                   "objective": ("objectives",)}

PASS, FAIL, NOT_RUN, MEASURED = "pass", "fail", "not_run", "measured"
ACCEPTED, REJECTED, INCONCLUSIVE = "accepted", "rejected", "inconclusive"


@dataclass(frozen=True)
class Requirement:
    """Uma exigência, com o tipo e a etapa que o tipo permite.

    ``condition`` é uma expressão sobre os números do projeto (a mesma gramática das restrições do auditor de
    dataset: ``"thickness_mm >= 2"``, ``"mass_kg <= 50"``, ``"clearance_mm > 5"``) ou ``func(values)`` para o que
    não cabe numa expressão (a física medida, um objetivo que vem do solver). ``why`` guarda de onde a exigência
    veio — a correção que a criou, a cláusula de projeto — porque uma regra sem procedência é a primeira a ser
    ignorada.
    """

    name: str
    kind: str
    stage: str
    condition: str = ""
    func: Callable[[Mapping[str, Any]], Any] | None = None
    sense: str = "min"
    unit: str = ""
    why: str = ""

    def __post_init__(self) -> None:
        if not str(self.name).strip():
            raise ValueError("uma exigência precisa de nome: é por ele que o relatório diz o que reprovou")
        if self.kind not in KINDS:
            raise ValueError(f"kind deve ser um de {KINDS}, veio {self.kind!r}")
        if self.stage not in _STAGES_OF_KIND[self.kind]:
            raise ValueError(f"uma exigência {self.kind!r} é checada em {_STAGES_OF_KIND[self.kind]}, "
                             f"não em {self.stage!r}")
        if self.kind == "objective":
            if self.condition:
                raise ValueError(f"{self.name!r}: um objetivo é um número medido, não uma condição a satisfazer")
            if self.sense not in ("min", "max"):
                raise ValueError("um objetivo minimiza ('min') ou maximiza ('max')")
        elif bool(self.condition) == (self.func is not None):
            raise ValueError(f"{self.name!r}: dê a condição (expressão) ou a função, não as duas nem nenhuma")
        if self.condition:
            from pinneapple_data.expressions import names_in
            names_in(self.condition)                       # sintaxe errada falha aqui, não no projeto do cliente

    # -- construtores: o tipo no nome, a etapa amarrada a ele ---------------------------------------------
    @classmethod
    def hard(cls, condition: str, *, name: str = "", stage: str = "geometry", why: str = "") -> Requirement:
        return cls(name=name or condition, kind="hard", stage=stage, condition=condition, why=why)

    @classmethod
    def physics(cls, condition: str = "", *, func: Callable[[Mapping[str, Any]], Any] | None = None,
                name: str = "", stage: str = "physics", why: str = "") -> Requirement:
        """Uma exigência sobre a solução: ``stage="convergence"`` para o que diz se a corrida fechou,
        ``stage="physics"`` para o que diz se o resultado respeita a física."""
        return cls(name=name or condition or getattr(func, "__name__", "physics"), kind="physics", stage=stage,
                   condition=condition, func=func, why=why)

    @classmethod
    def objective(cls, name: str, *, sense: str = "min",
                  func: Callable[[Mapping[str, Any]], Any] | None = None, unit: str = "") -> Requirement:
        """Um alvo a otimizar, não a impor: o valor vem de ``func(values)`` ou da coluna ``name`` da solução."""
        return cls(name=name, kind="objective", stage="objectives", func=func, sense=sense, unit=unit)

    @classmethod
    def from_correction(cls, correction: Any, *, kind: str = "hard", stage: str = "") -> Requirement:
        """A correção de especialista do #173 como exigência: a regra já nasce checável, com a procedência junto.

        ``kind="hard"`` (o padrão) para uma regra sobre a geometria, decidida antes de rodar; ``kind="physics"``
        para uma regra que só pode ser dita sobre a solução."""
        return cls(name=f"correction {correction.rule_id}", kind=kind,
                   stage=stage or _STAGES_OF_KIND[kind][0], condition=correction.condition,
                   why=f"{correction.statement} ({correction.reviewer}, {correction.date})")

    # -- a checagem ---------------------------------------------------------------------------------------
    def check(self, values: Mapping[str, Any]) -> RequirementResult:
        if self.kind == "objective":
            return self._measure(values)
        if self.condition:
            from pinneapple_data.audit import constraint_mask
            from pinneapple_data.expressions import names_in
            missing = sorted(n for n in names_in(self.condition) if n not in values)
            if missing:
                return RequirementResult(self.name, self.kind, self.stage, NOT_RUN,
                                         reason=f"o projeto não traz {', '.join(missing)}")
            try:
                ok = bool(constraint_mask(self.condition, {n: [values[n]] for n in names_in(self.condition)})[0])
            except Exception as exc:  # noqa: BLE001 -- uma exigência que não pôde ser perguntada não passa
                return RequirementResult(self.name, self.kind, self.stage, NOT_RUN,
                                         reason=f"a condição {self.condition!r} não pôde ser avaliada: {exc}")
            return RequirementResult(self.name, self.kind, self.stage, PASS if ok else FAIL,
                                     reason="" if ok else f"{self.condition} não vale")
        try:
            out = self.func(values)                                 # type: ignore[misc]
        except Exception as exc:  # noqa: BLE001
            return RequirementResult(self.name, self.kind, self.stage, NOT_RUN, reason=f"a checagem falhou: {exc}")
        if out is None:
            return RequirementResult(self.name, self.kind, self.stage, NOT_RUN,
                                     reason="a checagem não pôde ser feita sobre esta solução")
        if isinstance(out, tuple):
            ok, reason = bool(out[0]), str(out[1]) if len(out) > 1 else ""
        elif hasattr(out, "passed"):                                # um FieldCheck / Check: o mesmo vocabulário
            ok, reason = out.passed, getattr(out, "detail", "") or getattr(out, "reason", "")
            if ok is None:
                return RequirementResult(self.name, self.kind, self.stage, NOT_RUN, reason=reason)
        else:
            ok, reason = bool(out), ""
        return RequirementResult(self.name, self.kind, self.stage, PASS if ok else FAIL, reason="" if ok else reason)

    def _measure(self, values: Mapping[str, Any]) -> RequirementResult:
        if self.func is not None:
            value = float(self.func(values))                         # type: ignore[misc]
        else:
            value = float(values.get(self.name, float("nan")))
        if value != value:
            return RequirementResult(self.name, self.kind, self.stage, NOT_RUN,
                                     reason=f"o projeto não traz o objetivo {self.name!r}")
        return RequirementResult(self.name, self.kind, self.stage, MEASURED, value=value, unit=self.unit)


@dataclass
class RequirementResult:
    name: str
    kind: str
    stage: str
    status: str
    value: float | None = None
    unit: str = ""
    reason: str = ""

    @property
    def failed(self) -> bool:
        return self.status == FAIL

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "kind": self.kind, "stage": self.stage, "status": self.status,
                "value": self.value, "unit": self.unit, "reason": self.reason}


@dataclass
class StageRecord:
    """O que aconteceu numa etapa: as exigências que rodaram, o veredito e o porquê."""

    stage: str
    status: str                       # "pass" | "fail" | "not_run" | "empty"
    results: list[RequirementResult] = field(default_factory=list)
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "status": self.status, "reason": self.reason,
                "results": [r.to_dict() for r in self.results]}


@dataclass
class DesignDecision:
    """O veredito de um projeto: aceito, recusado (com a etapa e a exigência) ou sem resposta (com o motivo)."""

    name: str
    status: str
    rejected_by: str = ""             # a etapa onde parou; "" quando aceito ou inconclusivo por falta de dado
    reason: str = ""
    records: list[StageRecord] = field(default_factory=list)
    objectives: dict[str, float] = field(default_factory=dict)
    ran: bool = False                 # se o solver chegou a rodar para este projeto

    @property
    def accepted(self) -> bool:
        return self.status == ACCEPTED

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "status": self.status, "rejected_by": self.rejected_by, "reason": self.reason,
                "ran": self.ran, "objectives": dict(self.objectives),
                "records": [r.to_dict() for r in self.records]}

    def table(self) -> str:
        """Uma linha por etapa, para o log da campanha."""
        out = [f"{self.name}: {self.status}" + (f" em {self.rejected_by}" if self.rejected_by else "")]
        for rec in self.records:
            mark = {"pass": "ok", "fail": "FAIL", "not_run": "sem resposta", "empty": "-"}[rec.status]
            out.append(f"  {rec.stage:<12} {mark}" + (f" -- {rec.reason}" if rec.reason else ""))
        if self.objectives:
            out.append("  objetivos: " + ", ".join(f"{k}={v:g}" for k, v in self.objectives.items()))
        return "\n".join(out)


class DesignGate:
    """As exigências de um projeto, agrupadas pelas etapas, na ordem fixa.

    ``requirements`` de um mesmo tipo e etapa rodam juntos; a etapa reprova se qualquer exigência dela reprova. Sem
    ``run``, as etapas que precisam de solução ficam sem resposta em vez de passar: a malha e a física não são
    decididas sobre um projeto que ninguém rodou.
    """

    def __init__(self, requirements: Iterable[Requirement] = ()) -> None:
        self.requirements: list[Requirement] = []
        self.add(*requirements)

    def add(self, *requirements: Requirement) -> DesignGate:
        for r in requirements:
            if any(existing.name == r.name and existing.stage == r.stage for existing in self.requirements):
                raise ValueError(f"a exigência {r.name!r} já existe na etapa {r.stage!r}")
            self.requirements.append(r)
        return self

    @property
    def by_stage(self) -> dict[str, list[Requirement]]:
        out: dict[str, list[Requirement]] = {}
        for r in self.requirements:
            out.setdefault(r.stage, []).append(r)
        return out

    def evaluate(self, design: Mapping[str, Any], *, run: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None
                 = None, name: str = "design") -> DesignDecision:
        """Roda as etapas em ordem e para na primeira que reprova. ``run`` (o solver) só é chamado quando a
        geometria e a malha passaram."""
        records: list[StageRecord] = []
        values: dict[str, Any] = dict(design)
        solution: Mapping[str, Any] | None = None
        ran = False
        objectives: dict[str, float] = {}
        for stage in STAGES:
            reqs = self.by_stage.get(stage, [])
            if not reqs:
                records.append(StageRecord(stage, "empty"))
                continue
            if stage != "geometry" and solution is None and _needs_solution(reqs, values):
                if run is None:
                    results = [RequirementResult(r.name, r.kind, stage, NOT_RUN,
                                                 reason="sem solver: esta etapa só pode ser dita sobre uma solução")
                               for r in reqs]
                    records.append(StageRecord(stage, NOT_RUN, results,
                                               "a etapa não foi checada: nenhuma solução foi dada"))
                    return DesignDecision(name, INCONCLUSIVE, reason=records[-1].reason, records=records,
                                          objectives=objectives, ran=ran)
                solution = run(values)
                ran = True
                values = {**values, **solution}
            results = [r.check(values) for r in reqs]
            failed = [r for r in results if r.failed]
            unanswered = [r for r in results if r.status == NOT_RUN]
            if stage == "objectives":
                objectives = {r.name: r.value for r in results if r.value is not None}  # type: ignore[misc]
            if failed:
                records.append(StageRecord(stage, FAIL, results, f"{failed[0].name}: {failed[0].reason}".strip(": ")))
                return DesignDecision(name, REJECTED, rejected_by=stage, reason=records[-1].reason, records=records,
                                      objectives=objectives, ran=ran)
            if unanswered:
                records.append(StageRecord(stage, NOT_RUN, results,
                                           f"{unanswered[0].name}: {unanswered[0].reason}".strip(": ")))
                return DesignDecision(name, INCONCLUSIVE, rejected_by=stage, reason=records[-1].reason,
                                      records=records, objectives=objectives, ran=ran)
            records.append(StageRecord(stage, PASS, results))
        missing = [s for s in STAGES if s not in self.by_stage]
        return DesignDecision(name, ACCEPTED, records=records, objectives=objectives, ran=ran,
                              reason="; ".join(f"nada em {s}" for s in missing))

    def evaluate_all(self, designs: Iterable[Mapping[str, Any]], *,
                     run: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
                     names: Sequence[str] | None = None) -> list[DesignDecision]:
        """A campanha inteira: cada projeto com o veredito e a etapa que o parou."""
        designs = list(designs)
        labels = list(names) if names is not None else [f"design-{i:03d}" for i in range(len(designs))]
        return [self.evaluate(d, run=run, name=n) for d, n in zip(designs, labels, strict=True)]


def _needs_solution(reqs: Sequence[Requirement], values: Mapping[str, Any]) -> bool:
    """Se alguma exigência da etapa lê algo que o projeto ainda não traz (``func`` lê a solução por definição)."""
    from pinneapple_data.expressions import names_in

    for r in reqs:
        if r.func is not None:
            return True
        if r.condition and names_in(r.condition) - set(values):
            return True
    return False


def summarise(decisions: Sequence[DesignDecision]) -> dict[str, Any]:
    """Quantos projetos passaram, quantos foram recusados e por qual etapa (o registro da campanha)."""
    by_stage: dict[str, int] = {}
    by_requirement: dict[str, int] = {}
    for d in decisions:
        if d.rejected_by:
            by_stage[d.rejected_by] = by_stage.get(d.rejected_by, 0) + 1
            for rec in d.records:
                if rec.stage == d.rejected_by:
                    for r in rec.results:
                        if r.failed:
                            by_requirement[f"{d.rejected_by}/{r.name}"] = \
                                by_requirement.get(f"{d.rejected_by}/{r.name}", 0) + 1
    return {"total": len(decisions),
            "accepted": sum(d.accepted for d in decisions),
            "rejected": sum(d.status == REJECTED for d in decisions),
            "inconclusive": sum(d.status == INCONCLUSIVE for d in decisions),
            "rejected_by_stage": by_stage, "rejected_by_requirement": by_requirement,
            "ran": sum(d.ran for d in decisions)}
