"""pinneapple_analysis.trust.trust_report — relatório de confiança para decidir o deploy de um modelo.

Junta checagens independentes (acurácia contra referência, consistência física, OOD, calibração de
incerteza, comparação com dado experimental) numa decisão única: APPROVED, REVIEW ou REJECT. Cada
checagem usa o mesmo vocabulário das provas do E11 do PINNeAPPle-CFD:

- ``supports``: a checagem passou no critério;
- ``contradicts``: a checagem falhou no critério;
- ``not_run``: a checagem não foi executada (com o motivo). Não entra no score, mas conta na cobertura.

Regras (escolhas de projeto, documentadas e revogáveis, não normas):

1. Qualquer checagem ``critical`` com ``contradicts`` → REJECT.
2. Qualquer outra ``contradicts`` → REVIEW.
3. Cobertura (checagens executadas / total) abaixo de ``min_coverage`` → REVIEW.
4. Score (média ponderada das checagens executadas) abaixo de ``min_score`` → REVIEW.
5. Caso contrário → APPROVED.

``from_trust_score`` converte o ``TrustScore`` do ``TrustGate`` (OOD, resíduo, ensemble) em checagens.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

SUPPORTS = "supports"
CONTRADICTS = "contradicts"
NOT_RUN = "not_run"
_STATUS = (SUPPORTS, CONTRADICTS, NOT_RUN)

APPROVED = "APPROVED"
REVIEW = "REVIEW"
REJECT = "REJECT"


@dataclass(frozen=True)
class Check:
    name: str
    status: str
    critical: bool = False
    weight: float = 1.0
    value: Optional[float] = None
    limit: Optional[float] = None
    source: str = ""
    reason: str = ""

    def __post_init__(self) -> None:
        if self.status not in _STATUS:
            raise ValueError(f"invalid status: {self.status!r} (use {_STATUS})")
        if self.weight < 0:
            raise ValueError("weight não pode ser negativo")


@dataclass
class TrustReport:
    checks: List[Check]
    min_coverage: float = 0.6
    min_score: float = 0.8
    model: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def executadas(self) -> List[Check]:
        return [c for c in self.checks if c.status != NOT_RUN]

    @property
    def cobertura(self) -> float:
        return len(self.executadas) / len(self.checks) if self.checks else 0.0

    @property
    def score(self) -> Optional[float]:
        """Média ponderada das checagens executadas (supports = 1, contradicts = 0). None se nenhuma foi executada."""
        ex = self.executadas
        peso = sum(c.weight for c in ex)
        if not ex or peso == 0:
            return None
        return sum(c.weight for c in ex if c.status == SUPPORTS) / peso

    @property
    def decisao(self) -> str:
        if any(c.critical and c.status == CONTRADICTS for c in self.checks):
            return REJECT
        if any(c.status == CONTRADICTS for c in self.checks):
            return REVIEW
        if self.cobertura < self.min_coverage:
            return REVIEW
        s = self.score
        if s is None or s < self.min_score:
            return REVIEW
        return APPROVED

    @property
    def motivos(self) -> List[str]:
        out = []
        for c in self.checks:
            if c.status == CONTRADICTS:
                out.append(f"{c.name}: falhou" + (" (crítica)" if c.critical else "") + (f" — {c.reason}" if c.reason else ""))
            elif c.status == NOT_RUN:
                out.append(f"{c.name}: não executada" + (f" — {c.reason}" if c.reason else ""))
        if self.cobertura < self.min_coverage:
            out.append(f"cobertura {self.cobertura:.0%} abaixo do mínimo de {self.min_coverage:.0%}")
        if self.score is not None and self.score < self.min_score:
            out.append(f"score {self.score:.2f} abaixo do mínimo de {self.min_score:.2f}")
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modelo": self.model,
            "decisao": self.decisao,
            "score": self.score,
            "cobertura": self.cobertura,
            "motivos": self.motivos,
            "checagens": [c.__dict__ for c in self.checks],
            "min_coverage": self.min_coverage,
            "min_score": self.min_score,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrustReport":
        """Inverse of ``to_dict`` (the decision is recomputed from the checks, not copied)."""
        return cls(checks=[Check(**c) for c in d.get("checagens", [])], min_coverage=d.get("min_coverage", 0.6),
                   min_score=d.get("min_score", 0.8), model=d.get("modelo", ""))

    def to_markdown(self) -> str:
        s = self.score
        linhas = [
            f"# Relatório de confiança — {self.model or 'modelo'}",
            "",
            f"**Decisão:** {self.decisao}  ",
            f"**Score:** {'—' if s is None else f'{s:.2f}'}  ·  **Cobertura:** {self.cobertura:.0%}",
            "",
            "| Checagem | Status | Crítica | Valor | Limite | Fonte |",
            "|---|---|---|---|---|---|",
        ]
        for c in self.checks:
            val = "—" if c.value is None else f"{c.value:g}"
            lim = "—" if c.limit is None else f"{c.limit:g}"
            linhas.append(f"| {c.name} | {c.status} | {'sim' if c.critical else 'não'} | {val} | {lim} | {c.source or '—'} |")
        if self.motivos:
            linhas += ["", "## Motivos", ""] + [f"- {m}" for m in self.motivos]
        return "\n".join(linhas) + "\n"


def _status_por_limite(valor: Optional[float], limite: float, maior_e_melhor: bool) -> str:
    if valor is None:
        return NOT_RUN
    ok = valor >= limite if maior_e_melhor else valor <= limite
    return SUPPORTS if ok else CONTRADICTS


def checagem_erro_relativo(nome: str, erro: Optional[float], limite: float, *, fonte: str = "", critica: bool = False,
                           peso: float = 1.0, motivo_nao_executada: str = "") -> Check:
    """Acurácia contra referência: erro relativo menor ou igual ao limite."""
    st = _status_por_limite(erro, limite, maior_e_melhor=False)
    return Check(nome, st, critical=critica, weight=peso, value=erro, limit=limite, source=fonte,
                 reason="" if erro is not None else motivo_nao_executada or "sem medição")


def checagem_booleana(nome: str, passou: Optional[bool], *, fonte: str = "", critica: bool = False, peso: float = 1.0,
                      motivo_nao_executada: str = "") -> Check:
    """Consistência física (conservação, condições de contorno, simetria) ou comparação qualitativa."""
    if passou is None:
        return Check(nome, NOT_RUN, critical=critica, weight=peso, source=fonte, reason=motivo_nao_executada or "não avaliada")
    return Check(nome, SUPPORTS if passou else CONTRADICTS, critical=critica, weight=peso, source=fonte)


def from_trust_score(ts: Any, *, limite_sub_score: float = 0.7, fonte: str = "TrustGate") -> List[Check]:
    """Converte o ``TrustScore`` do ``TrustGate`` em checagens (OOD, resíduo e ensemble).

    Um sub-score ``None`` vira ``not_run`` com o motivo. ``limite_sub_score`` é uma escolha de projeto.
    """
    checks = []
    for nome, valor, critica in (("ood", ts.ood_score, True), ("residuo_pde", ts.residual_score, True),
                                 ("ensemble_incerteza", ts.ensemble_score, False)):
        st = _status_por_limite(valor, limite_sub_score, maior_e_melhor=True)
        checks.append(Check(nome, st, critical=critica, value=valor, limit=limite_sub_score, source=fonte,
                            reason="" if valor is not None else "sub-score não calculado"))
    return checks
