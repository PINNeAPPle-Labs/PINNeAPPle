"""As parcelas da discrepância: física, erro numérico, erro de modelo e ruído de medida (#212).

``observação = física verdadeira + erro numérico + erro de modelo + ruído``. Até aqui a biblioteca tinha duas metades
disso: :mod:`pinneapple_analysis.verification.convergence` media o erro de discretização de um modelo (Richardson,
GCI) e :func:`pinneapple_analysis.verification.causal_discrepancy.fit_discrepancy_model` ajustava um ``delta(x)``
sobre o resíduo de um modelo de física — que é o erro numérico **mais** o de modelo **mais** o ruído, somados, sem
dizer quanto é cada um. Este módulo separa os quatro.

A separação não vem de um ajuste: vem de duas identidades medidas nos dados, e é por isso que ela pode ser conferida.

**Identidade 1 — a variância.** Com ``k`` réplicas da observação em cada um dos ``n`` pontos de projeto,

    MSE(modelo fino, observações) = viés²  +  ruído²

exatamente, onde o viés é ``média_r(y_ir) - u_fino`` e o ruído é a variância de **erro puro** dentro das réplicas
``SSE / (n(k-1))``. É a decomposição da soma de quadrados da ANOVA, escrita para este caso: a réplica não vê a física,
então o que ela espalha só pode ser medida. ``closure`` no resultado é essa identidade conferida — não uma tolerância
escolhida a dedo.

**Identidade 2 — o viés.** O viés contra o modelo mais fino se divide exatamente, ponto a ponto,

    média_r(y_ir) - u_fino  =  (u_extrapolado - u_fino)  +  (média_r(y_ir) - u_extrapolado)
                              \\_______numérico________/     \\________erro de modelo_______/

porque o termo do meio se cancela. ``u_extrapolado`` é a extrapolação de Richardson dos três refinamentos mais finos
(:func:`~pinneapple_analysis.verification.convergence.richardson_extrapolate`): o valor que o próprio modelo daria com
``h -> 0``. O primeiro termo é o erro de discretização do modelo que se usa; o segundo é o que sobra entre a física e o
dado, que é o "erro de modelo" no sentido de Kennedy & O'Hagan.

Cada parcela sai com **intervalo**:

* ruído — intervalo qui-quadrado sobre o desvio-padrão de erro puro, com os graus de liberdade que as réplicas dão;
* numérico — a banda de Roache (``GCI_fino``, fator de segurança ``Fs`` incluso) aplicada ao valor do modelo fino;
* modelo — a incerteza da média das réplicas (t de Student com a variância de erro puro, não a de cada ponto sozinho)
  combinada em quadratura com a banda numérica do extrapolado, porque ``u_extrapolado`` é ele mesmo incerto.

O que este módulo **não** faz, dito porque um benchmark encontra: ele não separa erro numérico de erro de modelo quando
os três refinamentos não estão na faixa assintótica (``fraction_asymptotic`` diz qual fração dos pontos está), e ele
trata uma grandeza de interesse escalar por ponto — um campo inteiro tem de ser reduzido a um escalar por ponto antes,
ou decomposto ponto a ponto.

O que ele devolve entra no relatório de confiança pelas duas checagens que são de fato checagens: a identidade da
variância fechou, e a extrapolação está na faixa assintótica.

Referência das fórmulas: Roache, P.J., "Perspective: A Method for Uniform Reporting of Grid Refinement Studies",
ASME J. Fluids Eng. 116(3), 405-413 (1994); Celik et al., "Procedure for Estimation and Reporting of Uncertainty
due to Discretization", ASME J. Fluids Eng. 130(7) (2008); Kennedy, M.C. & O'Hagan, A., "Bayesian Calibration of
Computer Models", J. R. Stat. Soc. B 63(3), 425-464 (2001).

    >>> import numpy as np
    >>> from pinneapple_analysis.verification.discrepancy_decomposition import decompose_discrepancy
    >>> x = np.linspace(0.1, 0.9, 9)
    >>> h = (0.2, 0.1, 0.05)                        # três malhas, fator 2 entre elas
    >>> def solve(xx, hh):                          # erro de discretização O(h²), solver determinístico
    ...     return np.sin(np.pi * xx) * (1.0 - hh ** 2)
    >>> rng = np.random.default_rng(7)
    >>> obs = np.sin(np.pi * x)[:, None] + 0.01 * rng.standard_normal((x.size, 5))
    >>> d = decompose_discrepancy(solve, x, obs, h)
    >>> bool(d.closure < 1e-12), d.components["noise"].covers(0.01)
    (True, True)
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import stats

from .convergence import richardson_extrapolate

__all__ = [
    "ComponentEstimate",
    "PointDecomposition",
    "DiscrepancyDecomposition",
    "decompose_discrepancy",
    "COMPONENTS",
]

#: As três parcelas medidas, na ordem em que a identidade do viés as soma.
COMPONENTS: tuple[str, ...] = ("numerical", "model_form", "noise")

_NOISE_METHOD = "variância de erro puro entre réplicas, com intervalo qui-quadrado"
_NUMERICAL_METHOD = "Richardson a três malhas; banda de Roache (GCI com fator de segurança) sobre o valor fino"
_MODEL_METHOD = ("média das réplicas contra o valor extrapolado; t de Student sobre a variância de erro puro, "
                 "combinada em quadratura com a banda numérica")

_CITATION = (
    "Roache, P.J., ASME J. Fluids Eng. 116(3), 405-413 (1994); Celik et al., ASME J. Fluids Eng. 130(7) (2008); "
    "Kennedy & O'Hagan, J. R. Stat. Soc. B 63(3), 425-464 (2001)."
)


@dataclass(frozen=True)
class ComponentEstimate:
    """Uma parcela, com o intervalo em que ela cai no nível de confiança pedido.

    ``value`` é a magnitude da parcela (raiz da média dos quadrados sobre os pontos de projeto, para viés; o
    desvio-padrão para o ruído). ``low``/``high`` vêm do intervalo de cada ponto levado à mesma raiz quadrática, então
    o intervalo é conservador em relação a um intervalo sobre a média dos módulos.
    """

    name: str
    kind: str  # "bias" | "noise"
    value: float
    low: float
    high: float
    confidence: float
    method: str
    assumptions: str

    @property
    def interval(self) -> tuple[float, float]:
        return (self.low, self.high)

    def covers(self, true_value: float) -> bool:
        """O valor injetado cai no intervalo? É a pergunta do aceite, feita explicitamente."""
        return bool(self.low <= float(true_value) <= self.high)

    def relative_half_width(self) -> float:
        if self.value <= 0.0:
            return float("inf")
        return 0.5 * (self.high - self.low) / self.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "kind": self.kind, "value": self.value,
            "low": self.low, "high": self.high, "confidence": self.confidence,
            "method": self.method, "assumptions": self.assumptions,
        }


@dataclass(frozen=True)
class PointDecomposition:
    """A decomposição de um ponto, onde as duas identidades são exatas (são contas, não ajustes)."""

    x: float
    observed_mean: float
    finest_model: float
    extrapolated: float
    numerical: float  # u_extrapolado - u_fino
    model_form: float  # média_r(y) - u_extrapolado
    bias: float  # média_r(y) - u_fino == numerical + model_form
    numerical_half_width: float
    model_form_half_width: float
    replicate_std: float
    observed_order: float
    is_asymptotic: bool

    def to_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class DiscrepancyDecomposition:
    """O resultado: as três parcelas com intervalo, a identidade conferida e o que a sustenta."""

    points: list[PointDecomposition]
    components: dict[str, ComponentEstimate]
    total_mse: float
    total_rmse: float
    bias_rms: float
    noise_std: float
    closure: float  # |MSE - (viés² + ruído²)| / MSE -- a identidade 1, medida
    bias_closure: float  # max_i |viés - (numérico + modelo)|, relativo -- a identidade 2, medida
    bias_correlation: float  # corr(numérico_i, modelo_i); a soma em quadratura só é exata com ela em 0
    fraction_asymptotic: float
    resolutions: tuple[float, ...]
    confidence: float
    citation: str = _CITATION

    def component(self, name: str) -> ComponentEstimate:
        if name not in self.components:
            raise KeyError(f"no component {name!r}; the decomposition has {sorted(self.components)}")
        return self.components[name]

    def covers(self, injected: Mapping[str, float]) -> dict[str, bool]:
        """Um booleano por parcela pedida: o valor injetado conhecido cai no intervalo declarado?"""
        return {name: self.component(name).covers(value) for name, value in injected.items()}

    def checks(self) -> list[Any]:
        """As duas perguntas que são de verdade checagem, no vocabulário do relatório de confiança.

        * ``discrepancy.closure`` — a identidade da variância fechou (é uma conta; se não fechou, há bug ou forma
          errada dos dados);
        * ``discrepancy.asymptotic`` — a extrapolação está na faixa assintótica em quase todos os pontos, que é a
          condição em que a parcela numérica significa alguma coisa.
        """
        from pinneapple_analysis.trust.trust_report import SUPPORTS, Check

        def _check(name: str, value: float, limit: float, *, critical: bool, reason: str) -> Check:
            return Check(name=name, status=SUPPORTS if value <= limit else "contradicts", critical=critical,
                         value=float(value), limit=float(limit), source="discrepancy_decomposition", reason=reason)

        return [
            _check("discrepancy.closure", self.closure, 1e-9, critical=True,
                   reason=f"|MSE - (viés² + ruído²)|/MSE = {self.closure:.3e}"),
            _check("discrepancy.asymptotic", 1.0 - self.fraction_asymptotic, 0.1, critical=False,
                   reason=f"{self.fraction_asymptotic:.0%} dos pontos na faixa assintótica"),
        ]

    def report(self) -> str:
        """O número de cada parcela, com o intervalo, em uma página — o que se escreve num relatório."""
        total = self.components["total"]
        lines = [
            f"discrepância total: {total.value:.4g} (RMSE sobre {len(self.points)} pontos, "
            f"{self.confidence:.0%})",
            f"  ruído de medida: {self.components['noise'].value:.4g}  "
            f"[{self.components['noise'].low:.4g}, {self.components['noise'].high:.4g}]",
            f"  erro de modelo:  {self.components['model_form'].value:.4g}  "
            f"[{self.components['model_form'].low:.4g}, {self.components['model_form'].high:.4g}]",
            f"  erro numérico:   {self.components['numerical'].value:.4g}  "
            f"[{self.components['numerical'].low:.4g}, {self.components['numerical'].high:.4g}]",
            f"identidade da variância: {self.closure:.2e} de resíduo relativo",
            f"identidade do viés:      {self.bias_closure:.2e} de resíduo relativo",
            f"correlação numérico×modelo: {self.bias_correlation:+.3f} "
            "(a soma em quadratura supõe 0)",
        ]
        share = 1.0 - (self.bias_rms ** 2 / self.total_mse) if self.total_mse > 0 else 0.0
        lines.append(f"peso do ruído na variância: {share:.1%} (o resto é viés)")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "components": {k: v.to_dict() for k, v in self.components.items()},
            "points": [p.to_dict() for p in self.points],
            "total_mse": self.total_mse, "total_rmse": self.total_rmse,
            "bias_rms": self.bias_rms, "noise_std": self.noise_std,
            "closure": self.closure, "bias_closure": self.bias_closure,
            "bias_correlation": self.bias_correlation,
            "fraction_asymptotic": self.fraction_asymptotic,
            "resolutions": list(self.resolutions), "confidence": self.confidence,
            "citation": self.citation,
        }


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def _rms_band(values: np.ndarray, half_widths: np.ndarray) -> tuple[float, float]:
    """Raiz quadrática dos intervalos ponto a ponto: os bounds do agregado, conservadores por construção."""
    low = _rms(np.maximum(np.abs(values) - half_widths, 0.0))
    high = _rms(np.abs(values) + half_widths)
    return low, high


def _series_from_sequence(solve: Any, x: np.ndarray, resolutions: Sequence[float]) -> list[np.ndarray]:
    """``solve`` pode ser um callable ``solve(x, h)`` ou uma sequência de resultados já calculados (um por
    refinamento) — o segundo caso é o de quem pagou o solver em outro passo e não quer pagar de novo."""
    if callable(solve):
        series = []
        for h in resolutions:
            values = np.asarray(solve(x, h), dtype=np.float64).ravel()
            if values.shape != x.shape:
                raise ValueError(
                    f"solve(x, h={h}) returned {values.shape[0]} values for {x.size} design points: the model has to "
                    "give one quantity of interest per point (a field is reduced to a scalar first)"
                )
            series.append(values)
        return series
    series = [np.asarray(s, dtype=np.float64).ravel() for s in solve]
    if len(series) != len(resolutions):
        raise ValueError(f"got {len(series)} pre-computed solves for {len(resolutions)} resolutions")
    for s in series:
        if s.shape != x.shape:
            raise ValueError(f"a pre-computed solve has {s.size} values for {x.size} design points")
    return series


def decompose_discrepancy(
    solve: Callable[[np.ndarray, float], Any] | Sequence[Any],
    x: Any,
    observations: Any,
    resolutions: Sequence[float],
    *,
    safety_factor: float = 1.25,
    confidence: float = 0.95,
    asymptotic_tolerance: float = 0.10,
    p_assumed: float | None = None,
) -> DiscrepancyDecomposition:
    """Separa a discrepância entre observação e física em erro numérico, erro de modelo e ruído, com intervalo.

    Parâmetros
    ----------
    solve : ``solve(x, h)`` que devolve uma grandeza de interesse por ponto de projeto, **ou** uma sequência de
        resultados já calculados (um por refinamento, na mesma ordem e com o mesmo tamanho de ``resolutions``).
    x : os ``n`` pontos de projeto, 1-D.
    observations : ``(n, k)`` — ``k`` réplicas da mesma medida em cada ponto. ``k >= 2`` é exigência, não sugestão:
        sem réplica não existe estimativa de erro puro, e o ruído seria um palpite.
    resolutions : os espaçamentos ``h`` usados, do mais grosso ao mais fino. Usa-se os **três mais finos**; os demais
        ficam de fora (a razão de refino sai deles mesmos).

    Levanta ``ValueError`` quando não há dado para a pergunta: menos de três refinamentos, menos de duas réplicas,
    tamanhos que não batem, ou refinamento que não está em ordem decrescente.
    """
    xv = np.asarray(x, dtype=np.float64).ravel()
    obs = np.asarray(observations, dtype=np.float64)
    if obs.ndim != 2:
        raise ValueError(f"observations must be (n_points, n_replicates); got shape {obs.shape}")
    if obs.shape[0] != xv.size:
        raise ValueError(f"{xv.size} design points but observations have {obs.shape[0]} rows")
    n_points, k = obs.shape
    if n_points < 2:
        raise ValueError(f"at least 2 design points are needed for a spread over points; got {n_points}")
    if k < 2:
        raise ValueError(
            f"at least 2 replicates per point are needed to measure the measurement noise; got {k} -- with one "
            "replicate the noise would have to be assumed, and this function does not assume it"
        )
    if not np.all(np.isfinite(obs)):
        raise ValueError("observations contain NaN or Inf")

    hs = np.asarray(resolutions, dtype=np.float64).ravel()
    if hs.size < 3:
        raise ValueError(f"three refinements are needed for Richardson extrapolation; got {hs.size}")
    hs = hs[-3:]
    if not np.all(np.diff(hs) < 0):
        raise ValueError(f"resolutions must go from coarse to fine (descending h); got {list(hs)}")
    r21 = float(hs[1] / hs[2])
    r32 = float(hs[0] / hs[1])
    if not (r21 > 1.0 and r32 > 1.0):
        raise ValueError(f"the refinement ratios must be > 1; got r21={r21:g}, r32={r32:g}")

    coarse, medium, fine = _series_from_sequence(solve, xv, tuple(float(h) for h in hs))

    # -- ruído: variância de erro puro (ANOVA), o que a réplica espalha e a física não explica --
    observed_mean = obs.mean(axis=1)
    within = obs - observed_mean[:, None]
    dof_noise = n_points * (k - 1)
    sigma2 = float(np.sum(within ** 2) / dof_noise)
    sigma = float(np.sqrt(sigma2))

    alpha = 1.0 - float(confidence)
    chi2_lo = float(stats.chi2.ppf(alpha / 2.0, dof_noise))
    chi2_hi = float(stats.chi2.ppf(1.0 - alpha / 2.0, dof_noise))
    noise_low = float(np.sqrt(dof_noise * sigma2 / chi2_hi))
    noise_high = float(np.sqrt(dof_noise * sigma2 / chi2_lo))
    # a média das réplicas carrega uma fração do ruído; entra no intervalo do erro de modelo, não no valor
    t_quantile = float(stats.t.ppf(0.5 + confidence / 2.0, dof_noise))

    points: list[PointDecomposition] = []
    for i in range(n_points):
        f_c, f_m, f_f = float(coarse[i]), float(medium[i]), float(fine[i])
        if f_c == f_m == f_f:
            extrapolated, hw_num, order, is_asym = f_f, 0.0, float("nan"), True
        else:
            try:
                conv = richardson_extrapolate(
                    f_c, f_m, f_f, r=(r21, r32), p_assumed=p_assumed,
                    safety_factor=safety_factor, asymptotic_tolerance=asymptotic_tolerance,
                )
                extrapolated = float(conv.extrapolated_value)
                hw_num = float(conv.gci_fine * abs(f_f))
                order = float(conv.observed_order)
                is_asym = bool(conv.is_asymptotic)
            except (ValueError, ZeroDivisionError, FloatingPointError):
                # dois refinamentos iguais ou valor nulo: não há ordem a estimar, e dizer isso é melhor que inventar
                extrapolated, hw_num, order, is_asym = f_f, 0.0, float("nan"), False

        numerical = extrapolated - f_f
        model_form = float(observed_mean[i]) - extrapolated
        bias = float(observed_mean[i]) - f_f
        hw_model = float(np.hypot(t_quantile * np.sqrt(sigma2 / k), hw_num))
        points.append(PointDecomposition(
            x=float(xv[i]), observed_mean=float(observed_mean[i]), finest_model=f_f,
            extrapolated=extrapolated, numerical=numerical, model_form=model_form, bias=bias,
            numerical_half_width=hw_num, model_form_half_width=hw_model,
            replicate_std=float(np.std(obs[i], ddof=1)) if k > 1 else 0.0,
            observed_order=order, is_asymptotic=is_asym,
        ))

    num_vals = np.array([p.numerical for p in points])
    num_hw = np.array([p.numerical_half_width for p in points])
    mod_vals = np.array([p.model_form for p in points])
    mod_hw = np.array([p.model_form_half_width for p in points])
    bias_vals = np.array([p.bias for p in points])

    num_low, num_high = _rms_band(num_vals, num_hw)
    mod_low, mod_high = _rms_band(mod_vals, mod_hw)

    total_mse = float(np.mean((obs - fine[:, None]) ** 2))
    bias_rms = _rms(bias_vals)
    # a parcela do ruído no MSE é SS_dentro/(nk) -- o mesmo estimador de erro puro, com o fator (k-1)/k que vem do
    # número de réplicas; é ele que fecha a identidade da ANOVA, não o sigma de erro puro sozinho
    noise_mse = float(np.sum(within ** 2) / (n_points * k))
    closure = abs(total_mse - (bias_rms ** 2 + noise_mse)) / total_mse if total_mse > 0 else 0.0
    scale = max(bias_rms, float(np.finfo(float).eps))
    bias_closure = float(np.max(np.abs(bias_vals - (num_vals + mod_vals)))) / scale
    if np.std(num_vals) > 0 and np.std(mod_vals) > 0:
        correlation = float(np.corrcoef(num_vals, mod_vals)[0, 1])
    else:
        correlation = 0.0
    asym = [p.is_asymptotic for p in points if np.isfinite(p.observed_order)]

    components = {
        "numerical": ComponentEstimate(
            name="numerical", kind="bias", value=_rms(num_vals), low=num_low, high=num_high,
            confidence=confidence, method=_NUMERICAL_METHOD,
            assumptions="convergência monotônica e faixa assintótica nas três malhas mais finas; h_i do mais grosso "
                        "ao mais fino e razão de refino constante ou declarada",
        ),
        "model_form": ComponentEstimate(
            name="model_form", kind="bias", value=_rms(mod_vals), low=mod_low, high=mod_high,
            confidence=confidence, method=_MODEL_METHOD,
            assumptions="as réplicas medem a mesma grandeza no mesmo ponto; o ruído é homocedástico o suficiente "
                        "para a variância de erro puro valer como variância da média",
        ),
        "noise": ComponentEstimate(
            name="noise", kind="noise", value=sigma, low=noise_low, high=noise_high,
            confidence=confidence, method=_NOISE_METHOD,
            assumptions="erro de medida aditivo, independente entre réplicas e da física, com variância igual em "
                        "todos os pontos (a pureza do erro é o que a réplica mostra)",
        ),
        "total": ComponentEstimate(
            name="total", kind="bias", value=float(np.sqrt(total_mse)), low=float(np.sqrt(total_mse)),
            high=float(np.sqrt(total_mse)), confidence=confidence,
            method="RMSE do modelo mais fino contra todas as réplicas",
            assumptions="nenhuma: é a discrepância medida, sem decompor",
        ),
    }

    return DiscrepancyDecomposition(
        points=points,
        components=components,
        total_mse=total_mse,
        total_rmse=float(np.sqrt(total_mse)),
        bias_rms=bias_rms,
        noise_std=sigma,
        closure=closure,
        bias_closure=bias_closure,
        bias_correlation=correlation,
        fraction_asymptotic=(sum(asym) / len(asym)) if asym else 0.0,
        resolutions=tuple(float(h) for h in hs),
        confidence=float(confidence),
    )
