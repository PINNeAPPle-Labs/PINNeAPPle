"""Simulação orientada a evento: ``pp.Event(condition=..., action=...)`` (#210).

Um impacto, uma mudança de fase, uma válvula que abre, um contato, um limiar de reação: são todos a mesma coisa em
termos de integração — em algum instante ``t*`` uma condição sobre o estado cruza zero, e no instante seguinte a
dinâmica é outra. Integrar com passo fixo e testar a condição **depois** do passo erra o instante por até um passo
inteiro (e num impacto, errar o instante é errar a energia); integrar com passo fixo e simplesmente refletir a
velocidade quando o estado ficou abaixo do chão *cria* energia, porque o rebote acontece em ``h < 0``.

Este módulo faz o que um integrador de eventos faz: detecta a troca de sinal da condição dentro do passo, localiza o
instante por bissecção até a tolerância pedida, aplica a ação **naquele** instante e retoma a integração com o passo
que sobrou. O instante que sai no registro é o localizado, com a largura final do intervalo de bissecção junto — é o
que permite comparar com o tempo analítico.

    >>> import numpy as np
    >>> from pinneapple_simulation.events import Event, reflect, run_events
    >>> g, h0 = 9.81, 1.0
    >>> floor = Event(name="floor", condition=lambda t, y, p: y[0], action=reflect(component=1, factor=-1.0))
    >>> trace = run_events(lambda t, y, p: np.array([y[1], -g]), [h0, 0.0], (0.0, 2.0), 0.01, [floor])
    >>> round(trace.first("floor"), 6), round(np.sqrt(2 * h0 / g), 6)     # o tempo analítico da queda
    (0.451523, 0.451523)

Por que a condição é uma função e não um número: ``condition(t, y, params) -> float`` é a função de cruzamento, e o
evento dispara onde ela muda de sinal na direção pedida (``falling`` por omissão: de positivo para negativo, que é o
caso de um objeto que chega ao chão). ``action(t, y, params) -> Reaction`` devolve o estado novo, muda parâmetros ou
para a simulação — as três coisas que uma ação de verdade faz.

Limites declarados, porque um benchmark os encontra:

* a trajetória entre dois instantes é integrada por um passo de Runge-Kutta 4, o mesmo do integrador principal; a
  localização do evento herda essa aproximação (não há interpolação densa de ordem superior aqui);
* a detecção é o teste de sinal **nos extremos do passo**: um evento cujo cruzamento inteiro cabe dentro de um passo
  não é visto. Num rebote isso acontece quando o voo entre dois quiques fica menor que ``dt`` — e é por isso que o
  passo tem de resolver o evento, não o contrário: com ``dt`` grande demais a bola "atravessa" o chão em silêncio no
  `run_events`, o que é o mesmo erro que qualquer integrador de passo fixo comete;
* um evento que dispara sem parar (chattering: a condição continua satisfeita logo depois da própria ação) estoura
  ``max_events`` e o módulo levanta :class:`EventLimit` em vez de girar para sempre. O caso de Zeno (bola com
  restituição menor que 1 quicando cada vez mais rápido) tem o mesmo fim *se* o passo ainda o resolver; mais cedo,
  ele cai na limitação anterior — use ``reflect(stop_below=...)`` para fechá-lo por física em vez de por contagem;
* depois de disparar, o evento é desativado por omissão (``repeat=False``), que é o que a maioria dos casos quer; um
  rebote que deve repetir declara ``repeat=True`` e um piso de velocidade na ação (ver ``reflect(stop_below=...)``).
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

__all__ = ["Event", "Reaction", "Trigger", "EventTrace", "EventLimit", "run_events",
           "reflect", "set_params", "stop_simulation", "chain"]

_Directions = Literal["falling", "rising", "any"]


class EventLimit(RuntimeError):
    """An event fired more times than ``max_events`` allowed -- the case is chattering or Zeno, and this says so."""


@dataclass
class Reaction:
    """O que uma ação faz no instante do evento: novo estado, parâmetros alterados, ou parar.

    Qualquer combinação é válida. ``state=None`` mantém o estado; ``params`` é mesclado; ``stop=True`` encerra a
    simulação **naquele** instante, e o registro diz por quê em ``reason``.
    """

    state: np.ndarray | None = None
    params: dict[str, Any] | None = None
    stop: bool = False
    reason: str = ""


@dataclass(frozen=True)
class Event:
    """Uma condição de cruzamento e a ação que ela dispara.

    ``condition(t, y, params) -> float`` é a função de cruzamento: o evento dispara onde ela troca de sinal na
    direção pedida. ``action(t, y, params) -> Reaction`` diz o que passa a valer dali em diante.
    """

    name: str
    condition: Callable[[float, np.ndarray, Mapping[str, Any]], float]
    action: Callable[[float, np.ndarray, Mapping[str, Any]], Reaction]
    direction: _Directions = "falling"
    repeat: bool = False
    tolerance: float = 1e-10

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("an event needs a name: it is how the trace says which one fired")
        if self.direction not in ("falling", "rising", "any"):
            raise ValueError(f"direction must be 'falling', 'rising' or 'any'; got {self.direction!r}")
        if self.tolerance <= 0:
            raise ValueError("tolerance is a width: it must be > 0")

    def crosses(self, before: float, after: float) -> bool:
        """A condição passou para o outro lado, na direção pedida, entre os dois valores?

        O zero exato conta como o lado **de onde se vem**: um passo que começa em ``g == 0`` e termina negativo é um
        cruzamento descendente — é o caso do rebote, que termina exatamente na superfície e cujo passo seguinte entra
        no lado proibido. O contrário também vale: sair de ``g == 0`` para positivo não é cruzamento nenhum, então o
        objeto que acabou de quicar não dispara de novo enquanto sobe.
        """
        if self.direction == "falling":
            return before >= 0.0 > after
        if self.direction == "rising":
            return before <= 0.0 < after
        return before >= 0.0 > after or before <= 0.0 < after


@dataclass(frozen=True)
class Trigger:
    """Um disparo, com o instante localizado, a largura do intervalo e quantas avaliações custou."""

    name: str
    time: float
    width: float
    evaluations: int
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "time": self.time, "width": self.width,
                "evaluations": self.evaluations, "reason": self.reason}


@dataclass
class EventTrace:
    """O que a simulação deixou: a trajetória, os disparos e se ela parou por ação de um evento."""

    times: np.ndarray
    states: np.ndarray
    triggers: list[Trigger] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)
    stopped: bool = False
    stop_reason: str = ""

    def names(self) -> list[str]:
        return [t.name for t in self.triggers]

    def times_of(self, name: str) -> list[float]:
        return [t.time for t in self.triggers if t.name == name]

    def first(self, name: str) -> float:
        """O primeiro instante em que o evento ``name`` disparou; ``KeyError`` se ele nunca disparou."""
        for trigger in self.triggers:
            if trigger.name == name:
                return trigger.time
        raise KeyError(f"event '{name}' never fired (fired: {self.names()})")

    def state_at(self, time: float) -> np.ndarray:
        """O estado registrado no instante mais próximo de ``time`` (a grade é a dos passos dados)."""
        index = int(np.argmin(np.abs(self.times - float(time))))
        return self.states[index]

    def summary(self) -> dict[str, Any]:
        return {"steps": int(len(self.times)), "t_end": float(self.times[-1]) if len(self.times) else float("nan"),
                "triggers": [t.to_dict() for t in self.triggers], "stopped": self.stopped,
                "stop_reason": self.stop_reason, "params": dict(self.params)}


# ------------------------------------------------------------------------------------------------ ações prontas
def reflect(component: int, factor: float, *, stop_below: float | None = None, speed_component: int | None = None
            ) -> Callable[[float, np.ndarray, Mapping[str, Any]], Reaction]:
    """A ação de um impacto: multiplica uma componente do estado por ``factor`` (``-1`` = rebote elástico).

    ``stop_below`` para a simulação quando ``|y[speed_component]|`` (por omissão a própria componente) fica abaixo
    dele — é o que fecha um caso de Zeno com ``restitution < 1`` em vez de deixá-lo girar.
    """
    def action(t: float, y: np.ndarray, params: Mapping[str, Any]) -> Reaction:  # noqa: ARG001 -- action contract
        new = np.array(y, dtype=float, copy=True)
        new[component] *= float(factor)
        index = component if speed_component is None else int(speed_component)
        if stop_below is not None and abs(float(new[index])) < float(stop_below):
            return Reaction(state=new, stop=True, reason=f"|y[{index}]| < {stop_below:g}: the bounce is over")
        return Reaction(state=new)

    return action


def set_params(**values: Any) -> Callable[[float, np.ndarray, Mapping[str, Any]], Reaction]:
    """A ação de uma válvula que abre, de uma bomba que liga: altera parâmetros que o lado direito lê."""
    def action(t: float, y: np.ndarray, params: Mapping[str, Any]) -> Reaction:  # noqa: ARG001 -- action contract
        return Reaction(params=dict(values))

    return action


def stop_simulation(reason: str = "") -> Callable[[float, np.ndarray, Mapping[str, Any]], Reaction]:
    """A ação de um limiar de falha: a simulação termina no instante do evento."""
    def action(t: float, y: np.ndarray, params: Mapping[str, Any]) -> Reaction:  # noqa: ARG001 -- action contract
        return Reaction(stop=True, reason=reason)

    return action


def chain(*actions: Callable[[float, np.ndarray, Mapping[str, Any]], Reaction]
          ) -> Callable[[float, np.ndarray, Mapping[str, Any]], Reaction]:
    """Aplica as ações em ordem, acumulando estado e parâmetros (a última que pedir para parar decide)."""
    def action(t: float, y: np.ndarray, params: Mapping[str, Any]) -> Reaction:
        state, merged, stopped, reason = y, dict(params), False, ""
        for step in actions:
            reaction = step(t, state, merged)
            if reaction.state is not None:
                state = np.asarray(reaction.state, dtype=float)
            if reaction.params:
                merged.update(reaction.params)
            stopped = stopped or reaction.stop
            reason = reaction.reason or reason
        changed = {k: v for k, v in merged.items() if k not in params or params[k] != v}
        return Reaction(state=state, params=changed or None, stop=stopped, reason=reason)

    return action


# ------------------------------------------------------------------------------------------------ integrador
def _rk4(rhs: Callable[..., np.ndarray], t: float, y: np.ndarray, h: float, params: Mapping[str, Any]) -> np.ndarray:
    """Um passo de Runge-Kutta 4 clássico. O mesmo passo é usado para avançar e para avaliar a trajetória dentro do
    intervalo durante a bissecção, então a localização do evento tem a precisão do integrador, não outra."""
    k1 = np.asarray(rhs(t, y, params), dtype=float)
    k2 = np.asarray(rhs(t + h / 2.0, y + h * k1 / 2.0, params), dtype=float)
    k3 = np.asarray(rhs(t + h / 2.0, y + h * k2 / 2.0, params), dtype=float)
    k4 = np.asarray(rhs(t + h, y + h * k3, params), dtype=float)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def _locate(event: Event, rhs: Callable[..., np.ndarray], t_a: float, y_a: np.ndarray, h: float,
            before: float, after: float, params: Mapping[str, Any]) -> tuple[float, float, int]:
    """Bissecção no intervalo ``[t_a, t_a + h]`` sobre ``g(t, integrar(y_a até t))``, até ``event.tolerance``."""
    lo, hi = 0.0, h
    g_lo = before
    evaluations = 1
    while (hi - lo) > event.tolerance:
        mid = 0.5 * (lo + hi)
        g_mid = float(event.condition(t_a + mid, _rk4(rhs, t_a, y_a, mid, params), params))
        evaluations += 1
        if not np.isfinite(g_mid):
            break
        if (g_lo > 0.0 > g_mid) or (g_lo < 0.0 < g_mid):
            hi = mid
        else:
            lo, g_lo = mid, g_mid
        if g_mid == 0.0:
            lo = hi = mid
            break
    t_event = t_a + 0.5 * (lo + hi)
    return float(t_event), float(hi - lo), evaluations


def run_events(
    rhs: Callable[[float, np.ndarray, Mapping[str, Any]], np.ndarray],
    y0: Any,
    t_span: Sequence[float],
    dt: float,
    events: Sequence[Event],
    *,
    params: Mapping[str, Any] | None = None,
    max_events: int = 1000,
    min_step: float = 1e-12,
) -> EventTrace:
    """Integra ``dy/dt = rhs(t, y, params)`` de ``t_span[0]`` a ``t_span[1]``, parando em cada evento.

    ``rhs(t, y, params)`` devolve o lado direito (o mesmo ``params`` que as ações alteram). O passo é ``dt``; quando
    um evento é localizado dentro do passo, a integração é retomada exatamente no instante localizado, com o resto do
    passo, em vez de recomeçar do fim do passo.

    Levanta :class:`EventLimit` quando os disparos passam de ``max_events`` (chattering/Zeno) e ``ValueError`` para
    ``dt <= 0``, ``t_span`` invertido ou passo menor que ``min_step``.
    """
    y = np.array(y0, dtype=float, copy=True)
    t0, t1 = float(t_span[0]), float(t_span[1])
    if not t1 > t0:
        raise ValueError(f"t_span must be increasing; got {t_span!r}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0; got {dt}")
    if dt < min_step:
        raise ValueError(f"dt={dt:g} is below min_step={min_step:g}: the step has to resolve something")

    current: dict[str, Any] = dict(params or {})
    active = list(events)
    times: list[float] = [t0]
    states: list[np.ndarray] = [y.copy()]
    triggers: list[Trigger] = []
    stopped, stop_reason = False, ""

    t = t0
    while t < t1 - min_step:
        h = min(dt, t1 - t)
        y_next = _rk4(rhs, t, y, h, current)
        fired = False
        for event in tuple(active):
            before = float(event.condition(t, y, current))
            after = float(event.condition(t + h, y_next, current))
            if not event.crosses(before, after):
                continue
            t_event, width, evaluations = _locate(event, rhs, t, y, h, before, after, current)
            y_event = _rk4(rhs, t, y, t_event - t, current)
            reaction = event.action(t_event, y_event, current)
            if reaction.state is not None:
                y_event = np.asarray(reaction.state, dtype=float)
            if reaction.params:
                current.update(reaction.params)
            if not event.repeat:
                active.remove(event)
            triggers.append(Trigger(name=event.name, time=t_event, width=width, evaluations=evaluations,
                                    reason=reaction.reason))
            if len(triggers) > max_events:
                raise EventLimit(
                    f"more than {max_events} events: '{event.name}' keeps firing (chattering or a Zeno case -- a "
                    "restitution >= 1 against a floor, or a condition that is still satisfied right after its own "
                    "action). Give the action a floor (reflect(stop_below=...)) or set repeat=False"
                )
            y, t = y_event, t_event
            times.append(t_event)
            states.append(y.copy())
            if reaction.stop:
                stopped, stop_reason = True, reaction.reason
                break
            fired = True
            # o resto do passo é reavaliado do zero a partir do estado pós-evento: a dinâmica mudou aqui
            break
        if stopped:
            break
        if fired:
            continue
        y, t = y_next, t + h
        times.append(t)
        states.append(y.copy())

    return EventTrace(times=np.asarray(times), states=np.asarray(states), triggers=triggers,
                      params=dict(current), stopped=stopped, stop_reason=stop_reason)
