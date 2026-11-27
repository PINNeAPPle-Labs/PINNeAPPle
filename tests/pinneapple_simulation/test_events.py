"""Eventos disparando no instante certo, comparados com o tempo analítico (#210).

O aceite do #210 é "um exemplo de bola quicando ou de válvula abrindo dispara eventos no instante certo dentro da
tolerância". Os dois exemplos estão aqui, e nos dois o instante do disparo tem forma fechada para comparar:

**bola quicando** — ``h(t) = h0 - g t²/2`` até o chão, então o primeiro impacto é ``t* = sqrt(2 h0 / g)`` e, com
restituição ``e``, o segundo é ``t* (1 + 2e)``. O que se cobra não é "disparou": é o instante, e também a altura de
cada quique, que é ``e² h`` vezes a anterior — a energia que o rebote deve conservar ou perder.

**válvula de um tanque** — o tanque enche com vazão ``Q`` e, quando o nível chega a ``h_open``, a saída abre com
coeficiente ``k``. A abertura é ``t1 = h_open A / Q`` (enquanto não há saída) e, depois dela, ``dh/dt = (Q - k h)/A``,
cuja solução é fechada: o instante em que o nível volta a ``h_close`` é
``t2 = t1 + ln((h_open - Q/k) / (h_close - Q/k)) / (-k)``. O teste compara os dois instantes com esses números.

Além dos dois exemplos, os recusas: passo não positivo, intervalo invertido, evento sem nome, direção inválida, e
chattering (uma ação que deixa a condição satisfeita de novo) — que levanta ``EventLimit`` em vez de girar.
"""
from __future__ import annotations

import numpy as np
import pytest

from pinneapple_simulation.events import (
    Event,
    EventLimit,
    Reaction,
    chain,
    reflect,
    run_events,
    set_params,
    stop_simulation,
)

# --------------------------------------------------------------------------------------------- exemplo 1: a bola
G = 9.81


def ball_rhs(t, y, params):  # noqa: ARG001 -- the integrator's own signature
    return np.array([y[1], -G])


def floor_event(restitution: float = 0.8, *, stop_below: float | None = None) -> Event:
    return Event(name="floor", condition=lambda t, y, p: y[0],
                 action=reflect(component=1, factor=-restitution, stop_below=stop_below), repeat=True)


def test_the_first_impact_is_the_analytic_free_fall_time():
    """t* = sqrt(2 h0 / g): o instante localizado, não o fim do passo em que ele cai."""
    h0 = 1.0
    trace = run_events(ball_rhs, [h0, 0.0], (0.0, 1.0), 0.01, [floor_event()])
    analytic = float(np.sqrt(2.0 * h0 / G))
    assert trace.first("floor") == pytest.approx(analytic, abs=1e-9)
    # e a bissecção diz a que largura ela chegou -- não é um número mais preciso do que o método que o produziu
    assert trace.triggers[0].width <= 1e-9


def test_the_second_impact_is_the_analytic_rebound_time():
    """Com restituição e, o segundo impacto é t* (1 + 2e) -- o rebote devolve a velocidade certa no instante certo."""
    h0, e = 1.0, 0.8
    trace = run_events(ball_rhs, [h0, 0.0], (0.0, 3.0), 0.01, [floor_event(e)])
    first = float(np.sqrt(2.0 * h0 / G))
    second = first * (1.0 + 2.0 * e)
    assert trace.times_of("floor")[0] == pytest.approx(first, abs=1e-9)
    assert trace.times_of("floor")[1] == pytest.approx(second, abs=1e-9)


def test_each_bounce_returns_to_the_square_of_the_restitution():
    """A altura depois de cada quique é ``e²`` da anterior: é a física, medida na trajetória registrada."""
    h0, e = 1.0, 0.8
    trace = run_events(ball_rhs, [h0, 0.0], (0.0, 3.0), 0.002, [floor_event(e)])
    times = trace.times_of("floor")
    apexes = []
    for start, end in zip(times[:-1], times[1:], strict=True):
        window = (trace.times > start) & (trace.times < end)
        if window.any():
            apexes.append(float(trace.states[window, 0].max()))
    assert len(apexes) >= 2
    for previous, current in zip(apexes[:-1], apexes[1:], strict=True):
        assert current / previous == pytest.approx(e ** 2, rel=0.02)


def test_a_bounce_can_end_on_a_velocity_floor_and_say_so():
    """``stop_below`` fecha o caso: a simulação para no impacto e o registro diz de que se tratava."""
    trace = run_events(ball_rhs, [1.0, 0.0], (0.0, 20.0), 0.01, [floor_event(0.8, stop_below=0.1)])
    assert trace.stopped is True
    assert "the bounce is over" in trace.stop_reason
    assert len(trace.triggers) == len(trace.times_of("floor"))
    # parou num impacto, com a altura no chão
    assert abs(float(trace.states[-1, 0])) < 1e-6


# ------------------------------------------------------------------------------------------ exemplo 2: a válvula
A_TANK, Q_IN, H_OPEN, H_CLOSE, K_OUT = 1.0, 0.01, 0.5, 0.25, 0.1


def tank_rhs(t, y, params):  # noqa: ARG001 -- the integrator's own signature
    return np.array([(params["Q"] - params["k"] * y[0]) / A_TANK])


def valve_events() -> list[Event]:
    return [
        Event(name="open", condition=lambda t, y, p: y[0] - H_OPEN, direction="rising", action=set_params(k=K_OUT)),
        Event(name="close", condition=lambda t, y, p: y[0] - H_CLOSE, direction="falling", action=set_params(k=0.0)),
    ]


def test_the_valve_opens_at_the_analytic_level_crossing_time():
    trace = run_events(tank_rhs, [0.0], (0.0, 80.0), 0.05, valve_events(), params={"Q": Q_IN, "k": 0.0})
    assert trace.first("open") == pytest.approx(H_OPEN * A_TANK / Q_IN, abs=1e-8)


def test_the_valve_closes_at_the_analytic_solution_of_the_drain_equation():
    """Depois de abrir, ``dh/dt = (Q - k h)/A`` tem solução fechada, e o instante de fechar sai dela."""
    trace = run_events(tank_rhs, [0.0], (0.0, 80.0), 0.05, valve_events(), params={"Q": Q_IN, "k": 0.0})
    t_open = H_OPEN * A_TANK / Q_IN
    # dh/dt = (Q - k h)/A  =>  h(t) = Q/k + (h_open - Q/k) exp(-k (t - t_open)/A)
    t_close = t_open + (A_TANK / K_OUT) * np.log((H_OPEN - Q_IN / K_OUT) / (H_CLOSE - Q_IN / K_OUT))
    assert trace.first("close") == pytest.approx(t_close, abs=1e-6)
    # e o nível está no limiar no instante em que a válvula fechou
    assert trace.state_at(trace.first("close"))[0] == pytest.approx(H_CLOSE, abs=1e-3)


def test_the_action_changes_the_right_hand_side_for_the_rest_of_the_run():
    """A ação altera parâmetros que o lado direito lê: a saída abre e o nível passa a cair."""
    trace = run_events(tank_rhs, [0.0], (0.0, 80.0), 0.05, valve_events(), params={"Q": Q_IN, "k": 0.0})
    t_open = trace.first("open")
    before = trace.state_at(t_open - 1.0)[0]
    after = trace.state_at(t_open + 1.0)[0]
    assert before < H_OPEN                 # enchia
    assert after < H_OPEN                  # e agora esvazia, porque a saída abriu
    assert trace.params["k"] == pytest.approx(0.0)  # e fechou de novo no segundo evento


def test_a_condition_that_crosses_in_the_other_direction_does_not_fire():
    """A direção é a do sinal da condição: uma válvula que só abre ao subir não dispara enquanto o nível desce."""
    only_rising = [Event(name="open", condition=lambda t, y, p: y[0] - H_CLOSE, direction="rising",
                         action=set_params(k=K_OUT))]
    trace = run_events(tank_rhs, [H_OPEN], (0.0, 20.0), 0.05, only_rising, params={"Q": Q_IN, "k": K_OUT})
    assert trace.triggers == []            # o nível só desceu


def test_a_failure_threshold_stops_the_run_at_the_analytic_time():
    """Um limiar de falha (aqui, temperatura subindo a taxa constante) para a simulação no instante do evento."""
    rate, limit = 2.0, 7.0
    event = Event(name="failure", condition=lambda t, y, p: y[0] - limit, direction="rising",
                  action=stop_simulation("temperature limit reached"))
    trace = run_events(lambda t, y, p: np.array([rate]), [0.0], (0.0, 100.0), 0.1, [event])
    assert trace.first("failure") == pytest.approx(limit / rate, abs=1e-9)
    assert trace.stopped is True and trace.stop_reason == "temperature limit reached"
    assert trace.times[-1] == pytest.approx(limit / rate, abs=1e-9)
    assert trace.times[-1] < 100.0         # não integrou até o fim


def test_chained_actions_accumulate_state_and_parameters():
    """``chain`` aplica as ações em ordem: uma reflete a velocidade e a outra muda um parâmetro no mesmo impacto."""
    event = Event(
        name="impact",
        condition=lambda t, y, p: y[0],
        action=chain(reflect(component=1, factor=-1.0), set_params(bounces=1)),
    )
    trace = run_events(ball_rhs, [1.0, 0.0], (0.0, 1.0), 0.01, [event], params={"bounces": 0})
    assert trace.params["bounces"] == 1
    # a velocidade refletida: logo depois do impacto ela é positiva (a bola subiu)
    assert trace.state_at(trace.first("impact") + 1e-3)[1] > 0.0


# --------------------------------------------------------------------------------------------- o que é recusado
def test_a_condition_that_keeps_being_satisfied_right_after_its_own_action_is_stopped_with_a_reason():
    """Chattering: a ação devolve o estado para logo acima do limiar e a dinâmica o empurra de volta, para sempre."""
    chattering = Event(name="level", condition=lambda t, y, p: y[0] - 0.5, direction="falling",
                       action=lambda t, y, p: Reaction(state=np.array([0.5001, -1.0])), repeat=True)
    with pytest.raises(EventLimit, match="more than 20 events"):
        run_events(lambda t, y, p: np.array([y[1], 0.0]), [0.5001, -1.0], (0.0, 100.0), 0.1, [chattering],
                   max_events=20)


def test_a_negative_step_is_refused():
    with pytest.raises(ValueError, match="dt must be > 0"):
        run_events(ball_rhs, [1.0, 0.0], (0.0, 1.0), -0.01, [])


def test_an_inverted_interval_is_refused():
    with pytest.raises(ValueError, match="t_span must be increasing"):
        run_events(ball_rhs, [1.0, 0.0], (1.0, 0.0), 0.01, [])


def test_an_event_needs_a_name_and_a_direction_that_exists():
    with pytest.raises(ValueError, match="needs a name"):
        Event(name="", condition=lambda t, y, p: y[0], action=reflect(component=1, factor=-1.0))
    with pytest.raises(ValueError, match="direction must be"):
        Event(name="x", condition=lambda t, y, p: y[0], action=reflect(component=1, factor=-1.0),
              direction="sideways")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="tolerance"):
        Event(name="x", condition=lambda t, y, p: y[0], action=reflect(component=1, factor=-1.0), tolerance=0.0)


def test_a_trace_without_the_event_says_so_by_name():
    trace = run_events(ball_rhs, [1.0, 0.0], (0.0, 0.1), 0.01, [floor_event()])
    with pytest.raises(KeyError, match="never fired"):
        trace.first("floor")
    assert trace.summary()["steps"] == len(trace.times)
