"""Trajetória de um jato livre e estimativa de onde o pico de energia cinética turbulenta vai cair.

Duas perguntas de engenharia que aparecem sempre que um jato entra num banho:

1. **Onde o jato toca a superfície?** Um jato lançado com ``u = (0, u_y, u_z)`` de uma altura ``h`` acima do banho
   percorre ``x = u_y (u_z + √(u_z² + 2 g h)) / g`` na horizontal antes de tocar. É o alcance balístico: só depende do
   vetor, da altura e da gravidade — não da vazão nem do diâmetro do furo.
2. **Onde fica o pico medido?** Em escoamento real o máximo de ``k`` não está no toque, e sim um pouco a jusante, onde
   a pluma arrasta ar, quebra e mistura. Quem tem as duas parcelas medidas (toque → joelho do perfil, joelho → pico)
   estima a posição do pico para outra condição de entrada **sem re-rodar**:

   ``s_pico ≈ alcance(u, h) + atraso_toque + atraso_pico``

As duas parcelas de atraso vêm do próprio caso (ver ``field_lines.joelho`` e ``field_lines.pico``). A inversa,
:func:`velocidade_para_pico`, responde "que velocidade de entrada poria o pico em s?" — útil para separar hipóteses
antes de gastar uma rodada.

    from pinneapple_simulation.pipelines.free_jet import alcance, estimar_pico, velocidade_para_pico
    alcance((0, 9.85, 1.74), h=0.75)                       # -> 5.976...
    estimar_pico((0, 9.85, 1.74), 0.75, 0.30, 0.99)        # -> 7.27
    velocidade_para_pico(5.68, 0.75, 0.30, 0.99, theta_deg=10.02)   # -> ~7.9 m/s

Tudo em SI. ``u`` é o vetor de velocidade na entrada, na mesma base das alturas: só ``u_y`` (horizontal) e ``u_z``
(vertical) entram.
"""
from __future__ import annotations

import math
from collections.abc import Sequence

G = 9.81


def _uy_uz(u: Sequence[float]) -> tuple[float, float]:
    u = tuple(float(x) for x in u)
    if len(u) < 3:
        raise ValueError("u precisa ter 3 componentes (x, y, z)")
    return u[1], u[2]


def alcance(u: Sequence[float], h: float, g: float = G) -> float:
    """Alcance horizontal (m) de um jato livre que queda ``h`` metros até tocar a superfície.

    ``h`` é a diferença de cota entre a entrada e a superfície do banho; ``h <= 0`` significa entrada já submersa e o
    alcance perde sentido, então levanta ``ValueError`` em vez de devolver um número sem significado.
    """
    if h <= 0:
        raise ValueError(f"h precisa ser positivo (entrada acima do banho); recebi {h}")
    uy, uz = _uy_uz(u)
    if uy <= 0:
        raise ValueError("a componente horizontal u_y precisa ser positiva (jato para jusante)")
    return uy * (uz + math.sqrt(uz ** 2 + 2.0 * g * h)) / g


def tempo_de_voo(u: Sequence[float], h: float, g: float = G) -> float:
    """Tempo (s) entre a saída e o toque na superfície: ``(u_z + √(u_z² + 2gh)) / g``."""
    if h <= 0:
        raise ValueError(f"h precisa ser positivo (entrada acima do banho); recebi {h}")
    _, uz = _uy_uz(u)
    return (uz + math.sqrt(uz ** 2 + 2.0 * g * h)) / g


def vetor_de(velocidade: float, theta_deg: float) -> tuple[float, float, float]:
    """Vetor ``(0, U cosθ, U senθ)`` a partir do módulo e do ângulo acima da horizontal."""
    t = math.radians(theta_deg)
    return 0.0, velocidade * math.cos(t), velocidade * math.sin(t)


def estimar_pico(u: Sequence[float], h: float, atraso_toque: float = 0.0, atraso_pico: float = 0.0,
                 g: float = G) -> float:
    """Posição do pico (m) estimada por três parcelas: ``alcance + atraso_toque + atraso_pico``.

    ``atraso_toque``: do toque geométrico até o joelho medido do perfil de k (quanto o campo demora a reagir);
    ``atraso_pico``: do joelho até o máximo (a pluma se desenvolvendo).
    """
    return alcance(u, h, g) + float(atraso_toque) + float(atraso_pico)


def velocidade_para_alcance(x: float, theta_deg: float, h: float, g: float = G) -> float:
    """Módulo da velocidade de entrada que faz o jato tocar a superfície em ``x`` metros (inversa por bisseção).

    O alcance cresce monotonicamente com a velocidade, então a bisseção converge; o intervalo é ampliado até conter a
    solução e a tolerância é de 1 mm no alcance.
    """
    alvo = float(x)
    lo, hi = 1e-3, 1.0
    while alcance(vetor_de(hi, theta_deg), h, g) < alvo:
        hi *= 2.0
        if hi > 1e4:
            raise ValueError(f"sem solução: alcance não chega a {x} m com h = {h} m")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if alcance(vetor_de(mid, theta_deg), h, g) < alvo:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-9:
            break
    return 0.5 * (lo + hi)


def velocidade_para_pico(s_pico: float, h: float, atraso_toque: float = 0.0, atraso_pico: float = 0.0,
                         theta_deg: float = 0.0, g: float = G) -> float:
    """Módulo da velocidade de entrada que poria o pico em ``s_pico``, mantendo o ângulo e os atrasos medidos."""
    return velocidade_para_alcance(float(s_pico) - float(atraso_toque) - float(atraso_pico), theta_deg, h, g)


__all__ = ["alcance", "tempo_de_voo", "vetor_de", "estimar_pico", "velocidade_para_alcance", "velocidade_para_pico"]
