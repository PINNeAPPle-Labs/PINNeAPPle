"""The valve experiment itself, so the library, the command line and the app run the *same* thing.

A CFD case is easy to fork: one script on someone's laptop, another inside an app, and after two
weeks the numbers disagree and nobody knows which one is the case. This module is the single source:
:func:`globe_valve_experiment` builds the 3-D domain, climbs the pressure in stages, solves in
physical time, takes the time mean of the last quarter and assembles the checks, the figures, the
renders and the video. Everything else is a caller —

- ``examples/cfd_valvula_globo/run.py`` (command line),
- the Flow node ``cfd.compressible_passage`` with ``modo="transiente_3d"`` (the engineering app).

The choices that are physics, and why each one is here and not in a caller:

``totalPressure`` + ``totalTemperature`` at the inlet
    the pair has to be stagnation on both sides. Stagnation pressure with a *static* temperature
    leaks stagnation enthalpy (measured: +1,93 % in an adiabatic passage); the correct pair gives
    +1,20 % and moves the Joule-Thomson drop from −1,7 K to −7,15 K, against −7,45 K from the
    isenthalpic reference.

staged pressure ramp
    80 bar on a domain at rest sub-expands the gas and the Peng-Robinson ``T(h)`` inversion blows up
    (``Maximum number of iterations exceeded ... when starting from T0``). The stages are 62 → 70 →
    80 bar, each warmed by the previous field, with a progressively less diffusive scheme.

the time mean
    the answer is the mean over the last quarter of the window, not the last iteration. The window
    is what decides the number: 3,5 ms gives 2,1 % of mass imbalance and ΔT −1,7 K; 27,5 ms with the
    mean over the last quarter gives 0,22 % and ΔT −7,15 K.
"""
from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

__all__ = ["globe_valve_experiment", "STAGES", "QUICK_STAGES"]

#: (estagnação na entrada, bar; janela, s; início da média; esquema)
STAGES: tuple[tuple[float, float, float, str], ...] = (
    (62.0, 4.0e-4, 0.60, "upwind"),
    (70.0, 4.0e-4, 0.60, "limitedLinear"),
    (80.0, 3.5e-3, 0.25, "linearUpwind"),
)

#: the same ladder, short enough to check an installation (the numbers are not a result)
QUICK_STAGES: tuple[tuple[float, float, float, str], ...] = (
    (62.0, 2.0e-4, 0.60, "upwind"),
    (70.0, 2.0e-4, 0.60, "limitedLinear"),
    (80.0, 1.0e-3, 0.25, "linearUpwind"),
)

NOTES = (
    "as dimensões vêm do CAD, seção a seção: o STEP traz só o metal, não o volume de fluido, e um "
    "booleano dos 30 sólidos contra um envelope devolve lascas de 0,05 a 0,11 cm³",
    "entrada em pressão total e temperatura de estagnação — o par correto; pressão total com "
    "temperatura estática vaza entalpia de estagnação",
    "a cortina da sede tem ~2 células na malha padrão: é o que explica o Kv 10 % abaixo da "
    "correlação IEC 60534-2-1 e o desbalanceamento de massa acima de 0,1 %",
)


def globe_valve_experiment(folder: str, *, malha_mm: float = 2.0, p_in_bar: float = 80.0,
                           p_out_bar: float = 60.0, T_in_K: float = 300.0, gas: str = "methane",
                           stages: tuple[tuple[float, float, float, str], ...] = STAGES,
                           render: bool = True, video: bool = True, lift: float = 4.0,
                           log: Callable[[str], None] | None = None,
                           timeout: float = 21600.0) -> dict[str, Any]:
    """Run the whole case in ``folder`` and return what :func:`assemble_study` returns.

    ``stages`` may be :data:`STAGES` (the case of the report) or :data:`QUICK_STAGES` (a smoke
    test). Everything else is the physics of the case; a caller chooses the *size* of the run, never
    the physics.
    """
    from . import Conditions, StudySpec, globe_valve
    from .study import assemble_study, solve_once

    say = log or (lambda _m: None)
    previous = None
    last = None
    for k, (p0, window, average_from, scheme) in enumerate(stages):
        cond = Conditions(p_out_bar=p_out_bar, T_in_K=T_in_K, p_in_target_bar=p0, transient=True,
                          end_time_s=window, end_time_transits=1e9, average_from=average_from,
                          max_co=1.0, p_min_factor=0.5 if k == 0 else 0.1,
                          p_max_factor=1.5 if k == 0 else 2.0, n_outer=2, scheme=scheme)
        spec = StudySpec(passage=globe_valve(lift=lift, solid=True, mesh_size=malha_mm), gas=gas,
                         conditions=cond, title=f"Válvula globo — estágio {k + 1}",
                         render=False, video=False)
        stage_folder = os.path.join(folder, f"estagio_{k}")
        say(f"estágio {k + 1}: p0 {p0:g} bar, janela {window * 1e3:g} ms, média a partir de "
            f"{average_from * 100:.0f} %, {scheme}")
        last = solve_once(spec, stage_folder, restart_from=previous, timeout=timeout)
        r = last.result
        say(f"  {last.run.n_iter} passos | mdot {r.mdot_kg_s:.4f} kg/s (saída {r.mdot_out_kg_s:.4f}) "
            f"| p_in {r.p_in_pa / 1e5:.2f} bar | Kv {r.kv:.2f} | Ma {r.ma_max:.3f} | dT {r.dT_K:+.2f} K")
        previous = stage_folder

    final = stages[-1]
    spec = StudySpec(passage=globe_valve(lift=lift, solid=True, mesh_size=malha_mm), gas=gas,
                     conditions=Conditions(p_out_bar=p_out_bar, T_in_K=T_in_K,
                                           p_in_target_bar=final[0], transient=True,
                                           end_time_s=final[1]),
                     title="Válvula globo, abertura 4 mm", render=render, video=video, notes=NOTES)
    return assemble_study(spec, folder, last, [])
