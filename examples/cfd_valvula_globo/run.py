"""Válvula globo com gás real: o caso inteiro pelo pipeline, do CAD ao relatório.

A peça é a do documento que originou o caso: uma válvula globo de metano, disco a 4 mm da sede, 80 bar de
estagnação na entrada e 60 bar na saída, paredes adiabáticas. O escoamento atravessa uma cortina anular
estreita e o gás real se resfria por laminação isentálpica — é o caso em que um gás ideal dá a resposta
errada (0 K em vez de −7 K).

O domínio **não** é um sólido de revolução: o corpo tem o pescoço da haste num eixo perpendicular ao
escoamento, e uma malha que revolve um perfil em torno do eixo do fluxo não tem esse ramo. Ele é construído
como sólido (:mod:`pinneapple_simulation.pipelines.compressible_internal.solid3d`) e malhado em
hexa-dominante com ``snappyHexMesh``.

    python examples/cfd_valvula_globo/run.py --rapido     # malha grossa e janela curta, para conferir
    python examples/cfd_valvula_globo/run.py              # o caso do relatório (≈1 h nesta máquina)

Saída em ``_out/``: os casos do OpenFOAM, ``resumo.json``, as figuras, os renders e o vídeo. Requer OpenFOAM
nativo; sem Blender/ffmpeg o estudo roda e as figuras do matplotlib saem.

O procedimento completo, com os números medidos e as ressalvas, está em
``PINNeAPPle-CFD/casos/cfd08-valvula-globo/RELATORIO.md``.
"""
from __future__ import annotations

import argparse
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="válvula globo, metano 80 → 60 bar, pelo pipeline compressível")
    ap.add_argument("--rapido", action="store_true", help="malha grossa e janela curta (≈10 min)")
    ap.add_argument("--malha-mm", type=float, default=2.0, help="célula na parede, malha 3-D")
    ap.add_argument("--out", default=os.path.join(HERE, "_out"))
    ap.add_argument("--sem-render", action="store_true")
    ap.add_argument("--sem-video", action="store_true")
    ap.add_argument("--p-entrada", type=float, default=80.0,
                    help="pressão total de entrada, bar absolutos (70 é a leitura gauge)")
    a = ap.parse_args(argv)

    from pinneapple_simulation.pipelines.compressible_internal import globe_valve_experiment
    from pinneapple_simulation.pipelines.compressible_internal.experiment import (
        QUICK_STAGES,
        STAGES,
    )

    # o experimento é uma função da biblioteca: o app roda esta mesma função pelo fluxo
    # `cfd.compressible_passage` com `modo="transiente_3d"`
    saida = globe_valve_experiment(
        a.out, malha_mm=2.5 if a.rapido else a.malha_mm, p_in_bar=a.p_entrada,
        stages=QUICK_STAGES if a.rapido else STAGES,
        render=not a.sem_render, video=not a.sem_video and not a.rapido,
        log=lambda m: print(m, flush=True))

    print("\n" + saida["summary"])
    if a.rapido:
        print("\nATENÇÃO: --rapido serve para conferir a instalação, não é resultado. A malha de 3 mm e as "
              "janelas curtas não resolvem a cortina da sede, e os números acima não valem — as próprias "
              "checagens recusam. Rode sem --rapido para o caso do relatório.")
    return 0 if not saida["decision"]["failed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
