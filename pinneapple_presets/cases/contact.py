"""Junta aparafusada com pré-tensão: a capacidade de escorregamento de uma ligação por atrito, medida no CalculiX.

O aperto de uma junta real não é uma pressão que alguém digita: é a força que o parafuso esticado devolve às chapas.
O caso ``parafuso_pretensao`` de :mod:`pinneapple_simulation.external_solvers.calculix.contact` é o modelo do
repositório em que isso acontece de verdade — a pré-tensão entra por um ``*PRE-TENSION SECTION``, cada ponta da viga
do parafuso chega à sua chapa por um ``*DISTRIBUTING COUPLING`` sobre a pegada da cabeça (com peso de área, não por um
nó só), e a junta é puxada tangencialmente até a interface deslizar. O que o preset entrega é a conta que a ligação
precisa responder:

* quanto a junta transmite antes de escorregar, ``mu`` vezes o aperto **medido** (e não o pedido);
* se o aperto chega à estrutura (a integral de ``CPRESS`` na interface no fim do passo de pré-tensão);
* se a cabeça pressiona como área (pico sobre média da pressão sob a pegada) em vez de morder um nó;
* a tensão nominal no fuste e a utilização contra a tensão de prova da classe.

Números medidos nos padrões do preset (chapa de 10 mm de lado, 4 mm de espessura, pegada de raio 3 mm, malha 6,
pré-tensão de 5,10 kN, mu = 0,30), com o CalculiX 2.23:

======================================  ==================================================================
aperto medido                           5,06 kN contra 5,10 kN pedidos (perda de 0,9 %)
carga de escorregamento medida           1,53 kN contra ``mu*N`` = 1,53 kN (0,1 %)
pico/média da pressão sob a cabeça       1,34 — e 1,63 no mesmo caso com o parafuso de um nó só, na mesma malha
cone de Coulomb                          ``|tau|/p`` = 0,3000 em cada nó carregado de uma interface com mu = 0,30
abertura da interface                    0 mm (a junta não abre sob o puxamento)
malha 4 / 6 / 8                          a carga de escorregamento varia 0,02 % entre os três
======================================  ==================================================================

As entradas do preset estão em SI (m, N, Pa), como as dos outros presets; o caso trabalha em mm, N e MPa, que são as
unidades do deck do CalculiX e as que o PrePoMax abre. A conversão está em :meth:`BoltedJoint.case_params` e é a
única linha em que ela acontece.

O deck e os arquivos do solver ficam em ``runs/presets/bolted_joint_preload/run-<id>/`` (ignorado pelo git;
``$PINNEAPPLE_PRESETS_RUNS`` muda a base), um diretório por corrida porque o ``ccx`` escreve ``.frd``/``.dat``/``.sta``
ao lado do deck e duas corridas na mesma pasta sobrescreveriam os resultados uma da outra. ``result.fields["deck"]``
nomeia o ``.inp`` que o solver leu — o mesmo arquivo que o PrePoMax abre.

Sem um binário ``ccx`` (o PrePoMax distribui o dele; ``CCX_BIN`` aponta para ele) o deck é escrito e o resultado diz
que nada foi resolvido: nenhuma checagem vira aprovação por ausência de medida.
"""
from __future__ import annotations

import math
import os
import tempfile

from pinneapple_simulation.external_solvers.calculix import contact as K
from pinneapple_simulation.external_solvers.calculix.contact import run as R
from pinneapple_simulation.external_solvers.calculix.contact import verify as V

from ..core import Output, Param, Preset, Reference, check

CASE = "parafuso_pretensao"

ISO_898 = "ISO 898-1:2013 Tab. 3: classe 8.8, tensão de prova S_p = 580 MPa para d <= 16 mm"
VDI_2230 = ("VDI 2230 Blatt 1 (2015) Tab. A5 e eq. (5): pré-tensão de montagem F_M = 0,70 S_p A_s numa junta única; "
            "mu de projeto 0,24 a 0,35 para aço-aço seco")
SHIGLEY = "Shigley, Mechanical Engineering Design, 10a ed., sec. 8-8: F_i = 0,75 S_p A_t para junta reutilizada"
CCX_MANUAL = "CalculiX 2.23 manual, *PRE-TENSION SECTION e *DISTRIBUTING COUPLING"
COULOMB = "Coulomb: a interface escorrega quando a força tangencial passa de mu x N"
RICHMOND = "Richmond & DeVries / Shigley sec. 8-11: a ligação por atrito não escorrega em serviço, e a carga de "
RICHMOND += "escorregamento é mu x aperto"

#: the case's own checks, in the language of the engineering decision
LABEL = {
    "solver.convergiu": "o CalculiX terminou o deck do contato",
    "parafuso.aperto": "o aperto medido é a pré-tensão aplicada",
    "equilibrio.aperto": "o aperto não se perde quando a junta é puxada",
    "parafuso.pegada": "a cabeça do parafuso pressiona como área, não como um nó",
    "atrito.saturacao": "a força tangencial satura em mu x N quando a interface desliza",
    "atrito.limite": "a reação tangencial transmitida não passa do limite de atrito",
    "atrito.cone": "|tau| <= mu p em cada nó carregado (cone de Coulomb)",
    "contato.fechado": "a junta não abre sob o puxamento (COPEN <= 0 em todos os nós)",
}

#: how the case's outcome maps to the trust report (None = not measured, which is never a pass)
OUTCOME = {"pass": True, "fail": False, "inconclusive": None}


def runs_dir(name: str) -> str:
    """Where the deck and the solver's own files of a preset are kept: ``$PINNEAPPLE_PRESETS_RUNS`` when set, else
    the repository's ``runs/presets/`` (git-ignored) when it is writable, else a folder under the system's temporary
    directory."""
    base = os.environ.get("PINNEAPPLE_PRESETS_RUNS")
    if not base:
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        base = (os.path.join(root, "runs", "presets") if os.access(root, os.W_OK)
                else os.path.join(tempfile.gettempdir(), "pinneapple_presets"))
    return os.path.join(base, name)


def case_run_dir(name: str) -> str:
    """A folder of this run alone. The solver reads its argument as a file stem and writes ``.frd``/``.dat``/``.sta``
    beside it, so two runs sharing a folder overwrite each other's results — and the runner of the Hub and the Flow
    node do run in parallel. A directory per call is the only safe way to keep what a check read."""
    os.makedirs(runs_dir(name), exist_ok=True)
    return tempfile.mkdtemp(prefix="run-", dir=runs_dir(name))


class BoltedJoint(Preset):
    name = "bolted_joint_preload"
    title = "Junta aparafusada com pré-tensão: a carga de escorregamento da ligação, medida no CalculiX"
    sector = "Estrutural / ligações aparafusadas, máquinas e estruturas metálicas"
    problem = ("Duas chapas de aço apertadas por um parafuso cuja pré-tensão é aplicada por um *PRE-TENSION SECTION, "
               "com a cabeça e a porca chegando às chapas por *DISTRIBUTING COUPLING sobre a pegada (carga "
               "distribuída por área, como a de uma cabeça real). A junta é puxada tangencialmente até a interface "
               "deslizar: o deck é resolvido pelo CalculiX 2.23, os resultados são lidos de volta por "
               "pinneapple_data.cae e a força de escorregamento é comparada com mu vezes o aperto medido.")
    question = ("Quanta força cisalhante esta junta transmite antes de escorregar, a pré-tensão chega mesmo às chapas, "
                "e a cabeça pressiona como área ou morde um nó? A junta atende a carga de serviço com o fator de "
                "segurança pedido, e qual pré-tensão seria necessária se não atendesse?")
    params = (
        Param("preload", "N", 5100.0, VDI_2230 + "; " + ISO_898 + " com a haste de raio 2 mm (A = 12,57 mm²)",
              500.0, 60000.0),
        Param("mu", "-", 0.30, VDI_2230, 0.05, 0.60,
              help="o limite é mu x N, independente da área aparente; acima de ~0,6 o deslizamento prescrito pode "
                   "não ser suficiente para a interface chegar ao limite, e a checagem fica inconclusiva"),
        Param("plate_side", "m", 0.010, "premissa: chapa de 10 mm de lado, para a pegada de 3 mm caber com folga",
              0.004, 0.200),
        Param("plate_thickness", "m", 0.004, "premissa: duas chapas de 4 mm (o comprimento aparafusado)", 0.001, 0.050),
        Param("bolt_radius", "m", 0.002, "premissa: haste de raio 2 mm (M4/M5); só dimensiona a viga do parafuso",
              0.0005, 0.020),
        Param("head_radius", "m", 0.003, "raio da pegada da cabeça e da porca (a área por onde o aperto entra)",
              0.0012, 0.030,
              help="a pegada precisa de ao menos quatro nós da face: raio menor que ~meia célula da malha é "
                   "recusado pelo caso, com a mensagem que pede a malha mais fina ou a pegada maior"),
        Param("slip_imposed", "m", 5e-5, "deslizamento prescrito no segundo passo, grande o bastante para a "
                                         "interface chegar ao limite de atrito", 0.0, 1e-3),
        Param("mesh", "-", 6, "células por direção nas chapas; a carga de escorregamento varia 0,02 % entre "
                              "4, 6 e 8 células", 4, 16),
        Param("E", "Pa", 210e9, "aço carbono (BPVC II-D Tab. TM-1, ~201 GPa; aqui o aço do caso)", 70e9, 220e9),
        Param("nu", "-", 0.3, "aço carbono", 0.25, 0.35),
        Param("service_load", "N", 1000.0, "premissa: 1 kN de cisalhamento de serviço na junta", 10.0, 50000.0),
        Param("safety_factor", "-", 1.2, RICHMOND + " com 20 % de margem (ligação que não pode deslizar)", 1.0, 3.0),
        Param("proof_strength", "Pa", 580e6, ISO_898, 200e6, 1200e6),
    )
    outputs = (
        Output("F_clamp", "N", "aperto medido: integral de CPRESS na interface no fim do passo de pré-tensão"),
        Output("clamp_loss", "-", "perda entre a pré-tensão pedida e o aperto medido (0,009 = 0,9 % a menos)"),
        Output("F_slip", "N", "capacidade de escorregamento do atrito: mu vezes o aperto medido"),
        Output("F_slip_measured", "N", "força tangencial medida no apoio quando a interface desliza (a prova do "
                                       "F_slip)"),
        Output("slip_margin", "-", "F_slip_measured / (fator de segurança x carga de serviço); >= 1 atende"),
        Output("F_preload_required", "N", "pré-tensão que a carga de serviço exige: fator x carga / mu (conta fechada)"),
        Output("p_head_mean", "Pa", "pressão média de contato sob a pegada da cabeça"),
        Output("p_head_peak", "Pa", "pressão de pico sob a pegada"),
        Output("head_ratio", "-", "pico / média sob a pegada: 1 é carga distribuída, muito acima disso é um nó"),
        Output("bolt_stress", "Pa", "tensão nominal no fuste do parafuso com a pré-tensão"),
        Output("bolt_utilisation", "-", "tensão no fuste / tensão de prova da classe"),
        Output("slip_measured", "m", "escorregamento relativo medido na interface no fim do puxamento"),
        Output("contact_opening", "m", "maior abertura (COPEN) na interface: a junta não pode abrir em serviço"),
        Output("razao_coulomb", "-", "maior |tau| / p na interface carregada: toca mu quando a junta escorrega"),
    )
    validity = ("Duas chapas de aço elásticas, contato de Coulomb seco com mu constante, parafuso modelado como viga "
                "de um elemento substituída pela restrição da pré-tensão, pegada quadrada (o bloco de nós dentro do "
                "raio) e deslizamento prescrito pequeno em relação ao lado da chapa. Não cobre fadiga, relaxação, "
                "pressão de contato na borda do furo, rosca nem plastificação local sob a cabeça.")
    pipeline = ("geometria: duas chapas 10 x 4 x 10 mm e a pegada de raio 3 mm em torno do eixo do parafuso",
                "modelo: malha hexaédrica estruturada (pinneapple_simulation.external_solvers.calculix.contact.mesh), "
                "deck com *PRE-TENSION SECTION e *DISTRIBUTING COUPLING (contact.deck)",
                "solver: CalculiX 2.23 (ccx) em dois passos — pré-tensão e depois puxamento tangencial prescrito",
                "pós: .frd/.dat lidos por pinneapple_data.cae — CPRESS, CSHEAR, COPEN, reações",
                "decisão: carga de escorregamento contra a carga de serviço com o fator, aperto contra a pré-tensão, "
                "cabeça como área, tensão no fuste contra a classe")
    library = ("pinneapple_simulation.external_solvers.calculix.contact.run_case",
               "pinneapple_data.cae.contact")
    missing = {"surrogate": "cada ponto do DOE custa uma corrida de 15 a 25 s do ccx em dois passos; o domínio é "
                            "pequeno e a conta que importa (mu x N) é fechada, então um surrogate não pagaria o "
                            "custo de treino",
               "inversion": "mu não é medido pela análise: ele é a entrada que a torna conservadora ou não. "
                            "Recuperá-lo exigiria ensaio de escorregamento, não uma corrida de elementos finitos",
               "optimization": "a pré-tensão que a carga de serviço exige é uma conta fechada, publicada em "
                               "F_preload_required; otimizá-la com o sólver só confirmaria a conta a 20 s por ponto",
               "reason": "não há caso de junta aparafusada no pinneapple_reason"}
    has_3d = True
    has_variants = True
    needs = ("ccx",)              # the deck is solved by the CalculiX binary; without it nothing is measured
    variant_outputs = ("F_slip_measured", "slip_margin", "F_clamp")
    #: the contact imprint is the same picture in every variant (it is the head, not the load), so the variants are
    #: compared by their numbers and the bar chart, not by five identical stills
    variant_gallery = False
    limitations = (
        "o parafuso não é modelado volumetricamente: a flexibilidade dele é a do comprimento aparafusado, e não há "
        "concentração de tensão na rosca (a tensão do fuste é a nominal, pré-tensão sobre pi r²)",
        "a pegada é o bloco de nós dentro do raio, não um anel: o furo do parafuso e a concentração de pressão na "
        "borda do furo não estão no modelo",
        "contato e atrito são resolvidos com a formulação do CalculiX (penalidade + atrito de Coulomb); "
        "escorregamento parcial e micro-deslizamento da borda não são capturados",
        "o deslizamento é prescrito, não uma força aplicada: é o que faz a interface chegar ao limite sem divergir, "
        "mas a curva força-deslizamento depois do escorregamento não é o resultado",
    )

    # ------------------------------------------------------------------------------------- the SI <-> case bridge
    def case_params(self, p: dict) -> dict:
        """The case's own units (mm, N, MPa), converted from the preset's SI data sheet. The only place the units
        change: below this line everything is what the deck and the PrePoMax read."""
        return {"pretensao": float(p["preload"]), "friccao": float(p["mu"]),
                "lado": 1e3 * p["plate_side"], "espessura": 1e3 * p["plate_thickness"],
                "raio_parafuso": 1e3 * p["bolt_radius"], "raio_pegada": 1e3 * p["head_radius"],
                "deslizamento": 1e3 * p["slip_imposed"], "malha": float(p["mesh"]),
                "E": 1e-6 * p["E"], "nu": float(p["nu"])}

    # ---------------------------------------------------------------------------------------------------- the solve
    def solve(self, p):
        area_bolt = math.pi * p["bolt_radius"] ** 2
        stress = p["preload"] / area_bolt
        out = {"F_preload_required": p["safety_factor"] * p["service_load"] / p["mu"],
               "bolt_stress": stress, "bolt_utilisation": stress / p["proof_strength"],
               "F_slip": p["mu"] * p["preload"], "area_bolt": area_bolt, "case": CASE}
        cp = self.case_params(p)
        _, _, built = K.build(CASE, cp)
        solver = R.ccx_binary()
        out["solver"] = {"binary": solver, "available": solver is not None,
                         "version": R.version(solver) if solver else None}
        missing = {"F_clamp": None, "clamp_loss": None, "F_slip_measured": None, "slip_margin": None,
                   "p_head_mean": None, "p_head_peak": None, "head_ratio": None, "slip_measured": None,
                   "contact_opening": None, "razao_coulomb": None, "reaction_vertical": None}
        if solver is None:                       # the deck is written, and nothing is solved from it
            out.update(missing)
            out["case_checks"] = []
            deck = os.path.join(case_run_dir(self.name), f"{CASE}.inp")
            with open(deck, "w", encoding="utf-8") as fh:
                fh.write(built.deck)
            out["_fields"] = {"case_params": cp, "deck": deck, "deck_chars": len(built.deck), "solve_time_s": None}
            return out
        run_dir = case_run_dir(self.name)
        res = R.run_case(CASE, cp, run_dir, binary=solver)
        deck = os.path.join(run_dir, f"{CASE}.inp")
        parsed = R.read_results(deck, built.times)
        weights = V.tributary_areas(built.coords, built.interface, plane="y")
        at_preload = parsed.get("contact_until", {}).get("preload") or {}
        clamped = V.resultant(at_preload, weights)
        rows = R.footprint(parsed, built)
        press = [r["pressure"] for r in rows if r["pressure"] > 0.0]
        reactions = (res.get("measured") or {}).get("reactions") or {}
        base = reactions.get("BFIX") or {}
        opened = [row.get("opening", 0.0) for node, row in (parsed.get("contact") or {}).items()
                  if node in set(built.interface)]
        out["_fields"] = {"case_params": cp, "deck": deck, "solve_time_s": res.get("seconds"),
                          "footprint": rows, "radii_mm": [r["radius"] for r in rows],
                          "pressures_mpa": [r["pressure"] for r in rows],
                          "preload_n": p["preload"], "service_n": p["service_load"],
                          "head_radius_mm": cp["raio_pegada"], "side_mm": cp["lado"], "mesh": cp["malha"],
                          "coords": built.coords, "interface": list(built.interface), "level_mm": cp["espessura"],
                          "contact_at_preload": {int(n): float(r.get("pressure", 0.0))
                                                 for n, r in at_preload.items() if n in set(built.interface)}}
        out["case_checks"] = [c.as_dict() for c in res.get("checks") or []]
        if res.get("status") != "ok" or clamped is None:
            out.update(missing)
            out["solver"]["detail"] = res.get("detail") or res.get("error")
            out["solver"]["status"] = res.get("status")
            return out
        out["F_clamp"] = clamped
        out["clamp_loss"] = (p["preload"] - clamped) / p["preload"]
        out["F_slip"] = p["mu"] * clamped
        out["F_slip_measured"] = abs(float(base.get("fx", 0.0)))
        out["slip_margin"] = out["F_slip_measured"] / (p["safety_factor"] * p["service_load"])
        if press:
            mean = sum(press) / len(press)
            out["p_head_mean"] = mean * 1e6
            out["p_head_peak"] = max(press) * 1e6
            out["head_ratio"] = max(press) / mean
        else:
            out.update({k: None for k in ("p_head_mean", "p_head_peak", "head_ratio")})
        slip = (res.get("measured") or {}).get("max_slip")
        out["slip_measured"] = None if slip is None else slip * 1e-3
        out["razao_coulomb"] = (res.get("measured") or {}).get("coulomb_ratio")
        out["contact_opening"] = max(opened) * 1e-3 if opened else 0.0
        out["reaction_vertical"] = abs(float(base.get("fy", 0.0)))
        return out

    # --------------------------------------------------------------------------------------------------- the checks
    def checks(self, p, out):
        rows = [check("CalculiX (ccx) disponível para resolver o deck", out["solver"]["available"],
                      value=out["solver"]["version"], source="ccx -v",
                      reason="" if out["solver"]["available"] else
                      "sem ccx nada foi resolvido: o deck foi escrito e nenhuma medida existe (instale o ccx, que o "
                      "PrePoMax distribui, ou aponte CCX_BIN para ele)", critical=True)]
        for c in out.get("case_checks") or []:
            ok = OUTCOME.get(c["outcome"])
            rows.append(check(LABEL.get(c["name"], c["name"]), ok, value=c["value"], limit=c["reference"],
                              source=c["source"], reason=c["detail"] if ok is not True else ""))
        solved = out.get("F_slip_measured") is not None
        rows.append(check("a carga de escorregamento atende a carga de serviço com o fator",
                          out["slip_margin"] >= 1.0 if solved else None, value=out.get("slip_margin"),
                          limit=1.0, source=RICHMOND,
                          reason="" if solved else "sem solver não há carga de escorregamento medida"))
        rows.append(check("a tensão no fuste fica em 75 % da tensão de prova ou menos",
                          out["bolt_utilisation"] <= 0.75, value=out["bolt_utilisation"], limit=0.75,
                          source=ISO_898 + " e " + SHIGLEY))
        rows.append(check("o aperto medido é a pré-tensão a menos de 2 %",
                          abs(out["clamp_loss"]) <= 0.02 if solved else None, value=out.get("clamp_loss"), limit=0.02,
                          source="equilíbrio do *PRE-TENSION SECTION: a força de aperto é a pré-tensão aplicada",
                          reason="" if solved else "sem solver não há aperto medido"))
        rows.append(check("equilíbrio vertical da junta (reação fora do plano nula)",
                          out["reaction_vertical"] <= 1e-3 * p["preload"] if solved else None,
                          value=out.get("reaction_vertical"), limit=1e-3 * p["preload"],
                          source="soma das reações em y no apoio é nula: nada empurra a junta fora do plano",
                          reason="" if solved else "sem solver não há reações"))
        return rows

    # ------------------------------------------------------------------------------------------------- the decision
    def decide(self, p, out):
        if out.get("F_slip_measured") is None:
            return {"verdict": "N/A", "reasons": [
                f"sem CalculiX (ccx) o deck {CASE} foi escrito e nada foi resolvido: a capacidade de projeto "
                f"(mu x pré-tensão) é {out['F_slip']:.0f} N, mas não há aperto nem escorregamento medidos",
                "instale o ccx (o PrePoMax distribui o binário) ou aponte CCX_BIN para ele e rode de novo"]}
        margin, loss = out.get("slip_margin"), out.get("clamp_loss")
        ratio, util, opening = out.get("head_ratio"), out.get("bolt_utilisation"), out.get("contact_opening")
        if None in (margin, loss, ratio, util, opening):
            return {"verdict": "N/A", "reasons": [
                f"a corrida do CalculiX não deixou todas as medidas (status {out['solver'].get('status')!r}): "
                f"{out['solver'].get('detail', 'sem detalhe')}",
                "sem a pressão de contato ou as reações não há decisão: a checagem do trust report diz qual falta"]}
        ok = margin >= 1.0 and abs(loss) <= 0.02 and ratio <= 1.5 and util <= 0.75 and opening <= 0.0
        reasons = [
            f"escorrega a {1e-3 * out['F_slip_measured']:.2f} kN (mu x aperto medido = {1e-3 * out['F_slip']:.2f} kN) "
            f"contra {1e-3 * p['service_load']:.2f} kN de serviço x {p['safety_factor']:.2f} = margem "
            f"{margin:.2f}",
            f"aperto medido {1e-3 * out['F_clamp']:.2f} kN para {1e-3 * p['preload']:.2f} kN de pré-tensão "
            f"({100 * loss:+.1f} %), pressão sob a cabeça {1e-6 * out['p_head_mean']:.0f} MPa média e "
            f"{1e-6 * out['p_head_peak']:.0f} MPa de pico (pico/média {ratio:.2f})",
            f"tensão no fuste {1e-6 * out['bolt_stress']:.0f} MPa = {100 * util:.0f} % da tensão "
            f"de prova de {1e-6 * p['proof_strength']:.0f} MPa",
        ]
        if not ok:
            need = out["F_preload_required"]
            reasons.append(f"para atender são precisos {1e-3 * need:.2f} kN de pré-tensão (fator x carga / mu); a "
                           f"junta está com {1e-3 * p['preload']:.2f} kN")
        return {"verdict": "OK" if ok else "NOT OK", "reasons": reasons,
                "F_slip": out["F_slip_measured"], "slip_margin": margin,
                "F_preload_required": out["F_preload_required"],
                "governing": ("escorregamento" if margin < 1.0 else "tensão no fuste" if util > 0.75 else
                              "aperto perdido" if abs(loss) > 0.02 else
                              "cabeça pontual" if ratio > 1.5 else "nenhum: a junta atende")}

    # ----------------------------------------------------------------------------------------------- the references
    def references(self):
        """The closed forms a bolted joint is judged by, each at its own operating point: the clamp force *is* the
        preload, and the slip load *is* mu times it — at two very different friction coefficients, so a solver that
        ignores the interface or applies mu to the wrong surface cannot pass both."""
        return [
            _ref("aperto = pré-tensão (8 kN)", {"preload": 8000.0}, "F_clamp", 8000.0, "N", 0.03,
                 "equilíbrio do *PRE-TENSION SECTION: a força que o parafuso devolve às chapas é a pré-tensão "
                 "(medido -1,7 %, a mesma perda que a checagem do caso declara em 3 %)", "analytic", CCX_MANUAL),
            _ref("escorregamento em mu x N com mu = 0,5 e 4 kN", {"mu": 0.5, "preload": 4000.0}, "F_slip_measured",
                 2000.0, "N", 0.03, COULOMB, "analytic", CCX_MANUAL),
            _ref("escorregamento em mu x N com mu = 0,15 e 6 kN (junta oleada)", {"mu": 0.15, "preload": 6000.0},
                 "F_slip_measured", 900.0, "N", 0.03, COULOMB, "analytic", CCX_MANUAL),
            _ref("cone de Coulomb: |tau|/p na interface igual a mu = 0,5", {"mu": 0.5, "preload": 4000.0},
                 "razao_coulomb", 0.5, "-", 0.05,
                 "cada nó carregado fica dentro do cone de atrito, e o maior |tau|/p o toca quando a interface "
                 "desliza", "analytic", CCX_MANUAL),
        ]

    def validate(self):
        """Every reference case, each configuration solved once (the cone row shares the run of its own mu), plus the
        mesh study. Without a ``ccx`` there is nothing to validate, and the result says so instead of returning an
        empty pass."""
        if R.ccx_binary() is None:
            return {"preset": self.name, "references": [], "convergence": None, "all_pass": False,
                    "error": ("no CalculiX (ccx) binary: the deck is written but no reference case can be solved. "
                              "Set CCX_BIN or put ccx on PATH (PrePoMax ships one).")}
        cache: dict[tuple, dict] = {}

        def outputs(params: dict) -> dict:
            key = tuple(sorted(params.items()))
            if key not in cache:
                cache[key] = self.run(**params).outputs
            return cache[key]

        rows = []
        for ref in self.references():
            v = outputs(dict(ref.params))[ref.output]
            err = abs(float(v) - ref.value) / abs(ref.value) if ref.value else abs(float(v))
            rows.append({"case": ref.name, "output": ref.output, "unit": ref.unit, "computed": float(v),
                         "reference": ref.value, "rel_error": err, "tol": ref.tol, "pass": bool(err <= ref.tol),
                         "kind": ref.kind, "source": ref.source, "params": dict(ref.params)})
        conv = self.convergence()
        return {"preset": self.name, "references": rows, "convergence": conv,
                "all_pass": all(r["pass"] for r in rows) and (conv is None or conv.get("pass", True))}

    def convergence(self, values=(4, 6, 8)) -> dict | None:
        """The mesh study: the slip load is a property of the joint, not of the grid, so refining the plates must not
        move it (it varies 0,02 % between 4 and 8 cells)."""
        if R.ccx_binary() is None:
            return None
        runs = []
        for n in values:
            o = self.run(mesh=n).outputs
            runs.append({"mesh": n, "F_slip_measured": o.get("F_slip_measured"), "F_clamp": o.get("F_clamp"),
                         "head_ratio": o.get("head_ratio")})
        slip = [r["F_slip_measured"] for r in runs]
        if any(v is None for v in slip):
            return None
        rel = abs(slip[-1] - slip[-2]) / abs(slip[-1]) if len(slip) > 1 and slip[-1] else None
        return {"parameter": "malha (células por direção nas chapas)", "values": list(values),
                "F_slip_measured": slip, "F_clamp": [r["F_clamp"] for r in runs],
                "head_ratio": [r["head_ratio"] for r in runs], "rel_change_last": rel,
                "pass": bool(rel is not None and rel < 0.01)}

    # -------------------------------------------------------------------------------------------------- the designs
    def variants(self):
        return {"aço seco mu 0,30 (base)": {},
                "superfície oleada mu 0,10": {"mu": 0.10},
                "pré-tensão de 3 kN": {"preload": 3000.0},
                "serviço de 2 kN": {"service_load": 2000.0},
                "parafuso 8.8 acima do limite": {"preload": 6000.0}}

    # --------------------------------------------------------------------------------------------------- the figure
    def figures(self, result, out_dir):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f = result.fields
        radii, press = f.get("radii_mm") or [], f.get("pressures_mpa") or []
        if not radii:
            return []
        o = result.outputs
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 3.6))
        ax1.plot(radii, press, "C0o-", ms=4, label="CalculiX (CPRESS)")
        ax1.axhline(o["p_head_mean"] * 1e-6, color="C3", ls="--", lw=1,
                    label=f"média {1e-6 * o['p_head_mean']:.0f} MPa")
        ax1.axvline(f["head_radius_mm"], color="k", ls=":", lw=1,
                    label=f"raio da pegada {f['head_radius_mm']:g} mm")
        ax1.set(xlabel="distância ao eixo do parafuso [mm]", ylabel="pressão de contato [MPa]",
                title=f"a cabeça é uma área: pico/média {o['head_ratio']:.2f}")
        ax1.legend(fontsize=7)
        labels = ["pré-tensão\npedida", "aperto\nmedido", "escorrega em\nmu x aperto", "serviço x\nfator"]
        vals = [1e-3 * f["preload_n"], 1e-3 * o["F_clamp"], 1e-3 * o["F_slip_measured"],
                1e-3 * result.params["service_load"] * result.params["safety_factor"]]
        cols = ["#264653", "#2a9d8f", "#2a9d8f" if o["slip_margin"] >= 1 else "#e76f51", "#8d99ae"]
        ax2.bar(labels, vals, color=cols)
        for i, v in enumerate(vals):
            ax2.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8)
        ax2.set(ylabel="força [kN]", title=f"margem de escorregamento {o['slip_margin']:.2f}")
        ax2.tick_params(labelsize=7)
        fig.tight_layout()
        path = os.path.join(out_dir, "joint.png")
        fig.savefig(path, dpi=90)
        plt.close(fig)
        return [path]

    # ----------------------------------------------------------------------------------------------------- the scene
    def scene(self, result):
        """The contact pressure on the interface plane, over the base plate: the imprint of the head, coloured by
        ``CPRESS`` at the end of the preload step — which is what shows the load entering over an area. The upper
        plate is left out so the interface can be seen, the plane sits a hair above the plate's face so the two do not
        fight for the same pixels, and the thickness goes on the viewer's up axis so the joint reads as a slab."""
        import numpy as np

        from ..viz3d import Scene, Surface, box
        f = result.fields
        side, thk, n = f["side_mm"], f["level_mm"], int(f["mesh"])
        pre = f.get("contact_at_preload") or {}
        coords = f.get("coords") or {}
        lift = 0.02 * thk                                      # the interface floats just over the plate's face
        # the interface lattice: the nodes at the mid-plane, in mesh order, so the quads join the right corners
        xs = np.linspace(0.0, side, n + 1)
        at = {(round(coords[i][0], 6), round(coords[i][2], 6)): i for i in f["interface"]}
        ids = np.array([[at.get((round(x, 6), round(z, 6)), -1) for z in xs] for x in xs])
        V3 = np.array([[[x, z, thk + lift] for z in xs] for x in xs], float).reshape(-1, 3)
        idx = np.arange(len(V3)).reshape(n + 1, n + 1)
        quad = np.stack([idx[:-1, :-1], idx[1:, :-1], idx[1:, 1:], idx[:-1, 1:]], -1).reshape(-1, 4)
        flat = ids.ravel()
        keep = np.array([all(flat[q] > 0 for q in face) for face in quad]) if len(quad) else np.zeros(0, bool)
        quad = quad[keep]
        tri = np.r_[quad[:, [0, 1, 2]], quad[:, [0, 2, 3]]] if len(quad) else np.zeros((0, 3), int)
        p = np.array([pre.get(int(i), 0.0) for i in flat], float)          # MPa
        interface = Surface("interface", V3, tri, "steel", {"p": p})
        base = Surface("chapa inferior", *box([0, 0, 0], [side, side, thk], n=(4, 4, 2)), "steel")
        return Scene([base, interface], title=self.title,
                     labels={"p": "pressão de contato CPRESS [MPa]"})


def _ref(name, params, output, value, unit, tol, what, kind, source):
    return Reference(name=name, params=params, output=output, value=value, unit=unit, source=f"{what} ({source})",
                     kind=kind, tol=tol)
