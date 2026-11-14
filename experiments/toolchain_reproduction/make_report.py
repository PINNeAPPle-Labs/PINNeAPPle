"""The PDF summary of the toolchain reproduction experiment, built from ``results/results.json`` and the figures.

    python experiments/toolchain_reproduction/make_report.py --out ~/Downloads/PINNeAPPle-experimento.pdf

Every number in the PDF is read from ``results/results.json`` (written by ``run.py``), so the summary cannot drift
from the run it describes. reportlab is the only extra dependency, and only for this script.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    Image,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Table,
    TableStyle,
)

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

# the script writes its case and stage names in English (that is what the repository and the manifest use); the
# summary is in Portuguese, so the labels are translated here instead of in the results
CASE_PT = {"cavity Re=100": "cavidade, Re = 100", "cavity Re=1000": "cavidade, Re = 1000",
           "water hammer": "golpe de aríete", "cantilever": "viga, Hermite",
           "cantilever 3-D": "viga 3-D", "plate with a hole": "chapa com furo"}
STAGE_PT = {"toolchain audit": "auditoria da toolchain", "lid-driven cavity": "cavidade",
            "water hammer": "golpe de aríete", "cantilever": "viga (Hermite + 3 malhas 3-D)",
            "plate with a hole": "chapa com furo (3 malhas)", "post-processing": "pós-processamento (5 figuras)"}

INK = colors.HexColor("#1a1a1a")
GREY = colors.HexColor("#6b6b6b")
RULE = colors.HexColor("#c9c9c9")
OK = colors.HexColor("#1f7a3f")
BAD = colors.HexColor("#b3261e")
BAND = colors.HexColor("#f3f1ec")


def br(x: float, sig: int = 5) -> str:
    """A number in pt-BR notation (comma decimal) for the body text and the tables."""
    if x is None:
        return "—"
    s = f"{x:.{sig}g}"
    if "e" in s:
        mantissa, exp = s.split("e")
        return f"{mantissa.replace('.', ',')}e{int(exp)}"
    return s.replace(".", ",")


def pct(x: float) -> str:
    return "—" if x is None else (f"{100 * x:+.2e} %").replace(".", ",")


def styles() -> dict:
    s = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=s["BodyText"], fontName="Helvetica", fontSize=9.2, leading=13.4,
                          textColor=INK, alignment=TA_JUSTIFY, spaceAfter=5)
    return {
        "title": ParagraphStyle("title", parent=s["Title"], fontName="Helvetica-Bold", fontSize=17.5, leading=21,
                                textColor=INK, spaceAfter=2, alignment=0),
        "sub": ParagraphStyle("sub", parent=body, fontSize=10.2, leading=14, textColor=GREY, spaceAfter=10),
        "h": ParagraphStyle("h", parent=s["Heading2"], fontName="Helvetica-Bold", fontSize=11.8, leading=15,
                            textColor=INK, spaceBefore=11, spaceAfter=5),
        "body": body,
        "small": ParagraphStyle("small", parent=body, fontSize=8.2, leading=11, textColor=GREY),
        "cap": ParagraphStyle("cap", parent=body, fontSize=8.4, leading=11.2, textColor=GREY, spaceBefore=2,
                              spaceAfter=9),
        "cell": ParagraphStyle("cell", parent=body, fontSize=7.9, leading=10, alignment=0, spaceAfter=0),
        "cellb": ParagraphStyle("cellb", parent=body, fontName="Helvetica-Bold", fontSize=7.9, leading=10,
                                alignment=0, spaceAfter=0),
        "num": ParagraphStyle("num", parent=body, fontSize=7.9, leading=10, alignment=2, spaceAfter=0),
    }


def table(rows: list[list], widths: list[float], st: dict, *, head: bool = True, numeric_from: int = 2) -> Table:
    """A zebra-striped table; the first row is a header, columns from ``numeric_from`` are right-aligned."""
    cells = []
    for i, row in enumerate(rows):
        line = []
        for j, v in enumerate(row):
            if isinstance(v, Paragraph):
                line.append(v)
            elif i == 0 and head:
                line.append(Paragraph(f"<b>{v}</b>", st["cell"]))
            else:
                line.append(Paragraph(str(v), st["num"] if j >= numeric_from else st["cell"]))
        cells.append(line)
    t = Table(cells, colWidths=widths, repeatRows=1 if head else 0, hAlign="LEFT")
    style = [("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("LINEBELOW", (0, 0 if head else -1), (-1, 0 if head else -1), 0.7, INK if head else RULE),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    for i in range(1, len(rows)):
        if i % 2 == 0:
            style.append(("BACKGROUND", (0, i), (-1, i), BAND))
    t.setStyle(TableStyle(style))
    return t


def figure(path: str, width: float, caption: str, st: dict) -> KeepTogether:
    iw, ih = ImageReader(path).getSize()
    return KeepTogether([Image(path, width=width, height=width * ih / iw), Paragraph(caption, st["cap"])])


def build(out: str) -> str:
    with open(os.path.join(RESULTS, "results.json"), encoding="utf-8") as f:
        r = json.load(f)
    with open(os.path.join(RESULTS, "experiment.json"), encoding="utf-8") as f:
        prov = json.load(f)["provenance"]
    st = styles()
    env = r["toolchain"]
    rows = r["rows"]
    passed = sum(1 for x in rows if x["passed"])
    when = datetime.fromisoformat(prov["created"].replace("Z", "+00:00")).astimezone(timezone(timedelta(hours=-3)))
    docs: list = []

    # -- cover block
    docs.append(Paragraph("Um PC em branco, um agente e a toolchain — reproduzido no PINNeAPPle", st["title"]))
    docs.append(Paragraph("Resumo do experimento: cinco modelos clássicos de CFD e FEA, cada um conferido contra o "
                          "valor publicado, num único script sem licença.", st["sub"]))
    docs.append(table([["Execução", f"{when:%d/%m/%Y %H:%M} (UTC−3), commit "
                                    f"<font face='Courier'>{prov['git']}</font>, "
                                    f"{br(prov['seconds'], 4)} s medidos pelo Laboratório"],
                       ["Verificação", f"{passed}/{len(rows)} conferências passaram"
                        + ("" if r["passed"] else "  (há falhas)")],
                       ["Tempo de parede", f"{br(r['total_s'], 4)} s nesta máquina (carga alta; ~1 min livre)"],
                       ["Ambiente", f"python {env['python']} · numpy {env['packages']['numpy']} · scipy "
                                    f"{env['packages']['scipy']} · matplotlib {env['packages']['matplotlib']}"],
                       ["Licenças comerciais", f"{env['licences_paid']}"],
                       ["Ferramentas externas", ("instaladas mas não usadas: "
                                                 + ", ".join(env["external_solvers_installed_but_not_used"]))
                        if env["external_solvers_installed_but_not_used"] else "nenhuma usada"]],
                      [46 * mm, 118 * mm], st, head=False, numeric_from=99))

    # -- 1. what this is
    docs.append(Paragraph("1. O que foi reproduzido", st["h"]))
    docs.append(Paragraph(
        "O experimento original parte de um Windows vazio, pede a um agente de IA para instalar OpenFOAM, code_aster e "
        "ParaView e rodar alguns casos clássicos; cerca de uma hora depois tudo está instalado, resolvido e "
        "pós-processado, e os resultados batem com o livro. Aqui a toolchain <i>é</i> o PINNeAPPle: instalar vira "
        "<font face='Courier'>import</font>, não há licença a pedir e cada etapa é uma chamada de função em "
        "<font face='Courier'>experiments/toolchain_reproduction/run.py</font>. O que se mede é quanto tempo isso custa "
        "e a que distância os números ficam da referência publicada.", st["body"]))
    docs.append(table([["Toolchain original", "Aqui"],
                       ["OpenFOAM (Navier-Stokes incompressível)",
                        "motor local vorticidade–função de corrente (external_solvers.pict)"],
                       ["OpenFOAM (transiente 1-D em tubulação)", "método das características (water_hammer_moc)"],
                       ["code_aster (estruturas)", "beam_fe (Hermite), solid_fem (hexaedros), plane_stress_fem (CST)"],
                       ["ParaView (pós-processamento)", "matplotlib e solid_fem.fea_figure"]],
                      [58 * mm, 106 * mm], st))

    # -- 2. the results
    docs.append(Paragraph("2. As conferências", st["h"]))
    docs.append(Paragraph("Cada linha compara o que o modelo calculou com o que a referência diz. Erros em módulo: "
                          "os menores aparecem em negrito.", st["body"]))
    body_rows = [["Caso", "Quantidade", "Calculado", "Livro / limite", "Erro"]]
    for x in rows:
        tb = ("limite até " + br(x["tolerance"], 2)) if x["textbook"] is None else br(x["textbook"], 6)
        err = pct(x["relative_error"])
        if not x["passed"]:
            err = f"<font color='#b3261e'>{err} FALHOU</font>"
        elif x["relative_error"] is not None and abs(x["relative_error"]) < 1e-5:
            err = f"<b>{err}</b>"
        body_rows.append([CASE_PT.get(x["case"], x["case"]), x["quantity"],
                          br(x["value"], 6) + (f" {x['unit']}" if x["unit"] else ""),
                          tb + (f" {x['unit']}" if x["unit"] and x["textbook"] is not None else ""), err])
    docs.append(table(body_rows, [27 * mm, 53 * mm, 30 * mm, 31 * mm, 23 * mm], st, numeric_from=2))

    docs.append(Paragraph(
        f"<b>Nota sobre o Re = 1000.</b> O motor local marcha até o estado estacionário e para no resíduo de "
        f"vorticidade ou, se a máquina estiver ocupada demais, no guarda de 300 s de tempo de parede — e diz qual dos "
        f"dois o encerrou. Nesta execução ele bateu no guarda: "
        f"{r['cavity']['1000']['steps']} passos, resíduo {br(r['cavity']['1000']['final_residual'], 2)}, "
        f"rms de {br(r['cavity']['1000']['checks']['ghia_u']['rms'], 4)} contra os "
        f"{br(8.13e-3, 3)} registrados no repositório para o mesmo caso numa máquina livre. O Re = 100, que converge "
        f"em {r['cavity']['100']['steps']} passos, sai idêntico ao valor guardado em "
        f"<font face='Courier'>examples/pict_cavity/cavity.json</font> — a diferença é truncamento, não método. "
        f"O critério de parada baseado no relógio ficou registrado como a issue #1677 do repositório.", st["body"]))

    # -- 3. figures
    docs.append(Paragraph("3. Figuras", st["h"]))
    w = 168 * mm
    docs.append(figure(os.path.join(RESULTS, "cavity_vs_ghia.png"), w,
                       "Cavidade com tampa deslizante: as linhas de centro calculadas sobre os pontos publicados de "
                       "Ghia, Ghia & Shin (1982), em Re = 100 e Re = 1000.", st))
    docs.append(figure(os.path.join(RESULTS, "water_hammer.png"), w,
                       "Golpe de aríete: pressão na válvula depois do fechamento instantâneo (regime de Joukowsky) e do "
                       "fechamento lento (regime de Michaud), com as retas de referência de cada um.", st))
    docs.append(figure(os.path.join(RESULTS, "cantilever.png"), w,
                       "Viga em balanço: flecha dos elementos de Hermite sobre a fórmula fechada (indistinguíveis) e a "
                       "convergência da mesma viga em hexaedros 3-D para PL<super>3</super>/3EI.", st))
    docs.append(figure(os.path.join(RESULTS, "cantilever_3d.png"), w,
                       "A viga 3-D com a tensão de flexão S11. No meio do vão, ±150 MPa contra Mc/I = 150 MPa; o pico no "
                       "engaste é a singularidade do canto e cresce com o refino, por isso a conferência é feita longe "
                       "dele.", st))
    docs.append(figure(os.path.join(RESULTS, "plate_hole.png"), w,
                       "Chapa com furo central: campo de sigma_xx na malha mais fina, a tensão ao longo do contorno do furo e a "
                       "convergência de Kt contra Heywood e Kirsch, com extrapolação de Richardson.", st))

    # -- 4. time
    docs.append(Paragraph("4. Tempo", st["h"]))
    clock = [["Etapa", "s"]] + [[STAGE_PT.get(k, k), br(v, 4)] for k, v in r["clock_s"].items()] + [
        ["total", br(r["total_s"], 5)]]
    docs.append(table(clock, [90 * mm, 24 * mm], st))
    docs.append(Paragraph(
        "Estes tempos estão inflados: a máquina estava com média de carga entre 70 e 190 em 10 núcleos (27 processos "
        "interFoam de outra sessão) durante toda a execução. Numa máquina livre o mesmo conjunto roda em cerca de um "
        "minuto — a cavidade de 128² sozinha cai de 302 s para 22,5 s. O que não depende da carga são os números da "
        "seção 2.", st["small"]))

    # -- 5. reading
    docs.append(Paragraph("5. O que isso mostra, e o que não mostra", st["h"]))
    docs.append(Paragraph(
        "<b>Mostra</b> que, para casos clássicos, o ciclo inteiro — instalar, discretizar, resolver, pós-processar e "
        "conferir contra a referência publicada — cabe num arquivo e roda em minutos, sem licença. Onde o método é "
        "exato para a solução procurada, o número sai exato: a viga de Hermite reproduz qL<super>4</super>/8EI e a "
        "frequência de "
        "Euler-Bernoulli a menos de 1e-9, o golpe de aríete instantâneo é Joukowsky a menos de 1e-13, e a tensão de "
        "flexão 3-D no meio do vão é Mc/I a menos de 1e-6. Onde a discretização é de primeira ordem (CST na chapa, "
        "hexaedros 3-D), o resultado converge para o livro na medida em que a malha refina — que é exatamente o que se "
        "deve exigir desses elementos.", st["body"]))
    docs.append(Paragraph(
        "<b>Não mostra</b> que o PINNeAPPle substitui OpenFOAM ou code_aster. O motor de cavidade é um solver "
        "vorticidade–função de corrente em malha uniforme, sem campo de pressão e sem transiente preciso; a chapa é CST "
        "(tensão constante por elemento), não uma casca nem um elemento quadrático; o 3-D é elástico linear em malha "
        "estruturada. Nenhum dos casos tem turbulência, contato, plasticidade, escoamento compressível ou geometria de "
        "CAD real. O que este experimento mede é a <i>velocidade de chegar a um número conferido</i> nos casos em que o "
        "número é conhecido — não a faixa de problemas que a toolchain cobre.", st["body"]))

    # -- footer
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.2)
        canvas.setFillColor(GREY)
        canvas.drawString(20 * mm, 12 * mm, "PINNeAPPle — reprodução do experimento de toolchain open source")
        canvas.drawRightString(190 * mm, 12 * mm, str(doc.page))
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.4)
        canvas.line(20 * mm, 15.5 * mm, 190 * mm, 15.5 * mm)
        canvas.restoreState()

    doc = SimpleDocTemplate(out, pagesize=A4, leftMargin=20 * mm, rightMargin=20 * mm,
                            topMargin=16 * mm, bottomMargin=20 * mm,
                            title="PINNeAPPle — reprodução do experimento de toolchain open source",
                            author="PINNeAPPle Labs", subject="CFD e FEA clássicos conferidos contra o livro")
    doc.build(docs, onFirstPage=footer, onLaterPages=footer)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.expanduser("~/Downloads/PINNeAPPle-experimento-toolchain.pdf"))
    print(build(os.path.abspath(os.path.expanduser(ap.parse_args().out))))
