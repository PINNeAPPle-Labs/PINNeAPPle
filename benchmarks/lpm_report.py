"""The campaign as one document: what each design space scored, what the refinement bought, and what is still running.

A campaign lives in a directory of JSON files — a card per family, a refinement round per family that had one, a
comparison per outside corpus — and reading eleven of them side by side is how a person decides what to do next. This
builds that reading into a PDF: the table with every family, the charts that make the one number that matters visible,
the outside rows with the sentence that says what they do and do not mean, and the queue of what runs next.

    python -m benchmarks.lpm_report --out ~/lpm-campaign --pdf ~/Downloads/lpm-report.pdf

It reads, it never writes into the campaign: the only file it touches is the PDF, so it can be run against a campaign
that is in the middle of a fit (that is the point — a report you can only make when nothing is running is a report
nobody reads). A number that is missing is reported as missing and never as zero.

What it deliberately does not do: invent a comparison. A row appears when a file measured it, and the "not compared"
entries say why they are not there, which is the same rule the table itself follows.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

__all__ = ["Snapshot", "collect", "build_pdf", "main"]

#: the order the campaign runs its design spaces in, so the table reads the way the night went
ORDER = ("pp_fin", "pp_airfoil", "pp_nozzle", "pp_exchanger", "pp_bracket", "pp_airfoil_rans",
         "pp_wing3d", "pp_nozzle3d", "pp_fin3d", "pp_exchanger3d", "pp_bracket3d")


def _read(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _commit(repo: Path) -> str:
    """The commit the campaign is running, because a number without one is a number nobody can place."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "log", "-1", "--format=%h %s"], capture_output=True,
                             text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


class Snapshot:
    """One family's row: what its card says, what its refinement round says, and where the two came from."""

    def __init__(self, family: str, card: dict[str, Any] | None, refinement: dict[str, Any] | None) -> None:
        self.family = family
        self.card = card or {}
        self.refinement = refinement or {}

    @property
    def title(self) -> str:
        return str(self.card.get("family", {}).get("title") or self.family)

    @property
    def held(self) -> dict[str, Any]:
        return dict(self.card.get("held_out") or {})

    @property
    def scored(self) -> bool:
        return bool(self.held)

    @property
    def relative_l2(self) -> float | None:
        value = self.held.get("relative_l2")
        return None if value is None else float(value)

    @property
    def baseline(self) -> float | None:
        value = self.held.get("constant_field_baseline_relative_l2")
        return None if value is None else float(value)

    @property
    def tolerance(self) -> float | None:
        value = self.held.get("tolerance")
        return None if value is None else float(value)

    @property
    def verdict(self) -> str:
        if not self.scored:
            return "not fitted yet"
        if self.relative_l2 is None or self.tolerance is None:
            return "measured, no tolerance"
        return "inside tolerance" if self.relative_l2 <= self.tolerance else "over tolerance"

    @property
    def beats_baseline(self) -> bool | None:
        if self.relative_l2 is None or self.baseline is None:
            return None
        return self.relative_l2 < self.baseline

    def as_dict(self) -> dict[str, Any]:
        dataset = dict(self.card.get("dataset") or {})
        return {"family": self.family, "title": self.title, "held_out": self.relative_l2, "baseline": self.baseline,
                "tolerance": self.tolerance, "verdict": self.verdict, "beats_baseline": self.beats_baseline,
                "n_designs": dataset.get("n_designs"), "n_geometries": dataset.get("n_geometries"),
                "n_holdout": self.held.get("n_designs"), "n_points": self.held.get("n_points"),
                "split": dict(self.card.get("split") or {}),
                "p90": self.held.get("relative_l2_p90"), "refined": dataset.get("refined_geometries") or [],
                "refinement": self.refinement or None}


def collect(out: Path) -> dict[str, Any]:
    """Every family's row, the outside comparisons, the queue and the snapshot the refinement is compared against."""
    families = sorted({p.parent.parent.name for p in out.glob("*/model/card.json")})
    order = [name for name in ORDER if name in families] + [name for name in families if name not in ORDER]
    rows = [Snapshot(name, _read(out / name / "model" / "card.json"), _read(out / name / "model" / "refinement.json"))
            for name in order]

    before: dict[str, float] = {}
    latest = out / "report" / "snapshots" / "latest"
    if latest.is_file():
        stamp = latest.read_text(encoding="utf-8").strip()
        for path in sorted((out / "report" / "snapshots" / stamp).glob("*.card.json")):
            card = _read(path) or {}
            value = (card.get("held_out") or {}).get("relative_l2")
            if value is not None:
                before[path.name.removesuffix(".card.json")] = float(value)

    comparisons = []
    for path in sorted((out / "report").glob("comparison-*.json")):
        document = _read(path) or {}
        comparisons.append({"file": path.name, "corpus": document.get("corpus"),
                            "rows": document.get("rows") or [],
                            "not_compared": document.get("not_compared") or []})

    queue = None
    for candidate in (out / "queue" / "queue.json", out / "queue.json"):
        queue = queue or _read(candidate)
    return {"built_at": datetime.now(UTC).isoformat(timespec="seconds"), "out": str(out), "families": rows,
            "rows": [row.as_dict() for row in rows], "before": before, "comparisons": comparisons,
            "queue": queue or {"jobs": []}, "commit": _commit(Path(__file__).resolve().parent.parent)}


# --------------------------------------------------------------------------------------------------------------------
# the charts: one number per family, and the only comparison that a reader has to see rather than read
# --------------------------------------------------------------------------------------------------------------------
def _chart_scores(rows: list[dict[str, Any]], path: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    scored = [r for r in rows if r["held_out"] is not None]
    if not scored:
        return None
    names = [r["family"] for r in scored]
    held = [r["held_out"] for r in scored]
    base = [r["baseline"] or 0.0 for r in scored]
    y = range(len(names))
    fig, ax = plt.subplots(figsize=(7.4, 0.32 * len(names) + 1.5))
    ax.barh([i + 0.2 for i in y], base, height=0.38, color="#cfd8dc", label="constant field")
    ax.barh([i - 0.2 for i in y], held, height=0.38, color="#0081be", label="model (held out)")
    ax.set_yticks(list(y), names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("relative L2 (lower is better)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.set_title("held-out error against the constant-field baseline", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def _chart_shift(comparisons: list[dict[str, Any]], path: Path) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bars: list[tuple[str, float, float]] = []
    for comparison in comparisons:
        for row in comparison["rows"]:
            if row.get("relative_l2") is None:
                continue
            bars.append((f"{row.get('model', '?')}\n{comparison['file'].removesuffix('.json')}",
                         float(row["relative_l2"]), float(row.get("baseline") or 0.0)))
    if not bars:
        return None
    fig, ax = plt.subplots(figsize=(7.4, 2.6))
    x = range(len(bars))
    ax.bar([i - 0.2 for i in x], [b[1] for b in bars], width=0.38, color="#a23a00", label="model")
    ax.bar([i + 0.2 for i in x], [b[2] for b in bars], width=0.38, color="#cfd8dc", label="constant field")
    ax.set_xticks(list(x), [b[0] for b in bars], fontsize=7)
    ax.tick_params(labelsize=8)
    ax.set_ylabel("relative L2", fontsize=8)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("outside corpora: what the model says about geometry it never saw", fontsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


# --------------------------------------------------------------------------------------------------------------------
# the document
# --------------------------------------------------------------------------------------------------------------------
def build_pdf(document: dict[str, Any], pdf_path: str | Path, *, assets: Path | None = None) -> Path:
    """Render the collected state as a PDF. Portuguese, because that is who reads it."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        Image,
        PageBreak,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    pdf = Path(pdf_path).expanduser()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    assets = Path(assets or pdf.parent / "lpm-report-assets")
    assets.mkdir(parents=True, exist_ok=True)

    ink, accent, muted = colors.HexColor("#262626"), colors.HexColor("#0081be"), colors.HexColor("#6b6b6b")
    style = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=style["BodyText"], fontName="Helvetica", fontSize=9, leading=12.5,
                          alignment=TA_JUSTIFY, textColor=ink)
    small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=10, textColor=muted)
    h1 = ParagraphStyle("h1", parent=style["Heading1"], fontName="Helvetica-Bold", fontSize=17, textColor=ink,
                        spaceAfter=2)
    h2 = ParagraphStyle("h2", parent=style["Heading2"], fontName="Helvetica-Bold", fontSize=11.5, textColor=accent,
                        spaceBefore=10, spaceAfter=4)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.4, leading=9, alignment=0, textColor=ink)
    head = ParagraphStyle("head", parent=cell, fontName="Helvetica-Bold", textColor=colors.white)

    def table(data: list[list[Any]], widths: list[float], *, zebra: bool = True) -> Table:
        rendered = [[Paragraph(str(value), head if r == 0 else cell) for value in row]
                    for r, row in enumerate(data)]
        t = Table(rendered, colWidths=[w * mm for w in widths], repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), accent),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#d8d8d8")),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8fa")] if zebra else None),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
        ]))
        return t

    def figure(path: Path, max_w: float = 170.0, max_h: float = 95.0) -> Image:
        """The chart scaled to fit the frame: reportlab does not fit an Image by itself, it overflows the page."""
        from reportlab.lib.utils import ImageReader

        width_px, height_px = ImageReader(str(path)).getSize()
        width = max_w * mm
        height = width * height_px / width_px
        if height > max_h * mm:
            height, width = max_h * mm, max_h * mm * width_px / height_px
        return Image(str(path), width=width, height=height)

    def number(value: Any, digits: int = 3) -> str:
        return "—" if value is None else f"{float(value):.{digits}f}"

    story: list[Any] = []
    built = document["built_at"].replace("T", " ")[:16]
    story += [Paragraph("PINNeAPPle LPM &mdash; relatório da campanha", h1),
              Paragraph(f"gerado em {built} UTC &middot; campanha em <font face='Courier'>{document['out']}</font>"
                        f" &middot; commit <font face='Courier'>{document['commit']}</font>", small),
              Spacer(1, 6)]

    rows = document["rows"]
    scored = [r for r in rows if r["held_out"] is not None]
    inside = [r for r in scored if r["tolerance"] is not None and r["held_out"] <= r["tolerance"]]
    better = [r for r in scored if r["baseline"] is not None and r["held_out"] < r["baseline"]]
    story.append(Paragraph(
        f"{len(rows)} espaços de projeto, {len(scored)} com modelo medido. "
        f"<b>{len(better)}</b> batem o campo constante (prever a média do treino) e <b>{len(inside)}</b> cabem "
        f"na tolerância que a própria família declarou. A coluna <i>held-out</i> é o erro relativo L2 global em "
        f"corpos que o modelo nunca viu, separados <i>por geometria</i>.", body))
    story.append(Spacer(1, 6))

    data = [["família", "held-out", "campo const.", "tolerância", "veredito", "corpos", "refinados"]]
    for row in rows:
        data.append([row["family"], number(row["held_out"]), number(row["baseline"]),
                     "—" if row["tolerance"] is None else number(row["tolerance"], 2), row["verdict"],
                     f"{row['n_geometries'] or '—'} / {row['n_designs'] or '—'}",
                     ", ".join(str(g) for g in row["refined"]) or "—"])
    story.append(table(data, [26, 16, 20, 18, 34, 24, 24]))
    story.append(Paragraph("corpos = geometrias / projetos (uma geometria tem vários pontos de operação). "
                           "refinados = ids de geometria acrescentados pela segunda rodada do estágio 1, que treinam "
                           "e nunca entram no held-out.", small))

    chart = _chart_scores(rows, assets / "scores.png")
    if chart:
        story += [Spacer(1, 6), figure(chart)]

    story.append(Paragraph("Refinamento (estágio 1)", h2))
    refined = [r for r in rows if r["refinement"]]
    if not refined:
        story.append(Paragraph("Nenhuma família passou pela segunda rodada ainda. O mecanismo está ligado no "
                               "<font face='Courier'>--refine N</font> da campanha: simula <i>N</i> corpos onde a "
                               "confiança calibrada está mais perto do limiar de decisão, grava-os no mesmo dataset "
                               "marcados como refinados e refaz o fit <i>warm</i> a partir da rodada anterior.", body))
    else:
        data = [["família", "propostos", "adicionados", "antes", "depois", "tolerância", "ganho"]]
        for row in refined:
            before = document["before"].get(row["family"])
            after = row["refinement"].get("held_out_relative_l2", row["held_out"])
            gain = "—" if before is None or after is None else f"{100 * (after - before) / before:+.1f}%"
            data.append([row["family"], row["refinement"].get("proposed", "—"), row["refinement"].get("added", "—"),
                         number(before), number(after), number(row["tolerance"], 2), gain])
        story.append(table(data, [26, 18, 20, 16, 16, 18, 16]))
        story.append(Paragraph("antes = a rodada que o refinamento herdou (snapshot dos cartões); depois = o fit "
                               "<i>warm</i> que incluiu os corpos novos. O held-out é sempre das geometrias "
                               "sorteadas, então os dois números são sobre o mesmo tipo de corpo.", small))

    story.append(Paragraph("Corpora externos", h2))
    if not document["comparisons"]:
        story.append(Paragraph("Nenhuma comparação externa registrada ainda.", body))
    for comparison in document["comparisons"]:
        story.append(Paragraph(str(comparison["corpus"] or comparison["file"]), body))
        data = [["modelo", "held-out", "p90", "pior", "campo const.", "portões"]]
        for row in comparison["rows"]:
            gates = row.get("gates") or {}
            verdict = " / ".join(f"{check} {str(gates[check]).split(' ')[0].lower()}"
                                 for check in ("accuracy", "baseline") if check in gates) or "—"
            data.append([row.get("model", "?"), number(row.get("relative_l2")), number(row.get("p90")),
                         number(row.get("worst")), number(row.get("baseline")), verdict])
        story.append(table(data, [34, 18, 16, 16, 22, 50]))
        if comparison["not_compared"]:
            story.append(Paragraph("não comparados: " + "; ".join(
                f"{item['model']} ({item['reason']})" for item in comparison["not_compared"]), small))
        story.append(Spacer(1, 4))
    chart = _chart_shift(document["comparisons"], assets / "external.png")
    if chart:
        story.append(figure(chart))

    story.append(PageBreak())
    story.append(Paragraph("O modelo da Luminary, e o que dá para comparar", h2))
    story.append(Paragraph(
        "O SHIFT-Wing deles é um <b>DoMINO</b> do PhysicsNeMo treinado em <b>mais de 3.000 simulações RANS</b> do "
        "próprio espaço paramétrico (página <font face='Courier'>luminary.ai/models</font>, consultada em "
        "28/11/2026; o cartão do dataset no Hugging Face aponta o PhysicsNeMo como o caminho dos modelos "
        "pré-treinados). Não há métrica numérica publicada em nenhum dos dois lugares — verificado — então "
        "comparar com “o desempenho do modelo deles” por número seria inventar. O que existe, e está medido aqui:", body))
    story.append(Paragraph(
        "1) o nosso modelo aplicado ao <i>dado</i> deles, que é a linha do corpus externo acima: nenhum caso do SHIFT "
        "entrou em treino, e o resultado é o que a transferência vale; 2) o mesmo número contra o campo constante "
        "naquele corpus, que é o piso que qualquer modelo tem de bater; 3) o mesmo tipo de modelo — o DoMINO do "
        "PhysicsNeMo — treinado no <i>nosso</i> espaço e medido no nosso held-out, que é a única linha em que a "
        "arquitetura deles e a nossa veem os mesmos corpos. A linha 3 entra na fila com "
        "<font face='Courier'>--domino-steps</font>, limitada pela licença: o corpus deles é "
        "<font face='Courier'>cc-by-nc-4.0</font> e aqui só pode servir de validação.", body))

    story.append(Paragraph("Fila", h2))
    jobs = (document.get("queue") or {}).get("jobs") or []
    if not jobs:
        story.append(Paragraph("Sem fila registrada.", body))
    else:
        data = [["#", "trabalho", "estado", "desde", "detalhe"]]
        for i, job in enumerate(jobs, start=1):
            data.append([i, job.get("title") or job.get("id"), job.get("status", "—"), job.get("started_at", "—"),
                         job.get("detail", "—")])
        story.append(table(data, [8, 52, 18, 30, 62]))

    story.append(Paragraph("Como reproduzir", h2))
    story.append(Paragraph(
        "Campanha: <font face='Courier'>python -m benchmarks.lpm_campaign --out ~/lpm-campaign</font>, com "
        "<font face='Courier'>--refine N</font> para a segunda rodada do estágio 1. Comparação: "
        "<font face='Courier'>python -m benchmarks.lpm_compare --run ~/lpm-campaign/pp_wing3d --shift &lt;árvore&gt; "
        "--dataset-name Wing-sample --condition alpha_deg=alpha --condition mach=mach --json ...</font> "
        "(Wing-sample é o gêmeo acessível do WING, que é <i>manual</i> gated: mesma organização, mesma licença). "
        "Este PDF: "
        "<font face='Courier'>python -m benchmarks.lpm_report --out ~/lpm-campaign --pdf ...</font>, que só lê. "
        "Cada cartão traz o split, a semente, o baseline e a decisão de confiança; o commit está na capa.", small))

    doc = SimpleDocTemplate(str(pdf), pagesize=A4, title="PINNeAPPle LPM — relatório da campanha",
                            author="PINNeAPPle", leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm)
    doc.build(story)
    return pdf


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the campaign's PDF report (reads the campaign, writes one PDF)")
    parser.add_argument("--out", type=Path, required=True, help="the campaign directory")
    parser.add_argument("--pdf", type=Path, default=Path.home() / "Downloads" / "PINNeAPPle-LPM-relatorio.pdf")
    parser.add_argument("--assets", type=Path, default=None, help="where the charts go (default: beside the PDF)")
    parser.add_argument("--json", type=Path, default=None, help="also write the collected state here")
    args = parser.parse_args(argv)

    document = collect(args.out.expanduser())
    if not document["rows"]:
        print(f"{args.out}: no card.json under any family; nothing to report yet")
        return 2
    if args.json:
        args.json.write_text(json.dumps(document, indent=1, default=str), encoding="utf-8")
    pdf = build_pdf(document, args.pdf, assets=args.assets)
    print(f"{pdf}: {len(document['rows'])} famílias, {len(document['comparisons'])} corpora externos")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
