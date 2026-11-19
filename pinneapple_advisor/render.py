"""The same plan, drawn four ways: Mermaid source, Graphviz DOT, a self-contained SVG, and an HTML page.

The diagram is a rendering of the plan object, never a second description of it: the boxes, their order, their
status colours and the arrows all come from ``StudyPlan.stages`` and each stage's ``depends_on``. A stage that the
user already has is green, one to be built is blue, one to confirm is amber, an optional one is grey and one that is
not needed is dashed.

The SVG is generated here rather than by a JavaScript library so the file opens on a machine with no network -- the
same reason the library's other viewers are self-contained. Mermaid is still offered, because GitHub and most
documentation tools render it from the source text.
"""
from __future__ import annotations

import html as _html
import re
from collections.abc import Sequence

from .schema import Stage, StudyPlan, text_of

STATUS_STYLE = {
    "have": {"fill": "#e6f4ea", "stroke": "#1e7b34", "label_pt": "você já tem", "label_en": "you already have"},
    "generate": {"fill": "#e7f0fb", "stroke": "#0b5cad", "label_pt": "a fazer", "label_en": "to build"},
    "confirm": {"fill": "#fdf3e0", "stroke": "#a23a00", "label_pt": "a confirmar", "label_en": "to confirm"},
    "optional": {"fill": "#f2f2f2", "stroke": "#6b6b6b", "label_pt": "opcional", "label_en": "optional"},
    "skip": {"fill": "#ffffff", "stroke": "#b0b0b0", "label_pt": "não precisa", "label_en": "not needed"},
}

KIND_LABEL = {
    "physics": ("física", "physics"), "geometry": ("geometria", "geometry"), "mesh": ("malha", "mesh"),
    "solver": ("solver", "solver"), "dataset": ("dataset", "dataset"), "training": ("treino", "training"),
    "inverse": ("inversão", "inverse"), "optimization": ("otimização", "optimisation"),
    "verification": ("verificação", "verification"), "visualization": ("visualização", "visualisation"),
}


def diagram(plan: StudyPlan, kind: str = "mermaid", lang: str | None = None) -> str:
    """``mermaid``, ``dot``, ``svg``, ``html`` or ``markdown`` -- every one built from ``plan``."""
    lang = lang or str(plan.provenance.get("language") or "pt")
    if kind == "mermaid":
        return mermaid(plan, lang)
    if kind == "dot":
        return dot(plan, lang)
    if kind == "svg":
        return svg(plan, lang)
    if kind == "html":
        return html(plan, lang)
    if kind in ("markdown", "md"):
        return markdown(plan, lang)
    raise ValueError(f"unknown diagram kind {kind!r}; use mermaid, dot, svg, html or markdown")


# ---------------------------------------------------------------------------------------------------- helpers

_TAG = re.compile(r"<[^>]*>")


def _safe_label(text: str) -> str:
    """A label for a source-text diagram (Mermaid, DOT): no markup and no double quotes, so a problem description
    cannot carry a tag into a renderer that treats labels as HTML."""
    return _TAG.sub("", str(text)).replace('"', "'").strip()


def _label(stage: Stage, lang: str) -> str:
    kind = KIND_LABEL.get(stage.kind, (stage.kind, stage.kind))[0 if lang == "pt" else 1]
    status = STATUS_STYLE.get(stage.status, STATUS_STYLE["generate"])
    return f"{kind} · {text_of(stage.title, lang)} — {status['label_' + ('pt' if lang == 'pt' else 'en')]}"


def _wrap(text: str, width: int = 38, max_lines: int = 4) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines[:max_lines]


def layers(plan: StudyPlan) -> list[list[Stage]]:
    """The stages grouped by dependency depth: every stage sits one row below the deepest stage it depends on."""
    by_id = {s.id: s for s in plan.stages}
    depth: dict[str, int] = {}

    def resolve(stage: Stage, seen: tuple[str, ...] = ()) -> int:
        if stage.id in depth:
            return depth[stage.id]
        if stage.id in seen:                     # a cycle: stop, the plan never has one and rendering must not hang
            return 0
        deps = [by_id[d] for d in stage.depends_on if d in by_id]
        value = 1 + max([resolve(d, seen + (stage.id,)) for d in deps], default=-1)
        depth[stage.id] = value
        return value

    for s in plan.stages:
        resolve(s)
    rows: dict[int, list[Stage]] = {}
    for s in plan.stages:
        rows.setdefault(depth[s.id], []).append(s)
    return [rows[k] for k in sorted(rows)]


def _ordered_rows(plan: StudyPlan) -> list[list[Stage]]:
    """The dependency rows, with each row ordered by the average column of its parents: a child sits under what it
    needs, which is most of what keeps the arrows from crossing."""
    rows = layers(plan)
    column: dict[str, float] = {}
    ordered: list[list[Stage]] = []
    for i, row in enumerate(rows):
        if i:                                    # keep the order stable within equal barycentres
            parent_x = {s.id: [column[d] for d in s.depends_on if d in column] for s in row}
            row = sorted(row, key=lambda s: (sum(parent_x[s.id]) / len(parent_x[s.id])) if parent_x[s.id] else 0.0)
        for j, stage in enumerate(row):
            column[stage.id] = j
        ordered.append(row)
    return ordered


def _estimate_line(plan: StudyPlan, lang: str) -> str:
    est = plan.total_estimate().to_dict()
    hours = est["total_hours"]
    money = est.get("cost_usd")
    if lang == "pt":
        text = f"Estimativa da campanha: {hours:.2f} h de máquina"
        if money:
            text += f", US$ {money:.0f}"
        return text + f" ({est['confidence']})"
    text = f"Campaign estimate: {hours:.2f} machine-hours"
    if money:
        text += f", US$ {money:.0f}"
    return text + f" ({est['confidence']})"


# ---------------------------------------------------------------------------------------------------- mermaid

def mermaid(plan: StudyPlan, lang: str = "pt") -> str:
    """Mermaid ``flowchart TD``: one node per stage, coloured by status, one arrow per dependency."""
    lines = ["flowchart TD"]
    for s in plan.stages:
        lines.append(f'  {s.id}["{_safe_label(_label(s, lang))}"]:::{s.status}')
    for s in plan.stages:
        for dep in s.depends_on:
            if any(x.id == dep for x in plan.stages):
                lines.append(f"  {dep} --> {s.id}")
    for status, style in STATUS_STYLE.items():
        dash = ",stroke-dasharray: 4 3" if status == "skip" else ""
        lines.append(f"  classDef {status} fill:{style['fill']},stroke:{style['stroke']},stroke-width:1.5px{dash}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- dot

def dot(plan: StudyPlan, lang: str = "pt") -> str:
    lines = ["digraph pipeline {", '  rankdir=TB;', '  node [shape=box, style="rounded,filled", fontname="Helvetica"];']
    for s in plan.stages:
        style = STATUS_STYLE.get(s.status, STATUS_STYLE["generate"])
        label = _safe_label(_label(s, lang))
        lines.append(f'  {s.id} [label="{label}", fillcolor="{style["fill"]}", color="{style["stroke"]}"];')
    for s in plan.stages:
        for dep in s.depends_on:
            if any(x.id == dep for x in plan.stages):
                lines.append(f"  {dep} -> {s.id};")
    lines.append("}")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------- svg

_MIN_W = 820
_BOX_W = 300
_BOX_H = 98
_GAP_X = 40
_GAP_Y = 96
_MARGIN = 36
_HEADER = 118


def _node_lines(stage: Stage, lang: str) -> list[str]:
    label = text_of(stage.title, lang)
    return _wrap(label, 36, 3) + [_status_text(stage, lang)]


def _status_text(stage: Stage, lang: str) -> str:
    style = STATUS_STYLE.get(stage.status, STATUS_STYLE["generate"])
    kind = KIND_LABEL.get(stage.kind, (stage.kind, stage.kind))
    return f"[{kind[0] if lang == 'pt' else kind[1]}] {style['label_' + ('pt' if lang == 'pt' else 'en')]}"


def _layout(plan: StudyPlan) -> tuple[dict[str, tuple[float, float]], float, float]:
    """Where every stage sits, and the size of the canvas: the one layout the SVG and the video both draw."""
    rows = _ordered_rows(plan)
    widest = max((len(r) for r in rows), default=1)
    # at least wide enough for the header lines: a one-per-row plan is narrow, and a clipped subtitle reads as a bug
    width = max(_MARGIN * 2 + widest * _BOX_W + (widest - 1) * _GAP_X, _MIN_W)
    height = _HEADER + max(len(rows), 1) * (_BOX_H + _GAP_Y) + 64
    pos: dict[str, tuple[float, float]] = {}
    y = _HEADER
    for row in rows:
        row_w = len(row) * _BOX_W + (len(row) - 1) * _GAP_X
        x = (width - row_w) / 2
        for stage in row:
            pos[stage.id] = (x, y)
            x += _BOX_W + _GAP_X
        y += _BOX_H + _GAP_Y
    return pos, width, height


def svg(plan: StudyPlan, lang: str = "pt") -> str:
    """A self-contained SVG: rows are the dependency depth, arrows are the ``depends_on`` edges."""
    pos, width, height = _layout(plan)
    body: list[str] = []

    # edges first, so the boxes cover their ends
    for stage in plan.stages:
        sx, sy = pos[stage.id]
        for dep in stage.depends_on:
            if dep not in pos:
                continue
            dx, dy = pos[dep]
            x1, y1 = dx + _BOX_W / 2, dy + _BOX_H
            x2, y2 = sx + _BOX_W / 2, sy
            mid = (y1 + y2) / 2
            body.append(f'<path d="M {x1:.0f} {y1:.0f} C {x1:.0f} {mid:.0f}, {x2:.0f} {mid:.0f}, {x2:.0f} {y2:.0f}" '
                        f'fill="none" stroke="#9aa0a6" stroke-width="1.6" marker-end="url(#arrow)"/>')

    for stage in plan.stages:
        x, y = pos[stage.id]
        style = STATUS_STYLE.get(stage.status, STATUS_STYLE["generate"])
        dash = ' stroke-dasharray="5 4"' if stage.status == "skip" else ""
        body.append(f'<g><rect x="{x:.0f}" y="{y:.0f}" width="{_BOX_W}" height="{_BOX_H}" rx="12" '
                    f'fill="{style["fill"]}" stroke="{style["stroke"]}" stroke-width="1.6"{dash}/>')
        # the title wraps to at most two lines and is elided with an ellipsis rather than cut mid-word
        lines = _wrap(text_of(stage.title, lang), 36, 2)
        if len(lines) == 2 and len(" ".join(_wrap(text_of(stage.title, lang), 36, 9))) > len(" ".join(lines)):
            lines[1] = lines[1][:33].rstrip() + "…"
        for i, line in enumerate(lines):
            body.append(f'<text x="{x + 14:.0f}" y="{y + 22 + i * 16:.0f}" '
                        f'font-family="Helvetica, Arial, sans-serif" font-size="12.5" font-weight="600" '
                        f'fill="#262626">{_html.escape(line)}</text>')
        sub = _status_text(stage, lang)
        if stage.estimate is not None:
            sub += f" · {_human_time(stage.estimate.total_seconds)}"
        status_y = y + 22 + 16 * len(lines) + 13
        body.append(f'<text x="{x + 14:.0f}" y="{status_y:.0f}" font-family="Helvetica, Arial, sans-serif" '
                    f'font-size="11" fill="{style["stroke"]}">{_html.escape(sub)}</text>')
        if stage.answers:
            answers = ", ".join(str(a) for a in sorted(stage.answers))
            word = "perguntas" if lang == "pt" and len(stage.answers) > 1 else (
                "pergunta" if lang == "pt" else ("questions" if len(stage.answers) > 1 else "question"))
            body.append(f'<text x="{x + 14:.0f}" y="{status_y + 18:.0f}" '
                        f'font-family="Helvetica, Arial, sans-serif" font-size="10" fill="#6b6b6b">'
                        f'{_html.escape(f"{word} {answers}")}</text>')
        body.append("</g>")

    title = "Pipeline sugerido" if lang == "pt" else "Suggested pipeline"
    subtitle = _estimate_line(plan, lang)
    legend_y = height - 34
    header = [
        f'<text x="{_MARGIN}" y="44" font-family="Helvetica, Arial, sans-serif" font-size="21" font-weight="700" '
        f'fill="#262626">{_html.escape(title)}</text>',
        f'<text x="{_MARGIN}" y="68" font-family="Helvetica, Arial, sans-serif" font-size="13" fill="#4a4a4a">'
        f'{_html.escape(_elide(subtitle, _fit_chars(width - 2 * _MARGIN, 13)))}</text>',
        f'<text x="{_MARGIN}" y="90" font-family="Helvetica, Arial, sans-serif" font-size="11.5" fill="#6b6b6b">'
        f'{_html.escape(_elide(text_of(plan.problem, lang), _fit_chars(width - 2 * _MARGIN, 11.5)))}</text>',
    ]
    legend = []
    x = _MARGIN
    for status in ("have", "generate", "confirm", "optional", "skip"):
        style = STATUS_STYLE[status]
        label = style["label_pt"] if lang == "pt" else style["label_en"]
        legend.append(f'<rect x="{x}" y="{legend_y - 11}" width="14" height="12" rx="3" fill="{style["fill"]}" '
                      f'stroke="{style["stroke"]}"/>')
        legend.append(f'<text x="{x + 20}" y="{legend_y}" font-family="Helvetica, Arial, sans-serif" font-size="11" '
                      f'fill="#4a4a4a">{_html.escape(label)}</text>')
        x += 26 + 8 * len(label)

    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" height="{height:.0f}" '
            f'viewBox="0 0 {width:.0f} {height:.0f}" role="img" aria-label="{_html.escape(title)}">\n'
            f'<defs><marker id="arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#9aa0a6"/></marker></defs>\n'
            f'<rect width="100%" height="100%" fill="#ffffff"/>\n'
            + "\n".join(header) + "\n" + "\n".join(body) + "\n" + "\n".join(legend) + "\n</svg>\n")


def _fit_chars(pixels: float, font_size: float) -> int:
    """How many characters of this font size fit in that many pixels (0.55 em is the average advance of Helvetica)."""
    return max(int(pixels / (font_size * 0.55)), 12)


def _elide(text: str, limit: int) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def _human_time(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f} h"
    if seconds >= 60:
        return f"{seconds / 60:.0f} min"
    return f"{seconds:.0f} s"


# ---------------------------------------------------------------------------------------------------- html / md

def html(plan: StudyPlan, lang: str = "pt") -> str:
    """One self-contained page: the drawing, every stage with its calls/checks/views, the decisions and the open
    questions."""
    t = {k: _t(k, lang) for k in _T}
    rows = []
    for s in plan.stages:
        tools = "".join(f"<li><code>{_html.escape(x.call)}</code> — {_html.escape(text_of(x.note, lang))}</li>"
                        for x in s.tools)
        checks = "".join(f"<li>{_html.escape(text_of(c['what'], lang))} "
                         f"<span class='muted'>({_html.escape(str(c['tool']))})</span></li>"
                         for c in s.checks)
        views = "".join(f"<li>{_html.escape(text_of(v['what'], lang))} "
                        f"<span class='muted'>({_html.escape(str(v['tool']))})</span></li>"
                        for v in s.views)
        est = ""
        if s.estimate is not None:
            d = s.estimate.to_dict()
            est = (f"<p class='est'>{_human_time(d['total_seconds'])} · {d['n_evaluations']} × "
                   f"{d['seconds_per_evaluation']:.3g} s · <span class='muted'>{_html.escape(d['basis'])}</span></p>")
        style = STATUS_STYLE.get(s.status, STATUS_STYLE["generate"])
        rows.append(
            f"<section class='stage' style='border-left:4px solid {style['stroke']}'>"
            f"<h3>{_html.escape(text_of(s.title, lang))} "
            f"<span class='badge' style='background:{style['fill']};color:{style['stroke']}'>"
            f"{_html.escape(style['label_pt'] if lang == 'pt' else style['label_en'])}</span></h3>"
            f"<p>{_html.escape(text_of(s.why, lang))}</p>{est}"
            f"<p class='muted'>{t['answers']}: {', '.join(str(a) for a in s.answers)}</p>"
            f"<h4>{t['tools']}</h4><ul>{tools}</ul><h4>{t['checks']}</h4><ul>{checks or '<li>' + t['none'] + '</li>'}</ul>"
            f"<h4>{t['views']}</h4><ul>{views or '<li>' + t['none'] + '</li>'}</ul></section>")
    decisions = "".join(
        f"<tr><td>{_html.escape(d.topic)}</td><td><strong>{_html.escape(str(d.choice))}</strong></td>"
        f"<td>{_html.escape(text_of(d.why, lang))}</td>"
        f"<td class='muted'>{_html.escape(d.source)}</td></tr>" for d in plan.decisions)
    questions = "".join(f"<li><strong>{_html.escape(q['id'])}</strong> — "
                        f"{_html.escape(q['pt'] if lang == 'pt' else q['en'])}</li>" for q in plan.open_questions)
    warnings = "".join(f"<li>{_html.escape(text_of(w, lang))}</li>" for w in plan.warnings)
    title = "Pipeline sugerido" if lang == "pt" else "Suggested pipeline"
    return f"""<!doctype html>
<html lang="{lang}"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_html.escape(title)}</title>
<style>
:root {{ --ink:#262626; --muted:#6b6b6b; --line:#e3e3e3; --brand:#0081be; }}
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; color: var(--ink);
       margin: 0; padding: 28px; max-width: 1080px; }}
h1 {{ font-size: 24px; margin: 0 0 4px; }} h2 {{ font-size: 18px; margin-top: 34px; }}
h3 {{ font-size: 15px; margin: 0 0 6px; }} h4 {{ font-size: 12px; text-transform: uppercase; letter-spacing: .06em;
     color: var(--muted); margin: 12px 0 4px; }}
.muted {{ color: var(--muted); }} .est {{ font-size: 13px; }}
.badge {{ font-size: 11px; padding: 2px 8px; border-radius: 999px; margin-left: 8px; vertical-align: middle; }}
.stage {{ border: 1px solid var(--line); border-radius: 10px; padding: 14px 18px; margin: 14px 0; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
td, th {{ border-bottom: 1px solid var(--line); padding: 8px; text-align: left; vertical-align: top; }}
code {{ background: #f5f6f7; padding: 1px 5px; border-radius: 4px; font-size: 12px; }}
ul {{ margin: 4px 0 0 18px; padding: 0; }} li {{ margin: 2px 0; font-size: 13px; }}
.diagram {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 10px; }}
</style></head><body>
<h1>{_html.escape(title)}</h1>
<p class="muted">{_html.escape(_estimate_line(plan, lang))}</p>
<div class="diagram">{svg(plan, lang)}</div>
<h2>{t['decisions']}</h2>
<table><tr><th>{t['topic']}</th><th>{t['choice']}</th><th>{t['why']}</th><th>{t['source']}</th></tr>{decisions}</table>
{"<h2>" + t['warnings'] + "</h2><ul>" + warnings + "</ul>" if warnings else ""}
<h2>{t['questions']}</h2><ul>{questions or '<li>' + t['none'] + '</li>'}</ul>
<h2>{t['stages']}</h2>{"".join(rows)}
<p class="muted">{t['foot']}</p>
</body></html>
"""


_T = {
    "answers": {"pt": "Responde às perguntas", "en": "Answers questions"},
    "tools": {"pt": "Chamadas PINNeAPPle", "en": "PINNeAPPle calls"},
    "checks": {"pt": "Checagens", "en": "Checks"},
    "views": {"pt": "O que olhar", "en": "What to look at"},
    "decisions": {"pt": "Decisões e o porquê", "en": "Decisions and why"},
    "warnings": {"pt": "Avisos", "en": "Warnings"},
    "questions": {"pt": "Perguntas em aberto", "en": "Open questions"},
    "stages": {"pt": "Etapas em detalhe", "en": "Stages in detail"},
    "none": {"pt": "nenhuma", "en": "none"},
    "topic": {"pt": "Assunto", "en": "Topic"},
    "choice": {"pt": "Escolha", "en": "Choice"},
    "why": {"pt": "Por quê", "en": "Why"},
    "source": {"pt": "Fonte", "en": "Source"},
    "foot": {"pt": "Sugestão gerada pela biblioteca PINNeAPPle: nada aqui foi executado.",
             "en": "Suggestion generated by the PINNeAPPle library: nothing here was executed."},
}


def _t(key: str, lang: str) -> str:
    return _T[key][lang if lang in ("pt", "en") else "pt"]



def markdown(plan: StudyPlan, lang: str = "pt") -> str:
    """A page for a repository: the Mermaid source, the stages as a table and the decisions as a list."""
    title = "Pipeline sugerido" if lang == "pt" else "Suggested pipeline"
    out = [f"# {title}", "", f"_{_estimate_line(plan, lang)}_", ""]
    if text_of(plan.problem, lang):
        out += [f"> {text_of(plan.problem, lang)}", ""]
    out += ["```mermaid", mermaid(plan, lang).rstrip(), "```", ""]
    head = ("| # | etapa | tipo | situação | checagem principal |", "|---|---|---|---|---|") if lang == "pt" else (
        "| # | stage | kind | status | main check |", "|---|---|---|---|---|")
    out += [head[0], head[1]]
    for i, s in enumerate(plan.stages, 1):
        status = STATUS_STYLE.get(s.status, STATUS_STYLE["generate"])
        check = text_of(s.checks[0]["what"], lang) if s.checks else ""
        out.append(f"| {i} | {text_of(s.title, lang)} | {KIND_LABEL.get(s.kind, (s.kind, s.kind))[0 if lang == 'pt' else 1]} "
                   f"| {status['label_pt' if lang == 'pt' else 'label_en']} | {check} |")
    out += ["", f"## {'Decisões' if lang == 'pt' else 'Decisions'}", ""]
    for d in plan.decisions:
        out.append(f"- **{d.topic}: {d.choice}** — {text_of(d.why, lang)}")
    if plan.open_questions:
        out += ["", f"## {'Perguntas em aberto' if lang == 'pt' else 'Open questions'}", ""]
        out += [f"- `{q['id']}` — {q['pt'] if lang == 'pt' else q['en']}" for q in plan.open_questions]
    return "\n".join(out) + "\n"


def animate(plan: StudyPlan, path: str, *, lang: str | None = None, fps: int = 4, dpi: int = 110,
            hold_s: float = 2.5, reveal: int = 3) -> str:
    """The same plan as a video: the stages appear one by one, each with its status, its estimate and the arrows
    that carry it from what it needs.

    ``.mp4`` writes with ffmpeg (when the machine has it), anything else writes an animated ``.gif``; both come from
    matplotlib, which the library already depends on. The drawing is the SVG's own layout, so the video and the page
    cannot disagree. Returns the path written.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.animation import FFMpegWriter, PillowWriter
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    lang = lang or str(plan.provenance.get("language") or "pt")
    pos, width, height = _layout(plan)
    # the drawing's own legend strip is not in the video: the frame ends just below the last box
    height = max(y for _, y in pos.values()) + _BOX_H + 46 if pos else height
    scale = 100.0
    fig = plt.figure(figsize=(width / scale, height / scale), dpi=dpi)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)                     # y grows downwards, like the SVG
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    def header() -> None:
        ax.text(_MARGIN, 46, "Pipeline sugerido" if lang == "pt" else "Suggested pipeline", fontsize=19,
                fontweight="bold", color="#262626", va="center")
        ax.text(_MARGIN, 70, _elide(_estimate_line(plan, lang), _fit_chars(width - 2 * _MARGIN, 13)), fontsize=11,
                color="#4a4a4a", va="center")
        ax.text(_MARGIN, 92, _elide(text_of(plan.problem, lang), _fit_chars(width - 2 * _MARGIN, 11.5)),
                fontsize=9.5, color="#6b6b6b", va="center")

    def boxes(shown: Sequence[Stage], alpha: float) -> None:
        for stage in shown:
            a = alpha if stage is shown[-1] else 1.0
            x, y = pos[stage.id]
            style = STATUS_STYLE.get(stage.status, STATUS_STYLE["generate"])
            ax.add_patch(FancyBboxPatch((x, y), _BOX_W, _BOX_H, boxstyle="round,pad=0,rounding_size=12",
                                        linewidth=1.6, edgecolor=style["stroke"], facecolor=style["fill"],
                                        alpha=a, mutation_aspect=1))
            lines = _wrap(text_of(stage.title, lang), 33, 2)
            if len(lines) == 2 and len(" ".join(_wrap(text_of(stage.title, lang), 33, 9))) > len(" ".join(lines)):
                lines[1] = lines[1][:30].rstrip() + "…"
            for i, line in enumerate(lines):
                ax.text(x + 14, y + 24 + i * 15, line, fontsize=9.8, fontweight="bold", color="#262626",
                        va="center", alpha=a)
            sub = _status_text(stage, lang)
            if stage.estimate is not None:
                sub += f" · {_human_time(stage.estimate.total_seconds)}"
            ax.text(x + 14, y + 24 + 15 * len(lines) + 14, sub, fontsize=8.5, color=style["stroke"], va="center",
                    alpha=a)
            if stage.answers:
                word = "perguntas" if lang == "pt" and len(stage.answers) > 1 else (
                    "pergunta" if lang == "pt" else ("questions" if len(stage.answers) > 1 else "question"))
                ax.text(x + 14, y + 24 + 15 * len(lines) + 30, f"{word} " + ", ".join(str(n) for n in
                        sorted(stage.answers)), fontsize=8, color="#6b6b6b", va="center", alpha=a)

    def arrows(shown: Sequence[Stage]) -> None:
        ids = {s.id for s in shown}
        for stage in shown:
            sx, sy = pos[stage.id]
            for dep in stage.depends_on:
                if dep not in ids:
                    continue
                dx, dy = pos[dep]
                ax.add_patch(FancyArrowPatch((dx + _BOX_W / 2, dy + _BOX_H), (sx + _BOX_W / 2, sy),
                                             arrowstyle="-|>", mutation_scale=12, linewidth=1.4,
                                             color="#9aa0a6", shrinkA=0, shrinkB=2))

    frames: list[tuple[list[Stage], float]] = [([], 0.0)]
    shown: list[Stage] = []
    for stage in plan.stages:
        shown = shown + [stage]
        for k in range(1, max(int(reveal), 1) + 1):
            frames.append((shown, k / max(int(reveal), 1)))
    frames += [(shown, 1.0)] * max(int(hold_s * fps), 1)

    def draw(frame: tuple[list[Stage], float]) -> None:
        current, alpha = frame
        ax.clear()
        ax.set_xlim(0, width)
        ax.set_ylim(height, 0)
        ax.axis("off")
        header()
        arrows(current)
        if current:
            boxes(current, alpha)

    writer = FFMpegWriter(fps=fps) if str(path).lower().endswith(".mp4") else PillowWriter(fps=fps)
    with writer.saving(fig, str(path), dpi=dpi):
        for frame in frames:
            draw(frame)
            writer.grab_frame()
    plt.close(fig)
    return str(path)


__all__ = ["diagram", "mermaid", "dot", "svg", "html", "markdown", "animate", "layers", "STATUS_STYLE"]
