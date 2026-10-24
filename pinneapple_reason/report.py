"""Reports of a chain: ``chain.json``, ``report.md``, ``report.html`` (self-contained), ``trust.json`` and figures.

Figures (``figures/``):

* ``timeline.png``: version 0 -> version N as cards; each arrow names the failure found, the number that proves it
  and the countermeasure taken;
* ``margins.png``: every check (rows) on every version (columns), PASS / FAIL with demand / capacity;
* ``demand_capacity.png``: one panel per failure mode, demand and capacity across the versions;
* domain figures when the chain has the mode: minimum roof pitch by covering (``roof_min_slope.png``), natural
  frequency against the walking bands (``resonance.png``), head at the valve after closure (``water_hammer.png``).
"""
from __future__ import annotations

import base64
import html
import json
import math
import textwrap
from pathlib import Path
from typing import Any

from .chain import Chain, _fmt, cost_text, element_label, param_text, text_of
from .library import LIBRARY

__all__ = ["write_report", "timeline_figure", "margins_figure", "demand_capacity_figure"]

# reference palette of the data-viz method: text ink, surface, grid, two categorical slots, status good / critical
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#8a8984", "#fcfcfb", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
GOOD, CRITICAL = "#0ca30c", "#d03b3b"
GOOD_BG, CRITICAL_BG = "#e3f4e3", "#fbe5e5"

_T = {
    "pt": {"status": {"resolved": "resolvido", "unresolved": "sem solução na biblioteca", "max_steps": "parou no máximo "
                      "de passos"}, "version": "Versão", "all_pass": "todos os requisitos passam", "fails": "falha",
           "req": "Requisitos", "timeline": "Linha do tempo: versão 0 → versão N, e por quê", "options": "Opções testadas",
           "chosen": "Escolhida", "why": "Por quê", "final": "Verificações da versão final", "premises": "Premissas e "
           "parâmetros da versão final", "trust": "Confiança", "card": "Cartão de confiança (6 etapas)", "sources":
           "Fontes", "assumptions": "Hipóteses", "figures": "Figuras", "check": "Verificação", "demand": "Demanda",
           "capacity": "Capacidade", "margin": "Margem", "method": "Método", "value": "Valor", "source": "Fonte",
           "cost": "Custo", "fixes": "Resolve", "yes": "sim", "no": "não", "infeasible": "inviável", "decision":
           "Decisão", "proof": "Prova", "mechanism": "Mecanismo", "quantities": "Números medidos", "param":
           "Parâmetro", "stage": "Etapa", "evidence": "Evidência", "next": "Próximo passo", "changes": "Mudanças"},
    "en": {"status": {"resolved": "resolved", "unresolved": "no solution in the library", "max_steps": "stopped at the "
                      "step limit"}, "version": "Version", "all_pass": "every requirement passes", "fails": "fails",
           "req": "Requirements", "timeline": "Timeline: version 0 -> version N, and why", "options": "Options tried",
           "chosen": "Chosen", "why": "Why", "final": "Checks of the final version", "premises": "Premises and "
           "parameters of the final version", "trust": "Trust", "card": "Trust card (six stages)", "sources":
           "Sources", "assumptions": "Assumptions", "figures": "Figures", "check": "Check", "demand": "Demand",
           "capacity": "Capacity", "margin": "Margin", "method": "Method", "value": "Value", "source": "Source",
           "cost": "Cost", "fixes": "Fixes", "yes": "yes", "no": "no", "infeasible": "infeasible", "decision":
           "Decision", "proof": "Proof", "mechanism": "Mechanism", "quantities": "Measured numbers", "param":
           "Parameter", "stage": "Stage", "evidence": "Evidence", "next": "Next step", "changes": "Changes"},
}


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.titlecolor": INK, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.grid": False})
    return plt


def _wrap(s: str, width: int) -> str:
    return "\n".join(textwrap.wrap(s, width)) if s else ""


# ------------------------------------------------------------------------------------------------ figures
def timeline_figure(chain: Chain, path: str | Path, lang: str = "pt") -> Path:
    """Version cards left to right; the arrow between two cards says what failed (with its numbers) and what was
    added."""
    plt = _plt()
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    T = _T[lang]
    steps = chain.steps
    n = len(steps)
    card_w, gap, card_h = 3.0, 2.1, 2.9
    W = n * card_w + (n - 1) * gap + 0.4
    fig = plt.figure(figsize=(max(W, 6.0), card_h + 1.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, max(W, 6.0))
    ax.set_ylim(-0.25, card_h + 0.95)
    ax.axis("off")
    ax.text(0.2, card_h + 0.62, text_of(chain.title, lang), fontsize=12, color=INK, weight="bold", va="center")
    seen: list[str] = []
    for i, s in enumerate(steps):
        x0 = 0.2 + i * (card_w + gap)
        ok = s.finding is None
        ax.add_patch(FancyBboxPatch((x0, 0.25), card_w, card_h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=GOOD_BG if ok else CRITICAL_BG, ec=GOOD if ok else CRITICAL, lw=1.4))
        ax.text(x0 + 0.15, card_h + 0.02, f"v{s.version}", fontsize=13, weight="bold", color=INK, va="top")
        new = [e for e in s.concept.elements if e not in seen]
        lines = [("+ " if e in new and i > 0 else "· ") + element_label(e, lang) for e in s.concept.elements]
        seen = list(s.concept.elements)
        prev = steps[i - 1].countermeasure if i > 0 else None
        if prev is not None:
            lines += [f"  {param_text(k, x, lang)}" for k, x in list(prev.changes.items())[:4]]
        ax.text(x0 + 0.15, card_h - 0.42, "\n".join(lines[:9]), fontsize=8, color=INK, va="top", family="monospace")
        states = "  ".join(f"{v.requirement} {'✓' if v.passed else '✗'}" for v in s.verifications)
        ax.text(x0 + 0.15, 1.15, _wrap(states, 40), fontsize=8, color=INK2, va="bottom")
        if ok:
            status = "✓ " + T["all_pass"]
        else:
            v = s.finding.verification
            status = (f"✗ {s.finding.requirement}: {text_of(s.finding.title, lang)}\n"
                      f"{_fmt(v.demand)} > {_fmt(v.capacity)} {v.unit}")
        ax.text(x0 + 0.15, 0.4, _wrap(status.split("\n")[0], 34) + ("\n" + status.split("\n")[1]
                                                                   if "\n" in status else ""),
                fontsize=8, color=INK, va="bottom", weight="bold")
        if i < n - 1 and s.countermeasure is not None:
            xa, xb = x0 + card_w + 0.05, x0 + card_w + gap - 0.05
            ax.add_patch(FancyArrowPatch((xa, card_h * 0.55), (xb, card_h * 0.55), arrowstyle="-|>",
                                         mutation_scale=14, color=INK2, lw=1.4))
            label = _wrap(text_of(s.countermeasure.title, lang), 24)
            ax.text((xa + xb) / 2, card_h * 0.55 + 0.12, label, fontsize=7.5, color=INK, ha="center", va="bottom")
            c = s.countermeasure.cost
            ax.text((xa + xb) / 2, card_h * 0.55 - 0.12, _wrap(f"{T['cost']}: {cost_text(c, lang)}", 26),
                    fontsize=7, color=INK2, ha="center", va="top")
    ax.text(0.2, 0.06, ("Custo: a quantidade física que a mudança acrescenta. Entre opções do mesmo passo, o motor "
                        "compara índices relativos = quantidade × custo unitário (premissa, ver relatório)."
                        if lang == "pt" else
                        "Cost: the physical quantity the change adds. Among options of the same step the engine "
                        "compares relative indices = quantity x unit cost (a premise, see the report)."),
            fontsize=7, color=INK2, va="bottom")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def margins_figure(chain: Chain, path: str | Path, lang: str = "pt") -> Path:
    plt = _plt()
    rows = [(v.requirement, v.mode) for v in chain.steps[0].verifications]
    n, m = len(rows), len(chain.steps)
    fig, ax = plt.subplots(figsize=(1.6 + 1.7 * m, 0.6 + 0.62 * n))
    for j, s in enumerate(chain.steps):
        for i, (_r, mode) in enumerate(rows):
            v = s.verification(mode)
            ok = v.passed
            ax.add_patch(plt.Rectangle((j + 0.03, n - i - 1 + 0.06), 0.94, 0.88, fc=GOOD_BG if ok else CRITICAL_BG,
                                       ec=GOOD if ok else CRITICAL, lw=1.0))
            ax.text(j + 0.5, n - i - 0.38, ("PASS" if ok else "FAIL"), ha="center", va="center", fontsize=8,
                    weight="bold", color=INK)
            ax.text(j + 0.5, n - i - 0.7, f"{_fmt(v.demand)} / {_fmt(v.capacity)} {v.unit}", ha="center",
                    va="center", fontsize=6.8, color=INK2)
    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(m)], [f"v{s.version}" for s in chain.steps])
    ax.set_yticks([n - i - 0.5 for i in range(n)], [f"{r}: {mode}" for r, mode in rows])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title("demand / capacity" if lang == "en" else "demanda / capacidade", fontsize=9, color=INK2, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def demand_capacity_figure(chain: Chain, path: str | Path, lang: str = "pt") -> Path:
    plt = _plt()
    T = _T[lang]
    rows = [(v.requirement, v.mode) for v in chain.steps[0].verifications]
    k = len(rows)
    cols = min(k, 3)
    nr = math.ceil(k / cols)
    fig, axs = plt.subplots(nr, cols, figsize=(3.4 * cols, 2.6 * nr), squeeze=False)
    xs = [s.version for s in chain.steps]
    for idx, (r, mode) in enumerate(rows):
        ax = axs[idx // cols][idx % cols]
        vs = [s.verification(mode) for s in chain.steps]
        dem = [v.demand if math.isfinite(v.demand) else float("nan") for v in vs]
        cap = [v.capacity if math.isfinite(v.capacity) else float("nan") for v in vs]
        ax.plot(xs, dem, "-o", color=S1, lw=2, ms=6, label=T["demand"])
        ax.plot(xs, cap, "--s", color=S2, lw=2, ms=6, label=T["capacity"])
        for x, v in zip(xs, vs, strict=True):
            ax.annotate("✓" if v.passed else "✗", (x, v.demand if math.isfinite(v.demand) else 0),
                        textcoords="offset points", xytext=(0, 7), ha="center", fontsize=9,
                        color=GOOD if v.passed else CRITICAL)
        ax.set_title(f"{r}: {text_of(LIBRARY.mode(mode).title, lang)}", fontsize=8.5, loc="left")
        ax.set_ylabel(vs[0].unit)
        ax.set_xticks(xs, [f"v{x}" for x in xs])
        ax.grid(axis="y", color=GRID, lw=0.6)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for idx in range(k, nr * cols):
        axs[idx // cols][idx % cols].axis("off")
    axs[0][0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return Path(path)


def _roof_min_slope_figure(chain: Chain, path: Path, lang: str) -> Path | None:
    from .cases import ROOFING
    from .modes.water import minimum_slope
    p = chain.final.concept.params
    if "rain_intensity" not in p:
        return None
    margin = float(p.get("design_margin", 0.0))
    names, a0, a1 = [], [], []
    for g in ROOFING.values():
        pp = {**p, **{k: v for k, v in g.items() if k != "label"}}
        names.append(text_of(g["label"], lang))
        a0.append(minimum_slope(pp, 0.0))
        a1.append(minimum_slope(pp, margin))
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    y = range(len(names))
    ax.barh([i + 0.18 for i in y], a0, height=0.34, color=S1, label=("limite físico" if lang == "pt" else
                                                                     "physical limit"))
    ax.barh([i - 0.18 for i in y], a1, height=0.34, color=S3, label=(f"com margem de {margin:.0%}" if lang == "pt"
                                                                     else f"with {margin:.0%} margin"))
    for i in y:
        ax.text(a1[i] + 0.4, i - 0.18, f"{a1[i]:.1f}°", va="center", fontsize=8, color=INK)
        ax.text(a0[i] + 0.4, i + 0.18, f"{a0[i]:.1f}°", va="center", fontsize=8, color=INK2)
    ax.set_yticks(list(y), names)
    ax.set_xlabel("inclinação mínima (graus)" if lang == "pt" else "minimum pitch (degrees)")
    ax.set_title(("Inclinação mínima por material\n(capilaridade + vento + lâmina, chuva de projeto)" if lang == "pt"
                  else "Minimum pitch by covering\n(capillarity + wind + film, design rain)"), fontsize=9, loc="left")
    ax.set_xlim(0, max(a1) * 1.35)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _resonance_figure(chain: Chain, path: Path, lang: str) -> Path | None:
    vs = [(s.version, s.verification("pedestrian_resonance")) for s in chain.steps]
    vs = [(k, v) for k, v in vs if v is not None]
    if not vs:
        return None
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    for k, (lo, hi) in enumerate(vs[0][1].series["bands"], start=1):
        ax.axvspan(lo, hi, color=GRID, alpha=1.0 if k == 1 else 0.55, lw=0)
        ax.text((lo + hi) / 2, len(vs) - 0.35, (f"passo, {k}º harm." if lang == "pt" else f"walking, harm. {k}"),
                ha="center", fontsize=7.5, color=INK2)
    for i, (k, v) in enumerate(vs):
        f1 = v.series["f1"]
        ax.plot([f1], [i], "o", ms=8, color=CRITICAL if not v.passed else GOOD)
        ax.text(f1 + 0.08, i, f"v{k}: f1 = {f1:.2f} Hz, a = {_fmt(v.demand)} m/s² {'✓' if v.passed else '✗'}",
                va="center", fontsize=8, color=INK)
    ax.set_yticks([])
    ax.set_xlim(0, 8)
    ax.set_ylim(-0.6, len(vs) - 0.1)
    ax.set_xlabel("frequência (Hz)" if lang == "pt" else "frequency (Hz)")
    ax.set_title(("1ª frequência natural vertical contra as faixas do passo humano" if lang == "pt" else
                  "First vertical natural frequency against the walking bands"), fontsize=9, loc="left")
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _water_hammer_figure(chain: Chain, path: Path, lang: str) -> Path | None:
    vs, seen = [], set()
    for s in chain.steps:
        v = s.verification("water_hammer")
        if v is not None and v.series and v.series.get("t"):
            key = (round(v.demand, 6), len(v.series["t"]))
            if key not in seen:
                seen.add(key)
                vs.append((s.version, v))
    if not vs:
        return None
    plt = _plt()
    from .props import WATER, G
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    colors = [CRITICAL, S1, S3, S2]
    for i, (k, v) in enumerate(vs):
        tc = chain.steps[k].concept.params.get("valve_close_time")
        ax.plot(v.series["t"], v.series["h_valve"], color=colors[i % 4], lw=1.6,
                label=f"v{k}: t_c = {_fmt(tc)} s, p_max = {_fmt(v.demand)} MPa")
    rating = vs[0][1].capacity * 1e6 / (WATER["rho"] * G)
    ax.axhline(rating, color=INK2, lw=1.2, ls="--")
    ax.text(ax.get_xlim()[1], rating, (" classe do tubo" if lang == "pt" else " pipe class"), va="bottom",
            ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("carga na válvula (m)" if lang == "pt" else "head at the valve (m)")
    ax.set_title(("Golpe de aríete: carga na válvula após o início do fechamento (MOC)" if lang == "pt" else
                  "Water hammer: head at the valve after closure starts (MOC)"), fontsize=9, loc="left")
    ax.legend(frameon=False, fontsize=7.5)
    ax.grid(axis="y", color=GRID, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def _figures(chain: Chain, out: Path, lang: str) -> dict[str, Path]:
    fd = out / "figures"
    fd.mkdir(parents=True, exist_ok=True)
    figs = {"timeline": timeline_figure(chain, fd / "timeline.png", lang),
            "margins": margins_figure(chain, fd / "margins.png", lang),
            "demand_capacity": demand_capacity_figure(chain, fd / "demand_capacity.png", lang)}
    modes = {v.mode for v in chain.steps[0].verifications}
    extra = {"roof_min_slope": ("lap_water_ingress", _roof_min_slope_figure),
             "resonance": ("pedestrian_resonance", _resonance_figure),
             "water_hammer": ("water_hammer", _water_hammer_figure)}
    for name, (mode, fn) in extra.items():
        if mode in modes:
            p = fn(chain, fd / f"{name}.png", lang)
            if p is not None:
                figs[name] = p
    return figs


# ------------------------------------------------------------------------------------------------ text
def _md(chain: Chain, figs: dict[str, Path], trust: dict[str, Any], card: dict[str, Any], lang: str) -> str:
    T = _T[lang]
    L = [f"# {text_of(chain.title, lang)}", "",
         f"**Status:** {T['status'][chain.status]} · **{T['decision']}:** {trust['decisao']} · "
         f"{len(chain.steps) - 1} {'passos' if lang == 'pt' else 'steps'} · "
         f"{chain.meta.get('evaluations', 0)} {'avaliações físicas' if lang == 'pt' else 'physical evaluations'}", "",
         f"## {T['req']}", ""]
    L += [f"- **{r.id}** — {text_of(r.text, lang)} (`{'`, `'.join(r.modes)}`)" for r in chain.requirements]
    L += ["", f"## {T['timeline']}", "", f"![timeline](figures/{figs['timeline'].name})", ""]
    for s in chain.steps:
        L.append(f"### {T['version']} {s.version} — {', '.join(element_label(e, lang) for e in s.concept.elements)}")
        L.append("")
        if s.finding is None:
            L += [f"✓ {T['all_pass']}.", ""]
            continue
        v = s.finding.verification
        L += [f"**✗ {s.finding.requirement} {T['fails']}: {text_of(s.finding.title, lang)}.** "
              f"{text_of(v.mechanism, lang)}", "",
              f"*{T['proof']}:* {text_of(v.demand_label, lang)} = {_fmt(v.demand)} {v.unit} > "
              f"{text_of(v.capacity_label, lang)} = {_fmt(v.capacity)} {v.unit} — {v.method}.", ""]
        if s.options:
            L += [f"{T['options']}:", "", f"| | {T['fixes']} | {T['margin']} | {T['cost']} | {T['changes']} |",
                  "|---|---|---|---|---|"]
            for i, o in enumerate(s.options):
                mark = "**→**" if i == s.chosen else ""
                if not o.feasible:
                    L.append(f"| {mark} {o.countermeasure} | {T['infeasible']}: {text_of(o.reason, lang)} | | | |")
                    continue
                c = o.cost
                ch = ", ".join(param_text(k, x, lang) for k, x in o.changes.items())
                L.append(f"| {mark} {o.countermeasure} | {T['yes'] if o.fixes_target else T['no']} | "
                         f"{_fmt(o.target_margin_before)} → {_fmt(o.target_margin_after)} | "
                         f"{cost_text(c, lang)} ({'índice' if lang == 'pt' else 'index'} "
                         f"{_fmt(c.get('index', 0))}) | {ch} |")
            L.append("")
        cm = s.countermeasure
        if cm is not None:
            L += [f"**{T['chosen']}: {text_of(cm.title, lang)}.** {text_of(cm.effect, lang)}. "
                  f"*{T['why']}:* {text_of(s.why, lang)}.", ""]
        elif s.why:
            L += [f"**{text_of(s.why, lang)}**", ""]
    L += [f"## {T['final']}", "", f"| {T['check']} | {T['demand']} | {T['capacity']} | {T['margin']} | {T['method']} |",
          "|---|---|---|---|---|"]
    for v in chain.final.verifications:
        L.append(f"| {'✓' if v.passed else '✗'} {v.requirement}/{v.mode} | {_fmt(v.demand)} {v.unit} | "
                 f"{_fmt(v.capacity)} {v.unit} | {_fmt(v.margin)} | {v.method} |")
    L += ["", f"### {T['quantities']}", ""]
    for v in chain.final.verifications:
        L.append(f"- **{v.mode}**: " + "; ".join(f"{x[2]} = {_fmt(x[0])} {x[1]}" for x in v.quantities.values()))
    L += ["", f"## {T['figures']}", ""]
    L += [f"![{k}](figures/{p.name})" for k, p in figs.items() if k != "timeline"]
    c = chain.final.concept
    L += ["", f"## {T['premises']}", "", f"| {T['param']} | {T['value']} | {T['source']} |", "|---|---|---|"]
    for k, x in sorted(c.params.items()):
        L.append(f"| `{k}` | {_fmt(x) if isinstance(x, float) else x} | {c.sources.get(k, '')} |")
    L += ["", f"## {T['trust']}", "", f"**{T['decision']}:** {trust['decisao']} · score "
          f"{_fmt(trust['score']) if trust['score'] is not None else '—'} · {len(trust['checagens'])} checks", ""]
    if trust["motivos"]:
        L += [f"- {m}" for m in trust["motivos"]] + [""]
    L += [f"| {T['check']} | status | {T['value']} | limit | {T['source']} |", "|---|---|---|---|---|"]
    for ck in trust["checagens"]:
        L.append(f"| {ck['name']} | {ck['status']} | {'—' if ck['value'] is None else _fmt(ck['value'])} | "
                 f"{'—' if ck['limit'] is None else _fmt(ck['limit'])} | {(ck['source'] or '')[:90]} |")
    if card:
        L += ["", f"### {T['card']}", "", f"| {T['stage']} | status | {T['evidence']} | {T['next']} |",
              "|---|---|---|---|"]
        for st in card.values():
            L.append(f"| {st['title']} | {st['status']} | {'<br>'.join(st['evidence'][:3])} | {st.get('next', '')} |")
    srcs = sorted({x for v in chain.final.verifications for x in v.sources})
    asm = sorted({text_of(a, lang) for s in chain.steps for v in s.verifications for a in v.assumptions})
    L += ["", f"## {T['assumptions']}", ""] + [f"- {a}" for a in asm]
    L += ["", f"## {T['sources']}", ""] + [f"- {x}" for x in srcs]
    return "\n".join(L) + "\n"


_CSS = """
:root{--surface:#fcfcfb;--page:#f9f9f7;--ink:#0b0b0b;--ink2:#52514e;--rule:#e4e3df;--good:#0ca30c;--bad:#d03b3b}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;
--ink2:#c3c2b7;--rule:#383835}}
:root[data-theme="dark"]{--surface:#1a1a19;--page:#0d0d0d;--ink:#f0efec;--ink2:#c3c2b7;--rule:#383835}
body{background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif;
margin:0;padding:0 16px}
main{max-width:1100px;margin:24px auto;background:var(--surface);padding:24px;border-radius:10px}
h1{font-size:1.6em;margin-top:0}h2{border-bottom:1px solid var(--rule);padding-bottom:4px;margin-top:32px}
table{border-collapse:collapse;width:100%;font-size:.88em;margin:8px 0;display:block;overflow-x:auto}
td,th{border-bottom:1px solid var(--rule);padding:4px 8px;text-align:left;vertical-align:top}
img{max-width:100%;height:auto;background:#fcfcfb;border-radius:6px;margin:8px 0}
.ok{color:var(--good);font-weight:600}.bad{color:var(--bad);font-weight:600}
code{font-size:.9em}.muted{color:var(--ink2)}
"""


def _md_to_html(md: str, figs: dict[str, Path], title: str) -> str:
    """Small Markdown subset -> HTML (headings, tables, lists, images inlined, bold / italic / code)."""
    import re
    imgs = {p.name: base64.b64encode(p.read_bytes()).decode() for p in figs.values()}

    def inline(s):
        s = html.escape(s, quote=False).replace("&lt;br&gt;", "<br>")
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", s)
        s = re.sub(r"`(.+?)`", r"<code>\1</code>", s)
        return s.replace("✓", '<span class="ok">✓</span>').replace("✗", '<span class="bad">✗</span>')

    out, i, lines = [], 0, md.splitlines()
    while i < len(lines):
        ln = lines[i]
        m = re.match(r"!\[(.*?)\]\(figures/(.+?)\)", ln)
        if m:
            out.append(f'<img alt="{html.escape(m.group(1))}" src="data:image/png;base64,{imgs.get(m.group(2), "")}">')
        elif ln.startswith("#"):
            k = len(ln) - len(ln.lstrip("#"))
            out.append(f"<h{k}>{inline(ln[k:].strip())}</h{k}>")
        elif ln.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                if not re.match(r"^\|[-|\s]+\|$", lines[i]):
                    rows.append([c.strip() for c in lines[i].strip("|").split("|")])
                i += 1
            head, body = rows[0], rows[1:]
            out.append("<table><thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head) + "</tr></thead><tbody>"
                       + "".join("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
                       + "</tbody></table>")
            continue
        elif ln.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif ln.strip():
            out.append(f"<p>{inline(ln)}</p>")
        i += 1
    return (f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' "
            f"content='width=device-width,initial-scale=1'><title>{html.escape(title)}</title><style>{_CSS}</style>"
            f"</head><body><main>{''.join(out)}</main></body></html>")


def write_report(chain: Chain, out_dir: str | Path, lang: str = "pt", card: bool = True) -> dict[str, Path]:
    """Write the chain, its figures, the trust report and the reports; returns the paths."""
    from .trust import trust_card, trust_report
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = _figures(chain, out, lang)
    rep = trust_report(chain)
    trust = rep.to_dict()
    stages = {}
    if card:
        inputs = out / "inputs"
        inputs.mkdir(exist_ok=True)
        for s in chain.steps:                       # the inputs of every version, for the trust card's first stage
            (inputs / f"v{s.version}.json").write_text(json.dumps(s.concept.to_dict(), indent=1, default=str))
        try:
            stages = trust_card(chain, str(out), rep)
        except ImportError:                         # pinneapple_lab not importable: the report keeps the checks
            stages = {}
    paths = {"chain": chain.save(out / "chain.json")}
    (out / "trust.json").write_text(json.dumps({"report": trust, "card": stages}, indent=1, default=str))
    paths["trust"] = out / "trust.json"
    md = _md(chain, figs, trust, stages, lang)
    (out / "report.md").write_text(md)
    (out / "report.html").write_text(_md_to_html(md, figs, text_of(chain.title, lang)))
    paths.update({"md": out / "report.md", "html": out / "report.html"}, **{f"fig_{k}": p for k, p in figs.items()})
    return paths
