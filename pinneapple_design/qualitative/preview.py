"""Qualitative preview: how each geometry, or each change to a part, should affect the result, before any
simulation; then the quantitative check.

>>> from pinneapple_design.qualitative import preview
>>> p = preview({"baseline": body, "longer tail": body2, "blunt rear": body3}, objective="minimizar arrasto",
...             conditions={"U": 30})
>>> print(p.text())                # direction, strength, mechanism and confidence for each variant
>>> p.figure("preview.png")        # surface maps (where the physics happens) and the expected changes
>>> check = p.quantify(lambda name, geom: run_cfd(geom))   # numbers only for the variants worth simulating

The preview is meant to be read first: it says *why* (the mechanisms that move the objective and by how much
relative to each other), flags side effects on the other quantities, and lowers its confidence where the cheap model
is known to be fragile (near a separation threshold, in the transition range, short beams, low fin efficiency...).
``quantify`` then compares those expectations with an accurate computation (CFD, FEM, ``pp.solve``, experiments) and
reports where the qualitative reading was right or wrong.
"""
from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .descriptors import Assembly
from .models import (
    MODELS,
    Cantilever,
    ConvectiveCooling,
    Evaluation,
    ExternalFlow,
    QualitativeModel,
)

__all__ = ["Objective", "QualitativePreview", "QuantitativeCheck", "PartSensitivity", "part_sensitivity", "preview",
           "parse_objective"]

# words of an objective (pt / en) -> (model, quantity)
_VOCAB = [
    (r"rigidez\s*(por|/)\s*(massa|peso)|stiffness\s*(per|to|/)\s*(mass|weight)|specific stiffness", "cantilever", "stiffness_to_mass"),
    (r"frequ[eê]ncia|frequency|vibra", "cantilever", "first_frequency"),
    (r"defle|flecha|desloc|deflection|displacement", "cantilever", "tip_deflection"),
    (r"tens[aã]o|stress|seguran|safety", "cantilever", "max_stress"),
    (r"rigidez|stiff", "cantilever", "stiffness"),
    (r"resist[eê]ncia t[eé]rmica|thermal resistance", "convective_cooling", "thermal_resistance"),
    (r"resfri|refriger|dissip|calor|cool|heat|t[eé]rmic|thermal", "convective_cooling", "heat_rate"),
    (r"coeficiente de arrasto|drag coefficient|\bcd\b", "external_flow", "drag_coefficient"),
    (r"arrasto|drag|aerodin|resist[eê]ncia ao avan", "external_flow", "drag"),
    (r"sustenta|lift|downforce|pressão para baixo", "external_flow", "lift"),
    (r"esteira|wake", "external_flow", "wake_area"),
    (r"massa|peso|mass|weight|leve|light", None, "mass"),
]
_MIN = r"minimi|reduz|diminu|menor|baixar|minimi[sz]e|reduce|lower|less|decrease|smaller"
_MAX = r"maximi|aument|maior|elevar|melhor|maximi[sz]e|increase|raise|more|higher|larger|improve"


@dataclass
class Objective:
    quantity: str
    sense: str                       # "min" or "max"
    model: str


def parse_objective(objective: str | tuple | Objective, model: QualitativeModel | None = None) -> Objective:
    """``"minimizar arrasto"``, ``"maximize heat rejection"``, ``("tip_deflection", "min")`` or an ``Objective``.
    Downforce means lift with sense "min"."""
    if isinstance(objective, Objective):
        return objective
    if isinstance(objective, tuple):
        q, sense = objective
        mname = model.name if model is not None else next((m for m, cls in MODELS.items() if q in cls.quantities), None)
        if mname is None:
            raise ValueError(f"no model provides {q!r}; pass model=...")
        return Objective(q, sense, mname)
    text = objective.lower()
    for pat, mname, q in _VOCAB:
        if re.search(pat, text):
            if mname is None:
                mname = model.name if model is not None else "cantilever"
            cls = MODELS.get(mname)
            better = cls.quantities[q].better if cls is not None and q in cls.quantities else "min"
            sense = "min" if re.search(_MIN, text) else "max" if re.search(_MAX, text) else better
            if re.search(r"downforce|pressão para baixo", text):
                sense = "min"
            return Objective(q, sense, mname)
    raise ValueError(f"could not read an objective from {objective!r}: name the quantity (drag, lift, heat, "
                     "stiffness, deflection, stress, mass, frequency) or pass (quantity, 'min'|'max')")


def _strength(pct: float, lang: str) -> str:
    a = abs(pct)
    table = [(3, "desprezível", "negligible"), (10, "leve", "slight"), (30, "moderado", "moderate"),
             (1e300, "forte", "strong")]
    for lim, pt, en in table:
        if a < lim:
            return pt if lang == "pt" else en
    return ""


def _arrow(pct: float) -> str:
    return "≈" if abs(pct) < 3 else ("↑" if pct > 0 else "↓")


@dataclass
class VariantPreview:
    name: str
    evaluation: Evaluation
    change: dict[str, float]                 # quantity -> % change against the baseline
    mechanisms: list[tuple[str, float]]      # (mechanism, % of the baseline objective it moved), largest first
    parts: list[tuple[str, float]]           # (part, % of the baseline objective), largest first
    confidence: str                          # "alta" / "média" / "baixa" (or high / medium / low)
    reasons: list[tuple[str, str]]
    good: bool | None                        # does it move the objective the right way (None for the baseline)


@dataclass
class QualitativePreview:
    model: QualitativeModel
    objective: Objective
    baseline: str
    variants: dict[str, VariantPreview]
    geometries: dict[str, Any]
    lang: str = "pt"

    # ------------------------------------------------------------------ reading
    def ranking(self) -> list[str]:
        q, s = self.objective.quantity, self.objective.sense
        return sorted(self.variants, key=lambda k: self.variants[k].evaluation.quantities[q] * (1 if s == "min" else -1))

    def to_simulate(self, k: int = 3) -> list[str]:
        """Variants worth the quantitative step first: the best expected ones, then the least certain among the
        rest (where the qualitative reading could be wrong), always with the baseline."""
        rank = [v for v in self.ranking() if v != self.baseline]
        order = {"alta": 0, "high": 0, "média": 1, "medium": 1, "baixa": 2, "low": 2}
        best = [v for v in rank if self.variants[v].good][: max(1, k - 1)]
        rest = sorted((v for v in rank if v not in best), key=lambda v: -order[self.variants[v].confidence])
        return [self.baseline] + (best + rest)[: k - 1]

    def _q(self, q):
        Q = self.model.quantities[q]
        return Q.pt if self.lang == "pt" else Q.en

    def text(self, lang: str | None = None) -> str:
        lang = lang or self.lang
        pt = lang == "pt"
        o = self.objective
        L = [f"# {'Prévia qualitativa' if pt else 'Qualitative preview'}: "
             f"{('minimizar ' if o.sense == 'min' else 'maximizar ') if pt else (o.sense + 'imize ')}{self._q(o.quantity)}",
             "", f"{'Modelo' if pt else 'Model'}: {self.model.title[0 if pt else 1]}. "
             f"{'Referência' if pt else 'Baseline'}: **{self.baseline}**.", ""]
        base = self.variants[self.baseline]
        for t in base.evaluation.notes:
            L.append(f"- {t[0 if pt else 1]}")
        L.append("")
        for name in self.ranking():
            v = self.variants[name]
            if name == self.baseline:
                continue
            pct = v.change[o.quantity]
            verdict = ("melhora" if v.good else "piora") if pt else ("improves" if v.good else "worsens")
            if abs(pct) < 3:
                verdict = "não muda o objetivo" if pt else "leaves the objective unchanged"
            L.append(f"## {name}: {self._q(o.quantity)} {_arrow(pct)} {pct:+.0f}% ({_strength(pct, lang)}, {verdict}); "
                     f"{'confiança' if pt else 'confidence'} {v.confidence}")
            if v.mechanisms:
                L.append(f"- {'Por quê' if pt else 'Why'}:")
                for m, share in [t for t in v.mechanisms if abs(t[1]) >= 0.5][:3]:
                    expl = self.model.mechanisms.get(m, (m, m))[0 if pt else 1]
                    L.append(f"  - {expl}: {share:+.0f}% {'do valor de referência' if pt else 'of the baseline value'}")
            if len(v.parts) > 1:
                top = ", ".join(f"{p} {s:+.0f}%" for p, s in v.parts[:3] if abs(s) >= 1)
                if top:
                    L.append(f"- {'Peças que mais mudaram o resultado' if pt else 'Parts that moved it most'}: {top}")
            side = [(q, c) for q, c in v.change.items() if q != o.quantity and np.isfinite(c) and abs(c) >= 3]
            if side:
                L.append(f"- {'Efeitos colaterais' if pt else 'Side effects'}: "
                         + ", ".join(f"{self._q(q)} {_arrow(c)} {c:+.0f}%" for q, c in side))
            for r in v.reasons:
                L.append(f"- ⚠ {r[0 if pt else 1]}")
            L.append("")
        L.append(f"**{'Simular primeiro' if pt else 'Simulate first'}:** {', '.join(self.to_simulate())}")
        L.append("")
        L.append(("_Estimativas de ordem de grandeza para ordenar e explicar; os números vêm da etapa quantitativa "
                  "(`quantify`)._") if pt else
                 "_Order-of-magnitude estimates to rank and explain; the numbers come from the quantitative step "
                 "(`quantify`)._")
        return "\n".join(L)

    def to_dict(self) -> dict[str, Any]:
        return {"objective": self.objective.__dict__, "baseline": self.baseline, "ranking": self.ranking(),
                "variants": {k: {"quantities": v.evaluation.quantities, "change_pct": v.change,
                                 "mechanisms": v.mechanisms, "parts": v.parts, "confidence": v.confidence,
                                 "reasons": [r[0 if self.lang == "pt" else 1] for r in v.reasons]}
                             for k, v in self.variants.items()}}

    # ------------------------------------------------------------------ picture
    def figure(self, path: str | None = None, lang: str | None = None, max_faces: int = 6000,
               view: tuple[float, float] = (22, -125), same_scale: bool = True):
        """Surface map of each variant (where the physics acts, same colour scale for all) and the expected change
        of the objective split into mechanisms."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        lang = lang or self.lang
        pt = lang == "pt"
        names = [self.baseline] + [n for n in self.ranking() if n != self.baseline]
        k = len(names)
        fig = plt.figure(figsize=(3.4 * k, 7.2))
        vals = np.concatenate([self.variants[n].evaluation.face_field for n in names])
        lo, hi = np.nanpercentile(vals, [2, 98])
        if lo < 0 < hi:
            lo, hi = -max(-lo, hi), max(-lo, hi)
        cmap = plt.get_cmap(self.model.cmap)
        o = self.objective
        allV = np.vstack([self.variants[n].evaluation.mesh[0] for n in names])
        half = np.ptp(allV, axis=0) / 2
        half = np.maximum(half, 0.2 * half.max())          # keep flat or slender parts visible
        flow = self.model.conditions.get("direction") if self.model.conditions.get("U", 0) else None
        for i, name in enumerate(names):
            v = self.variants[name]
            V, F = v.evaluation.mesh
            fld = v.evaluation.face_field
            idx = np.arange(len(F)) if len(F) <= max_faces else np.linspace(0, len(F) - 1, max_faces).astype(int)
            ax = fig.add_subplot(2, k, i + 1, projection="3d")
            pc = Poly3DCollection(V[F[idx]], facecolors=cmap(np.clip((fld[idx] - lo) / (hi - lo + 1e-300), 0, 1)),
                                  edgecolor="none")
            ax.add_collection3d(pc)
            c = 0.5 * (V.min(0) + V.max(0))
            hv = half if same_scale else np.maximum(np.ptp(V, axis=0) / 2, 0.2 * np.ptp(V, axis=0).max() / 2)
            r = hv.max()
            ax.set(xlim=(c[0] - hv[0], c[0] + hv[0]), ylim=(c[1] - hv[1], c[1] + hv[1]), zlim=(c[2] - hv[2], c[2] + hv[2]))
            ax.set_box_aspect(tuple(hv / hv.max()))
            ax.view_init(*view)
            if flow is not None:
                dvec = np.asarray(flow, float) / np.linalg.norm(flow)
                p0 = c - dvec * 0.95 * r + np.array([0, 0, 0.6 * r])
                ax.quiver(*p0, *(dvec * 0.5 * r), color="black", arrow_length_ratio=0.3, lw=1.2)
            ax.set_axis_off()
            pct = v.change[o.quantity]
            sub = (("referência" if pt else "baseline") if name == self.baseline else
                   f"{_arrow(pct)} {pct:+.0f}% · {'conf.' if pt else 'conf.'} {v.confidence}")
            ax.set_title(f"{name}\n{sub}", fontsize=9,
                         color="black" if name == self.baseline or abs(pct) < 3 else ("green" if v.good else "firebrick"))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(lo, hi))
        cb = fig.colorbar(sm, ax=fig.axes, shrink=0.35, location="right", pad=0.02)
        cb.set_label(self.variants[self.baseline].evaluation.field_label[0 if pt else 1], fontsize=8)
        axb = fig.add_subplot(2, 1, 2)
        others = names[1:]
        mechs = [m for m in self.model.mechanisms if any(dict(self.variants[n].mechanisms).get(m) for n in others)]
        base_obj = self.variants[self.baseline].evaluation.quantities[o.quantity]
        bottom_pos, bottom_neg = np.zeros(len(others)), np.zeros(len(others))
        colors = plt.get_cmap("tab10")
        for j, m in enumerate(mechs):
            h = np.array([dict(self.variants[n].mechanisms).get(m, 0.0) for n in others])
            bot = np.where(h >= 0, bottom_pos, bottom_neg)
            axb.bar(range(len(others)), h, bottom=bot, color=colors(j), label=self.model.mechanisms[m][0 if pt else 1][:60])
            bottom_pos += np.where(h >= 0, h, 0)
            bottom_neg += np.where(h < 0, h, 0)
        tot = [self.variants[n].change[o.quantity] for n in others]
        axb.scatter(range(len(others)), tot, color="black", zorder=5, marker="D", label="total" if not pt else "total")
        axb.axhline(0, color="grey", lw=0.8)
        axb.set_xticks(range(len(others)))
        axb.set_xticklabels(others, rotation=15, fontsize=8)
        axb.set_ylabel(f"{self._q(o.quantity)}: % {'vs referência' if pt else 'vs baseline'}")
        axb.legend(fontsize=7, loc="best")
        axb.set_title(("Mudança esperada e os mecanismos que a explicam" if pt else
                       "Expected change and the mechanisms behind it") + f" ({'ref.' if pt else 'base'} = {base_obj:.3g} "
                      f"{self.model.quantities[o.quantity].unit})", fontsize=10)
        fig.suptitle(("Prévia qualitativa: " if pt else "Qualitative preview: ")
                     + (("minimizar " if o.sense == "min" else "maximizar ") if pt else o.sense + "imize ")
                     + self._q(o.quantity), fontsize=12, fontweight="bold")
        if path is not None:
            fig.savefig(path, dpi=110, bbox_inches="tight")
            plt.close(fig)
        return fig

    # ------------------------------------------------------------------ numbers
    def quantify(self, compute: Callable[[str, Any], float], variants: list[str] | None = None) -> QuantitativeCheck:
        """Run an accurate computation of the objective (``compute(name, geometry) -> value``: CFD, FEM, a
        ``pp.solve`` problem, an experiment) on ``variants`` (default ``to_simulate()``, always with the baseline) and
        compare with the qualitative expectations: direction of each change, ranking, and where it was wrong."""
        names = variants or self.to_simulate()
        if self.baseline not in names:
            names = [self.baseline] + list(names)
        vals, times = {}, {}
        for n in names:
            t0 = time.perf_counter()
            vals[n] = float(compute(n, self.geometries[n]))
            times[n] = time.perf_counter() - t0
        return QuantitativeCheck(self, vals, times)


@dataclass
class QuantitativeCheck:
    preview: QualitativePreview
    values: dict[str, float]
    seconds: dict[str, float]

    @property
    def table(self) -> list[dict[str, Any]]:
        p = self.preview
        q = p.objective.quantity
        b = self.values[p.baseline]
        rows = []
        for n, val in self.values.items():
            exp = p.variants[n].change[q]
            got = 100 * (val - b) / abs(b) if b else float("nan")
            same = (abs(exp) < 3 and abs(got) < 3) or np.sign(exp) == np.sign(got)
            rows.append({"variant": n, "expected_pct": exp, "computed_pct": got, "value": val,
                         "direction_ok": bool(same) if n != p.baseline else None, "confidence": p.variants[n].confidence})
        return rows

    @property
    def direction_agreement(self) -> float:
        r = [row["direction_ok"] for row in self.table if row["direction_ok"] is not None]
        return float(np.mean(r)) if r else float("nan")

    @property
    def rank_correlation(self) -> float:
        """Spearman correlation between the expected and the computed objective over the computed variants."""
        p = self.preview
        names = list(self.values)
        if len(names) < 3:
            return float("nan")
        a = np.argsort(np.argsort([p.variants[n].evaluation.quantities[p.objective.quantity] for n in names]))
        b = np.argsort(np.argsort([self.values[n] for n in names]))
        return float(np.corrcoef(a, b)[0, 1])

    def text(self, lang: str | None = None) -> str:
        pt = (lang or self.preview.lang) == "pt"
        L = ["| " + (" | ".join(["variante", "esperado", "calculado", "direção", "confiança"]) if pt else
                     " | ".join(["variant", "expected", "computed", "direction", "confidence"])) + " |",
             "|---|---|---|---|---|"]
        for r in self.table:
            ok = "—" if r["direction_ok"] is None else ("✔" if r["direction_ok"] else "✘")
            L.append(f"| {r['variant']} | {r['expected_pct']:+.0f}% | {r['computed_pct']:+.0f}% | {ok} | {r['confidence']} |")
        L.append("")
        L.append((f"Direção certa em {self.direction_agreement:.0%} das variantes; correlação de ordem "
                  f"{self.rank_correlation:.2f}.") if pt else
                 (f"Right direction for {self.direction_agreement:.0%} of the variants; rank correlation "
                  f"{self.rank_correlation:.2f}."))
        wrong = [r["variant"] for r in self.table if r["direction_ok"] is False]
        if wrong:
            L.append(("A leitura qualitativa errou em: " if pt else "The qualitative reading was wrong for: ")
                     + ", ".join(wrong) + (" (veja os avisos de confiança)." if pt else " (see the confidence notes)."))
        return "\n".join(L)


# ------------------------------------------------------------------------------------------------ main entry
# quantities whose change is explained by the terms of another one (sign: +1 same direction, -1 inverse)
_RELATED = {"drag": ("drag", 1), "drag_coefficient": ("drag", 1), "heat_rate": ("heat_rate", 1),
            "mean_h": ("heat_rate", 1), "thermal_resistance": ("heat_rate", -1), "cooling_time": ("heat_rate", -1),
            "tip_deflection": ("tip_deflection", 1), "stiffness": ("tip_deflection", -1),
            "stiffness_to_mass": ("tip_deflection", -1), "first_frequency": ("tip_deflection", -1)}


def _terms_for(q: str, ev: Evaluation):
    tq, sign = _RELATED.get(q, (q, 1))
    return (tq, sign) if tq in ev.terms else (None, 1)


def _confidence(evb: Evaluation, ev: Evaluation, mech: list[tuple[str, float]], total: float, geo_change: float,
                lang: str) -> tuple[str, list[tuple[str, str]]]:
    reasons = [(r[0], r[1]) for r in ev.caveats]
    score = sum(r[2] if len(r) > 2 else 1 for r in ev.caveats)
    pos = sum(s for _, s in mech if s > 0)
    neg = -sum(s for _, s in mech if s < 0)
    if min(pos, neg) > 0.5 * max(pos, neg) and max(pos, neg) > 5:
        reasons.append(("mecanismos competindo em sentidos opostos com pesos parecidos: o saldo é incerto",
                        "competing mechanisms of similar size pull in opposite directions: the net effect is uncertain"))
        score += 1
    if abs(total) < 5 and max(pos, neg, 0) > 3:
        reasons.append(("mudança pequena perto da margem do modelo", "small change, within the model's margin"))
        score += 1
    if geo_change > 1.0:
        reasons.append(("mudança geométrica grande (mais que dobra alguma dimensão): extrapola as correlações",
                        "large geometric change (some dimension more than doubles): the correlations are stretched"))
        score += 1
    levels = (("alta", "média", "baixa") if lang == "pt" else ("high", "medium", "low"))
    return levels[min(score, 2)], reasons


def preview(geometries: Mapping[str, Any], objective: str | tuple | Objective,
            conditions: Mapping[str, Any] | None = None, baseline: str | None = None,
            model: QualitativeModel | str | None = None, lang: str = "pt") -> QualitativePreview:
    """Qualitative comparison of ``geometries`` (name -> mesh tuple, trimesh-like object or ``Assembly``) for an
    ``objective`` such as "minimizar arrasto", "maximize heat rejection", "aumentar rigidez por massa" or
    ``("tip_deflection", "min")``. ``conditions`` are the model's operating conditions (speed, temperature
    difference, material, load...). The first geometry is the baseline unless ``baseline`` is given."""
    obj = parse_objective(objective, model if isinstance(model, QualitativeModel) else None)
    if isinstance(model, QualitativeModel):
        m = model
        if conditions:
            m.conditions.update(conditions)
    else:
        m = MODELS[model or obj.model](**(conditions or {}))
    if obj.quantity not in m.quantities:
        raise ValueError(f"{m.name} does not provide {obj.quantity!r}")
    names = list(geometries)
    base = baseline or names[0]
    evs = {n: m.evaluate(g) for n, g in geometries.items()}
    evb = evs[base]
    q = obj.quantity
    b = evb.quantities[q]
    ext_b = np.ptp(evb.mesh[0], axis=0)
    out = {}
    for n, ev in evs.items():
        change = {}
        for k, v in ev.quantities.items():
            b0 = evb.quantities[k]
            # percentages of a baseline that is (numerically) zero mean nothing
            change[k] = 100 * (v - b0) / abs(b0) if abs(b0) > 1e-6 * (abs(v) + abs(b0)) and b0 != 0 else (
                0.0 if v == b0 else float("nan"))
        tq, sign = _terms_for(q, ev)
        mech, parts = [], []
        if tq is not None:
            bt = evb.terms[tq]
            scale = abs(evb.quantities[tq]) or 1.0
            mech = [(k, sign * 100 * (ev.terms[tq].get(k, 0.0) - bt.get(k, 0.0)) / scale) for k in bt]
            if q.endswith("_to_mass") and "mass" in change:      # ratio objectives: the denominator is a mechanism
                mech.append(("mass", -change["mass"]))
            mech.sort(key=lambda t: -abs(t[1]))
            if tq in ev.by_part and ev.by_part[tq]:
                bp = evb.by_part.get(tq, {})
                keys = list(dict.fromkeys(list(bp) + list(ev.by_part[tq])))
                parts = sorted(((k, 100 * (ev.by_part[tq].get(k, 0.0) - bp.get(k, 0.0)) / scale) for k in keys),
                               key=lambda t: -abs(t[1]))
        elif hasattr(m, "log_attribution") and n != base:
            la = m.log_attribution(geometries[base], geometries[n], q)
            mech = sorted(((k, 100 * (math.exp(v) - 1)) for k, v in la.items()), key=lambda t: -abs(t[1]))
        ext = np.ptp(ev.mesh[0], axis=0)
        geo = float(np.max(np.abs(np.log(np.maximum(ext, 1e-12) / np.maximum(ext_b, 1e-12))))) / math.log(2)
        conf, reasons = _confidence(evb, ev, mech, change[q], geo, lang)
        good = None if n == base else (change[q] < 0 if obj.sense == "min" else change[q] > 0)
        out[n] = VariantPreview(n, ev, change, mech, parts, conf, reasons, good)
    _ = b
    return QualitativePreview(m, obj, base, out, dict(geometries), lang)


# ------------------------------------------------------------------------------------------------ parts
@dataclass
class PartSensitivity:
    objective: Objective
    model: QualitativeModel
    base_value: float
    rows: list[dict[str, Any]] = field(default_factory=list)
    lang: str = "pt"

    def text(self, lang: str | None = None, top: int = 12) -> str:
        pt = (lang or self.lang) == "pt"
        Q = self.model.quantities[self.objective.quantity]
        sense = ("minimizar" if self.objective.sense == "min" else "maximizar") if pt else self.objective.sense + "imize"
        L = [f"# {'Alavancas por peça' if pt else 'Levers by part'}: {sense} {Q.pt if pt else Q.en}", ""]
        L += ["| " + ("peça | mudança | efeito no objetivo | bom?" if pt else "part | change | effect on the objective | good?")
              + " |", "|---|---|---|---|"]
        for r in self.rows[:top]:
            L.append(f"| {r['part']} | {r['change_pt' if pt else 'change_en']} | {_arrow(r['pct'])} {r['pct']:+.1f}% | "
                     f"{'✔' if r['good'] else ('—' if abs(r['pct']) < 0.5 else '✘')} |")
        return "\n".join(L)

    def figure(self, path: str | None = None, lang: str | None = None):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        pt = (lang or self.lang) == "pt"
        rows = self.rows[::-1]
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(rows) + 1.5))
        ax.barh(range(len(rows)), [r["pct"] for r in rows],
                color=["seagreen" if r["good"] else "indianred" for r in rows])
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([f"{r['part']}: {r['change_pt' if pt else 'change_en']}" for r in rows], fontsize=8)
        ax.axvline(0, color="black", lw=0.8)
        Q = self.model.quantities[self.objective.quantity]
        ax.set_xlabel(f"{Q.pt if pt else Q.en}: %")
        ax.set_title("Onde mexer primeiro (verde ajuda o objetivo)" if pt else "Where to act first (green helps)")
        fig.tight_layout()
        if path is not None:
            fig.savefig(path, dpi=110)
            plt.close(fig)
        return fig


def part_sensitivity(assembly: Assembly, objective: str | tuple | Objective, conditions: Mapping[str, Any] | None = None,
                     step: float = 0.2, model: QualitativeModel | str | None = None, lang: str = "pt",
                     axes: str = "xyz", about: str = "min") -> PartSensitivity:
    """For each part: grow it by ``step`` along each axis (about its minimum corner by default, i.e. away from where
    it is usually attached) and remove it; the change of the objective ranks where to act first."""
    obj = parse_objective(objective, model if isinstance(model, QualitativeModel) else None)
    m = model if isinstance(model, QualitativeModel) else MODELS[model or obj.model](**(conditions or {}))
    b = m.evaluate(assembly).quantities[obj.quantity]
    rows = []
    for part in assembly.names:
        trials = []
        for i, a in enumerate("xyz"):
            if a not in axes:
                continue
            f = [1.0, 1.0, 1.0]
            f[i] = 1 + step
            trials.append((assembly.scaled(part, f, about=about), f"+{step:.0%} em {a}", f"+{step:.0%} along {a}"))
        if len(assembly.names) > 1:
            trials.append((assembly.without(part), "remover", "remove"))
        for g, cpt, cen in trials:
            try:
                v = m.evaluate(g).quantities[obj.quantity]
            except Exception:  # noqa: BLE001 - e.g. a beam that loses its only section
                continue
            pct = 100 * (v - b) / abs(b) if b else 0.0
            rows.append({"part": part, "change_pt": cpt, "change_en": cen, "pct": pct,
                         "good": (pct < -0.5) if obj.sense == "min" else (pct > 0.5)})
    rows.sort(key=lambda r: -abs(r["pct"]))
    return PartSensitivity(obj, m, b, rows, lang)


_ = (Cantilever, ConvectiveCooling, ExternalFlow)
