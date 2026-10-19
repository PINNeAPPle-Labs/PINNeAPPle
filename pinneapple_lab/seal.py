"""The PINNeAPPle quality seal: a small badge with the tier the Lab curation gave an item, stamped on its figures,
renders and movies.

The tier always comes from the curation record (``<lab>/curation.json``, written by ``pinneapple_lab.curation``);
there is no argument to set it by hand, and stamping again after a re-curation picks up the new tier. The one-line
reason under the wordmark is the item's ``headline`` from the same record: the passing check it was compared with
(``check_line``), or for E and F the failing check, never free text.

Every tier is sealed, A to F, so a figure always says where its result stands:

    A  green        independent reference            D  orange      exploratory, not validated
    B  light green  reference pending                E  red         known problems
    C  yellow       partly checked                   F  dark red    not usable

E and F seals always carry the main problem (the failing check, the crash or the divergence) and the next step to
the tier above it (``headline["hint"]``), whatever ``reason`` says, and have no compact form. ``tiers=("A", "B")``
on :func:`seal_for` or :func:`stamp_all` (``--tiers A,B`` on the command line) keeps a selection to the validated
items: anything else raises :class:`SealError` or is skipped.

    from pinneapple_lab.seal import seal_for, render_seal, stamp_figure, stamp_file
    s = seal_for("repo_results/ahmed_body")             # tier and reason from lab/curation.json
    render_seal(s).save("seal.png"); open("seal.svg", "w").write(render_seal(s, fmt="svg", lang="pt"))
    stamp_figure(fig, s)                                 # matplotlib: a corner without colour bars or text
    stamp_file("render.jpg", s)                          # PNG / JPG / GIF / MP4 -> render.sealed.jpg
    with sealed_savefig(s): fig.savefig("plot.png")      # every savefig in the block is stamped

    python -m pinneapple_lab seal repo_results/ahmed_body -o seal.svg
    python -m pinneapple_lab stamp repo_results/ahmed_body figure.png -o figure_sealed.png
    python -m pinneapple_lab stamp --all --out export/sealed                 # every figure of every item, A to F
    python -m pinneapple_lab stamp --all --tiers A,B --out export/validated  # only the validated ones
    python -m pinneapple_lab legend -o docs/assets/seals/seal_legend_en.png
"""
from __future__ import annotations

import contextlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = ["CORNERS", "SEALABLE", "TIER_COLOURS", "TIER_INK", "TIER_LINE", "TIER_TEXT", "VALIDATED", "Seal",
           "SealError", "contrast_ratio", "corner_scores", "items", "legend_seals", "load_curation", "parse_tiers",
           "pick_corner", "render_legend",
           "render_seal", "save_seal", "seal_for", "sealed_savefig", "stamp_all", "stamp_figure", "stamp_file",
           "stamp_image", "stamp_video"]

BRAND_YELLOW = "#FFD60A"
BADGE_BG = "#0F1420"
# fill of the letter disc; a ramp from green to dark red, the letter always printed so no tier is told by colour alone
TIER_COLOURS = {"A": "#2FBF71", "B": "#A6D96A", "C": "#FFE066", "D": "#FDAE61", "E": "#F0603C", "F": "#A50026"}
TIER_INK = {t: (BADGE_BG if t != "F" else "#FFFFFF") for t in TIER_COLOURS}          # the letter on the disc
TIER_LINE = {**TIER_COLOURS, "F": "#FF8A80"}          # outline and tier line: dark red is too dark on the badge
TIER_TEXT = {"en": {"A": "INDEPENDENT REFERENCE", "B": "REFERENCE PENDING", "C": "PARTLY CHECKED",
                    "D": "EXPLORATORY · NOT VALIDATED", "E": "KNOWN PROBLEMS", "F": "NOT USABLE"},
             "pt": {"A": "REFER\u00caNCIA INDEPENDENTE", "B": "REFER\u00caNCIA PENDENTE", "C": "VERIFICA\u00c7\u00c3O PARCIAL",
                    "D": "EXPLORAT\u00d3RIO · N\u00c3O VALIDADO", "E": "PROBLEMAS CONHECIDOS",
                    "F": "N\u00c3O UTILIZ\u00c1VEL"}}
SEALABLE = tuple(TIER_COLOURS)                # every tier is sealed
VALIDATED = ("A", "B")                        # ``tiers=VALIDATED``: only items with an independent check
MUST_EXPLAIN = ("E", "F")                     # always printed with the failing check and the next step
CORNERS = ("br", "tr", "bl", "tl")          # preference order when two corners are equally free
IMAGE_EXT = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
VIDEO_EXT = (".mp4", ".webm", ".mov", ".m4v")


class SealError(ValueError):
    """The item cannot carry a seal here (its tier is outside the ``tiers`` asked for)."""


@dataclass(frozen=True)
class Seal:
    item: str
    tier: str
    title: str
    reason: str          # the curation headline: the check it was compared with, or for E / F the failing check
    reason_kind: str     # "reference", "benchmark", "verification", "failing" or ""
    curated: str         # when the curation that gave the tier ran
    experiment: str
    run_id: str
    hint: str = ""       # the next step to the tier above ("D: fix the failing check"; "A+: ..." for A)
    reason_pt: str = ""
    hint_pt: str = ""

    @property
    def colour(self) -> str:
        return TIER_COLOURS[self.tier]

    @property
    def ink(self) -> str:
        return TIER_INK[self.tier]

    @property
    def line(self) -> str:
        return TIER_LINE[self.tier]

    def text(self, lang: str = "en") -> tuple[str, str]:
        """(reason, hint) in ``lang``, the English line where there is no Portuguese one."""
        if lang == "pt":
            return self.reason_pt or self.reason, self.hint_pt or self.hint
        return self.reason, self.hint


# ---------------------------------------------------------------------------------------------- curation records
def load_curation(lab: Any = None) -> dict[str, Any]:
    """The curation record: ``lab`` is a lab folder, a ``curation.json`` path, a ``LabStore``, an already loaded
    dict, or None for the default lab (``$PINNEAPPLE_LAB`` or ``./lab``)."""
    if isinstance(lab, dict):
        return lab
    if lab is None:
        from .store import default_root
        lab = default_root()
    root = getattr(lab, "root", lab)
    path = root if str(root).endswith(".json") else os.path.join(root, "curation.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run `python -m pinneapple_lab curate` first")
    with open(path) as f:
        out = json.load(f)
    out.setdefault("_root", os.path.dirname(os.path.abspath(path)))
    return out


def parse_tiers(tiers) -> tuple[str, ...]:
    """``"A,B"``, ``["A", "B"]`` or None (every tier) as a tuple of known tier letters."""
    if tiers is None:
        return SEALABLE
    if isinstance(tiers, str):
        tiers = tiers.split(",")
    out = tuple(dict.fromkeys(t.strip().upper() for t in tiers if t.strip()))
    bad = [t for t in out if t not in SEALABLE]
    if bad:
        raise ValueError(f"unknown tier {', '.join(bad)}: use letters from {', '.join(SEALABLE)}")
    return out


def items(lab: Any = None, tiers=SEALABLE) -> list[str]:
    """Item keys (``experiment`` or ``experiment/case``) curated at one of ``tiers``."""
    cur = load_curation(lab)
    tiers = parse_tiers(tiers)
    return sorted(k for k, v in cur.get("items", {}).items() if v.get("tier") in tiers)


def seal_for(item: str, lab: Any = None, tiers=None) -> Seal:
    """The seal of a curated item, with the tier, reason and (E / F) next step of its curation record. Raises
    ``KeyError`` for an unknown item and :class:`SealError` when ``tiers`` (e.g. ``"A,B"``) leaves its tier out."""
    cur = load_curation(lab)
    rec = cur.get("items", {}).get(item)
    if rec is None:
        raise KeyError(f"{item!r} is not in the curation; known: {', '.join(sorted(cur.get('items', {}))[:12])} ...")
    tier = rec.get("tier")
    if tier not in SEALABLE:
        raise SealError(f"{item}: tier {tier!r} is not a curation tier (re-run `python -m pinneapple_lab curate`)")
    if tier not in parse_tiers(tiers):
        raise SealError(f"{item} is tier {tier} ({rec.get('tier_name', '')}): outside the tiers asked for "
                        f"({','.join(parse_tiers(tiers))})")
    hl = rec.get("headline") or {}
    reason = hl.get("label") or hl.get("text", "")
    return Seal(item=item, tier=tier, title=rec.get("title") or item, reason=reason,
                reason_kind=hl.get("kind", ""), curated=str(cur.get("generated", "")),
                experiment=rec.get("experiment", item.split("/")[0]), run_id=rec.get("best_run", ""),
                hint=hl.get("hint") or _first_step(rec), reason_pt=hl.get("label_pt") or hl.get("text_pt", "") or reason,
                hint_pt=hl.get("hint_pt") or _first_step(rec, "pt"))


def _first_step(rec: dict[str, Any], lang: str = "en") -> str:
    """The seal's "next" line: the first gap of the item's path to the next letter (``rec["next_steps"]``)."""
    steps = rec.get("next_steps") or []
    if not steps:
        return ""
    g = steps[0]
    return f"{g['to']}: {g['short_pt'] if lang == 'pt' else g['short']}"


def run_dir(seal: Seal, lab: Any = None) -> str:
    cur = load_curation(lab)
    return os.path.join(cur["_root"], "runs", seal.experiment, seal.run_id)


# ---------------------------------------------------------------------------------------------- drawing
def _font(bold: bool = True):
    from matplotlib import font_manager as fm
    for fam in ("Inter", "DejaVu Sans"):
        path = fm.findfont(fm.FontProperties(family=fam, weight="bold" if bold else "medium"),
                           fallback_to_default=True)
        if path:
            return path
    return None


def _pil_font(size: float, bold: bool = True):
    from PIL import ImageFont
    path = _font(bold)
    try:
        return ImageFont.truetype(path, max(1, int(round(size))))
    except Exception:  # noqa: BLE001 - no TrueType font: PIL's default bitmap font
        return ImageFont.load_default()


def _symbol_font(size: float):
    """DejaVu Sans Bold for the check marks (Inter's half-circle is a bar at small sizes)."""
    from matplotlib import font_manager as fm
    from PIL import ImageFont
    try:
        return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans", weight="bold")),
                                  max(1, int(round(size))))
    except Exception:  # noqa: BLE001
        return None


_MARKS = ("\u2713", "\u25d0", "\u2717", "\u2191")      # passed, partial, failed, next step


def draw_text(draw, xy, text: str, font, fill, anchor: str = "ls") -> None:
    """``ImageDraw.text`` with a leading mark (\u2713, \u25d0, \u2717 or \u2191) drawn in DejaVu Sans."""
    if text[:1] in _MARKS:
        sf = _symbol_font(font.size)
        if sf is not None:
            draw.text(xy, text[0], font=sf, fill=fill, anchor=anchor)
            dx = sf.getlength(text[0] + " ")
            draw.text((xy[0] + dx, xy[1]), text[2:] if text[1:2] == " " else text[1:], font=font, fill=fill,
                      anchor=anchor)
            return
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _bezier(p0, p1, p2, p3, n=12):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(p, float) for p in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


# the pineapple mark of apps/playground/site/icon.svg (64-unit box), as cubic segments
_LEAVES = [[((32, 6), (35, 11), (35, 15), (32, 19)), ((32, 19), (29, 15), (29, 11), (32, 6))],
           [((22, 10), (27, 12), (30, 16), (30, 21)), ((30, 21), (25, 20), (22, 16), (22, 10))],
           [((42, 10), (37, 12), (34, 16), (34, 21)), ((34, 21), (39, 20), (42, 16), (42, 10))]]
_LEAF_PATHS = ["M32 6c3 5 3 9 0 13-3-4-3-8 0-13z", "M22 10c5 2 8 6 8 11-5-1-8-5-8-11z",
               "M42 10c-5 2-8 6-8 11 5-1 8-5 8-11z"]
_HATCH = [((20, 30), (44, 48)), ((18, 40), (38, 55)), ((24, 24), (46, 41)), ((44, 30), (20, 48)),
          ((46, 40), (26, 55)), ((40, 24), (18, 41))]
_MARK_BOX = (14.0, 4.0, 50.0, 60.0)          # x0, y0, x1, y1 of the mark inside the icon box


_LINE_STEP = 0.19                             # height of one extra small-print line (E / F hint), x the base height


def _mark(seal: Seal) -> str:
    return {"reference": "\u2713 ", "failing": "\u2717 ", "": ""}.get(seal.reason_kind, "\u25d0 ")


def _wrap(text: str, size: float, width: float, max_lines: int) -> list[str]:
    """``text`` broken at spaces into at most ``max_lines`` lines of at most ``width`` px at font ``size``; only
    what still does not fit after that is cut (with "…"). Continuation lines are indented past the mark."""
    f = _pil_font(size * 4, False)
    fits = lambda t: f.getlength(t) / 4 <= width  # noqa: E731
    out, rest = [], text
    while rest and len(out) < max_lines:
        if fits(rest) or len(out) == max_lines - 1:
            line, rest = rest, ""
        else:
            words, line = rest.split(" "), ""
            for i, wd in enumerate(words):
                cand = (line + " " + wd) if line else wd
                if not fits(cand) and line:
                    rest = " ".join(words[i:])
                    break
                line = cand
            else:
                rest = ""
        if not fits(line):
            while line and not fits(line + "…"):
                line = line[:-1]
            line = line.rstrip(" ,;·") + "…"
        out.append(line if not out else "   " + line.lstrip())
    return out


def _layout(seal: Seal, height: float, reason: bool, compact: bool, lang: str = "en",
            next_step: bool = True) -> dict[str, Any]:
    """Positions in pixels for a badge ``height`` tall (both renderers use it). An E or F badge is never compact and
    always has the reason and the next-step lines, so it is taller than ``height`` by one small-print line."""
    h = float(height)
    pad = 0.16 * h
    must = seal.tier in MUST_EXPLAIN
    compact = compact and not must
    reason = reason or must
    lines = []
    if compact:
        lines.append(("PINNeAPPle", 0.36 * h, BRAND_YELLOW, True, 0.5 * h + 0.13 * h))
    else:
        top = 0.30 * h
        lines.append(("PINNeAPPle", top, BRAND_YELLOW, True, pad + 0.80 * top))
        tier_text = TIER_TEXT.get(lang, TIER_TEXT["en"])[seal.tier]
        lines.append((f"TIER {seal.tier} · {tier_text}", 0.145 * h, seal.line, True,
                      pad + top + 0.21 * h))
        why, hint = seal.text(lang)
        small = []
        if reason and why:
            small.append((_mark(seal) + why, "#E6E9F0"))
        if (must or next_step) and hint:
            small.append(("\u2191 " + hint, seal.line))
        k = 0
        for text, col in small:                 # up to two lines each, so the whole check and action can be read
            for part in _wrap(text, 0.13 * h, 3.4 * h, 2):
                lines.append((part, 0.13 * h, col, False, pad + top + (0.40 + _LINE_STEP * k) * h))
                k += 1
    H = h + _LINE_STEP * h * max(0, len(lines) - 3)
    widths = [_pil_font(size * 4, bold).getlength(text) / 4 for text, size, _, bold, _ in lines]
    mark_h = h - 2 * pad
    mark_w = mark_h * (_MARK_BOX[2] - _MARK_BOX[0]) / (_MARK_BOX[3] - _MARK_BOX[1])
    x = pad + mark_w + 0.12 * h
    disc = 0.66 * h
    tw = max(widths)
    w = x + tw + 0.18 * h + disc + pad
    return {"w": w, "h": H, "base": h, "pad": pad, "mark": (pad, (H - mark_h) / 2, mark_w, mark_h), "x": x,
            "lines": lines, "widths": widths, "disc": (w - pad - disc / 2, H / 2, disc / 2), "radius": 0.22 * h}


def render_seal(seal: Seal, *, height: int = 96, fmt: str = "png", reason: bool = True,
                compact: bool | None = None, opacity: float = 1.0, lang: str = "en", next_step: bool = True):
    """The badge as a PIL RGBA image (``fmt="png"``) or an SVG string (``fmt="svg"``). Below 56 px it switches to
    the compact form (mark, wordmark and tier letter) so it stays legible; ``compact`` forces either form. ``lang``
    ("en", "pt") sets the language of the tier line, the reason and the hint. E and F seals ignore ``compact`` and
    ``reason=False``: they always show the failing check and the next step (and are one line taller). The full form
    of an A-D seal also carries the "next" line (what the item lacks for the next letter, ``seal.hint``) unless
    ``next_step=False``."""
    compact = height < 56 if compact is None else compact
    L = _layout(seal, height, reason, compact, lang, next_step)
    return _svg(seal, L, opacity) if fmt == "svg" else _png(seal, L, opacity)


def _png(seal: Seal, L: dict[str, Any], opacity: float):
    from PIL import Image, ImageDraw
    s = 4                                                   # supersampling, then a Lanczos downscale
    W, H = int(round(L["w"] * s)), int(round(L["h"] * s))
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    lw = max(1, int(0.035 * L["base"] * s))
    d.rounded_rectangle((0, 0, W - 1, H - 1), radius=L["radius"] * s, fill=_rgba(BADGE_BG, 0.92),
                        outline=_rgba(seal.line, 1.0), width=lw)
    mx, my, mw, mh = (v * s for v in L["mark"])
    k = mh / (_MARK_BOX[3] - _MARK_BOX[1])

    def tr(p):
        return (mx + (p[0] - _MARK_BOX[0]) * k, my + (p[1] - _MARK_BOX[1]) * k)

    for leaf in _LEAVES:
        pts = np.vstack([_bezier(*seg) for seg in leaf])
        d.polygon([tr(p) for p in pts], fill=_rgba("#3FBF7F", 1.0))
    body = (tr((17, 20)), tr((47, 58)))
    d.ellipse((body[0][0], body[0][1], body[1][0], body[1][1]), fill=_rgba("#F2B632", 1.0))
    hatch = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    hd = ImageDraw.Draw(hatch)
    for a, b in _HATCH:
        hd.line((tr(a), tr(b)), fill=_rgba("#B37A10", 0.7), width=max(1, int(2 * k)))
    mask = Image.new("L", (W, H), 0)
    ImageDraw.Draw(mask).ellipse((body[0][0], body[0][1], body[1][0], body[1][1]), fill=255)
    im.alpha_composite(Image.composite(hatch, Image.new("RGBA", (W, H), (0, 0, 0, 0)), mask))
    for (text, size, col, bold, base), _ in zip(L["lines"], L["widths"], strict=True):
        f = _pil_font(size * s, bold)
        draw_text(d, (L["x"] * s, base * s), text, f, _rgba(col, 1.0))
    cx, cy, r = (v * s for v in L["disc"])
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=_rgba(seal.colour, 1.0))
    d.text((cx, cy), seal.tier, font=_pil_font(1.25 * r, True), fill=_rgba(seal.ink, 1.0), anchor="mm")
    out = im.resize((max(1, int(round(L["w"]))), max(1, int(round(L["h"])))), Image.LANCZOS)
    if opacity < 1:
        a = np.asarray(out).copy()
        a[..., 3] = (a[..., 3] * float(opacity)).astype(np.uint8)
        out = Image.fromarray(a, "RGBA")
    return out


def _svg(seal: Seal, L: dict[str, Any], opacity: float) -> str:
    from xml.sax.saxutils import escape
    mx, my, mw, mh = L["mark"]
    k = mh / (_MARK_BOX[3] - _MARK_BOX[1])
    w, h, b = L["w"], L["h"], L["base"]
    label = f"PINNeAPPle seal: tier {seal.tier}, {seal.item}"
    if seal.tier in MUST_EXPLAIN:
        label += f": {seal.reason}; {seal.hint}"
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w:.0f}" height="{h:.0f}" viewBox="0 0 {w:.2f} {h:.2f}" '
         f'opacity="{opacity:g}" role="img" aria-label="{escape(label, {chr(34): "&quot;"})}">',
         f"<title>PINNeAPPle tier {seal.tier} · {escape(seal.item)} · curated {escape(seal.curated)}</title>",
         '<defs><clipPath id="pnp-body"><ellipse cx="32" cy="39" rx="15" ry="19"/></clipPath></defs>',
         f'<rect x="{0.0175 * b:.2f}" y="{0.0175 * b:.2f}" width="{w - 0.035 * b:.2f}" height="{h - 0.035 * b:.2f}" '
         f'rx="{L["radius"]:.2f}" fill="{BADGE_BG}" fill-opacity="0.92" stroke="{seal.line}" '
         f'stroke-width="{0.035 * b:.2f}"/>',
         f'<g transform="translate({mx - _MARK_BOX[0] * k:.3f} {my - _MARK_BOX[1] * k:.3f}) scale({k:.4f})">']
    o += [f'<path d="{p}" fill="#3FBF7F"/>' for p in _LEAF_PATHS]
    o.append('<ellipse cx="32" cy="39" rx="15" ry="19" fill="#F2B632"/>')
    o.append('<g clip-path="url(#pnp-body)" stroke="#B37A10" stroke-width="2" stroke-linecap="round" opacity=".7">'
             + "".join(f'<line x1="{a[0]}" y1="{a[1]}" x2="{b[0]}" y2="{b[1]}"/>' for a, b in _HATCH) + "</g></g>")
    for (text, size, col, bold, base), tw in zip(L["lines"], L["widths"], strict=True):
        o.append(f'<text x="{L["x"]:.2f}" y="{base:.2f}" font-family="Inter, \'DejaVu Sans\', Arial, sans-serif" '
                 f'font-size="{size:.2f}" font-weight="{700 if bold else 500}" fill="{col}" textLength="{tw:.2f}" '
                 f'lengthAdjust="spacingAndGlyphs">{escape(text)}</text>')
    cx, cy, r = L["disc"]
    o.append(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{seal.colour}"/>')
    o.append(f'<text x="{cx:.2f}" y="{cy + 0.44 * r:.2f}" text-anchor="middle" font-family="Inter, \'DejaVu Sans\', '
             f'Arial, sans-serif" font-size="{1.25 * r:.2f}" font-weight="800" fill="{seal.ink}">{seal.tier}</text>')
    o.append("</svg>")
    return "\n".join(o)


def _luminance(hex_colour: str) -> float:
    c = np.array(_rgba(hex_colour, 1.0)[:3]) / 255.0
    c = np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    return float(0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2])


def contrast_ratio(fg: str, bg: str = BADGE_BG) -> float:
    """WCAG 2 contrast ratio of two ``#rrggbb`` colours (4.5 is the minimum for small text)."""
    a, b = sorted((_luminance(fg), _luminance(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def _rgba(hex_colour: str, alpha: float) -> tuple[int, int, int, int]:
    h = hex_colour.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), int(round(255 * alpha))


def save_seal(seal: Seal, path: str, **kw) -> str:
    """Write the badge as ``.svg`` or a raster image (by extension)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if path.lower().endswith(".svg"):
        with open(path, "w") as f:
            f.write(render_seal(seal, fmt="svg", **kw))
    else:
        render_seal(seal, **kw).save(path)
    return path


# ---------------------------------------------------------------------------------------------- where to put it
def _box(corner: str, W: int, H: int, w: int, h: int, m: int) -> tuple[int, int, int, int]:
    x = W - w - m if corner[1] == "r" else m
    y = H - h - m if corner[0] == "b" else m
    return x, y, x + w, y + h


def _overlap(a, b) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


QUIET = 4.0          # corner score below which the badge covers nothing that matters (plain background ~ 0-1)


def pick_corner(frames, w: int, h: int, margin: int, avoid=()) -> str:
    """The corner where a ``w`` x ``h`` badge covers least: lowest mean colour gradient plus colour spread under
    it (colour bars, axis text and the subject all score high), never one overlapping a box in ``avoid``
    ((x0, y0, x1, y1) pixels from the top left). ``frames``: one image or several (a movie)."""
    scores = corner_scores(frames, w, h, margin, avoid)
    if not scores:
        return CORNERS[0]
    best = min(scores.values())
    return next(c for c in CORNERS if c in scores and scores[c] <= best + 1e-9)


def corner_scores(frames, w: int, h: int, margin: int, avoid=()) -> dict[str, float]:
    """Busyness of each usable corner (see :func:`pick_corner`); below ``QUIET`` the corner is plain background."""
    if not isinstance(frames, (list, tuple)):
        frames = [frames]
    H, W = np.asarray(frames[0]).shape[:2]
    scores = {}
    for c in CORNERS:
        b = _box(c, W, H, w, h, margin)
        if any(_overlap(b, a) for a in avoid):
            continue
        s = 0.0
        for fr in frames:
            reg = np.asarray(fr)[max(0, b[1] - margin // 2):b[3] + margin // 2,
                                 max(0, b[0] - margin // 2):b[2] + margin // 2, :3].astype(float)
            if reg.size == 0:
                continue
            g = np.abs(np.diff(reg, axis=0)).mean() + np.abs(np.diff(reg, axis=1)).mean()
            s += g + 0.5 * reg.reshape(-1, 3).std(axis=0).mean()
        scores[c] = s / len(frames)
    return scores


def _size_badge(seal: Seal, W: int, H: int, size: float, reason: bool, lang: str = "en", next_step: bool = True):
    """The badge scaled so its width is ``size`` x the image width (at least 28 px tall, at most 30 % of it). E and
    F badges keep their full form, so they are at least 56 px tall unless the image is too small for that."""
    probe = render_seal(seal, height=100, reason=reason, lang=lang, next_step=next_step)
    aspect = probe.width / probe.height
    lo = 56 if seal.tier in MUST_EXPLAIN else 28
    hgt = int(min(max(size * W / aspect, lo), max(0.3 * H, 28)))
    return render_seal(seal, height=hgt, reason=reason, lang=lang, next_step=next_step)


# ---------------------------------------------------------------------------------------------- stamping
def stamp_image(image, seal: Seal, *, corner: str = "auto", size: float = 0.24, opacity: float = 0.95,
                margin: float = 0.02, reason: bool = True, avoid=(), extend: str | bool = "auto", lang: str = "en",
                next_step: bool = True):
    """Stamp a PIL image (or an array, or a path) and return the stamped PIL image. With ``corner="auto"`` the
    badge goes in the emptiest corner; when every corner is busy (a plot filled to its edges) and ``extend`` is
    "auto", the canvas grows by a strip in the image's border colour and the badge sits there, covering nothing.
    ``extend=True`` always adds the strip, ``False`` never. ``lang`` ("en", "pt") is the seal's language."""
    from PIL import Image
    im = Image.open(image) if isinstance(image, (str, os.PathLike)) else \
        Image.fromarray(np.asarray(image)) if not hasattr(image, "convert") else image
    mode = im.mode
    base = im.convert("RGBA")
    badge = _size_badge(seal, base.width, base.height, size, reason, lang, next_step)
    m = int(round(margin * min(base.size)))
    rgb = np.asarray(base.convert("RGB"))
    if corner == "auto":
        sc = corner_scores(rgb, badge.width, badge.height, m, avoid)
        corner = pick_corner(rgb, badge.width, badge.height, m, avoid)
        if extend == "auto":
            extend = bool(not sc or min(sc.values()) > QUIET)
    if extend is True:
        base = _extend(base, badge.height + 2 * m, rgb)
        corner = "br"
    x0, y0, _, _ = _box(corner, base.width, base.height, badge.width, badge.height, m)
    a = np.asarray(badge).copy()
    a[..., 3] = (a[..., 3] * float(opacity)).astype(np.uint8)
    base.alpha_composite(Image.fromarray(a, "RGBA"), (x0, y0))
    return base if mode == "RGBA" else base.convert("RGB")


def _extend(base, dh: int, rgb: np.ndarray):
    """``base`` with a strip ``dh`` px tall under it, in the colour of its bottom edge."""
    from PIL import Image
    col = tuple(int(v) for v in np.median(rgb[-1], axis=0)) + (255,)
    out = Image.new("RGBA", (base.width, base.height + dh), col)
    out.alpha_composite(base, (0, 0))
    return out


def stamp_figure(fig, seal: Seal, *, corner: str = "auto", size: float = 0.24, opacity: float = 0.95,
                 margin: float = 0.015, reason: bool = True, extend: str | bool = "auto", lang: str = "en",
                next_step: bool = True):
    """Stamp a matplotlib figure in place (an image axes in figure coordinates, drawn on top) and return it. With
    ``corner="auto"`` it keeps off colour bars and picks the emptiest remaining corner; when every corner is busy
    (``extend="auto"``) the figure grows by a strip at the bottom, its contents move up, and the seal goes there.
    Stamping again replaces the previous seal."""
    for ax in list(fig.axes):
        if ax.get_label() == "pinneapple-seal":
            ax.remove()
    fig.canvas.draw()
    W, H = fig.canvas.get_width_height()
    badge = _size_badge(seal, W, H, size, reason, lang, next_step)
    m = int(round(margin * min(W, H)))
    if corner == "auto":
        rgb = np.asarray(fig.canvas.buffer_rgba())[..., :3]
        avoid = []
        for ax in fig.axes:
            if getattr(ax, "_colorbar", None) is not None or ax.get_label() == "<colorbar>":
                bb = ax.get_tightbbox(fig.canvas.get_renderer())
                avoid.append((bb.x0, H - bb.y1, bb.x1, H - bb.y0))
        sc = corner_scores(rgb, badge.width, badge.height, m, avoid)
        corner = pick_corner(rgb, badge.width, badge.height, m, avoid)
        if extend == "auto":
            extend = bool(not sc or min(sc.values()) > QUIET)
    if extend is True:
        H = _grow_figure(fig, badge.height + 2 * m)
        corner = "br"
    x0, y0, x1, y1 = _box(corner, W, H, badge.width, badge.height, m)
    ax = fig.add_axes((x0 / W, 1 - y1 / H, (x1 - x0) / W, (y1 - y0) / H), label="pinneapple-seal")
    a = np.asarray(badge).astype(float) / 255.0
    a[..., 3] *= opacity
    ax.imshow(a, interpolation="lanczos")
    ax.set_axis_off()
    ax.set_zorder(1e6)
    return fig


def _grow_figure(fig, dh_px: int) -> int:
    """Add ``dh_px`` pixels under the figure's contents (layout frozen first) and return the new height."""
    W, H = fig.canvas.get_width_height()
    if fig.get_layout_engine() is not None:
        fig.set_layout_engine("none")
    k = H / (H + dh_px)
    for ax in fig.axes:
        p = ax.get_position()
        ax.set_position((p.x0, 1 - (1 - p.y0) * k, p.width, p.height * k))
    for t in fig.texts:
        x, y = t.get_position()
        t.set_position((x, 1 - (1 - y) * k))
    w_in, h_in = fig.get_size_inches()
    fig.set_size_inches(w_in, h_in * (H + dh_px) / H, forward=False)
    fig.canvas.draw()
    return fig.canvas.get_width_height()[1]


@contextlib.contextmanager
def sealed_savefig(seal: Seal, **stamp_kw):
    """Inside the block, every ``Figure.savefig`` stamps the figure first (a savefig hook)."""
    from matplotlib.figure import Figure
    orig = Figure.savefig

    def savefig(self, *a, **kw):
        stamp_figure(self, seal, **stamp_kw)
        return orig(self, *a, **kw)

    Figure.savefig = savefig
    try:
        yield
    finally:
        Figure.savefig = orig


def _ffmpeg() -> str:
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001
        exe = shutil.which("ffmpeg")
        if not exe:
            raise RuntimeError("stamping a video needs ffmpeg (pip install imageio-ffmpeg)") from None
        return exe


def _video_frames(path: str, n: int = 5) -> tuple[list[np.ndarray], int, int]:
    import imageio_ffmpeg
    rd = imageio_ffmpeg.read_frames(path)
    meta = rd.__next__()
    W, H = meta["size"]
    frames = []
    for i, fr in enumerate(rd):
        if i % 10 == 0:
            frames.append(np.frombuffer(fr, np.uint8).reshape(H, W, 3))
        if len(frames) >= n:
            break
    rd.close()
    return frames, W, H


def stamp_video(src: str, seal: Seal, out: str, *, corner: str = "auto", size: float = 0.24, opacity: float = 0.95,
                margin: float = 0.02, reason: bool = True, extend: str | bool = "auto", lang: str = "en",
                next_step: bool = True) -> str:
    """Stamp a GIF (frame by frame, keeping the frame times) or an MP4 / WebM / MOV (ffmpeg overlay, H.264 yuv420p,
    faststart, audio copied). The corner is chosen once, on several frames; when every corner is busy the frames
    grow by a strip that carries the badge (``extend``, as for images)."""
    from PIL import Image, ImageSequence
    if src.lower().endswith(".gif"):
        g = Image.open(src)
        frames, durs = [], []
        for fr in ImageSequence.Iterator(g):
            frames.append(fr.convert("RGB"))
            durs.append(fr.info.get("duration", g.info.get("duration", 100)))
        probe = [np.asarray(f) for f in frames[:: max(1, len(frames) // 5)]]
        badge = _size_badge(seal, frames[0].width, frames[0].height, size, reason, lang, next_step)
        m = int(round(margin * min(frames[0].size)))
        c = corner
        if corner == "auto":
            c = pick_corner(probe, badge.width, badge.height, m)
            if extend == "auto":
                extend = bool(min(corner_scores(probe, badge.width, badge.height, m).values()) > QUIET)
        outs = [stamp_image(f, seal, corner=c, size=size, opacity=opacity, margin=margin, reason=reason,
                            extend=extend is True, lang=lang, next_step=next_step) for f in frames]
        outs[0].save(out, save_all=True, append_images=outs[1:], duration=durs, loop=g.info.get("loop", 0),
                     optimize=False, disposal=2)
        return out
    frames, W, H = _video_frames(src)
    badge = _size_badge(seal, W, H, size, reason, lang, next_step)
    m = int(round(margin * min(W, H)))
    c = corner
    if corner == "auto":
        c = pick_corner(frames, badge.width, badge.height, m)
        if extend == "auto":
            extend = bool(min(corner_scores(frames, badge.width, badge.height, m).values()) > QUIET)
    pad = ""
    if extend is True:
        dh = badge.height + 2 * m
        dh += dh % 2                                          # yuv420p needs even sizes
        col = "#{:02x}{:02x}{:02x}".format(*tuple(int(v) for v in np.median(frames[0][-1], axis=0)))
        pad = f"pad=iw:ih+{dh}:0:0:color={col},"
        H, c = H + dh, "br"
    x0, y0, _, _ = _box(c, W, H, badge.width, badge.height, m)
    a = np.asarray(badge).copy()
    a[..., 3] = (a[..., 3] * float(opacity)).astype(np.uint8)
    with tempfile.TemporaryDirectory() as tmp:
        png = os.path.join(tmp, "seal.png")
        Image.fromarray(a, "RGBA").save(png)
        cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-i", src, "-i", png, "-filter_complex",
               f"[0:v]{pad}null[b];[b][1:v]overlay={x0}:{y0}:format=auto,format=yuv420p[v]", "-map", "[v]", "-map", "0:a?",
               "-c:v", "libx264", "-crf", "20", "-preset", "medium", "-movflags", "+faststart", "-c:a", "copy", out]
        subprocess.run(cmd, check=True, capture_output=True)
    return out


def _default_out(path: str) -> str:
    stem, ext = os.path.splitext(path)
    return f"{stem}.sealed{ext}"


def stamp_file(path: str, seal: Seal, out: str | None = None, **kw) -> str:
    """Stamp an image (PNG, JPG, WebP), a GIF or a video file; returns the output path (``name.sealed.ext`` by
    default). SVG and PDF are not raster files and are refused. ``kw`` go to :func:`stamp_image` /
    :func:`stamp_video` (``corner``, ``size``, ``opacity``, ``reason``, ``extend``, ``lang``)."""
    out = out or _default_out(path)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    if ext == ".gif" or ext in VIDEO_EXT:
        return stamp_video(path, seal, out, **kw)
    if ext in IMAGE_EXT:
        im = stamp_image(path, seal, **kw)
        if out.lower().endswith((".jpg", ".jpeg")):
            im.convert("RGB").save(out, quality=92)
        else:
            im.save(out)
        return out
    raise ValueError(f"cannot stamp {ext or path!r}: images ({', '.join(IMAGE_EXT)}), .gif and videos only")


def stamp_all(lab: Any = None, out_dir: str = "export/sealed", *, tiers=SEALABLE, only: list[str] | None = None,
              lang: str = "en", **kw) -> dict[str, Any]:
    """Batch mode: every figure, render and movie of every item at one of ``tiers`` (all six by default; ``"A,B"``
    for the validated ones only), from its best run's ``figures/``, stamped into ``out_dir/<item>/``, with the
    item's seal as SVG and PNG and a ``manifest.json`` (item, tier, reason, next step, curation date, files).
    Re-running after a re-curation rewrites every stamp with the new tiers."""
    cur = load_curation(lab)
    tiers = parse_tiers(tiers)
    keys = only or items(cur, tiers)
    manifest = {"curated": cur.get("generated", ""), "tiers": list(tiers), "items": {}}
    for key in keys:
        try:
            s = seal_for(key, cur, tiers)
        except SealError:
            continue
        safe = re.sub(r"[^\w.-]+", "_", key)
        dst = os.path.join(out_dir, safe)
        os.makedirs(dst, exist_ok=True)
        save_seal(s, os.path.join(dst, "seal.svg"), lang=lang)
        save_seal(s, os.path.join(dst, "seal.png"), lang=lang)
        files = []
        src_dir = os.path.join(run_dir(s, cur), "figures")
        for f in sorted(os.listdir(src_dir)) if os.path.isdir(src_dir) else []:
            ext = os.path.splitext(f)[1].lower()
            if ext not in IMAGE_EXT and ext != ".gif" and ext not in VIDEO_EXT:
                continue
            try:
                files.append(os.path.basename(stamp_file(os.path.join(src_dir, f), s, os.path.join(dst, f), lang=lang,
                                                         **kw)))
            except Exception as e:  # noqa: BLE001 - one unreadable file must not stop the batch
                files.append(f"{f}: not stamped ({e})")
        manifest["items"][key] = {"tier": s.tier, "reason": s.reason, "reason_kind": s.reason_kind, "hint": s.hint,
                                  "files": files}
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=1)
    return manifest


# ---------------------------------------------------------------------------------------------- the legend
LEGEND = {   # what each letter means, printed in the legend's sample seals (no item, no number)
    "en": {"title": "PINNeAPPle quality seals", "light": "on a light figure", "dark": "on a dark figure",
           "compact": "compact",
           "A": "compared with an independent reference", "B": "physics, baseline or unseen-data check passes",
           "C": "checks pass, trust card partly answered", "D": "runs, no failing check, no evidence yet",
           "E": "the failing check", "F": "the crash, divergence or failed physics check",
           "hint_A": "A+: what would strengthen it", "hint_B": "A: what it lacks for A",
           "hint_C": "B: what it lacks for B", "hint_D": "C: what it lacks for C",
           "hint_E": "D: the next step from the curation", "hint_F": "E: the next step from the curation",
           "note": "The tier comes from the Lab curation (lab/curation.json), never set by hand. The ↑ line is what "
                   "the item lacks for the next letter. E and F always show the main problem and the next step; they "
                   "have no compact form."},
    "pt": {"title": "Selos de qualidade PINNeAPPle", "light": "sobre figura clara", "dark": "sobre figura escura",
           "compact": "compacto",
           "A": "comparado com uma referência independente", "B": "física, baseline ou dados não vistos passam",
           "C": "verificações passam, cartão em parte respondido", "D": "executa, nada falha, ainda sem evidência",
           "E": "a verificação que falha", "F": "a falha, a divergência ou a física que falha",
           "hint_A": "A+: o que a reforçaria", "hint_B": "A: o que falta para A",
           "hint_C": "B: o que falta para B", "hint_D": "C: o que falta para C",
           "hint_E": "D: o próximo passo segundo a curadoria", "hint_F": "E: o próximo passo segundo a curadoria",
           "note": "O nível vem da curadoria do Lab (lab/curation.json), nunca definido à mão. A linha ↑ é o que "
                   "falta para o nível seguinte. E e F mostram sempre o problema principal e o próximo passo; não "
                   "têm forma compacta."},
}
_LEGEND_KIND = {"A": "reference", "B": "benchmark", "C": "verification", "D": "", "E": "failing", "F": "failing"}


def legend_seals(lang: str = "en") -> list[Seal]:
    """One sample seal per tier, its reason line saying what the letter means (``LEGEND``)."""
    t = LEGEND.get(lang, LEGEND["en"])
    out = []
    for tier in SEALABLE:
        hint = t.get(f"hint_{tier}", "")
        out.append(Seal(item=f"legend {tier}", tier=tier, title="", reason=t[tier], reason_kind=_LEGEND_KIND[tier],
                        curated="", experiment="", run_id="", hint=hint, reason_pt=t[tier], hint_pt=hint))
    return out


def render_legend(lang: str = "en", height: int = 72):
    """The six seals, A to F, each on a light and on a dark background, with the compact form where there is one:
    a PIL RGB image."""
    from PIL import Image, ImageDraw
    t = LEGEND.get(lang, LEGEND["en"])
    seals = legend_seals(lang)
    full = [render_seal(s, height=height, lang=lang, compact=False) for s in seals]
    small = [render_seal(s, height=28, lang=lang) if s.tier not in MUST_EXPLAIN else None for s in seals]
    pad, gap = 24, 16
    cw = max(im.width for im in full) + 2 * pad
    sw = max(im.width for im in small if im is not None) + 2 * pad
    title, head = 60, 104
    rows = [max(im.height, height) + gap for im in full]
    foot = 76
    W, H = 2 * cw + sw, head + sum(rows) + gap + foot
    out = Image.new("RGB", (W, H), (255, 255, 255))
    d = ImageDraw.Draw(out)
    d.rectangle((cw, title, 2 * cw, H - foot), fill=(9, 12, 20))
    d.rectangle((2 * cw, title, W, H - foot), fill=(236, 239, 244))
    d.text((pad, 42), t["title"] + " · A–F", font=_pil_font(26), fill=(15, 20, 32), anchor="ls")
    for x, key, col in ((pad, "light", (90, 96, 110)), (cw + pad, "dark", (200, 205, 215)),
                        (2 * cw + pad, "compact", (90, 96, 110))):
        d.text((x, 90), t[key], font=_pil_font(16, False), fill=col, anchor="ls")
    y = head
    for im, sm, rh in zip(full, small, rows, strict=True):
        out.paste(im, (pad, y), im)
        out.paste(im, (cw + pad, y), im)
        if sm is not None:
            out.paste(sm, (2 * cw + pad, y + (im.height - sm.height) // 2), sm)
        y += rh
    note, f = t["note"], _pil_font(14, False)
    words, line, yy = note.split(), "", H - foot + 32
    for wd in words:                                    # two lines at most
        if f.getlength(line + " " + wd) > W - 2 * pad and line:
            d.text((pad, yy), line, font=f, fill=(90, 96, 110), anchor="ls")
            line, yy = wd, yy + 20
        else:
            line = (line + " " + wd).strip()
    d.text((pad, yy), line, font=f, fill=(90, 96, 110), anchor="ls")
    return out
