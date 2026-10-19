"""Animated showcase panel of the Lab's best results, generated from the curation.

``panel()`` lays out a 1080 x 1350 (4:5) panel on ``#090C14``: the PINNeAPPle header, one hero tile with a slow
Ken Burns zoom, a 3 x 3 grid of Tier A tiles, a row of two Tier B tiles and the footer, and writes an MP4 (H.264,
yuv420p, faststart), a GIF (720 wide, 12 fps, palettegen) and a still PNG.

What goes in it, and every word of evidence on it, comes from ``<lab>/curation.json``:

* tiles are picked from the curation, Tier A first (hero and grid), then the strongest Tier B (the B row); an item
  curated at C to F is never shown, whatever the override list says (the seals of C to F exist, for stamping the
  item's own figures, but the panel shows validated results only);
* an override list (``picks``) chooses items and which of their figures to show, in order; an item lands in the
  grid only if the curation gave it A, in the B row if it gave B, so a re-curation moves it;
* each tile's green "✓" (A) or amber "◐" (B) line is the item's curation headline, the passing check it was
  compared with (``pinneapple_lab.curation.headline``); a tile without one is not shown;
* each tile carries the item's seal (``pinneapple_lab.seal``), so the tier on the tile is the curation's tier.

Animated figures (GIF, MP4) play in their tiles, resampled to run once over the loop (8 s at 15 fps); static ones
stay still. Tiles are cropped only at empty rows and columns of the figure, so a colour bar or an axis label is
either whole or left out, never cut in half; renders (photographs, no text) may be cropped to fill.

    from pinneapple_lab.showcase import panel
    panel(lang="en", out_dir="docs/assets/showcase")              # showcase_en.mp4 / .gif / .png
    python -m pinneapple_lab showcase --lang en,pt --out docs/assets/showcase
"""
from __future__ import annotations

import json
import os
import subprocess
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .seal import (
    BRAND_YELLOW,
    TIER_COLOURS,
    VALIDATED,
    Seal,
    SealError,
    draw_text,
    load_curation,
    render_seal,
    run_dir,
    seal_for,
)

__all__ = ["DEFAULT_PICKS", "LABELS", "Tile", "choose_tiles", "fit_crop", "panel"]

W, H = 1080, 1350
BG = (9, 12, 20)                          # #090C14
MARGIN = 40
GAP = 14

LABELS = {
    "en": {"tagline": "Physics AI you can check",
           "subtitle": "Results from the PINNeAPPle Lab, graded by its curation: tier and evidence on every tile",
           "tier_a": "TIER A · physics ✓ · independent reference ✓",
           "tier_b": "TIER B · physics ✓ · benchmark ✓ · independent reference still missing",
           "footer1": "Every result goes through 6 questions: data, model, physics, benchmark, uncertainty, decision",
           "footer2": "Open source (Apache-2.0) · pip install pinneapple · github.com/PINNeAPPle-Labs/PINNeAPPle"},
    "pt": {"tagline": "IA para física que você pode verificar",
           "subtitle": "Resultados do PINNeAPPle Lab, classificados pela curadoria: nível e evidência em cada quadro",
           "tier_a": "TIER A · física ✓ · referência independente ✓",
           "tier_b": "TIER B · física ✓ · benchmark ✓ · referência independente ainda em falta",
           "footer1": "Todo resultado passa por 6 perguntas: dados, modelo, física, benchmark, incerteza, decisão",
           "footer2": "Código aberto (Apache-2.0) · pip install pinneapple · github.com/PINNeAPPle-Labs/PINNeAPPle"},
}

# Display names (the evidence line is never taken from here: it is the curation headline)
TITLES = {
    "repo_results/ahmed_body": ("Ahmed body, OpenFOAM", "Corpo de Ahmed, OpenFOAM"),
    "particle_suspension": ("Stirred tank, particles", "Tanque agitado, partículas"),
    "calculix_case/l_bracket": ("L-bracket, CalculiX", "Suporte em L, CalculiX"),
    "calculix_case/plate_hole": ("Plate with a hole", "Placa com furo"),
    "solid_fem/simply_supported": ("Simply supported beam", "Viga bi-apoiada"),
    "solid_fem/torsion": ("Square bar in torsion", "Barra quadrada em torção"),
    "solid_fem/cantilever": ("Cantilever beam", "Viga em balanço"),
    "bar_wear/tungsten carbide": ("Bar-end wear, WC", "Desgaste de barra, WC"),
    "pipe_flow/bend_90": ("90° pipe bend", "Curva de tubo 90°"),
    "pipe_flow/laminar_pipe": ("Laminar pipe flow", "Escoamento laminar em tubo"),
    "pipe_flow/kenics_mixer": ("Kenics static mixer", "Misturador estático Kenics"),
    "rom_study/cylinder_wake": ("Cylinder wake, ROM", "Esteira de cilindro, ROM"),
    "repo_results/prius_pmsm": ("Prius motor, magnetic field", "Motor do Prius, campo magnético"),
    "repo_results/em_verification": ("Electromagnetics, FDTD", "Eletromagnetismo, FDTD"),
    "repo_results/airfoil_handoff": ("NACA 2412, OpenFOAM", "NACA 2412, OpenFOAM"),
    "repo_results/airliner_cfd": ("A320-class airliner, CFD", "Avião classe A320, CFD"),
    "repo_results/lunar_rover": ("Lunar rover, terramechanics", "Rover lunar, terramecânica"),
}

# The panel of 2026-10 (hand-picked, then made a function): (item, figure substring, framing). Tiers still come from
# the curation; the e-machine field is always included wherever its tier puts it. ``focus`` is the subject's box in
# figure fractions (x0, y0, x1, y1), or several boxes laid side by side, drawn so its border is background (the margin
# repeats it); axes, legends and colour bars outside it are left out whole; ``fill`` crops the tile from the box's centre instead of showing all
# of it. ``rotate`` turns a tall render a quarter turn to fit a wide tile.
DEFAULT_PICKS = [
    ("repo_results/ahmed_body", "streamlines", {}),                                   # hero
    ("repo_results/ahmed_body", "cp", {"focus": (0.10, 0.14, 0.80, 0.815), "fill": True}),   # bar out
    ("repo_results/prius_pmsm", "pmsm_field",                                        # stator and rotor
     {"focus": (0.28, 0.07, 0.81, 0.89), "fill": True}),
    ("particle_suspension", ".gif", {"focus": (0.10, 0.17, 0.58, 0.90)}),             # the 3-D render, not the plots
    ("calculix_case/l_bracket", "render", {"focus": (0.24, 0.06, 0.72, 0.83)}),
    ("calculix_case/plate_hole", "render", {"focus": (0.08, 0.25, 0.88, 0.80)}),
    ("solid_fem/simply_supported", ".gif", {"focus": (0.26, 0.23, 0.88, 0.66)}),       # the deforming body
    ("solid_fem/torsion", ".gif", {"focus": (0.30, 0.26, 0.86, 0.74)}),
    ("bar_wear/tungsten carbide", "worn_end_render", {"focus": (0.22, 0.10, 0.80, 0.625)}),
    ("pipe_flow/bend_90", "streamlines", {"focus": (0.25, 0.10, 0.66, 0.97), "rotate": True}),
    ("repo_results/ahmed_body", "mid_plane", {}),
    ("rom_study/cylinder_wake", "wake_truth_vs_rom",                                  # both flow panels, no titles
     {"focus": [(0.26, 0.095, 0.74, 0.475), (0.26, 0.585, 0.74, 0.965)], "fill": True}),
    ("pipe_flow/kenics_mixer", "streamlines", {"focus": (0.03, 0.25, 0.88, 0.65)}),
]
_SKIP = ("loss", "history", "card", "convergence", "mesh")


@dataclass
class Tile:
    item: str
    tier: str
    figure: str
    title: str
    evidence: str
    evidence_kind: str
    seal: Seal = field(repr=False)
    frame: dict = field(default_factory=dict)     # framing: focus box(es), fill, rotate
    lang: str = "en"


# ---------------------------------------------------------------------------------------------- picking
def _figures(seal: Seal, cur) -> list[str]:
    d = os.path.join(run_dir(seal, cur), "figures")
    if not os.path.isdir(d):
        return []
    fs = [os.path.join(d, f) for f in sorted(os.listdir(d))
          if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".mp4", ".webm"))]
    return [f for f in fs if not any(s in os.path.basename(f).lower() for s in _SKIP)] or fs


def _best_figure(figs: list[str], want: str = "") -> str | None:
    if want:
        hit = [f for f in figs if want.lower() in os.path.basename(f).lower()]
        if hit:
            return hit[0]
    def order(f):
        b = os.path.basename(f).lower()
        return (not b.endswith((".gif", ".mp4", ".webm")),                       # movies first,
                not any(k in b for k in ("render", "beauty", "streamline")),     # then renders,
                not b.endswith((".jpg", ".jpeg")))                               # then photos, then plots

    return sorted(figs, key=order)[0] if figs else None


def _title(item: str, rec: dict, lang: str) -> str:
    t = TITLES.get(item)
    if t:
        return t[1] if lang == "pt" else t[0]
    return str(rec.get("title") or item).replace("_", " ").strip().capitalize()


def evidence_line(rec: dict, lang: str = "en", short: bool = False) -> str:
    """The tile's evidence line: the curation headline's display label in ``lang`` (``label_pt`` for Portuguese),
    its short form for a small tile when there is one, else the English label, else the raw check line."""
    hl = rec.get("headline") or {}
    pt = lang == "pt"
    keys = (["label_short_pt"] if short and pt else []) + (["label_short"] if short and not pt else []) + \
        (["label_pt"] if pt else []) + ["label", "text"]
    return next((hl[k] for k in keys if hl.get(k)), "")


def choose_tiles(lab: Any = None, picks=None, lang: str = "en", n_grid: int = 9, n_b: int = 2) -> dict[str, Any]:
    """Hero, grid (Tier A) and B row from the curation. ``picks``: [(item, figure substring[, framing])], tried
    first in order; the rest is filled automatically (A by best score, one tile per experiment first; B by best
    score)."""
    cur = load_curation(lab)
    recs = cur.get("items", {})
    picks = DEFAULT_PICKS if picks is None else picks
    used: set[tuple[str, str]] = set()

    def make(item, want="", frame=None):
        try:
            s = seal_for(item, cur, VALIDATED)
        except (KeyError, SealError):
            return None                                     # unknown, or C to F: never shown
        if not s.reason:
            return None                                     # no evidence line, no tile
        f = _best_figure([x for x in _figures(s, cur) if (item, x) not in used], want)
        if f is None:
            return None
        used.add((item, f))
        return Tile(item, s.tier, f, _title(item, recs[item], lang), evidence_line(recs[item], lang), s.reason_kind, s,
                    dict(frame or {}))

    a_tiles, b_tiles = [], []
    for pick in picks:
        item, want, frame = (tuple(pick) + ({},))[:3]
        t = make(item, want, frame)
        if t is not None:
            (a_tiles if t.tier == "A" else b_tiles).append(t)
    score = lambda kv: -(kv[1].get("best_score") or 0)  # noqa: E731
    ranked = sorted(recs.items(), key=score)
    seen_exp = {recs[t.item].get("experiment") for t in a_tiles}
    for second_pass in (False, True):                       # one tile per experiment first, then any
        for item, rec in ranked:
            if len(a_tiles) >= n_grid + 1:
                break
            if rec.get("tier") != "A" or any(t.item == item for t in a_tiles):
                continue
            if not second_pass and rec.get("experiment") in seen_exp:
                continue
            t = make(item)
            if t is not None:
                a_tiles.append(t)
                seen_exp.add(rec.get("experiment"))
    for item, rec in ranked:
        if len(b_tiles) >= n_b:
            break
        if rec.get("tier") == "B" and not any(t.item == item for t in b_tiles):
            t = make(item)
            if t is not None:
                b_tiles.append(t)
    hero = next((t for t in a_tiles if t.figure.lower().endswith((".jpg", ".jpeg"))), a_tiles[0] if a_tiles else None)
    grid = [t for t in a_tiles if t is not hero][:n_grid]
    for t in grid + b_tiles:                               # small tiles: the short form of the line, when there is one
        t.evidence = evidence_line(recs[t.item], lang, short=True)
    for t in a_tiles + b_tiles:
        t.lang = lang
    return {"hero": hero, "grid": grid, "b": b_tiles[:n_b], "curated": cur.get("generated", "")}


# ---------------------------------------------------------------------------------------------- media
class _Source:
    """A figure as frames: one frame for an image, all frames (and their times) for a GIF or a video."""

    def __init__(self, path: str, rotate: bool = False, focus=None):
        from PIL import Image, ImageSequence
        self.path = path
        ext = os.path.splitext(path)[1].lower()
        frames, times = [], []
        if ext == ".gif":
            g = Image.open(path)
            t = 0.0
            for fr in ImageSequence.Iterator(g):
                frames.append(np.asarray(fr.convert("RGB")))
                times.append(t)
                t += (fr.info.get("duration") or g.info.get("duration") or 100) / 1000.0
            self.duration = t
        elif ext in (".mp4", ".webm", ".mov"):
            import imageio_ffmpeg
            rd = imageio_ffmpeg.read_frames(path)
            meta = rd.__next__()
            w, h = meta["size"]
            fps = meta.get("fps") or 25
            for i, fr in enumerate(rd):
                frames.append(np.frombuffer(fr, np.uint8).reshape(h, w, 3).copy())
                times.append(i / fps)
            self.duration = len(frames) / fps
        else:
            frames.append(np.asarray(Image.open(path).convert("RGB")))
            times.append(0.0)
            self.duration = 0.0
        if focus:
            frames = [_focus(f, focus) for f in frames]
        if rotate:
            frames = [np.ascontiguousarray(np.rot90(f)) for f in frames]
        self.frames, self.times = frames, np.asarray(times)
        self.animated = len(frames) > 1

    def at(self, phase: float) -> np.ndarray:
        """The frame at ``phase`` in [0, 1) of the loop (the source plays exactly once per loop)."""
        if not self.animated:
            return self.frames[0]
        k = int(np.searchsorted(self.times, phase * self.duration, side="right") - 1)
        return self.frames[max(0, min(k, len(self.frames) - 1))]


def _focus(a: np.ndarray, focus) -> np.ndarray:
    """The subject of a frame: one box in figure fractions, or several laid side by side at a common height."""
    from PIL import Image
    boxes = [focus] if isinstance(focus[0], (int, float)) else list(focus)
    h, w = a.shape[:2]
    parts = [a[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] for x0, y0, x1, y1 in boxes]
    if len(parts) == 1:
        return np.ascontiguousarray(parts[0])
    hh = min(p.shape[0] for p in parts)
    parts = [np.asarray(Image.fromarray(p).resize((max(1, round(p.shape[1] * hh / p.shape[0])), hh), Image.LANCZOS))
             for p in parts]
    return np.concatenate(parts, axis=1)


def _cover(w: int, h: int, aspect: float) -> tuple[int, int, int, int]:
    """The centred window of a ``w`` x ``h`` image with width / height ``aspect``."""
    if w / h > aspect:
        nw = int(round(h * aspect))
        return (w - nw) // 2, 0, (w - nw) // 2 + nw, h
    nh = int(round(w / aspect))
    return 0, (h - nh) // 2, w, (h - nh) // 2 + nh


def _bg_colour(a: np.ndarray) -> np.ndarray:
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]]).astype(float)
    return np.median(border, axis=0)


def _segments(active: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Runs of active rows / columns separated by at least ``min_gap`` empty ones."""
    idx = np.flatnonzero(active)
    if idx.size == 0:
        return []
    segs, start, prev = [], idx[0], idx[0]
    for i in idx[1:]:
        if i - prev > min_gap:
            segs.append((start, prev + 1))
            start = i
        prev = i
    segs.append((start, prev + 1))
    return segs


def fit_crop(img: np.ndarray, aspect: float, *, photo: bool | None = None) -> tuple[tuple[int, int, int, int], bool]:
    """The window (x0, y0, x1, y1) of ``img`` to show in a tile of width / height ``aspect`` and whether it must be
    letterboxed. Plots are trimmed to their content and cut only at empty rows / columns (a colour bar or a label
    is kept whole or dropped whole, the largest block is always kept); photographs and renders (no empty bands,
    ``photo``) are cropped to fill around their centre."""
    a = np.asarray(img)
    h, w = a.shape[:2]
    bg = _bg_colour(a)
    mask = np.abs(a[..., :3].astype(float) - bg).max(axis=2) > 18
    ys, xs = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    if ys.size == 0:
        return (0, 0, w, h), True
    y0, y1, x0, x1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1
    content = mask[y0:y1, x0:x1]
    if photo is None:
        photo = content.mean() > 0.55 and not _segments(content.mean(axis=0) > 0.003, 3)[1:]
    if photo:
        cw, ch = x1 - x0, y1 - y0
        if cw / ch > aspect:
            nw = int(ch * aspect)
            cx = x0 + (cw - nw) // 2
            return (cx, y0, cx + nw, y1), False
        nh = int(cw / aspect)
        cy = y0 + (ch - nh) // 2
        return (x0, cy, x1, cy + nh), False
    best = None
    colsegs = _segments(content.mean(axis=0) > 0.003, max(4, int(0.012 * w)))
    rowsegs = _segments(content.mean(axis=1) > 0.003, max(4, int(0.012 * h)))
    big_c = max(range(len(colsegs)), key=lambda i: colsegs[i][1] - colsegs[i][0])
    big_r = max(range(len(rowsegs)), key=lambda i: rowsegs[i][1] - rowsegs[i][0])
    for i in range(big_c + 1):
        for j in range(big_c, len(colsegs)):
            for k in range(big_r + 1):
                for m in range(big_r, len(rowsegs)):
                    cx0, cx1 = colsegs[i][0], colsegs[j][1]
                    cy0, cy1 = rowsegs[k][0], rowsegs[m][1]
                    kept = content[cy0:cy1, cx0:cx1].sum() / max(content.sum(), 1)
                    ar = (cx1 - cx0) / max(cy1 - cy0, 1)
                    fill = min(ar, aspect) / max(ar, aspect)
                    score = fill + 1.5 * kept                    # keep the content, then fill the tile
                    if best is None or score > best[0]:
                        best = (score, (x0 + cx0, y0 + cy0, x0 + cx1, y0 + cy1))
    bx0, by0, bx1, by1 = best[1]
    pad = int(0.02 * max(bx1 - bx0, by1 - by0))
    return (max(0, bx0 - pad), max(0, by0 - pad), min(w, bx1 + pad), min(h, by1 + pad)), True


def _font(size: float, weight: str = "Bold"):
    from PIL import ImageFont
    for p in (f"/usr/share/fonts/opentype/inter/Inter-{weight}.otf",):
        if os.path.exists(p):
            return ImageFont.truetype(p, int(size))
    from matplotlib import font_manager as fm
    return ImageFont.truetype(fm.findfont(fm.FontProperties(family="DejaVu Sans",
                                                            weight="bold" if "Bold" in weight else "normal")),
                              int(size))


def _round_mask(w: int, h: int, r: int):
    from PIL import Image, ImageDraw
    m = Image.new("L", (w * 4, h * 4), 0)
    ImageDraw.Draw(m).rounded_rectangle((0, 0, w * 4 - 1, h * 4 - 1), radius=r * 4, fill=255)
    return m.resize((w, h), Image.LANCZOS)


def _fit_text(text: str, font, width: float) -> str:
    if font.getlength(text) <= width:
        return text
    while text and font.getlength(text + "\u2026") > width:
        text = text[:-1]
    return text.rstrip(" ,;\u00b7") + "\u2026"


def _fit_line(text: str, size: float, width: float, weight: str = "SemiBold", min_size: float = 10):
    """(text, font) that fits ``width`` on one line without cutting a word: the font shrinks down to ``min_size``,
    then trailing " \u00b7 " clauses are dropped whole."""
    parts = text.split(" \u00b7 ")
    while True:
        t = " \u00b7 ".join(parts)
        sz = size
        while sz >= min_size:
            f = _font(sz, weight)
            if f.getlength(t) <= width:
                return t, f
            sz -= 0.5
        if len(parts) == 1:
            return t, _font(min_size, weight)
        parts = parts[:-1]


class _TileRenderer:
    def __init__(self, tile: Tile, w: int, h: int, *, hero: bool = False, rotate: bool = False):
        from PIL import Image, ImageDraw
        self.tile, self.w, self.h, self.hero = tile, w, h, hero
        self.src = _Source(tile.figure, rotate=rotate or bool(tile.frame.get("rotate")), focus=tile.frame.get("focus"))
        f0 = self.src.frames[0]
        # only the hero is cropped to fill (its Ken Burns window must stay inside the image); every other tile shows
        # the whole content, cut at empty bands only, and fills the margin by extending the edge pixels
        if tile.frame.get("focus"):           # framed by hand: the whole subject as large as it fits, or filling
            fill = bool(tile.frame.get("fill"))   # the tile when the box already holds only the subject
            self.box = _cover(f0.shape[1], f0.shape[0], w / h) if fill else (0, 0, f0.shape[1], f0.shape[0])
            self.letterbox = not fill
        else:
            self.box, self.letterbox = fit_crop(f0, w / h, photo=True if hero else False)
        self.bg = tuple(int(v) for v in _bg_colour(f0))
        self.mask = _round_mask(w, h, 16 if hero else 12)
        self.lift = 0.0 if hero else 0.14 if tile.frame.get("focus") else 0.22   # keep it above the title band
        # overlay: bottom gradient band, title, evidence line, seal
        ov = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        band = int(h * (0.48 if not hero else 0.34))
        g = np.zeros((band, w, 4), np.uint8)
        g[..., :3] = (6, 8, 14)
        g[..., 3] = (np.clip(np.linspace(0, 1.25, band), 0, 1) ** 1.4 * 238).astype(np.uint8)[:, None]
        ov.alpha_composite(Image.fromarray(g, "RGBA"), (0, h - band))
        d = ImageDraw.Draw(ov)
        ts, es = (30, 19) if hero else (19 if w > 400 else 17, 13 if w > 400 else 12)
        ft, fe = _font(ts, "Bold"), _font(es, "SemiBold")
        col = TIER_COLOURS[tile.tier]
        mark = "✓ " if tile.tier == "A" else "◐ "
        pad = 14 if hero else 10
        title, ft = _fit_line(tile.title, ts, w - 2 * pad, "Bold", min_size=0.75 * ts)
        d.text((pad, h - pad - es - 6), title, font=ft, fill=(255, 255, 255, 255),
               anchor="ls")
        line, fe = _fit_line(mark + tile.evidence, es, w - 2 * pad)
        draw_text(d, (pad, h - pad), line, fe, col)
        if hero:                                            # the full seal on the hero only
            seal = render_seal(tile.seal, height=58, compact=False, reason=False, lang=tile.lang)
        else:                                               # a small compact seal, top right, see-through
            seal = render_seal(tile.seal, height=19, compact=True, opacity=0.85)
        ov.alpha_composite(seal, (w - seal.width - pad, pad))
        self.overlay = ov
        self._static = None

    def frame(self, phase: float, t_loop: float):
        if self._static is not None and not self.src.animated and not self.hero:
            return self._static
        im = self._base(phase, t_loop).convert("RGBA")
        im.alpha_composite(self.overlay)
        im.putalpha(self.mask)
        if not self.src.animated and not self.hero:
            self._static = im
        return im

    def _base(self, phase: float, t_loop: float):
        from PIL import Image
        a = self.src.at(phase)
        x0, y0, x1, y1 = self.box
        if self.hero:                                       # Ken Burns: zoom 1 -> 1.12 and a slow pan, inside the image
            z = 1 + 0.12 * (0.5 - 0.5 * np.cos(2 * np.pi * t_loop))
            cw, ch = (x1 - x0) / z, (y1 - y0) / z
            fx = 0.5 + 0.35 * np.sin(2 * np.pi * t_loop)
            fy = 0.5 + 0.25 * np.sin(2 * np.pi * t_loop + 1.0)
            cx0 = x0 + fx * ((x1 - x0) - cw)
            cy0 = y0 + fy * ((y1 - y0) - ch)
            im = Image.fromarray(a).resize((self.w, self.h), Image.BICUBIC, box=(cx0, cy0, cx0 + cw, cy0 + ch))
        else:
            crop = Image.fromarray(a).crop(self.box)
            if self.letterbox:
                s = min(self.w / crop.width, (1 - self.lift) * self.h / crop.height)
                inner = np.asarray(crop.resize((max(1, int(crop.width * s)), max(1, int(crop.height * s))),
                                               Image.LANCZOS))
                dx, dy = self.w - inner.shape[1], self.h - inner.shape[0]
                top = max(0, (dy - int(self.lift * self.h)) // 2)
                pad = ((top, dy - top), (dx // 2, dx - dx // 2), (0, 0))
                im = Image.fromarray(np.pad(inner, pad, mode="edge"))     # focus boxes end on background
            else:
                im = crop.resize((self.w, self.h), Image.LANCZOS)
        return im


# ---------------------------------------------------------------------------------------------- the panel
def _layout():
    inner = W - 2 * MARGIN
    gw = (inner - 2 * GAP) // 3
    bw = (inner - GAP) // 2
    y = {"hero": 172, "hero_h": 300}
    y["a_head"] = y["hero"] + y["hero_h"] + 30
    y["grid"] = y["a_head"] + 16
    gh = 176
    y["b_head"] = y["grid"] + 3 * gh + 2 * GAP + 30
    y["b"] = y["b_head"] + 16
    bh = 150
    y["footer"] = y["b"] + bh + 26
    return inner, gw, gh, bw, bh, y


def _background(labels: dict[str, str], tiles: dict[str, Any]):
    from PIL import Image, ImageDraw
    bg = Image.new("RGBA", (W, H), BG + (255,))
    d = ImageDraw.Draw(bg)
    inner, gw, gh, bw, bh, y = _layout()
    d.text((MARGIN, 88), "PINNeAPPle", font=_font(66, "ExtraBold"), fill=BRAND_YELLOW, anchor="ls")
    d.text((MARGIN, 126), labels["tagline"], font=_font(27, "SemiBold"), fill=(240, 242, 248), anchor="ls")
    d.text((MARGIN, 156), _fit_text(labels["subtitle"], _font(17, "Medium"), inner), font=_font(17, "Medium"),
           fill=(150, 158, 176), anchor="ls")
    if tiles["grid"]:
        d.text((MARGIN, y["a_head"]), labels["tier_a"], font=_font(19, "Bold"), fill=TIER_COLOURS["A"], anchor="ls")
    if tiles["b"]:
        d.text((MARGIN, y["b_head"]), _fit_text(labels["tier_b"], _font(19, "Bold"), inner), font=_font(19, "Bold"),
               fill=TIER_COLOURS["B"], anchor="ls")
    f1 = _font(15, "Medium")
    d.text((W // 2, y["footer"]), _fit_text(labels["footer1"], f1, inner), font=f1, fill=(200, 205, 218), anchor="ms")
    d.text((W // 2, y["footer"] + 24), _fit_text(labels["footer2"], f1, inner), font=f1, fill=(140, 148, 166),
           anchor="ms")
    return bg


def _render_frames(tiles: dict[str, Any], labels: dict[str, str], seconds: float, fps: int):
    inner, gw, gh, bw, bh, y = _layout()
    bg = _background(labels, tiles)
    placed = []
    if tiles["hero"] is not None:
        placed.append(((MARGIN, y["hero"]), _TileRenderer(tiles["hero"], inner, y["hero_h"], hero=True)))
    for i, t in enumerate(tiles["grid"][:9]):
        r, c = divmod(i, 3)
        placed.append(((MARGIN + c * (gw + GAP), y["grid"] + r * (gh + GAP)), _TileRenderer(t, gw, gh)))
    for i, t in enumerate(tiles["b"][:2]):
        placed.append(((MARGIN + i * (bw + GAP), y["b"]), _TileRenderer(t, bw, bh)))
    n = int(round(seconds * fps))
    for k in range(n):
        ph = k / n
        im = bg.copy()
        for (x, yy), tr in placed:
            im.alpha_composite(tr.frame(ph, ph), (x, yy))
        yield np.asarray(im.convert("RGB"))


def _ffmpeg() -> str:
    from .seal import _ffmpeg as f
    return f()


def panel(lab: Any = None, *, lang: str = "en", out_dir: str = "docs/assets/showcase", picks=None,
          seconds: float = 8.0, fps: int = 15, mp4_kbps: int = 2400, gif_width: int = 720, gif_fps: int = 12,
          name: str | None = None) -> dict[str, Any]:
    """Render the panel; returns the files written and the tiles (item, tier, figure, evidence) it used."""
    labels = LABELS[lang]
    tiles = choose_tiles(lab, picks=picks, lang=lang)
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, name or f"showcase_{lang}")
    mp4, gif, png = base + ".mp4", base + ".gif", base + ".png"
    cmd = [_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(fps), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-b:v", f"{mp4_kbps}k",
           "-maxrate", f"{int(1.3 * mp4_kbps)}k", "-bufsize", f"{2 * mp4_kbps}k", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", mp4]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    still = None
    for k, fr in enumerate(_render_frames(tiles, labels, seconds, fps)):
        if k == 0:
            still = fr
        proc.stdin.write(fr.tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed to write the MP4")
    from PIL import Image
    Image.fromarray(still).save(png, optimize=True)
    flt = (f"fps={gif_fps},scale={gif_width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=160:"
           "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle")
    subprocess.run([_ffmpeg(), "-y", "-loglevel", "error", "-i", mp4, "-filter_complex", flt, "-loop", "0", gif],
                   check=True)
    used = [{"slot": slot, "item": t.item, "tier": t.tier, "figure": os.path.basename(t.figure),
             "evidence": t.evidence} for slot, ts in (("hero", [tiles["hero"]] if tiles["hero"] else []),
                                                      ("grid", tiles["grid"]), ("b", tiles["b"])) for t in ts]
    meta = {"curated": tiles["curated"], "lang": lang, "files": {"mp4": mp4, "gif": gif, "png": png},
            "sizes_bytes": {k: os.path.getsize(v) for k, v in (("mp4", mp4), ("gif", gif), ("png", png))},
            "tiles": used}
    with open(base + ".json", "w") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    return meta
