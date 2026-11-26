"""Colour maps for animations (jet and diverging blue-white-red), colour bars burnt into images, and multi-panel
pictures and videos (a grid of renders with labels and one common colour bar).

    from pinneapple_tools.visualization.studio.compose import compose_grid, compose_grid_video
    compose_grid(["a.png", "b.png", "c.png", "d.png"], "grid.png", labels=["A320", "ATR", "Glider", "Jet"],
                 colorbar=("n_z [g]", 0.0, 2.0, "jet"))
    compose_grid_video(["a.mp4", "b.mp4"], "grid.mp4", labels=["A", "B"], ncols=2, crf=28)

Colour maps: ``"jet"`` (CFD convention, blue = low, red = high; the same stops as :mod:`.colormap` and
studio-core.js) and the diverging ``"coolwarm"`` / ``"bwr"`` / ``"rdbu"`` (ColorBrewer RdBu reversed; blue =
negative, neutral midpoint, red = positive), used with a range symmetric about zero (:func:`symmetric_range`).
"""
from __future__ import annotations

import math
import os
import shutil
import subprocess
import tempfile
from collections.abc import Sequence

import numpy as np

from .colormap import JET, MAGMA, RDBU, ticks

# stops: (position 0..1, (r, g, b) 0..255), sRGB; studio-core.js has the same tables
COOLWARM = [(0.0, (59, 76, 192)), (0.25, (141, 176, 254)), (0.5, (221, 221, 221)), (0.75, (244, 154, 123)),
            (1.0, (180, 4, 38))]                                        # Moreland's diverging map, as in matplotlib
BWR = [(0.0, (0, 0, 255)), (0.5, (255, 255, 255)), (1.0, (255, 0, 0))]
CMAPS = {"jet": JET, "coolwarm": COOLWARM, "bwr": BWR, "rdbu": RDBU, "magma": MAGMA}
DIVERGING = {"coolwarm", "bwr", "rdbu"}


def stops(cmap: str):
    if cmap not in CMAPS:
        raise ValueError(f"cmap={cmap!r}: one of {sorted(CMAPS)}")
    return CMAPS[cmap]


def cmap_color(cmap: str, t: float) -> tuple[float, float, float]:
    """sRGB colour (floats in [0, 1]) of ``cmap`` at t in [0, 1] (NaN -> 0)."""
    t = float(t)
    t = 0.0 if not np.isfinite(t) else min(1.0, max(0.0, t))
    st = stops(cmap)
    for (a, ca), (b, cb) in zip(st, st[1:], strict=False):
        if t <= b:
            f = (t - a) / ((b - a) or 1)
            return tuple((ca[k] + f * (cb[k] - ca[k])) / 255 for k in range(3))
    return tuple(c / 255 for c in st[-1][1])


def symmetric_range(values) -> tuple[float, float]:
    """(-m, m) with m the largest finite |value| (a diverging map keeps zero at its neutral midpoint)."""
    a = np.asarray(values, float)
    m = float(np.nanmax(np.abs(a))) if np.isfinite(a).any() else 1.0
    return (-(m or 1.0), m or 1.0)


def default_range(values, cmap: str = "jet") -> tuple[float, float]:
    """min..max for jet, symmetric about zero for a diverging map."""
    if cmap in DIVERGING:
        return symmetric_range(values)
    a = np.asarray(values, float)
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if hi == lo:
        d = abs(lo) * 0.01 or 1.0
        lo, hi = lo - d, hi + d
    return lo, hi


#: The typefaces an annotation is drawn with, tried in order: DejaVu is what a Linux box and matplotlib ship, Arial
#: and the system fonts are what macOS and Windows have. Pillow's own bitmap default is the last resort and has no
#: punctuation at all -- a "×" in a caption comes out as an empty box -- so it is only reached when nothing opens.
_FONTS = {
    False: ("DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc", "/System/Library/Fonts/SFNS.ttf", "arial.ttf"),
    True: ("DejaVuSans-Bold.ttf", "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
           "/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc", "arialbd.ttf"),
}


def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    for name in _FONTS[bool(bold)]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size=size)          # Pillow >= 10.1 sizes the bitmap fallback
    except TypeError:                                     # pragma: no cover - older Pillow
        return ImageFont.load_default()


def colorbar_image(width: int, title: str, lo: float, hi: float, cmap: str = "jet", scale: float = 1.0,
                   background=(255, 255, 255, 225)):
    """A colour bar (title, gradient, five ticks) as an RGBA PIL image ``width`` pixels wide."""
    from PIL import Image, ImageDraw
    fs, fb = _font(max(9, int(15 * scale))), _font(max(10, int(16 * scale)), True)
    pad, bh = int(8 * scale) + 2, max(6, int(14 * scale))
    h = pad + int(20 * scale) + bh + int(22 * scale) + pad
    im = Image.new("RGBA", (width, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle([0, 0, width - 1, h - 1], radius=int(8 * scale) + 1, fill=background)
    d.text((pad, pad), title, fill=(15, 23, 42), font=fb)
    yb, bw = pad + int(20 * scale), width - 2 * pad
    for i in range(bw):
        d.line([(pad + i, yb), (pad + i, yb + bh)], fill=tuple(int(255 * c) for c in cmap_color(cmap, i / max(1, bw - 1))))
    for k, t in enumerate(ticks(lo, hi, 5)):
        tw = d.textlength(t, font=fs)
        d.text((pad + bw * k / 4 - tw * k / 4, yb + bh + int(3 * scale)), t, fill=(15, 23, 42), font=fs)
    return im


def draw_colorbars(path: str, bars: Sequence[tuple], corner: str = "bottom-right", width_frac: float = 0.3) -> str:
    """Burn colour bars ``(title, lo, hi, cmap)`` into the image at ``path`` (overwritten), stacked from ``corner``."""
    from PIL import Image
    if not bars:
        return path
    im = Image.open(path).convert("RGBA")
    W, H = im.size
    scale = max(0.7, W / 1280)
    w = max(120, int(width_frac * W))
    m = max(4, int(10 * scale))
    y = H - m if corner.startswith("bottom") else m
    for b in bars:
        title, lo, hi = b[0], b[1], b[2]
        cb = colorbar_image(w, title, lo, hi, b[3] if len(b) > 3 else "jet", scale)
        x = W - w - m if corner.endswith("right") else m
        if corner.startswith("bottom"):
            y -= cb.height
            im.alpha_composite(cb, (x, y))
            y -= m // 2
        else:
            im.alpha_composite(cb, (x, y))
            y += cb.height + m // 2
    im.convert("RGB").save(path)
    return path


def _panel(d, box, dark: bool, radius: int):
    """The plate a bar or a caption sits on: dark glass over a bright picture, light glass over a dark one."""
    fill = (12, 16, 22, 190) if dark else (255, 255, 255, 210)
    d.rounded_rectangle(box, radius=radius, fill=fill)


def _save(im, path: str, alpha: bool) -> str:
    """Write the annotated picture back, keeping the alpha when it had one (a figure that goes on a page of another
    colour) and dropping it for a JPEG, which has nowhere to put it."""
    if path.lower().endswith((".jpg", ".jpeg")):
        im.convert("RGB").save(path, "JPEG", quality=94)
    elif alpha:
        im.save(path, "PNG")
    else:
        im.convert("RGB").save(path, "PNG")
    return path


def draw_scale_bar(path: str, px_per_metre: float, *, corner: str = "bottom-left", width_frac: float = 0.28,
                   dark: bool = False) -> str:
    """Burn a metric scale bar into the picture at ``path`` (overwritten): alternating segments, end ticks and the
    length as a number a reader can hold in mind ("250 mm", "1 m").

    ``px_per_metre`` is the image scale at the subject (``scale.px_per_metre``), so the bar is metrically right at the
    plane the camera focused on -- the convention a photograph with a scale bar uses. The bar's length is snapped to
    a round value near ``width_frac`` of the picture's width and kept inside 48 px .. 55 % of it."""
    from PIL import Image, ImageDraw

    from .scale import fmt_length, scale_bar_length

    if not (px_per_metre and px_per_metre > 0):
        raise ValueError(f"px_per_metre={px_per_metre!r}: a metric bar needs a positive image scale")
    im = Image.open(path)
    alpha = "A" in im.getbands()
    im = im.convert("RGBA")
    W, H = im.size
    k = max(0.7, W / 1280)
    f = _font(max(10, int(14 * k))), _font(max(11, int(15 * k)), True)
    target_m = width_frac * W / float(px_per_metre)
    bar_m = scale_bar_length(target_m)
    lo_px, hi_px = 48 * k, 0.55 * W
    if not (lo_px <= bar_m * px_per_metre <= hi_px):        # a decade that does not fit the frame is no measure
        ladder = []
        e0 = int(math.floor(math.log10(max(target_m, 1e-12)))) - 1
        for e in range(e0, e0 + 5):
            for m in (1.0, 2.0, 5.0):
                v = m * 10.0 ** e
                px = v * px_per_metre
                if lo_px <= px <= hi_px:
                    ladder.append((abs(px - width_frac * W), v))
        if ladder:
            bar_m = min(ladder)[1]
    bar_px = int(round(bar_m * px_per_metre))
    pad = int(10 * k) + 2
    bar_h = max(8, int(11 * k))
    label_w = max(f[1].getlength(fmt_length(bar_m)), f[0].getlength("0")) / 2.0
    pw = int(bar_px + 2 * pad + 2 * label_w)
    ph = int(pad + bar_h + int(20 * k) + pad * 0.6)
    panel = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(panel)
    _panel(d, [0, 0, pw - 1, ph - 1], dark, int(8 * k) + 1)
    ink = (242, 245, 249) if dark else (15, 23, 42)
    x0, y0 = pad + int(label_w), pad
    for i in range(2):                                       # two alternating segments: one is a "half", one is not
        seg = bar_px // 2 + (bar_px % 2 if i else 0)
        sh = ImageDraw.Draw(panel)
        sh.rectangle([x0 + i * (bar_px // 2), y0, x0 + i * (bar_px // 2) + seg - 1, y0 + bar_h - 1],
                     fill=ink if i == 0 else (ink[0], ink[1], ink[2], 40), outline=ink)
    for x in (x0, x0 + bar_px):                              # end ticks, so the ends can be read against the part
        d.line([(x, y0 - int(5 * k)), (x, y0 + bar_h + int(5 * k))], fill=ink, width=max(1, int(1.5 * k)))
    d.text((x0, y0 + bar_h + int(3 * k)), "0", fill=ink, font=f[0])
    lab = fmt_length(bar_m)
    d.text((x0 + bar_px - f[1].getlength(lab), y0 + bar_h + int(3 * k)), lab, fill=ink, font=f[1])
    x = int(12 * k)
    y = H - ph - int(12 * k) if corner.startswith("bottom") else int(12 * k)
    im.alpha_composite(panel, (x if corner.endswith("left") else W - pw - int(12 * k), y))
    return _save(im, path, alpha)


def draw_caption(path: str, lines: Sequence[str], *, corner: str = "top-left", title: str | None = None,
                 dark: bool = False) -> str:
    """Burn a block of text into the picture at ``path`` (overwritten) -- the technical caption a figure carries: the
    subject's extents, the grid square, the prop that gives the scale. One line each, ``title`` above them in bold."""
    from PIL import Image, ImageDraw

    lines = [str(v) for v in (lines or []) if str(v).strip()]
    if not lines and not title:
        return path
    im = Image.open(path)
    alpha = "A" in im.getbands()
    im = im.convert("RGBA")
    W, H = im.size
    k = max(0.7, W / 1280)
    fs, fb = _font(max(10, int(14 * k))), _font(max(11, int(15 * k)), True)
    pad = int(10 * k) + 2
    lh = int(19 * k)
    rows = ([title] if title else []) + lines
    widths = [fb.getlength(title or "")] + [fs.getlength(t) for t in lines]
    pw = int(max(widths) + 2 * pad)
    ph = int(pad + lh * len(rows) + pad * 0.6)
    panel = Image.new("RGBA", (pw, ph), (0, 0, 0, 0))
    d = ImageDraw.Draw(panel)
    _panel(d, [0, 0, pw - 1, ph - 1], dark, int(8 * k) + 1)
    ink = (242, 245, 249) if dark else (15, 23, 42)
    y = pad
    if title:
        d.text((pad, y), title, fill=ink, font=fb)
        y += lh
    for t in lines:
        d.text((pad, y), t, fill=ink, font=fs)
        y += lh
    x = int(12 * k) if corner.endswith("left") else W - pw - int(12 * k)
    y = int(12 * k) if corner.startswith("top") else H - ph - int(12 * k)
    im.alpha_composite(panel, (x, y))
    return _save(im, path, alpha)


def _label_strip(im, labels, cell, ncols, scale):
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    f = _font(max(10, int(18 * scale)), True)
    for i, lab in enumerate(labels or []):
        if not lab:
            continue
        x, y = (i % ncols) * cell[0], (i // ncols) * cell[1]
        p = int(6 * scale) + 2
        tw = d.textlength(lab, font=f)
        x = x + cell[0] - tw - 4 * p                               # top-right corner: clear of a HUD at top-left
        d.rounded_rectangle([x + p, y + p, x + 3 * p + tw, y + 2 * p + int(22 * scale)], radius=p, fill=(15, 23, 42, 185))
        d.text((x + 2 * p, y + int(1.4 * p)), lab, fill=(248, 250, 252), font=f)


def _overlay(size, cell, n, ncols, labels, colorbar, strip_h):
    """Transparent RGBA picture of the grid: labels in each cell and the colour bar strip at the bottom."""
    from PIL import Image
    W, H = size
    scale = max(0.5, cell[0] / 640)
    im = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    _label_strip(im, labels, cell, ncols, scale)
    if colorbar:
        cb = colorbar_image(min(W - 8, max(160, int(0.5 * W))), colorbar[0], colorbar[1], colorbar[2],
                            colorbar[3] if len(colorbar) > 3 else "jet", scale)
        im.alpha_composite(cb, ((W - cb.width) // 2, H - strip_h + max(0, (strip_h - cb.height) // 2)))
    return im


def _strip_height(cell_w: int, colorbar) -> int:
    if not colorbar:
        return 0
    return colorbar_image(100, "x", 0, 1, "jet", max(0.5, cell_w / 640)).height + 8


def compose_grid(images: Sequence[str], out_png: str, labels: Sequence[str] | None = None, ncols: int = 2,
                 colorbar: tuple | None = None, background=(233, 237, 242)) -> str:
    """One picture of ``images`` on a grid of ``ncols`` columns (every image scaled to the first one's size), a
    label in each cell's top-right corner and one common colour bar ``(title, lo, hi, cmap)`` under the grid."""
    from PIL import Image
    ims = [Image.open(p).convert("RGB") for p in images]
    cw, ch = ims[0].size
    nrows = -(-len(ims) // ncols)
    sh = _strip_height(cw, colorbar)
    out = Image.new("RGBA", (cw * ncols, ch * nrows + sh), (*background, 255))
    for i, im in enumerate(ims):
        out.paste(im.resize((cw, ch)) if im.size != (cw, ch) else im, ((i % ncols) * cw, (i // ncols) * ch))
    out.alpha_composite(_overlay(out.size, (cw, ch), len(ims), ncols, labels, colorbar, sh))
    out.convert("RGB").save(out_png)
    return out_png


def _probe_size(exe_probe: str, path: str) -> tuple[int, int]:
    r = subprocess.run([exe_probe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                        "-of", "csv=p=0", path], capture_output=True, text=True, check=True)
    w, h = r.stdout.strip().split(",")[:2]
    return int(w), int(h)


def compose_grid_video(mp4_paths: Sequence[str], out_mp4: str, labels: Sequence[str] | None = None, ncols: int = 2,
                       colorbar: tuple | None = None, crf: int = 28, webm: bool = False,
                       fps: float | None = None, background: str = "0xE9EDF2") -> dict:
    """A grid video of ``mp4_paths`` (ffmpeg ``xstack``; every clip scaled to the first one's size, the shorter ones
    hold their last frame), with the labels and one colour bar strip ``(title, lo, hi, cmap)`` burnt in.
    H.264 ``crf`` (yuv420p, even size, faststart); ``webm=True`` also writes a VP9 .webm next to it.
    Returns ``{"mp4": path, "webm": path | None}``."""
    exe, probe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if exe is None or probe is None:
        raise RuntimeError("compose_grid_video needs ffmpeg and ffprobe on the PATH")
    n = len(mp4_paths)
    cw, ch = _probe_size(probe, mp4_paths[0])
    cw, ch = cw - cw % 2, ch - ch % 2
    nrows = -(-n // ncols)
    sh = _strip_height(cw, colorbar)
    sh += sh % 2
    W, H = cw * ncols, ch * nrows + sh
    tmp = tempfile.mkdtemp()
    ov = os.path.join(tmp, "overlay.png")
    _overlay((W, H), (cw, ch), n, ncols, labels, colorbar, sh).save(ov)
    cmd = [exe, "-y", "-loglevel", "error"]
    for p in mp4_paths:
        cmd += ["-i", p]
    cmd += ["-i", ov]
    parts = [f"[{i}:v]scale={cw}:{ch},setsar=1,tpad=stop=-1:stop_mode=clone{',fps=' + str(fps) if fps else ''}[v{i}]"
             for i in range(n)]
    layout = "|".join(f"{(i % ncols) * cw}_{(i // ncols) * ch}" for i in range(n))
    stack = "".join(f"[v{i}]" for i in range(n))
    if n > 1:
        parts.append(f"{stack}xstack=inputs={n}:layout={layout}:fill={background}:shortest=0[g]")
    else:
        parts.append("[v0]null[g]")
    parts.append(f"[g]pad={W}:{H}:0:0:color={background}[p]")
    parts.append(f"[p][{n}:v]overlay=0:0:shortest=0,format=yuv420p[out]")
    # tpad makes every input endless: stop at the longest clip's duration
    durs = []
    for p in mp4_paths:
        r = subprocess.run([probe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p],
                           capture_output=True, text=True)
        try:
            durs.append(float(r.stdout.strip()))
        except ValueError:
            pass
    tmax = max(durs) if durs else None
    base = cmd + ["-filter_complex", ";".join(parts), "-map", "[out]"] + (["-t", f"{tmax:.3f}"] if tmax else [])
    subprocess.run(base + ["-c:v", "libx264", "-crf", str(crf), "-preset", "medium", "-pix_fmt", "yuv420p",
                           "-movflags", "+faststart", out_mp4], check=True)
    out = {"mp4": out_mp4, "webm": None}
    if webm:
        wp = os.path.splitext(out_mp4)[0] + ".webm"
        subprocess.run([exe, "-y", "-loglevel", "error", "-i", out_mp4, "-c:v", "libvpx-vp9", "-b:v", "0",
                        "-crf", str(min(63, crf + 6)), "-row-mt", "1", "-pix_fmt", "yuv420p", wp], check=True)
        out["webm"] = wp
    shutil.rmtree(tmp, ignore_errors=True)
    return out
