"""The CFD colour scale (jet: blue = low, red = high) and a colour bar burnt into an image."""
from __future__ import annotations

from typing import Optional, Tuple

JET = [(0.0, (0, 0, 143)), (0.125, (0, 0, 255)), (0.375, (0, 255, 255)), (0.625, (255, 255, 0)), (0.875, (255, 0, 0)),
       (1.0, (128, 0, 0))]


def jet(t: float) -> Tuple[float, float, float]:
    """sRGB jet colour for t in [0, 1], as floats in [0, 1]."""
    t = min(1.0, max(0.0, float(t)))
    for (a, ca), (b, cb) in zip(JET, JET[1:]):
        if t <= b:
            f = (t - a) / (b - a)
            return tuple((ca[k] + f * (cb[k] - ca[k])) / 255 for k in range(3))
    return tuple(c / 255 for c in JET[-1][1])


def colorbar(path: str, title: str, lo: float, hi: float, nd: Optional[int] = None, lo_txt: str = "",
             hi_txt: str = "", corner: str = "bottom-right") -> str:
    """Draw a jet colour bar with 5 ticks and a title in a corner of the image at ``path`` (overwritten)."""
    from PIL import Image, ImageDraw, ImageFont
    im = Image.open(path).convert("RGB")
    W, H = im.size
    s = W / 1920
    try:
        f = ImageFont.truetype("DejaVuSans.ttf", max(10, int(22 * s)))
        fb = ImageFont.truetype("DejaVuSans-Bold.ttf", max(11, int(24 * s)))
    except OSError:
        f = fb = ImageFont.load_default()
    if nd is None:
        span = abs(hi - lo) or abs(hi) or 1.0
        nd = max(0, min(6, 2 - int(__import__("math").floor(__import__("math").log10(span)))))
    d = ImageDraw.Draw(im, "RGBA")
    bw, bh, pad = int(520 * s), int(22 * s), int(22 * s)
    box_w = max(bw, int(d.textlength(title, font=fb)))
    box_h = int(150 * s)
    x0 = W - box_w - 3 * pad if corner.endswith("right") else 3 * pad
    y0 = H - box_h - pad if corner.startswith("bottom") else 2 * pad
    d.rounded_rectangle([x0 - pad, y0 - pad, x0 + box_w + pad, y0 + box_h], radius=int(12 * s), fill=(255, 255, 255, 225))
    d.text((x0, y0), title, fill=(15, 23, 42), font=fb)
    yb = y0 + int(40 * s)
    for i in range(bw):
        c = jet(i / (bw - 1))
        d.line([(x0 + i, yb), (x0 + i, yb + bh)], fill=tuple(int(255 * v) for v in c))
    for k in range(5):
        t = f"{lo + (hi - lo) * k / 4:.{nd}f}"
        tw = d.textlength(t, font=f)
        d.text((x0 + bw * k / 4 - tw * k / 4, yb + bh + int(6 * s)), t, fill=(15, 23, 42), font=f)
    if lo_txt:
        d.text((x0, yb + bh + int(36 * s)), lo_txt, fill=(71, 85, 105), font=f)
    if hi_txt:
        d.text((x0 + bw - d.textlength(hi_txt, font=f), yb + bh + int(36 * s)), hi_txt, fill=(71, 85, 105), font=f)
    im.save(path, quality=92)
    return path
