"""Finishing a figure after Cycles writes it: the contrast, the bloom and the vignette.

These are the last steps of a photographic pipeline, and they act on the written file, so they work the same on any
machine and on any Blender version -- the scene compositor and the AgX contrast looks moved between Blender 4 and 5,
and a figure should not depend on which one is installed. They are PNG/JPEG only: an EXR is a set of render passes to
be graded elsewhere, and must not be touched.

    render(scene, "hero.png", camera="product", look="punchy", post="bloom+vignette")

``look``: "neutral" (nothing), "low", "medium_low", "medium_high", "high", "very_high" -- an S-curve on the display
values, so the mid tones keep their shape and only the shoulders move -- or "punchy", the same curve plus a little
saturation, which is what a product shot gets before it ships.

``post``: "bloom" (the highlights bleed, as a bright lamp does on a sensor), "vignette" (the corners fall off) or
both joined by "+".
"""
from __future__ import annotations

import os

import numpy as np

#: name -> (S-curve strength, saturation gain); 0.0 of each is the picture as Cycles wrote it
LOOKS = {"neutral": (0.0, 1.0), "low": (0.15, 1.0), "medium_low": (0.25, 1.0), "medium_high": (0.40, 1.02),
         "high": (0.55, 1.04), "very_high": (0.75, 1.06), "punchy": (0.55, 1.12)}


def _load(path):
    from PIL import Image
    im = Image.open(path)
    alpha = im.getchannel("A") if im.mode in ("RGBA", "LA") else None
    return np.asarray(im.convert("RGB"), float) / 255.0, alpha


def _save(path, rgb, alpha=None):
    from PIL import Image
    arr = np.clip(rgb, 0.0, 1.0)
    im = Image.fromarray((arr * 255.0 + 0.5).astype(np.uint8), "RGB")
    if alpha is not None:
        im.putalpha(alpha)
    fmt = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
    im.save(path, format=fmt, quality=95) if fmt == "JPEG" else im.save(path, format=fmt)


def look(rgb: np.ndarray, name: str) -> np.ndarray:
    """An S-curve on the display values (and, for "punchy", a little saturation): the contrast step a colourist would
    apply before the figure ships. ``name`` as in :data:`LOOKS`; an unknown one is an error, so a typo never silently
    renders flat."""
    key = str(name or "neutral").strip().lower()
    if key not in LOOKS:
        raise ValueError(f"look={name!r}: {', '.join(LOOKS)}")
    strength, sat = LOOKS[key]
    out = rgb
    if strength:
        # smoothstep against the identity: pulls shadows down and highlights up, keeps 0 and 1 fixed
        out = out + strength * (out * out * (3.0 - 2.0 * out) - out)
    if sat != 1.0:
        lum = out @ np.array([0.2126, 0.7152, 0.0722])
        out = lum[..., None] + (out - lum[..., None]) * sat
    return np.clip(out, 0.0, 1.0)


def vignette(rgb: np.ndarray, strength: float = 0.55, softness: float = 0.55) -> np.ndarray:
    """Darken towards the corners: an elliptical falloff that leaves the middle untouched, the way a wide-open lens
    does. ``strength`` is how far the corners go down, ``softness`` how wide the transition is."""
    h, w = rgb.shape[:2]
    y, x = np.mgrid[0:h, 0:w]
    cx, cy = (w - 1) / 2.0, (h - 1) / 2.0
    r = np.sqrt(((x - cx) / cx) ** 2 + ((y - cy) / cy) ** 2) / np.sqrt(2.0)      # 0 centre, 1 corner
    ramp = np.clip((r - (1.0 - softness)) / max(softness, 1e-6), 0.0, 1.0)
    return rgb * (1.0 - strength * ramp)[..., None]


def bloom(rgb: np.ndarray, threshold: float = 0.85, radius: float = 0.02, strength: float = 0.55) -> np.ndarray:
    """The highlights bleed into their surroundings: what a bright lamp does on a sensor. ``radius`` is a fraction of
    the image's long side, so the same look holds at any resolution."""
    from PIL import Image, ImageFilter
    lum = rgb.max(axis=2, keepdims=True)
    hot = np.clip((lum - threshold) / max(1.0 - threshold, 1e-6), 0.0, 1.0)
    src = (rgb * hot * 255.0).astype(np.uint8)
    im = Image.fromarray(src, "RGB").filter(ImageFilter.GaussianBlur(radius * max(rgb.shape[:2])))
    spread = np.asarray(im, float) / 255.0
    return np.clip(rgb + strength * spread, 0.0, 1.0)


def apply(path: str, *, look_name: "str | None" = None, post: "str | None" = None) -> str:
    """Finish the PNG/JPEG at ``path`` in place: the look, then bloom, then the vignette. Returns ``path``.

    Anything else (an EXR, a file another process is still writing) is left alone: a render pass set is the input to
    a grade, not something to burn a contrast curve into."""
    if not path.lower().endswith((".png", ".jpg", ".jpeg")):
        return path
    want_look = str(look_name or "neutral").strip().lower()
    kinds = [k for k in ("bloom", "vignette") if k in str(post or "").lower()]
    if look_name is not None and want_look not in LOOKS:
        raise ValueError(f"look={look_name!r}: {', '.join(LOOKS)}")
    if post is not None and str(post).strip().lower() not in ("", "none") and not kinds:
        raise ValueError(f"post={post!r}: bloom, vignette or bloom+vignette")
    if want_look == "neutral" and not kinds:
        return path
    rgb, alpha = _load(path)
    rgb = look(rgb, want_look)
    for k in kinds:                                   # bloom before the vignette: light bleeds, then corners fall
        rgb = bloom(rgb) if k == "bloom" else vignette(rgb)
    _save(path, rgb, alpha)
    return path


__all__ = ["LOOKS", "look", "bloom", "vignette", "apply"]
