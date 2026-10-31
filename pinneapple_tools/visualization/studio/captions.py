"""Captions, title cards and narration for rendered stills and videos: the part of a product-style film that is
not the 3-D render itself.

    caption(path, "OpenFOAM k-omega SST, slant 25 deg, 60 m/s", sub="surface pressure coefficient")
    title_card("intro.png", (1280, 720), "Ahmed body", ["From CFD to a digital twin", "and back to the wind tunnel"])
    wav = narrate("Fifty-eight OpenFOAM runs.", "line1.wav")             # eSpeak NG (``espeak-ng`` binary)
    mux_audio("film.mp4", [(0.0, wav)], "film_voiced.mp4")                 # ffmpeg

Style (:data:`STYLE`): a near-black ground, light text and one accent colour (default the green of a dark product
stage, ``#76b900``) used for the title underline, the caption marker and the progress bar; fonts from DejaVu (always
installed with matplotlib). Every function works on PNG or JPEG files in place or to a new path and needs only
Pillow; :func:`narrate` and :func:`mux_audio` need the ``espeak-ng`` and ``ffmpeg`` binaries and say so when they
are missing.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from typing import Any

__all__ = ["STYLE", "caption", "title_card", "narrate", "mux_audio", "audio_duration"]

STYLE: dict[str, Any] = {"accent": "#76b900", "ink": "#f2f4f5", "muted": "#9aa3a8", "ground": "#0b0d0f",
                         "panel": (11, 13, 15, 200)}


def _rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return tuple(int(h[j:j + 2], 16) for j in (0, 2, 4))  # type: ignore[return-value]


def _font(size: int, bold: bool = False):
    import matplotlib
    from PIL import ImageFont
    base = os.path.join(os.path.dirname(matplotlib.__file__), "mpl-data", "fonts", "ttf")
    for cand in ((os.path.join(base, "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")),
                 "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(cand, size)
        except OSError:
            continue
    return ImageFont.load_default()


def caption(path: str, text: str, *, sub: str = "", out: str | None = None, corner: str = "top-left",
            progress: float | None = None, footer: str = "", footer_corner: str = "bottom-left",
            style: dict[str, Any] | None = None) -> str:
    """Burn a caption (``text`` bold, ``sub`` below it) with an accent marker into the image at ``path``
    (``out`` defaults to in place); ``progress`` in 0..1 draws a thin accent bar along the bottom edge and
    ``footer`` a small line of muted text above it in ``footer_corner`` (the source, a licence, a frame counter;
    bottom-left by default, away from a colour bar at the bottom right)."""
    from PIL import Image, ImageDraw
    st = {**STYLE, **(style or {})}
    im = Image.open(path).convert("RGB")
    W, H = im.size
    fs = max(10, int(round(H * 0.036)))
    fb, f, fsmall = _font(fs, True), _font(int(fs * 0.78)), _font(max(8, int(fs * 0.6)))
    d = ImageDraw.Draw(im, "RGBA")
    pad = max(6, fs // 2)
    lines = [(text, fb)] + ([(sub, f)] if sub else [])
    w = max(d.textlength(t, font=ft) for t, ft in lines)
    h = sum(int(ft.size * 1.3) for _, ft in lines)
    x0 = pad if corner.endswith("left") else W - pad * 3 - w - pad
    y0 = pad if corner.startswith("top") else H - pad * 3 - h - (int(fs * 0.9) if progress is not None else 0)
    d.rounded_rectangle([x0, y0, x0 + w + 3 * pad, y0 + h + 2 * pad], radius=pad, fill=st["panel"])
    d.rectangle([x0, y0 + pad // 2, x0 + max(3, pad // 3), y0 + h + 2 * pad - pad // 2], fill=_rgb(st["accent"]))
    y = y0 + pad
    for t, ft in lines:
        d.text((x0 + 2 * pad, y), t, fill=_rgb(st["ink"]) if ft is fb else _rgb(st["muted"]), font=ft)
        y += int(ft.size * 1.3)
    if footer:
        tw = d.textlength(footer, font=fsmall)
        fx = pad if footer_corner.endswith("left") else W - pad - tw
        d.text((fx, H - pad - fsmall.size - (max(2, H // 240) if progress is not None else 0)), footer,
               fill=_rgb(st["muted"]), font=fsmall)
    if progress is not None:
        bh = max(2, H // 240)
        d.rectangle([0, H - bh, W, H], fill=(255, 255, 255, 40))
        d.rectangle([0, H - bh, int(W * min(max(progress, 0.0), 1.0)), H], fill=_rgb(st["accent"]))
    out = out or path
    im.save(out, quality=92) if out.lower().endswith((".jpg", ".jpeg")) else im.save(out)
    return out


def title_card(out: str, size: tuple[int, int], title: str, lines: Sequence[str] = (), *, kicker: str = "",
               footer: str = "", style: dict[str, Any] | None = None) -> str:
    """A full-frame card on the dark ground: ``kicker`` (small, accent) above a large ``title`` with an accent
    rule, then ``lines`` in light text and ``footer`` in muted text at the bottom."""
    from PIL import Image, ImageDraw
    st = {**STYLE, **(style or {})}
    W, H = size
    im = Image.new("RGB", (W, H), _rgb(st["ground"]))
    d = ImageDraw.Draw(im)
    for y in range(H):                                                  # a faint vertical gradient
        g = int(4 * (1 - y / H))
        d.line([(0, y), (W, y)], fill=tuple(min(255, c + g) for c in _rgb(st["ground"])))
    fs = max(14, int(H * 0.085))
    ft, fk, fl, ff = _font(fs, True), _font(int(fs * 0.36), True), _font(int(fs * 0.42)), _font(int(fs * 0.3))
    x, y = int(W * 0.08), int(H * 0.26)
    if kicker:
        d.text((x, y), kicker.upper(), fill=_rgb(st["accent"]), font=fk)
        y += int(fk.size * 1.8)
    d.text((x, y), title, fill=_rgb(st["ink"]), font=ft)
    y += int(fs * 1.25)
    d.rectangle([x, y, x + int(W * 0.09), y + max(3, H // 180)], fill=_rgb(st["accent"]))
    y += int(fs * 0.55)
    for ln in lines:
        d.text((x, y), ln, fill=_rgb(st["ink"]), font=fl)
        y += int(fl.size * 1.45)
    if footer:
        d.text((x, H - int(H * 0.08) - ff.size), footer, fill=_rgb(st["muted"]), font=ff)
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    im.save(out)
    return out


def narrate(text: str, out_wav: str, *, voice: str = "en-us", speed_wpm: int = 165, exe: str = "espeak-ng") -> str:
    """Synthesise ``text`` to ``out_wav`` with eSpeak NG (English by default). Raises ``RuntimeError`` naming the
    missing binary when it is not installed."""
    path = shutil.which(exe)
    if path is None:
        raise RuntimeError(f"{exe} not found: install espeak-ng (apt) to narrate")
    os.makedirs(os.path.dirname(os.path.abspath(out_wav)), exist_ok=True)
    r = subprocess.run([path, "-v", voice, "-s", str(speed_wpm), "-w", out_wav, text], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{exe} failed: {r.stderr[-300:]}")
    return out_wav


def audio_duration(path: str) -> float:
    """Seconds of a WAV file (header only)."""
    import wave
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


def mux_audio(video: str, cues: Sequence[tuple[float, str]], out: str, *, gain_db: float = 0.0) -> str:
    """Lay the WAV files of ``cues`` (``(start_s, wav_path)``) on the video's timeline (silence elsewhere) and write
    ``out`` (H.264 copied, AAC audio) with ffmpeg; raises ``RuntimeError`` when ffmpeg is missing."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise RuntimeError("ffmpeg not found: install it to add a sound track")
    cmd = [exe, "-y", "-loglevel", "error", "-i", video]
    filt = []
    for k, (t0, wav) in enumerate(cues):
        cmd += ["-i", wav]
        filt.append(f"[{k + 1}:a]adelay={int(round(1000 * t0))}:all=1,volume={gain_db}dB[a{k}]")
    if cues:
        filt.append("".join(f"[a{k}]" for k in range(len(cues))) + f"amix=inputs={len(cues)}:normalize=0,apad[mix]")
        cmd += ["-filter_complex", ";".join(filt), "-map", "0:v", "-map", "[mix]", "-shortest"]
    cmd += ["-c:v", "copy", "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", out]
    subprocess.run(cmd, check=True)
    return out
