"""Render the right-hand panel -- "FIGURES OF THE RUN" -- from an asciinema cast.

    python examples/end_to_end/airfoil_trust/demo_panel.py session.cast panel.mp4 [--speed 1.0]

The panel's clock is the cast's own: every ``[figure] file.png  caption`` line the session printed marks the instant
the panel starts showing that figure, so nothing here depends on the wall clock and the two panes stay aligned even
when a stage took longer than expected.

The stills are drawn once per segment and joined with the concat demuxer; no frame is rendered twice.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw, ImageFont

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
MARK = re.compile(r"\[figure\]\s+(\S+)\s*(.*)")

BG = (13, 17, 23)
INK = (230, 237, 243)
MUTED = (139, 148, 158)
DIM = (88, 166, 255)
LINE = (48, 54, 61)

FONTS = ("/System/Library/Fonts/Supplemental/Arial.ttf", "/System/Library/Fonts/Helvetica.ttc",
         "/Library/Fonts/Arial.ttf")
MONO = ("/System/Library/Fonts/Menlo.ttc", "/System/Library/Fonts/Supplemental/Courier New.ttf")


def font(size: int, mono: bool = False):
    for path in (MONO if mono else FONTS):
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    return ImageFont.load_default()


def wrap(text: str, fnt, width: int) -> list[str]:
    """Break the caption at the panel width, measuring instead of counting characters."""
    words, lines, line = text.split(), [], ""
    for w in words:
        trial = f"{line} {w}".strip()
        if fnt.getlength(trial) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = w
    return lines + ([line] if line else [])


def markers(cast_path: str) -> tuple[list[tuple[float, str, str]], float]:
    """The instants at which the session announced a figure, and the instant the recording ends.

    Asciicast v3 stores the interval since the previous event, v2 stores the instant; the header says which, and the
    running sum is only taken in the first case.
    """
    events, clock, end, version = [], 0.0, 0.0, 3
    with open(cast_path, encoding="utf-8") as fh:
        for n, raw in enumerate(fh):
            raw = raw.strip()
            if not raw:
                continue
            if n == 0 and raw.startswith("{"):
                version = int(json.loads(raw).get("version", 3))
                continue
            try:
                t, _kind, data = json.loads(raw)
            except (ValueError, TypeError):
                continue
            clock = clock + float(t) if version >= 3 else float(t)
            end = max(end, clock)
            if not isinstance(data, str):
                continue
            for line in ANSI.sub("", data).splitlines():
                m = MARK.search(line)
                if m:
                    events.append((clock, m.group(1), m.group(2).strip()))
    return events, end


def still(index: int, total: int, fig_file: str, caption: str, size: tuple[int, int]) -> Image.Image:
    """One panel frame: header, the figure fitted, the caption and where it came from, centred as one block."""
    w, h = size
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    pad = 24
    d.text((pad, 36), "FIGURES OF THE RUN", font=font(15), fill=MUTED)
    d.text((w - pad, 36), f"{index}/{total}", font=font(15, mono=True), fill=MUTED, anchor="ra")
    d.line([(pad, 64), (w - pad, 64)], fill=LINE, width=1)

    top, bottom = 88, h - 46
    cap_font = font(23)
    lines = wrap(caption, cap_font, w - 2 * pad)[:4] if caption else []
    if fig_file and os.path.exists(fig_file):
        im = Image.open(fig_file).convert("RGB")
        box_w, box_h = w - 2 * pad, int((bottom - top) * 0.66)
        scale = min(box_w / im.width, box_h / im.height)
        im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.LANCZOS)
        block = im.height + 26 + 12 + len(lines) * 31
        y = top + max(0, ((bottom - top) - block) // 2)
        x = (w - im.width) // 2
        img.paste(im, (x, y))
        d.rectangle([x - 1, y - 1, x + im.width, y + im.height], outline=LINE, width=1)
        y += im.height + 26
        d.text((pad, y), os.path.basename(fig_file), font=font(14, mono=True), fill=MUTED)
        y += 12
    else:
        y = (top + bottom) // 2 - len(lines) * 15
        d.text((w // 2, y - 40), "waiting for the first figure", font=font(18), fill=MUTED, anchor="mm")
        y += 10
    for line in lines:
        d.text((pad, y), line, font=cap_font, fill=INK)
        y += 31
    d.text((pad, h - 34), f"figure {index} of {total}", font=font(14, mono=True), fill=MUTED)
    return img


def title_card(size: tuple[int, int]) -> Image.Image:
    w, h = size
    img = Image.new("RGB", size, BG)
    d = ImageDraw.Draw(img)
    d.text((24, 36), "FIGURES OF THE RUN", font=font(15), fill=MUTED)
    d.line([(24, 64), (w - 24, 64)], fill=LINE, width=1)
    d.text((24, int(h * 0.30)), "Trust beyond accuracy", font=font(38), fill=INK)
    for i, line in enumerate(("the PINNeAPPle verification pipeline,",
                              "and the samples it draws while it runs")):
        d.text((24, int(h * 0.30) + 64 + 30 * i), line, font=font(21), fill=MUTED)
    d.text((24, int(h * 0.30) + 190), "each figure appears when the stage\nthat measured it finishes",
           font=font(19), fill=DIM, spacing=8)
    return img


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("cast")
    ap.add_argument("out")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--width", type=int, default=840)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--figures", default=os.path.join(here, "results", "figures"))
    ap.add_argument("--fps", type=int, default=30)
    a = ap.parse_args()

    events, end = markers(a.cast)
    total = len(events)
    size = (a.width, a.height)
    print(f"{total} figures announced, cast ends at {end:.1f} s, panel speed {a.speed}")

    work = tempfile.mkdtemp(prefix="panel-")
    plan = [(0.0, None, "")]
    plan += [(t / a.speed, os.path.join(a.figures, name), caption) for t, name, caption in events]
    plan.append((end / a.speed + 1.0, None, ""))

    frames, durations = [], []
    for i in range(len(plan) - 1):
        start, fig, caption = plan[i]
        stop = plan[i + 1][0] - start
        if stop <= 0.05:
            continue
        path = os.path.join(work, f"{len(frames):03d}.png")
        img = title_card(size) if i == 0 else still(i, total, fig, caption, size)
        img.save(path)
        frames.append(path)
        durations.append(stop)

    listing = os.path.join(work, "list.txt")
    with open(listing, "w") as fh:
        for path, dur in zip(frames, durations, strict=True):
            fh.write(f"file '{path}'\nduration {dur:.3f}\n")
        fh.write(f"file '{frames[-1]}'\n")

    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", listing,
                    "-vf", f"fps={a.fps},scale={a.width}:{a.height},format=yuv420p",
                    "-t", f"{sum(durations):.3f}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "18", a.out], check=True)
    print(f"panel: {a.out} ({sum(durations):.1f} s, {len(frames)} segments)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
