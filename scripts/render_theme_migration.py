"""Compose the before/after contact sheet a theme-migration pull request shows (#918).

The sheet is how a reviewer sees that a batch of examples changed look and not content: one row per example, the
figure it produced before the migration on the left, after on the right, the same size and the same crop.

    python scripts/render_theme_migration.py docs/assets/house_theme/batch1_antes_depois.png \
        --pair "first_example:antes/first.png:depois/first.png" --title "Batch 1"

Each ``--pair`` is ``label:before:after``. The two runs have to come from the same script version, otherwise the sheet
compares the code and not the theme — run the example once on the branch and once on ``origin/main``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinneapple_tools.visualization.style import PALETTE  # noqa: E402

INK, MUTED, RULE = "#111111", "#52514e", "#d8d8d8"
FONT = Path(__import__("matplotlib").get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf"
FONT_BOLD = FONT.with_name("DejaVuSans-Bold.ttf")
CELL = (700, 260)         # every figure is fitted into this box, so the two columns line up
PAD, GUTTER, HEADER = 28, 40, 84


def _font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(path), size)
    except OSError:                                        # pragma: no cover - matplotlib without its fonts
        return ImageFont.load_default()


def _fit(path: Path, box: tuple[int, int]) -> Image.Image:
    """Scale the figure to fit the box, keeping its aspect: a wide figure and a square one still share a column."""
    image = Image.open(path).convert("RGB")
    scale = min(box[0] / image.width, box[1] / image.height)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS)


def build(pairs: list[tuple[str, Path, Path]], out: Path, *, title: str, note: str) -> Path:
    rows = [(_fit(before, CELL), _fit(after, CELL), label) for label, before, after in pairs]
    width = PAD * 2 + CELL[0] * 2 + GUTTER
    height = PAD + HEADER + len(rows) * (CELL[1] + PAD + 26) + (60 if note else PAD)
    sheet = Image.new("RGB", (width, height), "#ffffff")
    draw = ImageDraw.Draw(sheet)
    bold, small = _font(FONT_BOLD, 26), _font(FONT, 15)

    draw.text((PAD, PAD - 8), title, font=bold, fill=INK)
    draw.line([(PAD, PAD + HEADER - 26), (width - PAD, PAD + HEADER - 26)], fill=RULE, width=1)
    for column, caption in enumerate(("before", "after")):
        draw.text((PAD + column * (CELL[0] + GUTTER), PAD + HEADER - 22), caption, font=small, fill=MUTED)
    draw.line([(PAD + CELL[0] + GUTTER // 2, PAD + HEADER - 26), (PAD + CELL[0] + GUTTER // 2, height - PAD)],
              fill=RULE, width=1)

    y = PAD + HEADER
    for before, after, label in rows:
        for column, image in enumerate((before, after)):
            x = PAD + column * (CELL[0] + GUTTER) + (CELL[0] - image.width) // 2
            sheet.paste(image, (x, y + (CELL[1] - image.height) // 2))
        draw.text((PAD, y + CELL[1] + 4), label, font=small, fill=MUTED)
        draw.line([(PAD, y + CELL[1] + 24), (width - PAD, y + CELL[1] + 24)], fill=RULE, width=1)
        y += CELL[1] + PAD + 26
    if note:
        draw.text((PAD, height - PAD - 26), note, font=small, fill=PALETTE[1])
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out, optimize=True)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="render_theme_migration", description=__doc__.splitlines()[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--pair", action="append", required=True, metavar="LABEL:BEFORE:AFTER")
    ap.add_argument("--title", default="House theme, before and after")
    ap.add_argument("--note", default="")
    a = ap.parse_args(argv)
    pairs = []
    for spec in a.pair:
        try:
            label, before, after = spec.split(":")
        except ValueError:
            ap.error(f"--pair wants LABEL:BEFORE:AFTER, got {spec!r}")
        pairs.append((label, Path(before), Path(after)))
    print(build(pairs, a.out, title=a.title, note=a.note))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
