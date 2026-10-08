"""OCR for scanned pages (pages without a text layer), with Tesseract.

Datasheets are tables, and table grid lines are what OCR gets wrong: a border next to "1.27" is read as "427". So each
page is rendered at 300 dpi, deskewed, its grid lines are detected and erased before Tesseract reads it, and the
detected lines are then used to put every word back in its table cell. The result has the same shape as a page with a
text layer (text lines + table rows), so the same rules read it, and every word keeps Tesseract's confidence: values
read by OCR are marked as such, with their confidence, for the engineer to check against the scan.

Needs the ``tesseract`` program (``apt install tesseract-ocr``, or set ``TESSERACT_CMD``); page rendering uses
pypdfium2, which pdfplumber already depends on.
"""
from __future__ import annotations

import io
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

__all__ = ["ocr_available", "OcrPage", "ocr_pdf_page", "ocr_image"]

DPI = 300


def _cmd() -> Optional[str]:
    return os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")


def ocr_available() -> bool:
    return _cmd() is not None


@dataclass
class OcrPage:
    lines: List[str]                                   # free text, one string per line
    tables: List[List[List[str]]]                      # tables -> rows -> cells
    confidence: float                                  # mean word confidence, 0-100
    word_conf: Dict[str, float] = field(default_factory=dict)   # cell/line text -> lowest word confidence in it
    angle: float = 0.0                                 # deskew applied (degrees)


# --------------------------------------------------------------------------- image work (numpy)
def _deskew(img):
    """Small-angle deskew: the rotation that makes text rows sharpest (max variance of the row profile)."""
    import numpy as np
    from PIL import Image
    small = img.convert("L").resize((max(1, img.width // 3), max(1, img.height // 3)))
    best, best_angle = -1.0, 0.0
    for a10 in range(-30, 31, 2):
        a = a10 / 10
        arr = 255 - np.asarray(small.rotate(a, fillcolor=255, resample=Image.BILINEAR), dtype=np.float32)
        score = float(np.var(arr.sum(axis=1)))
        if score > best:
            best, best_angle = score, a
    if abs(best_angle) < 0.05:
        return img, 0.0
    return img.rotate(best_angle, fillcolor=255, resample=Image.BICUBIC, expand=False), best_angle


def _runs(mask, axis: int, length: int):
    """Pixels that belong to a straight run of at least ``length`` dark pixels along ``axis``."""
    import numpy as np
    m = mask.astype(np.int32)
    if axis == 1:
        m = m.T
    c = np.cumsum(np.pad(m, ((1, 0), (0, 0))), axis=0)
    full = (c[length:] - c[:-length]) == length                    # window starting at row i is all dark
    out = np.zeros_like(mask.T if axis == 1 else mask, dtype=bool)
    for k in range(length):                                          # spread back over the window
        out[k:k + full.shape[0]] |= full
    return out.T if axis == 1 else out


def _segments(mask, axis: int, min_len: int) -> List[Tuple[int, int, int]]:
    """Line segments from a run mask: (position across, start, end) along ``axis`` (1 = horizontal lines)."""
    import numpy as np
    m = mask if axis == 1 else mask.T
    rows = np.where(m.any(axis=1))[0]
    segs: List[Tuple[int, int, int]] = []
    if rows.size == 0:
        return segs
    groups, cur = [], [rows[0]]
    for r in rows[1:]:
        if r - cur[-1] <= 2:
            cur.append(r)
        else:
            groups.append(cur)
            cur = [r]
    groups.append(cur)
    for g in groups:
        band = m[g[0]:g[-1] + 1].any(axis=0)
        xs = np.where(band)[0]
        start = prev = xs[0]
        for x in list(xs[1:]) + [None]:
            if x is None or x - prev > 3:
                if prev - start >= min_len:
                    segs.append((int(round((g[0] + g[-1]) / 2)), int(start), int(prev)))
                if x is not None:
                    start = x
            if x is not None:
                prev = x
    return segs


# --------------------------------------------------------------------------- tesseract
def _tesseract(img, psm: int = 4) -> List[dict]:
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "page.png")
        img.save(path)
        out = subprocess.run([_cmd(), path, "stdout", "--psm", str(psm), "-l", os.environ.get("TESSERACT_LANG", "eng"),
                              "tsv"], capture_output=True, text=True, timeout=300)
    if out.returncode != 0:
        raise RuntimeError(f"tesseract failed: {out.stderr.strip()[:200]}")
    rows = out.stdout.splitlines()
    head = rows[0].split("\t")
    words = []
    for r in rows[1:]:
        parts = r.split("\t")
        if len(parts) != len(head):
            continue
        w = dict(zip(head, parts))
        text = w.get("text", "").strip()
        if not text or float(w.get("conf", -1)) < 0:
            continue
        left, top, width, height = (int(w[k]) for k in ("left", "top", "width", "height"))
        words.append({"text": text, "conf": float(w["conf"]), "x0": left, "x1": left + width, "y0": top,
                      "y1": top + height, "cx": left + width / 2, "cy": top + height / 2,
                      "line": (int(w["block_num"]), int(w["par_num"]), int(w["line_num"]))})
    return words


def ocr_image(img) -> OcrPage:
    """OCR one page image (PIL)."""
    import numpy as np
    img, angle = _deskew(img.convert("L"))
    arr = np.asarray(img)
    # grid lines are often thin and grey: "dark" is relative to the paper, not an absolute level
    paper = float(np.median(arr))
    dark = arr < min(215.0, max(150.0, paper - 45.0))
    h_mask = _runs(dark, 1, max(60, img.width // 25))
    v_mask = _runs(dark, 0, max(60, img.height // 30))             # longer than any letter stroke
    hsegs = _segments(h_mask, 1, img.width // 12)                   # (y, x0, x1)
    vsegs = _segments(v_mask, 0, img.height // 120)                 # (x, y0, y1)
    clean = arr.copy()
    grid = h_mask | v_mask
    if grid.any():                                                   # erase the grid (and a 2 px halo)
        g = grid.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1), (2, 0), (-2, 0), (0, 2), (0, -2)):
            g |= np.roll(np.roll(grid, dy, axis=0), dx, axis=1)
        clean[g] = 255
    from PIL import Image
    words = _tesseract(Image.fromarray(clean))
    used = set()
    tables: List[List[List[str]]] = []
    conf: Dict[str, float] = {}

    # Table rows: bands between consecutive horizontal lines that overlap in x; cells split at vertical lines.
    hsegs.sort()
    cur_table: List[List[str]] = []
    prev_band = None
    for (ya, xa0, xa1), (yb, xb0, xb1) in zip(hsegs, hsegs[1:]):
        x0, x1 = max(xa0, xb0), min(xa1, xb1)
        if x1 - x0 < img.width // 12 or yb - ya < 12:
            continue
        ymid = (ya + yb) / 2
        seps = sorted({x for x, s0, s1 in vsegs if s0 <= ymid <= s1 and x0 - 8 <= x <= x1 + 8})
        edges = [x0] + [x for x in seps if x0 + 8 < x < x1 - 8] + [x1]
        cells: List[List[dict]] = [[] for _ in range(len(edges) - 1)]
        for i, w in enumerate(words):
            if ya < w["cy"] < yb and x0 - 4 <= w["cx"] <= x1 + 4:
                k = next((j for j in range(len(edges) - 1) if w["cx"] <= edges[j + 1]), len(edges) - 2)
                cells[k].append(w)
                used.add(i)
        if not any(cells):
            continue
        row = []
        for c in cells:
            c.sort(key=lambda w: (round(w["cy"] / 25), w["x0"]))
            text = " ".join(w["text"] for w in c).strip(" |[]")
            row.append(text)
            if c:
                conf[text] = min(w["conf"] for w in c)
        same = prev_band is not None and abs(prev_band[0] - x0) < 15 and abs(prev_band[1] - x1) < 15 and \
            prev_band[2] == ya
        if not same and cur_table:
            tables.append(cur_table)
            cur_table = []
        cur_table.append(row)
        prev_band = (x0, x1, yb)
    if cur_table:
        tables.append(cur_table)

    # Free text: the remaining words, in Tesseract's line order.
    by_line: Dict[Tuple[int, int, int], List[dict]] = {}
    for i, w in enumerate(words):
        if i not in used:
            by_line.setdefault(w["line"], []).append(w)
    lines = []
    loose: List[List[str]] = []           # borderless table: consecutive lines split into cells at wide gaps
    for key in sorted(by_line, key=lambda k: min(w["y0"] for w in by_line[k])):
        ws = sorted(by_line[key], key=lambda w: w["x0"])
        text = " ".join(w["text"] for w in ws).strip(" |")
        if not text:
            continue
        lines.append(text)
        conf[text] = min(w["conf"] for w in ws)
        height = sorted(w["y1"] - w["y0"] for w in ws)[len(ws) // 2]
        cells, cur = [], [ws[0]]
        for a, b in zip(ws, ws[1:]):
            if b["x0"] - a["x1"] > 2.5 * height:
                cells.append(cur)
                cur = []
            cur.append(b)
        cells.append(cur)
        if 2 <= len(cells) <= 6:
            row = [" ".join(w["text"] for w in c).strip(" |[]") for c in cells]
            for c, t in zip(cells, row):
                conf[t] = min(w["conf"] for w in c)
            loose.append(row)
        elif loose:
            tables.append(loose)
            loose = []
    if loose:
        tables.append(loose)
    mean = sum(w["conf"] for w in words) / len(words) if words else 0.0
    # Table rows are also part of the page text (as on a page with a text layer), after the free text.
    return OcrPage(lines=lines, tables=tables, confidence=mean, word_conf=conf, angle=angle)


def ocr_pdf_page(pdf_bytes: bytes, index: int, dpi: int = DPI) -> OcrPage:
    """OCR page ``index`` (0-based) of a PDF."""
    import pypdfium2 as pdfium
    doc = pdfium.PdfDocument(io.BytesIO(pdf_bytes))
    try:
        img = doc[index].render(scale=dpi / 72).to_pil()
    finally:
        doc.close()
    return ocr_image(img)
