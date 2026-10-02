"""OpenRadioss input-deck (``*_0000.rad`` starter / ``*_0001.rad`` engine) editing.

Only the minimal, verifiable edits a design-of-experiments study needs:
reading and replacing the ``Thick`` field of ``/PROP/SHELL/<id>`` cards, and
staging one case directory per design point. The deck is treated as text and
edited in place by card position, so every other card is preserved byte-for-byte.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

__all__ = [
    "read_shell_thickness",
    "set_shell_thickness",
    "stage_case",
    "write_engine_anim",
]

# /PROP/SHELL card layout (OpenRadioss reference guide, fixed 20-char fields):
#   line 0: /PROP/SHELL/<id>
#   line 1: title
#   then comment lines (#...) interleaved with 3 data lines:
#     Ishell Ismstr Ish3n Idrill        (4 x 10-char)
#     hm hf hr dm dn                    (5 x 20-char)
#     N Istrain Thick Ashear Ithick Iplas   (2 x 10-char, 2 x 20-char, 2 x 10-char)
_THICK_SLICE = slice(20, 40)


def _prop_shell_data_lines(lines: List[str], prop_id: int) -> List[int]:
    head = re.compile(rf"^/PROP/SHELL/{prop_id}\s*$")
    for i, ln in enumerate(lines):
        if head.match(ln.rstrip("\n")):
            data = []
            j = i + 2  # skip header + title
            while j < len(lines) and len(data) < 3:
                s = lines[j]
                if s.startswith("/"):
                    break
                if not s.lstrip().startswith("#") and s.strip():
                    data.append(j)
                j += 1
            if len(data) != 3:
                raise ValueError(f"/PROP/SHELL/{prop_id}: expected 3 data lines, found {len(data)}")
            return data
    raise KeyError(f"/PROP/SHELL/{prop_id} not found in deck")


def read_shell_thickness(deck_text: str, prop_id: int) -> float:
    """Return the ``Thick`` value of ``/PROP/SHELL/<prop_id>``."""
    lines = deck_text.splitlines(keepends=True)
    k = _prop_shell_data_lines(lines, prop_id)[2]
    return float(lines[k][_THICK_SLICE])


def set_shell_thickness(deck_text: str, thickness_by_prop: Mapping[int, float]) -> str:
    """Return a copy of ``deck_text`` with ``Thick`` replaced for each ``/PROP/SHELL`` id.

    The value is written right-aligned in its 20-character field, exactly like the
    original deck, and the result is re-read to confirm the edit took effect.
    """
    lines = deck_text.splitlines(keepends=True)
    for pid, t in thickness_by_prop.items():
        if not t > 0:
            raise ValueError(f"thickness for /PROP/SHELL/{pid} must be > 0, got {t}")
        k = _prop_shell_data_lines(lines, pid)[2]
        ln = lines[k]
        eol = ln[len(ln.rstrip("\r\n")):]  # keep the deck's own line ending (often CRLF)
        body = ln.rstrip("\r\n").ljust(40)
        lines[k] = body[:20] + f"{t:20.6f}" + body[40:] + eol
    out = "".join(lines)
    for pid, t in thickness_by_prop.items():
        got = read_shell_thickness(out, pid)
        if abs(got - t) > 1e-5:
            raise RuntimeError(f"/PROP/SHELL/{pid}: wrote {t}, re-read {got}")
    return out


def write_engine_anim(
    path: str | Path,
    run_name: str,
    t_end: float,
    anim_dt: float,
    *,
    extra_anim: Iterable[str] = ("/ANIM/VECT/DISP",),
    header: str = "",
    tail: str = "",
) -> Path:
    """Write an engine file requesting ANIM output (readable with the free
    ``anim_to_vtk`` converter) instead of H3D."""
    body = [header.rstrip("\n")] if header else []
    body += [f"/RUN/{run_name}/0/", f"{t_end:20g}", "/VERS/2022", "/ANIM/DT",
             "#   TSTART     TFREQ", f"0.000000 {anim_dt:f}", *extra_anim]
    if tail:
        body.append(tail.rstrip("\n"))
    p = Path(path)
    p.write_text("\n".join(body) + "\n")
    return p


def stage_case(
    template_starter: str | Path,
    engine_file: str | Path,
    out_dir: str | Path,
    thickness_by_prop: Mapping[int, float],
    *,
    extra_files: Optional[Dict[str, str | Path]] = None,
) -> Path:
    """Create ``out_dir`` with a thickness-modified copy of the starter deck and
    an unmodified copy of the engine deck."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    starter = Path(template_starter)
    with open(starter, newline="") as fh:  # newline="" keeps CRLF decks byte-identical
        txt = set_shell_thickness(fh.read(), thickness_by_prop)
    with open(out / starter.name, "w", newline="") as fh:
        fh.write(txt)
    shutil.copy2(engine_file, out / Path(engine_file).name)
    for name, src in (extra_files or {}).items():
        shutil.copy2(src, out / name)
    return out
