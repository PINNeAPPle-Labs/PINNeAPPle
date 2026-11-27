"""LS-DYNA's **open, ASCII** side: the keyword deck, the ``nodout`` history and nothing that needs a licence.

LS-DYNA's result database (the ``d3plot`` family) is a proprietary binary format. Reading it here would be a claim
this repository cannot check: there is no LS-DYNA licence in it, so a d3plot reader would be tested against its own
assumption of the layout and nothing else — which is exactly the failure mode #1500 forbids. What *is* readable
without a licence, and what this module covers, is the ASCII half of the same run:

* **the keyword deck** (``*KEYWORD`` … ``*END``) — nodes, elements, ``*PART`` names, ``*SECTION_SHELL`` thickness and
  ``*CONTROL_TERMINATION``. It is a fixed-field text format, so it can be and is read exactly;
* **``nodout``** — the displacement and velocity of every node at every output time. This supplies the time axis and
  the kinematics of a run; it does not supply the element fields;
* **not covered, and named as such**: ``d3plot`` (binary, no licence to check a reader against), ``glstat`` (energies;
  its ASCII layout differs between releases and no run was available to fix the columns against) and ``elout``
  (element stress and plastic strain). A run built from this module therefore carries displacement and velocity and
  no stress — :class:`~pinneapple_data.cae.structural.StructuralRun` allows that, and ``docs/guides/structural_ingestion.md``
  states it as a table row rather than letting the object look complete.

The deck is parsed with or without commas and at the 8- and 16-character field widths the format allows. ``nodout``
columns are taken from the header labels the file prints (``x-disp``, ``y-velocity``, …) when they are there, and
from the documented order when they are not; the numbers are extracted with a pattern rather than ``str.split``,
because a wide value and its neighbour can run together (``7.64418E-03-1.24413E+00``) and a split would read five
numbers where the file has six.

    from pinneapple_data.cae.lsdyna import read_keyword_deck, read_nodout, read_lsdyna_directory
    deck = read_keyword_deck("crash.key")              # nodes, elements, parts, shell thickness
    hist = read_nodout("crash/nodout")                 # times, node ids, displacement, velocity
"""
from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .model import CellBlock, Mesh

__all__ = ["DynaDeck", "DynaHistory", "DynaFormatError", "read_keyword_deck", "read_nodout", "deck_mesh",
           "read_lsdyna_directory", "find_database_file"]

_NUMBER = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][-+]?\d+)?")

#: element kinds a keyword deck can carry and the number of nodes each one's card holds
_ELEMENT_KINDS = {
    "ELEMENT_SHELL": ("shell", 4), "ELEMENT_SOLID": ("solid", 8), "ELEMENT_SOLID_ORTHO": ("solid", 8),
    "ELEMENT_TETRAHEDRON": ("tetra", 4), "ELEMENT_BEAM": ("beam", 2), "ELEMENT_DISCRETE": ("beam", 2),
}


class DynaFormatError(ValueError):
    """A file that is not the ASCII LS-DYNA subset this module covers."""


# ----------------------------------------------------------------------------------------------------- the deck
@dataclass
class DynaDeck:
    """One keyword deck as the ingestion needs it: geometry, the parts and what each part's section says."""
    nodes: dict[int, tuple[float, float, float]] = field(default_factory=dict)
    elements: dict[str, list[tuple[int, int, tuple[int, ...]]]] = field(default_factory=dict)   # kind -> [(eid, pid, nodes)]
    part_names: dict[int, str] = field(default_factory=dict)          # pid -> title
    part_section: dict[int, int] = field(default_factory=dict)        # pid -> section id
    shell_thickness: dict[int, float] = field(default_factory=dict)   # section id -> t1
    section_kind: dict[int, str] = field(default_factory=dict)        # section id -> "shell" / "solid"
    termination_time: float | None = None
    title: str = ""

    @property
    def part_ids(self) -> list[int]:
        return sorted(set(self.part_names) | {pid for cards in self.elements.values() for _, pid, _ in cards})

    def thickness_of_part(self, pid: int) -> float | None:
        sec = self.part_section.get(pid)
        return None if sec is None else self.shell_thickness.get(sec)

    def summary(self) -> dict:
        """What was found, for the provenance of a run built from this deck."""
        return {"nodes": len(self.nodes), "parts": len(self.part_ids), "title": self.title,
                "elements": {k: len(v) for k, v in self.elements.items() if v},
                "shell_thickness": {str(k): v for k, v in sorted(self.shell_thickness.items())},
                "termination_time": self.termination_time}


def _fields(line: str) -> list[str]:
    """The fields of one card line: comma separated when it has commas, else free format, else 8/16-character columns.

    A deck is written either comma separated or space padded (``free format``); both put every field we read in its
    own token, so the tokens are the fields. The fixed-width fallback covers the card that carries one lone field
    right-aligned in its columns, and the 16-character width the ``LONG=y`` decks use.
    """
    s = line.rstrip("\n").rstrip("\r")
    if "," in s:
        return [f.strip() for f in s.split(",")]
    tokens = s.split()
    if len(tokens) >= 2 or not tokens:
        return tokens
    width = 16 if len(s) > 80 else 8
    return [s[i:i + width].strip() for i in range(0, max(len(s), width), width)]


def _number(text: str) -> float | None:
    m = _NUMBER.search(text or "")
    if m is None:
        return None
    return float(m.group(0).replace("D", "E").replace("d", "e"))


def _int(text: str) -> int | None:
    v = _number(text)
    return None if v is None else int(v)


def _blocks(text: str) -> Iterator[tuple[str, list[str]]]:
    """The deck split into ``*KEYWORD`` blocks: (keyword without the options, the lines under it)."""
    keyword, lines = None, []
    for raw in text.splitlines():
        if raw.lstrip().startswith("*"):
            if keyword is not None:
                yield keyword, lines
            keyword, lines = raw.lstrip()[1:].split(",")[0].strip().upper(), []
        elif keyword is not None:
            lines.append(raw)
    if keyword is not None:
        yield keyword, lines


def read_keyword_deck(source: str | Path) -> DynaDeck:
    """Read a ``*KEYWORD`` deck from a path or from its text."""
    text = source if isinstance(source, str) and "\n" in source else Path(source).read_text(encoding="latin-1")
    if "*KEYWORD" not in text.upper():
        raise DynaFormatError("not an LS-DYNA keyword deck: no *KEYWORD card")
    deck = DynaDeck()
    for keyword, lines in _blocks(text):
        if keyword == "TITLE":
            deck.title = lines[0].strip() if lines else ""
        elif keyword == "NODE":
            for ln in lines:
                if not ln.strip() or ln.lstrip().startswith("$"):
                    continue
                f = _fields(ln)
                nid = _int(f[0]) if f else None
                if nid is None:
                    continue
                xyz = [_number(v) for v in f[1:4]]
                if len(xyz) == 3 and all(v is not None for v in xyz):
                    deck.nodes[nid] = (xyz[0], xyz[1], xyz[2])       # type: ignore[assignment]
        elif keyword in _ELEMENT_KINDS:
            kind, nnode = _ELEMENT_KINDS[keyword]
            cards = deck.elements.setdefault(kind, [])
            for ln in lines:
                if not ln.strip() or ln.lstrip().startswith("$"):
                    continue
                f = _fields(ln)
                if len(f) <= 1:                                    # a continuation of the previous card
                    if cards:
                        eid, pid, nodes = cards[-1]
                        more = tuple(int(v) for v in (_int(s) for s in f) if v is not None and v > 0)
                        cards[-1] = (eid, pid, nodes + more)
                    continue
                eid, pid = _int(f[0]), _int(f[1])
                if eid is None or pid is None:
                    continue
                nodes = tuple(int(v) for v in (_int(s) for s in f[2:2 + nnode + 4]) if v is not None and v > 0)
                cards.append((eid, pid, nodes))
        elif keyword == "PART":
            title = ""                                  # the title is the last text line before a part's data card
            for ln in lines:
                if not ln.strip() or ln.lstrip().startswith("$"):
                    continue
                f = _fields(ln)
                pid, sec = _int(f[0]), (_int(f[1]) if len(f) > 1 else None)
                if pid is None:
                    title = ln.strip()
                    continue
                deck.part_names[pid] = title or f"part {pid}"
                if sec is not None:
                    deck.part_section[pid] = sec
                title = ""
        elif keyword in ("SECTION_SHELL", "SECTION_SOLID", "SECTION_TSHELL"):
            kind = "solid" if keyword == "SECTION_SOLID" else "shell"
            data = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith("$")]
            i = 0
            while i < len(data):
                f = _fields(data[i])
                sec = _int(f[0]) if f else None
                if sec is None:
                    i += 1
                    continue
                deck.section_kind[sec] = kind
                if kind == "shell" and i + 1 < len(data):
                    t = _number(_fields(data[i + 1])[0]) if _fields(data[i + 1]) else None
                    if t is not None and t > 0:
                        deck.shell_thickness[sec] = t
                    i += 2
                else:
                    i += 1
        elif keyword == "CONTROL_TERMINATION":
            for ln in lines:
                if not ln.strip():
                    continue
                v = _number(_fields(ln)[0]) if _fields(ln) else None
                if v is not None:
                    deck.termination_time = v
                break
    if not deck.nodes:
        raise DynaFormatError("the deck has no *NODE card: there is no geometry to build a run from")
    return deck


def deck_mesh(deck: DynaDeck, *, part_ids: Iterable[int] | None = None) -> Mesh:
    """The undeformed mesh of the deck as a ``Mesh``, with one ``node_set`` per part and the part id per cell.

    The cell order is the deck's, grouped by element kind — the same order the ``cell_data`` of every frame follows.
    Nodes are renumbered by sorted id, so ``node_ids`` maps a row back to the LS-DYNA id.
    """
    want = None if part_ids is None else set(int(p) for p in part_ids)
    node_ids = np.array(sorted(deck.nodes), dtype=np.int64)
    if node_ids.size == 0:
        raise DynaFormatError("the deck has no nodes")
    local = {int(n): k for k, n in enumerate(node_ids.tolist())}
    xyz = np.array([deck.nodes[int(n)] for n in node_ids.tolist()], dtype=np.float64)
    blocks: list[CellBlock] = []
    cell_part: list[int] = []
    for kind in sorted(deck.elements):
        cards = [c for c in deck.elements[kind] if want is None or c[1] in want]
        if not cards:
            continue
        rows = []
        for eid, _, nodes in cards:
            missing = [n for n in nodes if n not in local]
            if missing:
                raise DynaFormatError(f"element {eid} uses node {missing[0]}, which has no *NODE card")
            rows.append([local[n] for n in nodes])
        if not rows:
            continue
        if len({len(r) for r in rows}) != 1:
            raise DynaFormatError(f"the {kind} elements do not all have the same number of nodes")
        blocks.append(CellBlock(kind, np.array(rows, dtype=np.int64)))
        cell_part.extend(c[1] for c in cards)
    mesh = Mesh(points=xyz, blocks=blocks, source={"format": "LS-DYNA keyword deck", "title": deck.title})
    mesh.cell_data["part_id"] = np.array(cell_part, dtype=np.float64)
    mesh.source["node_ids"] = node_ids
    for pid in deck.part_ids:
        idx = [k for k, p in enumerate(cell_part) if p == pid]
        if idx:
            mesh.cell_sets[deck.part_names.get(pid, f"part {pid}")] = np.array(idx, dtype=np.int64)
    return mesh


# ---------------------------------------------------------------------------------------------------- nodout
@dataclass
class DynaHistory:
    """The ``nodout`` history: the times, the node ids and the nodal quantities, ``(T, n[, 3])`` each."""
    times: np.ndarray
    node_ids: np.ndarray
    fields: dict[str, np.ndarray] = field(default_factory=dict)

    def displacement(self) -> np.ndarray | None:
        """``(T, n, 3)`` displacement, when the file carries all three components."""
        return self._vector(("x_disp", "y_disp", "z_disp"))

    def velocity(self) -> np.ndarray | None:
        return self._vector(("x_vel", "y_vel", "z_vel"))

    def _vector(self, names: Sequence[str]) -> np.ndarray | None:
        if not all(n in self.fields for n in names):
            return None
        return np.stack([self.fields[n] for n in names], axis=-1)


_LABEL = re.compile(r"\b([xyz])[-_](disp|velocity|vel|accel\w*|coord\w*|coordinate)\b", re.I)
_DEFAULT_COLUMNS = ["id", "x_disp", "y_disp", "z_disp", "x_vel", "y_vel", "z_vel", "x_coord", "y_coord", "z_coord"]


def _canonical(label: str) -> str | None:
    """``x-disp`` -> ``x_disp``; ``y-velocity`` -> ``y_vel`` (the names :class:`DynaHistory` hands out)."""
    m = _LABEL.search(label)
    if m is None:
        return None
    axis, what = m.group(1).lower(), m.group(2).lower()
    what = "vel" if what.startswith("vel") else "coord" if what.startswith("coord") else what
    return f"{axis}_{what}"


def read_nodout(path: str | Path) -> DynaHistory:
    """Read ``nodout``: one block per output time, with the columns named by the file's own header."""
    p = Path(path)
    text = p.read_text(encoding="latin-1")
    times: list[float] = []
    blocks: list[dict[str, np.ndarray]] = []
    block_ids: list[np.ndarray] = []
    columns: list[str] | None = None
    rows: list[list[float]] = []

    def flush() -> None:
        nonlocal rows
        if rows and columns:
            arr = np.array(rows, dtype=np.float64)
            kept = min(arr.shape[1], len(columns))
            block_ids.append(arr[:, 0].astype(np.int64))
            blocks.append({columns[j]: arr[:, j] for j in range(1, kept)})
        rows = []

    for raw in text.splitlines():
        low = raw.lower()
        if "time value" in low or re.match(r"^\s*time\s*=", low):
            v = _number(raw.split("=", 1)[1]) if "=" in raw else None
            if v is not None:
                flush()
                times.append(v)
            continue
        labels = [_canonical(tok) for tok in raw.split()]
        labels = [lab for lab in labels if lab]
        if labels and not _NUMBER.search(raw) and ("disp" in low or "velocity" in low or "nodal" in low):
            columns = ["id"] + list(dict.fromkeys(labels))
            continue
        if columns is None:
            continue
        vals = [float(m.group(0).replace("D", "E").replace("d", "e")) for m in _NUMBER.finditer(raw)]
        if vals:
            rows.append(vals[: len(columns)])
    flush()
    if not times:
        raise DynaFormatError(f"{p.name}: no `time value` line — not an LS-DYNA nodout file")
    if not blocks:
        raise DynaFormatError(f"{p.name}: no data block under a time header")
    n = max(len(b.get("x_disp", ())) for b in blocks)
    if n == 0:
        raise DynaFormatError(f"{p.name}: the blocks carry no displacement column")
    node_ids = block_ids[0][:n] if block_ids and len(block_ids[0]) >= n else np.arange(1, n + 1, dtype=np.int64)
    fields: dict[str, np.ndarray] = {}
    for name in columns or []:
        if name == "id":
            continue
        fields[name] = np.stack([np.pad(block[name], (0, n - len(block[name])), constant_values=np.nan)
                                 if name in block else np.full(n, np.nan) for block in blocks])
    return DynaHistory(times=np.array(times, dtype=np.float64), node_ids=node_ids, fields=fields)


def find_database_file(directory: str | Path, *names: str) -> Path | None:
    """The first of ``names`` that exists in ``directory``, trying the usual suffixes too (``nodout``/``nodout.gz``)."""
    root = Path(directory)
    for name in names:
        for cand in (name, f"{name}.asc", f"{name}.txt"):
            p = root / cand
            if p.is_file():
                return p
    return None


@dataclass
class DynaAsciiRun:
    """What the ASCII half of one LS-DYNA run gives: the deck, the history, and what was *not* readable."""
    deck: DynaDeck
    history: DynaHistory
    directory: str = ""
    missing: list[str] = field(default_factory=list)     # the files that were not there
    unsupported: list[str] = field(default_factory=list)  # the quantities this route cannot carry


def read_lsdyna_directory(directory: str | Path, *, deck: str | Path | None = None) -> DynaAsciiRun:
    """Read the ASCII half of a run directory: the keyword deck and ``nodout``."""
    root = Path(directory)
    if deck is None:
        cands = sorted(p for p in root.glob("*") if p.suffix.lower() in (".key", ".k", ".dyn", ".kfile"))
        if not cands:
            raise DynaFormatError(f"{root}: no keyword deck (*.key/*.k/*.dyn) — the deck carries the geometry that "
                                  "nodout does not")
        deck = cands[0]
    nod = find_database_file(root, "nodout")
    if nod is None:
        raise DynaFormatError(f"{root}: no `nodout` file — it is the only ASCII output that carries the time axis")
    return DynaAsciiRun(deck=read_keyword_deck(deck), history=read_nodout(nod), directory=str(root),
                        unsupported=["d3plot (binary, no licence to check a reader against)",
                                     "glstat (energies; ASCII layout differs per release and no run was available)",
                                     "elout (element stress and plastic strain)"])
