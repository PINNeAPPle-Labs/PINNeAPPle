"""LDraw interchange: write and read .ldr / .mpd files, and load real part shapes from the LDraw parts library.

LDraw (ldraw.org) is the open file format and parts library for interlocking-brick CAD; files written here open in
LDraw-compatible editors and renderers. Units are LDraw units (1 LDU = 0.4 mm) and the LDraw frame has -Y up:
a point (x, y, z) mm of this package is (x, -z, y) / 0.4 in LDraw. A rectangular brick or plate has its origin at
the centre of its top face (where the studs start) and its long side along LDraw X.

The parts library is distributed under CC BY 2.0 (LDraw.org Parts Library, attribution required). It is not
copied into this repository: :func:`fetch_parts_library` downloads ``complete.zip`` at run time into a cache
folder, and anything rendered from it must credit "LDraw.org Parts Library, CC BY 2.0".
"""
from __future__ import annotations

import os
import zipfile
from dataclasses import dataclass, field

import numpy as np

from .assembly import Assembly, Ground
from .parts import BY_LDRAW, LDU_MM, PITCH_MM, PLATE_MM, colour

LIBRARY_URLS = ("https://library.ldraw.org/library/updates/complete.zip",
                "https://www.ldraw.org/library/updates/complete.zip")
LIBRARY_CREDIT = "LDraw.org Parts Library (CC BY 2.0), https://www.ldraw.org"

# ours (x, y, z) -> LDraw (X, Y, Z) = (x, -z, y): a proper rotation
_TO_LDRAW = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
_ROT = {0: np.eye(3), 1: np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])}   # yaw about LDraw Y


def _fmt(v: float) -> str:
    s = f"{v:.4f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def _line(b) -> str:
    cx = (b.x + b.sx / 2) * PITCH_MM
    cy = (b.y + b.sy / 2) * PITCH_MM
    top = b.top * PLATE_MM
    X, Y, Z = _TO_LDRAW @ np.array([cx, cy, top]) / LDU_MM
    R = _ROT[b.rot % 2]
    if not b.part.ldraw:
        raise ValueError(f"part {b.part.name} has no LDraw equivalent")
    return f"1 {colour(b.colour).ldraw} {_fmt(X)} {_fmt(Y)} {_fmt(Z)} " + " ".join(_fmt(v) for v in R.ravel()) + \
        f" {b.part.ldraw}"


def to_ldr(asm: Assembly, name: str = "model.ldr", *, steps: bool = True, author: str = "PINNeAPPle") -> str:
    """LDraw text of the build; with ``steps`` one ``0 STEP`` per layer (bottom first), as building instructions."""
    lines = [f"0 {asm.name or 'brick build'}", f"0 Name: {name}", f"0 Author: {author}"]
    order = sorted(range(len(asm.bricks)), key=lambda i: (asm.bricks[i].z, asm.bricks[i].x, asm.bricks[i].y))
    last = None
    for i in order:
        b = asm.bricks[i]
        if steps and last is not None and b.z != last:
            lines.append("0 STEP")
        lines.append(_line(b))
        last = b.z
    if steps and order:
        lines.append("0 STEP")
    return "\n".join(lines) + "\n"


def write_ldr(asm: Assembly, path: str, **kw) -> str:
    with open(path, "w", newline="\r\n") as f:
        f.write(to_ldr(asm, os.path.basename(path), **kw))
    return path


def write_mpd(models: dict[str, Assembly], path: str) -> str:
    """Several builds in one multi-part document; the first is the main model and references the others."""
    names = list(models)
    out = []
    for k, n in enumerate(names):
        fname = n if n.endswith(".ldr") else n + ".ldr"
        out.append(f"0 FILE {fname}")
        body = to_ldr(models[n], fname).rstrip("\n").split("\n")
        out += body
        if k == 0:
            for m in names[1:]:
                out.append(f"1 16 0 0 0 1 0 0 0 1 0 0 0 1 {m if m.endswith('.ldr') else m + '.ldr'}")
        out.append("0 NOFILE")
    with open(path, "w", newline="\r\n") as f:
        f.write("\n".join(out) + "\n")
    return path


@dataclass
class ReadReport:
    assembly: Assembly
    skipped: list[str] = field(default_factory=list)       # parts or placements that are not on the brick grid


def _quarter_yaw(R: np.ndarray, atol: float = 1e-3) -> int | None:
    """0 or 1 (quarter turns, mod 2: the rectangular parts are symmetric under a half turn) when ``R`` is a yaw by a
    multiple of 90 degrees about LDraw Y, else None."""
    if not (np.isclose(R[1, 1], 1.0, atol=atol) and np.allclose(R @ R.T, np.eye(3), atol=atol)):
        return None
    c, s = R[0, 0], R[0, 2]
    if np.isclose(abs(c), 1.0, atol=atol):
        return 0
    if np.isclose(abs(s), 1.0, atol=atol):
        return 1
    return None


def _parse_files(text: str) -> tuple[dict[str, list[str]], str]:
    files: dict[str, list[str]] = {}
    main, cur = None, None
    for raw in text.splitlines():
        s = raw.strip()
        if s.upper().startswith("0 FILE "):
            cur = s[7:].strip().lower()
            files[cur] = []
            main = main or cur
            continue
        if s.upper() == "0 NOFILE":
            cur = None
            continue
        if cur is None and main is None:
            main = "__main__"
            files[main] = []
            cur = main
        if cur is not None:
            files[cur].append(s)
    return files, main or "__main__"


def from_ldr(text: str, *, ground: Ground | None = None, tol: float = 0.5) -> ReadReport:
    """Read .ldr or .mpd text. Catalogue parts on the grid (positions within ``tol`` LDU, quarter-turn yaws) become
    bricks; everything else is listed in ``skipped``. Sub-models of an .mpd are expanded."""
    files, main = _parse_files(text)
    asm = Assembly(ground=ground)
    rep = ReadReport(asm)
    inv = np.linalg.inv(_TO_LDRAW)

    def walk(name, M, t, depth):
        if depth > 32:
            rep.skipped.append(f"{name}: sub-model nesting too deep")
            return
        for s in files.get(name, []):
            tok = s.split()
            if len(tok) < 15 or tok[0] != "1":
                continue
            code = int(tok[1])
            vals = np.array([float(v) for v in tok[2:14]])
            ref = " ".join(tok[14:]).lower().replace("\\", "/")
            p = M @ vals[:3] + t
            R = M @ vals[3:].reshape(3, 3)
            if ref in files:
                walk(ref, R, p, depth + 1)
                continue
            pt = BY_LDRAW.get(os.path.basename(ref))
            if pt is None:
                rep.skipped.append(f"{ref}: not a catalogue part")
                continue
            rot = _quarter_yaw(R)
            if rot is None:
                rep.skipped.append(f"{ref}: rotation is not a quarter turn about the vertical")
                continue
            c = inv @ (p * LDU_MM)                                      # mm, ours: top-centre of the part
            sx, sy = (pt.ny, pt.nx) if rot else (pt.nx, pt.ny)
            gx, gy = c[0] / PITCH_MM - sx / 2, c[1] / PITCH_MM - sy / 2
            gz = c[2] / PLATE_MM - pt.h
            g = np.array([gx, gy, gz])
            if np.abs(g - np.round(g)).max() * PITCH_MM / LDU_MM > tol:
                rep.skipped.append(f"{ref}: off the stud grid at {np.round(c, 2).tolist()} mm")
                continue
            asm.add(pt, *np.round(g).astype(int), rot=rot, colour=colour(code).name)

    walk(main, np.eye(3), np.zeros(3), 0)
    return rep


def read_ldr(path: str, **kw) -> ReadReport:
    with open(path, encoding="utf-8", errors="replace") as f:
        return from_ldr(f.read(), **kw)


# ---------------------------------------------------------------------------------------------- parts library
def default_cache() -> str:
    return os.path.join(os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache")), "pinneapple", "ldraw")


def fetch_parts_library(cache_dir: str | None = None, urls=LIBRARY_URLS, timeout: float = 120.0) -> str:
    """Download and unpack the LDraw parts library (CC BY 2.0) into ``cache_dir`` once; returns the folder that
    holds ``parts/`` and ``p/``. Raises ``RuntimeError`` with every URL tried when it cannot be downloaded."""
    import urllib.request
    root = cache_dir or default_cache()
    lib = os.path.join(root, "ldraw")
    if os.path.isdir(os.path.join(lib, "parts")):
        return lib
    os.makedirs(root, exist_ok=True)
    zpath = os.path.join(root, "complete.zip")
    errors = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(zpath, "wb") as f:
                f.write(r.read())
            break
        except Exception as e:                      # network policy, DNS, HTTP errors: try the next mirror
            errors.append(f"{url}: {e}")
    else:
        raise RuntimeError("could not download the LDraw parts library: " + "; ".join(errors))
    with zipfile.ZipFile(zpath) as z:
        for m in z.namelist():                      # stay inside the cache folder
            if m.startswith("/") or ".." in m.split("/"):
                raise RuntimeError(f"unsafe path in archive: {m}")
        z.extractall(root)
    return lib


class PartsLibrary:
    """Resolve and triangulate LDraw .dat files (line types 1, 3, 4; BFC winding is not enforced)."""

    def __init__(self, root: str):
        self.root = root
        self._index: dict[str, str] = {}
        for sub in ("parts", os.path.join("parts", "s"), "p", os.path.join("p", "48"), "models", ""):
            d = os.path.join(root, sub)
            if os.path.isdir(d):
                for fn in os.listdir(d):
                    if fn.lower().endswith((".dat", ".ldr")):
                        key = (("s/" if sub.endswith(os.path.join("parts", "s")) else
                                "48/" if sub.endswith(os.path.join("p", "48")) else "") + fn).lower()
                        self._index.setdefault(key, os.path.join(d, fn))
        self._cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def path(self, name: str) -> str | None:
        return self._index.get(name.lower().replace("\\", "/"))

    def mesh_ldu(self, name: str, depth: int = 0) -> tuple[np.ndarray, np.ndarray]:
        key = name.lower().replace("\\", "/")
        if key in self._cache:
            return self._cache[key]
        p = self.path(key)
        if p is None or depth > 40:
            raise FileNotFoundError(f"LDraw file {name!r} not found under {self.root}")
        Vs, Fs, off = [], [], 0
        with open(p, encoding="utf-8", errors="replace") as f:
            for raw in f:
                t = raw.split()
                if not t:
                    continue
                if t[0] == "3" and len(t) >= 11:
                    Vs.append(np.array(t[2:11], float).reshape(3, 3))
                    Fs.append(np.array([[off, off + 1, off + 2]]))
                    off += 3
                elif t[0] == "4" and len(t) >= 14:
                    Vs.append(np.array(t[2:14], float).reshape(4, 3))
                    Fs.append(np.array([[off, off + 1, off + 2], [off, off + 2, off + 3]]))
                    off += 4
                elif t[0] == "1" and len(t) >= 15:
                    v = np.array(t[2:14], float)
                    sub_v, sub_f = self.mesh_ldu(" ".join(t[14:]), depth + 1)
                    if len(sub_v):
                        Vs.append(sub_v @ v[3:].reshape(3, 3).T + v[:3])
                        Fs.append(sub_f + off)
                        off += len(sub_v)
        out = (np.vstack(Vs), np.vstack(Fs)) if Vs else (np.zeros((0, 3)), np.zeros((0, 3), np.int64))
        self._cache[key] = out
        return out

    def part_mesh_mm(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Mesh of a part in millimetres in this package's frame (z up), origin at the part origin."""
        V, F = self.mesh_ldu(name)
        return (V * LDU_MM) @ np.linalg.inv(_TO_LDRAW).T, F

    def assembly_meshes(self, asm: Assembly) -> list[tuple[np.ndarray, np.ndarray, tuple[float, float, float]]]:
        """Exact LDraw meshes of every brick of ``asm`` placed in millimetres (falls back to nothing on a miss)."""
        inv = np.linalg.inv(_TO_LDRAW)
        out = []
        for b in asm.bricks:
            V, F = self.mesh_ldu(b.part.ldraw)
            R = _ROT[b.rot % 2]
            cx, cy, top = (b.x + b.sx / 2) * PITCH_MM, (b.y + b.sy / 2) * PITCH_MM, b.top * PLATE_MM
            Vw = (V @ R.T) * LDU_MM @ inv.T + np.array([cx, cy, top])
            out.append((Vw, F, colour(b.colour).rgb))
        return out
