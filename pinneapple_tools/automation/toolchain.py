"""Where the engineering applications are installed and which versions they are.

Found from environment variables first, then the usual install locations, then ``PATH``:

=====================  ======================================  =============================================
tool                   variable                                looked for
=====================  ======================================  =============================================
FreeCAD (GUI)          ``PINNEAPPLE_FREECAD``                  ``freecad``, ``FreeCAD``, ``/opt/freecad/usr/bin/freecad``
FreeCAD (console)      ``PINNEAPPLE_FREECADCMD``               ``freecadcmd``, ``FreeCADCmd``
CfdOF workbench        ``PINNEAPPLE_CFDOF``                    ``~/.local/share/FreeCAD/Mod/CfdOF``, ``/opt/freecad-mods/CfdOF``
OpenFOAM               ``FOAM_BASHRC``                         ``/usr/share/openfoam/etc/bashrc`` (Ubuntu ``openfoam``), ``/opt/openfoam*/etc/bashrc``
gmsh                   ``PINNEAPPLE_GMSH``                     next to FreeCAD (the AppImage ships it), ``gmsh``
ParaView (Python)      ``PINNEAPPLE_PVPYTHON``                 ``pvpython``
=====================  ======================================  =============================================
"""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

__all__ = ["Toolchain"]


def _first(*cands: str | None) -> str | None:
    for c in cands:
        if not c:
            continue
        p = shutil.which(c) if os.sep not in c else (c if Path(c).exists() else None)
        if p:
            return p
    return None


@dataclass
class Toolchain:
    freecad: str | None = None
    freecadcmd: str | None = None
    cfdof: str | None = None
    foam_bashrc: str | None = None
    gmsh: str | None = None
    pvpython: str | None = None

    @classmethod
    def discover(cls) -> Toolchain:
        e = os.environ.get
        freecad = _first(e("PINNEAPPLE_FREECAD"), "freecad", "FreeCAD", "/opt/freecad/usr/bin/freecad")
        fc_dir = str(Path(freecad).parent) if freecad else None
        freecadcmd = _first(e("PINNEAPPLE_FREECADCMD"), fc_dir and os.path.join(fc_dir, "freecadcmd"), "freecadcmd",
                            "FreeCADCmd")
        cfdof = _first(e("PINNEAPPLE_CFDOF"), os.path.expanduser("~/.local/share/FreeCAD/Mod/CfdOF"),
                       "/opt/freecad-mods/CfdOF")
        bashrc = _first(e("FOAM_BASHRC"), "/usr/share/openfoam/etc/bashrc",
                        *sorted(glob.glob("/opt/openfoam*/etc/bashrc"), reverse=True),
                        *sorted(glob.glob("/usr/lib/openfoam/openfoam*/etc/bashrc"), reverse=True))
        gmsh = _first(e("PINNEAPPLE_GMSH"), fc_dir and os.path.join(fc_dir, "gmsh"), "gmsh")
        return cls(freecad, freecadcmd, cfdof, bashrc, gmsh, _first(e("PINNEAPPLE_PVPYTHON"), "pvpython"))

    def to_dict(self) -> dict:
        return asdict(self)

    def missing(self, tools: list[str]) -> list[str]:
        """The names in ``tools`` (freecad, freecadcmd, cfdof, openfoam, gmsh, paraview, xvfb, xdotool) not found."""
        have = {"freecad": self.freecad, "freecadcmd": self.freecadcmd, "cfdof": self.cfdof,
                "openfoam": self.foam_bashrc, "gmsh": self.gmsh, "paraview": self.pvpython,
                "xvfb": shutil.which("Xvfb"), "xdotool": shutil.which("xdotool")}
        return [t for t in tools if not have.get(t)]

    def version(self, tool: str) -> str | None:
        """Version string of one tool as it reports it (freecad, cfdof, openfoam, gmsh, paraview); None when the
        tool is missing or does not answer. Cached per toolchain."""
        cache = self.__dict__.setdefault("_versions", {})
        if tool not in cache:
            cache[tool] = self._ask(tool)
        return cache[tool]

    def versions(self) -> dict[str, str | None]:
        return {t: self.version(t) for t in ("freecad", "cfdof", "openfoam", "gmsh", "paraview")}

    def _ask(self, tool: str) -> str | None:
        if tool == "freecad":
            return _version([self.freecadcmd, "--version"], r"FreeCAD\s+([\d.]+)") if self.freecadcmd else None
        if tool == "cfdof":
            if self.cfdof and Path(self.cfdof, "package.xml").is_file():
                m = re.search(r"<version>([^<]+)</version>", Path(self.cfdof, "package.xml").read_text(errors="replace"))
                return m.group(1).strip() if m else None
            return None
        if tool == "openfoam":
            return _version(["bash", "-c", f'source "{self.foam_bashrc}" >/dev/null 2>&1; echo "$WM_PROJECT_VERSION"'],
                            r"(\S+)") if self.foam_bashrc else None
        if tool == "gmsh":
            return _version([self.gmsh, "--version"], r"([\d.]+)") if self.gmsh else None
        if tool == "paraview":
            return _version([self.pvpython, "--version"], r"paraview version ([\d.]+)") if self.pvpython else None
        raise KeyError(tool)


def _version(cmd: list[str], pattern: str) -> str | None:
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(pattern, (r.stdout or "") + (r.stderr or ""))
    return m.group(1) if m else None
