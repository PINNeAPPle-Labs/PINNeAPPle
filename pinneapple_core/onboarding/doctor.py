"""Is this installation ready for what I want to do? One check per thing, one fix per failure.

``pp.info()`` answers "what is in the library". ``pp.doctor()`` answers the question a new user
actually has, which is narrower and more useful: *for the work I intend to do, is anything
missing?* The intent is named by a profile:

=============  ==========================================================================
profile        for
=============  ==========================================================================
``start``      the first week: solve a preset, plot a field, train one small network
``cfd``        geometry to mesh to an OpenFOAM/CalculiX campaign, plus 3-D viewing
``research``   PINNs, operators, uncertainty, JAX, publication figures
``enterprise`` serving, signatures, the physics linter in CI, the hub and the MCP server
``all``        every extra the package declares
=============  ==========================================================================

Every failure prints the exact command that fixes it, and nothing is reported as fine when it was
not checked: a check that cannot run says so.

>>> import pinneapple as pp
>>> report = pp.doctor("cfd")          # returns the report and prints it
>>> report.ok                          # True when nothing required is missing
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = ["DoctorReport", "Check", "doctor", "profiles", "PROFILES"]

# Distribution names probed per extra; the first that imports decides the extra is present.
_EXTRA_PROBES: Dict[str, Tuple[str, ...]] = {
    "cfd": ("manifold3d", "cadquery", "gmsh", "trimesh", "pyvista", "plotly", "meshio"),
    "enterprise": ("fastapi", "cryptography", "huggingface_hub", "mcp", "meshio"),
    "geometry": ("manifold3d", "huggingface_hub"),
    "fitmesh": ("cadquery", "gmsh", "trimesh"),
    "viz3d": ("pyvista", "plotly"),
    "pyvista": ("pyvista",),
    "serve": ("fastapi", "uvicorn"),
    "em": ("miepython",),
    "pinn": ("sympy",),
    "uq": ("scipy",),
    "jax_cfd": ("jax",),
    "check": ("meshio", "h5py"),
    "security": ("cryptography", "argon2"),
    "twin": ("basyx", "asyncua"),
    "export": ("onnx",),
    "mcp": ("mcp",),
    "fmi": ("fmpy",),
    "mujoco": ("mujoco",),
    "fenics": ("dolfin", "dolfinx"),
    "hub": ("huggingface_hub",),
    "orchestration": ("prefect",),
}

PROFILES: Dict[str, Dict[str, Any]] = {
    "start": {
        "description": "first week: solve a preset, plot a field, train one small network",
        "extras": (),
        "programs": (),
        "base": ("numpy", "scipy", "torch", "matplotlib", "pandas", "sympy"),
    },
    "cfd": {
        "description": "geometry, mesh and solver campaigns (OpenFOAM, CalculiX), 3-D viewing",
        "extras": ("cfd",),
        "programs": ("blockMesh", "simpleFoam", "gmsh", "ccx"),
        "base": ("numpy", "scipy", "torch", "matplotlib"),
    },
    "research": {
        "description": "PINNs, neural operators, uncertainty, JAX, publication figures",
        "extras": ("pinn", "uq", "viz3d", "jax_cfd"),
        "programs": (),
        "base": ("numpy", "scipy", "torch", "matplotlib", "pandas", "sympy"),
    },
    "enterprise": {
        "description": "serving, signatures, the physics linter in CI, the hub and the MCP server",
        "extras": ("enterprise",),
        "programs": (),
        "base": ("numpy", "scipy", "torch", "pandas"),
    },
    "all": {
        "description": "every extra the package declares",
        "extras": tuple(k for k in _EXTRA_PROBES if k not in ("start",)),
        "programs": ("blockMesh", "gmsh", "ccx"),
        "base": ("numpy", "scipy", "torch", "matplotlib", "pandas", "sympy"),
    },
}

# External programs that the Python side cannot install, with the reason each one matters.
_PROGRAMS: Dict[str, str] = {
    "blockMesh": "OpenFOAM meshing: required by the CFD campaigns",
    "simpleFoam": "OpenFOAM steady solver: required by the CFD campaigns",
    "pimpleFoam": "OpenFOAM transient solver: required by the CFD campaigns",
    "gmsh": "mesh generation for CAD geometry (pinneapple[fitmesh])",
    "ccx": "CalculiX structural solves (pinneapple-ccx uses it outside its envelope)",
    "FreeCADCmd": "headless FreeCAD for the parametric CAD pipelines",
}

_MIN_PYTHON = (3, 10)


@dataclass
class Check:
    """One thing that was checked. ``ok`` is None when the check could not run."""

    name: str
    ok: Optional[bool]
    detail: str = ""
    fix: str = ""
    required: bool = True

    def line(self) -> str:
        glyph = {True: "ok  ", False: "MISS", None: "?   "}[self.ok]
        out = f"  [{glyph}] {self.name:<26} {self.detail}"
        if self.fix:
            out += f"\n         fix: {self.fix}"
        return out


@dataclass
class DoctorReport:
    profile: str
    checks: List[Check] = field(default_factory=list)

    @property
    def failures(self) -> List[Check]:
        return [c for c in self.checks if c.ok is False and c.required]

    @property
    def ok(self) -> bool:
        return not self.failures

    def as_dict(self) -> Dict[str, Any]:
        return {
            "profile": self.profile, "ok": self.ok,
            "checks": [{"name": c.name, "ok": c.ok, "detail": c.detail, "fix": c.fix,
                        "required": c.required} for c in self.checks],
        }

    def format(self) -> str:
        bar = "-" * 78
        desc = PROFILES.get(self.profile, {}).get("description", "")
        out = [bar, f"  pinneapple doctor   profile: {self.profile}"
                    + (f"  ({desc})" if desc else ""), bar]
        out += [c.line() for c in self.checks]
        out.append(bar)
        if self.ok:
            out.append("  Nothing required is missing. Start with:")
            out.append("    import pinneapple as pp; pp.recommend('describe your problem')")
        else:
            n = len(self.failures)
            out.append(f"  {n} required item{'s' if n > 1 else ''} missing. Fix commands, in order:")
            for c in self.failures:
                if c.fix:
                    out.append(f"    {c.fix}")
        out.append(bar)
        return "\n".join(out)

    def __str__(self) -> str:
        return self.format()


def profiles() -> Dict[str, str]:
    """The profiles ``doctor`` accepts, with the sentence that describes each one."""
    return {k: v["description"] for k, v in PROFILES.items()}


def _installed(module: str) -> bool:
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def _check_python() -> Check:
    ok = sys.version_info[:2] >= _MIN_PYTHON
    return Check(
        "python", ok, f"{sys.version.split()[0]} ({platform.system()} {platform.machine()})",
        fix=f"pinneapple needs Python {_MIN_PYTHON[0]}.{_MIN_PYTHON[1]} or newer" if not ok else "",
    )


def _check_package() -> Check:
    try:
        import pinneapple

        version = getattr(pinneapple, "__version__", "unknown")
        where = os.path.dirname(os.path.abspath(pinneapple.__file__))
        editable = " (editable install)" if "site-packages" not in where else ""
        return Check("pinneapple", True, f"v{version}{editable}")
    except Exception as e:
        return Check("pinneapple", False, f"import failed: {e}", "pip install pinneapple")


def _check_base(base: Sequence[str]) -> List[Check]:
    out = []
    for mod in base:
        present = _installed(mod)
        out.append(Check(
            f"import {mod}", present, "" if present else "not importable",
            fix="" if present else "pip install pinneapple   # reinstalls the core dependencies",
        ))
    return out


def _check_device() -> Check:
    try:
        import torch

        if torch.cuda.is_available():
            n = torch.cuda.device_count()
            return Check("compute device", True, f"cuda, {n} device(s)")
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return Check("compute device", True, "apple mps (GPU)")
        return Check("compute device", True, "cpu only: fine for presets and small PINNs, slow for 3-D")
    except Exception as e:
        return Check("compute device", None, f"could not query torch: {e}")


def _check_extras(extras: Sequence[str]) -> List[Check]:
    out = []
    for extra in extras:
        probes = _EXTRA_PROBES.get(extra, ())
        if not probes:
            continue
        missing = [m for m in probes if not _installed(m)]
        if not missing:
            out.append(Check(f"extra [{extra}]", True, "installed"))
        elif len(missing) == len(probes):
            out.append(Check(
                f"extra [{extra}]", False, f"none of {', '.join(probes)} importable",
                fix=f'pip install "pinneapple[{extra}]"',
            ))
        else:
            out.append(Check(
                f"extra [{extra}]", True, f"partial: {', '.join(missing)} missing",
                fix=f'pip install "pinneapple[{extra}]"   # only if you need it',
                required=False,
            ))
    return out


def _check_programs(programs: Sequence[str]) -> List[Check]:
    out = []
    for prog in programs:
        path = shutil.which(prog)
        why = _PROGRAMS.get(prog, "")
        if path:
            out.append(Check(f"program {prog}", True, path))
        else:
            out.append(Check(
                f"program {prog}", False, f"not on PATH. {why}",
                fix=_program_fix(prog),
            ))
    return out


def _program_fix(prog: str) -> str:
    if prog.startswith(("blockMesh", "simpleFoam", "pimpleFoam")):
        return "install OpenFOAM (apt install openfoam, or the ESI build) and source its env"
    if prog == "gmsh":
        return "brew install gmsh   # or apt install gmsh, or the wheel: pip install gmsh"
    if prog == "ccx":
        return "install CalculiX (apt install calculix-ccx) or run pinneapple-ccx, which falls back to it"
    if prog == "FreeCADCmd":
        return "install FreeCAD; the parametric pipelines call it headless"
    return f"install {prog} and put it on PATH"


def _check_disk() -> Check:
    try:
        usage = shutil.disk_usage(os.getcwd())
        free_gb = usage.free / 1e9
        ok = free_gb > 2.0
        return Check("free disk", ok, f"{free_gb:.1f} GB",
                     fix="" if ok else "free at least 2 GB: runs and dataset caches are written here")
    except Exception:
        return Check("free disk", None, "could not read")


def _check_memory() -> Check:
    try:
        if sys.platform == "darwin":
            out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=5)
            total_gb = int(out.stdout.strip()) / 1e9
        else:
            total_gb = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        ok = total_gb >= 8.0
        return Check("system memory", True,
                     f"{total_gb:.0f} GB" + ("" if ok else " (8 GB or more is comfortable)"))
    except Exception:
        return Check("system memory", None, "could not read")


def doctor(profile: str = "start", *, verbose: bool = True) -> DoctorReport:
    """Check this installation against a profile and, with ``verbose``, print the report.

    ``profile`` is one of :func:`profiles`. Nothing is installed and nothing is downloaded; every
    missing item comes with the command that fixes it.
    """
    key = str(profile or "start").lower().strip()
    if key not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose one of: {', '.join(PROFILES)}")
    spec = PROFILES[key]
    checks: List[Check] = [_check_python(), _check_package(), _check_device(), _check_disk(), _check_memory()]
    checks += _check_base(spec["base"])
    checks += _check_extras(spec["extras"])
    checks += _check_programs(spec["programs"])
    report = DoctorReport(profile=key, checks=checks)
    if verbose:
        print(report.format())
    return report
