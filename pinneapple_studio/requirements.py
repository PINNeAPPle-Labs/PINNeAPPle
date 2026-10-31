"""What a script needs to run on this machine, read from its code, and the command that provides each missing piece.

* Python packages: the top-level modules it imports that are not in the standard library or the repository, with the
  ``pyproject.toml`` extra that installs them (``pip install -e ".[extra]"``) or the pip package;
* external programs it calls (OpenFOAM, FreeCAD, CalculiX, ParaView, Blender, Ollama, ...): ``shutil.which``;
* a GPU, when the script asks for CUDA;
* the network, when it downloads data, and credentials, when it reads a token or key from the environment.

:func:`check` returns one row per requirement with ``ok`` and the fix; nothing here imports the script.
"""
from __future__ import annotations

import ast
import importlib.util
import re
import shutil
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import tomllib

from .inventory import repo_root

# import name -> pip distribution, where they differ
DIST = {"cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml", "sklearn": "scikit-learn", "pxr": "usd-core",
        "physicsnemo": "nvidia-physicsnemo", "skimage": "scikit-image", "OCC": "cadquery-ocp", "fitz": "pymupdf",
        "ee": "earthengine-api", "dolfinx": "fenics-dolfinx", "jax": "jax", "mpl_toolkits": "matplotlib",
        "google": "google-cloud", "attr": "attrs", "dateutil": "python-dateutil", "bs4": "beautifulsoup4",
        "pyvista": "pyvista", "trimesh": "trimesh", "gmsh": "gmsh", "meshio": "meshio", "torch_geometric":
        "torch-geometric", "lightning": "lightning", "pytorch_lightning": "pytorch-lightning", "onnxruntime":
        "onnxruntime", "huggingface_hub": "huggingface_hub", "usdbrep": "usd-brep", "mujoco": "mujoco"}
PROGRAMS = {
    "OpenFOAM": (r"simpleFoam|pimpleFoam|icoFoam|blockMesh|snappyHexMesh|foamRun|openfoam", "simpleFoam",
                 "install OpenFOAM (https://openfoam.org/download/) and source its bashrc"),
    "FreeCAD": (r"freecadcmd|FreeCADCmd|import FreeCAD", "freecadcmd", "install FreeCAD 0.21+ (freecadcmd on PATH)"),
    "CalculiX": (r"\bccx\b|calculix", "ccx", "apt install calculix-ccx"),
    "ParaView": (r"pvpython|pvbatch|paraview\.simple", "pvpython", "install ParaView (pvpython on PATH)"),
    "Blender": (r"\bblender\b.*-b|bpy\b", "blender", "install Blender 4.x (blender on PATH)"),
    "Ollama": (r"ollama|11434", "ollama", "install Ollama (https://ollama.com) and `ollama pull <model>`"),
    "EMerge": (r"\bemerge\b", "emerge", "pip install emerge (external FEM solver)"),
}
NETWORK = re.compile(r"https?://(?!github\.com/PINNeAPPle|doi\.org|arxiv\.org|en\.wikipedia)|fetch_iv|urlopen|"
                     r"requests\.get|hf_hub_download|snapshot_download|earthengine|ee\.Initialize|wget ")
CREDENTIAL = re.compile(r"os\.environ(?:\.get)?\(?\[?\s*['\"]([A-Z0-9_]*(?:TOKEN|KEY|SECRET|PASSWORD)[A-Z0-9_]*)")
GPU = re.compile(r"device\s*=\s*['\"]cuda|\.cuda\(\)|require_gpu|torch\.cuda\.is_available\(\)\s*\)?\s*else\s*None")


@lru_cache(maxsize=4)
def extras(root: str) -> dict[str, list[str]]:
    """``{distribution (lower case): [extras that install it]}`` from ``pyproject.toml``."""
    py = tomllib.loads((Path(root) / "pyproject.toml").read_text())
    out: dict[str, list[str]] = {}
    for extra, reqs in py["project"].get("optional-dependencies", {}).items():
        for r in reqs:
            name = re.split(r"[<>=!~;\[ ]", r, maxsplit=1)[0].strip().lower().replace("_", "-")
            if name and not name.startswith("pinneapple"):
                out.setdefault(name, []).append(extra)
    return out


def _local(root: Path) -> set[str]:
    return {p.name for p in root.iterdir() if p.is_dir() and (p / "__init__.py").exists()} | \
        {p.stem for p in root.glob("*.py")}


def imports_of(src: str) -> set[str]:
    """Top-level modules a script imports (absolute imports only)."""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return set()
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            out |= {a.name.split(".")[0] for a in n.names}
        elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
            out.add(n.module.split(".")[0])
    return out


def requirements(paths: list[Path], root: Path | None = None) -> list[dict[str, Any]]:
    """The requirements of a set of scripts (an example folder), unchecked."""
    root = Path(root or repo_root())
    src = "\n".join(p.read_text(encoding="utf-8", errors="replace") for p in paths if p.is_file())
    here = {p.parent.name for p in paths} | {p.stem for p in paths}
    local = _local(root) | here | {"conftest"}
    rows = []
    ex = extras(str(root))
    for mod in sorted(imports_of(src)):
        if mod in sys.stdlib_module_names or mod in local or mod.startswith("pinneapple") or mod == "__future__":
            continue
        dist = DIST.get(mod, mod).lower().replace("_", "-")
        via = ex.get(dist, [])
        fix = f'pip install -e ".[{via[0]}]"' if via else f"pip install {DIST.get(mod, mod)}"
        rows.append({"kind": "package", "name": mod, "dist": dist, "extras": via, "fix": fix})
    for name, (pattern, exe, fix) in PROGRAMS.items():
        if re.search(pattern, src):
            rows.append({"kind": "program", "name": name, "exe": exe, "fix": fix})
    if GPU.search(src):
        rows.append({"kind": "gpu", "name": "CUDA GPU", "fix": "run on a machine with an NVIDIA GPU and CUDA torch"})
    if NETWORK.search(src):
        rows.append({"kind": "network", "name": "internet access", "fix": "run with network access (data is "
                     "downloaded at run time)"})
    for var in sorted(set(CREDENTIAL.findall(src))):
        rows.append({"kind": "credential", "name": var, "fix": f"set {var} in the environment (never in the repo)"})
    return rows


def check(rows: list[dict[str, Any]], *, allow_network: bool = True) -> list[dict[str, Any]]:
    """Each requirement with ``ok`` on this machine (packages by ``find_spec``, programs by ``which``)."""
    out = []
    for r in rows:
        r = dict(r)
        if r["kind"] == "package":
            try:
                r["ok"] = importlib.util.find_spec(r["name"]) is not None
            except (ImportError, ValueError):
                r["ok"] = False
        elif r["kind"] == "program":
            r["ok"] = shutil.which(r["exe"]) is not None
        elif r["kind"] == "gpu":
            r["ok"] = _cuda()
        elif r["kind"] == "network":
            r["ok"] = allow_network
        elif r["kind"] == "credential":
            import os
            r["ok"] = bool(os.environ.get(r["name"]))
        out.append(r)
    return out


@lru_cache(maxsize=1)
def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001 - no torch, no GPU
        return False
