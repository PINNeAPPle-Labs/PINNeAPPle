"""Rendering without a display: what this machine has, and the shortest recipe to a frame (#66).

PyVista and VTK open an OpenGL context even for an off-screen render, and on a server with no X or Wayland
display there is nothing to open one against: the render either fails or aborts the process. Three things fix
it, in order of how little they change: a virtual display (``xvfb-run``), an OSMesa build of VTK, or an EGL
context. This module says which of those this machine already has, which one to install, and then actually tries
the render -- VTK can abort the interpreter instead of raising, so the attempt happens in a subprocess.

    python -m pinneapple_tools.visualization.headless      # the diagnosis and a real test render

    import pinneapple as pp
    pp.viz.headless()                                      # the same, as a dict

Run it inside the container in ``docker/headless`` to check that the image is what it claims to be.
"""
from __future__ import annotations

import ctypes.util
import os
import platform
import shutil
import subprocess
import sys
import time

# The commands each recipe is made of: the documentation quotes them from here, and tests/test_headless_rendering.py
# fails when the two drift apart.
RECIPES: dict[str, list[str]] = {
    "ready": [],                                            # it renders here; nothing to install
    "install": [
        'pip install "pinneapple[pyvista]"',
    ],
    "xvfb": [
        "apt-get install -y xvfb",
        'xvfb-run -a python -m pinneapple_tools.visualization.headless',
    ],
    "osmesa": [
        "pip uninstall -y vtk",
        "pip install vtk-osmesa",
        'PYVISTA_OFF_SCREEN=true python -m pinneapple_tools.visualization.headless',
    ],
    "egl": [
        "apt-get install -y libegl1 libgl1 libglvnd0",
        'PYVISTA_OFF_SCREEN=true python -m pinneapple_tools.visualization.headless',
    ],
    "docker": [
        "docker build -t pinneapple-headless -f docker/headless/Dockerfile .",
        "docker run --rm -v \"$PWD:/work\" -w /work pinneapple-headless",
    ],
}

WHY = {
    "ready": "off-screen rendering works here as it stands",
    "install": "pyvista is not installed, so there is nothing to render with yet",
    "xvfb": "no usable display, but xvfb-run is installed: it gives the render a virtual X display",
    "osmesa": "no usable display and no xvfb, but libOSMesa is installed: VTK can render into a memory buffer",
    "egl": "no usable display and no xvfb, but libEGL is installed: VTK can open a context without a window server",
    "docker": "no usable display and no xvfb, OSMesa or EGL: run it in the image in docker/headless",
}


def _version(module: str) -> str | None:
    try:
        return __import__(module).__version__
    except Exception:
        return None


def _library(*names: str) -> str | None:
    for name in names:
        found = ctypes.util.find_library(name)
        if found:
            return found
    return None


def _environment() -> dict:
    """What this machine offers for off-screen rendering, without rendering anything."""
    system = platform.system()
    return {
        "platform": platform.platform(),
        "system": system,
        "python": sys.version.split()[0],
        "pyvista": _version("pyvista"),
        "vtk": _version("vtk"),
        # X and Wayland are a Linux matter: macOS and Windows always have a window server to render against.
        "display": bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")),
        "needs_display": system == "Linux",
        "xvfb_run": shutil.which("xvfb-run"),
        "osmesa": _library("OSMesa"),
        "egl": _library("EGL"),
        "off_screen_env": os.environ.get("PYVISTA_OFF_SCREEN"),
    }


def _recipe(env: dict | None = None, works: bool | None = None) -> str:
    """The shortest way to get a frame on this machine: one key of :data:`RECIPES`.

    ``works`` is what the probe saw. Left as ``None`` it is inferred from the environment, which is right on
    Linux and pessimistic on macOS and Windows, where a render with no ``DISPLAY`` is the normal case; pass the
    probe result when it matters.
    """
    env = _environment() if env is None else env
    if not env.get("pyvista") or not env.get("vtk"):
        return "install"
    if works is True:
        return "ready"
    if works is None and (env.get("display") or not env.get("needs_display", True)):
        return "ready"
    for name, present in (("xvfb", env.get("xvfb_run")), ("osmesa", env.get("osmesa")), ("egl", env.get("egl"))):
        if present:
            return name
    return "docker"


def _probe_source(size: tuple[int, int]) -> str:
    width, height = size
    return (
        "import sys\n"
        "import pyvista as pv\n"
        "pv.OFF_SCREEN = True\n"
        f"plotter = pv.Plotter(off_screen=True, window_size=[{width}, {height}])\n"
        "plotter.add_mesh(pv.Sphere(), color='#4c9be8')\n"
        "image = plotter.screenshot(return_img=True)\n"
        "plotter.close()\n"
        f"if image.shape != ({height}, {width}, 3):\n"
        "    print('unexpected frame shape', image.shape, file=sys.stderr); sys.exit(3)\n"
        "if image.max() == image.min():\n"
        "    print('the frame is one flat colour: the context opened but nothing was drawn', file=sys.stderr)\n"
        "    sys.exit(4)\n"
        "print(image.mean())\n"
    )


def _probe(size: tuple[int, int] = (320, 240), timeout: float = 120.0) -> dict:
    """Render a sphere off screen in a subprocess and report what happened.

    In a subprocess because VTK crashes the interpreter rather than raising when it cannot open a context, and
    because the whole point is to survive the attempt. The frame is not just 'some array': a flat one means the
    context opened on a window nothing was drawn into, which is the failure mode of a broken EGL/OSMesa setup.
    """
    started = time.time()
    try:
        result = subprocess.run([sys.executable, "-c", _probe_source(size)], capture_output=True, text=True,
                                timeout=timeout)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}", "seconds": time.time() - started,
                "size": list(size)}
    seconds = time.time() - started
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "").strip().splitlines()
        return {"ok": False, "error": message[-1] if message else f"exit status {result.returncode}",
                "seconds": seconds, "size": list(size)}
    return {"ok": True, "error": None, "seconds": seconds, "size": list(size)}


def headless(size: tuple[int, int] = (320, 240), run_probe: bool = True) -> dict:
    """Everything about rendering on this machine in one dict, and the commands that would fix it.

    ``can_render`` is ``None`` when nothing was attempted (``run_probe=False``): pyvista absent, or the caller
    only wants the environment. ``recipe`` names the path -- one key of :data:`RECIPES` -- and ``commands``
    holds the exact lines from it, so a caller (or a documentation page) never paraphrases them.
    """
    env = _environment()
    result = _probe(size) if run_probe else None
    works = None if result is None else result["ok"]
    plan = _recipe(env, works)
    why = WHY[plan]
    if works is False and plan == "ready":                  # a display is there, but rendering into it failed
        plan = _recipe(env, works=False)
        why = "a display is present, but the off-screen render failed: " + WHY[plan]
    return {
        "can_render": None if result is None else result["ok"],
        "error": None if result is None else result["error"],
        "seconds": None if result is None else result["seconds"],
        "size": list(size),
        "recipe": plan,
        "why": why,
        "commands": list(RECIPES[plan]),
        "environment": env,
    }


def _report_lines(report: dict) -> list[str]:
    env = report["environment"]
    width, height = report["size"]
    lines = [
        f"python     {env['python']} on {env['platform']}",
        f"pyvista    {env['pyvista'] or 'not installed'}",
        f"vtk        {env['vtk'] or 'not installed'}",
        f"display    {'yes' if env['display'] else 'no'}",
        f"xvfb-run   {env['xvfb_run'] or 'not installed'}",
        f"libOSMesa  {env['osmesa'] or 'not found'}",
        f"libEGL     {env['egl'] or 'not found'}",
        "",
        f"recipe     {report['recipe']}: {report['why']}",
    ]
    if report["can_render"] is True:
        lines.append(f"render     ok, {report['seconds']:.1f} s for one {width}x{height} frame")
    elif report["can_render"] is False:
        lines.append(f"render     failed: {report['error']}")
    else:
        lines.append("render     not attempted")
    if report["commands"]:
        lines.append("")
        lines.append("to fix it:")
        lines.extend(f"  {command}" for command in report["commands"])
    return lines


if __name__ == "__main__":                                   # python -m pinneapple_tools.visualization.headless
    report = headless()
    print("\n".join(_report_lines(report)))
    raise SystemExit(0 if report["can_render"] is not False else 1)
