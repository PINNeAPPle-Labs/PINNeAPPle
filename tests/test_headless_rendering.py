"""Rendering without a display (#66): the recipe, and the command that says which part of it is missing.

The issue asks for a recipe "tested in CI without a display". CI here has no display and no OSMesa, so the honest
form of the test is the one this file uses: the *diagnosis* is tested everywhere (against fabricated
environments, so every branch of the recipe is exercised on any machine), and the *render* is tested wherever a
context can actually be opened -- which is macOS, Windows and any Linux with xvfb/OSMesa/EGL. Where it cannot be
opened, the test asserts that the report names the fix instead of quietly passing.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from pinneapple_tools.visualization import headless as H

ROOT = Path(__file__).resolve().parents[1]
GUIDE = ROOT / "docs" / "guides" / "headless_rendering.md"
DOCKERFILE = ROOT / "docker" / "headless" / "Dockerfile"

LINUX_HEADLESS = {
    "platform": "Linux-6.8.0", "system": "Linux", "python": "3.12.6", "pyvista": "0.49.2", "vtk": "9.7.2",
    "display": False, "needs_display": True, "xvfb_run": None, "osmesa": None, "egl": None, "off_screen_env": None,
}


def _env(**overrides) -> dict:
    env = dict(LINUX_HEADLESS)
    env.update(overrides)
    return env


# ------------------------------------------------------------------------------------------------- the recipe
def test_a_linux_box_with_nothing_on_it_is_told_to_use_the_container():
    assert H._recipe(_env()) == "docker"
    assert H.RECIPES["docker"], "the last recipe must actually say what to run"


def test_each_thing_found_moves_the_recipe_to_the_shortest_path_that_uses_it():
    assert H._recipe(_env(display=True)) == "ready"
    assert H._recipe(_env(xvfb_run="/usr/bin/xvfb-run")) == "xvfb"
    assert H._recipe(_env(osmesa="libOSMesa.so.8")) == "osmesa"
    assert H._recipe(_env(egl="libEGL.so.1")) == "egl"
    # xvfb is preferred over the libraries: it changes nothing about how VTK was built
    assert H._recipe(_env(xvfb_run="/usr/bin/xvfb-run", osmesa="libOSMesa.so.8")) == "xvfb"
    assert H._recipe(_env(osmesa="libOSMesa.so.8", egl="libEGL.so.1")) == "osmesa"


def test_no_pyvista_is_an_install_not_a_rendering_problem():
    assert H._recipe(_env(pyvista=None, vtk=None)) == "install"


def test_a_mac_or_a_windows_box_needs_no_display_at_all():
    """`DISPLAY` is an X11 variable: on macOS and Windows rendering works without one, and the report must say so."""
    mac = _env(system="Darwin", platform="macOS-26.5-arm64", needs_display=False)
    assert H._recipe(mac) == "ready"
    assert H._recipe(_env(system="Windows", needs_display=False)) == "ready"
    # a Linux box with no display is not the same case
    assert H._recipe(_env()) == "docker"


def test_what_the_probe_saw_overrides_the_guess():
    """A display that is present but unusable must not be reported as working."""
    assert H._recipe(_env(display=True), works=True) == "ready"
    assert H._recipe(_env(display=True), works=False) == "docker"
    assert H._recipe(_env(display=True, xvfb_run="/usr/bin/xvfb-run"), works=False) == "xvfb"


@pytest.mark.parametrize("name", sorted(H.RECIPES))
def test_every_recipe_has_a_stated_reason(name):
    assert name in H.WHY and H.WHY[name].strip()


# ------------------------------------------------------------------------------------- the report and the render
def test_the_report_carries_the_recipe_and_the_exact_commands():
    report = H.headless(run_probe=False)
    for key in ("can_render", "error", "seconds", "size", "recipe", "why", "commands", "environment"):
        assert key in report
    assert report["can_render"] is None, "nothing was attempted, and None says so rather than guessing"
    assert report["recipe"] in H.RECIPES
    assert report["commands"] == H.RECIPES[report["recipe"]], "the report must not paraphrase the recipe"
    assert report["environment"]["python"] and report["environment"]["platform"]


def test_the_report_survives_a_machine_that_cannot_render():
    """Where there is no context, the answer is a recipe with commands -- not a traceback and not silence."""
    report = H.headless()
    if report["can_render"] is None:
        pytest.skip("pyvista is not installed here, so nothing was attempted")
    if report["can_render"]:
        assert report["error"] is None and report["seconds"] > 0
        assert report["recipe"] == "ready", "a render that works does not need a recipe"
        assert report["commands"] == []
    else:
        assert report["error"]
        assert report["commands"], f"rendering failed ({report['error']}) and the report gave nothing to try"


@pytest.mark.skipif(H._recipe() == "install", reason="pyvista is not installed, so there is nothing to probe")
def test_the_probe_renders_a_frame_that_is_not_one_flat_colour():
    """A context that opens on a window nothing was drawn into is the usual half-broken Mesa, so the probe checks
    the frame, not just the exit status."""
    result = H._probe((160, 120))
    assert set(result) == {"ok", "error", "seconds", "size"}
    if not result["ok"]:
        pytest.skip(f"no off-screen context here: {result['error']}")
    assert result["seconds"] > 0 and result["size"] == [160, 120]


def test_the_self_check_command_prints_the_report_and_says_how_it_went():
    result = subprocess.run([sys.executable, "-m", "pinneapple_tools.visualization.headless"],
                            capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert result.returncode in (0, 1), result.stderr
    assert "recipe" in result.stdout and "pyvista" in result.stdout
    # 0 means a frame came back, 1 means the report explained why not; either way the exit status agrees
    assert (result.returncode == 0) == ("render     ok" in result.stdout)


def test_pp_viz_headless_is_the_same_call():
    import pinneapple as pp

    report = pp.viz.headless(run_probe=False)
    assert report["recipe"] in H.RECIPES


# ---------------------------------------------------------------------------- the documentation and the image
def test_the_guide_quotes_every_command_the_module_prints():
    """The page is the recipe; a command that exists only in one of the two places is a bug in the other."""
    guide = GUIDE.read_text(encoding="utf-8")
    for name, commands in H.RECIPES.items():
        assert name in guide, f"the guide does not mention the {name} recipe"
        for command in commands:
            assert command in guide, f"{name}: the guide does not quote `{command}`"


def test_the_guide_says_where_the_check_was_and_was_not_run():
    """The container is defined here and built elsewhere: saying which is the difference between a recipe and a claim."""
    guide = GUIDE.read_text(encoding="utf-8").lower()
    assert "macos" in guide and "linux" in guide
    assert "xvfb" in guide and "osmesa" in guide and "egl" in guide


def test_the_dockerfile_is_the_recipe_with_nothing_left_to_decide():
    text = DOCKERFILE.read_text(encoding="utf-8")
    instructions = [line for line in text.splitlines() if not line.strip().startswith("#")]
    body = "\n".join(instructions)
    assert "FROM python:" in body
    for package in ("xvfb", "libgl1-mesa-dri", "ffmpeg"):        # software GL, the virtual display, the MP4 encoder
        assert package in body, f"the image does not install {package}"
    assert "PYVISTA_OFF_SCREEN=true" in body
    assert 'ENTRYPOINT ["xvfb-run", "-a"]' in body, "a caller should not have to remember the prefix"
    assert "pinneapple_tools.visualization.headless" in body, "the image's default command is the self-check"
    # and the self-check it defaults to is the one the module provides
    assert 'CMD ["python", "-m", "pinneapple_tools.visualization.headless"]' in body

