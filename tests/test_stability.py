"""What PINNeAPPle promises (#1127): the core, the experimental rest, the labs warning, the catalog field."""
import os
import subprocess
import sys
from pathlib import Path

from pinneapple_stability import CORE, CORE_PP, LABS, is_core, stability

ROOT = Path(__file__).resolve().parents[1]


def _env(**extra):
    return {**os.environ, "PYTHONPATH": os.pathsep.join([str(ROOT), os.environ.get("PYTHONPATH", "")]), **extra}


def test_core_and_experimental_by_module():
    assert stability("pinneapple_physics.pinn_solver.compiler") == "core"
    assert stability("pinneapple_design.geometry.brep.study") == "core"
    assert stability("pinneapple_physics.pinn_solverX") == "experimental"          # a prefix, not a substring
    assert stability("pinneapple_simulation.electromagnetics.particles") == "experimental"
    assert stability("pinneapple_arena.pointwise") == "experimental (labs)"
    assert stability("pinneapple_systems.sports.football") == "experimental (labs)"
    assert stability("pinneapple_systems.control") == "experimental"


def test_the_facade_by_name():
    assert stability("pinneapple", "solve") == "core" and stability("pinneapple", "science") == "experimental"
    assert is_core("pinneapple") and is_core("pinneapple", "validate") and not is_core("pinneapple", "vehicle")


def test_literal_calls():
    """Plain-string calls (also the Studio's smoke inputs)."""
    assert stability("pinneapple_presets.water_hammer") == "core"
    assert is_core("pinneapple_neural.trainer")


def test_lists_name_real_modules_and_do_not_overlap():
    import importlib.util
    for p in CORE + LABS:
        top = p.split(".")[0]
        assert importlib.util.find_spec(top) is not None or top == "pinneapple_app", p
    assert not set(CORE) & set(LABS) and len(CORE_PP) == len(set(CORE_PP))


def _run(code, **env):
    return subprocess.run([sys.executable, "-W", "always::UserWarning", "-c", code], capture_output=True, text=True,
                          env=_env(**env), cwd=ROOT, timeout=600)


def test_a_labs_package_warns_once_and_can_be_silenced():
    code = "import pinneapple_condense, importlib; importlib.reload(pinneapple_condense)"
    r = _run(code)
    assert r.returncode == 0, r.stderr
    assert r.stderr.count("ExperimentalWarning") == 1 and "pinneapple_condense is experimental" in r.stderr
    q = _run(code, PINNEAPPLE_QUIET_EXPERIMENTAL="1")
    assert q.returncode == 0 and "ExperimentalWarning" not in q.stderr


def test_a_core_import_does_not_warn():
    r = _run("import pinneapple_design.geometry.brep")
    assert r.returncode == 0 and "ExperimentalWarning" not in r.stderr, r.stderr


def test_the_stability_package_is_light():
    r = subprocess.run([sys.executable, "-c", "import sys, pinneapple_stability; "
                        "print(sorted(m for m in sys.modules if m.startswith(('pinneapple', 'torch', 'numpy'))))"],
                       capture_output=True, text=True, env=_env(), cwd=ROOT, timeout=120)
    assert r.stdout.strip() == "['pinneapple_stability']", r.stdout
