"""The external-solver contract (PINNeAPPle#1476): one interface, the environment verdict that never raises, one
definition of "finished", and the same physics through two different backends into one schema.

Two kinds of proof here:

* with stand-ins (a shell script named ``ccx``, as ``tests/pinneapple_core/test_workflows.py`` already does), so the
  contract is exercised on a machine with no solver installed at all -- the environment verdict, the state machine,
  the cancellation and the divergence-vs-failure distinction;
* against the real binaries when the machine has them (``~/ccx-env/bin/ccx``, ``FOAM_BASHRC`` or the cask's
  ``/Volumes/OpenFOAM-*/etc/bashrc``): a cantilever through CalculiX, a cavity through OpenFOAM, and the same
  cantilever through CalculiX and through the in-process hexahedral FEM, compared in the one schema.
"""
from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

from pinneapple_simulation.external_solvers import (
    BACKENDS,
    SOLVERS,
    CaseUnsupported,
    EnvironmentReport,
    ExternalSolver,
    Requirement,
    RunUnusable,
    SolverCase,
    SolverState,
    SolverStatus,
    SolverUnavailable,
    check_all,
    classify_run,
    get_solver,
    solver_names,
)
from pinneapple_simulation.external_solvers.backends import (
    CalculiXBackend,
    OpenFOAMBackend,
    SolidFEMBackend,
)

HERE = Path(__file__).resolve().parent
FRD = HERE.parent / "apps" / "simulation_comparator" / "examples" / "calculix_cantilever_results.frd"

E, NU, L, B, H, P = 210e9, 0.3, 1.0, 0.1, 0.1, 1000.0


def timoshenko_tip() -> float:
    """Clamped-tip deflection of the 1 x 0.1 x 0.1 m steel cantilever under its 1 kN tip load (shear included)."""
    i = B * H ** 3 / 12
    g = E / (2 * (1 + NU))
    return P * L ** 3 / (3 * E * i) + P * L / (5 / 6 * g * B * H)


def _fake_bin(directory: Path, name: str, body: str) -> Path:
    p = directory / name
    p.write_text("#!/bin/sh\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return p


def _no_solvers(monkeypatch, tmp_path) -> None:
    """A machine with no solver: an empty PATH, no OpenFOAM bashrc anywhere, no WM_PROJECT_DIR, no FOAM_BASHRC."""
    empty = tmp_path / "empty-bin"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("PATH", str(empty))
    for var in ("FOAM_BASHRC", "WM_PROJECT_DIR", "FOAM_PROJECT_VERSION", "ANSYSLMD_LICENSE_FILE", "LM_LICENSE_FILE"):
        monkeypatch.delenv(var, raising=False)


# ---------------------------------------------------------------------------------------------- the interface itself
def test_every_registered_backend_answers_the_same_six_questions():
    assert solver_names() == sorted(BACKENDS)
    assert set(solver_names()) == {"ansys-fluent", "calculix", "fenicsx", "matlab", "openfoam", "solid-fem"}
    for name in solver_names():
        solver = get_solver(name)
        assert isinstance(solver, ExternalSolver)
        for method in ("check_environment", "prepare", "run", "status", "results", "cancel"):
            assert callable(getattr(solver, method)), (name, method)
        assert solver.check_environment() is solver.check_environment()      # asked twice, probed once
        assert SOLVERS[name] is BACKENDS[name]


def test_a_third_party_backend_only_fills_in_the_hooks(tmp_path):
    """The point of the contract: a new solver is one small class, not a reading of the previous one's code."""
    from pinneapple_simulation.external_solvers import register_solver

    class Hello(ExternalSolver):
        name = "hello"
        kind = "hello-test"
        converged_markers = (r"^hello$",)

        def _requirements(self):
            return (Requirement("executable", "hello", True, "always here"),)

        def prepare(self, case):
            self.case = case
            return self.workdir

        def _start(self, timeout):
            from pinneapple_simulation.external_solvers import RunHandle

            self.log_path.write_text("hello\n")
            return RunHandle.new(self.name, self.workdir, log_path=self.log_path,
                                 status=classify_run(backend=self.name, log="hello\n", returncode=0,
                                                     converged=self.converged_markers))

        def _classify(self, handle, returncode):
            return classify_run(backend=self.name, log=handle.read_log(), returncode=returncode,
                                converged=self.converged_markers)

        def _results(self):
            from pinneapple_simulation.external_solvers import SolverResults

            return SolverResults(backend=self.name, fields={"u": np.zeros((3, 3))}, scalars={"n": 3})

    register_solver(Hello)
    try:
        solver = get_solver("hello-test", workdir=tmp_path)
        assert solver.check_environment().ok
        solver.prepare(SolverCase.cantilever())
        assert SolverState.QUEUED is solver.status().state
        solver.run()
        assert solver.status().ok and solver.results().scalars["n"] == 3
    finally:
        SOLVERS.pop("hello-test", None)


# ---------------------------------------------------------------------------------------------- environment: never raises
def test_check_environment_reports_a_probe_that_blows_up_instead_of_raising():
    class Broken(ExternalSolver):
        name = "broken"
        kind = "broken-test"

        def _requirements(self):
            return (self._probe("executable", "kaboom", self._boom), Requirement("licence", "fine", True, "none"))

        def _boom(self):
            raise OSError("the launcher cannot even be asked")

        def prepare(self, case):
            return self.workdir

        def _start(self, timeout):
            raise NotImplementedError

        def _classify(self, handle, returncode):
            raise NotImplementedError

        def _results(self):
            raise NotImplementedError

    report = Broken().check_environment()
    assert not report.ok
    assert [(r.kind, r.ok) for r in report.missing] == [("executable", False)]
    assert "OSError: the launcher cannot even be asked" in report.explain()

    class Worse(Broken):
        """Not even the list of requirements can be built."""

        def _requirements(self):
            raise RuntimeError("nope")

    report = Worse().check_environment()
    assert not report.ok and "nope" in report.explain()


def test_absent_solver_is_a_diagnosis_not_an_exception_in_the_middle_of_a_run(tmp_path, monkeypatch):
    """The case that hurt: the solver is not there, and the caller gets a paragraph, not a traceback."""
    _no_solvers(monkeypatch, tmp_path)
    report = CalculiXBackend().check_environment()
    assert not report.ok
    assert "not on PATH" in report.why()
    assert report.version is None
    assert "install CalculiX" in report.explain()
    with pytest.raises(SolverUnavailable) as excinfo:
        report.raise_if_missing()
    assert "CalculiX (ccx) not on PATH" in str(excinfo.value)

    # ... and the OpenFOAM half: the binaries are on PATH but the environment is not loaded
    _fake_bin(tmp_path / "empty-bin", "blockMesh", "exit 0\n")
    _fake_bin(tmp_path / "empty-bin", "icoFoam", "exit 0\n")
    foam = OpenFOAMBackend(solver="icoFoam", patterns=()).check_environment()
    assert not foam.ok
    assert "environment is not loaded" in foam.why()


def test_openfoam_environment_verdict_uses_the_bashrc_it_finds(tmp_path, monkeypatch):
    """A bashrc on the usual paths makes the environment ready even with an empty PATH."""
    from pinneapple_simulation.external_solvers.openfoam.environment import probe

    release = tmp_path / "OpenFOAM-v2606"
    (release / "etc").mkdir(parents=True)
    (release / "etc" / "bashrc").write_text("export WM_PROJECT_DIR=x\n")
    _no_solvers(monkeypatch, tmp_path)
    with_bashrc = OpenFOAMBackend(solver="icoFoam", patterns=(str(release / "etc" / "bashrc"),))
    assert with_bashrc.bashrc == str(release / "etc" / "bashrc")
    assert "v2606" in (with_bashrc.check_environment().version or "")      # read off the release directory name
    # a bashrc that does not load is reported with what bash said
    broken = probe(str(tmp_path / "missing" / "bashrc"))
    assert broken["sourced"] is False and broken["error"]


def test_the_workflow_adapters_share_the_contracts_verdict(tmp_path, monkeypatch):
    """One verdict for the step surface and the library surface; the path list cannot drift between them."""
    from pinneapple_core.workflows import adapters
    from pinneapple_simulation.external_solvers.openfoam.environment import FOAM_BASHRCS

    assert adapters._FOAM_BASHRCS == FOAM_BASHRCS
    _no_solvers(monkeypatch, tmp_path)
    monkeypatch.setattr(adapters, "_FOAM_BASHRCS", ())
    ok, why = adapters.CalculiXAdapter().available()
    assert not ok and "not on PATH" in why
    ok, why = adapters.OpenFOAMAdapter().available()
    assert not ok and "not on PATH" in why          # its binaries are not on the empty PATH either


# ---------------------------------------------------------------------------------------------- "finished", one definition
def _status(**kw) -> SolverStatus:
    return classify_run(backend="test", **kw)


def test_diverged_is_not_indistinguishable_from_failed():
    diverged = _status(log="Time = 1\nMin(p) = nan Max(p) = nan\nEnd\n", returncode=1,
                       converged=(r"^End$",), divergence=(r"Min\([^)]*\) = nan",))
    failed = _status(log="FOAM FATAL ERROR: cannot open file\n", returncode=1, converged=(r"^End$",),
                     divergence=(r"Min\([^)]*\) = nan",))
    assert diverged.state is SolverState.DIVERGED and failed.state is SolverState.FAILED
    assert diverged.state is not failed.state
    assert "Min(p) = nan" in diverged.evidence[0] and "FOAM FATAL" in failed.evidence[0]
    # the divergence marker outranks both the exit code and the solver's own end marker
    assert _status(log="Min(U) = inf\nEnd\n", returncode=0, converged=(r"^End$",),
                   divergence=(r"Min\([^)]*\) = inf",)).state is SolverState.DIVERGED
    # a clean exit without the end marker is a failure, not an assumed success
    assert _status(log="stopped\n", returncode=0, converged=(r"^End$",)).state is SolverState.FAILED
    # an artefact the solver had to write, missing: named in the reason
    status = _status(log="End\n", returncode=0, converged=(r"^End$",), artefacts={"case.frd": False})
    assert status.state is SolverState.FAILED and "case.frd" in status.reason
    # a backend with no end marker of its own (a script) is judged by its exit code and its artefact
    script = _status(log="", returncode=0, artefacts={"out.mat": True})
    assert script.state is SolverState.CONVERGED and "out.mat" in script.evidence[0]


def test_status_and_results_say_why_they_cannot_be_used(tmp_path, monkeypatch):
    _no_solvers(monkeypatch, tmp_path)
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_bin(bindir, "ccx", 'echo "*ERROR: no convergence in increment 1"\nexit 1\n')
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    solver = get_solver("calculix", workdir=tmp_path / "run")
    solver.prepare(SolverCase.cantilever())
    solver.run()
    status = solver.wait()
    assert status.state is SolverState.DIVERGED, status.explain()
    assert "*ERROR: no convergence" in status.evidence[0]
    with pytest.raises(RunUnusable) as excinfo:
        solver.results()
    assert "diverged" in str(excinfo.value) and "no convergence" in str(excinfo.value)


def test_cancel_is_idempotent_and_a_running_solver_reports_running(tmp_path, monkeypatch):
    if os.name == "nt":
        pytest.skip("a shell script as a stand-in solver")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_bin(bindir, "ccx", "sleep 30\n")
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    solver = get_solver("calculix", workdir=tmp_path / "run")
    solver.prepare(SolverCase.cantilever())
    handle = solver.run()
    assert solver.status().state is SolverState.RUNNING
    first = solver.cancel()
    second = solver.cancel()
    assert first.state is SolverState.CANCELLED and second.state is SolverState.CANCELLED
    assert first.reason == second.reason and not handle.running
    assert solver.cancel().state is SolverState.CANCELLED          # still idempotent, no exception
    with pytest.raises(RunUnusable):
        solver.results()


def test_a_case_the_backend_cannot_express_is_named(tmp_path):
    solver = get_solver("calculix", workdir=tmp_path)
    with pytest.raises(CaseUnsupported) as excinfo:
        solver.prepare(SolverCase.plane_cantilever())
    assert "3-D" in str(excinfo.value)
    with pytest.raises(CaseUnsupported):
        get_solver("openfoam", workdir=tmp_path).prepare(SolverCase.cantilever())
    with pytest.raises(CaseUnsupported):
        get_solver("fenicsx", workdir=tmp_path).prepare(SolverCase.cantilever())      # the bridge is plane only


def test_check_all_reports_every_backend_without_raising(tmp_path, monkeypatch):
    _no_solvers(monkeypatch, tmp_path)
    reports = check_all()
    assert set(reports) == set(solver_names())
    assert all(isinstance(r, EnvironmentReport) for r in reports.values())
    assert not reports["calculix"].ok and not reports["fenicsx"].ok and not reports["matlab"].ok
    assert sum(1 for r in reports.values() if not r.ok) >= 3


# ---------------------------------------------------------------------------------------------- the same physics, two backends
def _run(name: str, case: SolverCase, workdir: Path):
    solver = get_solver(name, workdir=workdir)
    solver.prepare(case)
    solver.run()
    return solver, solver.wait()


def test_two_different_backends_answer_the_same_case_in_one_schema(tmp_path, monkeypatch):
    """The lock on the contract: one neutral case, two solvers, one schema, provenance naming each.

    CalculiX runs as a stand-in here (so the test exists on every machine); the next test runs the same two backends
    for real, with the numbers compared.
    """
    if os.name == "nt":
        pytest.skip("a shell script as a stand-in solver")
    assert FRD.is_file(), FRD
    assert "DISP" in FRD.read_text(errors="replace")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    _fake_bin(bindir, "ccx", f'test -f "$2.inp" || exit 2\ncp "{FRD}" "$2.frd"\necho "Job finished"\n')
    monkeypatch.setenv("PATH", f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}")
    case = SolverCase.cantilever(P=P, nx=8, ny=2, nz=2)

    backends = ["calculix", "solid-fem"]
    results = {name: _run(name, case, tmp_path / name)[0].results() for name in backends}

    assert [r.backend for r in results.values()] == [get_solver(n).name for n in backends]
    for name, r in results.items():
        assert r.fields["U"].ndim == 2 and r.fields["U"].shape[1] >= 3, name
        assert r.scalars["max_displacement"] > 0, name
        upd = r.to_upd()
        assert upd.provenance["backend"] == get_solver(name).name      # provenance names the backend it ran in
        assert upd.provenance["backend_version"] == r.version
        assert upd.state["U"].shape[0] == r.fields["U"].shape[0]
    assert results["calculix"].version != results["solid-fem"].version            # each names its own version
    assert set(results["solid-fem"].fields) >= {"U", "von_mises"}                # same schema, its own extra fields


@pytest.mark.skipif(sys.platform == "win32", reason="shell and ccx")
def test_real_ccx_and_the_in_process_fem_agree_on_the_cantilever(tmp_path):
    """Acceptance of the issue: the same physics through two backends, in the same schema, with the numbers checked
    against beam theory rather than against each other only."""
    from pinneapple_simulation.external_solvers.calculix import ccx_backend

    if ccx_backend() is None:
        pytest.skip("CalculiX not available (no ccx, no Docker)")
    case = SolverCase.cantilever(P=P)
    ccx, ccx_status = _run("calculix", case, tmp_path / "ccx")
    fem, fem_status = _run("solid-fem", case, tmp_path / "fem")
    assert ccx_status.ok, ccx_status.explain()
    assert fem_status.ok, fem_status.explain()
    a, b = ccx.results(), fem.results()
    assert a.backend == "calculix" and b.backend == "pinneapple-solid-fem"
    assert a.version and b.version
    tip_a, tip_b = a.scalars["max_displacement"], b.scalars["max_displacement"]
    assert abs(tip_a / timoshenko_tip() - 1) < 0.02                 # CalculiX, C3D20R
    assert abs(tip_b / timoshenko_tip() - 1) < 0.02                 # in-process, C3D8I equivalent
    assert abs(tip_a / tip_b - 1) < 0.02                            # and they agree with each other
    assert a.coords and b.coords                                  # the same box, from two meshers
    assert float(a.coords["x"].max()) == pytest.approx(float(b.coords["x"].max()))


def test_real_openfoam_runs_a_case_through_the_contract(tmp_path):
    """The present path for the other family: a real icoFoam run, its log parsed and its fields read back."""
    if OpenFOAMBackend(solver="icoFoam").check_environment() is None:      # pragma: no cover - defensive
        pytest.skip("OpenFOAM not available")
    if not OpenFOAMBackend(solver="icoFoam").check_environment().ok:
        pytest.skip("OpenFOAM not available")
    from pinneapple_core.workflows.templates import CASES_DIR

    case = SolverCase.case_directory(CASES_DIR / "cavity", solver="icoFoam",
                                     edits=[{"file": "system/controlDict", "key": "endTime", "value": "1"}])
    solver, status = _run("openfoam", case, tmp_path / "foam")
    assert status.ok, status.explain()
    assert status.progress["steps"] > 0 and status.progress["courant_max"] is not None
    results = solver.results()
    assert results.scalars["last_time"] == pytest.approx(1.0)
    assert set(results.fields) == {"p", "U"} and results.fields["U"].shape[1] == 3
    assert results.provenance["solver"] == "icoFoam"
    assert results.to_upd().provenance["backend"] == "openfoam"


def test_fenicsx_reports_its_absence_and_names_the_fix_on_a_machine_without_it(tmp_path):
    """The Python-module family: no executable to look for, and no licence either -- what must be missing is the module,
    and the fix names the package that provides it."""
    report = get_solver("fenicsx", workdir=tmp_path).check_environment()
    if report.ok:
        assert report.version and "FEniCS" in get_solver("fenicsx", workdir=tmp_path).results.__doc__
        pytest.skip("FEniCS is installed here; the plane-stress case is covered by the bridge's own tests")
    assert any(r.kind == "module" and not r.ok for r in report.missing)
    assert "fenics-dolfinx" in report.explain()


def test_ansys_reports_the_licence_it_cannot_see(tmp_path, monkeypatch):
    monkeypatch.delenv("ANSYSLMD_LICENSE_FILE", raising=False)
    monkeypatch.delenv("LM_LICENSE_FILE", raising=False)
    report = get_solver("ansys-fluent", workdir=tmp_path, environ={"PATH": os.environ.get("PATH", "")}).check_environment()
    licence = next(r for r in report.requirements if r.kind == "licence")
    assert not licence.ok and "ANSYSLMD_LICENSE_FILE" in licence.detail and "licence" in licence.fix
    monkeypatch.setenv("ANSYSLMD_LICENSE_FILE", "1055@licence-server")
    set_licence = get_solver("ansys-fluent", workdir=tmp_path).check_environment()
    assert next(r for r in set_licence.requirements if r.kind == "licence").ok


def test_the_in_process_backend_needs_nothing_and_still_names_a_version(tmp_path):
    report = SolidFEMBackend(workdir=tmp_path).check_environment()
    assert report.ok and report.version
