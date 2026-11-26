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
    conduction_profile,
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


def _version_tuple(text: str):
    import re

    m = re.match(r"\s*(\d+)\.(\d+)", str(text or ""))
    return (int(m.group(1)), int(m.group(2))) if m else None


def _module_requirement_available(names):
    for name in names:
        try:
            __import__(name)
        except Exception:                             # noqa: BLE001 - absent is the answer
            continue
        return name
    return None


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
    """One call, every backend, an answer for each: what is missing here is a sentence with a fix, never an exception,
    and a backend that is ready says so without an empty "why"."""
    _no_solvers(monkeypatch, tmp_path)
    reports = check_all()
    assert set(reports) == set(solver_names())
    for name, report in reports.items():
        assert isinstance(report, EnvironmentReport)
        assert report.ok == (not report.missing), name
        assert bool(report.why()) != report.ok, name          # a missing item always explains itself
        assert report.explain().startswith(get_solver(name).name)      # it names the backend, not the registry key
    assert sum(1 for r in reports.values() if not r.ok) >= 2    # no ccx, no matlab, no licence on this box

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


def test_fenicsx_reports_what_it_lacks_and_names_the_fix(tmp_path):
    """The Python-module family: no executable to look for and no licence either. What can be missing is the module --
    and, once it is installed, the version: DOLFINx 0.10 removed the API the bridge speaks, so "installed" and
    "usable" are different answers and the report has to tell them apart."""
    report = get_solver("fenicsx", workdir=tmp_path).check_environment()
    if report.ok:
        assert report.version and _version_tuple(report.version) <= (0, 9)
        pytest.skip("FEniCS 0.8/0.9 is installed here; the runs are in the closed-form test below")
    kinds = {r.kind for r in report.missing}
    assert kinds <= {"module", "version"}
    assert "fenics-dolfinx" in report.explain()
    if "version" in kinds:
        # the module is there, the API is not: the fix pins the release, it does not say "install FEniCS"
        version = next(r for r in report.missing if r.kind == "version")
        assert "outside what this bridge speaks" in version.detail and "fenics-dolfinx=0.9" in version.fix


def test_fenicsx_version_requirement_follows_the_bridge_range(tmp_path):
    """A version requirement that cannot be satisfied by ``pip install`` alone is the point: the range is read off the
    bridge's own calls, and the report says which releases it speaks."""
    from pinneapple_simulation.external_solvers.backends import FEniCSxBackend

    backend = FEniCSxBackend(workdir=tmp_path)
    found = _module_requirement_available(("dolfinx", "dolfin"))
    if found is None:
        pytest.skip("no FEniCS here")
    released = _version_tuple(next(r.detail for r in backend.check_environment().requirements
                                   if r.kind == "module" and r.ok))
    ok = backend.check_environment().ok
    assert ok == (released is None or backend.DOLFINX_RANGE[0] <= released <= backend.DOLFINX_RANGE[1])


def test_a_too_new_dolfinx_is_a_missing_version_not_a_crash(tmp_path, monkeypatch):
    """The gate itself, on any machine: when the installed release is the one the bridge cannot speak (0.10 changed
    ``FunctionSpace`` and ``dolfinx.fem.petsc``), the answer is a missing *version* with the pin in the fix -- not a
    ``TypeError`` raised from inside ``prepare()`` a hundred cases into a campaign."""
    from pinneapple_simulation.external_solvers import backends

    def newer(name, *, fix="", kind="module"):
        return Requirement(kind, name, True, "0.11.0", "")

    monkeypatch.setattr(backends, "_module_requirement", newer)
    report = backends.FEniCSxBackend(workdir=tmp_path).check_environment()
    assert not report.ok
    version = next(r for r in report.missing if r.kind == "version")
    assert "0.11.0" in version.detail and "outside what this bridge speaks" in version.detail
    assert "fenics-dolfinx=0.9" in version.fix
    assert report.version is None                      # an unusable release is not a version for provenance


def test_fenicsx_plane_case_gives_every_value_a_point(tmp_path):
    """The plane-stress path returns its field *with* the coordinates of its degrees of freedom, in the same order --
    a vector space lists each node once while the field lists its components, so the contract repeats the row per
    component. The number is not compared with beam theory on purpose: P1 triangles lock, and the closed-form check is
    the conduction case below."""
    solver = get_solver("fenicsx", workdir=tmp_path)
    if not solver.check_environment().ok:
        pytest.skip("FEniCSx not usable here")
    case = SolverCase.plane_cantilever(L=1.0, h=0.1, b=0.1, q=100.0, nx=16, ny=4)
    solver.prepare(case)
    solver.run()
    assert solver.wait().ok, solver.status().explain()
    results = solver.results()
    u = np.asarray(results.fields["u"])
    x = np.asarray(results.coords["x"])
    assert u.size == x.size                                  # one point per component entry, in the same order
    assert np.asarray(results.fields["uy"]).size * 2 == u.size   # the components are interleaved per node
    assert np.asarray(results.coords["y"]).size == u.size
    assert results.scalars["max_deflection"] > 0


def test_fenicsx_runs_a_closed_form_case_through_the_contract(tmp_path):
    """#2477: the FEniCSx backend exercised numerically, against an answer that is known rather than against another
    solver. ``-k T'' = source`` on a rectangle with two held sides has the closed form ``conduction_profile``: a linear
    exact solution (no source) is *in* the P1 space, so it comes back to solver precision, and a sourced one (whose
    exact solution is quadratic, so outside the space) is reproduced at the nodes to ~1e-6 of the range at every
    refinement -- P1 is nodal-superconvergent for this configuration, and what remains lives inside the elements, where
    the degrees of freedom cannot see it."""
    import numpy as np

    solver = get_solver("fenicsx", workdir=tmp_path)
    report = solver.check_environment()
    if not report.ok:
        pytest.skip(f"FEniCSx not usable here: {report.why()[:120]}")

    linear = SolverCase.conduction(L=1.0, H=0.5, k=2.0, source=0.0, T_left=0.0, T_right=100.0, nx=8, ny=4)
    solver.prepare(linear)
    solver.run()
    status = solver.wait()
    assert status.ok, status.explain()
    results = solver.results()
    assert "T" in results.fields and results.coords, results.summary()
    x = np.asarray(results.coords["x"])
    exact = conduction_profile(linear, x)
    error = float(np.abs(np.asarray(results.fields["T"]) - exact).max())
    assert error / 100.0 < 1e-5, error                # 1.8e-6 of the range here: the Krylov tolerance, not the mesh
    assert float(np.abs(np.asarray(results.fields["T"]).max() - 100.0)) / 100.0 < 1e-5

    for nx in (8, 16, 32):
        case = SolverCase.conduction(L=1.0, H=0.5, k=1.0, source=1.0, T_left=0.0, T_right=1.0, nx=nx, ny=nx // 2)
        s = get_solver("fenicsx", workdir=tmp_path / f"src{nx}")
        s.prepare(case)
        s.run()
        assert s.wait().ok, s.status().explain()
        r = s.results()
        x = np.asarray(r.coords["x"])
        values = np.asarray(r.fields["T"])
        assert values.size == x.size
        assert float(np.abs(values - conduction_profile(case, x)).max()) < 1e-4, nx   # the closed form, on the range
        assert float(values.max()) == pytest.approx(float(conduction_profile(case, 1.0)), abs=1e-4)


def test_comsol_is_out_of_scope_and_the_absence_is_explicit():
    """#2477, the decision: COMSOL has no backend and the interface does not pretend otherwise. Asking for one names the
    six that exist instead of failing somewhere inside a run, and the scope is written down in the contract and in the
    guide, not only in an issue that closes."""
    from pinneapple_simulation.external_solvers import contract

    assert "comsol" not in solver_names()
    with pytest.raises(KeyError) as excinfo:
        get_solver("comsol")
    assert "openfoam" in str(excinfo.value) and "solid-fem" in str(excinfo.value)   # it lists what does exist
    assert "COMSOL" in (contract.__doc__ or "")
    guide = HERE.parent / "docs" / "guides" / "external_solver_contract.md"
    assert guide.is_file() and "COMSOL is out of scope" in guide.read_text()


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
