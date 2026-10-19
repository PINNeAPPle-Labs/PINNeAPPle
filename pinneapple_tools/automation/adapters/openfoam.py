"""OpenFOAM through its command line: run the mesh script and the solver script of a case (CfdOF's ``Allmesh`` and
``Allrun``, or any command list), then read what happened from the logs with
:func:`pinneapple_data.simulation_metadata.openfoam.parse_log` (end reached, fatal errors, residual history,
convergence message, checkMesh quality).

* ``openfoam.mesh``: ``Allmesh`` in the mesh case, then ``checkMesh``. Checks: a polyMesh exists, checkMesh says
  "Mesh OK".
* ``openfoam.solve``: ``Allrun`` (or ``commands``) in the case. Checks: the solver reached ``End`` without a fatal
  error or floating-point exception, and either OpenFOAM's residual control stopped it ("SIMPLE solution converged")
  or every initial residual of the last iteration is below ``residual_tol``. Evidence: the residual history plot.

Compatibility with OpenFOAM v1912 (the Ubuntu 24.04 ``openfoam`` package; CfdOF targets newer releases), each
edit recorded as an action of the step:

* CfdOF picks the ``transformPoints`` syntax from ``FOAM_API``, which v1912 does not set, so the mesh would stay in
  millimetres: :func:`patch_cfdof_allmesh` adds the v-numbered (ESI) case to the test in ``Allmesh``.
* CfdOF's velocity inlet (``fixedNormalInletOutletVelocity``) did not hold the prescribed flux on v1912:
  :func:`patch_cfdof_inlet` writes it as ``fixedValue`` (ESI releases before v2006 only).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from ..record import Check, Evidence
from .base import ActionFailed, ActionResult, Adapter, StepContext, register

__all__ = ["OpenFOAMAdapter", "patch_cfdof_allmesh", "patch_cfdof_inlet", "residual_plot", "solver_log"]

_FOAM_API_TEST = "if [ ! -z $FOAM_API ] || [ $WM_PROJECT_VERSION -lt 9 ]"


def patch_cfdof_allmesh(mesh_case: str | Path) -> str | None:
    """Make CfdOF's ``Allmesh`` use the ``-scale`` syntax on v-numbered (ESI) releases that do not set FOAM_API.
    Returns a description of the edit, or None when the script needed none."""
    p = Path(mesh_case) / "Allmesh"
    text = p.read_text()
    if _FOAM_API_TEST not in text:
        return None
    p.write_text(text.replace(_FOAM_API_TEST, 'if [ ! -z $FOAM_API ] || [[ "$WM_PROJECT_VERSION" == v* ]] '
                                              "|| [ $WM_PROJECT_VERSION -lt 9 ]"))
    return ("Allmesh: added the v-numbered (ESI) release test before transformPoints, so OpenFOAM v1912, which "
            "does not set FOAM_API, scales the gmsh mesh from mm to m with -scale")


_MIXED_INLET = re.compile(r"(\n(\s*)(\w+)\s*\{)(?:\s*//[^\n]*)*\s*type\s+fixedNormalInletOutletVelocity;.*?normalVelocity\s*\{[^}]*?"
                          r"value\s+([^;]+);\s*\}\s*value\s+[^;]+;\s*\}", re.S)


def patch_cfdof_inlet(case: str | Path) -> list[str]:
    """Replace CfdOF's ``fixedNormalInletOutletVelocity`` inlets in ``0/U`` by ``fixedValue`` with the same normal
    velocity. On OpenFOAM v1912 the mixed condition is not assignable, so with the zeroGradient inlet pressure the
    pressure equation changed the inlet flux (6 % below U*A on the channel example). Returns the edits made."""
    p = Path(case) / "0" / "U"
    text = p.read_text()
    names = [m.group(3) for m in _MIXED_INLET.finditer(text)]
    if not names:
        return []
    p.write_text(_MIXED_INLET.sub(lambda m: f"{m.group(1)}\n{m.group(2)}    type            fixedValue;\n"
                                            f"{m.group(2)}    value           {m.group(4)};\n{m.group(2)}}}", text))
    return [f"0/U: inlet {n!r} fixedNormalInletOutletVelocity -> fixedValue with the same velocity, so OpenFOAM "
            "v1912 keeps the prescribed inflow" for n in names]


def _old_esi(version: str | None) -> bool:
    """ESI releases before v2006, the ones whose behaviour the CfdOF fixes above were measured on."""
    m = re.fullmatch(r"v(\d{4})", version or "")
    return bool(m) and int(m.group(1)) < 2006


def solver_log(case: Path) -> tuple[Path | None, dict[str, Any]]:
    """The solver log of a case (log.simpleFoam, log.foamRun, ...) and its parsed content."""
    from pinneapple_data.simulation_metadata.openfoam import parse_log
    skip = {"log.createPatch", "log.renumberMesh", "log.potentialFoam", "log.checkMesh", "log.gmshToFoam",
            "log.transformPoints", "log.foamToSurface", "log.decomposePar", "log.reconstructPar"}
    logs = [p for p in sorted(case.glob("log.*")) if p.name not in skip and not p.name.startswith("log.gmsh")]
    if not logs:
        return None, {}
    log = max(logs, key=lambda p: p.stat().st_mtime)
    return log, parse_log(log.read_text(errors="replace"))


def residual_plot(parsed: dict[str, Any], path: Path, title: str) -> Path | None:
    res = parsed.get("residuals") or {}
    if not res:
        return None
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=110)
    t = np.asarray(parsed.get("time_steps") or [], float)
    for name, vals in sorted(res.items()):
        v = np.asarray(vals, float)
        ax.semilogy(t[:len(v)] if len(t) >= len(v) else np.arange(len(v)), v, lw=1.3, label=name)
    ax.set_xlabel("iteration")
    ax.set_ylabel("initial residual")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def _case(params: dict[str, Any], ctx: StepContext, key: str = "case") -> Path:
    p = Path(params[key])
    p = p if p.is_absolute() else ctx.run_dir / p
    if not p.is_dir():
        raise ActionFailed(f"{key} {p} does not exist")
    return p


class OpenFOAMAdapter(Adapter):
    name = "openfoam"
    mode = "api"
    requires = ("openfoam",)
    actions = ("mesh", "solve")

    def _bash(self, ctx: StepContext, script: str) -> str:
        return f'source "{ctx.toolchain.foam_bashrc}" >/dev/null 2>&1; {script}'

    def do_mesh(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        from pinneapple_data.simulation_metadata.openfoam import parse_log
        res = ActionResult()
        mesh_case = _case(params, ctx, "mesh_case")
        fix = patch_cfdof_allmesh(mesh_case)
        if fix:
            res.actions.append(fix)
        ctx.run(self._bash(ctx, "bash ./Allmesh"), mesh_case, "log.Allmesh")
        poly = mesh_case / "constant" / "polyMesh"
        ok = (poly / "faces").is_file() or (poly / "faces.gz").is_file()
        res.checks.append(Check("polyMesh written", ok, str(poly) if ok else "no constant/polyMesh/faces",
                                source="Allmesh (gmsh + gmshToFoam)"))
        if not ok:
            return res
        ctx.run(self._bash(ctx, "checkMesh"), mesh_case, "log.checkMesh", check=False)
        q = parse_log((ctx.step_dir / "log.checkMesh").read_text(errors="replace")).get("mesh_quality", {})
        res.outputs["mesh"] = q
        res.checks.append(Check("checkMesh", q.get("status") == "Mesh OK",
                                f"{q.get('status')}: {int(q.get('cells', 0))} cells, max non-orthogonality "
                                f"{q.get('max_non_orthogonality')}°, max skewness {q.get('max_skewness')}",
                                value=q.get("cells"), source="pinneapple_data.simulation_metadata.openfoam.parse_log"))
        bm = re.search(r"bounding box \(([^)]*)\) \(([^)]*)\)", (ctx.step_dir / "log.checkMesh").read_text(), re.I)
        if bm:
            lo, hi = (np.array(s.split(), float) for s in bm.groups())
            res.outputs["bounding_box_m"] = [lo.tolist(), hi.tolist()]
            if params.get("expected_size_m"):
                want = np.asarray(params["expected_size_m"], float)
                good = bool(np.allclose(hi - lo, want, rtol=1e-3))
                res.checks.append(Check("mesh is in metres", good, f"size {np.round(hi - lo, 5).tolist()} m, "
                                        f"expected {want.tolist()} m", source="checkMesh bounding box"))
        res.artifacts.append(poly)
        return res

    def do_solve(self, params: dict[str, Any], ctx: StepContext) -> ActionResult:
        res = ActionResult()
        case = _case(params, ctx)
        version = ctx.toolchain.version("openfoam")
        if _old_esi(version) and params.get("compat", True):
            res.actions += patch_cfdof_inlet(case)
        commands = params.get("commands") or ["bash ./Allrun"]
        for i, c in enumerate(commands):
            ctx.run(self._bash(ctx, c), case, f"log.step{i + 1}", check=True)
        log, parsed = solver_log(case)
        if log is None:
            raise ActionFailed(f"no solver log in {case}")
        ctx.logs.append(ctx.rel(log))
        tol = float(params.get("residual_tol", 1e-4))
        last = {k: float(v[-1]) for k, v in (parsed.get("residuals") or {}).items() if len(v) and np.isfinite(v[-1])}
        ended = bool(parsed.get("ended")) and not parsed.get("fatal") and not parsed.get("floating_point_exception")
        res.checks.append(Check("solver finished", ended,
                                "reached End" if ended else (parsed.get("fatal") or "no End in the log"),
                                source="pinneapple_data.simulation_metadata.openfoam.parse_log"))
        n_it = len(parsed.get("time_steps") or [])
        conv_msg = parsed.get("converged_message")
        worst = max(last.values()) if last else float("nan")
        converged = bool(conv_msg) or (bool(last) and worst <= tol)
        res.checks.append(Check("solver converged", converged,
                                (f"{conv_msg}" if conv_msg else f"after {n_it} iterations") +
                                f"; largest last initial residual {worst:.3g} ({'<=' if worst <= tol else '>'} "
                                f"{tol:g})", value=worst, threshold=tol,
                                source="pinneapple_data.simulation_metadata.openfoam.parse_log"))
        res.outputs.update({"case": str(case), "solver": parsed.get("exec"), "openfoam_version": parsed.get("version"),
                            "iterations": n_it, "converged_message": conv_msg, "last_residuals": last,
                            "execution_time_s": parsed.get("execution_time_s")})
        png = residual_plot(parsed, ctx.step_dir / "residuals.png",
                            f"{parsed.get('exec') or 'solver'} residuals ({log.name})")
        if png:
            res.evidence.append(Evidence(ctx.rel(png), "plot", f"initial residuals per iteration, read from {log.name}"))
        times = sorted((p for p in case.iterdir() if p.is_dir() and re.fullmatch(r"\d+(\.\d+)?", p.name)
                        and p.name != "0"), key=lambda p: float(p.name))
        if times:
            res.outputs["last_time"] = times[-1].name
            res.artifacts.append(times[-1])
        res.artifacts.append(log)
        return res


register(OpenFOAMAdapter())
