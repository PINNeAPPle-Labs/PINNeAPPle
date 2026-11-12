"""The study: solve a passage, drive it onto the pressure drop asked for, check it, and write it up.

The pieces are deliberately separate from the geometry, so the same four calls work for a valve, an
orifice and a nozzle:

:func:`solve_once`
    write the case, run ``checkMesh`` and ``rhoSimpleFoam``, read the result and its CSV tables.
:func:`solve_to_pressure_drop`
    the outer loop. The inlet is driven by mass flow, so the inlet pressure is an **output**; this
    iterates the flow onto the pressure the caller asked for (``mdot ~ sqrt(dP)`` is the update, and
    it converges in two or three solves on a throttling passage).
:func:`run_study`
    the nominal case plus an optional sweep of the passage parameter (the valve's opening), the
    checks, the figures, the 3D scene, the video, and everything the report and the Flow node need.

Why the outer loop exists
-------------------------
Imposing static pressure at both ends of a small domain makes the first pressure correction take
the whole start-up imbalance: on this 73 cm3 valve the pressure jumped to 3.3e8 Pa in one iteration
and the run died. With the flow imposed and the inlet pressure read back, the same case converges
in about a thousand iterations, and the answer is the flow at the pressure drop asked for — which
is the question a valve is bought to answer.
"""
from __future__ import annotations

import csv
import math
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .case import RunResult, estimate_mass_flow, run_case, solver_for, write_case
from .checks import critical_pressure_ratio, run_checks
from .passage import PassageSpec
from .physics import Conditions, Gas, resolve_gas
from .post import FlowResult, read_case, write_tables

__all__ = ["StudySpec", "CaseOutcome", "solve_once", "solve_to_pressure_drop", "run_study",
           "assemble_study", "initial_mass_flow"]


def initial_mass_flow(spec: StudySpec, p_in_target_bar: float | None = None) -> float:
    """A first guess from ``mdot = Cd A sqrt(2 rho dP)`` (:func:`.case.estimate_mass_flow`)."""
    cond = Conditions(**{**spec.conditions.__dict__,
                         "p_in_target_bar": p_in_target_bar if p_in_target_bar is not None
                         else spec.conditions.p_in_target_bar})
    return estimate_mass_flow(spec.passage, spec.resolved_gas, cond)


@dataclass
class StudySpec:
    """A passage, a gas and the conditions — everything a run needs, no geometry in the code."""

    passage: PassageSpec
    gas: Gas | str = "methane"
    conditions: Conditions = field(default_factory=Conditions)
    title: str = ""
    notes: tuple[str, ...] = ()
    render: bool = True
    video: bool = True
    check_solver: bool = True

    @property
    def resolved_gas(self) -> Gas:
        return resolve_gas(self.gas)


@dataclass
class CaseOutcome:
    value: float
    passage: PassageSpec
    case_dir: str
    run: RunResult
    result: FlowResult
    fields: dict[str, Any]
    tables: list[str] = field(default_factory=list)
    solves: int = 1
    note: str = ""

    def as_row(self) -> dict[str, Any]:
        d = self.result.as_dict()
        d.update({"segundos": round(self.run.elapsed_s, 1), "iteracoes": self.run.n_iter,
                  "convergiu": self.run.converged, "solucoes_do_ciclo": self.solves})
        return d


def solve_once(spec: StudySpec, folder: str, *, timeout: float = 7200.0,
               restart_from: str | None = None) -> CaseOutcome:
    """One case: write, run, read, tabulate."""
    gas = spec.resolved_gas
    write_case(spec.passage, folder, gas, spec.conditions, restart_from=restart_from)
    run = run_case(folder, timeout=timeout, check=spec.check_solver, solver=solver_for(spec.conditions))
    result, fields = read_case(folder, spec.passage, spec.conditions, gas)
    tables = write_tables(os.path.join(folder, "csv"), result, fields)
    value = float(spec.passage.meta.get("parameter_value", 0.0))
    return CaseOutcome(value=value, passage=spec.passage, case_dir=folder, run=run, result=result,
                       fields=fields, tables=tables, note=run.note)


def solve_to_pressure_drop(spec: StudySpec, folder: str, *, tol: float = 0.01, max_solves: int = 5,
                           timeout: float = 7200.0, log: Callable[[str], None] | None = None) -> CaseOutcome:
    """Iterate the mass flow until the inlet pressure reaches ``conditions.p_in_target_bar``.

    ``mdot ~ sqrt(dP)`` on a throttling passage, so the update is ``m *= sqrt(target/measured)``,
    damped and clipped to a factor of two per step. Each solve is a fresh case, which keeps the
    result reproducible from the folder alone.
    """
    target = spec.conditions.p_in_target_bar
    say = log or (lambda m: None)
    if target is None:
        return solve_once(spec, folder, timeout=timeout)
    if spec.conditions.transient:
        # In physical time the inlet carries total pressure and the outlet static pressure, so the
        # flow is what the run measures. There is nothing to iterate: one march, one answer.
        say(f"  transiente: {spec.conditions.end_time_transits:g} travessias acústicas, "
            f"p0 = {target:g} bar na entrada, {spec.conditions.p_out_bar:g} bar na saída")
        return solve_once(spec, folder, timeout=timeout)

    best: CaseOutcome | None = None
    m = spec.conditions.mass_flow_kg_s or initial_mass_flow(spec, target)
    previous: str | None = None
    for k in range(max_solves):
        # first order to get there, the requested scheme to finish: the classic continuation, and it
        # is what lets a strongly throttled case reach the second-order answer at all
        scheme = "upwind" if k == 0 else spec.conditions.scheme
        cond = Conditions(**{**spec.conditions.__dict__, "mass_flow_kg_s": m, "scheme": scheme})
        cond.p_in_target_bar = target
        attempt = StudySpec(passage=spec.passage, gas=spec.gas, conditions=cond, title=spec.title,
                            notes=spec.notes, render=spec.render, video=spec.video,
                            check_solver=spec.check_solver)
        sub = os.path.join(folder, f"solve_{k:02d}") if max_solves > 1 else folder
        out = solve_once(attempt, sub, timeout=timeout, restart_from=previous)
        previous = sub
        out.solves = k + 1
        p_in_bar = out.result.p_in_pa / 1e5
        if not (math.isfinite(p_in_bar) and math.isfinite(out.result.mdot_kg_s)
                and out.result.mdot_kg_s > 0.0):
            # the run did not hold together: back off rather than trust a number that is not one.
            # A plateau is *not* this case: a bounded field with a residual that stops falling still
            # gives a usable mean (``sensibleEnthalpy`` runs on this geometry do exactly that), and it
            # is the ``converged`` check that says so, not the loop throwing the answer away.
            say(f"  ciclo {k + 1}: mdot {m:.5f} kg/s não convergiu ({out.run.note or 'sem resultado'}); "
                f"reduzindo a vazão")
            m *= 0.4
            continue
        err = (p_in_bar - target) / target
        say(f"  ciclo {k + 1}: mdot {m:.5f} kg/s -> p_entrada {p_in_bar:.3f} bar "
            f"(alvo {target:.3f}, erro {err * 100:+.2f} %)")
        if best is None or abs(err) < abs((best.result.p_in_pa / 1e5 - target) / target):
            best = out
        if abs(err) <= tol:
            best = out
            break
        # on a throttling passage the flow goes as sqrt(dP), so the update is the full square-root
        # step. The first guess comes from a Cd of 0.6 and the real coefficient of a 0.5 mm curtain is
        # nearer 0.1, i.e. the first solve lands an order of magnitude low -- clipping the step to a
        # factor of two (the safe-looking choice) then needs four passes instead of two.
        dp_measured = max(out.result.dp_pa, 1.0)
        dp_target = max((target - spec.conditions.p_out_bar) * 1e5, 1.0)
        factor = math.sqrt(dp_target / dp_measured)
        m = float(np.clip(m * float(np.clip(factor, 0.3, 8.0)), 1e-4, 50.0))
    if best is None:
        raise RuntimeError(f"no solve of {spec.passage.title} produced a usable field at {target:g} bar")
    return best


def run_study(spec: StudySpec, folder: str, *, sweep_values: list[float] | None = None,
              builder: Callable[..., PassageSpec] | None = None, timeout: float = 7200.0,
              jobs: int = 1, max_solves: int = 5,
              log: Callable[[str], None] | None = None) -> dict[str, Any]:
    """The nominal case, an optional sweep, the checks, the figures and the pictures."""
    say = log or (lambda m: None)
    os.makedirs(folder, exist_ok=True)
    kind = spec.passage.meta.get("kind", "passage")

    say(f"[{kind}] caso nominal")
    nominal = solve_to_pressure_drop(spec, os.path.join(folder, "nominal"), timeout=timeout,
                                     max_solves=max_solves, log=say)

    curve: list[CaseOutcome] = []
    if sweep_values:
        if builder is None:
            from .studies import STUDIES
            build = STUDIES[kind]
            param = spec.passage.meta.get("parameter_name", "lift_mm")
            builder = lambda v: build(**{param: v})          # noqa: E731
        def one(v: float) -> CaseOutcome:
            say(f"[{kind}] abertura {v:g}")
            passage = builder(v)
            cond = Conditions(**spec.conditions.__dict__)
            cond.mass_flow_kg_s = spec.conditions.mass_flow_kg_s
            sub = StudySpec(passage=passage, gas=spec.gas, conditions=cond, title=passage.title,
                            render=False, video=False, check_solver=spec.check_solver)
            return solve_to_pressure_drop(sub, os.path.join(folder, f"v{v:g}"), timeout=timeout,
                                          max_solves=max_solves, log=say)

        if jobs > 1:
            # each case is its own OpenFOAM subprocess, so threads are enough and each solver keeps a core
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=jobs) as pool:
                curve = list(pool.map(one, sweep_values))
        else:
            curve = [one(v) for v in sweep_values]

    return assemble_study(spec, folder, nominal, curve)


def assemble_study(spec: StudySpec, folder: str, nominal: CaseOutcome,
                   curve: list[CaseOutcome]) -> dict[str, Any]:
    """The report of a study whose cases were already solved — the same dict :func:`run_study` returns.

    Separate from :func:`run_study` so a caller can solve the cases its own way (in parallel, or in
    several machines) and still get the checks, the figures, the pictures and the file list.
    """
    gas = spec.resolved_gas
    kind = spec.passage.meta.get("kind", "passage")
    aggregate = _write_curve(folder, curve)
    figures = _figures(folder, nominal, curve, gas)

    pictures: dict[str, str] = {}
    if spec.render or spec.video:
        pictures = _pictures(folder, nominal, spec)

    checks = run_checks(nominal.result, nominal.fields, nominal.passage, gas, spec.conditions,
                        nominal.run, cfd_reported=spec.passage.meta.get("cfd_reported"))
    passed = sum(1 for c in checks if c["status"] == "pass")
    failed = [c for c in checks if c["status"] == "fail"]
    inconclusive = [c for c in checks if c["status"] == "inconclusive"]

    r = nominal.result
    summary = (f"{gas.name}, {spec.conditions.p_out_bar:g} bar na saída, {r.p_in_pa / 1e5:.2f} bar na entrada: "
               f"{r.mdot_kg_s:.4f} kg/s ({r.volume_flow_m3_h:.2f} m³/h), Kv {r.kv:.4f}, "
               f"ΔT {r.dT_K:+.2f} K, Ma_max {r.ma_max:.3f}; "
               f"{passed}/{len(checks)} checagens passaram" +
               (f", {len(failed)} falharam" if failed else "") +
               (f", {len(inconclusive)} sem referência" if inconclusive else ""))

    return {
        "kind": kind,
        "title": spec.title or spec.passage.title,
        "summary": summary,
        "nominal": nominal.as_row(),
        "curve": [c.as_row() for c in curve],
        "checks": checks,
        "assumptions": list(spec.notes) + _assumptions(spec, r),
        "files": [*nominal.tables, *aggregate, *figures, *pictures.values()],
        "figures": figures,
        "pictures": pictures,
        "gas": {"name": gas.name, "rho_in": round(r.rho_in, 3),
                "rho_ideal": round(r.p_in_pa / (gas.Rs * spec.conditions.T_in_K), 3),
                "critical_pressure_ratio": round(critical_pressure_ratio(gas), 4)},
        "conditions": spec.conditions.as_dict(),
        "passage": spec.passage.summary(),
        "decision": {"verdict": "usar com as ressalvas listadas" if failed else
                                ("usar" if not inconclusive else "usar, com uma checagem sem referência"),
                     "failed": [c["code"] for c in failed],
                     "inconclusive": [c["code"] for c in inconclusive]},
    }


def _assumptions(spec: StudySpec, r: FlowResult) -> list[str]:
    p = spec.passage
    out = [
        f"o domínio é um setor de {p.sector_deg:g}° do sólido de revolução; as duas faces de corte são "
        f"planos de simetria, o que é exato para esta geometria, e as vazões integrais são multiplicadas "
        f"por {360.0 / p.sector_deg:g} para o corpo inteiro",
        f"o eixo é representado por um núcleo de raio {p.r_core:g} mm com condição de simetria",
        "a passagem é tratada como adiabática (parede com fluxo de calor zero), como no documento de "
        "referência; não há troca de calor com o ambiente",
        f"o escoamento é permanente e a turbulência é RANS {spec.conditions.turbulence} com funções de "
        "parede; a camada limite não é resolvida até a parede",
        f"a vazão é imposta e a pressão de entrada é o resultado: o ciclo externo a levou a "
        f"{spec.conditions.p_in_target_bar or r.p_in_pa / 1e5:.3f} bar",
    ]
    return out


def _write_curve(folder: str, curve: list[CaseOutcome]) -> list[str]:
    if not curve:
        return []
    path = os.path.join(folder, "curva.csv")
    rows = [c.as_row() for c in curve]
    keys = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    return [path]


def _figures(folder: str, nominal: CaseOutcome, curve: list[CaseOutcome], gas: Gas) -> list[str]:
    """Matplotlib figures: the axial profile, the cut plane, and the curve over the parameter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .post import axial_profile

    out: list[str] = []
    prof = axial_profile(nominal.fields)
    if len(prof):
        fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
        ax[0].plot(prof[:, 0], prof[:, 1], lw=2, color="#1f77b4")
        ax[0].set_xlabel("x [mm]"), ax[0].set_ylabel("p estática [bar]")
        ax[0].set_title("Pressão ao longo do eixo"), ax[0].grid(alpha=.3)
        ax[1].plot(prof[:, 0], prof[:, 2], lw=2, color="#d62728")
        ax[1].set_xlabel("x [mm]"), ax[1].set_ylabel("T estática [K]")
        ax[1].set_title("Temperatura (efeito Joule–Thomson)"), ax[1].grid(alpha=.3)
        ax[2].plot(prof[:, 0], prof[:, 4], lw=2, color="#2ca02c")
        ax[2].set_xlabel("x [mm]"), ax[2].set_ylabel("Ma [-]")
        ax[2].set_title("Número de Mach"), ax[2].grid(alpha=.3)
        path = os.path.join(folder, "perfil_axial.png")
        fig.savefig(path, dpi=140), plt.close(fig)
        out.append(path)

    path = os.path.join(folder, "corte.png")
    if _cut_figure(nominal, path):
        out.append(path)

    if curve:
        fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
        xs = [c.result.open_fraction for c in curve] or [c.as_row()["open_fraction"] for c in curve]
        kv = [c.result.kv for c in curve]
        md = [c.result.mdot_kg_s for c in curve]
        ax.plot(xs, kv, "o-", color="#1f77b4", label="Kv")
        ax.set_xlabel("fração de abertura [-]"), ax.set_ylabel("Kv [m³/h]", color="#1f77b4")
        ax2 = ax.twinx()
        ax2.plot(xs, md, "s--", color="#d62728")
        ax2.set_ylabel("vazão mássica [kg/s]", color="#d62728")
        ax.set_title(f"Curva característica ({gas.name}), Δp = 10 bar")
        ax.grid(alpha=.3)
        fig.savefig(os.path.join(folder, "curva.png"), dpi=140)
        plt.close(fig)
        out.append(os.path.join(folder, "curva.png"))
    return out


def _meridian_figure(nominal: CaseOutcome, path: str) -> bool:
    """The same three panels for a **3-D** mesh, which has no symmetry plane to cut along.

    The slice is built by projecting the cells onto the meridian and averaging them into bins: the
    flow is axisymmetric everywhere except the bonnet, and that is exactly where a reader looks.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .visual import frame_on_passage

    c = nominal.fields.get("centres")
    if c is None or not len(c):
        return False
    (xc, _cy, _cz), radius = frame_on_passage(nominal.passage)
    half_x = 2.4 * radius * 1000.0
    lim = radius * 1000.0
    # fatia de verdade: só as células encostadas no plano z = 0, não a projeção do domínio inteiro
    dz = float(np.median(np.abs(np.diff(np.unique(np.round(c[:, 2], 6)))))) if len(c) > 10 else 2e-3
    band = np.abs(c[:, 2]) < max(1.5 * dz, 1e-3)
    if band.sum() < 200:
        band = np.ones(len(c), dtype=bool)
    x, y = c[band, 0] * 1000.0, c[band, 1] * 1000.0
    nx, ny = 150, 95
    xe = np.linspace(xc * 1000.0 - half_x, xc * 1000.0 + half_x, nx + 1)
    ye = np.linspace(-lim, lim, ny + 1)
    ix = np.clip(np.digitize(x, xe) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, ye) - 1, 0, ny - 1)
    flat = ix * ny + iy
    panels = {"pressão estática [bar]": nominal.fields["p"][band] / 1e5,
              "número de Mach [-]": nominal.fields["Ma"][band] if len(nominal.fields["Ma"])
              else np.zeros(int(band.sum())),
              "temperatura estática [K]": nominal.fields["T"][band]}
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
        for ax, (name, val) in zip(axes, panels.items(), strict=False):
            total = np.bincount(flat, weights=val, minlength=nx * ny)
            count = np.bincount(flat, minlength=nx * ny)
            img = np.where(count > 0, total / np.maximum(count, 1), np.nan).reshape(nx, ny).T
            im = ax.imshow(img, origin="lower", cmap="viridis", aspect="equal",
                           extent=[xe[0], xe[-1], ye[0], ye[-1]], interpolation="nearest")
            fig.colorbar(im, ax=ax, shrink=.92, pad=.02)
            ax.set_title(name, fontsize=10), ax.set_xlabel("x [mm]"), ax.set_ylabel("r [mm]")
        fig.suptitle(f"{nominal.passage.title} — plano meridiano (malha 3-D, eixo em x = "
                     f"{xc * 1000:.0f} mm)", fontsize=11)
        fig.savefig(path, dpi=150), plt.close(fig)
    except Exception:                                     # noqa: BLE001
        return False
    return True


def _cut_figure(nominal: CaseOutcome, path: str) -> bool:
    """p, Ma and T on the meridian cut, framed on the throttling region.

    A passage is 0.35 m of straight pipe around a 70 mm valve, so a picture of the whole domain is a
    picture of two pipes with a stub in the middle: the frames are set to the body, which is where
    every gradient in the case lives.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.tri import Triangulation

    from .post import patch_triangles
    from .visual import frame_on_passage

    try:
        v, t, cell = patch_triangles(nominal.case_dir, "sym0")
    except (OSError, ValueError, KeyError):
        return _meridian_figure(nominal, path)
    if len(t) == 0:
        return _meridian_figure(nominal, path)      # malha 3-D: não há plano de simetria
    x, y = v[:, 0] * 1000.0, v[:, 1] * 1000.0
    tri = Triangulation(x, y, t)
    fields = {"pressão estática [bar]": nominal.fields["p"][cell] / 1e5,
              "número de Mach [-]": nominal.fields["Ma"][cell] if len(nominal.fields["Ma"])
              else np.zeros(len(cell)),
              "temperatura estática [K]": nominal.fields["T"][cell]}
    (xc, _cy, _cz), radius = frame_on_passage(nominal.passage)
    half_x, xc_mm, r_mm = 2.4 * radius * 1000.0, xc * 1000.0, radius * 1000.0
    try:
        fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.4), constrained_layout=True)
        for ax, (name, val) in zip(axes, fields.items(), strict=False):
            # per-face values (one per cell) need flat shading: gouraud wants them at the vertices
            tpc = ax.tripcolor(tri, val, shading="flat", cmap="viridis")
            fig.colorbar(tpc, ax=ax, shrink=.92, pad=.02)
            ax.set_title(name, fontsize=10), ax.set_xlabel("x [mm]"), ax.set_ylabel("r [mm]")
            ax.set_xlim(xc_mm - half_x, xc_mm + half_x)
            ax.set_ylim(-r_mm * 0.85, r_mm * 0.85)
            ax.set_aspect("equal")
        fig.suptitle(f"{nominal.passage.title} — corte meridiano (eixo da válvula em x = "
                     f"{xc_mm:.0f} mm)", fontsize=11)
        fig.savefig(path, dpi=150)
        plt.close(fig)
    except (ValueError, RuntimeError):
        return False
    return True


def _pictures(folder: str, nominal: CaseOutcome, spec: StudySpec) -> dict[str, str]:
    """The Blender render and the video, if Blender is installed."""
    out: dict[str, str] = {}
    try:
        from .visual import FIELDS, build_scene, frame_on_passage, render, render_video
        scene = build_scene(nominal.passage, nominal.fields, title=spec.title or nominal.passage.title)
        frame = frame_on_passage(nominal.passage)
    except Exception as exc:
        return {"erro": f"a cena 3-D não pôde ser montada: {type(exc).__name__}: {exc}"}
    png = os.path.join(folder, "render_blender.png")
    # uma imagem por campo: o relatório precisa do mesmo 3-D para p, U, Ma e T, não só para um deles
    for campo in FIELDS:
        alvo = png if campo == "Ma" else os.path.join(folder, f"render_blender_{campo}.png")
        try:
            out["render" if campo == "Ma" else f"render_{campo}"] = render(
                scene, alvo, field=campo, samples=64, resolution=1200, frame=frame)
        except Exception as exc:
            out[f"render_erro_{field}"] = f"{type(exc).__name__}: {exc}"
    if spec.video:
        mp4 = os.path.join(folder, "simulacao.mp4")
        try:
            out["video"] = render_video(scene, mp4, field="Ma", frames=36, samples=14,
                                        resolution=720, frame=frame)
        except Exception as exc:
            out["video_erro"] = f"{type(exc).__name__}: {exc}"
    return out
