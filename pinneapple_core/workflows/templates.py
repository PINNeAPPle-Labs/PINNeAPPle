"""Workflow templates: a parameter space, the steps run per case, and the step whose outputs form the dataset.

These are the only workflows a public server should run (``Template.workflow`` checks every case against the
template's parameter space, so case values cannot inject text into commands or solver dictionaries):

==================  ========================================================================  ==================
``slab_heat``       transient conduction in a slab, exact series solution                        Python, seconds
                    (``pinneapple_physics.closed_form.heat.slab_transient``), then a summary
``beam_calculix``   tip-loaded cantilever: deck (``calculix.study.FEModel``), ``ccx``, and the    needs ``ccx``
                    deflection compared with Euler-Bernoulli
``cavity_openfoam`` lid-driven cavity (icoFoam): lid speed and viscosity edited into the case     needs OpenFOAM
==================  ========================================================================  ==================
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adapters import StepContext, get_adapter
from .dag import Step, Workflow, workflow_from_cases

CASES_DIR = Path(__file__).parent / "cases"


@dataclass
class Template:
    id: str
    title: str
    description: str
    space: Any                                    # pinneapple_data.contracts.ParameterSpace
    steps: list[Step]
    output_step: str
    final: list[Step] = field(default_factory=list)
    outputs: dict[str, str] = field(default_factory=dict)       # output name -> unit
    solver: str = "python"

    def availability(self) -> tuple[bool, str]:
        for s in self.steps + self.final:
            ok, why = get_adapter(s.kind).available(s)
            if not ok:
                return False, why
        return True, ""

    def example_cases(self, n: int = 6, seed: int = 0):
        """``n`` cases from a Latin hypercube of the template's space (``pinneapple_data.parameter_sampling``)."""
        from pinneapple_data.parameter_sampling import sample_parameters
        unit = sample_parameters({p.name: (0.0, 1.0) for p in self.space.free}, n, method="lhs", seed=seed)
        import numpy as np
        U = np.array([[row[p.name] for p in self.space.free] for row in unit])
        df = self.space.from_unit(U)
        df.insert(0, "case_id", [f"c{i + 1:02d}" for i in range(len(df))])
        return df

    def check_cases(self, cases) -> list[str]:
        """Problems with a case table (missing columns, values outside the space); empty when it can run."""
        import pandas as pd
        df = cases if isinstance(cases, pd.DataFrame) else pd.DataFrame(list(cases))
        missing = [p.name for p in self.space.parameters if p.free and p.name not in df.columns]
        if missing:
            return [f"missing columns: {missing}"]
        df = df.copy()
        for p in self.space.parameters:
            if not p.free and p.name not in df.columns:
                df[p.name] = p.value
            if p.kind in ("continuous", "integer"):
                bad = pd.to_numeric(df[p.name], errors="coerce").isna()
                if bad.any():
                    return [f"column {p.name}: non-numeric values in rows {list(df.index[bad][:5])}"]
                df[p.name] = pd.to_numeric(df[p.name])
        ok = self.space.feasible(df)
        if not ok.all():
            rows = list(df.index[~ok][:5])
            return [f"{int((~ok).sum())} case(s) outside the template's parameter range (rows {rows}); "
                    f"ranges: " + ", ".join(f"{p.name} [{p.low:g}, {p.high:g}]" for p in self.space.parameters
                                             if p.kind in ("continuous", "integer"))]
        return []

    def workflow(self, cases, *, name: str | None = None, id_column: str | None = "case_id", max_parallel: int = 2,
                 executor: str = "local", executor_options: dict | None = None) -> Workflow:
        import pandas as pd
        df = cases if isinstance(cases, pd.DataFrame) else pd.DataFrame(list(cases))
        problems = self.check_cases(df)
        if problems:
            raise ValueError("; ".join(problems))
        df = df.copy()
        for p in self.space.parameters:
            if not p.free and p.name not in df.columns:
                df[p.name] = p.value
        if id_column not in df.columns:
            df.insert(0, "case_id", [f"c{i + 1:02d}" for i in range(len(df))])
            id_column = "case_id"
        tmpl = [Step(**{**s.to_dict(), "executor": executor if s.kind != "python" else "local",
                        "executor_options": dict(executor_options or {}) if s.kind != "python" else {}})
                for s in self.steps]
        return workflow_from_cases(name or self.id, df.to_dict("records"), tmpl, id_column=id_column,
                                   final=self.final, max_parallel=max_parallel,
                                   metadata={"template": self.id, "output_step": self.output_step})


# -- step functions (called by the python adapter as fn(kwargs, ctx)) ---------------------------------------------
def slab_heat_solve(kw: dict, ctx: StepContext) -> dict:
    """Temperature profile of the slab at ``t_end`` from the exact series solution; writes ``profile.csv``."""
    import numpy as np

    from pinneapple_physics.closed_form.heat import slab_transient
    L, alpha, t = float(kw["L"]), float(kw["alpha"]), float(kw["t_end"])
    x = np.linspace(0.0, L, 201)
    T = slab_transient(x, t, L, alpha, float(kw["T_left"]), float(kw["T_right"]), float(kw["T0"]))
    steady = float(kw["T_left"]) + (float(kw["T_right"]) - float(kw["T_left"])) * x / L
    np.savetxt(ctx.workdir / "profile.csv", np.column_stack([x, T]), delimiter=",", header="x_m,T_K", comments="")
    ctx.log(f"profile at t = {t:g} s on 201 points written")
    dx = x[1] - x[0]
    return {"T_mid": float(np.interp(L / 2, x, T)), "T_mean": float(np.trapezoid(T, x) / L),
            "dTdx_left": float((T[1] - T[0]) / dx), "fourier": alpha * t / L**2,
            "approach_to_steady": float(np.max(np.abs(T - steady)) / max(abs(float(kw["T_left"]) -
                                                                        float(kw["T0"])), 1e-12))}


def collect_outputs(kw: dict, ctx: StepContext) -> dict:
    """Summary over every upstream step: count, and min / max of each numeric output; writes ``summary.csv``."""
    import csv
    rows = [{"step": k, **{n: v for n, v in u["outputs"].items() if isinstance(v, (int, float))}}
            for k, u in sorted(ctx.upstream.items())]
    names = sorted({n for r in rows for n in r if n != "step"})
    with open(ctx.workdir / "summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["step", *names])
        w.writeheader()
        w.writerows(rows)
    out: dict[str, Any] = {"n_cases": len(rows)}
    for n in names:
        vals = [r[n] for r in rows if n in r]
        out[f"{n}_min"], out[f"{n}_max"] = min(vals), max(vals)
    return out


def cantilever_deck(kw: dict, ctx: StepContext) -> dict:
    """``job.inp`` of a cantilever (C3D8I hexahedra) clamped at x = 0 with a tip load, and the Euler-Bernoulli tip
    deflection F L^3 / (3 E I) for comparison."""
    from pinneapple_simulation.external_solvers.calculix.study import FEModel, Material, Static
    L, b, h, E, F = (float(kw[k]) for k in ("L", "b", "h", "E", "F"))
    m = FEModel.from_box(L, b, h, int(kw.get("nx", 40)), 2, 4, element="C3D8I")
    m.material = Material("steel", E=E, nu=0.3, density=7850.0)
    root = m.nodes_where(lambda X: X[:, 0] < 1e-9)
    tip = m.nodes_where(lambda X: X[:, 0] > L - 1e-9)
    (ctx.workdir / "job.inp").write_text(m.deck(Static(fix=[(root, (1, 2, 3))], loads=[(tip, (0.0, 0.0, -F))])))
    inertia = b * h**3 / 12.0
    return {"tip_deflection_beam": F * L**3 / (3 * E * inertia), "nodes": int(len(m.nodes))}


def cantilever_compare(kw: dict, ctx: StepContext) -> dict:
    deck = next(u["outputs"] for k, u in ctx.upstream.items() if k.endswith(".deck"))
    solve = next(u["outputs"] for k, u in ctx.upstream.items() if k.endswith(".solve"))
    if solve.get("dry_run"):
        return {"tip_deflection_beam": deck["tip_deflection_beam"], "dry_run": True}
    fe, beam = solve["max_displacement"], deck["tip_deflection_beam"]
    return {"tip_deflection_fe": fe, "tip_deflection_beam": beam, "fe_over_beam": fe / beam}


def _templates() -> dict[str, Template]:
    from pinneapple_data.contracts import Parameter, ParameterSpace
    mod = __name__
    slab = Template(
        "slab_heat", "Slab heating (Python, exact solution)",
        "A slab at T0 whose left face is raised to T_left at t = 0 (right face held at T_right): temperature "
        "profile at t_end from the exact Fourier-series solution, then a summary over the cases.",
        ParameterSpace(name="slab_heat", parameters=[
            Parameter(name="L", low=0.01, high=0.1, unit="m", group="geometry"),
            Parameter(name="alpha", low=1e-7, high=1e-4, log=True, unit="m^2/s", group="material"),
            Parameter(name="T_left", low=320.0, high=500.0, unit="K", group="boundary"),
            Parameter(name="T_right", kind="constant", value=300.0, unit="K", group="boundary"),
            Parameter(name="T0", kind="constant", value=300.0, unit="K", group="boundary"),
            Parameter(name="t_end", low=1.0, high=600.0, unit="s", group="operating")]),
        steps=[Step("solve", "python", {"function": f"{mod}:slab_heat_solve",
                                        "kwargs": {k: "{" + k + "}" for k in ("L", "alpha", "T_left", "T_right", "T0",
                                                                             "t_end")}})],
        final=[Step("summary", "python", {"function": f"{mod}:collect_outputs"}, needs=["solve"])],
        output_step="solve",
        outputs={"T_mid": "K", "T_mean": "K", "dTdx_left": "K/m", "fourier": "", "approach_to_steady": ""})
    beam = Template(
        "beam_calculix", "Cantilever beam (CalculiX)",
        "A steel-like cantilever clamped at one end with a tip load: input deck from PINNeAPPle's CalculiX model "
        "builder, ccx, and the finite-element tip deflection against Euler-Bernoulli beam theory.",
        ParameterSpace(name="beam_calculix", parameters=[
            Parameter(name="L", low=0.5, high=2.0, unit="m", group="geometry"),
            Parameter(name="b", kind="constant", value=0.05, unit="m", group="geometry"),
            Parameter(name="h", low=0.02, high=0.1, unit="m", group="geometry"),
            Parameter(name="E", low=7.0e10, high=2.1e11, unit="Pa", group="material"),
            Parameter(name="F", low=100.0, high=2000.0, unit="N", group="boundary")]),
        steps=[Step("deck", "python", {"function": f"{mod}:cantilever_deck",
                                       "kwargs": {k: "{" + k + "}" for k in ("L", "b", "h", "E", "F")}}),
               Step("solve", "calculix", {"inp_from": "deck", "job": "job"}, needs=["deck"]),
               Step("compare", "python", {"function": f"{mod}:cantilever_compare"}, needs=["deck", "solve"])],
        output_step="compare", solver="calculix",
        outputs={"tip_deflection_fe": "m", "tip_deflection_beam": "m", "fe_over_beam": ""})
    cavity = Template(
        "cavity_openfoam", "Lid-driven cavity (OpenFOAM icoFoam)",
        "The OpenFOAM cavity tutorial (20 x 20 cells, 5 s): lid speed and kinematic viscosity edited into the case "
        "per run; the solver log's health (Courant number, end reached) is recorded.",
        ParameterSpace(name="cavity_openfoam", parameters=[
            Parameter(name="U_lid", low=0.5, high=2.0, unit="m/s", group="boundary"),
            Parameter(name="nu", low=0.005, high=0.05, unit="m^2/s", group="material")]),
        steps=[Step("solve", "openfoam", {
            "case": str(CASES_DIR / "cavity"), "solver": "icoFoam",
            "edits": [{"file": "0/U", "key": "boundaryField/movingWall/value", "value": "uniform ({U_lid} 0 0)"},
                      {"file": "constant/transportProperties", "key": "nu",
                       "value": "[0 2 -1 0 0 0 0] {nu}"}]})],
        output_step="solve", solver="openfoam", outputs={"latest_time": "s"})
    return {t.id: t for t in (slab, beam, cavity)}


TEMPLATES: dict[str, Template] = _templates()
