"""A full physical model for a problem no preset covers, from governing laws, as an editable, checked draft (#1175).

The Engineer states a GAP when no tool models a problem's physics (a battery cell's heating, CO2 in a ventilated
room). :func:`formulate` composes the model from **laws**: conservation equations and their closures, each written
in the equation text :mod:`pinneapple_physics.symbolic_pde.text` parses safely, with the units of every field and
coefficient and the source of the law. The draft is the system the physics implies, **never a simplified one**: a
reduction (a lumped body, a steady state, a 1-D slab) is offered only as a named :class:`Assumption` with the
number that decides whether it holds, and applied only when the caller accepts it.

What a :class:`ModelDraft` carries and checks:

* the PDEs, boundary and initial conditions as text, as LaTeX and as the parsed
  :class:`~pinneapple_physics.symbolic_pde.text.EquationSystem` (a PINN through ``pinn_problem``, a classical solve
  through the same system);
* the constants: value, unit and source, or "to be given" with what the value is;
* :meth:`ModelDraft.checks`: every term of every equation and condition has one dimension (from the units, via
  :func:`pinneapple_data.physical_units.parse_unit`); the system is closed (fields vs equations, every coefficient
  defined); each second-order field has a condition on every face and a time-dependent field an initial condition;
  a manufactured solution evaluated through the compiled residual (the path a PINN trains on) agrees with SymPy;
* :meth:`ModelDraft.edit`: a changed equation or condition is re-parsed and every check runs again; the loss code
  follows the equations.

Laws are written for 3-D (``x, y, z, t``); a draft in fewer dimensions drops the absent coordinates' terms.
"""
from __future__ import annotations

import math
import random
import re
from dataclasses import dataclass, field, replace
from typing import Any

import sympy as sp

R_GAS = 8.314462618                       # J/(mol K), CODATA 2018 (exact)

# --------------------------------------------------------------------------------------------- building blocks


@dataclass(frozen=True)
class Coefficient:
    name: str
    unit: str
    description: str
    value: float | None = None            # None: to be given
    source: str = ""                      # where the value comes from (a table, a datasheet, a definition)


@dataclass(frozen=True)
class Law:
    """One governing law or closure: equations in the parser's text, the fields it solves, its coefficients."""
    id: str
    title: str
    equations: tuple[str, ...]
    fields: dict[str, str]                # field -> unit
    coefficients: tuple[Coefficient, ...]
    source: str
    needs: tuple[str, ...] = ()           # other laws it couples to (their fields appear in its equations)


@dataclass(frozen=True)
class Assumption:
    """A reduction the caller may accept: what it drops, the number that decides it, and that number's limit."""
    id: str
    title: str
    drops: str
    criterion: str                        # e.g. "Bi = h*Lc/k"
    limit: str                            # e.g. "Bi < 0.1"
    source: str
    value: float | None = None            # the criterion evaluated from the stated values, when they are known
    holds: bool | None = None
    applied: bool = False                 # True: the draft already makes it (the caller asked, or the domain does)


AIR_300K = "Incropera, DeWitt, Bergman & Lavine, Fundamentals of Heat and Mass Transfer, 7th ed., Table A.4 (air, " \
           "300 K)"
_AIR = {"rho": Coefficient("rho", "kg/m^3", "air density", 1.1614, AIR_300K),
        "mu": Coefficient("mu", "Pa*s", "air dynamic viscosity", 184.6e-7, AIR_300K),
        "k": Coefficient("k", "W/(m*K)", "air thermal conductivity", 26.3e-3, AIR_300K),
        "cp": Coefficient("cp", "J/(kg*K)", "air specific heat", 1007.0, AIR_300K),
        "beta": Coefficient("beta", "1/K", "air thermal expansion coefficient (ideal gas: 1/T at 300 K)", 1 / 300.0,
                            "ideal gas, beta = 1/T (Incropera, eq. 9.10)")}

LAWS: dict[str, Law] = {
    "heat_conduction": Law(
        "heat_conduction", "Energy in a solid: transient conduction with a volumetric source",
        ("rho*cp*T_t = k*lap(T) + q",),
        {"T": "K"},
        (Coefficient("rho", "kg/m^3", "density"), Coefficient("cp", "J/(kg*K)", "specific heat"),
         Coefficient("k", "W/(m*K)", "thermal conductivity"), Coefficient("q", "W/m^3", "volumetric heat source")),
        "Incropera et al., 7th ed., eq. 2.19 (heat diffusion equation)"),
    "battery_heat": Law(
        "battery_heat", "Energy in a lithium-ion cell: conduction with Bernardi's heat generation, Joule heating "
                        "through a temperature-dependent internal resistance and the reversible entropic heat",
        ("rho*cp*T_t = k*lap(T) + (I**2*R0*exp(EaR*(1/T - 1/Tref)) - I*T*dUdT)/Vc",),
        {"T": "K"},
        (Coefficient("rho", "kg/m^3", "cell mean density (datasheet or measured)"),
         Coefficient("cp", "J/(kg*K)", "cell mean specific heat (calorimetry)"),
         Coefficient("k", "W/(m*K)", "cell effective thermal conductivity (anisotropic in a real cell: through "
                                     "the layers lower than along them)"),
         Coefficient("I", "A", "cell current (the input of the surrogate)"),
         Coefficient("R0", "ohm", "internal resistance at Tref (HPPC test)"),
         Coefficient("EaR", "K", "activation temperature of the resistance, Ea/R (fit of R(T))"),
         Coefficient("Tref", "K", "reference temperature of R0"),
         Coefficient("dUdT", "V/K", "entropic coefficient dU_ocv/dT (potentiometric measurement)"),
         Coefficient("Vc", "m^3", "cell volume (datasheet)")),
        "Bernardi, Pawlikowski & Newman, J. Electrochem. Soc. 132 (1985) 5-12 (heat generation); Arrhenius form of "
        "R(T)"),
    "surface_convection": Law(
        "surface_convection", "Convection from a surface to a fluid at Tamb (Newton's law of cooling)",
        (), {}, (Coefficient("h", "W/(m^2*K)", "convection coefficient"),
                 Coefficient("Tamb", "K", "fluid (ambient) temperature")),
        "Incropera et al., 7th ed., eq. 1.3a"),
    "incompressible_flow": Law(
        "incompressible_flow", "Mass and momentum of an incompressible fluid with buoyancy (Boussinesq)",
        ("u_x + v_y + w_z = 0",
         "rho*(u_t + u*u_x + v*u_y + w*u_z) = -p_x + mu*lap(u)",
         "rho*(v_t + u*v_x + v*v_y + w*v_z) = -p_y + mu*lap(v)",
         "rho*(w_t + u*w_x + v*w_y + w*w_z) = -p_z + mu*lap(w) + rho*g*beta*(T - T0)"),
        {"u": "m/s", "v": "m/s", "w": "m/s", "p": "Pa"},
        (_AIR["rho"], _AIR["mu"], _AIR["beta"],
         Coefficient("g", "m/s^2", "gravitational acceleration", 9.80665, "standard gravity (CGPM 1901)"),
         Coefficient("T0", "K", "reference temperature of the Boussinesq term")),
        "Bird, Stewart & Lightfoot, Transport Phenomena, 2nd ed., eqs. 3.1-4 and 3.5-6; Boussinesq: eq. 10.9-6",
        needs=("fluid_energy",)),
    "fluid_energy": Law(
        "fluid_energy", "Energy of the fluid: advection and conduction with a source",
        ("rho*cp*(T_t + u*T_x + v*T_y + w*T_z) = k*lap(T) + qT",),
        {"T": "K"},
        (_AIR["rho"], _AIR["cp"], _AIR["k"], Coefficient("qT", "W/m^3", "volumetric heat source (occupants, "
                                                                          "equipment)")),
        "Bird, Stewart & Lightfoot, 2nd ed., eq. 11.2-9", needs=("incompressible_flow",)),
    "species_transport": Law(
        "species_transport", "A species carried by the flow and diffusing, with a source",
        ("c_t + u*c_x + v*c_y + w*c_z = D*lap(c) + Sc",),
        {"c": "mol/m^3"},
        (Coefficient("D", "m^2/s", "diffusivity of the species in the fluid (CO2 in air: 1.77e-5 m^2/s at 317 K)",
                     1.77e-5, "Bird, Stewart & Lightfoot, 2nd ed., Table 17.1-1 (air-CO2, 317.2 K)"),
         Coefficient("Sc", "mol/(m^3*s)", "volumetric source of the species (occupants' exhalation, located where "
                                          "they sit)")),
        "Bird, Stewart & Lightfoot, 2nd ed., eq. 19.1-16", needs=("incompressible_flow",)),
}
# relations (algebraic models, e.g. what a measured curve follows): output = expression of inputs
RELATIONS: dict[str, dict[str, Any]] = {
    "arrhenius": {
        "title": "Arrhenius rate constant",
        "equation": "k = A*exp(-Ea/(Rg*T))",
        "inputs": {"T": "K"}, "output": ("k", "1/s"),
        "coefficients": (Coefficient("A", "1/s", "pre-exponential factor (to identify)"),
                         Coefficient("Ea", "J/mol", "activation energy (to identify)"),
                         Coefficient("Rg", "J/(mol*K)", "gas constant", R_GAS, "CODATA 2018 (exact)")),
        "source": "Arrhenius, Z. Phys. Chem. 4 (1889) 226; IUPAC Gold Book, 'Arrhenius equation'",
        "unknowns": ("A", "Ea"),
    },
}
# phenomenon -> the laws it needs
PHENOMENA: dict[str, tuple[str, ...]] = {
    "battery_cell_thermal": ("battery_heat", "surface_convection"),
    "solid_heat": ("heat_conduction", "surface_convection"),
    "room_air_species": ("incompressible_flow", "fluid_energy", "species_transport"),
}

# --------------------------------------------------------------------------------------------- units


def _dim(unit: str) -> tuple:
    from pinneapple_data.physical_units import parse_unit
    if unit in ("", "1", "-"):
        return _ZERO
    return tuple(float(x) for x in parse_unit(unit).dim)


def _add(a, b, k=1):
    return tuple(x + k * y for x, y in zip(a, b, strict=True))


def _scale(a, k):
    return tuple(x * k for x in a)


_ZERO = (0.0,) * 6                         # (m, kg, s, K, A, mol), as physical_units orders them


class DimensionError(ValueError):
    pass


def _dimension(expr, units: dict[str, tuple], coords: dict[str, tuple]):
    """The dimension of ``expr`` from the units of its symbols and fields; raises DimensionError naming the
    terms that disagree."""
    zero = _ZERO
    if expr.is_Number or expr in (sp.pi, sp.E):
        return zero
    if isinstance(expr, sp.Symbol):
        if expr.name not in units and expr.name not in coords:
            raise DimensionError(f"no unit for {expr.name!r}")
        return units.get(expr.name) or coords[expr.name]
    if isinstance(expr, sp.core.function.AppliedUndef):
        return units[expr.func.__name__]
    if isinstance(expr, sp.Derivative):
        d = _dimension(expr.expr, units, coords)
        for v, n in expr.variable_count:
            d = _add(d, coords[v.name], -int(n))
        return d
    if isinstance(expr, sp.Add):
        dims = [(t, _dimension(t, units, coords)) for t in expr.args]
        first = dims[0][1]
        for t, d in dims[1:]:
            if d != first:
                raise DimensionError(f"terms of different dimension: {sp.sstr(dims[0][0])} and {sp.sstr(t)}")
        return first
    if isinstance(expr, sp.Mul):
        d = zero
        for t in expr.args:
            d = _add(d, _dimension(t, units, coords))
        return d
    if isinstance(expr, sp.Pow):
        base = _dimension(expr.base, units, coords)
        if base == zero:
            if _dimension(expr.exp, units, coords) != zero:
                raise DimensionError(f"an exponent with a dimension: {sp.sstr(expr)}")
            return zero
        if not expr.exp.is_Number:
            raise DimensionError(f"a dimensional base to a symbolic power: {sp.sstr(expr)}")
        return _scale(base, float(expr.exp))
    if isinstance(expr, sp.Function):           # exp, sin, log, ...: dimensionless argument and result
        for a in expr.args:
            if _dimension(a, units, coords) != zero:
                raise DimensionError(f"{expr.func.__name__} of a dimensional argument: {sp.sstr(a)}")
        return zero
    raise DimensionError(f"cannot read the dimension of {sp.sstr(expr)}")


# --------------------------------------------------------------------------------------------- the draft


@dataclass
class ModelDraft:
    title: str
    coords: list[str]
    pdes: list[str]
    conditions: list[dict[str, str]]
    fields: dict[str, str]
    coefficients: dict[str, Coefficient]
    laws: list[str]
    sources: list[str]
    assumptions: list[Assumption] = field(default_factory=list)
    relation: dict[str, Any] | None = None      # an algebraic model instead of PDEs
    history: list[dict[str, Any]] = field(default_factory=list)

    # --- views
    def system(self):
        from pinneapple_physics.symbolic_pde.text import parse_system
        vals = {k: c.value for k, c in self.coefficients.items() if c.value is not None}
        return parse_system(self.pdes, self.conditions, coords=self.coords, fields=list(self.fields),
                            coefficients=vals)

    def latex(self) -> dict[str, list[str]]:
        if self.relation:
            lhs, rhs = self.relation["equation"].split("=", 1)
            return {"relation": [sp.latex(sp.Eq(sp.sympify(lhs.strip()), sp.sympify(rhs.strip())))]}
        s = self.system()
        args = r"{\left(" + ",".join(self.coords) + r" \right)}"

        def tex(e):                          # T(x, y, t) -> T: the coordinates are stated once, with the domain
            return sp.latex(sp.Eq(e.lhs, e.rhs)).replace(args, "")
        return {"pdes": [tex(e) for e in s.pdes],
                "conditions": [{"kind": c.kind, "on": c.on, "latex": tex(c)} for c in s.conditions]}

    def constants(self) -> list[dict[str, Any]]:
        return [{"name": c.name, "unit": c.unit, "description": c.description, "value": c.value,
                 "source": c.source or ("to be given" if c.value is None else "")} for c in self.coefficients.values()]

    def loss_code(self, bounds: dict[str, tuple[float, float]] | None = None) -> str:
        """The physics loss as code a PINN trains with (the library's ``pinn_problem`` on this very system)."""
        if self.relation:
            r = self.relation
            out, rhs = r["equation"].split("=", 1)
            free = [c for c in r["coefficients"] if c.name in r.get("unknowns", ())]
            fixed = {c.name: c.value for c in r["coefficients"] if c.name not in r.get("unknowns", ())}
            ins = list(r["inputs"])
            return "\n".join([
                "import numpy as np",
                "from pinneapple_analysis.inverse_problems.calibration import calibrate",
                "from pinneapple_analysis.inverse_problems.forward import CallableModel, Observations, Parameter",
                "",
                f"FIXED = {fixed!r}",
                "",
                "def model(theta, inputs):",
                f"    {', '.join(ins)}{',' if len(ins) == 1 else ''} = ({', '.join(f'inputs[{i!r}]' for i in ins)}"
                f"{',' if len(ins) == 1 else ''})",
                "    env = {**FIXED, **theta}",
                f"    {', '.join(c.name for c in free)}{',' if len(free) == 1 else ''} = "
                f"({', '.join(f'env[{c.name!r}]' for c in free)}{',' if len(free) == 1 else ''})",
                *[f"    {k} = env[{k!r}]" for k in fixed],
                f"    return {rhs.strip().replace('exp(', 'np.exp(')}   # {out.strip()} [{r['output'][1]}]",
                "",
                "# the bounds are yours to give (a physically possible range for each unknown)",
                "PARAMETERS = [" + ", ".join(f"Parameter({c.name!r}, value=..., lower=..., upper=..., unit={c.unit!r})"
                                             for c in free) + "]",
                f"cm = CallableModel(model, inputs={ins!r}, outputs=[{out.strip()!r}], parameters=PARAMETERS)",
                f"obs = Observations(inputs={{{', '.join(f'{i!r}: ...' for i in ins)}}}, values=..., sigma=...)",
                "fit = calibrate(cm, obs, PARAMETERS, holdout_fraction=0.2)   # estimates, intervals, held-out misfit",
            ])
        b = bounds or {c: (0.0, 1.0) for c in self.coords}
        given = {k: c.value for k, c in self.coefficients.items() if c.value is not None}
        todo = sorted(k for k, c in self.coefficients.items() if c.value is None and k in self._used())
        return "\n".join([
            "from pinneapple_physics.symbolic_pde.text import parse_system",
            "from pinneapple_physics.symbolic_pde.pinn import pinn_problem",
            "from pinneapple_analysis.validation.pinn import train_pinn, validate_pinn",
            "",
            f"PDES = {self.pdes!r}",
            f"CONDITIONS = {self.conditions!r}",
            f"COEFFICIENTS = {given!r}" + (f"  # to be given: {', '.join(todo)}" if todo else ""),
            f"system = parse_system(PDES, CONDITIONS, coords={self.coords!r}, fields={list(self.fields)!r}, "
            "coefficients=COEFFICIENTS)",
            f"SCALES = {self.scales()!r}  # field -> (offset, scale) from the stated values: the network trains near 1",
            f"problem, model = pinn_problem(system, {b!r}, scales=SCALES)",
            "net, info = train_pinn(problem, model, adam_steps=2000, lbfgs_steps=200)",
            "report = validate_pinn(net, problem)       # residuals on points never trained on",
        ])

    def scales(self) -> dict[str, tuple[float, float]]:
        """Per field, an offset and a scale from the constants of the same unit (temperatures: the lowest stated
        one the conditions name and their spread, at least 1 K; others: zero and the largest magnitude, else 1)."""
        out = {}
        for f, unit in self.fields.items():
            named = set()                   # the constants the conditions set this field against (T0, Tamb, Uin)
            for c in self.conditions:
                if re.search(rf"\b{f}(_[a-z]+)?\b", c["equation"]):
                    named |= set(re.findall(r"[A-Za-z][A-Za-z0-9]*", c["equation"]))
            vals = [c.value for c in self.coefficients.values()
                    if c.name in named and c.unit == unit and c.value is not None]
            if unit == "K" and vals:
                out[f] = (min(vals), max(max(vals) - min(vals), 1.0))
            else:
                out[f] = (0.0, max((abs(v) for v in vals if v), default=1.0))
        return out

    def _used(self) -> set[str]:
        if self.relation:
            return {c.name for c in self.relation["coefficients"]}
        return self.system().used_coefficients()

    # --- checks
    def checks(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []

        def add(code, ok, detail):
            out.append({"code": code, "status": "pass" if ok else "fail", "detail": detail})

        units = {k: _dim(c.unit) for k, c in self.coefficients.items()}
        if self.relation:
            r = self.relation
            units |= {k: _dim(u) for k, u in r["inputs"].items()}
            lhs, rhs = r["equation"].split("=", 1)
            try:
                d_l = _dimension(sp.sympify(lhs.strip()), units | {r["output"][0]: _dim(r["output"][1])}, {})
                d_r = _dimension(sp.sympify(rhs.strip()), units, {})
                add("model.dimensions", d_l == d_r, "both sides have one dimension" if d_l == d_r else
                    f"{lhs.strip()} and {rhs.strip()} differ in dimension")
            except DimensionError as e:
                add("model.dimensions", False, str(e))
            return out
        units |= {k: _dim(u) for k, u in self.fields.items()}
        coords = {c: (_dim("s") if c == "t" else _dim("m")) for c in self.coords}
        try:
            s = self.system()
        except Exception as e:  # noqa: BLE001 - a syntax error is the check's finding
            add("model.parses", False, str(e))
            return out
        add("model.parses", True, f"{len(s.pdes)} equations, {len(s.conditions)} conditions")
        bad = []
        for e in s.pdes + s.conditions:
            try:
                _dimension(e.lhs - e.rhs, units, coords)
            except DimensionError as err:
                bad.append(f"{e.text}: {err}")
        add("model.dimensions", not bad, "every term of every equation and condition has one dimension" if not bad
            else "; ".join(bad))
        closure = [i for i in s.issues() if "without a value" not in i]
        add("model.closed", not closure, f"{len(s.fields)} fields, {len(s.pdes)} equations" if not closure else
            "; ".join(closure))
        missing = self._missing_conditions(s)
        add("model.conditions", not missing, "every field has its boundary and initial conditions" if not missing
            else "missing: " + "; ".join(missing))
        given = {k for k, c in self.coefficients.items() if c.value is not None}
        todo = sorted(s.used_coefficients() - given)
        out.append({"code": "model.constants", "status": "inconclusive" if todo else "pass",
                    "detail": f"to be given: {', '.join(todo)}" if todo else "every constant has a value and a source"})
        ok, detail = self._mms(s, units)
        add("model.mms", ok, detail)
        return out

    def _missing_conditions(self, s) -> list[str]:
        spatial = [c for c in self.coords if c != "t"]
        out = []
        for f in self.fields:
            order2 = any(sp.Derivative(sp.Function(f)(*[sp.Symbol(c, real=True) for c in self.coords]),
                                       (sp.Symbol(x, real=True), 2)) in e.residual.atoms(sp.Derivative)
                         for e in s.pdes for x in spatial)
            if order2:
                for x in spatial:
                    for side in ("min", "max"):
                        face = f"{x}_{side}"
                        if not any(c.on in (face, "boundary") and f in c.fields_used() for c in s.conditions
                                   if c.kind != "initial"):
                            out.append(f"{f} on {face}")
            timed = any(isinstance(d, sp.Derivative) and any(v.name == "t" for v, _ in d.variable_count)
                        and d.expr.func.__name__ == f for e in s.pdes for d in e.residual.atoms(sp.Derivative))
            if timed and not any(c.kind == "initial" and f in c.fields_used() for c in s.conditions):
                out.append(f"{f} initial condition")
        return out

    def _mms(self, s, units) -> tuple[bool, str]:
        """A manufactured solution through the compiled residual (the PINN's path) against SymPy."""
        from pinneapple_physics.symbolic_pde.text import compile_expr, derivative_key
        rng = random.Random(0)
        syms = {c: sp.Symbol(c, real=True) for c in self.coords}
        sc = self.scales()                      # around the field's own magnitude (a temperature near 300 K)
        man = {f: sp.Float(sc[f][0]) + sp.Float(sc[f][1]) * (1 + sp.Rational(1, 10) * sp.sin(syms[self.coords[0]] + i)
                                                             * sp.exp(-syms.get("t", 0) / 3))
               for i, f in enumerate(self.fields)}
        vals = {k: (c.value if c.value is not None else 1.0 + 0.5 * i)
                for i, (k, c) in enumerate(self.coefficients.items())}
        worst = 0.0
        for e in s.pdes:
            res = e.residual
            for _ in range(3):
                pt = {c: rng.uniform(0.1, 0.9) for c in self.coords}
                env: dict[str, Any] = dict(pt) | vals
                for d in res.atoms(sp.Derivative):
                    f = d.expr.func.__name__
                    env[derivative_key(d)] = float(sp.diff(man[f], *[(syms[v.name], n) for v, n in d.variable_count])
                                                   .subs({syms[c]: pt[c] for c in syms}))
                for f in self.fields:
                    env[f] = float(man[f].subs({syms[c]: pt[c] for c in syms}))
                got = float(compile_expr(res)(env))
                ref = res
                for f in self.fields:
                    ref = ref.subs(sp.Function(f)(*syms.values()), man[f])
                want = float(ref.doit().subs({syms[c]: pt[c] for c in syms} | {sp.Symbol(k, real=True): v
                                                                                for k, v in vals.items()}))
                worst = max(worst, abs(got - want) / max(1.0, abs(want)))
        return worst < 1e-9, f"compiled residual vs SymPy on a manufactured solution: max relative difference " \
                             f"{worst:.1e}"

    # --- edits
    def edit(self, *, pde: tuple[int, str] | None = None, condition: tuple[int, dict] | None = None,
             coefficient: Coefficient | None = None, author: str = "user") -> ModelDraft:
        """A new draft with one equation, condition or constant changed; every check runs again on it (the caller
        keeps the old draft when a check fails)."""
        new = replace(self, pdes=list(self.pdes), conditions=list(self.conditions),
                      coefficients=dict(self.coefficients), history=list(self.history))
        if pde is not None:
            new.pdes[pde[0]] = pde[1]
        if condition is not None:
            if condition[0] < len(new.conditions):
                new.conditions[condition[0]] = condition[1]
            else:
                new.conditions.append(condition[1])
        if coefficient is not None:
            new.coefficients[coefficient.name] = coefficient
        new.history.append({"author": author, "pde": pde, "condition": condition,
                            "coefficient": coefficient.name if coefficient else None})
        return new

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "coords": self.coords, "pdes": self.pdes, "conditions": self.conditions,
                "fields": self.fields, "constants": self.constants(), "laws": self.laws, "sources": self.sources,
                "assumptions": [a.__dict__ for a in self.assumptions], "relation": (
                    {k: v for k, v in self.relation.items() if k != "coefficients"} if self.relation else None),
                "latex": self.latex(), "checks": self.checks(), "loss_code": self.loss_code(),
                "history": self.history}


# --------------------------------------------------------------------------------------------- composition


def _drop_absent(text: str, coords: list[str]) -> str:
    """Remove the terms of coordinates the draft does not have (a 2-D draft has no ``*_z`` terms or ``w``)."""
    absent = {"x", "y", "z"} - set(coords)
    vel = {"x": "u", "y": "v", "z": "w"}
    for c in absent:
        text = re.sub(rf"\s*\+\s*{vel[c]}\*[A-Za-z]+_{c}\b", "", text)
        text = re.sub(rf"\s*\+\s*[A-Za-z]+_{c}\b", "", text)
    return text


def formulate(phenomenon: str, *, dims: int = 3, steady: bool = False,
              values: dict[str, float] | None = None, faces: dict[str, list[dict[str, str]]] | None = None,
              title: str | None = None) -> ModelDraft:
    """The full model of ``phenomenon`` (a key of :data:`PHENOMENA` or :data:`RELATIONS`).

    ``values`` give constants (their source becomes "user"); ``faces`` override the default conditions per field
    (``{"T": [{"kind", "on", "equation"}, ...]}``). The default conditions are the ones the physics needs on every
    face, written with symbols to be given (an inlet velocity, a wall temperature), never values."""
    values = dict(values or {})
    if phenomenon in RELATIONS:
        r = RELATIONS[phenomenon]
        coefs = {c.name: (replace(c, value=values[c.name], source="user") if c.name in values else c)
                 for c in r["coefficients"]}
        return ModelDraft(title or r["title"], [], [], [], {}, coefs, [phenomenon], [r["source"]],
                          relation=r | {"coefficients": tuple(coefs.values())})
    if phenomenon not in PHENOMENA:
        raise KeyError(f"no laws for {phenomenon!r}; known: {sorted(PHENOMENA) + sorted(RELATIONS)}")
    coords = ["x", "y", "z"][:dims] + ([] if steady else ["t"])
    laws = [LAWS[k] for k in PHENOMENA[phenomenon]]
    fields: dict[str, str] = {}
    coefs: dict[str, Coefficient] = {}
    pdes: list[str] = []
    for law in laws:
        fields |= law.fields
        for c in law.coefficients:
            coefs.setdefault(c.name, c)
        for eq in law.equations:
            if not steady:
                pdes.append(_drop_absent(eq, coords))
            else:
                pdes.append(_drop_absent(re.sub(r"\b[A-Za-z]+_t\b", "0", eq), coords))
    if "T" in fields and "rho*g*beta*(T - T0)" in " ".join(pdes) and dims < 3:
        pdes = [p.replace(" + rho*g*beta*(T - T0)", "") for p in pdes]
        last = {2: "v", 1: "u"}[dims]                   # buoyancy acts along the last spatial axis
        pdes = [p + " + rho*g*beta*(T - T0)" if p.startswith(f"rho*({last}_t") else p for p in pdes]
    if dims < 3:
        fields.pop("w", None)
        pdes = [p for p in pdes if not p.startswith("rho*(w_t")]
        if dims < 2:
            fields.pop("v", None)
            pdes = [p for p in pdes if not p.startswith("rho*(v_t")]
    conds = _default_conditions(phenomenon, coords, fields, steady)
    for f, cs in (faces or {}).items():            # the caller's conditions replace every default one of that field
        conds = [c for c in conds if not re.search(rf"\b{f}(_[a-z]+)?\b", c["equation"])] + list(cs)
    conds = [{k: v for k, v in c.items() if k != "fields"} for c in conds]
    for c in _condition_coefficients(phenomenon):
        coefs.setdefault(c.name, c)
    for k, v in values.items():
        if k in coefs:
            coefs[k] = replace(coefs[k], value=float(v), source="user")
    draft = ModelDraft(title or phenomenon, coords, pdes, conds, fields, coefs, [law.id for law in laws],
                       list(dict.fromkeys(law.source for law in laws)))
    draft.assumptions = _assumptions(phenomenon, values) + _applied(phenomenon, dims, steady)
    return draft


def _applied(phenomenon: str, dims: int, steady: bool) -> list[Assumption]:
    """The reductions the draft makes, stated so none is silent."""
    out = []
    if dims < 3:
        out.append(Assumption("planar" if dims == 2 else "one_dimensional",
                              f"{dims}-D: nothing varies along the {'z axis' if dims == 2 else 'y and z axes'}",
                              "the terms along the absent axes", "extent along the absent axes vs the solved ones",
                              "the body or room is long (or uniform) along them", "the caller's dims", applied=True))
    if steady:
        out.append(Assumption("steady", "Steady state: no time derivative", "the time derivatives and the initial "
                              "conditions", "time constant vs the time scale of interest",
                              "the inputs are constant for many time constants", "the caller's steady=True",
                              applied=True))
    if phenomenon == "room_air_species":
        out.append(Assumption("full_face_openings", "Supply on the whole x_min wall, exhaust on the whole x_max wall",
                              "the size and position of the real openings (a box domain has only whole faces)",
                              "opening area vs wall area", "for a real diffuser or grille, mesh the room with its "
                              "openings (twin_step) and keep these equations", "box-domain conditions of "
                              "symbolic_pde.text", applied=True))
    return out


def _condition_coefficients(phenomenon: str) -> list[Coefficient]:
    if phenomenon in ("battery_cell_thermal", "solid_heat"):
        return [Coefficient("T0", "K", "initial temperature of the body")]
    if phenomenon == "room_air_species":
        return [Coefficient("Uin", "m/s", "supply air velocity (from the air changes: Uin = ACH*Vroom/Ain)"),
                Coefficient("Tin", "K", "supply air temperature"), Coefficient("cin", "mol/m^3", "species "
                                                                                        "concentration of the "
                                                                                        "supply air"),
                Coefficient("Twall", "K", "wall temperature"), Coefficient("c0", "mol/m^3", "initial "
                                                                                        "concentration")]
    return []


def _default_conditions(phenomenon: str, coords: list[str], fields: dict, steady: bool) -> list[dict]:
    spatial = [c for c in coords if c != "t"]
    out: list[dict] = []
    if phenomenon in ("battery_cell_thermal", "solid_heat"):
        for x in spatial:          # convection on every face: -k dT/dn = h (T - Tamb), n outward
            out.append({"kind": "robin", "on": f"{x}_min", "equation": f"k*T_{x} = h*(T - Tamb)", "fields": ["T"]})
            out.append({"kind": "robin", "on": f"{x}_max", "equation": f"-k*T_{x} = h*(T - Tamb)", "fields": ["T"]})
        if not steady:
            out.append({"kind": "initial", "on": "t_min", "equation": "T = T0", "fields": ["T"]})
        return out
    if phenomenon == "room_air_species":
        vel = [v for v in ("u", "v", "w") if v in fields]
        for x in spatial:
            for side in ("min", "max"):
                face = f"{x}_{side}"
                if face == "x_min":            # supply opening
                    out += [{"kind": "dirichlet", "on": face, "equation": "u = Uin"}] + [
                        {"kind": "dirichlet", "on": face, "equation": f"{v} = 0"} for v in vel[1:]]
                    out += [{"kind": "dirichlet", "on": face, "equation": "T = Tin"},
                            {"kind": "dirichlet", "on": face, "equation": "c = cin"}]
                elif face == "x_max":          # exhaust: zero gauge pressure, fully developed
                    out += [{"kind": "dirichlet", "on": face, "equation": "p = 0"}] + [
                        {"kind": "neumann", "on": face, "equation": f"{v}_x = 0"} for v in vel]
                    out += [{"kind": "neumann", "on": face, "equation": "T_x = 0"},
                            {"kind": "neumann", "on": face, "equation": "c_x = 0"}]
                else:                           # walls: no slip, wall temperature, impermeable
                    out += [{"kind": "dirichlet", "on": face, "equation": f"{v} = 0"} for v in vel]
                    out += [{"kind": "dirichlet", "on": face, "equation": "T = Twall"},
                            {"kind": "neumann", "on": face, "equation": f"c_{x} = 0"}]
        if not steady:
            out += [{"kind": "initial", "on": "t_min", "equation": f"{v} = 0"} for v in vel]
            out += [{"kind": "initial", "on": "t_min", "equation": "T = Tin"},
                    {"kind": "initial", "on": "t_min", "equation": "c = c0"}]
        return out
    return out


def _assumptions(phenomenon: str, values: dict[str, float]) -> list[Assumption]:
    if phenomenon in ("battery_cell_thermal", "solid_heat"):
        h, k, lc = values.get("h"), values.get("k"), values.get("Lc")
        bi = h * lc / k if h and k and lc else None
        return [Assumption("lumped", "Lumped body: one temperature for the whole cell",
                           "the conduction inside the body (lap(T)) and its temperature gradients",
                           "Bi = h*Lc/k, Lc = volume/surface", "Bi < 0.1",
                           "Incropera et al., 7th ed., eq. 5.10 and section 5.2", bi,
                           None if bi is None else bool(bi < 0.1))]
    if phenomenon == "room_air_species":
        return [Assumption("well_mixed", "Well-mixed room: one concentration for the whole room",
                           "the flow field and the spatial transport of the species",
                           "ventilation effectiveness from the flow, or a measured mixing factor", "close to 1",
                           "ASHRAE Standard 62.1, zone air distribution effectiveness", None, None)]
    return []


def known() -> list[str]:
    return sorted(PHENOMENA) + sorted(RELATIONS)


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)
