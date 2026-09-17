from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Union, Literal

import numpy as np

from .environment_typing import FieldNames

SelectorType = Literal["all", "tag", "callable"]


@dataclass(frozen=True)
class ConditionSpec:
    """
    Generic constraint spec.

    kind:
      - "dirichlet" -> u(x)=g(x)
      - "neumann"   -> n·∇u(x)=g(x)  (order=1, the default), or, when order>1,
                       d^order(u)/d(deriv_coord)^order = g(x) — a plain
                       repeated single-coordinate derivative (NOT a normal-
                       dot-gradient contraction generalized to higher tensor
                       order — deliberately narrow, added specifically for 1D
                       beam moment/shear conditions where there is only one
                       spatial coordinate and no directional ambiguity).
      - "robin"     -> a u + b n·∇u = g
      - "interface" -> coupling condition at a shared multi-domain boundary:
                       value continuity  field_A(x) - field_B(x) = g_value(x)
                       flux continuity   k_A n·∇field_A - k_B n·∇field_B = g_flux(x)
                       (default g_value=g_flux=0, i.e. plain continuity of
                       value and weighted normal flux across the interface,
                       as in conjugate heat transfer or fluid-structure
                       coupling). ``fields`` must be exactly the two field
                       names (field_A, field_B) evaluated on either side of
                       the interface; the flux weights are given via
                       ``interface_coeffs={"k_a":..., "k_b":...}``.
      - "initial"   -> u(x,t0)=g(x)
      - "data"      -> supervised constraint at points

    traction_map (only meaningful for kind="neumann", order<=1):
      Some elasticity presets declare their Neumann targets as TRACTION
      components (e.g. "tx", "ty") rather than a model output field's own
      normal derivative -- physically, traction is n . sigma (the stress
      tensor dotted with the boundary normal), not n . grad(field), and
      "tx"/"ty" are not fields the network ever predicts directly (the
      network predicts displacements ux/uy/uz). Without this mapping the
      compiler has no way to resolve "tx" against the model's own fields
      and raises a plain KeyError.

      ``traction_map`` names, for each traction component in ``fields``,
      which of the PDE's own displacement fields shares its spatial axis,
      e.g. ``{"tx": "ux", "ty": "uy"}``. When set on a "neumann" condition
      whose ``fields`` are not literal model outputs, ``compile_problem``
      builds the full elasticity stress tensor sigma from ALL of the PDE's
      displacement fields (not just the ones named here -- sigma_xy needs
      both ux and uy regardless of which single traction component is
      being matched) via the same lambda/mu constitutive relation the
      interior residual uses, then compares n . sigma (restricted to the
      row selected by each entry's mapped axis) against this condition's
      own ``value_fn`` target -- a real traction-from-stress evaluation,
      not a copy of the declared name. See ``compile.py``'s
      ``_elasticity_traction_from_stress`` for the implementation. This is
      a per-COMPONENT derivation (one vector component of n . sigma along
      a named axis) -- for the full scalar double contraction n . sigma . n
      (e.g. an internal-pressure boundary), see ``normal_stress_field``
      below instead; that is a genuinely different derivation, not a
      special case of this one.

    normal_stress_field (only meaningful for kind="neumann", order<=1):
      Some elasticity presets declare a Neumann target that is a PRESSURE
      MAGNITUDE acting normal to the boundary (e.g. the internal pressure
      of a pressure-vessel wall), not a single traction vector component.
      Physically this is the standard pressure-vessel boundary condition
      n^T . sigma . n = -p_internal (the normal-normal stress contraction,
      NOT one axis-aligned component of n . sigma) -- the minus sign is the
      standard convention that positive internal pressure produces
      COMPRESSIVE normal stress at the wall.

      ``normal_stress_field`` names the single field in ``fields`` that
      carries the pressure MAGNITUDE (e.g. ``"p_normal"``). When set,
      ``compile_problem`` builds sigma exactly as for ``traction_map``
      (same helper, same constitutive relation), then computes the full
      scalar contraction ``n . sigma . n`` and compares it against
      ``-value_fn(...)`` (the value_fn's own output is the pressure
      magnitude; the sign flip is applied by the compiler, not by the
      preset). Mutually exclusive with ``traction_map`` on the same
      condition. See ``compile.py``'s
      ``_elasticity_normal_stress_from_stress`` for the implementation.

    thermal_bc (only meaningful for kind="neumann", order<=1):
      Some thermal presets declare their Neumann/Robin boundary targets as
      a prescribed heat FLUX or a convection law, neither of which is a
      literal model output field (the model only ever outputs the
      temperature field itself, e.g. "T") -- physically:

        prescribed flux (Neumann):        -k * dT/dn = q_heat
        convection (Robin/Newton cooling): -k * dT/dn = h * (T - T_ref)

      where ``dT/dn`` is the boundary-normal derivative of the model's own
      temperature field (via autograd, the same ``norm_dot_grad`` machinery
      the plain Neumann branch already uses) and ``k`` is the PDE's own
      thermal conductivity (``params["k_eff"]`` if present, else
      ``params["k"]``, else 1.0 -- the same lookup order the
      ``heat_equation_steady*``/``heat_equation_transient`` residuals
      already use, so a thermal BC can never silently disagree with the
      interior residual's own conductivity).

      ``thermal_bc = {"kind": "flux", "T_field": "T"}`` for the prescribed-
      flux case (``fields=("q_heat",)``), or
      ``thermal_bc = {"kind": "convection", "T_field": "T"}`` for the
      convection case (``fields=("h", "T_ref")``, order not significant --
      resolved by name). ``T_field`` defaults to ``"T"`` if omitted. This
      mechanism is deliberately generic: it does not hardcode any one
      preset, and applies to any preset declaring this same
      ``q_heat``/``h``+``T_ref`` convention (e.g. ``car_brake_thermal``,
      ``cpu_heatsink_thermal``, ``pcb_thermal``,
      ``industrial_furnace_thermal``, the ``datacenter_*`` thermal
      presets), independent of whether that preset also has real geometry
      wired up yet. See ``compile.py``'s condition-loop
      ``thermal_bc`` branch for the implementation.

    selector:
      - "all": applies to all points of corresponding set
      - "tag": applies to points with ctx["tag_masks"][tag]==True
      - "callable": selector(X, ctx)->bool mask

    value_fn:
      - callable returning values for selected points

    order / deriv_coord:
      - only meaningful for kind="neumann". order=1 (default) is the existing
        n_bc-based normal derivative, unchanged. order>1 requires
        deriv_coord to be set (e.g. "z") and ignores n_bc entirely, computing
        d^order(u)/d(deriv_coord)^order directly via repeated autograd.
    """
    name: str
    kind: str
    fields: FieldNames
    selector_type: SelectorType = "all"
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None
    weight: float = 1.0
    order: int = 1
    deriv_coord: Optional[str] = None
    interface_coeffs: Optional[Dict[str, float]] = None
    traction_map: Optional[Dict[str, str]] = None
    normal_stress_field: Optional[str] = None
    thermal_bc: Optional[Dict[str, str]] = None

    def mask(self, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if self.selector_type == "all":
            return np.ones((X.shape[0],), dtype=bool)

        if self.selector_type == "tag":
            if not isinstance(self.selector, dict) or "tag" not in self.selector:
                raise ValueError(f"Condition '{self.name}' selector_type='tag' requires selector={{'tag':...}}")
            tag = self.selector["tag"]
            tag_masks = ctx.get("tag_masks", {})
            if tag not in tag_masks:
                return np.zeros((X.shape[0],), dtype=bool)
            m = np.asarray(tag_masks[tag], dtype=bool)
            if m.shape[0] != X.shape[0]:
                # If tag masks provided correspond to boundary set only, user must ensure X matches.
                raise ValueError(f"Tag mask '{tag}' shape mismatch: {m.shape} vs X={X.shape}")
            return m

        if self.selector_type == "callable":
            if not callable(self.selector):
                raise ValueError(f"Condition '{self.name}' selector_type='callable' requires callable selector")
            m = self.selector(X, ctx)
            return np.asarray(m, dtype=bool)

        raise ValueError(f"Unknown selector_type: {self.selector_type}")

    def values(self, X: np.ndarray, ctx: Dict[str, Any]) -> np.ndarray:
        if self.value_fn is None:
            # default zeros
            return np.zeros((X.shape[0], len(self.fields)), dtype=np.float32)
        v = self.value_fn(X, ctx)
        v = np.asarray(v, dtype=np.float32)
        if v.ndim == 1:
            v = v[:, None]
        return v


# ---------------------------------------------------------------------------
# Internal helper: build a value_fn from a dict of {field: scalar} values.
# ---------------------------------------------------------------------------

def _value_fn_from_dict(values: Dict[str, float]) -> Callable[[np.ndarray, Dict[str, Any]], np.ndarray]:
    """Return a value_fn that outputs a constant array from a {field: value} dict."""
    vals = list(values.values())
    n_fields = len(vals)
    def _fn(X: np.ndarray, ctx: Dict[str, Any], _vals=vals, _nf=n_fields) -> np.ndarray:
        out = np.zeros((X.shape[0], _nf), dtype=np.float32)
        for i, v in enumerate(_vals):
            out[:, i] = float(v)
        return out
    return _fn


# Convenience typed constructors.
#
# These support two call signatures:
#   1. Simple dict form (paper-style):
#      DirichletBC({"u": 0.0})
#      DirichletBC({"u": 0.0, "v": 0.0})
#
#   2. Full positional/keyword form (internal builder):
#      DirichletBC(name, fields, selector_type, selector, value_fn, weight)


def DirichletBC(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct a Dirichlet boundary condition.

    Simple dict form (paper-style)::

        DirichletBC({"u": 0.0})          # zero Dirichlet for field u
        DirichletBC({"u": 0.0, "v": 0.0})

    Full form::

        DirichletBC("wall", ("u",), "all", None, value_fn, 10.0)
    """
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="dirichlet_bc",
            kind="dirichlet",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="dirichlet",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def NeumannBC(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
    order: int = 1,
    deriv_coord: Optional[str] = None,
    traction_map: Optional[Dict[str, str]] = None,
    normal_stress_field: Optional[str] = None,
    thermal_bc: Optional[Dict[str, str]] = None,
) -> ConditionSpec:
    """Construct a Neumann boundary condition.

    Simple dict form::

        NeumannBC({"u": 1.0})

    order/deriv_coord: order=1 (default) is the standard normal-derivative
    condition (needs batch['n_bc'] at training time). order>1 requires
    deriv_coord (e.g. "z") and computes d^order(u)/d(deriv_coord)^order
    directly — see ConditionSpec's docstring for the exact scope (single-
    coordinate repeated derivative, added for 1D beam moment/shear BCs).

    traction_map: for elasticity presets whose Neumann targets are
    traction components (e.g. "tx"/"ty") rather than literal model
    output fields — see ConditionSpec's docstring for the exact mechanism.

    normal_stress_field: for elasticity presets whose Neumann target is a
    pressure MAGNITUDE (e.g. "p_normal", a pressure-vessel internal
    pressure) rather than a single traction component — see
    ConditionSpec's docstring for the n^T.sigma.n mechanism.

    thermal_bc: for thermal presets whose Neumann/Robin target is a
    prescribed heat flux or a convection law (e.g. fields=("q_heat",) or
    fields=("h","T_ref")) rather than a literal model output field — see
    ConditionSpec's docstring for the exact mechanism.
    """
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="neumann_bc",
            kind="neumann",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
            order=order,
            deriv_coord=deriv_coord,
            traction_map=traction_map,
            normal_stress_field=normal_stress_field,
            thermal_bc=thermal_bc,
        )
    return ConditionSpec(
        name=name,
        kind="neumann",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
        order=order,
        deriv_coord=deriv_coord,
        traction_map=traction_map,
        normal_stress_field=normal_stress_field,
        thermal_bc=thermal_bc,
    )


def RobinBC(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct a Robin boundary condition."""
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="robin_bc",
            kind="robin",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="robin",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def InterfaceBC(
    name: str,
    fields: FieldNames,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
    k_a: float = 1.0,
    k_b: float = 1.0,
) -> ConditionSpec:
    """Construct an interface / coupling boundary condition between two subdomains.

    Enforces, at a shared multi-domain boundary (e.g. a conjugate heat
    transfer interface or a fluid-structure interface), the two conditions
    that are standard for such couplings:

      1. Value continuity:   field_a(x)          - field_b(x)          = g_value(x)
      2. Flux continuity:    k_a * n.grad(field_a) - k_b * n.grad(field_b) = g_flux(x)

    where ``field_a``/``field_b`` are the same physical quantity evaluated
    on either side of the interface (e.g. ``("T_solid", "T_fluid")`` for
    conjugate heat transfer, with ``k_a``/``k_b`` the two sides' thermal
    conductivities), and ``g_value``/``g_flux`` default to zero (plain
    continuity of value and weighted normal flux). ``value_fn``, when
    given, must return an ``(N, 2)`` array: column 0 is the target value
    jump, column 1 is the target flux jump.

    Unlike ``DirichletBC``/``RobinBC``/etc., there is no dict-shorthand
    form: an interface condition inherently couples exactly two distinct
    field names, so ``fields`` must always be given explicitly as
    ``(field_a, field_b)``.

    Parameters
    ----------
    fields:
        Exactly two field names: ``(field_a, field_b)``.
    k_a, k_b:
        Flux weighting coefficients for ``field_a`` and ``field_b``
        respectively (e.g. thermal conductivities, elastic moduli).

    Example::

        InterfaceBC(
            "solid_fluid_interface",
            ("T_solid", "T_fluid"),
            selector_type="tag",
            selector={"tag": "chr_interface"},
            k_a=k_solid, k_b=k_fluid,
        )
    """
    if fields is None or len(fields) != 2:
        raise ValueError(
            "InterfaceBC requires exactly two field names (field_a, field_b) -- "
            "the same physical quantity evaluated on each side of the shared "
            "interface (e.g. ('T_solid', 'T_fluid') for conjugate heat "
            f"transfer); got fields={fields!r}"
        )
    return ConditionSpec(
        name=name,
        kind="interface",
        fields=tuple(fields),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
        interface_coeffs={"k_a": float(k_a), "k_b": float(k_b)},
    )


def InitialCondition(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct an initial condition.

    Simple dict form::

        InitialCondition({"u": 1.0})
    """
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="initial_condition",
            kind="initial",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="initial",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )


def DataConstraint(
    name: Union[str, Dict[str, float]],
    fields: Optional[FieldNames] = None,
    selector_type: SelectorType = "all",
    selector: Optional[Union[Dict[str, Any], Callable[[np.ndarray, Dict[str, Any]], np.ndarray]]] = None,
    value_fn: Optional[Callable[[np.ndarray, Dict[str, Any]], np.ndarray]] = None,
    weight: float = 1.0,
) -> ConditionSpec:
    """Construct a data constraint (supervised loss at specific points)."""
    if isinstance(name, dict):
        values = name
        _fields: FieldNames = tuple(values.keys())
        return ConditionSpec(
            name="data_constraint",
            kind="data",
            fields=_fields,
            selector_type="all",
            selector=None,
            value_fn=_value_fn_from_dict(values),
            weight=weight,
        )
    return ConditionSpec(
        name=name,
        kind="data",
        fields=fields or (),
        selector_type=selector_type,
        selector=selector,
        value_fn=value_fn,
        weight=weight,
    )