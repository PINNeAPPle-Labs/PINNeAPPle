"""``pp.transforms``: composable, invertible transforms for tables, datasets and problems, with provenance.

A transform works on a *table* (``{name: array or tensor}``, columns for coordinates and fields), on a
:class:`~pinneapple_core.data.PhysicsDataset`, and, for the scaling transforms, on a ``PhysicalProblem``::

    nd = Nondimensionalize(problem)               # coordinates to [0, 1], fields to O(1), PDE parameters rescaled
    small = nd.apply(problem)                     # a new problem; small.metadata["transforms"] records what was done
    solve = nd.pull_back(solution.predict)        # predictions in the original coordinates and units

    t = Scale({"x": 2.0}) >> Coordinate("log", ("t",), ("logt",)) >> FourierFeatures(("x",), 8)
    out = t.forward(table); back = t.inverse(out)   # back == table

``a >> b`` composes (inverse runs in reverse order); a composition is invertible when every step is. Every ``apply`` appends a
record ``{name, params, invertible, input/output fingerprint}`` to the object's provenance (``problem.metadata["transforms"]``,
``dataset.provenance``).

Transforms: :class:`Scale`, :class:`Nondimensionalize`, :class:`Coordinate` (polar, cylindrical, spherical, log),
:class:`Symmetry` (reflection, rotation, with vector fields), :class:`Periodic` (cos/sin embedding of a periodic coordinate),
:class:`FourierFeatures` (random Fourier features). Problem-level support is for the scaling transforms only: the others act on
data or on model inputs, and ``apply(problem)`` says so rather than guessing.
"""
from __future__ import annotations

import dataclasses
import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np

Table = Dict[str, Any]

__all__ = ["Transform", "Compose", "Scale", "Nondimensionalize", "Coordinate", "Symmetry", "Periodic", "FourierFeatures",
           "NotInvertible", "list_pde_rules"]


class NotInvertible(RuntimeError):
    """The transform (or one step of a composition) has no inverse."""


def _lib(a):
    if type(a).__module__.startswith("torch"):
        import torch

        return torch
    return np


def _atan2(lib, y, x):
    return lib.atan2(y, x) if lib is not np else np.arctan2(y, x)


def _need(table: Mapping[str, Any], names: Sequence[str], who: str) -> None:
    missing = [n for n in names if n not in table]
    if missing:
        raise KeyError(f"{who}: columns {missing} not in the table (have {list(table)})")


# -- base --------------------------------------------------------------------------
class Transform:
    """Base class. Subclasses implement ``forward`` / ``inverse`` on tables and describe themselves in ``params()``."""

    invertible = True
    name = "transform"

    def forward(self, table: Table) -> Table:
        raise NotImplementedError

    def inverse(self, table: Table) -> Table:
        raise NotInvertible(f"{self.name} has no inverse")

    def params(self) -> Dict[str, Any]:
        return {}

    def record(self) -> Dict[str, Any]:
        """JSON-friendly description of this transform."""
        return {"name": self.name, "params": self.params(), "invertible": self.invertible}

    def steps(self) -> List["Transform"]:
        return [self]

    def __rshift__(self, other: "Transform") -> "Compose":
        return Compose(self.steps() + other.steps())

    def __call__(self, obj):
        return self.apply(obj)

    # -- application to tables, datasets and problems
    def apply(self, obj):
        """Apply to a table (returns a table), a ``PhysicsDataset`` or a ``PhysicalProblem`` (return new objects with the
        record appended to their provenance)."""
        from .data import PhysicsDataset

        if isinstance(obj, PhysicsDataset):
            return self._apply_dataset(obj)
        if hasattr(obj, "to_pde_spec") and hasattr(obj, "fingerprint"):
            return self._apply_problem(obj)
        if isinstance(obj, Mapping):
            return self.forward(dict(obj))
        raise TypeError(f"cannot apply a transform to {type(obj).__name__}; pass a table, a PhysicsDataset or a PhysicalProblem")

    def _apply_problem(self, problem):
        raise NotImplementedError(f"{self.name} acts on data or model inputs and has no problem-level form; "
                                  f"apply it to a table or a dataset (problem-level: Scale, Nondimensionalize)")

    def _apply_dataset(self, ds):
        from .data import PhysicsDataset
        import torch

        if not ds.coords:
            raise ValueError("the dataset needs coords= (and fields= for targets) so columns can be named")
        table: Table = {c: ds.inputs[:, i] for i, c in enumerate(ds.coords)}
        if ds.targets is not None:
            if not ds.fields:
                raise ValueError("the dataset has targets but no fields= names")
            tg = ds.targets.reshape(ds.targets.shape[0], -1)
            table.update({f: tg[:, i] for i, f in enumerate(ds.fields)})
        out = self.forward(table)
        new_coords = [c for c in out if c not in (ds.fields or ())]
        new_fields = [c for c in out if c in (ds.fields or ())] if ds.fields else []
        # columns not originally fields are inputs; the transform may rename or add coordinate-like columns
        inputs = torch.stack([torch.as_tensor(out[c]) for c in new_coords], dim=1)
        targets = None
        if new_fields:
            targets = torch.stack([torch.as_tensor(out[f]) for f in new_fields], dim=1)
        return PhysicsDataset(inputs, targets, coords=new_coords, fields=new_fields or None, domain=None, name=ds.name,
                              provenance=list(ds.provenance) + [self.record()])

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.params()})"


class Compose(Transform):
    """Sequence of transforms; ``forward`` runs them in order and ``inverse`` in reverse."""

    name = "compose"

    def __init__(self, steps: Sequence[Transform]) -> None:
        flat: List[Transform] = []
        for s in steps:
            flat.extend(s.steps())
        if not flat:
            raise ValueError("Compose needs at least one transform")
        self._steps = flat

    @property
    def invertible(self) -> bool:  # type: ignore[override]
        return all(s.invertible for s in self._steps)

    def steps(self) -> List[Transform]:
        return list(self._steps)

    def forward(self, table: Table) -> Table:
        for s in self._steps:
            table = s.forward(table)
        return table

    def inverse(self, table: Table) -> Table:
        bad = [s.name for s in self._steps if not s.invertible]
        if bad:
            raise NotInvertible(f"cannot invert: {bad} are not invertible")
        for s in reversed(self._steps):
            table = s.inverse(table)
        return table

    def record(self) -> Dict[str, Any]:
        return {"name": "compose", "steps": [s.record() for s in self._steps], "invertible": self.invertible}

    def provenance(self) -> List[Dict[str, Any]]:
        return [s.record() for s in self._steps]

    def _apply_problem(self, problem):
        for s in self._steps:
            problem = s._apply_problem(problem)
        return problem

    def _apply_dataset(self, ds):
        for s in self._steps:
            ds = s._apply_dataset(ds)
        return ds

    def params(self) -> Dict[str, Any]:
        return {"steps": [s.name for s in self._steps]}


# -- scaling ----------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _PDERule:
    """How a PDE's parameters change when coordinates and fields are scaled, and how its source term scales."""
    adjust: Callable[[Dict[str, float], Dict[str, float], Dict[str, float], Sequence[str], Sequence[str]], Dict[str, float]]
    source: Callable[[Dict[str, float], Dict[str, float], Sequence[str], Sequence[str]], float]
    allow_field_shift: bool = True


def _spatial(scale: Mapping[str, float], coords: Sequence[str], who: str) -> float:
    sp = [scale.get(c, 1.0) for c in coords if c != "t"]
    if not sp:
        return 1.0
    if max(sp) / min(sp) - 1.0 > 1e-9:
        raise ValueError(f"{who}: anisotropic spatial scaling {sp} changes the operator; use one length scale for all space coordinates")
    return sp[0]


def _field_scale(scale: Mapping[str, float], fields: Sequence[str]) -> float:
    return scale.get(fields[0], 1.0)


def _burgers(params, scale, shift, coords, fields):
    L, T, U = _spatial(scale, coords, "burgers"), scale.get("t", 1.0), _field_scale(scale, fields)
    if abs(U * T / L - 1.0) > 1e-9:
        raise ValueError(f"burgers: the convective term u u_x keeps its coefficient only if U*T/L = 1 (got {U * T / L:g}); "
                         f"choose the velocity scale U = L/T")
    if shift.get(fields[0], 0.0) != 0.0:
        raise ValueError("burgers is not invariant under a shift of u (Galilean); use no field shift")
    return {"nu": params["nu"] * T / L ** 2}


def _heat(params, scale, shift, coords, fields):
    L, T = _spatial(scale, coords, "heat_equation"), scale.get("t", 1.0)
    key = "alpha" if "alpha" in params else "kappa"
    return {key: params[key] * T / L ** 2}


def _wave(params, scale, shift, coords, fields):
    L, T = _spatial(scale, coords, "wave_equation"), scale.get("t", 1.0)
    return {"c": params["c"] * T / L}


_RULES: Dict[str, _PDERule] = {
    "laplace": _PDERule(lambda p, s, h, c, f: (_spatial(s, c, "laplace"), {})[1], lambda s, h, c, f: 1.0),
    "poisson": _PDERule(lambda p, s, h, c, f: (_spatial(s, c, "poisson"), {})[1],
                        lambda s, h, c, f: _spatial(s, c, "poisson") ** 2 / _field_scale(s, f)),     # f' = f L^2 / U
    "heat_equation": _PDERule(_heat, lambda s, h, c, f: s.get("t", 1.0) / _field_scale(s, f)),        # q' = q T / U
    "wave_equation": _PDERule(_wave, lambda s, h, c, f: s.get("t", 1.0) ** 2 / _field_scale(s, f)),   # f' = f T^2 / U
    "burgers": _PDERule(_burgers, lambda s, h, c, f: 1.0, allow_field_shift=False),
}


def list_pde_rules() -> List[str]:
    """PDE kinds whose parameters ``Scale`` / ``Nondimensionalize`` know how to rescale."""
    return sorted(_RULES)


class Scale(Transform):
    """Affine scaling of named columns: ``y = (x - shift) / scale``. On a ``PhysicalProblem`` it rescales the coordinates, the
    domain, the conditions and, for the PDEs in :func:`list_pde_rules`, the PDE parameters; PDEs without a rule raise rather than
    be left inconsistent. A source term is rescaled with :meth:`source`."""

    name = "scale"

    def __init__(self, scale: Mapping[str, float], shift: Optional[Mapping[str, float]] = None) -> None:
        self.scale = {k: float(v) for k, v in scale.items()}
        self.shift = {k: float(v) for k, v in (shift or {}).items()}
        bad = [k for k, v in self.scale.items() if not (v > 0 and math.isfinite(v))]
        if bad:
            raise ValueError(f"scales must be positive and finite: {bad}")
        for k in self.shift:
            self.scale.setdefault(k, 1.0)

    def params(self) -> Dict[str, Any]:
        return {"scale": dict(self.scale), "shift": dict(self.shift)}

    def forward(self, table: Table) -> Table:
        out = dict(table)
        for k, s in self.scale.items():
            if k in out:
                out[k] = (out[k] - self.shift.get(k, 0.0)) / s
        return out

    def inverse(self, table: Table) -> Table:
        out = dict(table)
        for k, s in self.scale.items():
            if k in out:
                out[k] = out[k] * s + self.shift.get(k, 0.0)
        return out

    # -- problems
    def source(self, fn: Callable, problem) -> Callable:
        """Rescale a source function ``fn(X, ctx)`` (coordinates and values in original units) to the transformed problem:
        the new function takes the new coordinates and returns the new source values."""
        kind = problem.pde.kind
        if kind not in _RULES:
            raise ValueError(f"no scaling rule for PDE '{kind}'; known: {list_pde_rules()}")
        factor = _RULES[kind].source(self.scale, self.shift, problem.coords, problem.fields)
        sc = np.array([self.scale.get(c, 1.0) for c in problem.coords])
        sh = np.array([self.shift.get(c, 0.0) for c in problem.coords])
        return lambda X, ctx=None: np.asarray(fn(np.asarray(X) * sc + sh, ctx)) * factor

    def pull_back(self, predict: Callable[[np.ndarray], np.ndarray], coords: Sequence[str], fields: Sequence[str]):
        """Wrap ``predict(X) -> Y`` of the *transformed* problem so it takes the original coordinates and returns the
        original fields. ``coords`` and ``fields`` name the original columns, in column order (``problem.coords`` and
        ``problem.fields`` of the original problem)."""
        def original(X):
            X = np.asarray(X, dtype=np.float64)
            t = self.forward({c: X[:, i] for i, c in enumerate(coords)})
            Y = np.asarray(predict(np.stack([t[c] for c in coords], axis=1)), dtype=np.float64).reshape(len(X), -1)
            back = self.inverse({f: Y[:, i] for i, f in enumerate(fields)})
            return np.stack([back[f] for f in fields], axis=1)
        return original

    def _apply_problem(self, problem):
        from dataclasses import replace

        if problem.pde is None:
            raise ValueError("the problem has no PDE to rescale")
        kind = problem.pde.kind
        if kind not in _RULES:
            raise ValueError(f"no scaling rule for PDE '{kind}'; known: {list_pde_rules()}. A silent coordinate change would "
                             f"leave its coefficients wrong.")
        rule = _RULES[kind]
        coords, fields = tuple(problem.coords), tuple(problem.fields)
        if not rule.allow_field_shift and any(self.shift.get(f, 0.0) for f in fields):
            raise ValueError(f"{kind} is not invariant under a shift of its field (Galilean shift for Burgers)")
        sc = np.array([self.scale.get(c, 1.0) for c in coords])
        sh = np.array([self.shift.get(c, 0.0) for c in coords])
        fs = {f: self.scale.get(f, 1.0) for f in fields}
        fh = {f: self.shift.get(f, 0.0) for f in fields}
        # PDE parameters (current values: parameters override pde.params on export)
        values = dict(problem.pde.params)
        values.update(problem.parameter_values())
        new_vals = rule.adjust(values, self.scale, self.shift, coords, fields) or {}
        params = dict(problem.parameters)
        pde_params = dict(problem.pde.params)
        for k, v in new_vals.items():
            ratio = v / values[k] if values.get(k) else 1.0
            if k in params:
                p = params[k]
                params[k] = replace(p, value=v, bounds=None if p.bounds is None else (p.bounds[0] * ratio, p.bounds[1] * ratio))
            else:
                pde_params[k] = v
        conds = tuple(self._condition(c, coords, fields, sc, sh, fs, fh) for c in problem.conditions)
        bounds = {c: ((lo - self.shift.get(c, 0.0)) / self.scale.get(c, 1.0), (hi - self.shift.get(c, 0.0)) / self.scale.get(c, 1.0))
                  for c, (lo, hi) in problem.domain_bounds.items()}
        ranges = {f: ((lo - fh.get(f, 0.0)) / fs.get(f, 1.0), (hi - fh.get(f, 0.0)) / fs.get(f, 1.0))
                  for f, (lo, hi) in problem.field_ranges.items()}
        units = dict(problem.units)
        for n in list(coords) + list(fields):
            if n in units and (n in self.scale or n in self.shift):
                units[n] = "1"
        before = problem.fingerprint()
        meta = dict(problem.metadata)
        out = replace(problem, name=f"{problem.name}_{self.name}", conditions=conds, domain_bounds=bounds,
                      field_ranges=ranges, units=units, parameters=params, pde=replace(problem.pde, params=pde_params),
                      reference_solver={}, metadata=meta)
        rec = self.record()
        rec.update({"input_fingerprint": before, "output_fingerprint": out.fingerprint(),
                    "original_units": {n: problem.units[n] for n in units if n in problem.units and units[n] != problem.units[n]},
                    "note": "reference_solver dropped: it solves the original problem"})
        meta["transforms"] = list(meta.get("transforms", [])) + [rec]
        return replace(out, metadata=meta)

    def _condition(self, c, coords, fields, sc, sh, fs, fh):
        from dataclasses import replace

        if c.kind not in ("dirichlet", "initial", "data", "neumann"):
            raise ValueError(f"condition '{c.name}' is of kind '{c.kind}', which Scale cannot rescale (supported: dirichlet, initial, data, neumann)")
        if c.kind == "neumann" and c.order != 1:
            raise ValueError(f"condition '{c.name}': only first-order Neumann conditions can be rescaled")
        fsc = np.array([fs.get(f, 1.0) for f in c.fields])
        fsh = np.array([fh.get(f, 0.0) for f in c.fields])
        sel = c.selector
        if c.selector_type == "callable":
            old_sel = c.selector
            sel = lambda X, ctx=None, _o=old_sel: _o(np.asarray(X) * sc + sh, ctx)
        old_val = c.value_fn
        if c.kind == "neumann":                         # normal derivative: (dU/dL) times the field scale over the length scale
            L = _spatial(dict(zip(coords, sc)), coords, "neumann")
            val = lambda X, ctx=None, _o=old_val: np.asarray(_o(np.asarray(X) * sc + sh, ctx)) * L / fsc
        else:
            val = lambda X, ctx=None, _o=old_val: (np.asarray(_o(np.asarray(X) * sc + sh, ctx)) - fsh) / fsc
        return replace(c, selector=sel, value_fn=val)


class Nondimensionalize(Scale):
    """Scale a problem to dimensionless variables. ``Nondimensionalize(problem)`` takes each coordinate's extent as its length
    scale (so coordinates run over [0, 1]), the time extent as ``T``, and a field scale from ``field_ranges`` (or 1); for Burgers
    the velocity scale is ``L / T``. Override with ``length=``, ``time=``, ``field=`` or give ``scale=`` / ``shift=`` explicitly."""

    name = "nondimensionalize"

    def __init__(self, problem=None, *, length: Optional[float] = None, time: Optional[float] = None,
                 field: Optional[Union[float, Mapping[str, float]]] = None, scale: Optional[Mapping[str, float]] = None,
                 shift: Optional[Mapping[str, float]] = None) -> None:
        if problem is None:
            if scale is None:
                raise ValueError("pass a problem to derive the scales from, or scale= explicitly")
            super().__init__(scale, shift)
            return
        sc: Dict[str, float] = {}
        sh: Dict[str, float] = {}
        for c in problem.coords:
            lo, hi = problem.domain_bounds[c]
            ext = float(hi - lo)
            sc[c] = float(time if c == "t" and time else (length if c != "t" and length else ext))
            sh[c] = float(lo)
        if problem.pde is not None and problem.pde.kind == "burgers" and "t" in sc:
            spatial = [c for c in problem.coords if c != "t"]
            sc[problem.fields[0]] = (field if isinstance(field, (int, float)) else sc[spatial[0]] / sc["t"])
        else:
            for f in problem.fields:
                if isinstance(field, Mapping) and f in field:
                    sc[f] = float(field[f])
                elif isinstance(field, (int, float)):
                    sc[f] = float(field)
                elif f in problem.field_ranges:
                    lo, hi = problem.field_ranges[f]
                    sc[f] = float(max(abs(lo), abs(hi)) or 1.0)
                else:
                    sc[f] = 1.0
        if scale:
            sc.update({k: float(v) for k, v in scale.items()})
        if shift:
            sh.update({k: float(v) for k, v in shift.items()})
        super().__init__(sc, sh)


# -- coordinate changes -----------------------------------------------------------------
class Coordinate(Transform):
    """Change of coordinates on table columns. ``kind``: ``polar`` ((x, y) -> (r, theta)), ``cylindrical`` ((x, y, z) -> (r, theta, z)),
    ``spherical`` ((x, y, z) -> (r, theta from +z, phi)) and ``log`` (x -> ln x, x > 0). Vector fields are not transformed
    (their components stay Cartesian); give ``center`` to shift the origin first."""

    name = "coordinate"
    _ARITY = {"polar": 2, "cylindrical": 3, "spherical": 3, "log": 1}

    def __init__(self, kind: str, inputs: Sequence[str], outputs: Sequence[str], center: Optional[Sequence[float]] = None) -> None:
        if kind not in self._ARITY:
            raise ValueError(f"kind must be one of {sorted(self._ARITY)}")
        if len(inputs) != self._ARITY[kind] or len(outputs) != self._ARITY[kind]:
            raise ValueError(f"{kind} maps {self._ARITY[kind]} columns to {self._ARITY[kind]} columns")
        self.kind, self.inputs, self.outputs = kind, tuple(inputs), tuple(outputs)
        self.center = None if center is None else tuple(float(c) for c in center)
        if self.center is not None and len(self.center) != len(self.inputs):
            raise ValueError("center must have one value per input column")

    def params(self) -> Dict[str, Any]:
        return {"kind": self.kind, "inputs": list(self.inputs), "outputs": list(self.outputs), "center": self.center}

    def _c(self, i: int) -> float:
        return 0.0 if self.center is None else self.center[i]

    def forward(self, table: Table) -> Table:
        _need(table, self.inputs, "Coordinate")
        out = {k: v for k, v in table.items() if k not in self.inputs}
        a = [table[n] - self._c(i) for i, n in enumerate(self.inputs)]
        lib = _lib(a[0])
        if self.kind == "polar" or self.kind == "cylindrical":
            r = lib.sqrt(a[0] ** 2 + a[1] ** 2)
            vals = [r, _atan2(lib, a[1], a[0])] + a[2:]
        elif self.kind == "spherical":
            r = lib.sqrt(a[0] ** 2 + a[1] ** 2 + a[2] ** 2)
            safe = lib.where(r == 0, lib.ones_like(r), r)
            vals = [r, lib.acos(a[2] / safe) if lib is not np else np.arccos(a[2] / safe), _atan2(lib, a[1], a[0])]
        else:
            if bool((a[0] <= 0).any()):
                raise ValueError("log needs positive values")
            vals = [lib.log(a[0])]
        out.update(dict(zip(self.outputs, vals)))
        return out

    def inverse(self, table: Table) -> Table:
        _need(table, self.outputs, "Coordinate")
        out = {k: v for k, v in table.items() if k not in self.outputs}
        b = [table[n] for n in self.outputs]
        lib = _lib(b[0])
        if self.kind in ("polar", "cylindrical"):
            vals = [b[0] * lib.cos(b[1]), b[0] * lib.sin(b[1])] + b[2:]
        elif self.kind == "spherical":
            vals = [b[0] * lib.sin(b[1]) * lib.cos(b[2]), b[0] * lib.sin(b[1]) * lib.sin(b[2]), b[0] * lib.cos(b[1])]
        else:
            vals = [lib.exp(b[0])]
        vals = [v + self._c(i) for i, v in enumerate(vals)]
        out.update(dict(zip(self.inputs, vals)))
        return out


# -- symmetry --------------------------------------------------------------------------------
class Symmetry(Transform):
    """A symmetry operation on data, for augmentation or to test equivariance.

    ``Symmetry.reflect("x", coords=("x", "y"), vectors=[("u", "v")])`` mirrors ``x -> 2c - x`` and flips the matching component of
    each vector field; ``Symmetry.rotate(angle, coords=("x", "y"), vectors=[("u", "v")])`` rotates positions about ``center`` and the
    vector components by the same angle. Both are invertible (a reflection is its own inverse)."""

    name = "symmetry"

    def __init__(self, kind: str, coords: Sequence[str], vectors: Sequence[Sequence[str]] = (), *, axis: Optional[str] = None,
                 angle: float = 0.0, center: Optional[Sequence[float]] = None) -> None:
        if kind not in ("reflect", "rotate"):
            raise ValueError("kind must be 'reflect' or 'rotate'")
        self.kind, self.coords, self.vectors = kind, tuple(coords), tuple(tuple(v) for v in vectors)
        self.axis, self.angle = axis, float(angle)
        self.center = tuple(float(c) for c in center) if center is not None else (0.0,) * len(self.coords)
        if len(self.center) != len(self.coords):
            raise ValueError("center must have one value per coordinate")
        if kind == "reflect" and axis not in self.coords:
            raise ValueError(f"axis {axis!r} must be one of the coordinates {self.coords}")
        if kind == "rotate" and len(self.coords) != 2:
            raise ValueError("rotate acts on a pair of coordinates")
        for v in self.vectors:
            if len(v) != len(self.coords):
                raise ValueError(f"each vector needs {len(self.coords)} components, got {v}")

    @classmethod
    def reflect(cls, axis: str, coords: Sequence[str], vectors: Sequence[Sequence[str]] = (), center: Optional[Sequence[float]] = None):
        return cls("reflect", coords, vectors, axis=axis, center=center)

    @classmethod
    def rotate(cls, angle: float, coords: Sequence[str], vectors: Sequence[Sequence[str]] = (), center: Optional[Sequence[float]] = None):
        return cls("rotate", coords, vectors, angle=angle, center=center)

    def params(self) -> Dict[str, Any]:
        return {"kind": self.kind, "coords": list(self.coords), "vectors": [list(v) for v in self.vectors], "axis": self.axis,
                "angle": self.angle, "center": list(self.center)}

    def _move(self, table: Table, angle: float) -> Table:
        _need(table, self.coords, "Symmetry")
        out = dict(table)
        if self.kind == "reflect":
            i = self.coords.index(self.axis)
            c = self.center[i]
            out[self.axis] = 2 * c - table[self.axis]
            for v in self.vectors:
                _need(table, v, "Symmetry")
                out[v[i]] = -table[v[i]]
            return out
        cs, sn = math.cos(angle), math.sin(angle)
        x, y = table[self.coords[0]] - self.center[0], table[self.coords[1]] - self.center[1]
        out[self.coords[0]], out[self.coords[1]] = cs * x - sn * y + self.center[0], sn * x + cs * y + self.center[1]
        for v in self.vectors:
            _need(table, v, "Symmetry")
            out[v[0]], out[v[1]] = cs * table[v[0]] - sn * table[v[1]], sn * table[v[0]] + cs * table[v[1]]
        return out

    def forward(self, table: Table) -> Table:
        return self._move(table, self.angle)

    def inverse(self, table: Table) -> Table:
        return self._move(table, -self.angle)


# -- periodic embedding and Fourier features ------------------------------------------------------
class Periodic(Transform):
    """Replace a periodic coordinate by ``cos`` / ``sin`` of its phase, so a network is periodic in it by construction:
    ``x -> {x_cos1, x_sin1, ..., x_cosK, x_sinK}`` with phase ``2 pi k (x - lo) / period``. Inverse (from the first harmonic) maps back
    into ``[lo, lo + period)``."""

    name = "periodic"

    def __init__(self, coord: str, period: float, harmonics: int = 1, lo: float = 0.0) -> None:
        if period <= 0 or harmonics < 1:
            raise ValueError("period must be positive and harmonics >= 1")
        self.coord, self.period, self.harmonics, self.lo = coord, float(period), int(harmonics), float(lo)

    def params(self) -> Dict[str, Any]:
        return {"coord": self.coord, "period": self.period, "harmonics": self.harmonics, "lo": self.lo}

    def _names(self, k: int) -> Tuple[str, str]:
        return f"{self.coord}_cos{k}", f"{self.coord}_sin{k}"

    def forward(self, table: Table) -> Table:
        _need(table, [self.coord], "Periodic")
        out = {k: v for k, v in table.items() if k != self.coord}
        x = table[self.coord]
        lib = _lib(x)
        phase = 2 * math.pi * (x - self.lo) / self.period
        for k in range(1, self.harmonics + 1):
            c, s = self._names(k)
            out[c], out[s] = lib.cos(k * phase), lib.sin(k * phase)
        return out

    def inverse(self, table: Table) -> Table:
        c, s = self._names(1)
        _need(table, [c, s], "Periodic")
        lib = _lib(table[c])
        out = {k: v for k, v in table.items() if k not in {n for j in range(1, self.harmonics + 1) for n in self._names(j)}}
        phase = _atan2(lib, table[s], table[c]) % (2 * math.pi)
        out[self.coord] = phase * self.period / (2 * math.pi) + self.lo
        return out


class FourierFeatures(Transform):
    """Random Fourier features of some columns, ``[cos(2 pi B x), sin(2 pi B x)]`` with ``B ~ N(0, scale^2)`` (Tancik et al.), added as
    ``ff_cos_i`` / ``ff_sin_i`` columns next to the originals; the inverse drops them. ``scale`` sets the frequency band (large for
    high-frequency targets). :meth:`module` returns the same map as a torch layer."""

    name = "fourier_features"

    def __init__(self, coords: Sequence[str], n_features: int = 16, scale: float = 1.0, seed: int = 0, prefix: str = "ff") -> None:
        if n_features < 1:
            raise ValueError("n_features must be >= 1")
        self.coords, self.n_features, self.scale, self.seed, self.prefix = tuple(coords), int(n_features), float(scale), int(seed), prefix
        self.B = np.random.default_rng(seed).normal(0.0, scale, size=(len(self.coords), self.n_features))

    def params(self) -> Dict[str, Any]:
        return {"coords": list(self.coords), "n_features": self.n_features, "scale": self.scale, "seed": self.seed, "prefix": self.prefix}

    def _cols(self) -> List[str]:
        return [f"{self.prefix}_{k}_{i}" for k in ("cos", "sin") for i in range(self.n_features)]

    def embed(self, x):
        """``(N, d) -> (N, 2 n_features)`` for a numpy array or a torch tensor."""
        lib = _lib(x)
        B = self.B if lib is np else lib.as_tensor(self.B, dtype=x.dtype, device=x.device)
        proj = 2 * math.pi * (x @ B)
        return lib.cat([lib.cos(proj), lib.sin(proj)], dim=1) if lib is not np else np.concatenate([np.cos(proj), np.sin(proj)], axis=1)

    def forward(self, table: Table) -> Table:
        _need(table, self.coords, "FourierFeatures")
        lib = _lib(table[self.coords[0]])
        x = lib.stack([table[c] for c in self.coords], dim=1) if lib is not np else np.stack([table[c] for c in self.coords], axis=1)
        feats = self.embed(x)
        out = dict(table)
        for j, name in enumerate(self._cols()):
            out[name] = feats[:, j]
        return out

    def inverse(self, table: Table) -> Table:
        drop = set(self._cols())
        return {k: v for k, v in table.items() if k not in drop}

    def module(self):
        import torch

        outer = self

        class _Layer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.register_buffer("B", torch.as_tensor(outer.B, dtype=torch.get_default_dtype()))

            def forward(self, x):
                proj = 2 * math.pi * (x @ self.B)
                return torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)

        return _Layer()
