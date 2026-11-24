"""``PhysicsTensor``: a ``torch.Tensor`` that knows what it is (#186, roadmap item X2).

A value in this library usually has four things attached to it that a plain tensor forgets: a **name**, a **unit**,
the **domain** it was sampled on and what its **components** mean. Losing them is how ``u`` (m/s) ends up added to
``p`` (Pa) and the mistake surfaces three layers later as a bad number. This module keeps them attached and refuses
the operations where they do not fit:

>>> import pinneapple as pp
>>> u = pp.tensor([[1.0, 2.0], [3.0, 4.0]], name="velocity", units="m/s", coords=("x", "y"))
>>> u.name, u.units
('velocity', 'm/s')
>>> p = pp.tensor([[1.0, 2.0], [3.0, 4.0]], name="pressure", units="Pa")
>>> (u + p).units                                    # doctest: +SKIP
Traceback (most recent call last):
UnitError: cannot add 'm/s' (velocity) to 'Pa' (pressure): the dimensions differ

What it does, exactly:

* ``pp.tensor(...)`` builds one; ``.name``, ``.units``, ``.unit`` (parsed ``Unit``), ``.domain``, ``.coords``,
  ``.components`` and ``.semantics`` read the labels back;
* every ``torch`` operation keeps working and its result **is** a ``PhysicsTensor`` carrying the labels of the first
  labelled operand, so ``u * 2`` and ``u.sum()`` do not drop the unit;
* autograd is untouched -- the result of a labelled operation keeps its ``grad_fn``, and ``loss.backward()`` fills the
  ``.grad`` of a tensor that required grad;
* **additive operations and comparisons between two labelled tensors require the same dimension**
  (``u + p`` and ``u < p`` raise ``UnitError``); adding a plain tensor or a number is allowed and means "same unit";
* multiplying two *dimensioned* tensors is refused rather than guessed: combining ``m/s`` and ``Pa`` correctly needs
  a unit algebra this class does not implement, and a wrong guess is worse than an error. ``.convert("km/h")``
  converts the values to another unit of the same dimension.

Known limit, stated because it is the kind of thing a benchmark finds: the labels are metadata, not a physical type
system. A result computed from ``u`` and ``p`` in a way that goes through an unlabelled tensor (``.numpy()``, a loss
that returns a Python float) comes back unlabelled -- which is honest, since nothing in that path carried the unit.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch

from pinneapple_data.physical_units import Unit, UnitError, parse_unit, try_parse_unit

__all__ = ["PhysicsTensor", "tensor", "UnitError"]


# operations where two labelled operands must agree on the dimension; by name because torch hands the unbound method
# to __torch_function__ and the same name serves the method and the functional form
_ADDITIVE = frozenset({"add", "sub", "add_", "sub_", "subtract", "__add__", "__sub__", "__iadd__", "__isub__",
                       "eq", "ne", "lt", "le", "gt", "ge", "__eq__", "__ne__", "__lt__", "__le__", "__gt__", "__ge__"})
# products of two dimensioned operands would need a unit algebra (m/s * Pa = ?); refusing beats guessing
_MULTIPLICATIVE = frozenset({"mul", "div", "mul_", "div_", "multiply", "divide", "true_divide", "matmul", "mm", "dot",
                             "__mul__", "__truediv__", "__matmul__", "__imul__", "__itruediv__"})


def _unit_of(value: Any) -> Optional[Unit]:
    if isinstance(value, str):
        return parse_unit(value)
    return try_parse_unit(value) if value is not None else None


def _iterate(value: Any):
    if isinstance(value, (tuple, list)):
        for item in value:
            yield from _iterate(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iterate(item)
    else:
        yield value


class PhysicsTensor(torch.Tensor):
    """A tensor with a name, a unit, a domain and component labels. Use :func:`tensor` to build one."""

    # -- construction ----------------------------------------------------------------------------------
    @staticmethod
    def __new__(cls, data: Any, *, name: str = "", units: Any = None, domain: Any = None,
                coords: Optional[Sequence[str]] = None, components: Optional[Sequence[str]] = None,
                time: Any = None, meta: Optional[Mapping[str, Any]] = None,
                requires_grad: Optional[bool] = None) -> "PhysicsTensor":
        base = data if isinstance(data, torch.Tensor) else torch.as_tensor(data)
        if requires_grad is not None or base.is_leaf:
            # a leaf, so autograd fills *its* .grad: as_subclass on a leaf would return a non-leaf view and the
            # gradient would be recorded on the plain tensor behind it
            out = torch.Tensor._make_subclass(cls, base.detach(),
                                              require_grad=base.requires_grad if requires_grad is None
                                              else bool(requires_grad))
        else:
            # an intermediate of a graph: as_subclass keeps the history, so the labels survive a computation
            out = base.as_subclass(cls)
        out._pp_name = str(name or "")
        out._pp_unit = _unit_of(units)
        out._pp_units = str(units) if units is not None else ""
        out._pp_domain = domain
        out._pp_coords = tuple(coords) if coords is not None else None
        out._pp_components = tuple(components) if components is not None else None
        out._pp_time = time
        out._pp_meta = dict(meta or {})
        return out

    # -- labels ----------------------------------------------------------------------------------------
    @property
    def name(self) -> str:
        return getattr(self, "_pp_name", "")

    @property
    def units(self) -> str:
        return getattr(self, "_pp_units", "")

    @property
    def unit(self) -> Optional[Unit]:
        return getattr(self, "_pp_unit", None)

    @property
    def domain(self) -> Any:
        return getattr(self, "_pp_domain", None)

    @property
    def coords(self) -> Optional[Tuple[str, ...]]:
        return getattr(self, "_pp_coords", None)

    @property
    def components(self) -> Optional[Tuple[str, ...]]:
        return getattr(self, "_pp_components", None)

    @property
    def time(self) -> Any:
        return getattr(self, "_pp_time", None)

    @property
    def semantics(self) -> Dict[str, Any]:
        """The labels as one dict -- what to record next to a result so the unit is not lost in a log."""
        return {"name": self.name, "units": self.units, "domain": self.domain, "coords": self.coords,
                "components": self.components, "time": self.time, "meta": dict(getattr(self, "_pp_meta", {})),
                "shape": tuple(self.shape), "dtype": str(self.dtype)}

    def labelled(self) -> bool:
        """Whether anything at all is attached (a tensor with no unit, no name and no domain is just a tensor)."""
        return bool(self.name or self.units or self.domain is not None or self.coords or self.components)

    def without_labels(self) -> torch.Tensor:
        """A plain tensor with the same storage and history -- for a function that does not know about labels."""
        with torch._C.DisableTorchFunctionSubclass():
            return self.as_subclass(torch.Tensor)

    def convert(self, units: str) -> "PhysicsTensor":
        """The same physical quantity written in another unit of the same dimension (``km/h`` from ``m/s``)."""
        if self.unit is None:
            raise UnitError(f"'{self.name or 'tensor'}' has no unit to convert from",
                            fix=f"build it with pp.tensor(..., units='{units}'), or drop convert() and use the "
                                f"values as they are")
        target = parse_unit(units)
        if target.dim != self.unit.dim:
            raise UnitError(f"cannot convert '{self.units}' to '{units}': the dimensions differ",
                            fix="convert only inside one dimension (m -> km, Pa -> bar); a different dimension is "
                                "a different quantity, so build the tensor with the unit you meant")
        with torch._C.DisableTorchFunctionSubclass():
            values = (self.without_labels() * self.unit.factor + self.unit.offset - target.offset) / target.factor
        return PhysicsTensor(values, name=self.name, units=units, domain=self.domain, coords=self.coords,
                             components=self.components, time=self.time, meta=self.semantics["meta"])

    def clone(self, *args: Any, **kwargs: Any) -> "PhysicsTensor":
        out = super().clone(*args, **kwargs)
        return self._relabel(out)

    # -- the physics rule ------------------------------------------------------------------------------
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if not all(issubclass(t, torch.Tensor) for t in types):
            return NotImplemented
        name = getattr(func, "__name__", "")
        if name in _ADDITIVE:
            _same_dimension(name, args)
        elif name in _MULTIPLICATIVE:
            _no_unit_product(name, args)
        with torch._C.DisableTorchFunctionSubclass():
            out = func(*args, **(kwargs or {}))
        return _relabel_like(out, args)

    def _relabel(self, out: torch.Tensor) -> Optional["PhysicsTensor"]:
        return _relabel_like(out, (self,))

    def __repr__(self) -> str:
        inner = repr(self.without_labels())
        if not self.labelled():
            return f"PhysicsTensor({inner})"
        bits = [b for b in (f"name={self.name!r}" if self.name else "",
                            f"units={self.units!r}" if self.units else "") if b]
        if self.coords:
            bits.append(f"coords={self.coords}")
        if self.components:
            bits.append(f"components={self.components}")
        return f"PhysicsTensor({inner}, {', '.join(bits)})"


def _labelled(args: Any) -> list[PhysicsTensor]:
    return [v for v in _iterate(args) if isinstance(v, PhysicsTensor)]


def _same_dimension(op: str, args: Any) -> None:
    """Two labelled operands of an additive or comparative op have to be the same kind of quantity."""
    labelled = [t for t in _labelled(args) if t.unit is not None]
    for first, second in zip(labelled, labelled[1:]):
        if first.unit.dim != second.unit.dim:
            raise UnitError(f"cannot apply '{op.rstrip('_')}' to {_describe(first)} and {_describe(second)}: "
                            f"the dimensions differ",
                            fix="add or compare only quantities of the same dimension; convert one first "
                                "(u.convert('km/h')) or work on the raw values with .without_labels()")


def _describe(t: PhysicsTensor) -> str:
    label = f" ({t.name})" if t.name else ""
    return f"'{t.units}'{label}" if t.units else "an unlabelled tensor"


def _no_unit_product(op: str, args: Any) -> None:
    """Two dimensioned operands in a product: the unit of the result is not something this class can compute."""
    with_units = [t for t in _labelled(args) if t.unit is not None]
    if len(with_units) > 1:
        first, second = with_units[0], with_units[1]
        raise UnitError(
            f"cannot apply '{op.rstrip('_')}' to {_describe(first)} and {_describe(second)}: the unit of a product "
            f"is not defined here",
            fix="multiply the raw values (.without_labels()) and label the result with pp.tensor(..., units=...)")


def _relabel_like(out: Any, args: Any) -> Any:
    """Give the result the labels of the first labelled operand; recurse into tuples and lists."""
    source = next((t for t in _labelled(args) if t.labelled()), None)
    if source is None:
        return out
    if isinstance(out, torch.Tensor):
        if not isinstance(out, PhysicsTensor):
            out = out.as_subclass(PhysicsTensor)
        out._pp_name, out._pp_units, out._pp_unit = source.name, source.units, source.unit
        out._pp_domain, out._pp_coords = source.domain, source.coords
        out._pp_components, out._pp_time = source.components, source.time
        out._pp_meta = dict(getattr(source, "_pp_meta", {}))
        return out
    if isinstance(out, tuple):
        return tuple(_relabel_like(o, args) for o in out)
    if isinstance(out, list):
        return [_relabel_like(o, args) for o in out]
    return out


def tensor(data: Any, *, name: str = "", units: Any = None, domain: Any = None,
           coords: Optional[Sequence[str]] = None, components: Optional[Sequence[str]] = None,
           time: Any = None, meta: Optional[Mapping[str, Any]] = None,
           requires_grad: Optional[bool] = None) -> PhysicsTensor:
    """``pp.tensor(data, name=..., units=..., domain=...)``: a tensor that remembers what it is (#186).

    ``units`` is anything :func:`pinneapple_data.physical_units.parse_unit` accepts (``"m/s"``, ``"Pa"``,
    ``"W/(m·K)"``) or an already parsed ``Unit``; ``domain`` is whatever describes where the values live (a
    ``Domain``, a mesh, a mapping of bounds -- it is carried, not interpreted here); ``coords`` and ``components``
    name the axes. ``requires_grad=True`` makes it a leaf so autograd can fill ``.grad``.
    """
    return PhysicsTensor(data, name=name, units=units, domain=domain, coords=coords, components=components,
                         time=time, meta=meta, requires_grad=requires_grad)
