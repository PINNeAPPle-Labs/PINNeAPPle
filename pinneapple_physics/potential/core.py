"""A potential and its derivative tree.

A potential is one scalar function of its natural variables, ``Phi(x_1, ..., x_n)``. Its derivatives, organised by
order, are the tree: depth 1 gives the "forces" (entropy and pressure from a Helmholtz energy, stress from a strain
energy, momenta from a Lagrangian), depth 2 the response functions (heat capacity, compressibility, elastic tangent,
mass matrix), and so on. Every quantity of the tree comes from the same function, so the mixed partials agree and
the quantities are consistent with one another by construction.

    import torch
    from pinneapple_physics.potential import Potential, tree

    A = Potential(lambda T, V: -8.314 * T * torch.log(V) - 1.5 * 8.314 * T * torch.log(T),
                  {"T": "K", "V": "m3/mol"}, name="A", unit="J/mol",
                  names={("T",): ("S", -1, "J/(mol K)"), ("V",): ("p", -1, "Pa")})
    t = tree(A, depth=2, at={"T": 300.0, "V": 0.025})
    t["p"], t["S"], t[("T", "T")]                  # 8.314 * 300 / 0.025, ..., -1.5 * 8.314 / 300

Derivatives are taken by reverse-mode automatic differentiation in float64 (``torch.autograd``), batched over the
evaluation points. A potential written as a SymPy expression (:meth:`Potential.from_sympy`) also carries its
symbolic derivatives; :meth:`DerivativeTree.symbolic_agreement` compares the two.

Variables may be tensors (a deformation gradient ``F`` of shape ``(3, 3)``): the node ``("F",)`` then has the shape
of ``F`` and ``("F", "F")`` the shape ``(3, 3, 3, 3)``, per point.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

__all__ = ["Var", "Name", "Response", "Potential", "Node", "DerivativeTree", "tree"]

DTYPE = torch.float64


@dataclass(frozen=True)
class Var:
    """A natural variable: its unit (free text, e.g. ``"K"``) and its shape per point (``()`` for a scalar)."""

    unit: str = ""
    shape: tuple[int, ...] = ()

    @property
    def size(self) -> int:
        return int(math.prod(self.shape))


@dataclass(frozen=True)
class Name:
    """The physical name of a raw derivative: ``quantity = sign * d^k Phi / dx...``, with its unit."""

    symbol: str
    sign: float = 1.0
    unit: str | None = None
    description: str = ""


@dataclass(frozen=True)
class Response:
    """A quantity built from several raw derivatives (``c_p``, a speed of sound, a mass matrix).

    ``fn(d, x)`` receives ``d``, a mapping from derivative keys (tuples of variable names in variable order, ``()``
    for the potential itself) to tensors, and ``x``, the variables as tensors; it returns a tensor. ``depth`` is the
    highest derivative order it reads."""

    fn: Callable[[Mapping[tuple, torch.Tensor], Mapping[str, torch.Tensor]], torch.Tensor]
    depth: int
    unit: str = ""
    description: str = ""


def _as_var(v) -> Var:
    if isinstance(v, Var):
        return v
    if isinstance(v, str):
        return Var(unit=v)
    if isinstance(v, tuple) and len(v) == 2 and isinstance(v[0], str):
        return Var(unit=v[0], shape=tuple(v[1]))
    raise TypeError(f"variable spec must be a unit string, (unit, shape) or Var, got {v!r}")


def _as_name(v) -> Name:
    if isinstance(v, Name):
        return v
    if isinstance(v, str):
        return Name(v)
    return Name(*v)


class Potential:
    """A scalar potential ``Phi(x)`` of named natural variables.

    Parameters
    ----------
    fn:
        ``fn(**x) -> Phi``: takes one float64 tensor per variable (shape ``batch + var.shape``) and returns a tensor
        of shape ``batch``. Write it with ``torch`` operations so it can be differentiated.
    variables:
        names in order, or ``{name: unit}``, ``{name: (unit, shape)}`` or ``{name: Var}``.
    name, unit:
        symbol and unit of the potential.
    names:
        ``{key: (symbol, sign, unit)}`` naming raw derivatives, e.g. ``{("V",): ("p", -1, "Pa")}`` for
        ``p = -dA/dV``. Keys are tuples of variable names; their order does not matter.
    responses:
        ``{symbol: Response}`` for quantities that combine several derivatives.
    relation_names:
        ``{frozenset({x_i, x_j}): text}``: engineering names of the mixed-partial identities (see
        :func:`~pinneapple_physics.potential.relations.maxwell_relations`).
    expr, symbols:
        a SymPy expression of the same potential and its symbols (one per variable), for symbolic derivatives.
    """

    def __init__(self, fn: Callable[..., torch.Tensor], variables: Sequence[str] | Mapping[str, Any], *,
                 name: str = "Phi", unit: str = "", names: Mapping | None = None,
                 responses: Mapping[str, Response] | None = None, relation_names: Mapping | None = None,
                 expr: Any = None, symbols: Sequence[Any] | None = None, description: str = ""):
        if isinstance(variables, Mapping):
            self.variables = {k: _as_var(v) for k, v in variables.items()}
        else:
            self.variables = {k: Var() for k in variables}
        if not self.variables:
            raise ValueError("a potential needs at least one variable")
        self.fn = fn
        self.name, self.unit, self.description = name, unit, description
        self.names = {self.key(k): _as_name(v) for k, v in (names or {}).items()}
        self.responses = dict(responses or {})
        self.relation_names = {frozenset(k): v for k, v in (relation_names or {}).items()}
        self.expr, self.symbols = expr, (tuple(symbols) if symbols is not None else None)
        offsets, o = {}, 0
        for k, v in self.variables.items():
            offsets[k] = o
            o += v.size
        self._offsets, self.n_flat = offsets, o

    # ------------------------------------------------------------------ construction helpers
    @classmethod
    def from_sympy(cls, expr, variables: Sequence[Any] | Mapping[Any, str], *, params: Mapping | None = None,
                   **kwargs) -> Potential:
        """A potential from a SymPy expression. ``variables`` are SymPy symbols (in order) or ``{symbol: unit}``;
        ``params`` substitutes numeric values for the remaining symbols. The expression is turned into a torch
        function with :func:`sympy.lambdify` and kept for symbolic derivatives."""
        import sympy as sp

        if isinstance(variables, Mapping):
            syms, units = list(variables.keys()), list(variables.values())
        else:
            syms, units = list(variables), [""] * len(variables)
        e = sp.sympify(expr)
        if params:
            e = e.subs({(sp.Symbol(k) if isinstance(k, str) else k): v for k, v in params.items()})
        free = e.free_symbols - set(syms)
        if free:
            raise ValueError(f"symbols without a value: {sorted(map(str, free))}; pass them in params")
        f = sp.lambdify(syms, e, modules="torch")
        names = [str(s) for s in syms]

        def fn(**x):
            ref = x[names[0]]
            out = f(*[x[n] for n in names])
            return out if torch.is_tensor(out) else torch.full_like(ref, float(out))

        return cls(fn, {n: u for n, u in zip(names, units)}, expr=e, symbols=syms, **kwargs)

    def key(self, k) -> tuple[str, ...]:
        """A derivative key in canonical (variable) order."""
        if isinstance(k, str):
            k = (k,)
        order = list(self.variables)
        for v in k:
            if v not in self.variables:
                raise KeyError(f"{v!r} is not a variable of {self.name} ({order})")
        return tuple(sorted(k, key=order.index))

    def symbol_of(self, key: tuple[str, ...]) -> str:
        key = self.key(key)
        if key in self.names:
            return self.names[key].symbol
        if not key:
            return self.name
        den = "".join(f"d{v}" for v in key)
        return f"d{len(key) if len(key) > 1 else ''}{self.name}/{den}"

    def unit_of(self, key: tuple[str, ...]) -> str:
        key = self.key(key)
        if key in self.names and self.names[key].unit is not None:
            return self.names[key].unit
        if not key:
            return self.unit
        den = " ".join(self.variables[v].unit or "1" for v in key)
        return f"{self.unit or '1'}/({den})"

    # ------------------------------------------------------------------ evaluation
    def _prepare(self, point: Mapping[str, Any]) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Flatten the variables of ``point`` into ``(N, n_flat)`` float64 and return the batch shape."""
        missing = [k for k in self.variables if k not in point]
        if missing:
            raise KeyError(f"missing variables {missing} for {self.name}")
        arrays, batch = {}, []
        for k, v in self.variables.items():
            a = torch.as_tensor(np.asarray(point[k].detach().cpu() if torch.is_tensor(point[k]) else point[k],
                                           dtype=float), dtype=DTYPE)
            if tuple(a.shape[a.ndim - len(v.shape):]) != v.shape and v.shape:
                raise ValueError(f"{k} must end with shape {v.shape}, got {tuple(a.shape)}")
            arrays[k] = a
            batch.append(tuple(a.shape[:a.ndim - len(v.shape)]))
        bshape = tuple(np.broadcast_shapes(*batch))
        cols = []
        for k, v in self.variables.items():
            a = arrays[k].broadcast_to(bshape + v.shape)
            cols.append(a.reshape(-1, v.size) if bshape else a.reshape(1, v.size))
        return torch.cat(cols, dim=1), bshape

    def unflatten(self, z: torch.Tensor) -> dict[str, torch.Tensor]:
        """Columns of a ``(N, n_flat)`` tensor back to one tensor per variable."""
        return {k: z[:, self._offsets[k]:self._offsets[k] + v.size].reshape((z.shape[0],) + v.shape)
                for k, v in self.variables.items()}

    def call_flat(self, z: torch.Tensor) -> torch.Tensor:
        out = self.fn(**self.unflatten(z))
        return out.reshape(z.shape[0])

    def __call__(self, **point) -> np.ndarray:
        z, bshape = self._prepare(point)
        with torch.no_grad():
            out = self.call_flat(z).detach()
        return out.reshape(bshape).numpy() if bshape else out.reshape(()).numpy()

    def flat_derivatives(self, z: torch.Tensor, depth: int, create_graph: bool = False) -> dict[tuple, torch.Tensor]:
        """All partial derivatives up to ``depth`` with respect to the flattened components of ``z``.

        Returns ``{(i_1 <= ... <= i_k): tensor (N,)}``; ``()`` holds the potential. Symmetry of mixed partials is
        used (only sorted multi-indices are computed). ``create_graph=True`` keeps the graph for training."""
        if not z.requires_grad:
            z = z.detach().requires_grad_(True)
        phi = self.call_flat(z)
        out = {(): phi}
        frontier = {(): phi}
        for d in range(1, depth + 1):
            keep = create_graph or d < depth
            new = {}
            for idx, val in frontier.items():
                if not val.requires_grad:
                    g = torch.zeros_like(z)
                else:
                    (g,) = torch.autograd.grad(val.sum(), z, create_graph=keep, allow_unused=True)
                    if g is None:
                        g = torch.zeros_like(z)
                start = idx[-1] if idx else 0
                for j in range(start, self.n_flat):
                    new[idx + (j,)] = g[:, j]
            out.update(new)
            frontier = new
        return out

    def keys(self, depth: int) -> list[tuple[str, ...]]:
        """Derivative keys (variable-level, canonical order) up to ``depth``, starting with ``()``."""
        names = list(self.variables)
        out = [()]
        for d in range(1, depth + 1):
            out += [tuple(c) for c in itertools.combinations_with_replacement(names, d)]
        return out

    def assemble(self, flat: Mapping[tuple, torch.Tensor], key: tuple[str, ...], n: int) -> torch.Tensor:
        """The derivative block of variable-level ``key`` from flat derivatives: shape ``(N,) + shapes``."""
        key = self.key(key)
        if all(self.variables[v].size == 1 for v in key):
            idx = tuple(sorted(self._offsets[v] for v in key))
            return flat[idx]
        ranges = [range(self.variables[v].size) for v in key]
        vals = []
        for comb in itertools.product(*ranges):
            idx = tuple(sorted(self._offsets[v] + c for v, c in zip(key, comb)))
            vals.append(flat[idx])
        shape = sum((self.variables[v].shape for v in key), ())
        return torch.stack(vals, dim=1).reshape((n,) + shape)

    def derivatives(self, z: torch.Tensor, depth: int, create_graph: bool = False
                    ) -> tuple[dict[tuple, torch.Tensor], dict[str, torch.Tensor]]:
        """Variable-level derivative blocks up to ``depth`` and the responses whose depth fits, as tensors.

        ``d[key]`` for raw derivatives; the second mapping holds named quantities (with their sign applied) and
        responses."""
        flat = self.flat_derivatives(z, depth, create_graph=create_graph)
        n = z.shape[0]
        d = {k: self.assemble(flat, k, n) for k in self.keys(depth)}
        x = self.unflatten(z)
        named = {}
        for k, nm in self.names.items():
            if len(k) <= depth:
                named[nm.symbol] = nm.sign * d[k]
        for sym, r in self.responses.items():
            if r.depth <= depth:
                named[sym] = r.fn(d, x)
        return d, named


@dataclass
class Node:
    """One quantity of a derivative tree."""

    key: tuple[str, ...]
    symbol: str
    unit: str
    depth: int
    value: np.ndarray | None = None
    sign: float = 1.0
    expr: Any = None
    symbolic_value: np.ndarray | None = None
    description: str = ""


@dataclass
class DerivativeTree:
    """The derivative tree of a potential evaluated at a set of points.

    ``tree[key]`` with a key tuple returns the raw derivative; ``tree[symbol]`` a named quantity or response (with
    its sign applied). ``nodes`` holds every raw derivative, ``responses`` the derived quantities."""

    potential: Potential
    depth: int
    point: dict[str, np.ndarray]
    nodes: dict[tuple, Node] = field(default_factory=dict)
    quantities: dict[str, np.ndarray] = field(default_factory=dict)

    def __getitem__(self, k):
        if isinstance(k, str) and k in self.quantities:
            return self.quantities[k]
        if isinstance(k, str) and k not in self.potential.variables:
            raise KeyError(f"{k!r}: not a named quantity of {self.potential.name}; known: {sorted(self.quantities)}")
        return self.nodes[self.potential.key(k)].value

    def __contains__(self, k) -> bool:
        try:
            self[k]
            return True
        except KeyError:
            return False

    def level(self, depth: int) -> list[Node]:
        """The nodes at one depth (1: forces, 2: response functions, ...)."""
        return [n for n in self.nodes.values() if n.depth == depth]

    def symbolic_agreement(self) -> dict[str, float]:
        """Largest relative difference between autodiff and symbolic values, per node (needs a SymPy potential)."""
        out = {}
        for n in self.nodes.values():
            if n.symbolic_value is None:
                continue
            a, s = np.asarray(n.value, float), np.asarray(n.symbolic_value, float)
            scale = max(np.abs(s).max(), np.abs(a).max(), 1e-300)
            out[n.symbol] = float(np.abs(a - s).max() / scale)
        return out

    def table(self) -> list[dict[str, Any]]:
        """One row per node and response: depth, symbol, unit, value (first point when several)."""
        rows = []
        for n in self.nodes.values():
            v = np.asarray(n.value)
            rows.append({"depth": n.depth, "key": n.key, "symbol": n.symbol, "unit": n.unit,
                         "value": v.reshape(-1)[0].item() if v.size else None, "description": n.description})
        for sym, v in self.quantities.items():
            if any(n.symbol == sym for n in self.nodes.values()):
                continue
            r = self.potential.responses.get(sym)
            rows.append({"depth": r.depth if r else None, "key": None, "symbol": sym, "unit": r.unit if r else "",
                         "value": np.asarray(v).reshape(-1)[0].item(), "description": r.description if r else ""})
        return rows

    def __str__(self) -> str:
        lines = [f"derivative tree of {self.potential.name}({', '.join(self.potential.variables)}), depth "
                 f"{self.depth}"]
        for r in self.table():
            d = "R" if r["key"] is None else r["depth"]
            lines.append(f"  [{d}] {r['symbol']:<16s} = {r['value']: .6g} {r['unit']}")
        return "\n".join(lines)


def _symbolic_nodes(potential: Potential, depth: int, point: Mapping[str, np.ndarray], bshape) -> dict:
    import sympy as sp

    syms = potential.symbols
    vals = [np.broadcast_to(np.asarray(point[k], float), bshape) for k in potential.variables]
    out = {}
    for key in potential.keys(depth):
        e = potential.expr
        for v in key:
            e = sp.diff(e, syms[list(potential.variables).index(v)])
        f = sp.lambdify(syms, e, modules="numpy")
        out[key] = (e, np.broadcast_to(np.asarray(f(*vals), float), bshape).copy())
    return out


def tree(potential: Potential, depth: int = 2, at: Mapping[str, Any] | None = None,
         symbolic: bool | None = None) -> DerivativeTree:
    """The derivative tree of ``potential`` up to ``depth`` at the points ``at`` (one value or array per variable,
    broadcast together).

    ``symbolic`` (default: when the potential has a SymPy expression and only scalar variables) adds each node's
    symbolic derivative and its value, for :meth:`DerivativeTree.symbolic_agreement`."""
    if depth < 0:
        raise ValueError("depth must be >= 0")
    if at is None:
        raise ValueError("give the evaluation points: tree(potential, depth, at={var: value, ...})")
    z, bshape = potential._prepare(at)
    d, named = potential.derivatives(z, depth)

    def np_(t):
        a = t.detach().numpy()
        return a.reshape(bshape + a.shape[1:]) if bshape else a.reshape(a.shape[1:])

    point = {k: np.asarray(at[k], float) for k in potential.variables}
    use_sym = symbolic if symbolic is not None else (
        potential.expr is not None and all(v.size == 1 and not v.shape for v in potential.variables.values()))
    sym = _symbolic_nodes(potential, depth, point, bshape or (1,)) if use_sym else {}
    t = DerivativeTree(potential, depth, point)
    for key in potential.keys(depth):
        nm = potential.names.get(key)
        node = Node(key, potential.symbol_of(key), potential.unit_of(key), len(key), np_(d[key]),
                    sign=nm.sign if nm else 1.0, description=nm.description if nm else "")
        if key in sym:
            e, v = sym[key]
            node.expr, node.symbolic_value = e, (v if bshape else v.reshape(()))
        t.nodes[key] = node
    t.quantities = {k: np_(v) for k, v in named.items()}
    return t
