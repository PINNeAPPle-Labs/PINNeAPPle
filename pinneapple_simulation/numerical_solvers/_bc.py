"""Read Dirichlet conditions of a ProblemSpec for grid / node-based solvers.

``DirichletBC`` is a factory returning a ``ConditionSpec`` (kind="dirichlet"), so solvers must
check ``cond.kind`` (an ``isinstance`` check against the factory raises TypeError). The builder's
contract is ``selector(X, ctx) -> bool mask`` and ``value_fn(X, ctx) -> (N, n_fields)`` on NumPy
arrays; older call sites pass one-argument callables on tensors. Both are accepted here.
"""
from __future__ import annotations

import inspect

import numpy as np
import torch


def dirichlet_conditions(spec) -> list:
    return [c for c in getattr(spec, "conditions", ()) if getattr(c, "kind", None) == "dirichlet"]


def _accepts_two_positional(fn) -> bool:
    """Whether ``fn`` can be called as fn(X, ctx) -- checked by binding the SIGNATURE, not by
    calling fn and catching TypeError: catching would also swallow a real TypeError raised from
    inside a well-formed 2-argument fn (e.g. a bug in the caller's own selector/value_fn) and
    misreport it as "fn takes 1 argument", masking the actual failure (observed while writing
    tests/test_solver_exact_solutions_batch4.py: a bug elsewhere raised TypeError deep inside a
    2-arg lambda, and the old try/except here reported a misleading "missing argument 'ctx'")."""
    try:
        inspect.signature(fn).bind(None, None)
        return True
    except TypeError:
        return False
    except ValueError:
        return False  # signature not introspectable (e.g. some builtins): assume legacy 1-arg


def _call(fn, pts: torch.Tensor):
    """Try the builder contract fn(X_numpy, ctx) first, then the legacy fn(X_tensor)."""
    if _accepts_two_positional(fn):
        return fn(pts.detach().cpu().numpy(), {})
    return fn(pts)


def select(cond, pts: torch.Tensor, edges: dict[str, torch.Tensor] | None = None,
           boundary: torch.Tensor | None = None) -> torch.Tensor:
    """Indices of ``pts`` the condition applies to.

    ``edges`` maps tag names ("left", "right", ...) to indices; ``boundary`` is the index set used
    for selector None / "all" / {"tag": "boundary"}.
    """
    sel = cond.selector
    tag = sel.get("tag") if isinstance(sel, dict) else sel
    if tag is None or tag in ("all", "boundary"):
        if boundary is None:
            return torch.arange(pts.shape[0])
        return boundary
    if isinstance(tag, str):
        if edges is None or tag not in edges:
            raise KeyError(f"unknown boundary tag {tag!r}; known: {sorted(edges or {})}")
        return edges[tag]
    mask = torch.as_tensor(np.asarray(_call(sel, pts)), dtype=torch.bool).reshape(-1)
    return torch.where(mask)[0]


def values(cond, pts: torch.Tensor, field: int = 0) -> torch.Tensor:
    """Dirichlet values at ``pts`` for one field (column ``field`` of a multi-field value_fn)."""
    fn = cond.value_fn
    if not callable(fn):
        return torch.full((pts.shape[0],), float(fn), dtype=pts.dtype, device=pts.device)
    r = _call(fn, pts)
    v = (r if torch.is_tensor(r) else torch.as_tensor(np.asarray(r, dtype=np.float64))).to(pts.dtype).to(pts.device)
    if v.ndim == 2:
        v = v[:, field]
    return v.reshape(-1).expand(pts.shape[0]) if v.numel() == 1 else v.reshape(-1)


def boundary_rect(nodes: torch.Tensor, edges: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.unique(torch.cat(list(edges.values())))


__all__: tuple[str, ...] = ("dirichlet_conditions", "select", "values", "boundary_rect")
