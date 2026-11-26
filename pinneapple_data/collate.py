"""Collate functions for PINN batches and UPD supervision batches."""
from __future__ import annotations

from typing import Any

import torch

Tensor = torch.Tensor


def _stack_or_cat(xs: list[Tensor], dim: int = 0) -> Tensor:
    """Stack tensors if shapes match; otherwise concatenate along dim."""
    if not xs:
        raise ValueError("Empty tensor list")
    shape0 = tuple(xs[0].shape)
    if all(tuple(x.shape) == shape0 for x in xs):
        return torch.stack(xs, dim=dim)
    return torch.cat(xs, dim=dim)


def _collate_tuple_of_tensors(items: list[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
    """Collate [(t,x), (t,x), ...] -> (T, X) by concatenating each position along dim=0."""
    if not items:
        raise ValueError("No items to collate")
    k = len(items[0])
    if any(len(it) != k for it in items):
        raise ValueError("Inconsistent tuple lengths in collate")
    return tuple(torch.cat([it[j] for it in items], dim=0) for j in range(k))


def collate_pinn_batches(batches: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate a list of PINN batch dicts into a single batch dict.

    Handles keys: ``collocation``, ``conditions``, ``data``, ``meta``.
    """
    out: dict[str, Any] = {}
    if not batches:
        return out

    colls = [b.get("collocation") for b in batches if b.get("collocation") is not None]
    if colls:
        out["collocation"] = _collate_tuple_of_tensors(colls)

    conds = [b.get("conditions") for b in batches if b.get("conditions") is not None]
    if conds:
        n_cond = len(conds[0])
        if any(len(c) != n_cond for c in conds):
            raise ValueError("Inconsistent number of conditions across batches")
        out["conditions"] = [
            _collate_tuple_of_tensors([c[i] for c in conds]) for i in range(n_cond)
        ]

    datas = [b.get("data") for b in batches if b.get("data") is not None]
    if datas:
        out["data"] = (
            _collate_tuple_of_tensors([d[0] for d in datas]),
            torch.cat([d[1] for d in datas], dim=0),
        )

    metas = [b.get("meta") for b in batches if b.get("meta") is not None]
    if metas:
        out["meta"] = metas

    return out


def move_batch_to_device(batch: dict[str, Any], device: str | torch.device) -> dict[str, Any]:
    """Move all tensors in a collated PINN batch dict to *device*. Leaves ``meta`` unchanged."""
    dev = torch.device(device)

    def _t(x: Any) -> Any:
        return x.to(dev) if isinstance(x, torch.Tensor) else x

    out: dict[str, Any] = dict(batch)

    if out.get("collocation") is not None:
        out["collocation"] = tuple(_t(t) for t in out["collocation"])

    if out.get("conditions") is not None:
        out["conditions"] = [tuple(_t(t) for t in cond) for cond in out["conditions"]]

    if out.get("data") is not None:
        x, y = out["data"]
        out["data"] = (tuple(_t(t) for t in x), _t(y))

    return out


def collate_upd_supervised(samples: list[Any]) -> dict[str, Any]:
    """Collate a list of PhysicalSample objects for supervised training.

    Each ``sample.state`` dict must contain ``"x"`` (Tensor) and optionally ``"y"`` (Tensor).
    Returns ``{"x": Tensor, "y": Tensor, "meta": list}``.
    """
    if not samples:
        return {}

    xs: list[Tensor] = []
    ys: list[Tensor] = []
    metas: list[dict[str, Any]] = []

    for s in samples:
        st = getattr(s, "state", None)
        if st is None or not isinstance(st, dict):
            raise TypeError(
                "collate_upd_supervised expects PhysicalSample.state to be a dict."
            )
        if "x" not in st:
            raise KeyError("PhysicalSample.state missing key 'x'")

        x = st["x"]
        if not isinstance(x, torch.Tensor):
            raise TypeError("PhysicalSample.state['x'] must be a torch.Tensor")
        xs.append(x)

        if "y" in st and st["y"] is not None:
            y = st["y"]
            if not isinstance(y, torch.Tensor):
                raise TypeError("PhysicalSample.state['y'] must be a torch.Tensor")
            ys.append(y)

        metas.append({
            "provenance": getattr(s, "provenance", {}) or {},
            "domain": getattr(s, "domain", {}) or {},
            "schema": getattr(s, "schema", {}) or {},
        })

    out: dict[str, Any] = {"x": _stack_or_cat(xs, dim=0), "meta": metas}
    if ys:
        out["y"] = _stack_or_cat(ys, dim=0)
    return out
