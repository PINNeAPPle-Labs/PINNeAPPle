"""Model rows of a field-surrogate benchmark, built through one interface so that one trainer trains them all.

Every entry is a builder of an ``nn.Module`` whose forward is ``(x, *, part, part_id, thickness, time, mask) ->
(B, N, C)`` for a steady field or ``(B, T, N, C)`` for a transient one — the interface the trainer and the metrics
expect. The adapters below bring the library's own operators (``Transolver``, ``TransolverLite``) and the point
models into it without renaming anything.

Two things this module deliberately does not do: it never labels a stand-in with the name of a published model it is
not (an approximation carries its own name, and its note says what it approximates), and it never silently drops a
row — a model that cannot be built is reported as unavailable, with the reason.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from pinneapple_neural.architectures.crash_solver import CrashSolverModel, FourierTime
from pinneapple_neural.architectures.neural_operators.transolver import Transolver, TransolverLite
from pinneapple_neural.architectures.point_surrogates import (
    EdgeConvSurrogate,
    PartPointNetSurrogate,
    PointNetSurrogate,
)

__all__ = ["OperatorSetAdapter", "build_model_specs", "available_models"]


class OperatorSetAdapter(nn.Module):
    """Wrap a point operator (``(B, N, in_dim) -> (B, N, out_dim)``) into the set-to-set interface.

    The operator sees the coordinates plus any extra per-node channels, and — when the benchmark is transient — a
    Fourier encoding of the requested time, frame by frame. This is the honest way to put a steady operator on a
    transient benchmark: same weights for every frame, time as an input, no invented temporal head."""

    def __init__(self, operator: nn.Module, out_dim: int = 1, in_dim: int | None = None, time_dim: int = 8,
                 t_max: float = 1.0):
        super().__init__()
        self.op = operator
        self.out_dim = int(out_dim)
        self.time_enc = FourierTime(time_dim, t_max=t_max) if time_dim else None
        self.in_dim = in_dim                                 # channels the operator expects (coords + extras + tau)

    def forward(self, x: torch.Tensor, *, part=None, part_id=None, thickness=None, extra=None, time=None,
                mask=None, **_) -> torch.Tensor:
        if thickness is None:                                  # a benchmark without a per-node scalar still feeds one
            thickness = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
        feats = [x]
        if extra is not None:
            feats.append(extra)
        feats.append(thickness.unsqueeze(-1))
        base = torch.cat(feats, dim=-1)
        if time is None:
            out = self.op(base)
            return getattr(out, "y", out)
        frames = []
        t_feat = self.time_enc(time.to(x.dtype)) if self.time_enc is not None else None
        for ti in range(time.shape[1]):
            inp = base if t_feat is None else torch.cat(
                [base, t_feat[:, ti].unsqueeze(1).expand(-1, base.shape[1], -1)], dim=-1)
            out = self.op(inp)
            frames.append(getattr(out, "y", out))
        return torch.stack(frames, dim=1)


def build_model_specs(*, in_dim: int = 3, out_dim: int = 1, transient: bool = False, n_frames: int = 8,
                      n_parts: int = 8, scale: str = "small", extra_dim: int = 0) -> list[dict]:
    """The model rows of a benchmark, one spec per architecture (``{"name", "family", "build", "notes"}``).

    ``scale="small"`` keeps every model trainable on a laptop-sized corpus; ``scale="paper"`` uses the widths the
    vehicle-aerodynamics benchmark reports (a TransolverLarge of ~7.6 M parameters, a wide residual point model), which
    needs a GPU to be worth the wall time. ``transient`` rows add a temporal head (the hierarchical model) or the time
    as an input channel (the operators and the point models)."""
    if scale not in ("small", "paper"):
        raise ValueError("scale must be 'small' or 'paper'")
    big = scale == "paper"
    dim = 384 if big else 128
    specs: list[dict] = []

    def add(name, family, build, notes):
        specs.append({"name": name, "family": family, "build": build, "notes": notes,
                      "uses_parts": family in ("hierarchical", "point+part")})

    time_dim = 8                                                               # Fourier features of the time channel
    # coordinates + extra channels + tau, plus the time encoding when the field is transient
    op_in = in_dim + 1 + extra_dim + (time_dim if transient else 0)
    add("transolver", "operator",
        lambda: OperatorSetAdapter(Transolver(in_dim=op_in, out_dim=out_dim, dim=dim, depth=5, heads=8, slices=32),
                                   out_dim=out_dim, in_dim=op_in, time_dim=time_dim),
        "physics-attention slices over the whole point set (Wu et al., ICML 2024), time as an input channel")
    add("transolver_lite", "operator",
        lambda: OperatorSetAdapter(TransolverLite(in_dim=op_in, out_dim=out_dim, d_model=64 if not big else 128,
                                                  n_slices=8), out_dim=out_dim, in_dim=op_in),
        "the simplified slice-attention variant kept in the library for reproduction studies")
    if big:
        add("transolver_large", "operator",
            lambda: OperatorSetAdapter(Transolver(in_dim=op_in, out_dim=out_dim, dim=384, depth=7, heads=12,
                                                  slices=48), out_dim=out_dim, in_dim=op_in, time_dim=time_dim),
            "the wider Transolver of the vehicle-aerodynamics benchmark (48 slices, 384 latent)")
    add("pointnet", "point",
        lambda: _PointRow(PointNetSurrogate(in_dim=op_in, out_dim=out_dim,
                                            dim=1024 if big else 128, residual=False), time_dim=time_dim),
        "shared point MLP + global max pool (Qi et al., CVPR 2017)")
    add("pointnet_wide", "point",
        lambda: _PointRow(PointNetSurrogate(in_dim=op_in, out_dim=out_dim, dim=1024, residual=True), time_dim=time_dim),
        "wide residual point model: our stand-in for the PointNetLarge row of the vehicle benchmark, not its weights")
    add("edgeconv", "point",
        lambda: _PointRow(EdgeConvSurrogate(in_dim=op_in, out_dim=out_dim, dim=256 if not big else 512, k=16),
                               time_dim=time_dim),
        "k-NN EdgeConv graph rebuilt every layer (Wang et al., TOG 2019)")
    add("part_pointnet", "point+part",
        lambda: _PointRow(PartPointNetSurrogate(in_dim=op_in, out_dim=out_dim, dim=256 if not big else 1024,
                                               n_parts=max(n_parts, 1)), time_dim=time_dim),
        "PointNet with a learned embedding per semantic part")
    add("crash_solver", "hierarchical",
        lambda: CrashSolverModel(in_dim=in_dim - 3 + extra_dim, out_dim=out_dim, hidden=96 if not big else 128,
                                 local_slices=32, global_layers=2 if not big else 3, n_frames=n_frames if transient else 1,
                                 n_classes=max(n_parts, 2)),
        "per-part physics attention -> part-token transformer -> interface message passing -> temporal decoder "
        "(the hierarchical design of the crash benchmark)")
    return specs


class _PointRow(nn.Module):
    """Give a point model the optional extra channels the benchmark passes (thickness) and the time encoding."""

    def __init__(self, model: nn.Module, time_dim: int = 8):
        super().__init__()
        self.model = model
        self.time_enc = FourierTime(time_dim)

    def forward(self, x, *, part=None, part_id=None, thickness=None, extra=None, time=None, mask=None, **_):
        if thickness is None:
            thickness = torch.ones(x.shape[:2], dtype=x.dtype, device=x.device)
        feats = [x]
        if extra is not None:
            feats.append(extra)
        feats.append(thickness.unsqueeze(-1))
        base = torch.cat(feats, dim=-1)
        if time is None:
            return self.model(base, part=part, mask=mask)
        out = []
        t_feat = self.time_enc(time.to(x.dtype))
        for ti in range(time.shape[1]):
            out.append(self.model(torch.cat([base, t_feat[:, ti].unsqueeze(1).expand(-1, base.shape[1], -1)], -1),
                                  part=part, mask=mask))
        return torch.stack(out, dim=1)


def available_models(**kwargs) -> list[str]:
    """Names the specs report as buildable on this machine (a constructor is attempted with tiny sizes)."""
    ok = []
    for spec in build_model_specs(**kwargs):
        try:
            spec["build"]()
            ok.append(spec["name"])
        except Exception as exc:                                  # pragma: no cover - reported, not raised
            ok.append(f"{spec['name']} (unavailable: {type(exc).__name__})")
    return ok
