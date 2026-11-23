"""Chebyshev-basis neural operators and the Hadamard branch/trunk variant.

Three operators, from two published designs, reimplemented from the papers:

* :class:`ChebyshevPolynomialNeuralOperator` - Chen, B., Wang, J., Xie, H., Wang, Q.,
  Zhang, S., Xia, Y. & Zhang, J. (2026), *Physics-Informed Chebyshev Polynomial Neural
  Operator for Parametric Partial Differential Equations*, arXiv:2602.01737. A
  mesh-free operator: it answers one point at a time from a Chebyshev expansion of the
  coordinates, with the scenario entering only through a FiLM-style modulation of each
  layer.
* :class:`ChebyshevBranchOperator` and :class:`HadamardDeepONet` - the branch/trunk
  decomposition used for multiphase flow by Santos, E. S., Barros, G. F., Oliveira,
  A. C. N., Silva, R. M., Freitas, R. S. M., Valiveti, D. M., Wu, X.-H., Rochinha, F. A.
  & Coutinho, A. L. G. A. (2025), *Hybrid DeepONet Surrogates for Multiphase Flow in
  Porous Media*, arXiv:2511.02962, combined by a Hadamard product rather than the inner
  product of the original DeepONet (Lu et al. 2021): the branch emits the whole field
  for every instant, the trunk only reweights it in time.

Relationship to the existing :class:`~.deeponet.DeepONet`
--------------------------------------------------------
The registry's ``deeponet`` is the classical operator: a branch vector contracted with a
trunk basis over ``modes`` components. ``HadamardDeepONet`` here is the spatially
resolved variant: the branch keeps a full field per output channel and per timestep, so
it does not need a reduced basis to represent a sharp front. Both are in the registry
under distinct keys; they are not interchangeable and the difference is the reason both
exist.

Contract
--------
``ChebyshevPolynomialNeuralOperator`` is a point operator: ``(B, N, d)`` coordinates in,
``(B, N, out)`` values out, the same shape convention as the other point operators in
this package.

The two branch/trunk operators are grid operators and emit the full field for every
timestep: ``(B, C, T, H, W)``, matching what the FNO family returns for a 3-D volume.
They need the grid size at construction time because the branch output is reshaped into
it.

Validation status: these are faithful reimplementations of published architectures.
They are not in ``pinneapple_catalog`` as validated methods, and nothing here should be
reported as validated because of these tests.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import NeuralOperatorBase, OperatorOutput


def chebyshev_basis(z: torch.Tensor, order: int) -> torch.Tensor:
    """Chebyshev polynomials of the first kind, ``T_0 .. T_order``, by recurrence.

    ``z`` must carry a trailing singleton axis (shape ``(..., 1)``), which the expansion
    replaces: a ``(n, 1)`` input returns ``(n, order + 1)``. The recurrence is
    ``T_{k+1} = 2 z T_k - T_{k-1}``, the same one the paper uses (Eq. 10); computing it
    instead of fitting it is why the basis needs no training.
    """
    if order < 0:
        raise ValueError("order must be non-negative")
    terms = [torch.ones_like(z)]
    if order >= 1:
        terms.append(z)
    for _ in range(2, order + 1):
        terms.append(2.0 * z * terms[-1] - terms[-2])
    return torch.cat(terms, dim=-1)


def chebyshev_features(coords: torch.Tensor, order: int) -> torch.Tensor:
    """Expand every coordinate column and concatenate: ``(B, N, d) -> (B, N, d*(order+1))``."""
    if coords.dim() < 2:
        raise ValueError("coords must have shape (..., coordinates)")
    return torch.cat(
        [chebyshev_basis(coords[..., d : d + 1], order) for d in range(coords.shape[-1])], dim=-1
    )


class ModulatedPolynomialLayer(nn.Module):
    """One CPNO layer: grow the polynomial degree, then modulate it by the parameters.

    ``h <- GELU(omega(theta) * (A phi * h) + beta(theta))``, with ``omega`` and ``beta``
    produced from the scenario parameters alone. The modulation head starts at zero, so
    the layer begins as a plain polynomial projection (``omega = 1``, ``beta = 0``) and
    the conditioning grows into it instead of scrambling an untrained basis.
    """

    def __init__(self, phi_dim: int, hidden_dim: int, modulation_dim: int = 32,
                 parameter_dim: int = 2):
        super().__init__()
        self.poly = nn.Linear(phi_dim, hidden_dim)
        self.modulation = nn.Sequential(
            nn.Linear(parameter_dim, modulation_dim),
            nn.GELU(),
            nn.Linear(modulation_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, phi: torch.Tensor, theta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        omega_raw, beta = self.modulation(theta).chunk(2, dim=-1)
        omega = 1.0 + omega_raw
        if omega.dim() < hidden.dim():
            # (batch, hidden) has to broadcast over the point axis of (batch, points, hidden).
            omega = omega.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return F.gelu(omega * (self.poly(phi) * hidden) + beta)


class ChebyshevPolynomialNeuralOperator(NeuralOperatorBase):
    """Mesh-free Chebyshev operator conditioned on scalar parameters (CPNO).

    Parameters
    ----------
    coordinate_dim:
        Number of coordinates per query point (2 for (x, y), 3 with time).
    parameter_dim:
        Number of scalars that identify a scenario (well rates, boundary values, ...).
    out_dim:
        Output channels per query point.
    order, depth, hidden_dim, modulation_dim:
        Polynomial degree, number of modulated layers, hidden width, width of the
        modulation network. ``order=6, depth=6`` matches the configuration reported in
        the reference study.

    Forward
    -------
    ``forward(coords, theta)`` with ``coords`` of shape ``(B, N, coordinate_dim)`` already
    mapped into ``[-1, 1]``, and ``theta`` of shape ``(B, parameter_dim)``. Returns
    ``(B, N, out_dim)``. Nothing bounds the output: the reference measurement is that
    some saturation values leave the physical range, and that behaviour is kept rather
    than hidden, so a caller must clamp when the physics requires it.
    """

    def __init__(self, coordinate_dim: int = 3, parameter_dim: int = 2, out_dim: int = 2,
                 order: int = 6, depth: int = 6, hidden_dim: int = 64, modulation_dim: int = 32):
        super().__init__()
        self.coordinate_dim = int(coordinate_dim)
        self.parameter_dim = int(parameter_dim)
        self.out_dim = int(out_dim)
        self.order = int(order)
        self.depth = int(depth)
        self.hidden_dim = int(hidden_dim)
        self.phi_dim = self.coordinate_dim * (order + 1)
        self.entry = nn.Linear(self.phi_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [ModulatedPolynomialLayer(self.phi_dim, hidden_dim, modulation_dim, self.parameter_dim)
             for _ in range(depth)]
        )
        self.projection = nn.Linear(hidden_dim, out_dim)

    def forward(
        self,
        coords: torch.Tensor,
        theta: torch.Tensor,
        *,
        y_true: torch.Tensor | None = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        if coords.dim() != 3:
            raise ValueError(f"coords must be (B, N, {self.coordinate_dim}), got {tuple(coords.shape)}")
        if coords.shape[-1] != self.coordinate_dim:
            raise ValueError(f"expected {self.coordinate_dim} coordinates, got {coords.shape[-1]}")
        batch = coords.shape[0]
        if theta.shape[0] not in (1, batch):
            raise ValueError(f"theta has {theta.shape[0]} scenarios for a batch of {batch}")
        phi = chebyshev_features(coords, self.order)
        hidden = self.entry(phi)
        conditioning = theta if theta.shape[0] == batch else theta.expand(batch, -1)
        for layer in self.layers:
            hidden = layer(phi, conditioning, hidden)
        y = self.projection(hidden)

        losses = {"total": torch.zeros((), device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]
        return OperatorOutput(y=y, losses=losses, extras={"phi_dim": self.phi_dim})


class _BroadcastModulatedLayer(nn.Module):
    """The CPNO recurrence for a *shared* point set, evaluated for a whole batch.

    When the query points do not depend on the scenario, the polynomial projection is
    computed once and broadcast, and only the modulation varies per scenario. That is
    the same arithmetic as :class:`ModulatedPolynomialLayer` with the batch axis moved
    outside the point axis, and it is what makes a 81 x 58 grid times a batch fit in
    memory.
    """

    def __init__(self, phi_dim: int, hidden_dim: int, modulation_dim: int = 32,
                 parameter_dim: int = 2):
        super().__init__()
        self.poly = nn.Linear(phi_dim, hidden_dim)
        self.modulation = nn.Sequential(
            nn.Linear(parameter_dim, modulation_dim),
            nn.GELU(),
            nn.Linear(modulation_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(self, phi: torch.Tensor, theta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        poly_phi = self.poly(phi)                                    # (points, hidden)
        omega_raw, beta = self.modulation(theta).chunk(2, dim=-1)     # (batch, hidden)
        omega = (1.0 + omega_raw).unsqueeze(1)
        beta = beta.unsqueeze(1)
        return F.gelu(omega * (poly_phi.unsqueeze(0) * hidden) + beta)


class _ChebyshevBranch(nn.Module):
    """Spatial branch: CPNO's mechanism, emitting the field for every instant at once.

    Only the geometric coordinates are expanded in the Chebyshev basis. A spatially
    varying field input (normalised log permeability) is concatenated as a raw column,
    which is what the branch/trunk design does: the rock is not a coordinate, and
    expanding it would impose a polynomial shape on a field that is not polynomial. The
    parameter count is sensitive to this choice and the published numbers assume it.
    """

    def __init__(self, basis_dims: int, extra_dims: int, out_channels: int, order: int,
                 depth: int, hidden_dim: int, modulation_dim: int, parameter_dim: int):
        super().__init__()
        self.basis_dims = int(basis_dims)
        self.extra_dims = int(extra_dims)
        self.order = int(order)
        self.phi_dim = self.basis_dims * (order + 1) + self.extra_dims
        self.entry = nn.Linear(self.phi_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [_BroadcastModulatedLayer(self.phi_dim, hidden_dim, modulation_dim, parameter_dim)
             for _ in range(depth)]
        )
        self.projection = nn.Linear(hidden_dim, out_channels)

    def features(self, field_coords: torch.Tensor) -> torch.Tensor:
        basis = chebyshev_features(field_coords[:, : self.basis_dims], self.order)
        if self.extra_dims:
            basis = torch.cat([basis, field_coords[:, self.basis_dims :]], dim=-1)
        return basis

    def forward(self, field_coords: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        phi = self.features(field_coords)
        hidden = self.entry(phi).unsqueeze(0).expand(theta.shape[0], -1, -1)
        for layer in self.layers:
            hidden = layer(phi, theta, hidden)
        return self.projection(hidden)


class _ChebyshevTrunk(nn.Module):
    """Temporal trunk as a learned Chebyshev coefficient vector: one gain per instant."""

    def __init__(self, degree: int = 6):
        super().__init__()
        self.degree = int(degree)
        self.coefficients = nn.Parameter(torch.zeros(degree + 1))

    def forward(self, times: torch.Tensor) -> torch.Tensor:
        basis = chebyshev_basis((2.0 * times - 1.0).unsqueeze(-1), self.degree)
        return (basis * self.coefficients).sum(dim=-1)


class ChebyshevBranchOperator(NeuralOperatorBase):
    """Branch/trunk operator with a Chebyshev branch and a Chebyshev trunk.

    Parameters
    ----------
    n_steps:
        Number of time instants the operator emits at once.
    out_dim:
        Output channels (for example water saturation and pressure).
    grid:
        ``(nx, ny)`` of the field the branch covers.
    field_dim:
        Columns of the per-cell input: the first two are ``x`` and ``y`` and are expanded
        in the Chebyshev basis; any further columns (normalised log permeability, for
        example) are used as raw scalars. ``3`` for a heterogeneous 2-D reservoir.
    trunk_degree, branch_order, branch_depth, branch_hidden, modulation_dim:
        Architecture. ``branch_depth=6, branch_order=6`` reproduces the reported 41 025
        parameters at 61 instants; ``branch_depth=5`` the capacity-matched 35 681.

    Forward
    -------
    ``forward(field_coords, theta, times)``: ``field_coords`` is ``(N, field_dim)`` with
    the whole grid flattened and already in ``[-1, 1]``; ``theta`` is
    ``(B, parameter_dim)``; ``times`` is ``(1, n_steps)`` in ``[0, 1]``. Returns
    ``(B, out_dim, n_steps, nx, ny)``.
    """

    def __init__(self, n_steps: int, grid: tuple[int, int], out_dim: int = 2, field_dim: int = 3,
                 trunk_degree: int = 6, branch_order: int = 6, branch_depth: int = 6,
                 branch_hidden: int = 64, modulation_dim: int = 32, parameter_dim: int = 2,
                 saturation_channel: int = 0):
        super().__init__()
        if grid[0] * grid[1] <= 0:
            raise ValueError("grid must be a positive (nx, ny)")
        self.n_steps = int(n_steps)
        self.grid = (int(grid[0]), int(grid[1]))
        self.out_dim = int(out_dim)
        self.field_dim = int(field_dim)
        self.saturation_channel = int(saturation_channel)
        self.branch = _ChebyshevBranch(
            basis_dims=2,
            extra_dims=max(0, self.field_dim - 2),
            out_channels=n_steps * out_dim,
            order=branch_order,
            depth=branch_depth,
            hidden_dim=branch_hidden,
            modulation_dim=modulation_dim,
            parameter_dim=parameter_dim,
        )
        self.trunk = _ChebyshevTrunk(trunk_degree)

    def forward(
        self,
        field_coords: torch.Tensor,
        theta: torch.Tensor,
        times: torch.Tensor,
        *,
        y_true: torch.Tensor | None = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        batch = theta.shape[0]
        field = self.branch(field_coords, theta).view(
            batch, self.out_dim, self.n_steps, self.grid[0], self.grid[1]
        )
        gain = self.trunk(times).view(1, 1, self.n_steps, 1, 1)
        y = field * gain
        if self.saturation_channel is not None:
            channels = [y[:, index : index + 1] for index in range(self.out_dim)]
            channels[self.saturation_channel] = torch.sigmoid(channels[self.saturation_channel])
            y = torch.cat(channels, dim=1)

        losses = {"total": torch.zeros((), device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]
        return OperatorOutput(y=y, losses=losses, extras={"grid": self.grid})


class HadamardDeepONet(NeuralOperatorBase):
    """Branch/trunk operator with a convolutional branch and an MLP trunk.

    The branch sees the scenario as a spatial input (coordinates, normalised log
    permeability, and one bitmask channel per varying control) and emits the whole field
    for every instant; the trunk reads the whole time vector and returns one gain per
    instant. The two are combined elementwise.

    Parameters
    ----------
    in_channels:
        Branch input channels, ``3 + number of varying controls`` in the reference use.
    branch_width, branch_depth:
        The branch is ``in_channels -> width -> 2*width -> 2*width -> n_steps*out_dim``,
        with 3x3 convolutions and one final 1x1 projection.
    trunk_hidden:
        Hidden width of the MLP that maps the time vector to the gains.
    saturation_channel:
        Channel passed through a sigmoid, or ``None``. The reference found that the trunk
        alone can produce gains outside ``[0, 1]``; because every scenario shares the same
        trunk curve, that error contaminates all of them identically, and the sigmoid is
        what bounds it.

    Forward
    -------
    ``forward(branch_fields, times)`` with ``branch_fields`` of shape
    ``(B, in_channels, nx, ny)`` and ``times`` of shape ``(1, n_steps)``. Returns
    ``(B, out_dim, n_steps, nx, ny)``.
    """

    def __init__(self, in_channels: int, n_steps: int, grid: tuple[int, int], out_dim: int = 2,
                 branch_width: int = 16, branch_depth: int = 3, trunk_hidden: int = 32,
                 saturation_channel: int | None = 0):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_steps = int(n_steps)
        self.grid = (int(grid[0]), int(grid[1]))
        self.out_dim = int(out_dim)
        self.saturation_channel = saturation_channel

        layers: list[nn.Module] = []
        channels = self.in_channels
        for index in range(branch_depth):
            width = branch_width * (2**index) if index < 2 else branch_width * 2
            layers += [nn.Conv2d(channels, width, 3, padding=1), nn.SiLU()]
            channels = width
        layers.append(nn.Conv2d(channels, n_steps * out_dim, 1))
        self.branch = nn.Sequential(*layers)

        trunk_layers: list[nn.Module] = []
        width = n_steps
        for _ in range(2):
            trunk_layers += [nn.Linear(width, trunk_hidden), nn.SiLU()]
            width = trunk_hidden
        trunk_layers.append(nn.Linear(width, n_steps))
        self.trunk = nn.Sequential(*trunk_layers)

    def forward(
        self,
        branch_fields: torch.Tensor,
        times: torch.Tensor,
        *,
        y_true: torch.Tensor | None = None,
        return_loss: bool = False,
    ) -> OperatorOutput:
        if branch_fields.dim() != 4:
            raise ValueError(f"branch_fields must be (B, C, nx, ny), got {tuple(branch_fields.shape)}")
        batch = branch_fields.shape[0]
        field = self.branch(branch_fields).view(
            batch, self.out_dim, self.n_steps, self.grid[0], self.grid[1]
        )
        gain = self.trunk(times).view(1, 1, self.n_steps, 1, 1)
        y = field * gain
        if self.saturation_channel is not None:
            channels = [y[:, index : index + 1] for index in range(self.out_dim)]
            channels[self.saturation_channel] = torch.sigmoid(channels[self.saturation_channel])
            y = torch.cat(channels, dim=1)

        losses = {"total": torch.zeros((), device=y.device)}
        if return_loss and y_true is not None:
            losses["mse"] = self.mse(y, y_true)
            losses["total"] = losses["mse"]
        return OperatorOutput(y=y, losses=losses, extras={"grid": self.grid})


__all__ = [
    "chebyshev_basis",
    "chebyshev_features",
    "ModulatedPolynomialLayer",
    "ChebyshevPolynomialNeuralOperator",
    "ChebyshevBranchOperator",
    "HadamardDeepONet",
]
