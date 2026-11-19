"""The coordinate network that answers an AirfRANS case point by point, and its two physics residuals."""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import FEATURE_NAMES

__all__ = ["CoordinateNet", "WallAwareNet", "physics_loss", "wall_layer_profile"]


class CoordinateNet(nn.Module):
    """A point-wise MLP: eleven scaled features in, ``(u/U∞, v/U∞, p/U∞²)`` out.

    One architecture, deliberately: the campaign exists to measure what a cheap, well-conditioned coordinate network
    does on a benchmark whose leaders are much larger models, not to compete with them on parameter count."""

    def __init__(self, width: int = 256, depth: int = 4):
        super().__init__()
        layers, last = [], len(FEATURE_NAMES)
        for _ in range(depth):
            layers += [nn.Linear(last, width), nn.SiLU()]
            last = width
        layers.append(nn.Linear(last, 3))
        self.net = nn.Sequential(*layers)
        self.width, self.depth = width, depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def physics_loss(prediction: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
    """No penetration on the wall and Bernoulli's relation in the outer domain, both differentiable.

    ``features`` is the batch's feature block in the order of :data:`~.features.FEATURE_NAMES`: columns 2 (sdf/chord),
    4 and 5 (wall normals) and 6 (surface flag) are the ones the two terms read. The wall term uses every wall node of
    the batch; the outer term uses the points more than one chord from the airfoil, where the flow is essentially
    freestream."""
    normal_x, normal_y = features[:, 4], features[:, 5]
    surface, sdf = features[:, 6], features[:, 2]
    on_wall = surface > 0.5
    speed2 = prediction[:, 0] ** 2 + prediction[:, 1] ** 2
    penetration = prediction[:, 0] * normal_x + prediction[:, 1] * normal_y
    bernoulli = prediction[:, 2] - 0.5 * (1.0 - speed2)
    outer = sdf.abs() > 1.0
    total = prediction.sum() * 0.0
    if bool(on_wall.any()):
        total = total + (penetration[on_wall] ** 2).mean()
    if bool(outer.any()):
        total = total + (bernoulli[outer] ** 2).mean() + (speed2[outer] - 1.0).abs().mean()
    return total


def wall_layer_profile(distance, wall_layer: float):
    """The partition of unity that hands the first few percent of the boundary layer to the slope head.

    ``A(d) = 1 − exp(−(d/λ)²)`` vanishes at the wall *with zero derivative*, so ``U = A·U_outer + (1−A)·s·d`` has
    ``U(0) = 0`` exactly (no-slip, by construction rather than by penalty) and ``dU/dd|₀ = s``, the slope *away*
    from the wall — ``+wallShearStress/nu``, the quantity the evaluator reconstructs. ``λ`` is the height of the viscous sublayer in chords: at
    ``Re = 3·10⁶`` the first mesh cell sits at ``y⁺ ≈ 0.2`` and ``y⁺ = 5`` is ``λ ≈ 5·10⁻⁵`` chord, which is the band
    where the true profile is linear anyway."""
    distance = torch.clamp(distance, min=0.0)
    return 1.0 - torch.exp(-(distance / wall_layer) ** 2)


class WallAwareNet(nn.Module):
    """A residual coordinate network with a wall-slope head and a no-slip wall layer.

    The measured failure of the plain network is not the far field — it is the wall: the evaluator derives both the
    wall shear and every force coefficient from the velocity gradient at the surface nodes, and a point-wise MLP that
    misses the wall velocity by 3 % of ``U∞`` misses the gradient (and so the drag) by a factor of several, because the
    first cell of the mesh is ``10⁻⁶`` chords away. Predicting the slope explicitly, and making the near-wall field a
    linear layer with that slope, turns an unlearnable absolute-velocity requirement into a smooth regression on the
    wall."""

    def __init__(self, width: int = 512, depth: int = 4, fourier: int = 4, wall_layer: float = 5e-5,
                 slope_scale: float = 1e-3, shape_features: int = 0, slope_detach: bool = True):
        super().__init__()
        self.width, self.depth, self.fourier = width, depth, fourier
        self.shape_features = shape_features
        # the wall slope reads the trunk's representation but does not train it. Measured on the holdout: letting the
        # slope gradients reach the shared trunk either wrecked the field (the velocity MSE went from 0.027 to 0.45) or,
        # clipped, was starved of signal — a detached readout keeps the field that the data term fits and learns the
        # wall on top of it.
        self.slope_detach = slope_detach
        self.register_buffer("wall_layer", torch.tensor(float(wall_layer)))
        # the true scaled slope is O(10³); the head is asked for ``slope_scale`` times it, so it starts in a
        # well-conditioned range instead of having to grow its output by three orders of magnitude first
        self.register_buffer("slope_scale", torch.tensor(float(slope_scale)))
        self.embed = nn.Linear(len(FEATURE_NAMES) + shape_features + 4 * fourier, width)
        self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(width, width), nn.SiLU(),
                                                   nn.Linear(width, width), nn.SiLU()) for _ in range(depth)])
        self.head_velocity = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        self.head_slope = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 2))
        self.head_pressure = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        if self.fourier:
            bands = []
            for k in range(1, self.fourier + 1):
                for column in (0, 1):                  # x/c and y/c: the periodicity a coordinate MLP cannot invent
                    angle = (2.0 * 3.141592653589793 * k) * x[:, column:column + 1]
                    bands += [torch.sin(angle), torch.cos(angle)]
            x = torch.cat([x] + bands, dim=1)
        hidden = self.embed(x)
        for block in self.blocks:
            hidden = hidden + block(hidden)           # residual: the trunk starts as the identity and refines
        return hidden

    def forward(self, x: torch.Tensor, heads: bool = False):
        """``(N, 3)`` of ``(u/U∞, v/U∞, p/U∞²)`` with no-slip built in; the raw heads too when ``heads`` is set."""
        hidden = self._encode(x)
        outer = self.head_velocity(hidden)            # perturbation from the freestream, as in the plain network
        slope_features = hidden.detach() if self.slope_detach else hidden
        slope = self.head_slope(slope_features) / self.slope_scale   # dimensionless slope, as the wall truth is stored
        pressure = self.head_pressure(hidden)[:, 0]
        distance = x[:, 2]                            # sdf / chord
        weight = wall_layer_profile(distance, float(self.wall_layer))
        velocity = weight[:, None] * (outer + x[:, 7:9]) + (1.0 - weight)[:, None] * slope * distance[:, None]
        prediction = torch.cat([velocity, pressure[:, None]], dim=1)
        if heads:
            return {"prediction": prediction, "slope": slope, "outer": outer, "pressure": pressure}
        return prediction
