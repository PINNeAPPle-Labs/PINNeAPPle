"""The coordinate network that answers an AirfRANS case point by point, and its two physics residuals."""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import FEATURE_NAMES

__all__ = ["CoordinateNet", "physics_loss"]


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
