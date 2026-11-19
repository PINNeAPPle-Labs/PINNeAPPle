"""The coordinate network that answers an AirfRANS case point by point, and its two physics residuals."""
from __future__ import annotations

import torch
import torch.nn as nn

from .features import FEATURE_NAMES

__all__ = ["CoordinateNet", "WallAwareNet", "physics_loss", "wall_layer_profile", "flat_plate_slope"]

#: the release's kinematic viscosity, in m²/s. Every case of AirfRANS uses the same fluid; Reynolds number varies
#: through the inlet velocity and the chord, which is what the ``log_inlet_velocity`` channel and the geometry carry.
NU = 1.54981e-05

#: every airfoil of the release is normalised to this chord, so ``x_over_chord`` is the dimensional distance
CHORD = 1.0


def flat_plate_slope(edge_velocity: torch.Tensor, x_over_chord: torch.Tensor, inlet_velocity: torch.Tensor,
                     coefficient: float = 0.0576, exponent: float = 0.2) -> torch.Tensor:
    """The wall slope a flat plate would have at this edge speed and running length — the physics prior for ``τ_w``.

    ``τ_w = ½ρU_e²·C_f`` with ``C_f = 0.0576 Re_x^{-1/5}`` (turbulent flat plate), converted to the network's
    dimensionless slope ``τ_w/ν · c/U∞``. It is pointwise, needs nothing but quantities the model already has, and —
    the reason it exists — it moves the *magnitude* of the prediction from the network into a law. Measured on the
    training sample, the wall slope has rms 26 500 and median 6 500, and every attempt to regress it directly
    overfit the training cases' magnitudes: with this prior the readout only has to learn a multiplicative correction
    of order one, which is a far smaller thing to get wrong.

    ``edge_velocity`` is the *scaled* velocity just outside the wall layer, in units of ``U∞`` — the outer head plus the
    freestream channels, which travel as a unit direction — so the prior stays differentiable and carries no truth;
    ``inlet_velocity`` puts it back into metres per second.
    """
    inlet = inlet_velocity.clamp_min(1e-3)
    speed = torch.linalg.vector_norm(edge_velocity, dim=1).clamp_min(1e-3) * inlet      # m/s
    x = x_over_chord.abs().clamp_min(1e-4) * CHORD                                        # m
    reynolds_x = (speed * x / NU).clamp_min(1.0)
    friction = coefficient * reynolds_x ** (-exponent)
    wall_shear = 0.5 * speed ** 2 * friction
    return wall_shear / NU * (CHORD / inlet)



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
                 slope_scale: float = 1e-3, shape_features: int = 0, slope_detach: bool = True,
                 slope_prior: bool = False):
        super().__init__()
        self.width, self.depth, self.fourier = width, depth, fourier
        self.shape_features = shape_features
        # with the prior the readout predicts a log-correction to the flat-plate law instead of the slope itself: its
        # output is O(1) and its target is bounded, which is the whole point — see flat_plate_slope
        self.slope_prior = slope_prior
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
        # the wall readout is its own small network on the raw channels, not a head on the trunk. Measured three ways:
        # a linear probe on those channels reaches 74 % of the wall slope, a standalone 256-wide MLP 60 %, and a head
        # reading the trunk output 330 % — the shared encoder smooths away exactly the near-wall detail this needs,
        # and letting its gradients into the trunk costs the field. Two networks, two jobs.
        input_features = len(FEATURE_NAMES) + shape_features
        self.slope_width = 256
        # per-feature standardisation, not per-sample: LayerNorm across the channels mixes the 64 shape bins with the
        # local geometry and loses the magnitudes that carry the information — measured, that alone was the difference
        # between 30 % and 300 % on the wall slope. The statistics are filled from the training sample in train_wall.
        self.register_buffer("feature_mean", torch.zeros(input_features))
        self.register_buffer("feature_std", torch.ones(input_features))
        self.head_slope = nn.Sequential(nn.Linear(input_features, self.slope_width), nn.SiLU(),
                                        nn.Linear(self.slope_width, self.slope_width), nn.SiLU(),
                                        nn.Linear(self.slope_width, 1 if slope_prior else 2))
        self.head_pressure = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))

    def _inputs(self, x: torch.Tensor) -> torch.Tensor:
        """The raw channels, extended with the Fourier bands of the coordinates when the model uses them."""
        if not self.fourier:
            return x
        bands = []
        for k in range(1, self.fourier + 1):
            for column in (0, 1):                      # x/c and y/c: the periodicity a coordinate MLP cannot invent
                angle = (2.0 * 3.141592653589793 * k) * x[:, column:column + 1]
                bands += [torch.sin(angle), torch.cos(angle)]
        return torch.cat([x] + bands, dim=1)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(x)
        for block in self.blocks:
            hidden = hidden + block(hidden)            # residual: the trunk starts as the identity and refines
        return hidden

    def forward(self, x: torch.Tensor, heads: bool = False):
        """``(N, 3)`` of ``(u/U∞, v/U∞, p/U∞²)`` with no-slip built in; the raw heads too when ``heads`` is set."""
        hidden = self._encode(self._inputs(x))
        outer = self.head_velocity(hidden)             # perturbation from the freestream, as in the plain network
        # the statistics are the wall nodes' (train.py), so a far-field row standardises to a huge z-score; clipped,
        # because the readout's output is exponentiated below and an unbounded z there is an inf, not a large number
        wall_inputs = ((x - self.feature_mean) / self.feature_std).clamp(-8.0, 8.0)
        raw = self.head_slope(wall_inputs)             # independent of the trunk
        if self.slope_prior:
            # the law supplies the magnitude, the readout a log-correction and nothing else
            edge = outer + x[:, 7:9]
            prior = flat_plate_slope(edge, x[:, 0], torch.exp(x[:, 9]))
            direction = edge / torch.linalg.vector_norm(edge, dim=1, keepdim=True).clamp_min(1e-6)
            slope = prior[:, None] * torch.exp(raw.clamp(-6.0, 6.0)) * direction
        else:
            slope = raw / self.slope_scale
        pressure = self.head_pressure(hidden)[:, 0]
        distance = x[:, 2]                             # sdf / chord
        weight = wall_layer_profile(distance, float(self.wall_layer))
        velocity = weight[:, None] * (outer + x[:, 7:9]) + (1.0 - weight)[:, None] * slope * distance[:, None]
        prediction = torch.cat([velocity, pressure[:, None]], dim=1)
        if heads:
            return {"prediction": prediction, "slope": slope, "outer": outer, "pressure": pressure}
        return prediction
