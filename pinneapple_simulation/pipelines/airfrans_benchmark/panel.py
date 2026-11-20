"""A panel method: the inviscid field around a body, as the prior the competing entry feeds its network.

The measured gap is where the flow is hardest, not where it is easiest: on our best entry 47 % of the nodes sit within
0.01 chords of the wall and they carry **89.5 % of the velocity error energy** (5.79 % relative L2 there against 1.61 %
in the far field). The first entry of the FluidsBench AirfRANS leaderboard reaches 0.42 % overall, and its `submission.json`
says how: a potential-flow (panel-method) prior as an input. Potential flow is exact for the outer problem — the edge
velocity, the pressure and the far field — so a network that starts from it only has to learn the viscous correction,
which is what the boundary layer is.

This is the constant-strength source + constant-strength vortex method (Hess–Smith): each panel carries an unknown source
strength and the whole body a single vortex strength, the boundary condition is no penetration at every panel midpoint,
and the Kutta condition closes the system. It is validated against the exact cylinder solution the repository already
uses as its reference (`examples/fluidsbench_airfrans`), because a prior that is wrong by a sign is worse than no prior.

Nothing here knows about airfoils: it takes the contour, the freestream and the angle, and returns the velocity and the
pressure at whatever points are asked for.
"""
from __future__ import annotations

import numpy as np

__all__ = ["panel_influence", "solve_panels", "panel_flow", "airfoil_contour", "camber_line",
           "zero_lift_angle", "panel_prior", "PANEL_CHANNELS"]




def panel_influence(collocation: np.ndarray, first: np.ndarray, second: np.ndarray) -> tuple[np.ndarray, ...]:
    """The Hess–Smith influence coefficients of every panel on every collocation point.

    Returns ``(source_u, source_v, vortex_u, vortex_v)``, each ``(P, N)``: the velocity a unit-strength source or vortex
    on panel ``j`` induces at point ``i``, in the global frame. The logarithms are the source's and the angles the
    vortex's; the angle is unwrapped with ``atan2`` of the cross and dot products, which is what makes the far-field
    terms cancel instead of wrapping to the wrong branch."""
    collocation = np.asarray(collocation, float)
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    length = np.linalg.norm(second - first, axis=1)
    direction = (second - first) / length[:, None]
    normal = np.stack([-direction[:, 1], direction[:, 0]], axis=1)

    offset = collocation[:, None, :] - first[None, :, :]
    along = np.einsum("kj,pkj->pk", direction, offset)         # in the panel's own frame
    across = np.einsum("kj,pkj->pk", normal, offset)

    # a collocation point lies exactly on its own panel, and there `across` is a rounding residue whose *sign* decides
    # whether the self-term is +1/2 or −1/2. This frame's normal is the inward one for a counter-clockwise contour, so
    # the fluid side is `across` just *below* zero; pinning the residue to −0 makes the diagonal exactly +1/2 along the
    # outward normal, which is the limit the boundary condition is imposed in — and the value the field is later
    # evaluated with, so the solve and the evaluation cannot disagree by a whole panel strength
    across = np.where(np.abs(across) < 1e-12, -0.0, across)
    radius1 = np.hypot(along, across)
    radius2 = np.hypot(along - length[None, :], across)
    logarithm = np.log(np.maximum(radius1, 1e-30) / np.maximum(radius2, 1e-30))
    # the angle the panel subtends at the point: atan2 of the cross and dot products of the vectors to its two ends
    theta = np.arctan2(across * length[None, :],
                       along * along + across * across - along * length[None, :])

    # Katz & Plotkin's constant-strength forms: u = (σ/4π)·ln(r₁²/r₂²) = (σ/2π)·ln(r₁/r₂), v = (σ/2π)·θ, and the
    # vortex is the same pair with the two components exchanged and one sign flipped
    source_along = logarithm / (2.0 * np.pi)
    source_across = theta / (2.0 * np.pi)
    vortex_along = theta / (2.0 * np.pi)
    vortex_across = -logarithm / (2.0 * np.pi)

    source_u = direction[None, :, 0] * source_along + normal[None, :, 0] * source_across
    source_v = direction[None, :, 1] * source_along + normal[None, :, 1] * source_across
    vortex_u = direction[None, :, 0] * vortex_along + normal[None, :, 0] * vortex_across
    vortex_v = direction[None, :, 1] * vortex_along + normal[None, :, 1] * vortex_across
    return source_u, source_v, vortex_u, vortex_v


def solve_panels(first: np.ndarray, second: np.ndarray, freestream: np.ndarray,
                 kutta: bool = False, circulation: float | None = None) -> dict[str, np.ndarray]:
    """The source strength of every panel of one inviscid flow, and the vortex strength the contour carries.

    ``freestream`` is the undisturbed velocity vector in the same frame as the contour, and the no-penetration condition
    is imposed at every panel midpoint. The circulation is then a single scalar and there are three ways to close the
    system: none (a body with no sharp edge — the case the exact circular-cylinder solution validates), a prescribed
    value (thin-airfoil theory), or the Kutta condition on the two panels that meet at the trailing edge.

    A prescribed circulation has to enter the *solve*, not be added to the field afterwards: the vortex sheet induces a
    normal velocity of its own, and a source distribution solved without it then lets the flow through the wall by an
    amount of the order of the freestream. Measured on a case of the release, adding the vortex after the solve left a
    normal velocity eight times the freestream and a lift off by four orders of magnitude.

    The system that remains is not ill-conditioned once the contour is a contour: over the release's 1 000 sections its
    condition number is 18 at the median and 1.7·10⁴ at worst, which double precision solves exactly — a plain solve and
    a truncated one agree to every digit of the lift on 125 cases."""
    first = np.asarray(first, float)
    second = np.asarray(second, float)
    freestream = np.asarray(freestream, float)
    # the outward normal below is only outward for a counter-clockwise contour, and the self-influence's sign depends on
    # it, so a contour that arrives clockwise is walked the other way rather than silently inverted
    signed_area = 0.5 * np.sum(first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1])
    if signed_area < 0.0:
        first, second = second[::-1].copy(), first[::-1].copy()
    collocation = 0.5 * (first + second)
    length = np.linalg.norm(second - first, axis=1)
    direction = (second - first) / length[:, None]
    # the outward normal of a contour traversed counter-clockwise; the influence coefficients use their own frame, which
    # is validated against a brute-force integration in the tests
    normal = np.stack([direction[:, 1], -direction[:, 0]], axis=1)

    source_u, source_v, vortex_u, vortex_v = panel_influence(collocation, first, second)

    # normal and tangential components, at the collocation points, of what one unit of each unknown induces
    source_normal = source_u * normal[:, 0:1] + source_v * normal[:, 1:2]          # (N, N): panel j at point i
    # a collocation point lies on its own panel, where the induced normal velocity is ±½ depending on the side it is
    # approached from; the boundary condition is the *fluid* side, whose value is +½ along the outward normal
    np.fill_diagonal(source_normal, 0.5)
    source_tangent = source_u * direction[:, 0:1] + source_v * direction[:, 1:2]
    vortex_normal = (vortex_u * normal[:, 0:1] + vortex_v * normal[:, 1:2]).sum(axis=1)   # (N,): one γ for the body
    vortex_tangent = (vortex_u * direction[:, 0:1] + vortex_v * direction[:, 1:2]).sum(axis=1)
    free_normal = freestream[0] * normal[:, 0] + freestream[1] * normal[:, 1]

    count = len(first)
    if circulation is not None and not kutta:
        # the sheet strength that carries the asked circulation, and the normal velocity it induces — which the sources
        # then have to cancel, so the wall stays closed
        sheet = float(circulation) / float(np.sum(length))
        solution = np.linalg.solve(source_normal, -free_normal - sheet * vortex_normal)
        return {"source": solution, "vortex": sheet, "length": length, "direction": direction, "normal": normal,
                "first": first, "second": second}
    if not kutta:
        solution = np.linalg.solve(source_normal, -free_normal)
        return {"source": solution, "vortex": 0.0, "length": length, "direction": direction, "normal": normal,
                "first": first, "second": second}

    matrix = np.zeros((count + 1, count + 1))
    matrix[:count, :count] = source_normal
    matrix[:count, count] = vortex_normal
    # the Kutta condition: the tangential velocity is the same on the two panels that meet at the trailing edge
    # both panels are walked in the contour's own direction, so equal *magnitudes* at the trailing edge is a difference
    matrix[count, :count] = source_tangent[0] - source_tangent[-1]
    matrix[count, count] = vortex_tangent[0] - vortex_tangent[-1]
    right = np.zeros(count + 1)
    right[:count] = -free_normal
    right[count] = -(freestream[0] * (direction[0, 0] - direction[-1, 0])
                     + freestream[1] * (direction[0, 1] - direction[-1, 1]))

    solution = np.linalg.solve(matrix, right)
    return {"source": solution[:count], "vortex": float(solution[count]), "length": length,
            "direction": direction, "normal": normal, "first": first, "second": second}


def panel_flow(first: np.ndarray, second: np.ndarray, freestream: np.ndarray,
               points: np.ndarray, *, kutta: bool = False, circulation: float | None = None) -> dict[str, np.ndarray]:
    """Velocity and pressure of the inviscid flow at ``points`` — the prior the network reads.

    The pressure comes from Bernoulli in the release's own scaling (``p`` in m²/s², kinematic, unit density), which is
    what the network's pressure channel is normalised by."""
    solution = solve_panels(first, second, freestream, kutta=kutta, circulation=circulation)
    freestream = np.asarray(freestream, float)
    points = np.asarray(points, float)
    # the panels the solve used, in its own orientation: the strengths are ordered that way
    source_u, source_v, vortex_u, vortex_v = panel_influence(points, solution["first"], solution["second"])
    velocity = np.stack([
        freestream[0] + source_u @ solution["source"] + solution["vortex"] * vortex_u.sum(axis=1),
        freestream[1] + source_v @ solution["source"] + solution["vortex"] * vortex_v.sum(axis=1),
    ], axis=1)
    speed2 = np.einsum("ij,ij->i", velocity, velocity)
    pressure = 0.5 * (float(freestream @ freestream) - speed2)
    return {"velocity": velocity, "pressure": pressure, "solution": solution}


#: the channels :func:`panel_prior` appends to a case's features, in order: the potential-flow velocity in units of the
#: freestream and its kinematic pressure in units of the freestream's dynamic head — the same scaling the network's
#: targets use, so the prior is a prediction already in the network's own units
PANEL_CHANNELS = ("potential_u", "potential_v", "potential_p")

#: how far above the freestream the prior's speed is allowed to go. A panel contour has a corner wherever the section
#: turns fastest, and near the leading edge the discretised velocity there reaches 28 times the freestream on 0.2 % of
#: the release's nodes. The release's own flow never exceeds 4.2 times the freestream (its lowest pressure is
#: −8.7 U²), so the clip costs nothing real and keeps a handful of corner nodes from dominating a training block
PANEL_VELOCITY_CLIP = 4.5

#: panels per contour. The exact cylinder solution converges at 0.17 % (60 panels), 0.046 % (240); a section's contour
#: carries 1 000 nodes from the release, and 160 panels change the surface pressure by under 1 % from 120 to 240
PANEL_COUNT = 160


def _chord_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The section's own frame — ``(origin, chord_axis, normal_axis)`` — from its point cloud.

    The chord is the cloud's principal axis, not the global ``x``: taking it from the geometry keeps the contour right
    for a case that arrives rotated, and the release's cases differ in incidence while their sections do not."""
    cloud = np.asarray(points, float)
    origin = cloud.mean(axis=0)
    centred = cloud - origin
    values, vectors = np.linalg.eigh(centred.T @ centred / len(centred))
    del values
    chord_axis = vectors[:, -1]
    if chord_axis[0] < 0.0:
        chord_axis = -chord_axis
    return origin, chord_axis, np.array([-chord_axis[1], chord_axis[0]])


def _surface_masks(along: np.ndarray, across: np.ndarray, bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Which nodes lie on the upper and on the lower surface, from each chord bin's own mid height.

    Splitting about a *local* mid rather than the chord line is what makes it work on a cambered section, where the
    lower surface can sit above the chord line over the aft third."""
    span = max(float(np.ptp(along)), 1e-12)
    index = np.minimum(((along - along.min()) / span * bins).astype(int), bins - 1)
    above, below = np.zeros(len(along), bool), np.zeros(len(along), bool)
    for i in range(bins):
        rows = np.flatnonzero(index == i)
        if rows.size == 0:
            continue
        mid = 0.5 * (across[rows].max() + across[rows].min())
        above[rows] = across[rows] >= mid
        below[rows] = ~above[rows]
    return above, below


def _chain(surface: np.ndarray, start: int = 0) -> np.ndarray:
    """Order one surface's nodes along the surface, by walking to the nearest node not yet visited.

    Ordering by chord position is the obvious rule and it is wrong at the leading edge, where the section turns almost
    vertical and a dozen nodes share one chord position: their order in the walk is decided by a tie-break that has the
    opposite sign at the leading and the trailing edge. The walk has no such ambiguity, because within one surface the
    node spacing is uniform and no two nodes of it come near each other except their neighbours along the surface."""
    points = np.asarray(surface, float)
    remaining = np.ones(len(points), bool)
    remaining[start] = False
    order, current = [start], start
    for _ in range(len(points) - 1):
        candidates = np.flatnonzero(remaining)
        step = np.linalg.norm(points[candidates] - points[current], axis=1)
        current = int(candidates[np.argmin(step)])
        order.append(current)
        remaining[current] = False
    return np.asarray(order)


def _ordered_surface(along: np.ndarray, across: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """One surface's nodes in the order they lie along it.

    A few nodes of a cloud are filed on the wrong surface by the chord-bin split — 3 % on a synthetic section — and one
    of those at the far end of a surface is enough to make the walk jump the whole chord after it has already traversed
    the surface, which doubles the contour back over itself. The walk's step length says where that happens, so the
    longest unbroken run is kept and the strays are dropped: they belong to the other surface, and one node of four
    hundred is cheaper than a contour that crosses itself."""
    surface = np.stack([along[rows], across[rows]], axis=1)
    order = _chain(surface, int(np.argmin(surface[:, 0])))
    walk = surface[order]
    step = np.linalg.norm(np.diff(walk, axis=0), axis=1)
    if len(step) and step.max() > 20.0 * max(float(np.median(step)), 1e-12):
        edges = np.concatenate([[0], np.flatnonzero(step > 20.0 * np.median(step)) + 1, [len(walk)]])
        piece = int(np.argmax(np.diff(edges)))
        walk = walk[edges[piece]:edges[piece + 1]]
    return walk


def airfoil_contour(points: np.ndarray, panels: int = PANEL_COUNT,
                    frame: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
                    bins: int = 48, cluster: float = 0.0) -> np.ndarray:
    """A closed contour from the release's airfoil point cloud, its two surfaces resampled for a panel solve.

    The cloud arrives unordered — consecutive nodes jump between the two surfaces — so it cannot be used as a contour
    directly. What it does have is that each surface is a *graph over the chord*: binning the cloud on the chord and
    splitting every bin above and below its own mid height separates the two surfaces, and a nearest-neighbour walk
    within each then puts its nodes in order along it. Two shorter constructions fail and both were measured: ordering
    the whole cloud by the angle around the chord's midpoint doubles back over itself on the release's more distorted
    sections, and ordering a surface by chord position reverses at the leading edge, where the section turns almost
    vertical. Either mistake produces a contour that crosses itself, which solves to a field five orders of magnitude
    away rather than a few percent.

    ``cluster`` blends the panel distribution between uniform in arc length (0) and a cosine spacing that piles panels up
    at the leading and trailing edges (1). Uniform is the better prior and it is not the intuitive answer: clustering
    shortens the panels at the nose to a quarter of the release's node spacing and makes the no-penetration system
    nearly singular, and on sixty cases of the release the lift error is 0.070 median against 0.143.

    The result is counter-clockwise and ``panels`` long, so that ``[dy, -dx]`` on a panel is outward for every caller."""
    cloud = np.asarray(points, float)
    origin, chord_axis, normal_axis = _chord_frame(cloud) if frame is None else frame
    centred = cloud - origin
    along, across = centred @ chord_axis, centred @ normal_axis

    above, below = _surface_masks(along, across, bins)
    if not above.any() or not below.any():
        raise ValueError("the point cloud does not separate into two surfaces")
    upper = _ordered_surface(along, across, np.flatnonzero(above))
    lower = _ordered_surface(along, across, np.flatnonzero(below))

    count = panels // 2 + 1
    uniform = np.linspace(0.0, 1.0, count)
    parameter = uniform + cluster * (0.5 * (1.0 - np.cos(np.pi * uniform)) - uniform)
    resampled = []
    for surface in (upper, lower[::-1]):    # the second surface is walked back, so the contour closes
        keep = np.concatenate([[True], np.linalg.norm(np.diff(surface, axis=0), axis=1) > 1e-12])
        surface = surface[keep]
        arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(surface, axis=0), axis=1))])
        arc /= max(arc[-1], 1e-30)
        resampled.append(np.stack([np.interp(parameter, arc, surface[:, 0]),
                                   np.interp(parameter, arc, surface[:, 1])], axis=1))
    # the two surfaces keep the same sample parameter, which is what lets :func:`camber_line` pair them; the trailing
    # edge is kept once, in the first surface
    contour = np.concatenate([resampled[0], resampled[1][1:-1]])
    # counter-clockwise, so that ``[dy, -dx]`` on a panel is *outward* for every caller and the force quadrature has one
    # sign; a contour that arrives the other way is walked backwards rather than left for the reader to notice
    signed_area = 0.5 * np.sum(contour[:, 0] * np.roll(contour[:, 1], -1) - np.roll(contour[:, 0], -1) * contour[:, 1])
    if signed_area < 0.0:
        contour = contour[::-1]
    return origin + contour[:, :1] * chord_axis + contour[:, 1:] * normal_axis


def chord_length(points: np.ndarray) -> float:
    """The section's chord, from the extent of its point cloud along its own principal axis."""
    origin, chord_axis, _ = _chord_frame(points)
    along = (np.asarray(points, float) - origin) @ chord_axis
    return float(max(np.ptp(along), 1e-12))


def camber_line(contour: np.ndarray, frame: tuple[np.ndarray, np.ndarray, np.ndarray], bins: int = 48,
                trim: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
    """The camber line of a contour — ``(x over chord, y_c over chord)`` — from the mean of its two surfaces.

    The surfaces are separated with the same chord-bin split :func:`airfoil_contour` uses, so the answer does not depend
    on where the closed contour happens to be cut, and each is a graph over the chord: the lower one is interpolated
    onto the upper one's chord positions."""
    origin, chord_axis, normal_axis = frame
    centred = np.asarray(contour, float) - origin
    along, across = centred @ chord_axis, centred @ normal_axis
    above, below = _surface_masks(along, across, bins)
    if not above.any() or not below.any():
        return np.zeros(0), np.zeros(0)
    order = np.argsort(along[above])
    x_upper, y_upper = along[above][order], across[above][order]
    order = np.argsort(along[below])
    y_lower = np.interp(x_upper, along[below][order], across[below][order])
    extent = max(float(x_upper[-1] - x_upper[0]), 1e-12)
    x, y = (x_upper - x_upper[0]) / extent, 0.5 * (y_upper + y_lower) / extent
    # the slope below is taken against x, so repeated chord positions — which the leading edge always supplies — have
    # to go, or the difference falls on a zero step and the camber line comes back with infinities in it
    unique = np.concatenate([[True], np.diff(x) > 1e-9])
    x, y = x[unique], y[unique]
    inside = (x >= trim) & (x <= 1.0 - trim)
    return x[inside], y[inside]


def zero_lift_angle(contour: np.ndarray, frame: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    """The section's zero-lift incidence, in radians, from its camber line — thin-airfoil theory.

    ``α_L0 = (1/π)∫ (dy_c/dx)(1 − cos θ) dθ`` with ``x = ½(1 − cos θ)``: it comes from setting the total circulation
    — ``2πA₀ + πA₁`` with ``A₀ = α − (1/π)∫(dy_c/dx)dθ`` and ``A₁ = (2/π)∫(dy_c/dx)cos θ dθ`` — to zero. A circular-arc
    mean line of camber ``h`` gives ``−2h``, which is the classic check the tests use. The geometric incidence of a case
    is measured from the chord, and a cambered section lifts at zero incidence; without this the prescribed circulation
    is the symmetric one and the prior has the lift of the wrong airfoil."""
    x, camber = camber_line(contour, frame)
    if len(x) < 8:
        return 0.0
    slope = np.gradient(camber, x)
    theta = np.arccos(np.clip(1.0 - 2.0 * x, -1.0, 1.0))
    return float(np.trapezoid(slope * (1.0 - np.cos(theta)), theta) / np.pi)


def panel_prior(features: np.ndarray, airfoil_points: np.ndarray, panels: int = PANEL_COUNT,
                chunk: int = 16_384, cluster: float = 0.0, pressure: bool = True) -> np.ndarray:
    """The inviscid field of one case at its own points, as the three :data:`PANEL_CHANNELS` in the network's units.

    ``features`` is the release's per-point block ``(x, y, sdf, n_x, n_y, surface, ux, uy)``; the freestream is the same
    vector at every node of a case, so the first row's is used. The lift is prescribed from thin-airfoil theory rather
    than from a Kutta condition, because the release's sections have a finite-thickness trailing edge where the
    two-panel Kutta condition is not well posed, while the camber-line circulation is a well-conditioned scalar.

    ``pressure=False`` returns the third channel as zeros. It exists because the pressure is the one channel that
    measurably *hurt* when a network was given it: the inviscid pressure is the exact answer in the outer flow and
    wrong near the wall, where the release's own pressure carries its variance, and a held-out run at the scarce budget
    went from 4.75 % to 27.57 % on the domain pressure with the channel in and 11.77 % without it."""
    features = np.asarray(features, float)
    points = features[:, 0:2]
    origin, chord_axis, normal_axis = _chord_frame(airfoil_points)
    frame = (origin, chord_axis, normal_axis)
    contour = airfoil_contour(airfoil_points, panels, frame, cluster=cluster)
    first, second = contour, np.roll(contour, -1, axis=0)

    freestream = features[0, 6:8].astype(float)
    speed = float(np.hypot(*freestream))
    if speed <= 0.0:
        return np.zeros((len(points), len(PANEL_CHANNELS)), np.float32)
    incidence = float(np.arctan2(freestream @ normal_axis, freestream @ chord_axis))
    lift_angle = incidence - zero_lift_angle(contour, frame)
    circulation = np.pi * chord_length(airfoil_points) * speed * np.sin(lift_angle)

    out = np.empty((len(points), len(PANEL_CHANNELS)), np.float32)
    for start in range(0, len(points), chunk):
        stop = min(start + chunk, len(points))
        flow = panel_flow(first, second, freestream, points[start:stop], circulation=circulation)
        velocity = flow["velocity"] / speed
        magnitude = np.linalg.norm(velocity, axis=1)
        velocity *= np.minimum(1.0, PANEL_VELOCITY_CLIP / np.maximum(magnitude, 1e-12))[:, None]
        out[start:stop, 0] = velocity[:, 0]
        out[start:stop, 1] = velocity[:, 1]
        # Bernoulli on the clipped field, so the three channels stay one flow rather than two
        out[start:stop, 2] = 0.5 * (1.0 - np.einsum("ij,ij->i", velocity, velocity)) if pressure else 0.0
    return out
