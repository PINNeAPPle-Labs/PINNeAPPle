"""Leak detection in a buried pipeline: the transient the hole makes, what two pressure transmitters see of it, and
which of the two is lying.

A crude-oil line is modelled as a slightly compressible liquid in an elastic pipe, one-dimensional and unsteady, by the
method of characteristics on the characteristic grid ``dx = a dt`` (Courant number 1) with Darcy-Weisbach friction:
along ``dx/dt = +-a`` the compatibility equations

    C+:  H_P = C_P - B Q_P,   C_P = H_A + B Q_A - R Q_A |Q_A|
    C-:  H_P = C_M + B Q_P,   C_M = H_B - B Q_B + R Q_B |Q_B|

with ``B = a / (g A)``, ``R = f dx / (2 g D A^2)`` and ``a`` from Korteweg. Upstream the line is fed through a pump
whose head falls with flow, downstream it ends in a valve discharging to a receiving head; both ends therefore move
when the line does, which is what makes an arrival time measurable at both of them.

A leak is a hole at one node, discharging to the ambient head with the orifice law ``Q_L = C_d A_h sqrt(2 g (H - H_amb))``.
It is an internal boundary condition, not a source term: the flow is discontinuous across the node and continuity there
reads ``Q_up = Q_down + Q_L``. Substituting the two characteristics leaves one quadratic in ``sqrt(H - H_amb)``, so the
node is solved in closed form and nothing is iterated:

    H_L = H_0 - (B/2) C_v sqrt(H_L - H_amb),   H_0 = (C_P + C_M) / 2,   C_v = C_d A_h sqrt(2 g)

The first step after the hole opens is exact and is what the tests check: the head at the hole drops by
``a Q_L / (2 g A)`` and each side of the line takes half the leak, ``Q_up = Q_0 + Q_L/2`` and ``Q_down = Q_0 - Q_L/2``.

**What is measured, not asserted.** The two transmitters are sampled, quantised and noisy; the faults injected into
them are a bias, a drift, extra noise and a stuck value. A change-point detector (a one-sided CUSUM on the residual
against the nominal transient) finds the arrival; the leak is localised from the difference of the two arrival times,
``x_L = (x_a + x_b - a (t_b - t_a)) / 2``, and its rate from the mass imbalance ``Q_in - Q_out`` once the line has
settled. The separation of an instrument fault from a leak is physics, not a classifier: **a leak is seen by both
transmitters, in the same direction, and a fault by one**, and only a leak leaves the line with less in it than went
in. Every gate is reported per case, so the separation is a confusion matrix rather than a claim.

The wave is the fast sensor and the poor one: it reports in seconds but resolves a leak no better than the transducer
noise. The balance is the slow sensor and the sharp one: it resolves a hundredth of the flow but knows nothing until
the line has settled, ``2L/a`` later. A system wants both, which is what :func:`detect` is.

**What this is not.** Steady friction only. On this line that is already worth a third of the wave: a rarefaction
that leaves the hole 1.26 m high is 0.84 m high when it reaches a transmitter 9.3 km away, because ``R q^2`` per cell
turns against the flow change for as long as the front travels. A real line adds unsteady friction and, in a lined
pipe, viscoelasticity on top, loses more still, and the smallest leak it gives up is larger than the one found here. The nominal transient is computed from known boundary conditions: the pump curve
and the valve schedule are assumed known, and the detector re-zeroes itself on the first twenty seconds of the same
record to absorb a static error in them. A boundary condition that changes *during* the record without the model
being told is indistinguishable from a leak by the pressure alone -- the balance is what refuses it, and pays for the
refusal with the settle time. Column separation is not modelled, and neither is a leak whose area grows once it opens
(a crack).

numpy only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "G", "Pipeline", "Leak", "Sensors", "Fault",
    "wave_speed", "friction_factor", "simulate", "observe", "expected", "detect", "measurement_check",
    "minimum_detectable_leak",
]

G = 9.80665


def wave_speed(bulk_modulus: float, density: float, diameter: float, wall: float, youngs: float,
               c1: float = 1.0) -> float:
    """Pressure-wave speed in a liquid-filled elastic pipe (Korteweg): ``a = sqrt((K/rho) / (1 + c1 K D / (E e)))``.
    ``c1 = 1 - nu^2`` for a pipe anchored against axial movement, which is what a buried line is."""
    return math.sqrt((bulk_modulus / density) / (1.0 + c1 * bulk_modulus * diameter / (youngs * wall)))


def friction_factor(reynolds: float, roughness: float, diameter: float, iterations: int = 60,
                    tolerance: float = 1e-12) -> float:
    """Darcy friction factor from Colebrook-White, ``1/sqrt(f) = -2 log10(eps/(3.7 D) + 2.51/(Re sqrt(f)))``, by
    fixed-point iteration from ``f = 0.02``. Laminar flow (``Re < 2300``): ``f = 64/Re``."""
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive")
    if reynolds < 2300:
        return 64.0 / reynolds
    relative = roughness / diameter
    f = 0.02
    for _ in range(iterations):
        nxt = (-2.0 * math.log10(relative / 3.7 + 2.51 / (reynolds * math.sqrt(f)))) ** -2
        if abs(nxt - f) < tolerance:
            return nxt
        f = nxt
    return f


@dataclass(frozen=True)
class Pipeline:
    """A line carrying one liquid, in the steady state the transient starts from.

    ``head_upstream`` is the pump-discharge head at the rated flow and ``head_downstream`` the head the valve discharges
    to, so the valve's own drop is what is left after the friction loss (and must be positive). ``pump_resistance`` is
    the pump curve ``H = H_shutoff - k Q^2``: at steady flow ``k Q_0^2`` is the head the pump gives up when the demand
    rises from zero, 30 m on a typical line. With ``k = 0`` the upstream end is a constant-head reservoir, which
    absorbs the wave completely -- the reason a transmitter sitting *on* such a boundary would see nothing.
    """
    length: float
    diameter: float
    wall: float = 0.008
    roughness: float = 4.5e-5
    density: float = 850.0
    bulk_modulus: float = 1.5e9
    viscosity: float = 1.0e-2
    youngs: float = 210.0e9
    flow: float = 0.25
    head_upstream: float = 260.0
    head_downstream: float = 20.0
    pump_resistance: float = 480.0

    def __post_init__(self) -> None:
        if min(self.length, self.diameter, self.wall, self.flow) <= 0:
            raise ValueError("length, diameter, wall and flow must be positive")
        if self.friction_head >= self.head_upstream - self.head_downstream:
            raise ValueError("the friction loss leaves no head across the valve: raise head_upstream")

    @property
    def area(self) -> float:
        return math.pi * self.diameter ** 2 / 4.0

    @property
    def wave_speed(self) -> float:
        return wave_speed(self.bulk_modulus, self.density, self.diameter, self.wall, self.youngs)

    @property
    def velocity(self) -> float:
        return self.flow / self.area

    @property
    def reynolds(self) -> float:
        return self.density * self.velocity * self.diameter / self.viscosity

    @property
    def friction(self) -> float:
        return friction_factor(self.reynolds, self.roughness, self.diameter)

    @property
    def friction_head(self) -> float:
        """Steady head lost to friction over the whole line, ``f (L/D) v^2 / (2 g)``."""
        return self.friction * self.length / self.diameter * self.velocity ** 2 / (2.0 * G)

    @property
    def valve_drop(self) -> float:
        return self.head_upstream - self.friction_head - self.head_downstream

    @property
    def travel_time(self) -> float:
        """``L/a``: the longest a wave can take to cross the line, and so the widest gap two arrivals can have."""
        return self.length / self.wave_speed

    def steady_head(self, x) -> np.ndarray:
        """Steady head profile ``H_0(x) = H_up - f x/D v^2/(2 g)``."""
        x = np.asarray(x, dtype=float)
        return self.head_upstream - self.friction * x / self.diameter * self.velocity ** 2 / (2.0 * G)


@dataclass(frozen=True)
class Leak:
    """A hole at ``distance`` along the line, ``area`` wide, opening at ``t_open``.

    ``head_ambient`` is the head just outside the hole: 0 for a leak to atmosphere on a line whose heads are gauge,
    and the burial head (a few metres) for a hole under a river crossing."""
    distance: float
    area: float
    discharge: float = 0.62
    head_ambient: float = 0.0
    t_open: float = 0.0

    def capacity(self) -> float:
        """``C_v = C_d A_h sqrt(2 g)``: the coefficient of ``Q_L = C_v sqrt(H - H_amb)``."""
        return self.discharge * self.area * math.sqrt(2.0 * G)


@dataclass(frozen=True)
class Sensors:
    """The two transmitters and the two flow meters, and what the acquisition does to their signals.

    ``upstream``/``downstream`` are the distances along the line of the two pressure transmitters from the inlet and
    the outlet: they sit on the line, not on the station boundary, which is what lets them see the incident wave
    before the boundary reflects it. A transmitter sits on a node of the model or not at all, so the position is
    snapped to the grid and the snapped value is what the observation reports back. ``sigma`` is the standard deviation of the head noise, ``bits`` and
    ``full_scale`` the quantisation of an ``bits``-bit analogue input of range ``full_scale``.
    """
    upstream: float = 500.0
    downstream: float = 500.0
    sigma: float = 0.15
    bits: int = 12
    full_scale: float = 400.0
    flow_sigma: float = 1.25e-3
    sample: float = 0.0351

    @property
    def step(self) -> float:
        """One quantisation step of the analogue input, in metres of head."""
        return self.full_scale / 2 ** self.bits


@dataclass(frozen=True)
class Fault:
    """One transducer fault, on channel ``"a"`` (upstream) or ``"b"`` (downstream), starting at ``t_start``.

    ``kind``: ``"offset"`` adds ``magnitude`` metres of head, ``"drift"`` ramps at ``magnitude`` metres per second,
    ``"noise"`` multiplies the noise standard deviation by ``magnitude``, and ``"stuck"`` freezes the reading at the
    last value before ``t_start`` (a transmitter whose output board has latched)."""
    kind: str
    channel: str
    t_start: float
    magnitude: float

    def __post_init__(self) -> None:
        if self.kind not in ("offset", "drift", "noise", "stuck"):
            raise ValueError(f"kind must be offset, drift, noise or stuck, not {self.kind!r}")
        if self.channel not in ("a", "b"):
            raise ValueError("channel must be 'a' or 'b'")


def simulate(pipe: Pipeline, *, n_reaches: int = 500, t_end: float | None = None, leak: Leak | None = None,
             valve=None, monitor=None, record: bool = False) -> dict:
    """Run the transient from the steady state of the intact line.

    ``leak`` opens a hole at ``Leak.t_open``; ``valve`` is a schedule ``t -> tau`` for the receiving valve, ``tau = 1``
    being the steady opening, which is how the normal-operation manoeuvres are made; ``monitor`` is a list of
    distances along the line at which the head is recorded (the transmitters); ``record`` returns the head at every
    node at every step, for the figures.

    Returns the time ``t``, the heads and flows at the two ends, the recorded heads at ``monitor_x``, the grid
    (``dx``, ``dt``) and the steady line it started from.
    """
    a = pipe.wave_speed
    A = pipe.area
    dx = pipe.length / n_reaches
    dt = dx / a
    if t_end is None:
        t_end = 6.0 * pipe.travel_time
    n_steps = int(math.ceil(t_end / dt))
    B = a / (G * A)
    R = pipe.friction * dx / (2.0 * G * pipe.diameter * A ** 2)
    x = np.linspace(0.0, pipe.length, n_reaches + 1)
    H = pipe.steady_head(x)
    q_left = np.full(n_reaches + 1, pipe.flow)           # flow at a node, in the reach to its left
    q_right = np.full(n_reaches + 1, pipe.flow)          # flow at a node, in the reach to its right
    H_nodes = [H.copy()] if record else None
    steady = H.copy()

    monitored = [] if monitor is None else [int(round(m / dx)) for m in monitor]
    for i in monitored:
        if not 0 <= i <= n_reaches:
            raise ValueError("a monitor sits outside the line")
    monitor_x = np.array([i * dx for i in monitored], dtype=float)
    monitored_heads = [H[monitored].copy()] if monitored else None

    i_leak = -1 if leak is None else int(round(leak.distance / dx))
    if leak is not None and not 0 < i_leak < n_reaches:
        raise ValueError("the leak must sit on an interior node: 0 < distance < length")
    leak_x = None if leak is None else i_leak * dx
    capacity = 0.0 if leak is None else leak.capacity()
    h_amb = 0.0 if leak is None else leak.head_ambient
    t_open = math.inf if leak is None else leak.t_open

    dh0 = pipe.valve_drop
    k_pump = pipe.pump_resistance
    h_shutoff = pipe.head_upstream + k_pump * pipe.flow ** 2
    ts = [0.0]
    head_up, head_down = [float(H[0])], [float(H[-1])]
    flow_in, flow_out = [pipe.flow], [pipe.flow]
    leaky = [0.0]
    split = [(0.0, 0.0)]
    t = 0.0
    for n in range(1, n_steps + 1):
        t = n * dt
        c_plus = H[:-1] + B * q_right[:-1] - R * q_right[:-1] * np.abs(q_right[:-1])     # into the reach's right node
        c_minus = H[1:] - B * q_left[1:] + R * q_left[1:] * np.abs(q_left[1:])           # into the reach's left node
        Hn = np.empty_like(H)
        Hn[1:-1] = 0.5 * (c_plus[:-1] + c_minus[1:])
        Hn[0] = Hn[-1] = 0.0                              # the two ends are set by their boundary conditions below
        qn_left = np.empty_like(q_left)
        qn_right = np.empty_like(q_right)
        qn_left[1:-1] = qn_right[1:-1] = (c_plus[:-1] - c_minus[1:]) / (2.0 * B)

        # upstream: a pump curve H = h_shutoff - k Q^2 crossed with the C- characteristic H = c_minus[0] + B Q,
        # that is k Q^2 + B Q + (c_minus[0] - h_shutoff) = 0, taking the positive root
        if k_pump > 0.0:
            disc = B * B - 4.0 * k_pump * (c_minus[0] - h_shutoff)
            qn_left[0] = qn_right[0] = (math.sqrt(disc) - B) / (2.0 * k_pump)
            Hn[0] = h_shutoff - k_pump * qn_left[0] ** 2
        else:
            qn_left[0] = qn_right[0] = (h_shutoff - c_minus[0]) / B
            Hn[0] = h_shutoff

        # downstream: a valve discharging to head_downstream, Q = tau Q_0 sqrt(dH / dH_0), reverse flow allowed
        tau = 1.0 if valve is None else float(valve(t))
        c_p = c_plus[-1]
        if tau <= 0.0:
            q = 0.0
        else:
            c_v = (tau * pipe.flow) ** 2 / dh0
            d = c_p - pipe.head_downstream
            if d >= 0:
                q = 0.5 * (-c_v * B + math.sqrt((c_v * B) ** 2 + 4.0 * c_v * d))
            else:
                q = 0.5 * (c_v * B - math.sqrt((c_v * B) ** 2 - 4.0 * c_v * d))
        qn_left[-1] = qn_right[-1] = q
        Hn[-1] = c_p - B * q

        out_of_hole = 0.0
        if leak is not None and t > t_open:
            h0 = 0.5 * (c_plus[i_leak - 1] + c_minus[i_leak])
            k = 0.5 * B * capacity
            rise = h0 - h_amb
            if rise <= 0.0:
                Hn[i_leak] = h_amb
                qn_right[i_leak] = 0.0
            else:
                z = 0.5 * (math.sqrt(k * k + 4.0 * rise) - k)        # z = sqrt(H_L - H_amb)
                Hn[i_leak] = h_amb + z * z
                qn_right[i_leak] = (Hn[i_leak] - c_minus[i_leak]) / B    # flow leaving the node to the right
            qn_left[i_leak] = qn_right[i_leak] + capacity * math.sqrt(max(Hn[i_leak] - h_amb, 0.0))
            out_of_hole = float(qn_left[i_leak] - qn_right[i_leak])

        H, q_left, q_right = Hn, qn_left, qn_right
        ts.append(t)
        head_up.append(float(H[0]))
        head_down.append(float(H[-1]))
        flow_in.append(float(q_left[0]))
        flow_out.append(float(q_right[-1]))
        leaky.append(out_of_hole)
        split.append((float(q_left[i_leak]) - pipe.flow, float(q_right[i_leak]) - pipe.flow)
                     if leak is not None else (0.0, 0.0))
        if monitored:
            monitored_heads.append(H[monitored].copy())
        if record:
            H_nodes.append(H.copy())

    out = {"t": np.array(ts), "head_up": np.array(head_up), "head_down": np.array(head_down),
           "flow_in": np.array(flow_in), "flow_out": np.array(flow_out), "x": x, "steady": steady,
           "dx": dx, "dt": dt, "n_reaches": n_reaches, "wave_speed": a, "leak_node": leak_x,
           "leak_flow": np.array(leaky), "leak_split": np.array(split)}
    if monitored:
        out["monitor_x"] = monitor_x
        out["monitor_head"] = np.array(monitored_heads)
    if record:
        out["field"] = np.array(H_nodes)
    return out


def observe(sim: dict, pipe: Pipeline, sensors: Sensors, *, seed: int = 0,
            faults: tuple[Fault, ...] = ()) -> dict:
    """What the two transmitters and the two flow meters read while ``sim`` happens.

    The acquisition samples at ``sensors.sample``, quantises to ``sensors.bits`` over ``sensors.full_scale`` and adds
    white noise of ``sensors.sigma`` (head) and ``sensors.flow_sigma`` (flow). The faults are applied per channel, in
    the order given, on top of the noise -- which is the way a transmitter fails: the process is what it is and the
    instrument reports something else.

    Returns ``t``, ``head_a``, ``head_b`` (upstream and downstream transmitters), ``flow_in``, ``flow_out``, and the
    positions and acquisition settings the detector needs to read it back.
    """
    if "monitor_head" not in sim:
        raise ValueError("simulate() must be given the two transmitter positions as monitor=[x_a, x_b]")
    t_sim = sim["t"]
    duration = float(t_sim[-1])
    n = int(math.floor(duration / sensors.sample)) + 1
    t = np.arange(n) * sensors.sample
    rng = np.random.default_rng(seed)
    truth = {"a": np.interp(t, t_sim, sim["monitor_head"][:, 0]),
             "b": np.interp(t, t_sim, sim["monitor_head"][:, 1])}
    reading = {}
    for channel in ("a", "b"):
        value = truth[channel] + rng.normal(0.0, sensors.sigma, n)
        for fault in faults:
            if fault.channel != channel:
                continue
            on = t >= fault.t_start
            if fault.kind == "offset":
                value = value + np.where(on, fault.magnitude, 0.0)
            elif fault.kind == "drift":
                value = value + np.where(on, fault.magnitude * (t - fault.t_start), 0.0)
            elif fault.kind == "noise":
                extra = rng.normal(0.0, sensors.sigma * math.sqrt(max(fault.magnitude ** 2 - 1.0, 0.0)), n)
                value = value + np.where(on, extra, 0.0)
            elif fault.kind == "stuck":
                frozen = value[max(int(np.argmax(on)) - 1, 0)] if on.any() else value[-1]
                value = np.where(on, frozen, value)
        reading[channel] = np.round(value / sensors.step) * sensors.step
    flow_in = np.interp(t, t_sim, sim["flow_in"]) + rng.normal(0.0, sensors.flow_sigma, n)
    flow_out = np.interp(t, t_sim, sim["flow_out"]) + rng.normal(0.0, sensors.flow_sigma, n)
    return {"t": t, "head_a": reading["a"], "head_b": reading["b"], "truth_a": truth["a"], "truth_b": truth["b"],
            "flow_in": flow_in, "flow_out": flow_out,
            "x_a": float(sim["monitor_x"][0]), "x_b": float(sim["monitor_x"][1]),
            "sigma": sensors.sigma, "sample": sensors.sample}


def _cusum(u: np.ndarray, slack: float, threshold: float) -> tuple[int, int, np.ndarray]:
    """One-sided CUSUM for a *drop* in the standardised residual ``u``: ``S_k = max(0, S_{k-1} - u_k - slack)``.

    Returns where the excursion that crossed ``threshold`` started (the last sample before it at which ``S`` was
    zero), where it crossed, and ``S`` itself, for the figure. ``(-1, -1, S)`` when nothing crosses."""
    s = 0.0
    start = 0
    out = np.zeros_like(u)
    for k in range(len(u)):
        if s == 0.0:
            start = k
        s = max(0.0, s - u[k] - slack)
        out[k] = s
        if s > threshold:
            return start, k, out
    return -1, -1, out


def _arrival(t: np.ndarray, u: np.ndarray, onset: int, level: float = -3.0) -> float:
    """When the front crossed ``level`` standard deviations, interpolated between the two samples that bracket it.

    The step is only as sharp as the model's own time step and the acquisition's, so the sample the crossing lands in
    is worth ``a dt / 2`` of the line; interpolating inside it is what buys the rest. The search is anchored on the
    CUSUM's onset so that a chance excursion of the noise, which has no CUSUM behind it, cannot move the arrival."""
    for k in range(max(onset - 3, 1), min(onset + 4, len(u))):
        if u[k] <= level:
            slope = u[k] - u[k - 1]
            if slope >= 0.0:
                return float(t[k])
            return float(t[k - 1] + (t[k] - t[k - 1]) * (level - u[k - 1]) / slope)
    return float(t[min(max(onset, 0), len(t) - 1)])


def _stuck(reading: np.ndarray, step: float, window: int) -> int:
    """Index of the first sample after which ``reading`` has not moved by one quantisation step for ``window``
    samples: a transmitter whose output board has latched. ``-1`` if it never happens."""
    if len(reading) <= window:
        return -1
    flat = np.abs(np.diff(reading)) < step
    run = np.convolve(flat.astype(int), np.ones(window, dtype=int), mode="valid")
    hit = np.flatnonzero(run >= window)
    return int(hit[0]) + 1 if hit.size else -1


def expected(sim: dict, sensors: Sensors) -> dict:
    """The noiseless heads the model says the two transmitters should have read while ``sim`` happened, on the
    acquisition's own time base. This is the reference the residual is taken against: the model, not another noisy
    record -- a reference record would double the noise the leak has to climb out of."""
    if "monitor_head" not in sim:
        raise ValueError("simulate() must be given the two transmitter positions as monitor=[x_a, x_b]")
    t_sim = sim["t"]
    n = int(math.floor(float(t_sim[-1]) / sensors.sample)) + 1
    t = np.arange(n) * sensors.sample
    return {"t": t, "head_a": np.interp(t, t_sim, sim["monitor_head"][:, 0]),
            "head_b": np.interp(t, t_sim, sim["monitor_head"][:, 1])}


def detect(observation: dict, reference: dict, pipe: Pipeline, sensors: Sensors, *, slack: float = 0.75,
           threshold: float = 12.0, arrival_level: float = 3.0, baseline: float = 20.0,
           imbalance_k: float = 5.0, settle: float = 45.0) -> dict:
    """Read the two residuals against the nominal transient and say what happened: a leak, a broken instrument, a
    change of operation, or nothing.

    ``reference`` is :func:`expected` of the intact line. The verdict rests on four pieces of physics, and each is
    reported so that the answer can be argued with:

    * **one side or both.** A leak pushes a rarefaction in both directions, so within ``(x_b - x_a)/a`` of the first
      departure the other transmitter must see it too, and **in the same direction: a hole can only lower the head**.
      A disturbance confined to one channel, of either sign, is the instrument.
    * **the arrival difference.** ``x_L = (x_a + x_b - a (t_a - t_b))/2`` -- a leak between the two transmitters is
      placed by the head start one front has over the other. A leak outside them sends one front past both
      transmitters, which no time difference can place, and the verdict says so instead of pretending otherwise.
    * **the mass balance.** ``Q_in - Q_out`` is the leak and nothing else, but only once the line has settled
      (``settle`` seconds, a little over the ``2L/a`` the line takes to forget a manoeuvre), which is why it confirms
      a leak rather than triggering one -- and why a valve that moved without the model being told is not called a
      leak.
    * **a flat channel.** A transmitter that has not moved by one quantisation step for five seconds has latched,
      whatever the line is doing.

    Both ends moving but the balance unchanged is ``"operation"``: a manoeuvre or a pump change the model did not
    know about. ``slack`` and ``threshold`` are the CUSUM's drift and decision interval in standard deviations of the
    residual, ``arrival_level`` the level the front is timed at, and ``imbalance_k`` the number of standard errors the
    settled imbalance must exceed. ``baseline`` is the quiet stretch at the start of the record over which each
    channel's offset from the model is learned and removed: without it a static calibration error in the model -- a
    friction factor assumed for a pipe that has aged -- is a permanent residual and the detector never stops alarming.
    """
    sigma = float(observation["sigma"])
    t = np.asarray(observation["t"], dtype=float)
    r_a = (observation["head_a"] - reference["head_a"]) / sigma
    r_b = (observation["head_b"] - reference["head_b"]) / sigma
    warm = t <= baseline                              # the record calibrates the model on its own quiet start
    if warm.sum() >= 10:
        r_a = r_a - float(np.mean(r_a[warm]))
        r_b = r_b - float(np.mean(r_b[warm]))
    residuals = {"a": r_a, "b": r_b}
    drop = {"a": _cusum(r_a, slack, threshold), "b": _cusum(r_b, slack, threshold)}
    rise = {"a": _cusum(-r_a, slack, threshold), "b": _cusum(-r_b, slack, threshold)}
    x_a, x_b = float(observation["x_a"]), float(observation["x_b"])
    a = pipe.wave_speed
    span = (x_b - x_a) / a                          # the widest gap two arrivals of one leak can have
    window = max(int(round(5.0 / sensors.sample)), 2)
    stuck = {"a": _stuck(observation["head_a"], sensors.step, window),
             "b": _stuck(observation["head_b"], sensors.step, window)}

    # the settled mass balance, asked of every record including the ones where nothing alarmed
    tail_window = t >= max(float(t[-1]) - settle, float(t[-1]) - 0.25 * float(t[-1]))
    counts = int(tail_window.sum())
    imbalance = float(np.mean((observation["flow_in"] - observation["flow_out"])[tail_window])) if counts >= 10 else None
    imbalance_error = sensors.flow_sigma / math.sqrt(counts) if counts >= 10 else None

    verdict = {"alarm": False, "source": "none", "channel": None, "time": None, "alarm_time": None,
               "leak_distance": None, "leak_rate": None,
               "evidence": {"span_s": span, "arrival_a": None, "arrival_b": None, "delta_t": None,
                            "both_ends": False, "imbalance": imbalance, "imbalance_error": imbalance_error,
                            "outside_transmitters": False, "balance_window": None, "confirmed": None,
                            "stuck": {k: (None if v < 0 else float(t[v])) for k, v in stuck.items()},
                            "cusum": {k: v[2] for k, v in drop.items()}}}
    mine = sorted({k for k, v in drop.items() if v[1] >= 0} | {k for k, v in rise.items() if v[1] >= 0}
                  | {k for k, v in stuck.items() if v >= 0})
    if not mine:
        return verdict
    verdict["alarm"] = True

    first = min(mine, key=lambda k: min([i for i in (drop[k][1], rise[k][1], stuck[k]) if i >= 0]))
    crossed = min(i for i in (drop[first][1], rise[first][1], stuck[first]) if i >= 0)
    verdict["channel"] = first
    verdict["time"] = verdict["alarm_time"] = float(t[crossed])

    drop_both = all(drop[k][1] >= 0 for k in ("a", "b"))
    margin = max(3.0 * sensors.sample, 0.05 * span)
    if drop_both:
        drop_both = abs(t[drop["b"][0]] - t[drop["a"][0]]) <= span + margin
    if not drop_both:
        # one side only, or both sides moving in opposite or non-leak directions -- unless the line is losing mass,
        # in which case the hole is there and the transient simply was not heard well enough to be placed
        verdict["source"] = "instrument" if len(mine) == 1 or any(stuck[k] >= 0 for k in mine) else "operation"
        if verdict["source"] == "operation":
            verdict["channel"] = None
        if imbalance is not None and imbalance > imbalance_k * imbalance_error:
            verdict["source"] = "leak"
            verdict["leak_rate"] = imbalance
        return verdict

    t_a = _arrival(t, residuals["a"], drop["a"][0], -arrival_level)
    t_b = _arrival(t, residuals["b"], drop["b"][0], -arrival_level)
    x_leak = (x_a + x_b - a * (t_b - t_a)) / 2.0
    verdict["evidence"].update(both_ends=True, arrival_a=t_a, arrival_b=t_b, delta_t=float(t_b - t_a))
    verdict["evidence"]["outside_transmitters"] = not x_a <= x_leak <= x_b
    verdict["source"] = "leak"
    verdict["leak_distance"] = float(min(max(x_leak, x_a), x_b))

    settled = t >= max(t[drop["a"][0]], t[drop["b"][0]]) + settle
    which = "settled" if settled.sum() >= 10 else "tail"
    window_mask = settled if which == "settled" else tail_window
    counts = int(window_mask.sum())
    if counts >= 10:
        settled_balance = float(np.mean((observation["flow_in"] - observation["flow_out"])[window_mask]))
        verdict["evidence"].update(imbalance=settled_balance, balance_window=which,
                                   imbalance_error=sensors.flow_sigma / math.sqrt(counts),
                                   confirmed=bool(settled_balance >= imbalance_k * sensors.flow_sigma
                                                  / math.sqrt(counts)))
        verdict["leak_rate"] = settled_balance
        if not verdict["evidence"]["confirmed"]:
            # nothing is leaving the line: the two drops were the line changing operation, not a hole
            verdict["source"] = "operation"
            verdict["leak_distance"] = verdict["leak_rate"] = None
            verdict["channel"] = None
    return verdict


def measurement_check(observation: dict, reference: dict, pipe: Pipeline, sensors: Sensors, **kwargs) -> bool:
    """``True`` when the reading itself is not trustworthy -- one transmitter departed from the nominal transient and
    the other one did not, or one has gone flat, while the line's mass balance is unchanged. The shape of
    :meth:`pinneapple_physics.PhysicsEnsemble` expects for ``measurement_check=``: a predicate over one observation.

    The mirror of :func:`detect` by construction, so that a verdict shown next to a reading is the detector's verdict
    and not a second opinion; ``kwargs`` go to :func:`detect`."""
    return detect(observation, reference, pipe, sensors, **kwargs)["source"] == "instrument"


def minimum_detectable_leak(pipe: Pipeline, sensors: Sensors, *, waves: float = 1.2,
                            attenuation: float = 0.0) -> dict:
    """The leak whose wave at the transmitter is ``waves`` standard deviations of the head noise, and the reasoning.

    The head change a hole makes at the instant it opens is the Joukowsky relation applied to the leak flow,
    ``dH = a Q_L / (2 g A)`` -- half the flow goes each way, hence the 2. Re-arranged, a leak is invisible to a
    transducer of noise ``sigma`` below ``Q_L = 2 g A sigma / a``, independently of the detector: the signal is not
    in the record that a better algorithm could find. ``waves`` is the margin the CUSUM needs over the noise, and a
    smaller one buys sensitivity with false alarms. ``attenuation``, the fraction of the wave's height lost between
    the hole and the transmitter (zero at the hole, and a third of it over 9 km on the line of the example), divides
    the answer by ``1 - attenuation``: the transducer sees what is left.

    This is the amplitude argument and it is optimistic. A leak also has to clear the CUSUM's slack at *both*
    transmitters, and the far one sees less, which is why :func:`detect` finds the hole about half again larger than
    this. The number to quote is the measured one; this one says why it is where it is.
    """
    per_sigma = 2.0 * G * pipe.area * sensors.sigma / pipe.wave_speed / max(1.0 - attenuation, 1e-9)
    return {"leak_rate": waves * per_sigma, "per_sigma": per_sigma,
            "fraction_of_flow": waves * per_sigma / pipe.flow,
            "head_at_transmitter": waves * sensors.sigma, "attenuation": attenuation,
            "sampling_resolution": pipe.wave_speed * sensors.sample / 2.0}
