"""The pipeline transient, its two transmitters, and the detector that tells a hole from a broken instrument (#358).

Every tolerance here is a number the module was measured to reach: the leak node's closed form is reproduced to
`1.3e-12`, the intact steady state is preserved to `1e-12` m, the front travels at 1 120-1 135 m/s against the
Korteweg speed of 1 140 m/s, the wave reaches the transmitter 30 % shorter than it left the hole, and the detector
places a 10 mm hole within 5-60 m of where it was drilled.
"""
import math

import numpy as np
import pytest

from pinneapple_physics.pipeline_leak import (
    Fault,
    Leak,
    Pipeline,
    Sensors,
    detect,
    expected,
    friction_factor,
    measurement_check,
    minimum_detectable_leak,
    observe,
    simulate,
    wave_speed,
)

PIPE = Pipeline(length=20000.0, diameter=0.4)
SENSORS = Sensors(upstream=480.0, downstream=480.0)
MONITOR = (SENSORS.upstream, PIPE.length - SENSORS.downstream)
REACHES = 500
HOLE_10_MM = math.pi * (10e-3) ** 2 / 4.0


def hole(diameter_mm: float, distance: float, t_open: float = 20.0) -> Leak:
    return Leak(distance=distance, area=math.pi * (diameter_mm * 1e-3) ** 2 / 4.0, t_open=t_open)


@pytest.fixture(scope="module")
def nominal():
    return simulate(PIPE, n_reaches=REACHES, t_end=120.0, monitor=MONITOR)


@pytest.fixture(scope="module")
def leaky():
    return simulate(PIPE, n_reaches=REACHES, t_end=200.0, monitor=MONITOR, leak=hole(10.0, 9800.0))


# --------------------------------------------------------------------------------------------- the transient
def test_the_line_properties_are_the_ones_the_example_quotes():
    assert PIPE.wave_speed == pytest.approx(1140.31, abs=0.01)
    assert PIPE.friction == pytest.approx(0.020021, abs=1e-5)
    assert PIPE.friction_head == pytest.approx(202.01, abs=0.01)
    assert PIPE.travel_time == pytest.approx(17.539, abs=1e-3)
    assert PIPE.reynolds == pytest.approx(67641.0, abs=1.0)


def test_korteweg_and_colebrook_have_their_textbook_limits():
    # a rigid pipe: the wave speed is the liquid's own, sqrt(K/rho)
    assert wave_speed(1.5e9, 850.0, 0.4, 0.008, youngs=1e30) == pytest.approx(math.sqrt(1.5e9 / 850.0), rel=1e-12)
    assert friction_factor(1000.0, 4.5e-5, 0.4) == pytest.approx(0.064)          # laminar, 64/Re
    with pytest.raises(ValueError):
        friction_factor(0.0, 4.5e-5, 0.4)


def test_the_intact_line_stays_where_it_started(nominal):
    """No leak and no manoeuvre: the transient must be the steady state, to round-off, for 120 s."""
    steady = nominal["steady"][[int(round(m / nominal["dx"])) for m in MONITOR]]
    assert np.max(np.abs(nominal["monitor_head"] - steady)) < 1e-9
    assert np.max(np.abs(nominal["flow_in"] - PIPE.flow)) < 1e-12


def test_the_first_step_is_the_leak_node_closed_form(leaky):
    """The node is solved, not iterated: the head there falls by ``a Q_L/(2 g A)`` and each side takes ``Q_L/2``."""
    opened = int(np.argmax(leaky["t"] > 20.0))
    node = int(round(9800.0 / leaky["dx"]))
    leak_at_hole = leaky["leak_flow"][opened]
    drop = leaky["monitor_head"][opened] - leaky["monitor_head"][0]
    assert drop[0] == pytest.approx(0.0, abs=1e-12)          # the transmitters have not been reached yet
    field = simulate(PIPE, n_reaches=REACHES, t_end=40.0, leak=hole(10.0, 9800.0), record=True)
    at_node = field["field"][opened][node] - field["steady"][node]
    closed = -PIPE.wave_speed * leak_at_hole / (2.0 * 9.80665 * PIPE.area)
    assert at_node == pytest.approx(closed, rel=1e-10)
    split_up, split_down = (float(v) for v in field["leak_split"][opened])
    assert split_up == pytest.approx(leak_at_hole / 2.0, rel=1e-10)
    assert split_down == pytest.approx(-leak_at_hole / 2.0, rel=1e-10)


def test_the_front_leaves_the_hole_at_the_wave_speed():
    """The disturbance spreads at ``a``: over the first eight seconds its width is ``2 a t`` to within the grid."""
    run = simulate(PIPE, n_reaches=REACHES, t_end=40.0, leak=hole(10.0, 9800.0), record=True)
    closed = -PIPE.wave_speed * run["leak_flow"][int(np.argmax(run["t"] > 20.0))] / (2.0 * 9.80665 * PIPE.area)
    speeds = []
    for elapsed in (2.0, 4.0, 6.0, 8.0):
        k = int(np.argmin(np.abs(run["t"] - (20.0 + elapsed))))
        moved = np.abs(run["field"][k] - run["steady"]) > 0.02 * abs(closed)
        speeds.append((run["x"][moved].max() - run["x"][moved].min()) / (2.0 * elapsed))
    assert min(speeds) > 0.98 * PIPE.wave_speed and max(speeds) < PIPE.wave_speed * 1.001


def test_without_friction_the_front_keeps_its_height():
    """The check that the friction term is what shrinks the wave and not an indexing slip: with ``f = 0`` the
    transmitter sees exactly the head the hole dropped -- 1.6038 m for this hole, to round-off."""

    class Smooth(Pipeline):
        @property
        def friction(self) -> float:
            return 0.0

    pipe = Smooth(length=20000.0, diameter=0.4, head_upstream=260.0, head_downstream=20.0, pump_resistance=0.0)
    run = simulate(pipe, n_reaches=REACHES, t_end=40.0, leak=hole(10.0, 9800.0), monitor=MONITOR)
    opened = int(np.argmax(run["t"] > 20.0))
    closed = -pipe.wave_speed * run["leak_flow"][opened] / (2.0 * 9.80665 * pipe.area)
    step = run["monitor_head"][:, 0] - run["monitor_head"][0, 0]
    assert float(np.min(step)) == pytest.approx(closed, rel=1e-9)


def test_the_wave_loses_a_third_of_its_height_over_the_line(leaky):
    """Steady friction still costs: 1.261 m at the hole is 0.838 m at a transmitter 9.3 km away."""
    opened = int(np.argmax(leaky["t"] > 20.0))
    closed = PIPE.wave_speed * float(leaky["leak_flow"][opened]) / (2.0 * 9.80665 * PIPE.area)
    attenuation = 1.0 - abs(float(np.min(leaky["monitor_head"][:, 0] - leaky["monitor_head"][0, 0]))) / closed
    assert 0.25 < attenuation < 0.35


def test_the_leak_node_refuses_to_sit_on_a_boundary():
    with pytest.raises(ValueError):
        simulate(PIPE, n_reaches=REACHES, t_end=30.0, leak=hole(10.0, 10.0))
    with pytest.raises(ValueError):
        Pipeline(length=20000.0, diameter=0.4, head_upstream=200.0, head_downstream=20.0)


# --------------------------------------------------------------------------------------------- the transmitters
def test_the_acquisition_quantises_and_is_noisy(nominal):
    reading = observe(nominal, PIPE, SENSORS, seed=1)
    steps = reading["head_a"] / SENSORS.step
    assert np.max(np.abs(steps - np.round(steps))) < 1e-9                # every sample is on the quantisation grid
    assert reading["head_a"].std() == pytest.approx(SENSORS.sigma, rel=0.06)      # quantisation adds to it
    assert reading["sample"] == pytest.approx(SENSORS.sample)
    assert reading["x_a"] == pytest.approx(480.0) and reading["x_b"] == pytest.approx(19520.0)


def test_a_fault_starts_when_it_says_it_does(nominal):
    """Before ``t_start`` the channel is untouched: the fault is on the instrument, not on the record."""
    faults = (Fault("offset", "a", 40.0, 5.0), Fault("noise", "b", 40.0, 6.0))
    reading = observe(nominal, PIPE, SENSORS, seed=4, faults=faults)
    early = reading["t"] < 40.0
    steady = nominal["monitor_head"][0]
    assert reading["head_a"][early].mean() == pytest.approx(steady[0], abs=0.05)
    assert reading["head_a"][early].std() == pytest.approx(SENSORS.sigma, rel=0.1)
    assert reading["head_a"][~early].mean() - reading["head_a"][early].mean() == pytest.approx(5.0, abs=0.05)
    assert reading["head_b"][~early].std() > 3.0 * reading["head_b"][early].std()
    latched = observe(nominal, PIPE, SENSORS, seed=4, faults=(Fault("stuck", "a", 40.0, 0.0),))
    assert np.all(np.diff(latched["head_a"][latched["t"] > 41.0]) == 0.0)


def test_the_fault_names_are_checked():
    with pytest.raises(ValueError):
        Fault("melted", "a", 1.0, 1.0)
    with pytest.raises(ValueError):
        Fault("offset", "c", 1.0, 1.0)


# --------------------------------------------------------------------------------------------- the detector
@pytest.fixture(scope="module")
def long_nominal():
    return simulate(PIPE, n_reaches=REACHES, t_end=300.0, monitor=MONITOR)


@pytest.fixture(scope="module")
def reference(long_nominal):
    return expected(long_nominal, SENSORS)


def _verdict(run, reference, seed, **kwargs):
    return detect(observe(run, PIPE, SENSORS, seed=seed), reference, PIPE, SENSORS, **kwargs)


def test_a_hole_in_the_middle_is_a_leak_in_the_flight_time(leaky, reference):
    """8.2 s is the time the wave needs to reach a transmitter, not the detector's reaction: the two must agree."""
    run = simulate(PIPE, n_reaches=REACHES, t_end=300.0, monitor=MONITOR, leak=hole(10.0, 9800.0))
    verdict = _verdict(run, reference, seed=2)
    assert verdict["source"] == "leak"
    assert verdict["alarm_time"] - 20.0 == pytest.approx(8.17, abs=0.3)
    assert abs(verdict["leak_distance"] - 9800.0) < 150.0
    assert verdict["leak_rate"] == pytest.approx(float(run["leak_flow"][-1]), rel=0.05)


@pytest.mark.parametrize("distance", [3000.0, 15500.0])
def test_a_hole_near_an_end_is_late_by_less_and_placed_where_it_is(distance, reference):
    run = simulate(PIPE, n_reaches=REACHES, t_end=300.0, monitor=MONITOR, leak=hole(10.0, distance))
    verdict = _verdict(run, reference, seed=3)
    floor = min(distance - MONITOR[0], MONITOR[1] - distance) / PIPE.wave_speed
    assert verdict["source"] == "leak"
    assert verdict["alarm_time"] - 20.0 == pytest.approx(floor, abs=0.5)
    assert abs(verdict["leak_distance"] - distance) < 150.0


def test_a_hole_outside_the_transmitters_cannot_be_placed(reference):
    """Both fronts go past both transmitters, so their difference says nothing: the guess collapses onto the nearer
    one and the verdict admits it instead of inventing a position."""
    run = simulate(PIPE, n_reaches=REACHES, t_end=300.0, monitor=MONITOR, leak=hole(10.0, 300.0))
    verdict = _verdict(run, reference, seed=2)
    assert verdict["source"] == "leak"
    assert verdict["evidence"]["outside_transmitters"] is True
    assert verdict["leak_distance"] == pytest.approx(MONITOR[0])


def test_the_smallest_hole_the_wave_finds_is_the_transducer_noise(reference):
    """A 3 mm hole is under the noise and is not heard; a 10 mm one is. The gap between the two is the amplitude
    argument, and the module says so rather than claiming a smarter detector."""
    small = simulate(PIPE, n_reaches=REACHES, t_end=300.0, monitor=MONITOR, leak=hole(3.0, 9800.0))
    listening = [_verdict(small, reference, seed=s)["source"] for s in range(4)]
    assert listening.count("leak") <= 1
    floor = minimum_detectable_leak(PIPE, SENSORS, attenuation=0.305)
    assert floor["leak_rate"] == pytest.approx(5.596e-4, rel=0.02)
    assert 0.2 < floor["fraction_of_flow"] * 100 < 0.25


def test_an_instrument_fault_is_not_a_leak(nominal, reference):
    """Six ways for a transmitter to lie on a record of normal operation, and not one of them is a hole."""
    cases = {"bias up": (Fault("offset", "a", 40.0, 3.0),), "bias down": (Fault("offset", "a", 40.0, -3.0),),
             "drift": (Fault("drift", "a", 40.0, -0.02),), "noise": (Fault("noise", "a", 40.0, 8.0),),
             "latched A": (Fault("stuck", "a", 40.0, 0.0),), "latched B": (Fault("stuck", "b", 40.0, 0.0),)}
    quiet = simulate(PIPE, n_reaches=REACHES, t_end=140.0, monitor=MONITOR)
    quiet_reference = expected(quiet, SENSORS)
    for name, faults in cases.items():
        for seed in range(3):
            verdict = detect(observe(quiet, PIPE, SENSORS, seed=700 + seed, faults=faults), quiet_reference,
                             PIPE, SENSORS)
            assert verdict["source"] == "instrument", name
            assert verdict["alarm"] is True, name
    assert measurement_check(observe(quiet, PIPE, SENSORS, seed=700, faults=cases["drift"]), quiet_reference,
                             PIPE, SENSORS) is True


def test_a_hole_is_not_an_instrument_fault(leaky, reference):
    run = simulate(PIPE, n_reaches=REACHES, t_end=300.0, monitor=MONITOR, leak=hole(10.0, 9800.0))
    assert measurement_check(observe(run, PIPE, SENSORS, seed=2), reference, PIPE, SENSORS) is False


def test_normal_operation_raises_nothing(long_nominal, reference):
    sources = [_verdict(long_nominal, reference, seed=3000 + s)["source"] for s in range(10)]
    assert sources == ["none"] * 10


def test_an_unrecorded_closure_is_called_an_operation():
    """A valve the model was not told about moves both ends *up*, which no hole can do; the verdict says so."""
    def closing(t):
        if 40.0 <= t <= 45.0:
            return 1.0 - 0.10 * (t - 40.0) / 5.0
        return 0.90 if t > 45.0 else 1.0

    run = simulate(PIPE, n_reaches=REACHES, t_end=200.0, monitor=MONITOR, valve=closing)
    blind = expected(simulate(PIPE, n_reaches=REACHES, t_end=200.0, monitor=MONITOR), SENSORS)
    sources = [detect(observe(run, PIPE, SENSORS, seed=800 + s), blind, PIPE, SENSORS)["source"] for s in range(5)]
    assert sources == ["operation"] * 5


def test_the_balance_finds_a_leak_the_wave_has_lost(reference):
    """A 4 mm hole at 3 km: one transmitter hears it, the other does not, so there is a leak and no position. The
    mass balance is what makes the difference, and it does not claim a place it does not have."""
    run = simulate(PIPE, n_reaches=REACHES, t_end=300.0, monitor=MONITOR, leak=hole(4.0, 3000.0))
    verdict = _verdict(run, reference, seed=5)
    assert verdict["source"] == "leak"
    assert verdict["leak_distance"] is None
    assert verdict["leak_rate"] == pytest.approx(float(run["leak_flow"][-1]), rel=0.25)


def test_the_threshold_is_what_the_false_alarms_are_bought_with(long_nominal, reference):
    """A decision interval of 8 sigma alarms on noise; 12 does not. That is the trade the example plots."""
    eight = [_verdict(long_nominal, reference, seed=3000 + s, threshold=8.0)["source"] for s in range(20)]
    twelve = [_verdict(long_nominal, reference, seed=3000 + s, threshold=12.0)["source"] for s in range(20)]
    assert eight.count("none") < 20
    assert twelve == ["none"] * 20


def test_detect_refuses_a_reference_without_the_transmitters():
    run = simulate(PIPE, n_reaches=REACHES, t_end=30.0)
    with pytest.raises(ValueError):
        expected(run, SENSORS)
    with pytest.raises(ValueError):
        observe(run, PIPE, SENSORS)
