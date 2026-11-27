"""Tests for ``pinneapple_physics.embedded_bridge`` -- the flight-stack contract, tested without a flight stack (#350).

There is no PX4 and no ROS 2 on this machine (``ROS2_INSTALLED`` is a measurement, and the adapter that needs the
stack is marked *not covered*), so what is asserted here is the part that can be: the payload, the 69-byte frame, the
four ways a frame is refused, and a publisher/subscriber double that carries the *bytes* -- a bus that hands the
dataclass over would leave the wire format, the only thing a flight stack can break, untested.
"""
from __future__ import annotations

import json

import pytest

from pinneapple_physics.embedded_bridge import (
    FLAG_OFF_PHYSICS,
    FLAG_SENSOR_SUSPECT,
    FLAG_TRUSTED,
    FRAME_BYTES,
    MAGIC,
    MAX_HORIZON,
    ROS2_INSTALLED,
    TOPIC,
    VERSION,
    LoopbackBus,
    Subscription,
    TrustNode,
    TrustReport,
    readings,
    with_sequence,
)

HEALTHY = dict(stamp_us=1_000_000, active_model="ensemble/fno-seed3", trust=0.8, residual=0.25,
               trust_horizon_steps=40, step_us=440, sequence=0)


def test_the_frame_is_fixed_size_and_says_so_in_its_header():
    frame = TrustReport(**HEALTHY).encode()

    assert len(frame) == FRAME_BYTES == 69
    assert int.from_bytes(frame[0:2], "little") == MAGIC
    assert frame[2] == VERSION
    assert frame[3] == 0, "the sequence is the fourth byte, where a MAVLink reader looks for it"


def test_a_report_survives_the_wire_and_the_bus():
    sent = TrustReport(**HEALTHY, off_physics=True, sensor_suspect=True, channel="accelerometer")
    bus = LoopbackBus()
    sub = bus.subscribe(TOPIC)

    bus.publish(TOPIC, sent)
    back = sub.read()

    assert isinstance(sub, Subscription) and len(sub) == 0
    for field in ("stamp_us", "active_model", "trust_horizon_steps", "step_us", "channel", "sequence"):
        assert getattr(back, field) == getattr(sent, field), field
    assert back.trust == pytest.approx(sent.trust, rel=1e-6), "float32 on the wire, float64 in the dataclass"
    assert back.residual == pytest.approx(sent.residual, rel=1e-6)
    assert back.off_physics is True and back.sensor_suspect is True


def test_the_flags_say_what_the_report_says():
    healthy = TrustReport(**HEALTHY)
    sick = TrustReport(**{**HEALTHY, "trust": 0.2}, off_physics=True, sensor_suspect=True, channel="gyro")

    assert healthy.flags & FLAG_TRUSTED and not healthy.flags & FLAG_OFF_PHYSICS
    assert sick.flags & FLAG_OFF_PHYSICS and sick.flags & FLAG_SENSOR_SUSPECT and not sick.flags & FLAG_TRUSTED


def test_the_json_form_is_the_same_report():
    report = TrustReport(**HEALTHY, channel="gyro", sensor_suspect=True)

    parsed = json.loads(report.to_json())

    assert parsed["active_model"] == "ensemble/fno-seed3" and parsed["sensor_suspect"] is True
    assert parsed["channel"] == "gyro"


@pytest.mark.parametrize("changes,reason", [
    ({"trust": 1.5}, "a trust of 1.5 is more than the tolerance"),
    ({"trust": -0.1}, "negative trust"),
    ({"residual": -1.0}, "a negative distance"),
    ({"residual": float("inf")}, "an infinite residual is not a distance"),
    ({"trust_horizon_steps": MAX_HORIZON + 1}, "the field is a uint16"),
    ({"sequence": 256}, "the sequence wraps at 255"),
    ({"stamp_us": -1}, "a clock that ran backwards"),
    ({"active_model": "a-model-name-that-does-not-fit-in-24-bytes"}, "the frame is fixed size"),
    ({"channel": "a-channel-that-is-too-wide"}, "the frame is fixed size"),
    ({"sensor_suspect": True}, "a suspect sensor has to name its channel"),
])
def test_an_impossible_report_is_refused_before_it_is_sent(changes, reason):
    with pytest.raises(ValueError):
        TrustReport(**{**HEALTHY, **changes})


@pytest.mark.parametrize("frame,reason", [
    (b"", "an empty read"),
    (TrustReport(**HEALTHY).encode()[:-1], "a truncated frame"),
    (TrustReport(**HEALTHY).encode() + b"\x00", "a frame with a byte stuck to it"),
    (b"\x00\x00" + TrustReport(**HEALTHY).encode()[2:], "a frame from something else"),
])
def test_a_frame_that_is_not_one_is_refused_and_named(frame, reason):
    with pytest.raises(ValueError):
        TrustReport.decode(frame)


def test_a_corrupt_frame_is_not_a_plausible_frame():
    frame = bytearray(TrustReport(**HEALTHY).encode())
    frame[12] ^= 0xFF                                              # the trust byte
    with pytest.raises(ValueError, match="crc"):
        TrustReport.decode(bytes(frame))

    unknown = bytearray(TrustReport(**HEALTHY).encode())
    unknown[2] = 9                                                 # a version this reader does not speak
    with pytest.raises(ValueError, match="version"):
        TrustReport.decode(bytes(unknown))


def test_a_bus_that_carries_corruption_passes_it_on_to_be_refused():
    """The double does not sanitise: what the wire would deliver is what the subscriber has to refuse."""
    bus = LoopbackBus()
    sub = bus.subscribe(TOPIC)
    frame = bytearray(TrustReport(**HEALTHY).encode())
    frame[20] ^= 0x01

    bus.publish_raw(TOPIC, bytes(frame))

    with pytest.raises(ValueError):
        sub.read()


# --------------------------------------------------------------------------------------------------- the node
class _Output:
    """The four numbers the node reads off one step of the graph, without building a graph."""

    def __init__(self, trust: float, residual: float = 0.0, off_physics: bool = False, suspect: bool = False):
        self.trust, self.residual, self.off_physics, self.suspect = trust, residual, off_physics, suspect


def test_the_horizon_is_the_trend_extrapolated_to_the_threshold():
    bus = LoopbackBus()
    node = TrustNode(bus, active_model="ensemble/ridge", window=8, threshold=0.5)

    assert node.step(_Output(0.9), stamp_us=0).trust_horizon_steps == MAX_HORIZON, "one sample is not a trend"
    second = node.step(_Output(0.85), stamp_us=5000)
    steeper = node.step(_Output(0.70), stamp_us=10000)      # the fall is faster now: fewer steps left

    assert second.trust_horizon_steps == pytest.approx(7, abs=1)
    assert steeper.trust_horizon_steps < second.trust_horizon_steps
    assert node.step(_Output(0.49), stamp_us=15000).trust_horizon_steps == 0, "below the threshold: no horizon left"


def test_the_node_publishes_what_the_stack_reads_back():
    bus = LoopbackBus()
    sub = bus.subscribe(TOPIC)
    node = TrustNode(bus, active_model="ensemble/fno", threshold=0.5)

    node.step(_Output(0.9, residual=0.1), stamp_us=1000, step_us=430)
    published = node.step(_Output(0.2, residual=1.4, off_physics=True, suspect=True), stamp_us=2000, step_us=430)
    assert len(sub) == 2
    back = sub.latest()

    assert back.active_model == "ensemble/fno" and back.step_us == 430
    assert back.sensor_suspect is True and back.channel == "accelerometer"
    assert back.off_physics is True and back.sequence == published.sequence == 1


def test_a_whole_flight_is_one_run_of_the_node():
    bus = LoopbackBus()
    sub = bus.subscribe(TOPIC)
    node = TrustNode(bus, active_model="ensemble/fno")

    reports = node.run([_Output(0.9), _Output(0.8), _Output(0.7)], start_us=0, step_us=5000)

    assert [r.stamp_us for r in reports] == [0, 5000, 10000]
    assert [r.sequence for r in reports] == [0, 1, 2]
    assert len(readings(sub)) == 3 and readings(sub) == []


def test_the_sequence_wraps_like_a_mavlink_one():
    bus = LoopbackBus()
    node = TrustNode(bus)

    report = with_sequence(TrustReport(**HEALTHY), 300)

    assert report.sequence == 300 % 256
    assert node.step(_Output(0.9), stamp_us=0).sequence == 0


def test_the_stack_is_measured_not_assumed():
    assert isinstance(ROS2_INSTALLED, bool)
    assert FRAME_BYTES == 69
