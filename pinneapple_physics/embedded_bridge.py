"""The bridge to a flight stack, as a contract that can be tested without one (#350).

The node a vehicle would run is small and boring: one step of the embedded graph in, one report out, published on a
topic the rest of the stack reads. What is *not* boring is that this is the piece which cannot be checked on a
laptop -- there is no PX4, no ROS 2 and no board here (:data:`ROS2_INSTALLED` is measured, not assumed) -- so this
module ships the part that survives that: the payload, the wire format, the validation, and a test double with the
shape of a uORB publisher. The adapter to ``rclpy`` is two functions at the end, marked as untested, and they are the
only lines of the module that need the stack installed.

The payload
-----------
:class:`TrustReport` is what the flight stack needs to know and nothing else:

* ``active_model`` -- which model is flying. An ensemble switches experts; a stack that acted on the output of
  expert A must be able to see that expert B took over, so the name travels with every report;
* ``trust`` and ``trust_horizon_steps`` -- the confidence now, and how many control steps of the current trend it
  has left before it falls under the threshold. A horizon is the actionable form: "trust is 0.8" says nothing about
  when to hand over, ``trust_horizon_steps = 40`` says 200 ms at 200 Hz;
* ``sensor_suspect`` and ``channel`` -- which sensor is disagreeing, so the stack can stop trusting a reading rather
  than the whole estimate;
* ``off_physics`` -- the vehicle disagreeing with its own dynamics, which is a different action from a bad sensor.

The wire format
---------------
:meth:`TrustReport.encode` writes the fixed-size little-endian frame below -- fixed size and no strings of variable
length, because that is what a uORB message and a MAVLink payload both are, and because a fixed frame is what lets a
receiver validate a truncated read instead of interpreting it::

    offset  size  field
    0       2     magic 0x5042 ("PB")
    2       1     version (1)
    3       1     sequence, wrapping 0-255 like a MAVLink seq
    4       8     stamp_us, microseconds since boot (hrt_absolute_time)
    12      4     trust, float32 in [0, 1]
    16      4     residual, float32, >= 0
    20      2     trust_horizon_steps, uint16 (capped at 65535, which is where "no trend" lands)
    22      1     flags: bit0 off_physics, bit1 sensor_suspect, bit2 trusted
    23      24    active_model, ASCII, NUL padded
    47      16    channel, ASCII, NUL padded -- which sensor is suspect
    63      4     step_us, uint32 -- the dt the graph was called with
    67      2     crc16 (CCITT-FALSE over bytes 0..66)

69 bytes, and :meth:`TrustReport.decode` refuses anything else: a wrong magic, a wrong version, a wrong length and a
wrong CRC are four different errors, because a stack that receives a corrupt frame must not read it as a trusted one.

    >>> from pinneapple_physics.embedded_bridge import TrustReport, LoopbackBus, TOPIC
    >>> bus = LoopbackBus()
    >>> sub = bus.subscribe(TOPIC)
    >>> _ = bus.publish(TOPIC, TrustReport(stamp_us=1, active_model="ensemble/ridge", trust=0.9, residual=0.01,
    ...                                    trust_horizon_steps=40, step_us=5000))
    >>> report = sub.read()
    >>> (report.active_model, round(report.trust, 3), report.trust_horizon_steps)
    ('ensemble/ridge', 0.9, 40)
"""
from __future__ import annotations

import binascii
import json
import struct
from collections import deque
from dataclasses import dataclass, replace
from typing import Any, Iterable

__all__ = ["TrustReport", "FRAME", "FRAME_BYTES", "MAGIC", "VERSION", "TOPIC", "FLAG_OFF_PHYSICS",
           "FLAG_SENSOR_SUSPECT", "FLAG_TRUSTED", "MAX_HORIZON", "MODEL_CHARS", "CHANNEL_CHARS", "LoopbackBus",
           "Subscription", "TrustNode", "ROS2_INSTALLED", "ros2_publisher", "publish_ros2", "with_sequence",
           "readings"]

MAGIC = 0x5042                      # "PB" -- what a receiver checks before it believes a byte
VERSION = 1
#: The uORB-style topic the node publishes on: ``/fmu/out/vehicle_trust`` on a PX4 build would be the same name.
TOPIC = "/vehicle_trust"
FLAG_OFF_PHYSICS = 1 << 0
FLAG_SENSOR_SUSPECT = 1 << 1
FLAG_TRUSTED = 1 << 2
MAX_HORIZON = 65535                 # uint16: where "the trend does not cross the threshold inside the window" lands
MODEL_CHARS = 24                    # a uORB char[24]: long enough for "ensemble/fno-seed3", fixed by the format
CHANNEL_CHARS = 16                  # and a char[16] for the sensor channel: "accelerometer", "magnetometer"

#: magic, version, sequence, stamp_us, trust, residual, horizon, flags, model, channel, step_us -- then the CRC
_BODY = struct.Struct(f"<HBBQffHB{MODEL_CHARS}s{CHANNEL_CHARS}sI")
_CRC = struct.Struct("<H")
FRAME = struct.Struct(f"<HBBQffHB{MODEL_CHARS}s{CHANNEL_CHARS}sIH")
FRAME_BYTES = FRAME.size

#: measured, not assumed: the adapter at the bottom of this module is untested exactly when this is False
try:                                                        # pragma: no cover - the import is the measurement
    import rclpy  # noqa: F401
    ROS2_INSTALLED = True
except Exception:                                           # noqa: BLE001 - any failure means "no stack here"
    ROS2_INSTALLED = False


@dataclass(frozen=True)
class TrustReport:
    """What the flight stack is told, once per control step."""

    stamp_us: int                                   # microseconds since boot
    active_model: str = ""                          # which expert is flying
    trust: float = 0.0                              # 1 - residual/limit, clipped to [0, 1]
    residual: float = 0.0                           # the normalised residual the graph returned (1.0 = at the limit)
    trust_horizon_steps: int = 0                    # control steps before trust crosses the threshold at this trend
    step_us: int = 0                                # how long one step of the graph took on this machine
    off_physics: bool = False
    sensor_suspect: bool = False
    channel: str = ""                               # the sensor channel that is suspect, "" when none is
    sequence: int = 0

    def __post_init__(self) -> None:
        if not 0.0 <= float(self.trust) <= 1.0:
            raise ValueError(f"trust is {self.trust!r}: it is a fraction of the tolerance, so it lives in [0, 1]")
        if not (float(self.residual) >= 0.0 and float(self.residual) != float("inf")):
            raise ValueError(f"residual is {self.residual!r}: it is a normalised distance, so it is finite and >= 0")
        if not 0 <= int(self.trust_horizon_steps) <= MAX_HORIZON:
            raise ValueError(f"trust_horizon_steps is {self.trust_horizon_steps!r}: a uint16 holds 0..{MAX_HORIZON}")
        if not 0 <= int(self.sequence) <= 255:
            raise ValueError(f"sequence is {self.sequence!r}: it wraps at 255, like a MAVLink seq")
        if int(self.stamp_us) < 0 or int(self.step_us) < 0:
            raise ValueError("stamp_us and step_us are unsigned: a negative one is a clock that ran backwards")
        if len(self.active_model.encode("ascii", "strict")) > MODEL_CHARS:
            raise ValueError(f"active_model is longer than the {MODEL_CHARS} bytes the frame holds")
        if len(self.channel.encode("ascii", "strict")) > CHANNEL_CHARS:
            raise ValueError(f"channel is longer than the {CHANNEL_CHARS} bytes the frame holds")
        if self.sensor_suspect and not self.channel:
            raise ValueError("a suspect sensor has to name the channel it was suspected on")

    @property
    def flags(self) -> int:
        return ((FLAG_OFF_PHYSICS if self.off_physics else 0) | (FLAG_SENSOR_SUSPECT if self.sensor_suspect else 0)
                | (FLAG_TRUSTED if self.trust >= 0.5 and not self.sensor_suspect else 0))

    def encode(self) -> bytes:
        """The 69-byte frame: fixed layout, NUL-padded model and channel, CRC last."""
        body = _BODY.pack(MAGIC, VERSION, int(self.sequence), int(self.stamp_us), float(self.trust),
                          float(self.residual), int(self.trust_horizon_steps), self.flags,
                          self.active_model.encode("ascii"), self.channel.encode("ascii"), int(self.step_us))
        return body + _CRC.pack(binascii.crc_hqx(body, 0xFFFF))

    @classmethod
    def decode(cls, frame: bytes) -> TrustReport:
        """Read a frame, or say which of the four things is wrong with it -- never a plausible-looking report."""
        if len(frame) != FRAME_BYTES:
            raise ValueError(f"a frame is {FRAME_BYTES} bytes; {len(frame)} arrived, so it was truncated or joined")
        (magic, version, sequence, stamp_us, trust, residual, horizon, flags, model, channel,
         step_us, crc) = FRAME.unpack(frame)
        if magic != MAGIC:
            raise ValueError(f"bad magic 0x{magic:04x}, not 0x{MAGIC:04x}: this is not a trust frame")
        if version != VERSION:
            raise ValueError(f"frame version {version}, this reader speaks {VERSION}")
        expected = binascii.crc_hqx(_BODY.pack(MAGIC, version, sequence, stamp_us, trust, residual, horizon, flags,
                                               model, channel, step_us), 0xFFFF)
        if crc != expected:
            raise ValueError(f"crc16 0x{crc:04x} does not match the 0x{expected:04x} the bytes give: corrupt frame")
        return cls(stamp_us=int(stamp_us), active_model=model.split(b"\0")[0].decode("ascii"),
                   trust=float(trust), residual=float(residual), trust_horizon_steps=int(horizon),
                   step_us=int(step_us), off_physics=bool(flags & FLAG_OFF_PHYSICS),
                   sensor_suspect=bool(flags & FLAG_SENSOR_SUSPECT),
                   channel=channel.split(b"\0")[0].decode("ascii"), sequence=int(sequence))

    def to_json(self) -> str:
        """The same report as JSON, for a log or a ROS 2 topic that carries strings instead of a struct."""
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_dict(self) -> dict[str, Any]:
        return {"stamp_us": int(self.stamp_us), "active_model": self.active_model, "trust": float(self.trust),
                "residual": float(self.residual), "trust_horizon_steps": int(self.trust_horizon_steps),
                "step_us": int(self.step_us), "off_physics": bool(self.off_physics),
                "sensor_suspect": bool(self.sensor_suspect), "channel": self.channel,
                "sequence": int(self.sequence)}


# ----------------------------------------------------------------------------------------------- the test double
class Subscription:
    """A listener with the shape of a uORB subscription: whatever is published is *encoded and decoded* to reach it.

    The double carries the bytes on purpose. A bus that hands the object over would test the dataclass and leave the
    wire format -- the only part a flight stack can break -- untested.
    """

    def __init__(self, topic: str):
        self.topic = topic
        self.frames: deque[bytes] = deque()

    def __len__(self) -> int:
        return len(self.frames)

    def read(self) -> TrustReport:
        if not self.frames:
            raise IndexError(f"nothing published on {self.topic} yet")
        return TrustReport.decode(self.frames.popleft())

    def latest(self) -> TrustReport:
        if not self.frames:
            raise IndexError(f"nothing published on {self.topic} yet")
        return TrustReport.decode(self.frames[-1])

    def drain(self) -> list[bytes]:
        frames = list(self.frames)
        self.frames.clear()
        return frames


class LoopbackBus:
    """In-process publish/subscribe, the stand-in for ``rclpy`` and for uORB: same contract, one thread, no stack."""

    def __init__(self) -> None:
        self._subs: dict[str, list[Subscription]] = {}
        self.published: dict[str, bytes] = {}

    def subscribe(self, topic: str = TOPIC) -> Subscription:
        sub = Subscription(topic)
        self._subs.setdefault(topic, []).append(sub)
        return sub

    def publish(self, topic: str, report: TrustReport) -> bytes:
        return self.publish_raw(topic, report.encode())

    def publish_raw(self, topic: str, frame: bytes) -> bytes:
        """Publish bytes that did not come from :meth:`TrustReport.encode`, which is how a corrupt frame is tested."""
        self.published[topic] = frame
        for sub in self._subs.get(topic, ()):
            sub.frames.append(frame)
        return frame


class TrustNode:
    """The node: it turns one step of the embedded graph into one report, and publishes it.

    The trust horizon is a *measurement of a trend*, not a promise: the last ``window`` trust values are fitted with a
    least-squares line and the crossing of the threshold is extrapolated from its slope. A trend that is flat or
    rising does not cross anything, and reports :data:`MAX_HORIZON` -- the largest number the frame can carry, so a
    reader can tell "far away" from "unknown". A trend that is already below the threshold reports zero.
    """

    def __init__(self, bus: LoopbackBus, *, active_model: str = "", topic: str = TOPIC, threshold: float = 0.5,
                 window: int = 8):
        if window < 2:
            raise ValueError("a trend needs two samples: window must be at least 2")
        self.bus = bus
        self.active_model = active_model
        self.topic = topic
        self.threshold = float(threshold)
        self.window = int(window)
        self._trust: deque[float] = deque(maxlen=self.window)
        self._sequence = 0

    def horizon(self, trust: float) -> int:
        """Control steps until ``trust`` reaches the threshold at the trend of the last ``window`` samples."""
        self._trust.append(float(trust))
        if trust <= self.threshold:
            return 0
        if len(self._trust) < 2:
            return MAX_HORIZON
        n = len(self._trust)
        xs = list(range(n))
        mean_x, mean_y = sum(xs) / n, sum(self._trust) / n
        denominator = sum((x - mean_x) ** 2 for x in xs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, self._trust, strict=True)) / denominator
        if slope >= 0.0:
            return MAX_HORIZON
        return int(min(MAX_HORIZON, max(0.0, (trust - self.threshold) / -slope)))

    def step(self, output: Any, *, stamp_us: int, step_us: int = 0, channel: str = "") -> TrustReport:
        """One graph step (anything with ``trust``, ``residual``, ``off_physics`` and ``suspect``) in, one frame out."""
        report = TrustReport(stamp_us=int(stamp_us), active_model=self.active_model,
                             trust=float(output.trust), residual=float(output.residual),
                             trust_horizon_steps=self.horizon(float(output.trust)), step_us=int(step_us),
                             off_physics=bool(output.off_physics), sensor_suspect=bool(output.suspect),
                             channel=channel or ("accelerometer" if output.suspect else ""),
                             sequence=self._sequence)
        self._sequence = (self._sequence + 1) % 256
        self.bus.publish(self.topic, report)
        return report

    def run(self, outputs: Iterable[Any], *, start_us: int = 0, step_us: int = 0) -> list[TrustReport]:
        """A whole flight's worth: one ``stamp_us`` per step, counting from ``start_us`` at ``step_us`` each."""
        return [self.step(o, stamp_us=start_us + k * step_us, step_us=step_us) for k, o in enumerate(outputs)]


# --------------------------------------------------------------------------------------------- the untested lines
def ros2_publisher(node: Any = None, *, topic: str = TOPIC, frame: bool = True):       # pragma: no cover
    """The ROS 2 adapter: a publisher for this module's frames. **Not tested: no ROS 2 on this machine.**

    Two lines of translation and nothing else -- ``create_publisher`` with a ``std_msgs/msg/ByteMultiArray`` (or a
    ``String`` for the JSON form) and ``publish`` with the bytes :meth:`TrustReport.encode` already produced. It is
    left thin on purpose: what this module proves is the contract, and a contract that needs the stack installed to
    be checkable is a contract nobody checks in CI. On a Jetson or a Pi with ``ros-humble-rclpy`` this is the whole
    of the work that is *not* in this file.

    ``frame=True`` publishes the 69-byte frame (what a PX4/uORB side reads); ``frame=False`` publishes JSON.
    """
    import rclpy
    from std_msgs.msg import ByteMultiArray, String

    node = node or rclpy.create_node("pinneapple_trust")
    if frame:
        return node.create_publisher(ByteMultiArray, topic, 10)
    return node.create_publisher(String, topic, 10)


def publish_ros2(publisher: Any, report: TrustReport, *, frame: bool = True) -> None:  # pragma: no cover
    """Push one report through the publisher :func:`ros2_publisher` made. **Not tested: no ROS 2 here.**"""
    if frame:
        from std_msgs.msg import ByteMultiArray
        publisher.publish(ByteMultiArray(data=list(report.encode())))
    else:
        from std_msgs.msg import String
        publisher.publish(String(data=report.to_json()))


def with_sequence(report: TrustReport, sequence: int) -> TrustReport:
    """A copy with the wrap-around counter set: what a publisher increments between frames."""
    return replace(report, sequence=int(sequence) % 256)


def readings(sub: Subscription) -> list[dict[str, Any]]:
    """Everything a subscriber has, as dictionaries -- the convenience a test double can offer and a wire cannot."""
    return [TrustReport.decode(f).to_dict() for f in sub.drain()]
