"""A small embedded MQTT 3.1.1 broker for demos, tests and single-asset edge deployments.

Implements the part of MQTT 3.1.1 (OASIS standard, 2014) a sensor gateway and a twin need: CONNECT/CONNACK,
PUBLISH with QoS 0 and 1 (PUBACK), retained messages, SUBSCRIBE/SUBACK and UNSUBSCRIBE/UNSUBACK with ``+`` and
``#`` wildcards, PINGREQ/PINGRESP and DISCONNECT. Messages are delivered to subscribers at QoS 0. There is no
persistence, no QoS 2, no will messages and no TLS: put a production broker (Mosquitto, EMQX, HiveMQ) in front of
real equipment and point :class:`~pinneapple_systems.digital_twin.io.stream.MQTTStream` at it instead.

Optional username/password: when both are set, CONNECT packets with other credentials get return code 4.

>>> broker = MQTTBroker(port=0).start()      # port 0: any free port
>>> broker.port > 0
True
>>> broker.stop()
"""
from __future__ import annotations

import asyncio
import logging
import struct
import threading

logger = logging.getLogger(__name__)

CONNECT, CONNACK, PUBLISH, PUBACK, SUBSCRIBE, SUBACK = 1, 2, 3, 4, 8, 9
UNSUBSCRIBE, UNSUBACK, PINGREQ, PINGRESP, DISCONNECT = 10, 11, 12, 13, 14


def topic_matches(filt: str, topic: str) -> bool:
    """MQTT topic-filter matching (``+`` one level, ``#`` the rest; ``$`` topics need an explicit first level)."""
    f, t = filt.split("/"), topic.split("/")
    if topic.startswith("$") and f[0] in ("+", "#"):
        return False
    for i, part in enumerate(f):
        if part == "#":
            return True
        if i >= len(t):
            return False
        if part != "+" and part != t[i]:
            return False
    return len(f) == len(t)


def _encode_length(n: int) -> bytes:
    out = bytearray()
    while True:
        b, n = n % 128, n // 128
        out.append(b | (0x80 if n else 0))
        if not n:
            return bytes(out)


def _packet(ptype: int, flags: int, body: bytes) -> bytes:
    return bytes([(ptype << 4) | flags]) + _encode_length(len(body)) + body


def _utf8(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("!H", len(b)) + b


class _Client:
    def __init__(self, writer: asyncio.StreamWriter):
        self.writer = writer
        self.client_id = ""
        self.subscriptions: set[str] = set()


class MQTTBroker:
    """Run with :meth:`start` (background thread with its own event loop) and :meth:`stop`."""

    def __init__(self, host: str = "127.0.0.1", port: int = 1883, username: str | None = None,
                 password: str | None = None, max_packet: int = 256 * 1024):
        self.host, self.port = host, int(port)
        self.username, self.password = username, password
        self.max_packet = int(max_packet)
        self.clients: list[_Client] = []
        self.retained: dict[str, bytes] = {}
        self.n_published = 0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._server: asyncio.AbstractServer | None = None
        self._thread: threading.Thread | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None

    # ------------------------------------------------------------------ lifecycle
    def start(self, timeout: float = 5.0) -> MQTTBroker:
        self._thread = threading.Thread(target=self._run, name="mqtt-broker", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout):
            raise RuntimeError("MQTT broker did not start")
        if self._error is not None:
            raise RuntimeError(f"MQTT broker failed to start: {self._error}") from self._error
        return self

    def stop(self) -> None:
        loop = self._loop
        if loop is None:
            return
        for c in list(self.clients):
            loop.call_soon_threadsafe(c.writer.close)
        if self._server is not None:
            loop.call_soon_threadsafe(self._server.close)
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5.0)
        self._loop = None

    @property
    def running(self) -> bool:
        return self._loop is not None and self._ready.is_set() and self._error is None

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            self._server = loop.run_until_complete(asyncio.start_server(self._handle, self.host, self.port))
            self.port = self._server.sockets[0].getsockname()[1]
        except OSError as e:
            self._error = e
            self._ready.set()
            loop.close()
            self._loop = None
            return
        self._ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    # ------------------------------------------------------------------ protocol
    async def _read_packet(self, reader: asyncio.StreamReader) -> tuple[int, int, bytes]:
        head = await reader.readexactly(1)
        mult, length = 1, 0
        for _ in range(4):
            b = (await reader.readexactly(1))[0]
            length += (b & 0x7F) * mult
            if not b & 0x80:
                break
            mult *= 128
        else:
            raise ValueError("malformed remaining length")
        if length > self.max_packet:
            raise ValueError(f"packet of {length} bytes exceeds {self.max_packet}")
        body = await reader.readexactly(length) if length else b""
        return head[0] >> 4, head[0] & 0x0F, body

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        client = _Client(writer)
        try:
            ptype, _, body = await asyncio.wait_for(self._read_packet(reader), timeout=10.0)
            if ptype != CONNECT or not self._accept_connect(client, body):
                return
            self.clients.append(client)
            while True:
                ptype, flags, body = await self._read_packet(reader)
                if ptype == PUBLISH:
                    await self._on_publish(client, flags, body)
                elif ptype == SUBSCRIBE:
                    await self._on_subscribe(client, body)
                elif ptype == UNSUBSCRIBE:
                    pid, pos = struct.unpack("!H", body[:2])[0], 2
                    while pos < len(body):
                        n = struct.unpack("!H", body[pos:pos + 2])[0]
                        client.subscriptions.discard(body[pos + 2:pos + 2 + n].decode("utf-8"))
                        pos += 2 + n
                    writer.write(_packet(UNSUBACK, 0, struct.pack("!H", pid)))
                elif ptype == PINGREQ:
                    writer.write(_packet(PINGRESP, 0, b""))
                elif ptype == DISCONNECT:
                    return
                await writer.drain()
        except (asyncio.IncompleteReadError, ConnectionError, asyncio.TimeoutError):
            pass
        except (ValueError, UnicodeDecodeError, struct.error) as e:
            logger.warning("MQTT broker: dropping client %s: %s", client.client_id, e)
        finally:
            if client in self.clients:
                self.clients.remove(client)
            writer.close()

    def _accept_connect(self, client: _Client, body: bytes) -> bool:
        n = struct.unpack("!H", body[:2])[0]
        proto, pos = body[2:2 + n], 2 + n
        level, cflags = body[pos], body[pos + 1]
        pos += 4                                                      # level, flags, keepalive
        if proto not in (b"MQTT", b"MQIsdp") or level not in (3, 4):
            client.writer.write(_packet(CONNACK, 0, b"\x00\x01"))     # unacceptable protocol version
            return False
        n = struct.unpack("!H", body[pos:pos + 2])[0]
        client.client_id = body[pos + 2:pos + 2 + n].decode("utf-8")
        pos += 2 + n
        if cflags & 0x04:                                             # will topic + message: skip
            for _ in range(2):
                n = struct.unpack("!H", body[pos:pos + 2])[0]
                pos += 2 + n
        user = pw = None
        if cflags & 0x80:
            n = struct.unpack("!H", body[pos:pos + 2])[0]
            user = body[pos + 2:pos + 2 + n].decode("utf-8")
            pos += 2 + n
        if cflags & 0x40:
            n = struct.unpack("!H", body[pos:pos + 2])[0]
            pw = body[pos + 2:pos + 2 + n].decode("utf-8")
        if self.username and self.password and (user != self.username or pw != self.password):
            client.writer.write(_packet(CONNACK, 0, b"\x00\x04"))     # bad user name or password
            return False
        client.writer.write(_packet(CONNACK, 0, b"\x00\x00"))
        return True

    async def _on_publish(self, client: _Client, flags: int, body: bytes) -> None:
        qos, retain = (flags >> 1) & 0x03, flags & 0x01
        n = struct.unpack("!H", body[:2])[0]
        topic, pos = body[2:2 + n].decode("utf-8"), 2 + n
        if "+" in topic or "#" in topic:
            raise ValueError("wildcards are not allowed in a PUBLISH topic")
        if qos > 1:
            raise ValueError("QoS 2 is not supported by this broker")
        if qos == 1:
            pid = body[pos:pos + 2]
            pos += 2
            client.writer.write(_packet(PUBACK, 0, pid))
        payload = body[pos:]
        if retain:
            if payload:
                self.retained[topic] = payload
            else:
                self.retained.pop(topic, None)
        self.deliver(topic, payload)

    def deliver(self, topic: str, payload: bytes) -> None:
        """Send ``payload`` to every subscriber of ``topic`` (call from the broker's loop)."""
        self.n_published += 1
        pkt = _packet(PUBLISH, 0, _utf8(topic) + payload)
        for c in list(self.clients):
            if any(topic_matches(f, topic) for f in c.subscriptions):
                c.writer.write(pkt)

    async def _on_subscribe(self, client: _Client, body: bytes) -> None:
        pid, pos = body[:2], 2
        granted, new = bytearray(), []
        while pos < len(body):
            n = struct.unpack("!H", body[pos:pos + 2])[0]
            filt = body[pos + 2:pos + 2 + n].decode("utf-8")
            pos += 3 + n                                              # filter + requested QoS byte
            client.subscriptions.add(filt)
            new.append(filt)
            granted.append(0)
        client.writer.write(_packet(SUBACK, 0, pid + bytes(granted)))
        for topic, payload in self.retained.items():
            if any(topic_matches(f, topic) for f in new):
                client.writer.write(_packet(PUBLISH, 0x01, _utf8(topic) + payload))

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> None:
        """Publish from the hosting process (thread-safe), as if a client had sent it."""
        if self._loop is None:
            raise RuntimeError("broker is not running")
        if retain:
            self.retained[topic] = payload

        self._loop.call_soon_threadsafe(self.deliver, topic, payload)
