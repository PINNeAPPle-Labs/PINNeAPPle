"""Optional live OpenFOAM log adapter for Physics Dashboard Studio.

Run from the solver host. Only emits measured residuals; no synthetic samples.
A caller owns credentials and decides whether to retry failed publications.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

from .telemetry import TelemetryEmitter, http_publisher

_RESIDUAL = re.compile(
    r"Solving for\s+([A-Za-z][A-Za-z0-9_]*)\s*,\s*Initial residual\s*=\s*([0-9.eE+-]+)"
)


def follow_openfoam(path: Path, emitter: TelemetryEmitter, *, interval: float = 0.5,
                    start_at_end: bool = False, stop=None):
    """Follow solver logs across truncation/rotation with bounded memory.

    Iteration index counts parsed residual records, not OpenFOAM time steps.
    A partial line is held until newline. Rotation reopens the file.
    """
    import math
    offset = 0
    inode = None
    buffer = b""
    started = False
    step = 0
    while stop is None or not stop():
        try:
            stat = path.stat()
            identity = (stat.st_dev, stat.st_ino)
            if identity != inode or stat.st_size < offset:
                inode, offset, buffer = identity, 0, b""
                if start_at_end and not started:
                    offset = stat.st_size
                started = True
            with path.open("rb") as stream:
                stream.seek(offset)
                chunk = stream.read(65536)
                offset = stream.tell()
            if not chunk:
                time.sleep(interval)
                continue
            buffer += chunk
            if len(buffer) > 65536 and b"\n" not in buffer:
                buffer = b""
                continue
            *lines, buffer = buffer.split(b"\n")
            for raw in lines:
                if len(raw) > 65536:
                    continue
                match = _RESIDUAL.search(raw.decode("utf-8", "replace"))
                if not match:
                    continue
                value = float(match.group(2))
                if not math.isfinite(value):
                    continue
                emitter.emit("residuals", step, **{match.group(1): value})
                step += 1
        except FileNotFoundError:
            time.sleep(interval)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Stream OpenFOAM residual logs to Physics Dashboard")
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--endpoint", required=True, help="HTTPS origin of engineering API")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--from-end", action="store_true")
    parser.add_argument("--interval", type=float, default=0.5)
    args = parser.parse_args(argv)
    if not 0.1 <= args.interval <= 30:
        parser.error("--interval must be between 0.1 and 30 seconds")
    key = os.environ.get("PINNEAPPLE_DASHBOARD_API_KEY")
    if not key:
        parser.error("PINNEAPPLE_DASHBOARD_API_KEY is required")
    publisher = http_publisher(args.endpoint, key)
    emitter = TelemetryEmitter("simulation", args.case_id, publisher)
    follow_openfoam(args.log, emitter, interval=args.interval, start_at_end=args.from_end)


if __name__ == "__main__":
    main()
