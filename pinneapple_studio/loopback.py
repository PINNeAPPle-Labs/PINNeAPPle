"""Run a module with the loopback interface up: ``python -m pinneapple_studio.loopback <module> [args...]``.

A recorded example run has no network (``unshare -n``): its namespace starts with ``lo`` down, so even a
local rendezvous (``torchrun --standalone``) cannot connect. This brings ``lo`` up inside that namespace, which keeps
every other address unreachable, and then runs the module as ``python -m`` would. Outside a fresh namespace ``lo``
is already up and nothing changes."""
from __future__ import annotations

import fcntl
import runpy
import socket
import struct
import sys

SIOCGIFFLAGS, SIOCSIFFLAGS, IFF_UP = 0x8913, 0x8914, 0x1


def _loopback_up() -> bool:
    """Set ``lo`` up; ``False`` when this process may not (the run then goes on as it is)."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            flags = struct.unpack("16sH", fcntl.ioctl(s, SIOCGIFFLAGS, struct.pack("16sH", b"lo", 0)))[1]
            if not flags & IFF_UP:
                fcntl.ioctl(s, SIOCSIFFLAGS, struct.pack("16sH", b"lo", flags | IFF_UP))
        return True
    except OSError:
        return False


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        raise SystemExit("usage: python -m pinneapple_studio.loopback <module> [args...]")
    _loopback_up()
    sys.argv = argv
    runpy.run_module(argv[0], run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
