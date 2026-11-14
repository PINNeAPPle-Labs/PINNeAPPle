"""Building the C backend: one compiler invocation, cached by the fingerprint of what it compiles.

The library is a build artifact, so it lives where build artifacts live — a cache directory, never the
checkout (a committed binary would conflict on the next merge and could not be audited). The cache key is the
SHA-256 of the header, the source, the flags and the compiler version, so editing a kernel or changing a flag
rebuilds and nothing else does.

Two environment variables decide what happens:

* ``PINNEAPPLE_NATIVE=0`` — never build or load a native backend. The NumPy reference runs and every result
  says so. This is what a restricted CI runner or a locked-down machine wants.
* ``PINNEAPPLE_NATIVE_CACHE`` — where to put the cache (default: the platform's user cache directory, e.g.
  ``~/Library/Caches/pinneapple/native`` on macOS). Also ``PINNEAPPLE_NATIVE_LIBRARY`` to point at a library
  you built yourself instead.

A missing compiler is not an error of this library: it is a machine that cannot run this backend, and
:class:`~pinneapple_core.native.abi.CompilerNotFound` says which command was looked for.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from .abi import ABI_VERSION, BuildFailed, CompilerNotFound

__all__ = ["source_paths", "compiler", "cache_root", "build", "library_path", "enabled", "status", "CC_ENV"]

HERE = Path(__file__).resolve().parent
HEADER = HERE / "include" / "pinneapple_native.h"
SOURCE = HERE / "csrc" / "kernels.c"

LIBRARY_BASENAME = "libpinneapple_native"
CC_ENV = "PINNEAPPLE_NATIVE_CC"
FLAGS = ("-O3", "-fPIC", "-shared", "-std=c11", "-fno-fast-math", "-Wall", "-Wextra")

#: Compilers tried in order when PINNEAPPLE_NATIVE_CC is not set.
CANDIDATES = ("cc", "clang", "gcc")


def enabled() -> bool:
    """False when ``PINNEAPPLE_NATIVE=0``: the native path is opt-out, so a machine without a compiler never
    has to discover the failure halfway through a study."""
    return os.environ.get("PINNEAPPLE_NATIVE", "1").strip().lower() not in ("0", "false", "off", "no")


def source_paths() -> tuple[Path, Path]:
    """``(header, source)`` — the two files whose bytes define the library."""
    return HEADER, SOURCE


def compiler() -> str | None:
    """The C compiler this machine has, or ``None``. ``PINNEAPPLE_NATIVE_CC`` wins when it is set."""
    wanted = os.environ.get(CC_ENV)
    if wanted:
        return wanted if Path(wanted).exists() else shutil.which(wanted)
    for candidate in CANDIDATES:
        found = shutil.which(candidate)
        if found:
            return found
    return None


def cache_root() -> Path:
    """Where the built libraries live."""
    override = os.environ.get("PINNEAPPLE_NATIVE_CACHE")
    if override:
        return Path(override).expanduser()
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "pinneapple" / "native"
    if os.name == "nt":
        return Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "pinneapple" / "native"
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "pinneapple" / "native"


def library_name() -> str:
    suffix = {"darwin": ".dylib", "win32": ".dll"}.get(sys.platform, ".so")
    return LIBRARY_BASENAME + suffix


def _compiler_version(cc: str) -> str:
    try:
        out = subprocess.run([cc, "--version"], capture_output=True, text=True, timeout=30, check=False)
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - depends on the machine
        return f"unavailable: {exc}"
    return (out.stdout or out.stderr).splitlines()[0] if (out.stdout or out.stderr) else "unknown"


def fingerprint(cc: str) -> str:
    """SHA-256 over everything that decides the bytes of the library."""
    digest = hashlib.sha256()
    for path in source_paths():
        digest.update(path.read_bytes())
    digest.update("\0".join(FLAGS).encode())
    digest.update(f"\0abi={ABI_VERSION}\0cc={cc}\0{_compiler_version(cc)}".encode())
    digest.update(f"\0machine={platform.machine()}\0platform={sys.platform}".encode())
    return digest.hexdigest()[:16]


def _compile_command(cc: str, target: Path) -> list[str]:
    build_line = f"{Path(cc).name} {' '.join(FLAGS)} ({sys.platform}/{platform.machine()})"
    return [
        cc, *FLAGS,
        f'-DPP_BUILD="{build_line}"',
        "-I", str(HEADER.parent),
        "-o", str(target),
        str(SOURCE),
    ]


def build(force: bool = False) -> Path:
    """Build the C backend into the cache and return the library path.

    Raises :class:`~pinneapple_core.native.abi.CompilerNotFound` when there is no compiler and
    :class:`~pinneapple_core.native.abi.BuildFailed` with the command and its stderr when the compiler
    refuses the source. A cached library from the same fingerprint is returned without invoking a compiler.
    """
    cc = compiler()
    if cc is None:
        raise CompilerNotFound(
            "no C compiler found (looked for " + ", ".join(CANDIDATES) + f"; set {CC_ENV} to name one). "
            "The NumPy reference implementation still runs — nothing is silently wrong, it is slower."
        )
    target_dir = cache_root() / fingerprint(cc)
    target = target_dir / library_name()
    if target.exists() and not force:
        return target
    target_dir.mkdir(parents=True, exist_ok=True)
    command = _compile_command(cc, target)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.SubprocessError) as exc:
        raise BuildFailed(f"could not run the compiler: {exc}", command=command[1:]) from exc
    if result.returncode != 0 or not target.exists():
        raise BuildFailed(
            f"the compiler exited {result.returncode} without producing {target.name}",
            command=tuple(command), stderr=(result.stderr or result.stdout or "").strip(),
        )
    (target_dir / "build.json").write_text(
        json.dumps(
            {
                "abi_version": ABI_VERSION,
                "command": command,
                "compiler": cc,
                "compiler_version": _compiler_version(cc),
                "flags": list(FLAGS),
                "library": target.name,
                "fingerprint": fingerprint(cc),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return target


def library_path() -> Path | None:
    """The library to load: ``PINNEAPPLE_NATIVE_LIBRARY`` if set, else the cached build, else ``None``.

    This is the quiet form of :func:`build`: it never raises, so the dispatcher can fall back. Use
    :func:`status` when the reason matters — and it does, so a report can print it.
    """
    override = os.environ.get("PINNEAPPLE_NATIVE_LIBRARY")
    if override:
        path = Path(override).expanduser()
        return path if path.exists() else None
    if not enabled():
        return None
    try:
        return build()
    except (CompilerNotFound, BuildFailed):
        return None
    except OSError:  # pragma: no cover - an unwritable cache, not a programming error
        return None


def status() -> dict:
    """Why the native backend is or is not usable, as the sentence a report prints.

    Keys: ``enabled``, ``compiler``, ``library``, ``available``, ``reason``. ``available`` is ``True`` only
    when a library exists *and* the opt-out is not set — a built file left in the cache does not make a
    ``PINNEAPPLE_NATIVE=0`` run use it.
    """
    info = {"enabled": enabled(), "compiler": compiler(), "library": None, "available": False, "reason": ""}
    if not info["enabled"]:
        info["reason"] = "PINNEAPPLE_NATIVE=0: the native backend is switched off for this process"
        return info
    path = library_path()
    info["library"] = str(path) if path else None
    if path is None:
        info["reason"] = (
            "no C compiler found (looked for " + ", ".join(CANDIDATES) + f"); set {CC_ENV} to name one"
            if info["compiler"] is None
            else "the C source did not build; run pinneapple_core.native.build.build() to see the compiler output"
        )
        return info
    info["available"] = True
    info["reason"] = f"libpinneapple_native built with {info['compiler']}"
    return info
