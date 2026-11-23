"""Run CalculiX (``ccx``): native binary if on PATH, else the bundled Docker image.

The image is built from ``Dockerfile`` next to this module (Debian's official ``calculix-ccx`` package on the official
Debian image, pulled from the AWS mirror of the Docker library to avoid Docker Hub's anonymous rate limit), tag
``pinneapple/calculix:local``: no third-party container is pulled.

The image is multi-platform: the same ``Dockerfile`` builds for :data:`PLATFORMS` (an x86 server or CI runner and an
Apple-silicon laptop), and :func:`build_for` tags each build with the architecture it is for. :func:`image_smoke`
builds one platform and runs a real cantilever through it -- the gate CI runs (``.github/workflows/calculix-image.yml``)
and, while the repository's runners are unavailable, ``scripts/calculix_docker_smoke.py`` runs locally.
"""
from __future__ import annotations

import os
import platform as _platform
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

IMAGE = "pinneapple/calculix:local"
_HERE = os.path.dirname(os.path.abspath(__file__))

#: The Debian version the ``Dockerfile`` pins (``ARG CALCULIX_VERSION``). Provenance of a result has to carry it:
#: ccx 2.20 and 2.23 do not have to agree to the last digit.
IMAGE_VERSION = "2.20-1"

#: The architectures the image is built and tested for. The base image digest and the Debian package are the same on
#: both; the compiled binary is not, which is why both are run and not merely built.
PLATFORMS = ("linux/amd64", "linux/arm64")


def _docker_env():
    env = dict(os.environ)
    helper = "/Applications/Docker.app/Contents/Resources/bin"  # Docker Desktop credential helper (macOS)
    if os.path.isdir(helper) and helper not in env.get("PATH", ""):
        env["PATH"] = helper + os.pathsep + env.get("PATH", "")
    return env


def docker_available() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True, env=_docker_env(), timeout=30).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


#: The same official Debian image from three registries, tried in turn: a registry that rate limits anonymous pulls
#: (public.ecr.aws answers "429 Too Many Requests / Data limit exceeded" on busy shared CI runners, Docker Hub has
#: its own limit) must not make the build fail while another one serves the image. The digest is the manifest list's,
#: so it is one value for every architecture and the same bytes on both registries that answer anonymous pulls
#: (checked on public.ecr.aws and registry-1.docker.io for debian:bookworm-slim); the mirror that only answers with
#: credentials stays on the tag. ``PINNEAPPLE_CALCULIX_BASE`` puts a base of your choice first.
DEBIAN_DIGEST = "sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc"
BASE_IMAGES = (f"public.ecr.aws/docker/library/debian:bookworm-slim@{DEBIAN_DIGEST}",
               f"debian:bookworm-slim@{DEBIAN_DIGEST}",
               "mirror.gcr.io/library/debian:bookworm-slim")


def _bases() -> list[str]:
    first = os.environ.get("PINNEAPPLE_CALCULIX_BASE")
    return ([first] if first else []) + [b for b in BASE_IMAGES if b != first]


def host_platform() -> str:
    """The platform ``docker build`` produces by default on this machine, as a Docker platform string."""
    m = _platform.machine().lower()
    return "linux/arm64" if m in ("arm64", "aarch64") else "linux/amd64"


def platform_tag(platform: str, image: str = IMAGE) -> str:
    """``pinneapple/calculix:local`` + ``linux/arm64`` -> ``pinneapple/calculix:local-arm64``: one tag per platform, so
    the two images can live in the same store instead of overwriting each other."""
    return f"{image}-{platform.rsplit('/', 1)[-1]}"


def ensure_image(image: str = IMAGE, attempts: int = 3, platform: str | None = None) -> str:
    """Build the image if it is missing, for ``platform`` (the host's by default). The build downloads the base image
    and the Debian package, so a failed attempt is retried, each time from the next registry of :data:`BASE_IMAGES`
    (registry or mirror hiccups and rate limits); the last failure is raised with Docker's own output."""
    env = _docker_env()
    if subprocess.run(["docker", "image", "inspect", image], capture_output=True, env=env).returncode == 0:
        return image
    bases, plat = _bases(), platform or host_platform()
    for k in range(max(1, attempts)):
        base = bases[k % len(bases)]
        p = subprocess.run(["docker", "build", "-q", "--platform", plat, "--build-arg", f"BASE={base}", "-t", image,
                            _HERE], capture_output=True, text=True, env=env)
        if p.returncode == 0:
            return image
        if k + 1 < attempts:
            time.sleep(10 * (k + 1))
    raise RuntimeError(f"docker build of {image} for {plat} failed {attempts} time(s) (bases tried: "
                       f"{', '.join(bases[:max(1, attempts)])}); last output:\n{(p.stdout + p.stderr)[-4000:]}")


def build_for(platform: str, image: str = IMAGE, attempts: int = 3) -> str:
    """Build the image for one platform under its own tag (:func:`platform_tag`) and return that tag. The multi-platform
    entry point: on an arm64 machine ``linux/amd64`` is built under emulation (Docker Desktop's qemu or Rosetta), which
    is how the x86 image gets tested from a laptop."""
    return ensure_image(platform_tag(platform, image), attempts=attempts, platform=platform)


def ccx_backend() -> str | None:
    """'native', 'docker' or None."""
    if shutil.which("ccx"):
        return "native"
    if docker_available():
        return "docker"
    return None


@dataclass
class CcxRun:
    returncode: int
    stdout: str
    frd: str
    dat: str


def run_ccx(workdir: str, job: str, *, threads: int = 1, timeout: float = 3600.0, backend: str | None = None,
            image: str | None = None, platform: str | None = None) -> CcxRun:
    """Run ``ccx -i <job>`` in ``workdir`` (which must contain ``<job>.inp``). ``backend='docker'`` runs it in
    ``image`` (the bundled one by default), for ``platform`` when given: with two architectures of the same name in
    the store, the flag is what says which one."""
    backend = backend or ccx_backend()
    env = dict(_docker_env(), OMP_NUM_THREADS=str(threads))
    if backend == "native":
        cmd = ["ccx", "-i", job]
    elif backend == "docker":
        tag = image or ensure_image()
        cmd = ["docker", "run", "--rm"] + (["--platform", platform] if platform else [])
        cmd += ["-e", f"OMP_NUM_THREADS={threads}", "-v", f"{os.path.abspath(workdir)}:/work", "-w", "/work",
                tag, "-i", job]
    else:
        raise RuntimeError("CalculiX not available: install ccx or Docker")
    p = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout, env=env)
    return CcxRun(p.returncode, p.stdout + p.stderr, os.path.join(workdir, job + ".frd"),
                  os.path.join(workdir, job + ".dat"))


def image_version(image: str = IMAGE, platform: str | None = None) -> str:
    """The ``ccx`` version the image reports (``ccx -v``), e.g. ``'2.20'``; '' when it cannot be read."""
    cmd = ["docker", "run", "--rm"] + (["--platform", platform] if platform else []) + [image, "-v"]
    p = subprocess.run(cmd, capture_output=True, text=True, env=_docker_env(), timeout=300)
    m = re.search(r"Version\s+([\d.]+)", p.stdout + p.stderr)
    return m.group(1) if m else ""


def image_smoke(platform: str, image: str = IMAGE, timeout: float = 1800.0) -> dict:
    """The gate: build the image for one platform, run a real CalculiX case in it -- a C3D20R cantilever 1 x 0.1 x
    0.05 m under a 1 kN tip load, whose deflection Timoshenko's theory fixes to 1 % -- and compare. Returns what it
    measured; ``ok`` is the verdict. The workflow and the local script call this same function, so "the image is
    tested" means one thing in both places.
    """
    import numpy as np

    from .frd import read_frd
    from .inp import cantilever_hex_mesh, write_inp

    t0 = time.time()
    tag = build_for(platform, image)
    version = image_version(tag, platform)
    work = tempfile.mkdtemp(prefix="ccx-smoke-")
    try:
        L, B, H, P = 1.0, 0.1, 0.05, 1000.0
        m = cantilever_hex_mesh(L, B, H, nx=20, ny=2, nz=2, element_type="C3D20R")
        m.boundary = [("FIXED", 1, 3, 0.0)]
        m.cloads = [(str(n), 3, -P / len(m.nsets["TIP"])) for n in m.nsets["TIP"]]
        write_inp(m, os.path.join(work, "smoke.inp"))
        run = run_ccx(work, "smoke", backend="docker", image=tag, platform=platform, timeout=timeout)
        E, NU = 210e9, 0.3
        expected = (P * L ** 3 / (3 * E * (B * H ** 3 / 12))
                    + P * L / (5 / 6 * (E / (2 * (1 + NU))) * B * H))
        got = None
        if run.returncode == 0 and os.path.exists(run.frd):
            real = read_frd(run.frd)
            got = float(real.field("DISP")[np.isin(real.node_ids, m.nsets["TIP"]), 2].mean())
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ok = bool(got is not None and abs(abs(got) / expected - 1) < 0.02 and version == IMAGE_VERSION.split("-")[0])
    return {"platform": platform, "image": tag, "version": version, "expected_version": IMAGE_VERSION.split("-")[0],
            "tip_deflection_m": got, "timoshenko_m": expected,
            "rel_err": None if got is None else float(abs(got) / expected - 1),
            "returncode": run.returncode, "ok": ok, "seconds": round(time.time() - t0, 1),
            "log": "" if ok else run.stdout[-2000:]}
