"""The bundled CalculiX image: pinned, multi-platform, and proven by running a real ``ccx`` case inside it (#70).

``test_the_image_runs_a_real_ccx_case`` is the gate: it builds the image for the host's platform and runs a cantilever
through the container, so an image that builds but cannot solve fails here. It skips without Docker, like the rest of
the repository's optional tools, and ``scripts/calculix_docker_smoke.py`` runs the same function for both platforms --
the local stand-in for ``.github/workflows/calculix-image.yml`` while the repository's runners are unavailable.

The tests above the gate need no Docker: they hold the pin (a Dockerfile that rebuilds itself differently every day is
not reproducible) and the platform plumbing.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from pinneapple_simulation.external_solvers.calculix import runner

DOCKERFILE = Path(runner.__file__).with_name("Dockerfile")
needs_docker = pytest.mark.skipif(not runner.docker_available(), reason="Docker is not available")


def _arg(name: str) -> str:
    m = re.search(rf"^ARG {name}=(.*)$", DOCKERFILE.read_text(encoding="utf-8"), re.M)
    assert m, f"the Dockerfile declares no ARG {name}"
    return m.group(1).strip()


def test_the_base_image_is_pinned_by_the_digest_of_its_manifest_list():
    base = _arg("BASE")
    assert "@sha256:" in base, "the base is on a tag: the same commit would build a different Debian tomorrow"
    assert base.endswith(runner.DEBIAN_DIGEST), "runner.DEBIAN_DIGEST and the Dockerfile's BASE disagree"
    # one digest for every architecture is what makes a single Dockerfile multi-platform: it has to be a manifest list
    assert len(runner.DEBIAN_DIGEST) == len("sha256:") + 64


def test_the_solver_version_is_pinned_and_asserted_at_build_time():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert _arg("CALCULIX_VERSION") == runner.IMAGE_VERSION, "runner.IMAGE_VERSION and the Dockerfile disagree"
    assert 'apt-get' in text and 'install -y --no-install-recommends "calculix-ccx=${CALCULIX_VERSION}"' in text, \
        "the package is installed without its version: apt would take whatever the mirror serves"
    assert 'grep -qx "Version: ${CALCULIX_VERSION}"' in text and 'ccx -v 2>&1 | grep -q "Version ' in text, \
        "the build does not assert the version it installed"
    assert 'ENTRYPOINT ["ccx"]' in text


def test_the_architecture_the_image_really_is_written_at_build_time():
    """BuildKit's TARGETPLATFORM only arrives with a buildx builder, so the image records what dpkg installed -- the
    architecture of the bytes -- and refuses to build for one this project does not test."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert 'dpkg --print-architecture' in text and '/etc/pinneapple-arch' in text
    assert "case \"${arch}\" in amd64|arm64)" in text, "the build accepts architectures nobody tests"
    assert runner.PLATFORMS == ("linux/amd64", "linux/arm64")


def test_each_platform_gets_its_own_tag_and_the_host_is_one_of_them():
    assert runner.host_platform() in runner.PLATFORMS
    assert runner.platform_tag("linux/arm64") == f"{runner.IMAGE}-arm64"
    assert runner.platform_tag("linux/amd64") == f"{runner.IMAGE}-amd64"
    assert runner.platform_tag("linux/arm64") != runner.platform_tag("linux/amd64")


def test_the_build_and_the_run_say_which_platform_they_mean(monkeypatch):
    """Two architectures of the same image in one store are ambiguous: the flags are what tell Docker which to build
    and which to run."""
    seen = []

    class P:
        def __init__(self, rc):
            self.returncode, self.stdout, self.stderr = rc, "", ""

    def fake_run(cmd, **kw):
        seen.append(cmd)
        return P(1 if cmd[:3] == ["docker", "image", "inspect"] else 0)   # the image is missing, then builds
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    runner.build_for("linux/amd64", image="pinneapple/calculix:test")
    assert seen[-1][:2] == ["docker", "build"] and "linux/amd64" in seen[-1]
    assert runner.platform_tag("linux/amd64", "pinneapple/calculix:test") in seen[-1]
    seen.clear()
    runner.run_ccx(str(DOCKERFILE.parent), "job", backend="docker", image="pinneapple/calculix:test-arm64",
                   platform="linux/arm64")
    assert seen[-1][:2] == ["docker", "run"] and "--platform" in seen[-1]
    assert seen[-1][seen[-1].index("--platform") + 1] == "linux/arm64"


@needs_docker
def test_the_image_runs_a_real_ccx_case():
    row = runner.image_smoke(runner.host_platform())
    assert row["version"] == runner.IMAGE_VERSION.split("-")[0], f"the image runs ccx {row['version']}: {row['log']}"
    assert row["arch"] == row["expected_arch"], f"the image is {row['arch']}, the platform asked {row['platform']}"
    assert row["ok"], f"tip deflection {row['tip_deflection_m']} vs Timoshenko {row['timoshenko_m']}: {row['log']}"
    assert row["rel_err"] == pytest.approx(0.0, abs=0.02)
    assert row["image"].endswith(runner.host_platform().rsplit("/", 1)[-1])
