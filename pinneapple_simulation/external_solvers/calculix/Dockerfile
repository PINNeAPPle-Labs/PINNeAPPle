# CalculiX CrunchiX (ccx) from the official Debian package: reproducible, no third-party image, and one file for both
# architectures the project runs on (linux/amd64 servers and CI runners, linux/arm64 Apple-silicon laptops).
#
# Pinned twice, because a tag is not a version:
#   * the base image by the digest of its manifest list. One digest covers every architecture Debian publishes, so a
#     multi-platform build needs this single ARG; a rebuild of the same commit gets the same Debian.
#   * the calculix-ccx package by its Debian version, asserted at build time: apt would otherwise install whatever the
#     mirror serves that day. bookworm ships 2.20-1 on amd64 and on arm64. Pinning the exact version means a Debian
#     point release breaks the build until this file is updated -- which is the point: the image never changes the
#     solver under a commit.
#
# The official Debian image is pulled from the AWS mirror of the Docker library: anonymous pulls from Docker Hub are
# rate limited (429 Too Many Requests on shared CI runners). runner.ensure_image falls back to the other entries of
# BASE_IMAGES when one refuses; override with
# --build-arg BASE=debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc
ARG BASE=public.ecr.aws/docker/library/debian:bookworm-slim@sha256:734a72b3d6142a71993b4bc580ca01d8ca687f77a2c422d96a961d3376b179bc
FROM ${BASE}

ARG CALCULIX_VERSION=2.20-1
# BuildKit's platform meta-args, for the label only. They are empty under the plain docker driver (`docker build`, which
# is what runner.ensure_image uses without a buildx builder), so the architecture this image really is comes from dpkg
# below and is written into /etc/pinneapple-arch -- a fact about the bytes, not a declaration about the build.
ARG TARGETPLATFORM

LABEL org.opencontainers.image.title="pinneapple/calculix" \
      org.opencontainers.image.description="CalculiX CrunchiX (ccx) from the Debian package, linux/amd64 and linux/arm64" \
      org.opencontainers.image.version="2.20-1" \
      org.opencontainers.image.url="https://www.calculix.de" \
      org.opencontainers.image.source="https://github.com/PINNeAPPle-Labs/PINNeAPPle" \
      org.pinneapple.solver.plain-version="2.20" \
      org.pinneapple.platform="${TARGETPLATFORM}"

# The architecture is recorded and restricted to the two this project tests (an image for an untested architecture is a
# promise nobody checked); the Debian version is asserted; and then the binary itself, so a mirror that serves another
# build fails here instead of in somebody's benchmark.
RUN set -eu; \
    arch="$(dpkg --print-architecture)"; \
    case "${arch}" in amd64|arm64) ;; *) echo "unsupported architecture: ${arch}" >&2; exit 1 ;; esac; \
    printf '%s\n' "${arch}" > /etc/pinneapple-arch; \
    apt-get -o Acquire::Retries=5 update; \
    apt-get -o Acquire::Retries=5 install -y --no-install-recommends "calculix-ccx=${CALCULIX_VERSION}"; \
    rm -rf /var/lib/apt/lists/*; \
    dpkg -s calculix-ccx | grep -qx "Version: ${CALCULIX_VERSION}"; \
    ccx -v 2>&1 | grep -q "Version ${CALCULIX_VERSION%-*}"

WORKDIR /work
ENTRYPOINT ["ccx"]
