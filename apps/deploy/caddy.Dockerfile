# The deploy proxy: the pinned Caddy image with the vulnerable Alpine package upgraded at build time, so a CVE fixed in Alpine
# (CVE-2026-111497 in brotli-libs, still in caddy:2 as of 2026-10-29) does not wait for an upstream rebuild.
# CI builds this file and scans the result (.github/workflows/security.yml).
FROM caddy:2@sha256:685ae3d7c34d9b5f8caeddc5d14051db7bbd88ff1f5a2fa0c16bd8ca328fb7af
RUN apk add --no-cache --upgrade brotli-libs
