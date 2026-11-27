- **Release preflight** (#421, #422, #423): `scripts/release_preflight.py` reports every blocker for a tag at once —
  the three version files agreeing with the version asked for, nothing left in the CHANGELOG's `[Unreleased]`, no
  unfurled fragment in `changelog.d/`, a dated heading for the version, and no open item in the GitHub milestone
  (looked up by title, since the numbers are stable only by accident). It exits 1 when it blocks, prints a table or
  `--json`, and when `gh` is unavailable it says so in a note instead of inventing a blocker. On 2026-11-27 it
  measured: v0.8.0 with 10 open items and a CHANGELOG already dated 2026-11-01 (the release shipped ahead of its own
  exit criterion), v0.9.0 with 9 open, v1.0.0 with 8 open, and **241 changelog fragments** left unfurled since the
  last fold — the actual content of the next release notes, invisible to anyone reading `CHANGELOG.md`.
