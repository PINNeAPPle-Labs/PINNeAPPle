# Contributing to PINNeAPPle

Thanks for taking the time to contribute!

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

## Running tests

```bash
pytest -q
```

## Lint / format

```bash
ruff check .
ruff format .
```

## Pull requests
- Keep changes focused and well-scoped
- Add or update tests when possible
- Update docs/examples if behavior changes

## Changelog
Every user-visible change (new feature, bug fix, behavior change, removal) needs one line under
`## [Unreleased]` in [`CHANGELOG.md`](CHANGELOG.md), in the same pull request. Use the headings
Added / Changed / Fixed / Removed / Documentation / Known issues. While the major version is 0, start any
breaking or behavior-changing entry with **Breaking:** or **Behavior:**. Maintainers move the Unreleased
entries under a dated version heading when they cut a release (see below). Purely internal changes
(refactors, test-only, CI) do not need an entry.

## Releasing (maintainers)
1. Move the `[Unreleased]` entries under `## [x.y.z] - YYYY-MM-DD` and update the compare links at the bottom.
2. Bump the version in `pyproject.toml`, `CITATION.cff` and `pinneapple/__init__.py` (`__version__`, a separate copy).
3. Tag the release commit `vx.y.z` and push the tag. The `Release to PyPI` workflow checks that the tag, `pyproject.toml`,
   `CITATION.cff`, `pinneapple.__version__` and the changelog agree (`python scripts/check_release_version.py vx.y.z` runs the
   same check locally), builds, and publishes through PyPI Trusted Publishing. One-time setup: add the publisher on PyPI
   (workflow `release.yml`, environment `pypi`) and create the `pypi` environment on GitHub. Never upload by hand.

## Commit style
We recommend Conventional Commits (optional), e.g.:
- feat: add shard-aware iterator
- fix: correct zarr cache eviction
- docs: improve README examples

## Development history and internal engineering notes
[`docs/dev/ROADMAP_PHYSICS_AI_HUB.md`](docs/dev/ROADMAP_PHYSICS_AI_HUB.md) and
[`docs/dev/AUDIT_REPORT.md`](docs/dev/AUDIT_REPORT.md) are internal engineering
notes kept for historical context: a running roadmap and an audit log written
during development sessions. They are not polished, curated documentation —
expect first-person session narration, in-progress task tracking, and
references to work that may since have changed — but they're useful if you
want the backstory on why a design decision was made or what's already been
tried. For current, user-facing documentation see the [`docs/`](docs/)
directory and [`README.md`](README.md).
