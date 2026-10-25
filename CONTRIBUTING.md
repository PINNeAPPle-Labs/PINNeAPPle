# Contributing to PINNeAPPle

Thanks for taking the time to contribute!

## Reporting bugs and requesting features
Open an issue in the [issue tracker](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues) using the bug report or
feature request template. For a bug, include the PINNeAPPle version (`python -c "import pinneapple; print(pinneapple.__version__)"`),
your Python and PyTorch versions, a minimal script that reproduces the problem, and the full error message.
Security problems must not be reported in public issues: follow [SECURITY.md](SECURITY.md).

## Getting help
Ask usage questions in the [issue tracker](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues). The documentation is at
<https://pinneapple-labs.github.io/PINNeAPPle/>, and runnable examples are in [`examples/`](examples/).

## Code of conduct
Everyone taking part in this project is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Governance

### Who reviews
Pull requests are reviewed by maintainers of the [PINNeAPPle-Labs](https://github.com/PINNeAPPle-Labs) organization
(members with write access to this repository). Anyone may comment on issues and pull requests; only maintainers merge
to the default branch. At least one maintainer approval is required before merge. Release, security, and large
API or behavior changes may need a second maintainer look.

Code of Conduct reports are handled by the same maintainers, as described in [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

### Response times (best effort)
We are a small team and cannot guarantee SLAs, but we aim for:

| Kind of request | Target first response |
|---|---|
| Security vulnerability ([SECURITY.md](SECURITY.md)) | Within 2 business days |
| Bug reports and feature requests | Within 1 week |
| Pull requests | Initial review within 2 weeks |

Silence longer than this does not mean rejection — please bump the thread or open a discussion issue if you are blocked.

### Becoming a maintainer
There is no application form. Maintainers are invited when someone has shown sustained, high-quality contributions
(issues, pull requests, reviews, documentation) and works well with the community under the Code of Conduct.
Existing maintainers decide by consensus whether to extend write access. Invitations are occasional and depend on
project needs, not on a fixed schedule or quota.

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

## Where code goes
Reusable logic goes in the library; apps and examples only compose library calls. Physics, numerics, geometry,
data processing, metrics, plotting helpers, trust checks, IO, solvers and training loops belong in a `pinneapple*`
package, with a public API, docstrings and tests (against an independent reference when the code is numerical), so
users get them with `pip install pinneapple`. `apps/` keeps only glue (FastAPI routes, request schemas, HTML/JS,
deploy files) and `examples/` only the narrative of the example. `python scripts/check_app_helpers.py` (run in CI by
`tests/test_app_helpers.py`) fails on a new local helper module in `apps/` or `examples/`, or on a new local copy of a
helper the library already has; [docs/dev/LIBRARY_CONSOLIDATION.md](docs/dev/LIBRARY_CONSOLIDATION.md) is the
inventory of what is being moved.

## Pull requests
- Keep changes focused and well-scoped
- Add or update tests when possible
- Update docs/examples if behavior changes
- Do not commit generated files (`data/web/hub/index.json`, `docs/flow_studio/COVERAGE.md`,
  `docs/flow_studio/EXAMPLES_AS_FLOWS.md`): they are git-ignored and built where they are used. The commands that
  build them locally, and what is committed instead, are in [AGENTS.md](AGENTS.md#generated-files).

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
