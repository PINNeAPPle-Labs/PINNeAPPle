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

One command, then one command to know the checkout works:

```bash
make setup    # .venv with the package installed in editable mode and the [dev] extras
make test     # the fast suite, ~30 s on a laptop
```

`make` on its own lists the targets. Each target is one job, and each is what CI runs, so a green `make test`
locally means the same thing as a green workflow:

| Command | What it does | Time |
|---|---|---|
| `make setup` | `python3 -m venv .venv`, then `pip install -e ".[dev]"` | a few minutes, once |
| `make test` | the fast suite: public surface, front door, executed documentation, changelog and licensing rules, the module reference scripts | ~30 s |
| `make test-full` | `tests/`, the same selection the full-suite workflow runs | minutes |
| `make lint` | the ruff selection CI enforces (`E9`: what ruff reports when a file does not parse) | ~2 s |
| `make lint-backlog` | the wider selection declared in `pyproject.toml` (`E`, `F`, `I`, `B`, `UP`), as a statistic | ~2 s |
| `make gates` | `scripts/changelog.py check`, `scripts/reference/build.py --check`, the licensing check | ~5 s |
| `make docs` | regenerate the module reference and build the site (needs `pip install -r requirements-docs.txt`) | minutes |
| `make clean` | build and cache output, keeping `.venv` | — |

The commands the targets wrap are the ones below; use them directly if you prefer, or if your platform has no
`make`. Point the targets at another interpreter with `make test PY=/usr/bin/python3`, and at the CPU wheel of
torch with `make setup TORCH_INDEX=https://download.pytorch.org/whl/cpu`.

A dev container is defined in [`.devcontainer/devcontainer.json`](.devcontainer/devcontainer.json): open the
repository in a codespace, or in VS Code with the Dev Containers extension, and it runs `make setup` for you.

### Without make

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[dev]"
```

## Running tests

```bash
make test      # the fast suite, the list below
pytest tests/  # everything
```

```bash
pytest tests/test_public_api.py tests/test_front_door.py tests/test_onboarding.py \
       tests/test_agents_md.py tests/test_changelog_fragments.py tests/test_licensing_policy.py \
       tests/test_stability.py tests/test_solve_guide.py tests/test_dev_environment.py -q
```

The fast suite is deliberately short: anything that trains a network, renders an image or needs an external solver
belongs to `make test-full`, so the answer to "can I open a pull request" arrives in under a minute.

## Lint / format

```bash
make lint            # what CI enforces: ruff with `--select E9`, expected to pass
make lint-backlog    # the selection declared in pyproject.toml, currently ~21,000 findings
ruff format <files>  # a file you changed; there is no repository-wide format pass in CI
```

`[tool.ruff.lint]` in `pyproject.toml` selects `E`, `F`, `I`, `B` and `UP`, and the tree does not satisfy that
selection yet — mostly `UP006`/`UP045` (typing to builtin generics) and `I001` (import order). It is a backlog to
burn down one rule at a time, not a gate, which is why CI runs only `E9` and why `make lint` matches CI. A new file
should not add to the backlog, and the last rule burned down moves from `make lint-backlog` into `make lint`.

## Where code goes
Reusable logic goes in the library; apps and examples only compose library calls. Physics, numerics, geometry,
data processing, metrics, plotting helpers, trust checks, IO, solvers and training loops belong in a `pinneapple*`
package, with a public API, docstrings and tests (against an independent reference when the code is numerical), so
users get them with `pip install pinneapple`. `apps/` keeps only glue (FastAPI routes, request schemas, HTML/JS,
deploy files) and `examples/` only the narrative of the example. `python scripts/check_app_helpers.py` (run in CI by
`tests/test_app_helpers.py`) fails on a new local helper module in `apps/` or `examples/`, or on a new local copy of a
helper the library already has; [docs/dev/LIBRARY_CONSOLIDATION.md](docs/dev/LIBRARY_CONSOLIDATION.md) is the
inventory of what is being moved.

## Core and experimental
The **core** (`pinneapple_stability.CORE` and `CORE_PP`) is covered by semantic versioning. Do not break a core name.
To rename one, keep the old name working with a deprecation warning for at least one minor release. Everything else is
**experimental**: it is kept and developed, but carries no compatibility promise. An experimental module is promoted to
the core when it is fully tested and used. See [docs/api_stability.md](docs/api_stability.md).

## Pull requests
- Keep changes focused and well-scoped
- Add or update tests when possible
- Update docs/examples if behavior changes
- Do not commit generated files (`data/web/hub/index.json`, `docs/flow_studio/COVERAGE.md`,
  `docs/flow_studio/EXAMPLES_AS_FLOWS.md`): they are git-ignored and built where they are used. The commands that
  build them locally, and what is committed instead, are in [AGENTS.md](AGENTS.md#generated-files).

### The module reference

`docs/org/reference/` (published under `/PINNeAPPle/org/reference/`) is generated from the source, and it is
committed. Two commands, and CI runs both:

```bash
python scripts/reference/run_examples.py           # only when you changed a curated example
python scripts/reference/build.py                 # after any change to a public signature
```

A pull request that changes a public class, function or module and does not rebuild the reference fails the
`Module reference` workflow with the list of stale files and the command to run. A pull request that breaks a
curated example fails the same workflow, naming the example. Use `--check` on either script to verify without
writing anything:

```bash
python scripts/reference/build.py --check
python scripts/reference/run_examples.py --check [package ...]
```

## Licensing

Apache-2.0, no CLA and no required DCO sign-off: a contribution is accepted under the licence the project ships
under. GPL code is never copied into this repository, only the idea; an adapted file gets an entry in `NOTICE`; and
a new dependency is permissive unless its copyleft licence was decided in [`docs/licensing.md`](docs/licensing.md)
first. Details, and the command that prints the licence of every declared dependency, are in that page.

## Changelog
Every user-visible change (new feature, bug fix, behavior change, removal) needs one entry, in the same pull
request, as a **new file in [`changelog.d/`](changelog.d/README.md)**: `<slug>.<category>.md`, with category
`security`, `added`, `changed`, `fixed`, `removed`, `documentation` or `known-issues`, holding the entry exactly as
it would appear under that heading. Do not edit `CHANGELOG.md` in a pull request: with many pull requests open, every
merge made all the others conflict on its Unreleased section. While the major version is 0, start any breaking or
behavior-changing entry with **Breaking:** or **Behavior:**. `python scripts/changelog.py check` validates the
fragments and `python scripts/changelog.py render` shows the Unreleased section as it will be released. Purely
internal changes (refactors, test-only, CI) do not need an entry.

## Releasing (maintainers)
1. Run `python scripts/changelog.py release x.y.z`: it moves the `[Unreleased]` entries and every `changelog.d/` fragment
   under `## [x.y.z] - YYYY-MM-DD` and deletes the fragments. Then update the compare links at the bottom.
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
