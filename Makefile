# One command per job, so a contributor does not have to read the CI workflows to work locally.
#
#   make setup        create .venv and install the package with its test extras
#   make test         the fast suite: what has to pass before opening a pull request
#   make e2e          the blocking end-to-end tier (#37): ten flows, input to kept artifact
#   make test-full    everything, the same selection the full-suite workflow runs
#   make lint         the ruff selection CI enforces (syntax-level, expected to pass)
#   make lint-backlog the wider selection declared in pyproject.toml (a backlog, not a gate)
#   make gates        the repository's own checks: changelog, module reference, licensing
#   make docs         regenerate the module reference and build the documentation site
#   make clean        remove build and cache output (keeps .venv)
#
# The Python is a variable so a contributor (or CI) can point at another interpreter:
#     make test PY=/usr/bin/python3
#
# Measured on an M-series laptop on 2026-11-21 with the development environment installed:
# `make test` 17-29 s (127 tests, depending on what the import cache holds), `make lint` 2 s, `make gates` 5 s. The full suite is minutes, which is why it is a
# separate target.

PY ?= .venv/bin/python
PIP ?= $(PY) -m pip
# Point this at the CPU wheel on a machine without a GPU, or in a container, to avoid the CUDA build of torch:
#     make setup TORCH_INDEX=https://download.pytorch.org/whl/cpu
TORCH_INDEX ?=

# The fast suite: the tests that answer "is this checkout working" -- the public surface, the front door, the
# documentation that is executed, the changelog and licensing rules, the module reference scripts. Anything that
# trains a network, renders, or needs an external solver is deliberately not here; it belongs to test-full.
FAST = tests/test_public_api.py tests/test_front_door.py tests/test_onboarding.py \
       tests/test_agents_md.py tests/test_changelog_fragments.py tests/test_licensing_policy.py \
       tests/test_stability.py tests/test_solve_guide.py tests/test_dev_environment.py \
       tests/test_turntable.py

.PHONY: help setup test e2e test-full lint lint-backlog gates docs clean

help:
	@grep -E '^[a-z-]+:' $(MAKEFILE_LIST) | grep -v '^\.' | awk -F: '{printf "  make %-13s\n", $$1}'

setup:
	python3 -m venv .venv
	$(PIP) install --upgrade pip
	$(if $(TORCH_INDEX),$(PIP) install "torch>=2.2" --index-url $(TORCH_INDEX),)
	$(PIP) install -e ".[dev]"
	@echo "Ready. Activate with: source .venv/bin/activate"

test:
	$(PY) -m pytest $(FAST) -q -p no:warnings

# The blocking end-to-end tier (#37): ten flows from an input a user has to an artifact they keep, through the
# public API only. Each flow carries its own wall-clock budget and the ten budgets sum to under ten minutes, which
# is what makes this a tier that can block a merge. Measured at about 5 s together on an M-series laptop.
e2e:
	$(PY) -m pytest tests/e2e -q -p no:warnings

test-full:
	$(PY) -m pytest tests/ -q --continue-on-collection-errors -p no:warnings

# The same selection .github/workflows/tests.yml runs. E9 is what ruff reports when the file does not parse
# (syntax error, undefined name in a comprehension scope), which is worth failing on and passes today.
lint:
	$(PY) -m ruff check . --select E9 --target-version py310 --no-cache --output-format concise

# [tool.ruff.lint] in pyproject.toml selects E, F, I, B and UP. The tree does not satisfy that yet -- about
# 21,000 findings, mostly UP006/UP045 (typing to builtin generics) and I001 (import order) -- so this prints the
# backlog instead of pretending to be a gate. Burn it down a rule at a time, then move the selection into `lint`.
lint-backlog:
	@$(PY) -m ruff check . --no-cache --statistics | head -30
	@echo "Backlog, not a gate: CI runs 'make lint'."

gates:
	$(PY) scripts/changelog.py check
	$(PY) scripts/reference/build.py --check
	$(PY) scripts/licensing/dependency_licenses.py --fail-on-copyleft

docs:
	@$(PY) -c "import mkdocs" 2>/dev/null || \
	  { echo "mkdocs is not installed: $(PIP) install -r requirements-docs.txt"; exit 1; }
	$(PY) scripts/reference/build.py
	$(PY) -m mkdocs build --clean

clean:
	rm -rf .pytest_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
