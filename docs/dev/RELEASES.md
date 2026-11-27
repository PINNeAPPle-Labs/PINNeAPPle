# Cutting a release: the preflight, and the state of the three milestones

The three release epics of 2026 — [#421](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/421) for **0.8.0**,
[#422](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/422) for **0.9.0** and
[#423](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/423) for **1.0.0** — each end with the same exit criterion:
*every item closed, or explicitly moved to a later milestone with the reason written in the item*, plus a CHANGELOG
whose `[Unreleased]` section has become the dated version, docs that build, and a regenerated catalogue.

Most of that is a human decision. What is mechanical is the rest, and it is what `scripts/release_preflight.py` checks:

```bash
python scripts/release_preflight.py 0.9.0              # the local gates
python scripts/release_preflight.py 0.9.0 --github     # + the milestone's open items (needs gh authenticated)
python scripts/release_preflight.py --all --github     # the three at once, in a table
```

It exits 1 when anything blocks, so it can be a gate; `--json` gives the same report for a script. It reports **every**
blocker at once, which is the difference from `scripts/check_release_version.py` — that one stops at the version
disagreement, which is the smallest of the problems.

## The five things it checks

| gate | what it reads | what blocks |
|---|---|---|
| `version` | `pyproject.toml`, `CITATION.cff`, `pinneapple/__init__.py` | the three files disagree, or do not match the version asked for |
| `changelog` | the `[Unreleased]` body | anything still sitting there |
| `changelog` | `changelog.d/*.md` | fragments that were never folded — `scripts/changelog.py release <v>` folds them |
| `changelog` | the dated heading `## [<v>] - YYYY-MM-DD` | there is none |
| `milestone` | the GitHub milestone, by **title** (`v0.8.0`) | open items — each one has to be closed with evidence or moved with the reason written in it |

The milestone is looked up by title and not by number on purpose: the numbers are stable only by accident, and a
preflight that reads the wrong milestone is worse than one that reads none. When `gh` is missing or unauthenticated
the run says so in a note and **does not** pass or fail on the milestone — a gate that invents a blocker from a
network failure is a gate people learn to ignore.

## The state on 2026-11-27

Measured with `python scripts/release_preflight.py --all --github`:

| versão | milestone | versão nos arquivos | CHANGELOG datado | fragmentos por dobrar | bloqueios |
|---|---|---|---|---|---|
| 0.8.0 | 10 abertos / 24 fechados | 0.8.0 | **sim**, 2026-11-01 | 241 | 2 |
| 0.9.0 | 9 abertos / 14 fechados | 0.8.0 | não | 241 | 4 |
| 1.0.0 | 8 abertos / 8 fechados | 0.8.0 | não | 241 | 4 |

Two facts that table makes plain, and that neither issue says:

1. **0.8.0 is dated in the CHANGELOG (2026-11-01) and its milestone still has ten open items.** The release shipped
   ahead of its own exit criterion. Whether that was right is a product decision, but the epic cannot be closed by
   moving the milestone quietly — the criterion it wrote for itself says each item is closed or moved *with the
   reason written in the item*.
2. **241 changelog fragments have accumulated since the last fold.** They are the real content of the next release
   notes, and until `changelog.py release` runs they are invisible to anyone reading `CHANGELOG.md`.

## What the preflight does not do

- It does not tag, publish, or touch the milestone. Those are the release workflow's job
  (`scripts/check_release_version.py` guards the tag itself) and a person's decision.
- It does not check that the docs build or that the catalogue was regenerated: both are CI jobs with their own
  commands, and a local copy of them would be a second implementation to keep in step.
- It does not decide that an item may be moved. It only reports that it is still open.
