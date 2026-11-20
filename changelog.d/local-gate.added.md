`scripts/local_gate.sh`: the checks a reviewer reads from the checks tab, run locally. GitHub Actions has been
unavailable on this repository since 2026-11-05 (billing), so the gate that matters is the one that runs before the push:
`ruff` over what changed, `pytest` over the test files the change touches, and the repository's own structural checks,
with one line per stage so the output can be pasted into a pull request as evidence. `.githooks/pre-push` calls it, and
`git config core.hooksPath .githooks` makes that automatic; `--no-verify` skips it and `STRICT=1` also blocks on the
repository-wide checks, which read the whole tree and report a state the diff does not decide.
