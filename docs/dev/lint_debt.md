# Lint debt: recorded, and only going down

The ruff configuration in `pyproject.toml` (`E`, `F`, `I`, `B`, `UP`, `E501` ignored) describes the code the
repository wants. It is not the code it has: the tree carries **21 339 findings across 1 781 files**, almost all of
them older than the configuration that names them.

Two different problems were being answered with one tool, and neither well:

- **New debt.** A pull request that adds an unused import must fail. `.pre-commit-config.yaml` runs ruff (with
  `--fix`) on the files a commit touches — that is the half that works today.
- **Old debt.** The same command, run over the whole tree, fails on the first file and says nothing about whether the
  situation is improving. That half is `scripts/lint_debt.py`.

## The baseline

`docs/dev/lint_debt.json` is a count per **(file, rule)**, plus the totals by rule:

```bash
python scripts/lint_debt.py --record    # write the baseline from the current tree
python scripts/lint_debt.py --check     # the gate: fail on any pair above its count
python scripts/lint_debt.py --summary   # the debt by rule, and how much is ready to ratchet
```

Counting per pair, and not by total, is the whole design. A total lets one module pay for another: remove five `UP006`
in `analysis/` and add five in `data/`, and the number does not move — but the debt does. With per-pair counts the
gate fails on the second module, and `--check` still reports the first as a ratchet waiting to be recorded.

## Recording is a decision

`--record` writes the tree as it is. That is only honest when the author can say **why** the number moved: a batch of
fixes (down), or a new module that arrived with debt on purpose (up). Like the render baseline of the Studio
benchmark, the reason belongs in the commit message, not in the file — the file holds the number.

## What the gate does not promise

- It does not promise the tree is clean, and it will not be for a long time: 8 135 of the findings are `UP006`
  (annotations written `List[int]` instead of `list[int]`), a mechanical fix that touches thousands of signatures and
  is worth its own batch per package.
- It does not run per-file counts across a rename: a moved file shows up as one removal and one addition, and the
  addition fails until the baseline is recorded again. That is deliberate — a rename that carries debt into a file
  with a clean baseline should be seen.
- It does not replace `ruff check . --select E9` as the per-pull-request gate. That one is fast and catches the errors
  that break at import time; this one is the ledger.

## The order the debt is paid in

Suggested, from the cheapest to the most invasive, each step ending with a `--record` and a smaller baseline:

1. `F401` (unused import) and `I001` (unsorted import) — automatic with `--fix`, safe in bulk.
2. `UP035`/`UP037`/`UP045` (deprecated imports, quoted annotations, `Optional[X]`) — automatic with `--unsafe-fixes`
   reviewed file by file.
3. `UP006` (`List[int]` → `list[int]`) — automatic, but it touches the public signatures of every package; one package
   per pull request, with the API reference regenerated.
4. `E402`, `E702`, `E701`, `E741`, `B905`, `B008`, `B904`, `F841`, `F821` — one by one, because each one is a
   judgement about the code and not a formatting choice.
