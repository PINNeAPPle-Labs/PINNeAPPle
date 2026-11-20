# Licensing

What this repository promises about the code inside it, about code adapted from elsewhere, and about what it
depends on. The rule that can be checked by a machine is a script; the rest is a decision written down so a
contributor does not have to guess (issue #48).

## The repository

PINNeAPPle is distributed under the **Apache License 2.0** ([`LICENSE`](../LICENSE)). It is permissive, it grants a
patent licence, and it asks for the notices to be kept — which matters for a library that ships third-party material
next to its own.

## Contributions: inbound = outbound, no CLA

A contribution is accepted under Apache-2.0, the same terms the project ships under. There is **no Contributor
License Agreement** and **no required DCO sign-off**.

- **No CLA.** A CLA exists to let a project relicense contributions it received. This project has no such intention:
  it is Apache-2.0 today and the copyright stays with the contributors. Asking every contributor to sign away
  rights the project does not need is friction with no benefit.
- **No mandatory sign-off.** Apache-2.0 does not require a `Signed-off-by` line, and the repository's history does
  not carry one. Adding the requirement retroactively would fail every existing contribution. A sign-off is
  welcome and never required.

What a contribution does need is what the pull request template already asks: the change, its tests, a changelog
fragment when it is user-visible, and a licence that is compatible with the project's.

## Code adapted from elsewhere

- **GPL and AGPL code is never copied into this repository**, in whole or in part, not even into a module that is
  "not really used". The *idea* is not copyrightable and may be implemented from scratch; the *expression* is.
  This is the rule the pull request checklist states in one line.
- Every adapted file gets an entry in [`NOTICE`](../NOTICE) naming the source project, its copyright line and its
  licence. The entries are factual: file, source, licence. `docs/dev/rea_harvest/ledger.json` keeps the full
  provenance for the material harvested in 2026.
- Permissive licences (MIT, BSD-2/3, Apache-2.0, ISC, Zlib, MPL-2.0) are what this repository has accepted so far.
  A weak-copyleft licence (MPL-2.0) is acceptable for a dependency; a strong one (GPL, AGPL) is a decision that
  goes in this file first, never a silent addition.

## No per-file licence headers

There is one `LICENSE` at the root and one `NOTICE` for third-party material; individual files carry no banner.
A header repeated in two thousand modules drifts — one file ends up with a year that is wrong, another with a
holder that never existed — and it duplicates what the repository already states unambiguously. Where provenance
matters, it is in `NOTICE`, next to the file it names.

## Dependencies

The core's dependencies are declared in [`pyproject.toml`](../pyproject.toml). The policy is: **permissive by
default, copyleft as a decision**. `scripts/licensing/dependency_licenses.py` reports what every declared
dependency says about its own licence, and `--fail-on-copyleft` fails when one of them is GPL, AGPL, LGPL, SSPL or
EUPL:

```console
$ python scripts/licensing/dependency_licenses.py
numpy              BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0
scipy              BSD License
pandas             BSD License
pyyaml             MIT License
tqdm               MPL-2.0 AND MIT
typing-extensions  PSF-2.0
pydantic           MIT
matplotlib         Python Software Foundation License
torch              Apache-2.0 AND Apache-2.0 WITH LLVM-exception AND BSD-2-Clause AND BSD-3-Clause AND ... 
sympy              BSD License
xarray             Apache-2.0
zarr               MIT
```

The table above is what the command printed on 2026-11-20 with the development environment of that day; run it
again rather than trusting the copy. A dependency whose licence is not machine-readable is printed as
`unknown (see the distribution's licence file)` and checked by hand — it is never assumed permissive.

## Using this repository's code elsewhere

Apache-2.0 allows it, including commercially. Keep the `LICENSE` and `NOTICE` files with any substantial portion
you redistribute, and state what you changed. Nothing else is asked.
