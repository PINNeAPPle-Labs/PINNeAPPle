- Changelog entries now go in one new file per pull request under `changelog.d/` (`<slug>.<category>.md`) instead
  of the Unreleased section of `CHANGELOG.md`, so parallel pull requests no longer conflict on it.
  `python scripts/changelog.py check | render | release x.y.z` validates, previews and folds them in at release time.
