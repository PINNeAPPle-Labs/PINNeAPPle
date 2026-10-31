# Changelog fragments

Every pull request with a user-visible change adds **one new file** here instead of editing `CHANGELOG.md`, so
parallel pull requests never conflict on the changelog.

- Name: `<slug>.<category>.md`, for example `potential-core.added.md` or `973.fixed.md`. The slug only has to be
  unique; the branch name works.
- Category: `security`, `added`, `changed`, `fixed`, `removed`, `documentation` or `known-issues`.
- Body: the entry exactly as it would appear under that heading in `CHANGELOG.md`, usually one `- ` bullet.
  While the major version is 0, start a breaking or behavior-changing entry with **Breaking:** or **Behavior:**.

`python scripts/changelog.py check` validates the names (CI runs it through `tests/test_changelog_fragments.py`),
`render` shows the Unreleased section as it will be released, and maintainers run `release x.y.z` when cutting a
release (see CONTRIBUTING.md).
