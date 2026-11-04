# Experiment campaign, November 2026

Archive of the 79-issue experiment campaign opened in this repository: the backlogs the issues were derived from,
the writing specification, the drafted bodies exactly as they were created, and the tooling that created them.

Start with [CONTINUE.md](CONTINUE.md) — it holds the status, the re-check command and the procedure for extending
the campaign.

| path | what it is |
| --- | --- |
| `CONTINUE.md` | status, how to re-check, how to add an issue, facts not worth re-deriving |
| `SPEC.md` | the 17 mandatory `##` headings, the hard rules and the verified module inventory |
| `BACKLOG-A-B.md`, `BACKLOG-C.md`, `BACKLOG-D.md` | the concept notes, one section per issue, with its title and labels |
| `<ID>.md` | the issue body as created, one per entry of `manifest.tsv` |
| `manifest.tsv` | `id`, `title`, `labels` parsed from the backlogs |
| `created.tsv` | `id`, issue number and URL for every issue opened |
| `build_manifest.py` | rebuilds the manifest and checks the bodies (headings, word count, forbidden strings) |
| `create_issues.py` | creates the issues idempotently, prepending `Part of #1288.` to every body |

The campaign index — every issue number by series — is the body of [#1288](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1288).

The bodies are specifications for experiments that have not been run. Numbers in them are targets, acceptance
thresholds and compute budgets, not measurements.
