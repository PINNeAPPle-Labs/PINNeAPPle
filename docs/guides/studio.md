# PINNeAPPle Studio: every tool, example and app in one place

`pinneapple_studio` turns the whole library into a catalog of tools and the repository's examples into a gallery you
can run. It is the library side of the PINNeAPPle Studio: the full web interface lives in the Engineering API and
reads this package; `pinneapple studio` serves a local version (the Flow Studio UI plus the Studio API).

```bash
pinneapple studio summary                          # tools per category and status, gallery per state
pinneapple studio search "dissipador de calor com aletas" --category executable
pinneapple studio tool preset:water_hammer_valve_closure
pinneapple studio gallery --state blocked          # what cannot run here, and the command that fixes it
pinneapple studio run-tool preset:water_hammer_valve_closure --inputs '{"t_close": 2.0}'
pinneapple studio run example:lbm                  # the quick configuration of the example's flow
pinneapple studio                                  # http://127.0.0.1:8765, API under /api/studio
```

## The catalog of tools

Every public class and function of every `pinneapple*` package is a tool (`pinneapple_studio.catalog.load()`), plus
the industrial presets (`preset:<name>`) and the science experiments (`experiment:<name>`). Nothing is skipped:
`tests/test_studio.py` checks that each public symbol gets a category. The category comes from the code, with the
reason stored next to it:

| category | rule | in the Studio |
|---|---|---|
| `executable` | every required input has a type a form can hold (numbers, text, booleans, lists and dicts of those, `Literal` choices, paths, numeric arrays as lists, dataclasses of those as nested forms); a class also needs a method that does the work (`run`, `solve`, `predict`, ...) | a card with a form generated from the signature, a Run button (a job in the sandbox) and the result with its evidence |
| `composite` | it takes or is an object other tools build (a tensor, a model, a problem, a callable, an input without a type annotation), or it is a layer, a loss, a callback or an adapter | a block of a pipeline: the card shows the examples and modules that use it |
| `reference` | exceptions, type definitions, entry points (CLI, servers), internal, deprecated or placeholder modules, plain data records | shown with its description and the reason it is not a tool |

Each tool carries:

- its signature, parameters with annotations and defaults, and the docstring;
- the Portuguese prose of the map, whole (`docs/pt/_mapa/prosa`);
- its status and the evidence behind it;
- the tests and the examples that use it;
- its tasks, physics areas and products;
- related tools from the import graph.

`prose_unmatched` lists prose whose symbol is no longer in the code. A symbol without prose shows none; nothing is
generated.

### Status

The rules are those of the Engineering API's Capability Registry:

| status | rule |
|---|---|
| `verified` | a test that runs by default calls it and compares with a reference (`validate()` asserted, a reference / exact / analytic solution, a relative error); for a preset, its recorded `validation.json` shows every reference case passing |
| `tested` | a test that runs by default calls it, without a reference comparison |
| `discovered` | no test that runs by default calls it (examples and opt-in slow tests do not count) |

### Navigation and search

`pinneapple_studio.search.Index(catalog)` runs BM25 over the name, module, docstring, prose and tags, in Portuguese or
English:

- accents are folded and names are split (`HeatSink`, `heat_sink`);
- `calor` also finds `heat`, from a fixed table plus the glossary's terms in parentheses.

Filters cover category, status, task, area, product, package and kind. The tasks, areas and products are regular
expressions in `pinneapple_studio.taxonomy`, and each tag keeps the words that matched.

## The gallery

`pinneapple_studio.gallery.load()` lists every example folder, every paper reproduction (`examples/reproductions/*`)
and every app. Each entry's data sheet comes from `experiment.json`, the example flows of
`pinneapple_flow.examples` (their measured runs and comparisons), the README and the prose.

The requirements are read from its scripts: packages with the `pyproject.toml` extra that installs them, external
programs (OpenFOAM, FreeCAD, CalculiX, ParaView, Blender, Ollama), a GPU, the network and credentials. They are
checked on this machine. An entry that cannot run is `blocked`, with each missing piece and its fix. A cost is shown
only when a run was measured.

## Running

`pinneapple_studio.runner.queue()` is a `pinneapple_core.service.jobs.JobQueue` with two job kinds, `tool` and
`gallery`. Each job runs in its own process:

- with an environment from an allowlist, without tokens, keys, passwords or database URLs;
- under memory, CPU and open-file limits;
- with a time limit, logs, progress and cancel.

A gallery job runs the example's flow, quick by default, and compares it with the committed measurement. It never
writes to `measured.json`. It can also run one of the example's scripts in its own process.

Every result carries an evidence record: parameters, versions, wall time, files with SHA-256, and verifications that
are each `pass`, `fail`, `inconclusive` or `not_run`. The comparisons of an example flow and the checks of a preset or
an experiment become those verifications. A tool that just returned a value has no verification, so its outcome is
`not_run`.

## The API

`pinneapple_studio.api.router(queue)` is a FastAPI router. The app that mounts it brings the security: login, API
keys and rate limits.

| method | path | |
|---|---|---|
| GET | `/summary` | counts, labels of every navigation, facets, gallery per state |
| GET | `/tools?q=&category=&status=&task=&area=&product=&package=&kind=` | search with facets of the hits |
| GET | `/tools/{id}`, `/compare?ids=a,b`, `/tree`, `/modules/{module}`, `/glossary` | cards, comparison, the package tree |
| GET | `/gallery?kind=&state=&q=`, `/gallery/{id}` | entries and data sheets |
| POST | `/jobs` (`{"kind": "tool" \| "gallery", "params": {...}}`) | `202` and the job id; a composite tool or a blocked entry is refused with the reason |
| GET / POST | `/jobs/{id}`, `/jobs/{id}/logs`, `/jobs/{id}/cancel`, `/jobs/{id}/files/{name}` | status and result, logs, cancel, files |

The inventory, catalog and gallery are generated per commit under `~/.cache/pinneapple/studio`
(`PINNEAPPLE_STUDIO_CACHE`) and never committed. `PINNEAPPLE_ROOT` points the Studio at a checkout when the package is
installed from a wheel.
