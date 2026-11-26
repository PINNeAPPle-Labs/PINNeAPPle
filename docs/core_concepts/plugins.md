# Plugins: what installing one does, and what it does not guarantee

A plugin is an installed package that adds something to PINNeAPPle without a change here: a solver, a model, a
benchmark, an adapter. It declares the `pinneapple.plugins` entry point in its own `pyproject.toml`:

```toml
[project.entry-points."pinneapple.plugins"]
openfoam = "pinneapple_openfoam:PLUGIN"
```

and the value resolves to a `pinneapple_plugins.Plugin` (the contract is `pinneapple_plugins.contract`, importable
without the rest of the library). `import pinneapple` activates the installed, enabled plugins once per process;
`pp.plugins` is the question-answering side of the same thing:

```bash
pinneapple plugins list                     # every discovered plugin: name, version, source, contract, state
pinneapple plugins inspect openfoam         # capabilities, declared dependencies, entry point, where it came from
pinneapple plugins doctor                   # per plugin: dependencies present, contract compatible, what it registered
pinneapple plugins policy                   # what is checked, and what is deliberately not provided
pinneapple plugins disable openfoam         # a configuration file, not an uninstall
```

The policy below is `pinneapple_plugins.policy`, printed verbatim by `pinneapple plugins policy` and checked
sentence for sentence by `tests/test_plugins.py`: the page and the command cannot drift apart.

> Loading a plugin checks that its versions, dependencies and registrations match what it declares and records where it
> came from; it does not sandbox it — the plugin runs with the privileges of whoever imports it.

## Checked at load time, every time

- declared dependencies, resolved against the environment: the `pinneapple` version specifier, the other plugins it
  needs, the Python modules that must import, the executables that must be on PATH, and the configuration keys it
  requires
- the contract version it was written against, refused by name with both versions before `initialize()` runs
- where the plugin came from — distribution name, version and entry-point value — recorded, and written into the
  provenance of every run it takes part in
- the registry entries it adds when `initialize()` runs, against the `registers` it declares: a plugin that declares
  `registers` and then registers something it did not declare, or declares something it did not register, is refused
  and its registrations are rolled back; a plugin that declares nothing has what it registers recorded and shown by
  `pinneapple plugins doctor`, and not verified
- the permissions it declares (`read`, `run`, `write`, `network`) — recorded and shown by `pinneapple plugins doctor`,
  never enforced

The fourth one is the check worth explaining. The dangerous plugin is not the hostile one: it is the honest one
written against last year's API, which loads, registers half of what it promised and looks successful. So a plugin
declares what it registers —

```python
from pinneapple_plugins import Plugin, PluginContext

class OpenFOAM(Plugin):
    name = "openfoam"
    version = "0.3.1"
    contract_version = "1.1"
    capabilities = {"solver": ["cfd", "steady_state"]}      # what it can do, its own vocabulary, not verified
    registers = {"solver": ["openfoam_case"]}               # what initialize adds through pp.register, verified
    requires = {"executables": ["blockMesh"], "pinneapple": ">=0.8,<0.9"}

    def initialize(self, context: PluginContext) -> None:
        import pinneapple as pp
        pp.register(openfoam_case, kind="solver", name="openfoam_case")
```

— and the loader watches the registry while `initialize` runs. A mismatch refuses the plugin with both lists in the
message, and everything it registered is removed again:

```
plugin openfoam is broken at registers: declared in `registers` but not registered: solver=['erosion_case']. The
plugin was refused and its registrations were rolled back: a plugin that does half of what it declares loads and looks
successful, which is worse than one that refuses to load. ...
```

A plugin written against contract `1.0` stays compatible and loads; what it registers is recorded and shown by
`doctor`, and not verified, because it declared nothing to check it against. Bump `contract_version` when you adopt
`registers` (or when the API you build against changes: that is what the version is for).

## Not provided, by design

- process isolation, resource limits or syscall filtering: a plugin runs in this process, with the privileges of
  whoever imports it
- a sandbox of any kind, including for a plugin started in a subprocess — a subprocess is not a sandbox either
- enforcement of the declared permissions: they are recorded, never applied, so a plugin that declares `read` can still
  write and a plugin that declares neither can still do both
- revocation: an installed plugin stays importable until it is disabled or uninstalled, and what a running plugin
  already registered in the library stays registered until the process ends
- protection from a plugin that passes every check: the checks are about declarations that match reality, not about
  what the code does after `initialize()` returns

A deployment that needs isolation needs a process or a container: run the plugin there and hand results across the
boundary. Nothing in this loader substitutes for that, and nothing here claims to.

## Recorded, so a result can be attributed to the code that produced it

- every plugin that took part in a run, with its name, its version, the contract version it was written against, the
  distribution and version it came from, and the entry-point value that was loaded
- what each plugin registered, and what it declared but did not register
- the plugins that failed to load, at which check, and why — a failure is a record, never an exception

`pp.plugins.provenance()` is that record for the current process, and the simulation workflow runner writes it into a
run's provenance (`pinneapple_core.workflows`), so a result carries the plugin set that produced it next to the git
commit and the environment.

## Related

- [Add something to PINNeAPPle: six extension points](../dev/adding_to_pinneapple.md) — when a plugin is the right way
  to add something, and when it is not.
- [One registry for every extensible object type](registry.md) — the protocol a plugin registers through.
