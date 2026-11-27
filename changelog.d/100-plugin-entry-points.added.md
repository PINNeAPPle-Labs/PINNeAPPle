- **Plugins from installed packages** (`pinneapple.plugins` entry point, K1/#100): a third-party package implements
  the `pinneapple_plugins.Plugin` contract (name, version, contract version, requirements, capabilities, permissions,
  `initialize`, `shutdown`) and registers a solver, model, benchmark or adapter inside `initialize` — it shows up in
  `pp.list_models()` and in the benchmark with no change to this repository. A plugin that fails to load is a record,
  never an exception: `import pinneapple` and every other plugin keep working.
