- `pinneapple_core.artifact`: the `.ppx` container — one file with a result and everything needed to reproduce it. It
  holds a JSON manifest (problem, units, solver, config, metrics, validation, datasets, environment, provenance), the
  arrays it names (`arrays/*.npy`) and the model weights (`model/state.pt`), with the sha256 of every member in the
  manifest. `pp.save` writes it from an `Artifact`, a `Solution` or a torch model; `pp.load` refuses a member that
  does not match its hash (`IntegrityError`) and, by default, a file whose provenance is absent or incomplete
  (`MissingProvenance`); `pp.verify_artifact` reports the container, the integrity, the provenance, the sections, the
  units and the reproduction. A model is rebuilt from the class and keyword arguments recorded next to its weights and
  compared to the probe outputs stored with it; a solved problem reproduces through the callable that produced it,
  since the classical backends take a source function no container can carry.
