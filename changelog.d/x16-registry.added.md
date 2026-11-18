- `pinneapple_core.registry`: one registry protocol for the nine extensible object types — `physics` (problem presets),
  `model`, `solver`, `operator`, `dataset`, `benchmark`, `geometry`, `material` and `optimizer`. `pp.register(obj,
  kind=...)` (also a decorator), `pp.registered(kind)`, `pp.resolve(kind, name)` and `pp.unregister(kind, name)` are
  the same calls for every kind, and each kind declares once what it accepts, so a wrong type fails at registration
  with `WrongType` naming what the kind takes instead of failing later inside a solve. A registered solver reaches
  `pp.solve`, a preset reaches `pp.get_preset`, a model reaches `pp.list_models`, a dataset reaches `DatasetRegistry`
  and a material reaches `MATERIALS`; the kinds that had no registry of their own live here. Inference of `kind` is
  conservative: a plain callable raises `AmbiguousKind` instead of being filed as a solver when it could be an operator.
