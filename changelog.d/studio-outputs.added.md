- Every Studio tool records its outputs (`outputs`: name, type, unit and description of each value) and where they were
  read (`outputs_source`: `declared` from a module's `X_OUTPUTS` or a preset card, `annotation` from a `NamedTuple` /
  dataclass / plain return type, `docstring` from a numpy or Google `Returns` section with the unit in brackets, or
  `none`).
