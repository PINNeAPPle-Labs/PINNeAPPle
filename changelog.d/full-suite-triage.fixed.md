Found by the nightly full suite (#992):

- The Arena builds X-TFC. It passed `NetworkConfig`'s default activation ("tanh") to X-TFC, whose random Fourier
  features take none and refuse one. `extreme_tfc` is an alias of `xtfc`.
- `FileRef.of` accepts a relative path under an absolute root (or the reverse), as a flow's run folder gives it.
- The Ahmed-body flow runs end to end on a fresh checkout: `source="recorded"` reads the committed campaign
  (`results/cfd_runs`), the report works on a copy instead of rewriting the committed study, and the surrogate
  selection rule is registered with its references (Stone 1974; Vovk 2015; Barber et al. 2021).
- `lab/curation.json` is regenerated: the new `freecad_openfoam` example is curated, and three electromagnetism
  experiments, now documented by their guides, reach tier A.
