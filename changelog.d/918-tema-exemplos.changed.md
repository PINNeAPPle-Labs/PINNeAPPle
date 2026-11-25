- Fourth batch of the house theme (#918): 38 of the 102 example scripts that still drew with matplotlib
  defaults now call `use_theme("report")`, each one run after the edit, so `examples/.house_theme_baseline` went from
  102 lines to 64. Every line that remains is an example that already failed on `main` for a reason of its own — a
  missing optional package, a module the repository does not carry any more (`pinneapple_digital_twin`,
  `pinneapple_timeseries`), an external solver (getdp, OpenModelica, OpenFOAM), inputs the script is given rather than
  born with, or a defect in the example — or one whose training outlasts the window the batch had. **Behavior:** the
  figures of those 38 scripts now come out in the house palette, font stack and perceptually uniform maps, and
  two `plt.rcParams.update` calls that only restated the theme are gone (`electrodynamics/24_afpm_quasi3d`,
  `25_afpm_surrogates`). `scripts/apply_house_theme.py --check` fails on a new figure drawn with matplotlib defaults.
