- Presets carry their data sheet in English as well as in Portuguese (`title_en`, `sector_en`, `problem_en`,
  `question_en`, each parameter's `help_en`, and a scene's `labels_en`), so the Studio catalog is searched in both
  languages and a 3-D field view shows its labels in the language of the page: `pinneapple_presets.i18n` with
  `english()` applied by `pinneapple_presets.load`, `Scene.labels_en`, and `pinneapple_tools.visualization.studio
  .fieldview.from_scene(scene, lang=...)` to mount a library-built scene (a preset's own `scene(result)`) as a field
  view with its labels, units and colour ranges.
