A part drafted by `pinneapple_llm` (text to CAD) is built with a real boolean or refused: without a boolean engine
(`manifold3d`, extra `geometry`) `build_mesh` replaced a cut by a concatenation, so a washer or a plate with a hole
came out with the hole's volume added (PINNeAPPle-apps#5). `build_mesh(..., strict_booleans=True)` raises instead of
falling back; `build_recipe`, `export_recipe` and `measure_part` use it. The generic factory keeps its warning
fallback by default.
