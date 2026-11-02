`pinneapple_llm.check_preset_coherence(description, spec)`: does the chosen preset solve the problem described? It
compares the physics classes the description names (heat, diffusion, waves, fluid flow, structures, electrostatics,
electromagnetism, orbits, epidemics, finance, ...; English and Portuguese) with the preset's `pde.kind`, and checks
the regime (steady or transient) and an explicit dimension. `PhysicsGuardrail` re-checks the physics of the preset it
is given, never whether it is the right one; this check is meant to run before training. A description it cannot
classify is reported as `"unknown"`, never as confirmed.
