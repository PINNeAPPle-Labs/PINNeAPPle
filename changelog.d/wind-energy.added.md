- `pinneapple_systems.wind`: a wind-energy vertical built on public SCADA — the Penmanshiel loader with an
  evidence-based operating mask (recorded availability, downtime and curtailment instead of heuristics), IEC
  61400-12-1 density correction and binned power curves, a logarithmic-free **physics-informed monotone power
  model** whose constraints (non-negative, non-decreasing, capped at rated, zero outside cut-in/cut-out) hold by
  construction and are re-measured by `physical_guardrails`, the **Jensen/PARK and Gaussian (Bastankhah &
  Porté-Agel) wake models** with RSS superposition and a farm model, a farm-efficiency observation from SCADA and
  a sector-by-sector comparison against the wake models, a wake-decay calibration that fits on a validation window
  and scores on a test window, an operational-curtailment study, a farm-efficiency surrogate, and the figures for
  all of it. Guide: `docs/guides/wind_power.md`; runnable case: `examples/wind_farm_penmanshiel`.
  Check the fixed `farm_power` convention: `thrust_scale` now scales the derated machine's own power as well as its
  thrust coefficient, which removes a modelled curtailment gain of 10-16 % that came from producing full power with
  no wake.
- `pinneapple_systems.wind.twin` and `pinneapple_systems.wind.video`: the farm as a `pinneapple_twin3d.Scene`
  — plan-view wake footprint on a ground grid, swept over a full wind-direction cycle, with towers, rotor discs and
  one power sensor per turbine — exported to the bundled web viewer, and the same scene rasterised frame by frame
  into an H.264 demo film (`examples/wind_farm_penmanshiel/outputs/wind_farm_demo.mp4`).
- Two demonstration notebooks with their outputs committed,
  `examples/wind_farm_penmanshiel/notebooks/01_turbine_power_model.ipynb` and `02_wake_models.ipynb`: the same
  pipeline as the runnable case, cell by cell, from the operating mask through the guardrail checks and the
  applicability map to the SCADA-versus-wake-model comparison and one frame of the 3D twin.
  `figures.plot_wake_polar` now leaves an empty direction sector as NaN instead of emitting one
  "Mean of empty slice" warning per sector per call.
