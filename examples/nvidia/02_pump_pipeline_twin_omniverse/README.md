# 02 Pump and pipeline twin: PINNeAPPle + Omniverse (OpenUSD)

**What.** A fixed-speed pump (`H = 70 - 120 Q|Q|` m) lifts water through a 2 km DN600 steel main
(`a = 1150 m/s`) into a tank 20 m up; the valve at the end closes in 0, 6, 20 or 60 s. Water hammer by the method of
characteristics, rated against the PN16 pipe class.

**Why.** A digital twin is only as good as the solver behind its colours. PINNeAPPle verifies the transient solver
(Joukowsky head rise, wave period, steady state) before rating the closures, and the OpenUSD stage carries the
pressure as data keyed over time, so the Omniverse timeline plays the surge and every sensor has its own series.
Omniverse only: no PhysicsNeMo.

| Step | Who | Ran here |
|---|---|---|
| operating point (pump curve = Colebrook friction + valve), MOC transient, verification, PN16 rating | PINNeAPPle (`pinneapple_systems.process_components.water_hammer`) | yes |
| USD twin: pipe mesh with `pressure`/`head` time samples, pump, valve, 3 sensors, trust | OpenUSD (`usd-core`) | yes, validated |
| timeline playback / RTX in Omniverse Kit, live sensor feed through Nucleus | Omniverse | **no** (needs Omniverse and an RTX GPU) |

```bash
pip install usd-core
python examples/nvidia/02_pump_pipeline_twin_omniverse/run.py      # ~10 s
```

## Results (this run: 4-core Xeon 2.1 GHz, usd-core 26.11)

- Operating point: Q = 0.576 m^3/s, V = 2.04 m/s, Re = 1.22e6, f = 0.0128 (Colebrook), friction loss 9.06 m.
- Verification: Joukowsky head rise reproduced to < 1e-6 relative, period 4L/a = 6.96 s to < 1 %, steady state
  held to < 1e-8 m for 20 s, Courant number exactly 1.
- Max gauge pressure: 26.3 bar (instant), 25.5 bar (6 s), 23.1 bar (20 s), **14.7 bar (60 s)**; PN16 = 16 bar.
- Min gauge pressure: -8.1, -7.5, -6.1 bar for the three fast closures: below vapour pressure, so the water column
  separates and the single-phase model no longer holds there (the check fails and says so); -0.12 bar for 60 s.
- **Decision: APPROVED for the 60 s closure only**; the faster closures fail PN16 and column separation.
- USD: one 90-sample timeline on the pipe and the sensors, all validators clean.

Figures in `results/`: `architecture.png`, `transients.png` (valve pressure, envelope along the pipe, Joukowsky
verification), `checks.png`, `usd_preview.png` (stage read back at the pressure peak), `usd_timeline.gif` (the
time samples read back frame by frame). `results/kit_render.py` renders the stage in Omniverse Kit.
