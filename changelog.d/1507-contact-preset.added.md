- Contact with friction becomes a preset the front door can list, not only a library path: `pp.presets.load(
  "bolted_joint_preload")` is a preloaded bolted joint whose deck the CalculiX `ccx` solves — the preload enters
  through a `*PRE-TENSION SECTION` and the head and nut reach the plates through a `*DISTRIBUTING COUPLING` over the
  washer footprint — and whose answer is the joint's slip capacity, `mu` times the clamp force *measured* under the
  head instead of the one that was asked for. At its defaults (two 10 x 4 x 10 mm steel plates, 3 mm footprint,
  5,1 kN of preload, mu = 0,30) the run clamps at 5,06 kN (−0,9 %), slips at 1,53 kN against `mu*N` = 1,53 kN, keeps
  peak/mean contact pressure under the head at 1,34 (1,63 with the single-node bolt this case used to be), closes
  the Coulomb cone at `|tau|/p` = 0,3000 and does not open; the mesh study over 4, 6 and 8 cells moves the slip load
  by 0,02 %. `validate()` compares the clamp with the preload, the slip load with `mu*N` at two friction coefficients
  and the cone ratio with `mu`, and writes `examples/industrial_presets/bolted_joint_preload/results/validation.json`
  from those runs. A preset may now declare the external binaries it solves with (`needs`), which `card()` publishes:
  without a `ccx` the deck is still written, the trust report says nothing was solved and `validate()` returns the
  reason instead of an empty pass, and `tests/test_presets.py` skips the run rather than failing on the environment.
