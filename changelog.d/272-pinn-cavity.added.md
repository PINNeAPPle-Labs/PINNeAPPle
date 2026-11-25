- `pinneapple_simulation.cfd_validation.pinn`: a physics-informed network for the lid-driven cavity at Re = 100, 400
  and 1000, and one operator whose input carries `Re`, measured against Ghia (1982) at the same 17 stations and
  through the same relative L2 the OpenFOAM mesh study in `examples/cfd/classical_cases` uses, so the two numbers sit
  in one table. `results/cavity_pinn.json` carries the training budget, the machine, the profiles and the Reynolds
  number at which each model stops matching; the runner is `run_cases.py --only pinn`.
