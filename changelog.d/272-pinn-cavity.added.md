- `pinneapple_simulation.cfd_validation.pinn`: a physics-informed network for the lid-driven cavity at Re = 100, 400
  and 1000, and one operator whose input carries `Re`, measured against Ghia (1982) at the same 17 stations and
  through the same relative L2 the OpenFOAM mesh study in `examples/cfd/classical_cases` uses, so the two numbers sit
  in one table. `results/cavity_pinn.json` carries the training budget, the machine, the profiles and the Reynolds
  number at which each model stops matching; the runner is `run_cases.py --only pinn`.
- `pinneapple_simulation.cfd_validation.pinn.against`: the operator asked at a Reynolds number no column of the table
  has (`pinn.OFF_SAMPLE`, 250 and 750) and scored against `simpleFoam` at the same stations — an operator measured only
  on the three columns it was asked to reproduce has not been shown to be one. The record keeps the mesh ladder the
  reference was made on.
- `pinneapple_simulation.cfd_validation.cfdbench`: the CFDBench cavity family (Luo, Chen and Zhang, 2023) read from
  its own HuggingFace archive through `pinneapple_data.datasets.cfdbench`, sampled at Ghia's stations and scored with
  the same ruler, so a published dataset's numbers can be read in the column ours are in. `results/cfdbench.json`
  holds the rows; the runner is `run_cases.py --only cfdbench --cfdbench <unpacked archive>`.
- `pinneapple_simulation.cfd_validation.cavity.sample` takes an optional `stations`, which is how a solution at a
  Reynolds number the table does not publish is read at the stations it does.
