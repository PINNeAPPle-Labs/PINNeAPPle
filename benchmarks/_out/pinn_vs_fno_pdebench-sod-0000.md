# PINN vs FNO on `pdebench-sod-0000` (field `density`)

- split: 161 train / 40 test time steps (last 20% held out), 150 epochs, lr 0.001, seed 0
- both models get the same data, optimiser budget and seed; a CPU-sized budget, not a paper reproduction

| model | parameters | rel-L2 (held out) | wall time |
|---|---:|---:|---:|
| pinn | 8,577 | 1.4847e-02 | 69.8 s |
| fno | 15,721 | 4.0215e-03 | 390.7 s |

Leader: **fno** (lowest held-out relative L2, tolerance 5.0e-02).

## Verification gates

| gate | status | value | detail |
|---|---|---:|---|
| integrity | passed | 0.0 | the data bundle's own digest |
| units | passed | 0.0 | every declaration is present and consistent (the time axis is dimensionless (a non-dimensionalised case)) (0.00s) |
| location | passed | 0.0 | every field declares a location consistent with the mesh (0.00s) |
| frame | passed | 1.0967254638651058e-05 | axes strictly monotonic and uniformly spaced within 1.2e-04 (tolerance 1.0e-05 or the dtype's own rounding, whichever is larger): 't' is float32, which alone accounts for 4.8e-05, 'x' is float32, which alone accounts for 1.2e-04 (0.00s) |
| provenance | passed | 0.0 | source, licence (CC BY 4.0), citation and 1 input hash(es) recorded (0.00s) |
| reference | passed | 0.004021643262088804 | relative L2 error of 'density' against the reference (0.00s) |

Leader fno on 40 held-out time steps; tolerance 5.0e-02 -> within. Reduced CPU budget: this compares the two families on this data, not against published numbers.
