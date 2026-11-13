# PINN vs FNO on `pdebench-sod-0000` (field `density`)

- split: 161 train / 40 test time steps (last 20% held out), 20 epochs, lr 0.001, seed 0
- both models get the same data, optimiser budget and seed; a CPU-sized budget, not a paper reproduction

| model | parameters | rel-L2 (held out) | wall time |
|---|---:|---:|---:|
| pinn | 8,577 | 6.4249e-02 | 11.7 s |
| fno | 4,705 | 2.4845e-02 | 19.2 s |

Leader: **fno** (lowest held-out relative L2, tolerance 5.0e-02).

## Verification gates

| gate | status | value | detail |
|---|---|---:|---|
| integrity | passed | 0.0 | the data bundle's own digest |
| units | passed | 0.0 | every declaration is present and consistent (the time axis is dimensionless (a non-dimensionalised case)) (0.00s) |
| location | passed | 0.0 | every field declares a location consistent with the mesh (0.00s) |
| frame | passed | 1.0967254638651058e-05 | axes strictly monotonic and uniformly spaced within the tolerance their dtype allows ('t' is float32: 4.8e-05 of this deviation is rounding, 'x' is float32: 1.2e-04 of this deviation is rounding) (0.00s) |
| provenance | passed | 0.0 | source, licence (CC BY 4.0), citation and 1 input hash(es) recorded (0.00s) |
| reference | passed | 0.024844878756419598 | relative L2 error of 'density' against the reference (0.00s) |

Leader fno on 40 held-out time steps; tolerance 5.0e-02 -> outside. Reduced CPU budget: this compares the two families on this data, not against published numbers.
