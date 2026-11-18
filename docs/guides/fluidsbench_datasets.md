# FluidsBench beyond AirfRANS: what each dataset costs, and what this machine can carry

Five FluidsBench datasets accept submissions today. AirfRANS is the one the campaign covers end to end; this page
records what moving to another one would actually require — measured against the hardware this repository is running
on, not against a wish. The numbers are the benchmark's own published figures, quoted from
`benchmark-specs/<dataset>/` of the `FluidsBench/fluidsbench-submission` checkout, with the date they were read.

## Availability, as published

| dataset | submission status | splits that accept results | what a result must predict |
|---|---|---|---|
| AirfRANS | open | `full`, `scarce`, `aoa_extrapolation` (`reynolds` closed) | velocity and pressure at every native volume point |
| AhmedML | open | eight prescribed splits (`full`, `medium`, `scarce`, `super_scarce`, four diagnostic) | surface `pMean`/`wallShearStressMean` **and** volume `pMean`/`UMean`, native cell order |
| WindsorML | open | eight prescribed labels; `full` is 35 test cases, 233 distinct test cases overall | surface (point IDs) and volume (cell IDs) fields |
| DrivAerML | open | eight splits, 50 or 97 test cases each out of 484 public cases | surface + volume, or surface only |
| DrivAerNet++ | open | one split: pressure at every native surface point of 1,154 official test designs | surface pressure only |
| HiLiftAeroML | coming soon | owner-coordinated trials only | — |
| BlendedNet, Rotor37, VKI-LS59 | coming soon | evaluation support being prepared | — |

Every open dataset scores through the same composite machinery — `bounded_error` components with per-dataset caps and
`bounded_quality` R² components — but the fields, the support manifests and the evaluator are dataset-specific. Nothing
transfers by assumption; the support release pins exactly which points are scored.

## The footprint each one needs

| dataset | public data | prepared support | notes |
|---|---|---|---|
| AirfRANS | 14 GB (the whole release, all splits) | included in the checkout | the feature store this campaign uses is 3.7 GB |
| AhmedML | pinned Hugging Face revision `neashton/ahmedml` | ~41.45 GB derived cache for the 316-case test union | cache is downloadable, byte-identical and verified; the canonical VTP/VTU are still required for training |
| WindsorML | `neashton/windsorml`, per-case download | profile support for 233 cases bundled in the checkout | surface dual areas travel with the case data |
| DrivAerML | `neashton/drivaerml` | chunked immutable support | the guide states a reconstructed VTU is about 50 GB, which is why it ships chunked |
| DrivAerNet++ | 15 Harvard Dataverse archives, about 210 GB | about 5.5 GB prepared support | surface pressure only, 1,154 designs |

## This machine, measured (2026-11-18)

| resource | laptop | server `159.69.246.142` |
|---|---|---|
| CPU | Apple M5, 10 cores | 4 vCPU |
| memory | 16 GB | 7 GB |
| disk free | 24 GB | 20 GB |
| accelerator | MPS (8-core GPU) | **none** |

## What that means, dataset by dataset

- **AirfRANS — feasible, done.** 14 GB of data, a 3.7 GB feature store, and a 20-minute fit of a 1-2 M-parameter
  coordinate network on the MPS device. This is why the campaign runs here.
- **AhmedML — not feasible here.** The derived cache alone is 41.45 GB against 24 GB free, before the canonical case
  files that training needs; and a surface *and* volume prediction must be produced for 50 test cases. Two orders of
  magnitude of storage short.
- **WindsorML — not feasible here.** 233 test cases with volume fields, and the volume representation is the expensive
  part; the smallest label (`full`, 35 cases) still needs both surface and volume predictions per case.
- **DrivAerML — not feasible here.** About 50 GB for a single reconstructed VTU; the smallest split is 50 test cases.
- **DrivAerNet++ — not feasible here.** 210 GB of public data against 24 GB of disk.

The disk is only the first wall. The second is compute: every one of the four is a 3-D geometry-conditioned problem
(point clouds or graphs with 10⁵-10⁶ surface points per case), where a single training run on a GPU is measured in
hours and on this laptop's MPS device in days. Nothing about the AirfRANS result transfers, because the scored
quantities are surface fields on unstructured 3-D geometry rather than a 2-D volume field.

## The exact next command, once there is a machine that fits

```bash
# 1. the data (AhmedML, surface-only training loop; the evaluator still needs both fields for a full result)
hf download neashton/ahmedml --repo-type dataset --local-dir "$DATA/ahmedml"
python scripts/fetch_ahmedml_support.py --split super_scarce

# 2. the contract, before any model is written
python -m pinneapple_arena.fluidsbench dataset ahmedml        # splits, digests, composite, required fields
python scripts/ahmedml.py --help                              # the participant workflow of that dataset

# 3. one case end to end, then the split — never the other way round
python scripts/evaluate_ahmedml_candidate_case.py --case-id run_1 ...
```

The rule the campaign follows on AirfRANS applies unchanged: one case through the dataset's own evaluator before a
single split is trained, because a model that cannot be scored costs more than a model that is not built.
