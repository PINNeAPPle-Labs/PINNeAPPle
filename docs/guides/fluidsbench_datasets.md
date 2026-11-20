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

## The per-case numbers, read from the case records (2026-11-19)

The `case-support.json` records in the checkout state, for every case, the source artifact's size, its SHA-256 and the
native entity counts — which is the geometry representation and the evaluation contract, before anything is downloaded:

| dataset | what one case is, natively | measured per case | `full`-split total |
|---|---|---|---|
| AhmedML | surface `.vtp` (78.6 MB) + volume `.vtu` + cell areas | **5.55 GB** (median of 450 records, volume alone) | ~280 GB for 50 cases |
| WindsorML | surface points, 640 entities; volume support alongside | 0.5 MB of support per case | small; the volume fields are the expensive part |
| DrivAerML | volume cells, **8.4 M entities** per case, chunked | support ships chunked, 1 MB per artifact | the surface-only scope exists; the volume scope is the 50 GB-per-VTU case |
| DrivAerNet++ | surface points only, `p` alone | geometry **26.4 MB** (median of 1 154), support 5.0 MB | **30.5 GB geometry + 5.49 GB support** |

The AhmedML surface, fetched and parsed (2026-11-20): a `.vtp` of 82.7–106.4 MB with **1 101 574–1 235 746 boundary
entities** of quads (and 422 pentagons in `run_4`), `pMean` as one component in m²/s² and `wallShearStressMean` as three,
whose median magnitude is 1.5·10⁻³ m²/s² — five hundred times smaller than the pressure, which is the scale a fit has to
respect. The published `boundary_cell_area_<id>.npy` (4.4 MB) sums to **1.6843 m²** of wetted area on `run_4` and the
geometric fan area of the same polygons reproduces it to the last digit, so the weights are checkable against the mesh
instead of trusted.

**DrivAerNet++ contract, verified by fetching one case** (`E_S_WWC_WM_008`; the archive's SHA-256 matches the manifest's
`d22f6fa7…`): a VTK legacy surface with PointData `p`, **511 063 points**, 474 686 polygons, 5 connected components
(436 403 + 2×18 742 + 2×18 588 — body and four wheels), bounding box 4.85 × 2.25 × 1.33 m, total dual area
32.35 m², dynamic head 450 m²/s², `p` from −1 853.8 to 449.9 m²/s². The support `.npz` carries the truth `p`, the
`dual_area_m2` weights and the profile indices — and **no coordinates**: the geometry has to be fetched separately, which
is what makes the 30.5 GB the real cost of scoring this split. Two metrics are bound to the single quantity
(`surface_pressure_rel_l2`, area-weighted; `surface_pressure_equal_entity_rel_l2`, uniform), aggregated per geometry and
then averaged over the 1 154 designs, and it is the only component of the composite (cap 100).

## This machine, measured (2026-11-19)

| resource | laptop | server `159.69.246.142` |
|---|---|---|
| CPU | Apple M5, 10 cores | 4 vCPU |
| memory | 16 GB | 7 GB |
| disk free | 72 GB (after clearing 87 GB of caches, Docker's image and stale run outputs) | 20 GB |
| accelerator | MPS (8-core GPU) | **none** |

## What that means, dataset by dataset

- **AirfRANS — feasible, done.** 14 GB of data, a 3.7 GB feature store, and a 20-minute fit of a 1-2 M-parameter
  coordinate network on the MPS device. This is why the campaign runs here.
- **AhmedML — half of it is feasible, and now measured.** The surface is 83 MB per case and carries its own truth and
  its own weights, so the surface fit runs here (below). The volume is 5.55 GB per case, its prepared cache for one
  50-case split is **1.74 GB of download that installs to 6.61 GB**, and the contract's
  `complete_case_and_entity_coverage_required` means a surface-only prediction is refused rather than scored partially.
  A complete entry is therefore ~280 GB of release against this machine's 72 GB.
- **WindsorML — not feasible here.** 233 test cases with volume fields, and the volume representation is the expensive
  part; the smallest label (`full`, 35 cases) still needs both surface and volume predictions per case.
- **DrivAerML — not feasible here.** About 50 GB for a single reconstructed VTU; the smallest split is 50 test cases.
- **DrivAerNet++ — not feasible here, and the number is now exact.** It is the cheapest of the four by contract: one
  scalar field on the surface, no volume, one component in the composite. It costs **30.5 GB of geometry plus 5.49 GB of
  support** for the 1 154-design test split, against 24 GB free — and the support carries no coordinates, so the geometry
  cannot be avoided. With a 40 GB disk and this laptop it would still be a 5·10⁸-point inference problem.

The disk is only the first wall. The second is compute: every one of the four is a 3-D geometry-conditioned problem
(point clouds or graphs with 10⁵-10⁶ surface points per case), where a single training run on a GPU is measured in
hours and on this laptop's MPS device in days. Nothing about the AirfRANS result transfers, because the scored
quantities are surface fields on unstructured 3-D geometry rather than a 2-D volume field.

## What it takes to start DrivAerNet++, measured (2026-11-19)

The dataset is the cheapest of the four by contract and the most awkward by access. Both were checked end to end:

- **the contract is one scalar.** `drivaernetplusplus-surface-native-points-v1` binds two metrics to the same quantity
  — `surface_pressure_rel_l2` (area-weighted) and `surface_pressure_equal_entity_rel_l2` (uniform) — aggregated per
  geometry and then macro-averaged over 1 154 designs, and they are the whole composite. A submission needs no volume.
- **the data is fifteen family ZIPs, not case files.** The Dataverse API (`doi:10.7910/DVN/K7PWNJ`) lists 15 archives
  from 9.04 GB to 18.80 GB, **212.5 GB** in total, stored Globus-backed. A case is 26.4 MB inside one of them.
- **the download needs a guestbook.** `GET /api/access/datafile/10816650` answers HTTP 400 with *"You may not download
  this file without the required Guestbook response for guestbookID 601"*. That is a form filled by the account holder —
  it is not something a script should answer on someone's behalf, so the dataset's own pipeline stops there.
- **range requests do work**, so once the guestbook is accepted a handful of cases can be pulled without the 18 GB
  archive: the support ZIPs of the *scoring* side are already public on the release page and were fetched that way,
  with the archive SHA-256 matching the manifest.
- **the scoring support carries no coordinates.** The per-case `.npz` has the truth `p`, the `dual_area_m2` weights and
  the profile indices; the geometry has to come from the archive, which is why the 30.5 GB of surface files is the real
  cost of a full test split and why nothing can be predicted before the geometry is in hand.

`pinneapple_simulation.pipelines.drivaernetplusplus` implements everything that does not need the geometry — the VTK
reader, the support reader, both reductions and the macro aggregation — and its test suite checks the scoring side
against the released support of `E_S_WWC_WM_008` (511 063 points, 32.3518 m² of dual area, `p` from −1 853.76 to
449.888 m²/s²).

### Once the guestbook is accepted

```bash
# 1. one family archive, then the cases it holds (check the box, then the download is a normal ranged GET)
curl -L -H "Range: bytes=0-1" -o /dev/null -w '%{http_code}\n' \
  https://dataverse.harvard.edu/api/access/datafile/10816650      # 206 when the guestbook is settled

# 2. the case's geometry, and its scoring support from the public release
python -m pinneapple_simulation.pipelines.drivaernetplusplus --case E_S_WWC_WM_008 \
  --geometry E_S_WWC_WM_008.vtk --support E_S_WWC_WM_008/support.npz
```

## What it takes to start AhmedML, measured (2026-11-20)

AhmedML is the mirror image of DrivAerNet++: open where that one is gated, but split in two where that one is a single
scalar. The composite gives **25 % to the surface** (`surface_pressure_rel_l2`, cap 15, weight 0.15;
`surface_wall_shear_rel_l2`, cap 20, weight 0.10), **25 % to the volume** (`volume_velocity_rel_l2`, cap 12, 0.15;
`volume_pressure_rel_l2`, cap 15, 0.10), 25 % to the forces and 25 % to the diagnostics — and
`complete_case_and_entity_coverage_required` applies to both domains.

- **the access is not the obstacle.** The pinned revision of `neashton/ahmedml` serves individual files over plain
  HTTPS (no guestbook, unlike DrivAerNet++): the surface of a case is 83 MB and arrives in six seconds, and a whole
  50-case test split is 4.2 GB.
- **the surface is self-contained.** `boundary_<id>.vtp` carries `pMean` and `wallShearStressMean` as CellData and
  `boundary_cell_area_<id>.npy` carries the weights, so a surface model can be fitted *and* scored with nothing else.
  `geometry_parameters_<id>.csv` adds the eight geometric parameters the design vector needs.
- **two fits, both on the same 50 test cases** (`super_scarce` and `scarce`), scored by the module's own reductions:

| training cases of the split | surface pressure, area-weighted | surface pressure, equal entity | wall shear, area-weighted | wall shear, equal entity |
|---|---|---|---|---|
| 11 (`super_scarce`) | 26.10 % | 26.98 % | **17.81 %** | 18.82 % |
| 67 (`scarce`) | 22.76 % | 24.01 % | **15.37 %** | 16.40 % |
| 400 (`full`) | 20.18 % | 20.15 % | **12.70 %** | 12.79 % |
| the training mean of each field | 81.21 % | 88.18 % | 100 % | 100 % |

  The wall shear is under its 20 % cap at every size and keeps falling with data (17.8 → 15.4 → 12.7 %); the pressure
  is not under its 15 % cap and falls far more slowly (26.1 → 22.8 → 20.2 %), which says the pressure side is limited by
  the model rather than by the number of cases — each case is thinned to 120 000 of its ~1.2 M polygons, a tenth of the
  available resolution. The two heads are normalised separately precisely because the shear is 500× smaller than the
  pressure.
- **what is implemented and tested:** `pinneapple_simulation.pipelines.ahmedml` — the `.vtp` reader (VTK's own, because
  the connectivity array is not something to hand-parse), both reductions for both fields with the spec's vector rule,
  the macro average, the area-weighted fit, and a prediction path that needs no truth. Eight tests, none of which needs
  the release.

**The exact blocker for a complete entry** is the volume: 5.55 GB per case, of which `UMean` and `pMean` are CellData on
21.5 M cells, and the whole-split derived cache installs to 6.61 GB per 50 cases against a disk that holds 72. No model
of the volume exists here and none is claimed.

### The exact next command

```bash
# 1. the surface of one split, verified against the release's own digests (83 MB per case)
python fetch_ahmedml_surface.py scarce "$DATA/ahmedml-surf"

# 2. fit on the split's own training cases and score the complete test list
python run_ahmedml.py --split scarce --epochs 25 --cell-limit 250000 --points-per-case 150000

# 3. the volume — a machine with ~300 GB, and then the volume surrogate does not exist yet
python scripts/fetch_ahmedml_support.py --split scarce      # 1.7 GB down, 6.6 GB installed
```

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
