# The Ashton automotive datasets: three bodies, one loader, one out-of-distribution test

AhmedML, DrivAerML and WindsorML are three open road-vehicle CFD datasets published by Neil Ashton and
co-authors in the same on-disk layout, under the same licence, each a parametric family of one reference body
run with a scale-resolving method. Because the layout is shared, one reader serves all three -- and because the
*bodies* differ, a model trained on one family and tested on another is a real out-of-distribution test rather
than a recount of the same family's variation.

| dataset | cases | method | cells | size | paper |
|---|---|---|---|---|---|
| **AhmedML** | 500 Ahmed-body variants | hybrid RANS-LES, OpenFOAM v2212, ~80 convective time units | ~1.1 M surface, ~20 M volume | 2 TB | arXiv:2407.20801 |
| **DrivAerML** | 500 morphed DrivAer notchbacks (484 public) | hybrid RANS-LES, OpenFOAM v2212 | ~8.8 M surface, ~160 M volume | 31 TB | arXiv:2408.11969 |
| **WindsorML** | 355 Windsor bodies (350 run folders) | GPU wall-modelled LES, Cartesian immersed boundary | >280 M volume | 8 TB | arXiv:2407.19320 |

They are CC BY-SA 4.0. Share-alike travels with the data: a derived dataset redistributed from these has to
carry the same licence, which is a fact about a surrogate trained on them that whoever ships the surrogate has
to know. The credit each dataset asks for is in `pinneapple_data/ashton/catalog.py` (`ATTRIBUTION`) and travels in
the object every read returns -- see [Licence and credit](#licence-and-credit).

## What is in this module, and what is not

| piece | what it does | offline? |
|---|---|---|
| `pinneapple_data.ashton.catalog` | the three datasets: paper, licence, credit, layout, reference quantities, which file holds what | yes |
| `pinneapple_data.ashton.loader` | open a dataset, read a case, integrate its drag, subsample its volume by byte range | needs the data |
| `pinneapple_data.ashton.vtk_xml` | the VTK XML surface reader (partial reads) and the writer the fixture uses | yes |
| `pinneapple_data.ashton.fixture` | a synthetic dataset in the same layout: two parametric body families, a stated pressure model, one hand-computed case | yes |
| `pinneapple_data.ashton.benchmark` | the comparison table: published baselines *declared*, measurements where a case is on disk | yes on the fixture |
| `pinneapple_data.ashton.trust` | the cross-family experiment: does the trust signal move before the error does | yes on the fixture |

Nothing here downloads anything unless asked, and nothing here ships the data. What *is* measured, on what is
stated as synthetic, is listed at the end -- and separately from what needs the 31 TB.

## Reading a case

```python
from pinneapple_data.ashton import loader

data = loader.open_dataset("ahmedml", root="~/data/ahmedml")     # reads nothing yet
data.n_cases, data.licence, data.attribution.split(",")[0]
# (500, 'CC-BY-SA-4.0', 'Ashton, N.')

case = data.case(1)                     # force_mom_1.csv and geo_parameters_1.csv (66 KB together)
case.forces["cd"], sorted(case.parameters)[:3]

measured = data.measured_forces(1)      # -oint Cp n_x dA / A_ref, on boundary_1.vtp (612 MB, decoded once)
measured.cd, measured.cd_published      # the two routes to the same number, side by side
```

The layout, as the three dataset cards describe it:

```
run_<i>/                        one case (WindsorML runs are 0..349)
    boundary_<i>.vtp            surface field, VTK XML PolyData        (AhmedML, DrivAerML)
    boundary_<i>.vtu            surface field, VTK XML UnstructuredGrid (WindsorML)
    boundary_cell_area_<i>.npy  one area per native polygon, float32   (AhmedML, DrivAerML)
    boundary_dual_area_<i>.npy  one quadrature weight per point, float32 (WindsorML)
    force_mom_<i>.csv           time-averaged coefficients, constant reference area
    force_mom_varref_<i>.csv    the same, reference area of the case
    geo_parameters_<i>.csv      the design parameters of the case
    <body>_<i>.stl              the surface geometry (and .step/.stp for DrivAerML and WindsorML)
    volume_<i>.vtu              the volumetric field, 5.6 GB to 44 GB per case
force_mom_all.csv, geo_parameters_all.csv, splits/manifest.json   the aggregate tables and the splits
```

WindsorML is the one place the layout is not identical: a different container (`.vtu`) and a different
quadrature (a weight per **point**, not an area per polygon). Both are declared in the catalogue and used by the
integral, rather than sniffed at read time.

**A file that is not there raises with the URL and the call that fetches it**:

```
CaseNotPresent: drivaerml run 1: neither run_1/force_mom_1.csv nor run_1/boundary_1.vtp is under /data/drivaerml.
Fetch one with fetch('forces', 1, root='/data/drivaerml', dataset='drivaerml'), or by hand from
https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_1/force_mom_1.csv (31 TB ...)
```

and a dataset that has not been downloaded still opens: it reports zero cases, its licence, its credit and its
declared baselines, which is what a table built from published numbers needs.

## Partial reads: the surface, and the volume

The surface files are VTK XML: base64 arrays in one long text document, no index to seek with. `read_surface`
takes the arrays it needs and *skips the text of the others as it passes them* -- so asking for `Cp` on a
612 MB DrivAerML boundary file decodes one field instead of fifteen. The mesh (points, connectivity, offsets) is
always read; it is what the integral needs.

```python
from pinneapple_data.ashton.vtk_xml import read_surface

surface = read_surface("run_1/boundary_1.vtp", arrays=["Cp"])   # points, cells and Cp; the rest skipped
surface.n_points, surface.n_cells, sorted(surface.point_data), surface.skipped
manifest = read_surface("run_1/boundary_1.vtp", arrays=())      # the names, decoded none of them
manifest.available_cells, manifest.skipped
```

The volume files are 5.6 GB (AhmedML), 20 GB (WindsorML) and 44 GB (DrivAerML). `volume_subsample` reads
chosen points *by byte range*: base64 puts three bytes in four characters, so an element range is reachable from
any three-byte alignment and a hundred points out of 160 million cost a hundred short reads. The same contract
that reads HDF5 over HTTP (`pinneapple_data.datasets.remote_hdf5.HTTPRangeFile`) is what makes a range read of
a remote file possible at all; the sample comes back with the indices it took, so a point can be found again.

```python
sample = loader.volume_subsample("run_1/volume_1.vtu", 2000, seed=0, fields=["pMean", "UxMean"])
sample["points"].shape, sample["_index"][:3]
```

## The fixture: what the tests actually read

The three datasets cannot be in a test, so `pinneapple_data.ashton.fixture` writes a small one in the same
layout: a closed lofted body with a nose, a taper, a rear slant and a ground clearance, at Ahmed-body dimensions
(large enough that the catalogue's 0.112 m² reference area is the right one for it), with a stated pressure model
on its faces -- `1 - (V/Vinf)²` with a model speed and a base-suction term. Two families come out of the same
generator:

| family | superellipse exponent | rear slant | taper | what it looks like |
|---|---|---|---|---|
| `alpha` | 2.6-3.4 | 10-22° | 0.02-0.10 | rounded, low, gently tapered |
| `beta` | 5.0-7.0 | 28-42° | 0.14-0.24 | boxy, tall, heavily tapered |

It also writes three things that make the measurements checkable rather than circular:

* `force_mom_<i>.csv` is integrated by decomposing each quad into two triangles (`Heron`), a **different route**
  from the polygon (Newell) integral the loader uses -- so comparing the two checks the integral, not the same
  arithmetic twice;
* `write_box_case` writes the one case with a closed-form answer: a 2 x 1 x 1 box, Cp = 1 upstream, -0.4
  downstream, 0 elsewhere, `A_ref = 1 m²`, so `Cd = 1*1 + 0.4*1 = 1.4` and `Cl = Cs = 0`;
* a Windsor-shaped case (`dataset="windsorml"`) of the *same body and field*, so the point quadrature and the
  cell quadrature can be compared on identical geometry.

## The benchmark table

Two kinds of number go in one table, and the difference is the point:

* a **declared** number is one somebody published, quoted with its paper, its table and -- where the datasets
  scale their fields differently -- its scale. `BASELINES` holds them; nothing in the module can compute or
  overwrite one;
* a **measured** number is computed here from a case that is on disk. A row says which it is: `measured`,
  `declared` or `awaiting_data` (with the size and the call that would fill it).

```python
from pinneapple_data.ashton.benchmark import benchmark_table, constant_field_baseline

table = benchmark_table({"ahmedml": "data/ashton/ahmedml"}, datasets=["ahmedml", "drivaerml", "windsorml"])
print(table.to_markdown())          # declared rows only, on a machine with no data
```

The declared surface-pressure numbers are the AB-UPT paper's (Alkin et al., arXiv:2502.09692, v4): L2 0.0272 for
GP-UPT on AhmedML (Table 2), L2 0.0388 for Transolver and 0.0434 for GP-UPT on DrivAerML (Table 1). The declared
force coefficients are the dataset papers' own solver-versus-tunnel numbers: Cd 0.285 against 0.299 from
experiment (AhmedML, Table 1), Cd 0.274 against 0.255 (DrivAerML, Table 1), Cd 0.33 on the finest grid against
0.33 (WindsorML, Table 1).

Two things are stated in the table rather than rounded off:

* **WindsorML is not in the AB-UPT paper.** The version read here (v4) mentions it zero times. Its declared
  number is therefore the dataset's own solver-versus-tunnel result, which is a different kind of number -- a
  CFD solver against a wind tunnel, not a surrogate against a solver -- and the table says so in a row rather
  than leaving the column blank;
* **the two pressure columns are not on the same scale.** AhmedML's boundary file carries Cp, so its MSE is
  ~1e-3; DrivAerML's carries pressure, so its MSE is ~1e3. L2 is scale-free and comparable; MSE and MAE are not,
  and the table's footer says that.

A surface-pressure error belongs to a *model*, not to the data, so that column stays empty until one is passed:
`benchmark_table(..., model=my_predict)` calls `model(case)` for a predicted field on the case's own faces. The
baseline that runs offline is `constant_field_baseline`, the area-weighted mean field of the training cases --
a surrogate that predicts nothing, which every surrogate that predicts something has to beat.

On the fixture, with the constant baseline, the table reads (4 cases, `ahmedml`):

| quantity | measured | declared |
|---|---|---|
| drag coefficients | `cd_mean` 0.5686 against the CSV's 0.5686, max abs error 6.6e-09 | Cd 0.285 (the dataset's own CFD, arXiv:2407.20801 Table 1) |
| surface pressure | L2 0.352, MSE 0.332, MAE 0.395 | L2 0.0272 (GP-UPT, arXiv:2502.09692 Table 2) |

The 6.6e-09 is the two independent integrals agreeing. The 0.35 is what a surrogate that predicts nothing gets,
and it is why the published 0.0272 needs a scale next to it.

## The cross-family experiment

Train on one family, test on the other, and ask whether the trust layer says so before the number goes wrong.
The model is a ridge regression from *geometry descriptors* read off the surface (extents, ground clearance,
wetted and frontal area, the area fraction facing aft, flatness, shape factor) to the drag coefficient and the
area-weighted Cp quantiles. A ridge regression because the question is about the trust layer, not the model: a
stochastic training would put its own variance into the measurement.

The shift is graded: family `alpha` is the training family, and the evaluation families are the linear blend of
the two parameter boxes, `box(t) = (1 - t) box_alpha + t box_beta`, at `t = 0, 0.2, ..., 1`. `t = 0` is a fresh
sample of the training family (a real held-out split), `t = 1` is the other family, and the levels in between are
the shift arriving -- which is the only way to answer "before" rather than "as well as".

Three signals, all measured here:

* **embedding distance** -- the Mahalanobis distance of a case's descriptor vector to the training
  distribution, through `pinneapple_analysis.trust.TrustGate` (fitted on the training cases, scored per case).
  Its threshold is calibrated on a *held-out* in-family split at 99 %, so "flagged" means "further out than every
  genuine in-family case was", and the false-alarm rate that threshold produced is reported;
* **flagged fraction** -- the share of a level's cases over that threshold: the label-free alarm;
* **conformal coverage** -- a 90 % split-conformal interval on Cd calibrated on the held-out in-family split
  (`pinneapple_analysis.uncertainty.conformal.conformal_quantile`), and its coverage at each level.

The measured result, seed 0, 36 training and 12 calibration cases, 28 x 13 surface mesh, both directions:

| trained on | towards | t flagged | t error doubles | embedding distance 0 -> 1 | Cd error 0 -> 1 | coverage 0 -> 1 |
|---|---|---|---|---|---|---|
| `alpha` | `beta` | 0.4 | 0.6 | 2.89 -> 38.18 | 0.019 -> 0.110 | 100 % -> 0 % |
| `beta` | `alpha` | 0.2 | 0.6 | 2.90 -> 16.59 | 0.020 -> 0.209 | 90 % -> 0 % |

So in both directions the label-free alarm crosses its calibrated threshold a level before the error exceeds
twice its in-family value, and the conformal interval calibrated in-family covers none of the other family. That
is the answer the three datasets were read for, measured on synthetic geometry.

Two findings from the same run that are worth more than the headline:

* **the Cp *distribution* barely moves.** Its area-weighted quantiles stay within 0.026-0.044 relative L2 across
  every level, in both directions, while the drag error goes up by 5-10x. On this fixture the pressure model is
  nearly the same function of the local normal, so the shift moves the *integral*, not the field's shape. The
  experiment therefore claims its ordering on the drag coefficient -- the quantity these datasets exist for --
  and reports the descriptor error as the flat column it is, rather than quoting a flat number as a result;
* **the ordering is not free.** At a coarser fixture mesh (24 x 11) both crossings happen at the same level and
  no ordering is established, and across seeds 0-3 one direction failed because that training set fitted so well
  (in-family Cd error 0.0129) that "twice the in-family error" became a strict threshold. `CrossFamilyResult`
  reports both crossings and the boolean, so an experiment where the ordering does not hold says so.

## Licence and credit

All three datasets are **CC BY-SA 4.0** (the `license` field of each card, and their `LICENSE.txt`). The
attribution each one asks for, quoted from its card:

* **AhmedML** -- Ashton, N., Maddix, D., Gundry, S. and Shabestari, P. (2024), *AhmedML: High-Fidelity
  Computational Fluid Dynamics dataset for incompressible, low-speed bluff body aerodynamics*, arXiv:2407.20801.
* **DrivAerML** -- Ashton, N., Mockett, C., Fuchs, M., Fliessbach, L., Hetmann, H., Knacke, T., Schonwald, N.,
  Skaperdas, V., Fotiadis, G., Walle, A., Hupertz, B. and Maddix, D. (2024), *DrivAerML: High-Fidelity
  Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics*, arXiv:2408.11969.
* **WindsorML** -- Ashton, N., Angel, J., Ghate, A., Kenway, G., Long Wong, M., Kiris, C., Walle, A., Maddix, D.
  and Page, G. (2024), *WindsorML: High-Fidelity Computational Fluid Dynamics dataset for automotive
  aerodynamics*, arXiv:2407.19320 (NeurIPS 2024 Datasets and Benchmarks).

Both travel in the object: `open_dataset(...).licence` and `.attribution`, and a
`PhysicsDataBundle` made with `case_bundle` carries them in its provenance (`license`, `citation`, `url`) and in
its metadata, so a case cannot lose them by being passed along. The reference quantities each coefficient is
normalised with -- and where in each paper they come from -- are `FREESTREAM_SOURCE` in the catalogue.

## What was measured here, and what needs the 31 TB

Measured, offline, on the synthetic fixture of this module:

* the VTK XML reader on both containers, ASCII and inline binary, with partial reads (only the named arrays
  decoded) and a name manifest that decodes nothing;
* the drag integral against a hand-computed box: `Cd = 1.4` to 1e-9, from the published quadrature and from the
  faces' own areas; the same body integrated by both quadratures agrees to 2.2 % (point quadrature on a
  366-node surface), and `surface_area` matches to 1e-9;
* the measured drag against the fixture's independently integrated CSV: 6.6e-09 maximum absolute difference over
  six cases;
* the metric definitions on hand values, including what area-weighting does and does not change;
* the byte-range volume subsample against the full read: identical values at the sampled indices;
* the licence, the credit and the reference quantities of all three datasets, and the provenance of every
  declared baseline (paper, table, scale);
* the cross-family ordering, twice, with the numbers above -- and its two caveats.

Not measured here, and not claimed: any number on the real 41 TB. The table's `measured` column fills itself
when the data is there (`benchmark_table({"ahmedml": root, ...}, model=...)`); the cross-family experiment
becomes the real thing by pointing `ashton.loader` at two dataset roots instead of at the fixture. Until then
the measured column of the published comparison is empty, and the only numbers in it are the ones the papers
printed.
