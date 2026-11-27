# Verify somebody else's prediction: the five-minute quick start

`pp.trust` answers *how much can I trust this model?* This answers the other question a consumer asks: *does this
field satisfy the physics it claims, and where does it stop doing so?* You bring the prediction and the description
of the problem; you get back a **certificate** — a JSON document that says, per check, the number, the limit it was
judged against, the point in the domain where it is worst, and where the number came from.

Nothing in it is a second engine. The residuals are the equation set of the linter (`pinneapple_check.physics`), the
reading is the linter's field reader (`pinneapple_check.fields`), the units are `pinneapple_data.physical_units`, the
in-domain level follows the ensemble's `residual_baseline`, and the domain gate is `TrustGate`. What the certificate
adds is the decision, the provenance of every limit, and the two numbers that make the decision meaningful.

## 0. Install, and see that it imports

```bash
pip install pinneapple                 # or: pip install -e ".[check]" from a checkout
python -c "from pinneapple_analysis.trust.certificate import Problem, verify; print('ok')"
```

That is the whole setup: the certificate has no extra dependency beyond what `pinneapple_check` already needs. The
rest of this page is copy-paste — one field, one problem, one certificate, then the two numbers that say whether the
certificate itself can be believed.

## 1. A prediction you have, and the equation it claims

The exact solution of `dC/dt + c dC/dx = alpha d2C/dx2` — so you can see the certificate agree with something known:

```python
import numpy as np
from pinneapple_analysis.trust.certificate import Problem, verify

x, t = np.linspace(0, 2 * np.pi, 65)[:-1], np.linspace(0, 1, 41)
X, T = np.meshgrid(x, t, indexing="ij")
C = np.exp(-0.05 * T) * np.sin(X - T)                      # the exact solution, c = 1, alpha = 0.05

problem = Problem(physics="advection-diffusion",
                  axes={"x": x, "t": t},
                  parameters={"c": "1 m/s", "alpha": "0.05 m2/s"},   # SI numbers or strings with their unit
                  units={"C": "1", "x": "m", "t": "s"},
                  tolerance={"advection-diffusion": 0.05})           # what you accept for this law

cert = verify({"C": C}, problem)      # arrays; a .npz/.h5 grid or a VTK point cloud also work

cert.badge()      # 'good'
cert.verdict()    # 'certified'
print(cert)       # one line per check: PASS 0.0007477 (limit 0.05)
cert.write("certificate.json")      # the same document, for whoever has to sign it off
```

### Your own prediction

The field above is an analytic solution so you can see the certificate agree with something known. Yours is not:
hand it over as the file it already is. `verify` reads the coordinates from the file, so the `Problem` loses `axes`:

```python
# a .npz with the 1-D axes, the field and its unit -- what most surrogates save
np.savez("my_prediction.npz", x=x, t=t, C=C, units=np.array(["C=1", "x=m", "t=s"]))
problem = Problem(physics="advection-diffusion", parameters={"c": "1 m/s", "alpha": "0.05 m2/s"},
                  tolerance={"advection-diffusion": 0.05})

verify("my_prediction.npz", problem).badge()        # 'good'
```

| what you have | what you pass to `verify` | what you get |
|---|---|---|
| a NumPy grid | `{"C": C}` plus `Problem(axes={"x": x, "t": t})` | the residual on the grid's own derivatives |
| `.npz` / `.h5` with 1-D axes | the path — it carries the axes, and an optional `units` array of `"name=unit"` | the same, on the file's grid |
| `.vtu` / `.vtk` / `.xdmf` / `.msh` point cloud | the path (meshio reads it) | the residual on a point-cloud derivative, so looser than on a grid |
| `.csv` / `.parquet` point cloud | the path | the same as above |
| an OpenFOAM case folder | the path, and `time="5"` to pick the time directory | **read but not yet judged**: the certificate reports `input` as `unmeasured`, because the face sums are `pinneapple_check.fv` |
| NetCDF | — | not read; export to `.npz` or `.vtu` first |

## 2. The same field, advecting at the wrong speed

```python
wrong = np.exp(-0.05 * T) * np.sin(X - 1.6 * T)             # 1.6 where the problem declares 1.0
cert = verify({"C": wrong}, problem)                        # the same Problem as above

cert.badge()        # 'fading'
cert.failures()     # (Check(name='residual:advection-diffusion', value=0.2285, threshold=0.05,
                    #        where='x=3.142, t=0', ...),)
cert.separating()   # 'failing: residual:advection-diffusion -- 0.228473 > 0.05; worst at x=3.142, t=0 (...)'
```

`where` is the part a consumer needs: *not* "the model is bad", but *"here, by this much"*.

## 3. Where the limit comes from

The certificate never decides by itself what "good enough" is. In this order:

1. **the model's own in-domain level** — `Problem(reference={"advection-diffusion": 1e-3}, reference_source="the
   ensemble's own in-domain median")`. This is the honest one: a model whose residual is 0.04 everywhere inside its
   domain is not failing at 0.05 outside it. The `source` of the check repeats where the limit came from.
2. **your declared tolerance** — `Problem(tolerance={"continuity": 0.01})`, per law or per physics.
3. **nothing** — the check is `unmeasured`, the badge is `fading`, and the summary says `not measured: ...`. A
   residual nobody can judge is never a pass.

## 4. Inside the domain or outside it

```python
outside = Problem(physics="advection-diffusion", axes={"x": x, "t": t},
                  parameters={"c": "1 m/s", "alpha": "0.05 m2/s"}, units={"C": "1"},
                  tolerance={"advection-diffusion": 0.05}, domain={"t": (0.0, 0.5)})   # the field runs to t = 1

cert = verify({"C": C}, outside)
cert.badge()        # 'no_skill'
cert.separating()   # 'no_skill: domain -- 0.609756 < 1 (fraction of points inside the declared domain ...)'
```

That is a different sentence from "the residual is high", and it is the one that changes what a consumer does.

## 5. What else the certificate checks

| check | what it reads |
|---|---|
| `finite` | non-finite values, per field |
| `units:<field>` | the unit you declared against the one in the file, by dimension and factor (`PC501`/`PC502`) |
| `parameter:<name>` | the unit of an equation parameter; a wrong dimension **stops** the certificate instead of computing a residual from a differently-scaled number |
| `residual:<law>` | the equation's relative RMS, on the linter's derivative providers |
| `boundary:<field>` | a declared boundary condition, where you have one |
| `positivity:<field>` | a field that cannot go negative and does |
| `domain` | the fraction of query points inside the declared envelope |

The laws available are the ones the linter has: incompressible flow (continuity and momentum), heat,
advection-diffusion, shallow water, elasticity, conservation, Darcy and Maxwell.

## 6. Measuring the certificate itself

A certificate that flags everything is as useless as one that flags nothing. `calibrate` counts both, case by case,
against a truth the certificate cannot see — here on the two fields this page already has:

```python
from pinneapple_analysis.trust.certificate import calibrate

cal = calibrate([
    ("mine", {"C": C}, problem, True),                        # True: this field really is inside the domain
    ("mine, advecting at 1.6", {"C": wrong}, problem, False),  # False: this one left it
])
print(cal.table())
# gate                         detection  false alarms   (cases)
# finite                           0.000         0.000   (0/1 out, 0/1 in)
# units:C                          0.000         0.000   (0/1 out, 0/1 in)
# residual:advection-diffusion     1.000         0.000   (1/1 out, 0/1 in)
cal.rate("residual:advection-diffusion")   # (1.0, 0.0)
cal.to_dict()                              # schema pinneapple.calibration/1, with every case
```

The table that ships with the tests separates 0.039 (0.2 % of noise, a pass) from 0.095 (0.5 %, flagged) at a
tolerance of 0.05 on a 64 × 41 grid — a boundary of *that* grid and *that* tolerance, which is exactly why the
in-domain level of the model wins over a constant.

### The same question on five physics families and one CFD surrogate

The calibration above is yours to declare. The repository also measures the certificate itself, and the measurement
is reproducible in six seconds:

```bash
python scripts/certificate_benchmark.py                          # writes results/certificate_five_families.json
python scripts/certificate_benchmark.py --check                  # fails if the committed JSON is out of date
```

Each of the five families (`pinneapple_analysis.trust.benchmark`) is an exact analytic solution — advection-diffusion,
heat, the Taylor-Green vortex, a shallow-water vortex in gradient balance, a plane-strain equilibrium field — with
three in-domain fields (the solution and two noise levels) and three out-of-domain ones (a wrong parameter, another
family's solution, the sign inverted). The sixth group is a real surrogate: the repository's POD `OperatorSurrogate`
over the Taylor-Green vortex, asked four viscosities inside its fitted envelope and four outside it. As measured:

| group | cases | detection | false alarms |
|---|---|---|---|
| advection-diffusion | 3 in / 3 out | 0.667 (2/3) | 0.000 (0/3) |
| heat | 3 in / 3 out | 0.667 (2/3) | 0.000 (0/3) |
| incompressible-flow | 3 in / 4 out | 0.750 (3/4) | 0.000 (0/3) |
| shallow-water | 3 in / 3 out | 1.000 (3/3) | 0.000 (0/3) |
| elasticity | 3 in / 3 out | 0.667 (2/3) | 0.000 (0/3) |
| cfd-surrogate | 4 in / 4 out | 0.750 (3/4) | 0.000 (0/4) |

Nineteen in-domain fields out of nineteen certified; fourteen out-of-domain fields out of twenty refused. The six the
certificate let through are the finding, and they are named in the report instead of averaged into the rate:

- **the sign inverted**, on advection-diffusion, heat and elasticity. Those three laws are linear and homogeneous with
  the parameters declared, so the relative residual is invariant under `C -> -C`: 0.000748 where the exact solution
  also gives 0.000748, 0.000445, and 2e-13. A residual gate cannot see it — and a consumer who needs to does not have
  to wait for one, because a declared boundary is a different measurement:

  ```python
  problem = Problem(physics="advection-diffusion", axes={"x": x, "t": t},
                    parameters={"c": "1 m/s", "alpha": "0.05 m2/s"}, units={"C": "1"},
                    tolerance={"advection-diffusion": 0.05},
                    boundary={"t_min": {"C": lambda pts: np.sin(pts[:, 0])}})   # what C must be at t = 0

  verify({"C": C}, problem).badge()      # 'good'        -- the boundary check passes at 0.0
  verify({"C": -C}, problem).badge()     # 'fading'      -- boundary:t_min.C fails at 1.0 of the field's range
  ```
- **the undamped direction of a viscosity error**, on the flow family. The vortex that does not decay at all is judged
  against `nu = 0.02` and its momentum residual lands at 0.0279, under the 0.05 limit, while the same error in the
  other direction (decaying at `nu = 0.2`) is flagged at 0.2272. At this Reynolds number the viscous term the error
  changes is small next to the advection.
- **one small extrapolation of the surrogate**, `nu = 0.02` against a fit on `[0.05, 0.4]`. Its momentum residual is
  0.0412 and the fitted domain gate accepts the query, so it is certified; `nu = 0.6`, `0.8` and `1.2` are all
  rejected, on the residual (0.1346, 0.2383, 0.3865) and on the domain gate.

## 7. What this round does not do

- An OpenFOAM case folder is **read** — the mesh, the patches, the cell values — but it is **not judged**: on an `fv`
  result the certificate reports `input` as `unmeasured` and the badge is `fading`, because its residuals would need
  the conservative face sums of `pinneapple_check.fv` and this round has none. Export the same field on a grid or as a
  point cloud and it is judged like any other.
- There is no HTTP endpoint and no upload page yet: the library call is the whole surface.
- The measurement is the one on this page — five analytic families and one POD surrogate
  (`results/certificate_five_families.json`). It is not a survey of other people's models, and the out-of-domain kinds
  it covers are the three a surrogate produces when it leaves its regime: a wrong parameter, another law's solution,
  and the sign inverted. A surrogate that is confidently wrong in some fourth way is not represented here.

Where it sits: the certificate **embeds** a `Trust` record (schema `pinneapple.trust/1`) and keeps its sections — the
horizon, the readings, the calibrated uncertainty — as they are. It fills the residuals and the domain from its own
measurement and invents nothing for the sections it did not measure.
