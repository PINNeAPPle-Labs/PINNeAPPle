# Verify somebody else's prediction: the five-minute quick start

`pp.trust` answers *how much can I trust this model?* This answers the other question a consumer asks: *does this
field satisfy the physics it claims, and where does it stop doing so?* You bring the prediction and the description
of the problem; you get back a **certificate** — a JSON document that says, per check, the number, the limit it was
judged against, the point in the domain where it is worst, and where the number came from.

Nothing in it is a second engine. The residuals are the equation set of the linter (`pinneapple_check.physics`), the
reading is the linter's field reader (`pinneapple_check.fields`), the units are `pinneapple_data.physical_units`, the
in-domain level follows the ensemble's `residual_baseline`, and the domain gate is `TrustGate`. What the certificate
adds is the decision, the provenance of every limit, and the two numbers that make the decision meaningful.

## 1. A prediction you have, and the equation it claims

The exact solution of `dC/dt + c dC/dx = alpha d2C/dx2` — so you can see the certificate agree with something known:

```python
import numpy as np
from pinneapple_analysis.trust.certificate import Problem, verify

x, t = np.linspace(0, 2 * np.pi, 65)[:-1], np.linspace(0, 1, 41)
X, T = np.meshgrid(x, t, indexing="ij")
C = np.exp(-0.05 * T) * np.sin(X - T)                      # the exact solution, c = 1, alpha = 0.05

cert = verify(
    {"C": C},                                              # arrays, a .npz/.h5 grid, a VTK/VTU point cloud, a case
    Problem(physics="advection-diffusion",
            axes={"x": x, "t": t},
            parameters={"c": "1 m/s", "alpha": "0.05 m2/s"},   # SI numbers or strings with their unit
            units={"C": "1", "x": "m", "t": "s"},
            tolerance={"advection-diffusion": 0.05}),       # what you are willing to accept for this law
)

cert.badge()      # 'good'
cert.verdict()    # 'certified'
print(cert)       # one line per check: PASS 0.0007477 (limit 0.05)
```

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
cert = verify({"C": C}, Problem(..., domain={"t": (0.0, 0.5)}))     # the field runs to t = 1
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
against a truth the certificate cannot see:

```python
from pinneapple_analysis.trust.certificate import calibrate

cal = calibrate([
    ("in-1",  in_field_1,  problem, True),      # True: this case really is inside the model's domain
    ("out-1", out_field_1, problem, False),
])
print(cal.table())
# gate                         detection  false alarms   (cases)
# residual:advection-diffusion     1.000         0.000   (3/3 out, 0/3 in)
cal.rate("residual:advection-diffusion")   # (1.0, 0.0)
cal.to_dict()                              # schema pinneapple.calibration/1, with every case
```

The table that ships with the tests separates 0.039 (0.2 % of noise, a pass) from 0.095 (0.5 %, flagged) at a
tolerance of 0.05 on a 64 × 41 grid — a boundary of *that* grid and *that* tolerance, which is exactly why the
in-domain level of the model wins over a constant.

## 7. What this round does not do

- An OpenFOAM case folder is read, but its residuals are evaluated on the cell values as a point cloud, not with the
  conservative face sums (`pinneapple_check.fv`).
- There is no HTTP endpoint and no upload page yet: the library call is the whole surface.
- The calibration ships as a function and as a table in the tests; a published benchmark across the five families of
  the trust benchmark is the next slice.

Where it sits: the certificate **embeds** a `Trust` record (schema `pinneapple.trust/1`) and keeps its sections — the
horizon, the readings, the calibrated uncertainty — as they are. It fills the residuals and the domain from its own
measurement and invents nothing for the sections it did not measure.
