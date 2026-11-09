# Viscous vortex domains: 2-D Navier–Stokes without a mesh

`pinneapple_simulation.vortex_domains` solves the two-dimensional incompressible Navier–Stokes equations in
velocity–vorticity form, Lagrangian and mesh-free: the vorticity of the flow is carried by **vortex domains** —
blobs of circulation `gamma` and core radius `eps` — that move with the velocity the whole field induces on them,
while viscosity enters as a deterministic *drift* down the vorticity gradient. The method is Dynnikova's viscous
vortex domains (VVD): G. Ya. Dynnikova, *The Lagrangian approach to solving the time-dependent Navier–Stokes
equations*, Doklady Physics 49 (2004); G. Ya. Dynnikova & A. V. Dynnikov, *Viscous vortex domains method*, the
VVF formulation documented with the reference implementation [`vvflow`](https://github.com/vvflow/vvflow) (MIT),
reimplemented here from that documentation and the papers it cites. The reference case -- a NACA 0012 with a
deploying spoiler at `Re = 4e4` -- is the simulation of [the video that motivated
this](https://www.youtube.com/watch?v=H-snLmMQK0Y) (Yaroslav Dynnikov, `vvflow`).

Nothing here is a turbulence model and nothing decides in advance where the flow separates: the boundary layer, its
separation and the wake are all the same mechanism -- the wall hands vorticity to the fluid at the rate Lighthill's
relation asks for, the stream carries it away, and the drift spreads it.

## What it is made of

| Piece | Where | What it does |
|---|---|---|
| domains | `Domains` | the blobs: positions, circulations, core radii (second-neighbour distance, floored by a third of the panel), the merge that bounds their number, removal that reports the circulation it takes out |
| kernels | `kernels` | the regularised Lamb–Oseen Biot–Savart sum; the closed-form source and vortex sheets on straight panels; the two diffusion drifts (domain–domain and wall) |
| bodies | `Body`, `Motion`, `Assembly`, `PanelSet` | rigid polygonal bodies that translate, rotate and hinge; generators `cylinder`, `ellipse`, `plate`, `naca4`, `naca0012`, `deploying_spoiler` |
| wall | `SourceSheet`, `VortexSheet`, `vorticity_flux` | no through flow (source sheet, exact), no slip (vortex sheet, for the pitch/rotation cases), and Lighthill's vorticity flux `U_e dU_e/ds` |
| solver | `VVD2D`, `VVDConfig` | the time loop (midpoint, advect + drift), the forces (wall pressure, wall shear), the history of the run |
| fields | `grid_field`, `surface_pressure`, `surface_friction`, `streamlines` | velocity, vorticity, pressure, stream function, `Cp` and `Cf` on the wall, streamlines |
| cases | `cylinder_case`, `naca0012_spoiler_case` | the verification case and the reference case |
| study | `run_study` | mean coefficients, Strouhal, the physics checks, a `study.json` to save |
| verification | `verification.run` | what the checkout reproduces, check by check |

## The reference case: a NACA 0012 with a deploying spoiler

```python
from pinneapple_simulation.vortex_domains import naca0012_spoiler_case, run_study, surface_pressure, grid_field

case = naca0012_spoiler_case(re=40_000.0, panels=180, spoiler_panels=24,
                             t_start=0.4, deploy_time=0.5, t_end=1.4)   # the numbers of the reference video
result = run_study(case)
print(result.summary())                      # Cl, Cd, Cm before and after the spoiler stands up, and the checks
sim = result.simulation
mid, cp = surface_pressure(sim)              # Cp on the wall
field = grid_field(sim, extent=(-0.5, 2.0, -0.6, 0.6), n=200)
```

What a run of it shows (180 section panels, 24 spoiler panels, `Re = 4e4`, deployment starting at `t = 0.4` and
taking `0.5`):

- **Flush, before the deployment.** `Cl = 0.00` and `Cd = 0.02` -- a symmetric section at zero incidence owes an
  observer exactly zero lift, and gets it (to `1e-3`), with a drag of the right order.
- **While it rises.** Both loads move sharply -- in the runs measured here the lift swung through `-4.7` and the
  drag climbed to `0.98` -- and then **the run is lost**: the number of domains grows to thousands, the reported
  loads diverge, and the `displacement` check fails. This is reproducible: at 180 section panels and 24 spoiler
  panels the deployment is the configuration the current wall model does not survive (one longer run made earlier
  in the same code state did reach `t = 1.33` with `Cl = 0.70`, `Cd = 0.38`, and that is *not* reproducible, so do
  not rely on it). The spoiler is the honest open case of this module: the moving wall asks for vorticity at a
  scale the panels cannot hold, and the flux cap only bounds the symptom.

The geometry follows what the reference case states: a spoiler of 10 % of the chord whose trailing edge sits 20 %
of the chord ahead of the airfoil's trailing edge, hinged at `0.8 c` on the upper surface (so flush it covers the
last tenth of the chord) and rising to 90° about that hinge. `deploy()` is the smooth ramp; set
`deploy_angle_deg=0` to leave it flush.

Honest scope, in three points:

- **Resolution.** At `Re = 4e4` the boundary layer is of order `c/sqrt(Re) = 0.005 c` -- thinner than the panels a
  laptop-sized run can afford (`eps ~ 0.004 c` here). The case exercises the mechanism and produces a force
  history (lift collapses and drag rises as the spoiler stands up), and its numbers are qualitative.
- **Trailing edge.** The section is truncated at 97 % of the chord. A sharp trailing edge leaves a potential-flow
  singularity there (the edge velocity reaches 15 to 50 times the free stream, measured in the tests) and this
  solver imposes no Kutta condition to remove it; the blunt base keeps the edge velocity at the physical `1.2 U`.
- **No slip.** The wall creates vorticity through Lighthill's flux and never enforces zero tangential velocity, so
  the boundary layer is right in its vorticity flux and too thin in its resistance -- see "What is verified" below
  for what that costs on the drag.

## The verification case: a cylinder

```python
from pinneapple_simulation.vortex_domains import cylinder_case, run_study

result = run_study(cylinder_case(re=40.0, panels=96, t_end=10.0))
print(result.summary())
for check in result.checks:
    print(check["name"], check["ok"], check["detail"])
```

The references the case carries: `Cd = 1.3375` at `Re = 40` (Dennis & Chang, *J. Fluid Mech.* 42, 1970) and the
shedding law `St = 0.2731 - 1.1129/sqrt(Re) + 0.4821/Re` of Williamson & Brown (*J. Fluids Struct.* 12, 1998),
which gives `St = 0.1968` at `Re = 200`.

## What is verified, and what is not

`python -m pinneapple_simulation.vortex_domains` prints the table below for the checkout it runs in; `--full` adds
the two cylinder runs.

| Check | What it asks | Status |
|---|---|---|
| `panel_influence` | the closed-form source and vortex panels against a 20 000-point quadrature of the point kernels | **exact**: relative error `5e-10` |
| `potential_flow` | `Cp = 1 - 4 sin^2 theta` on a cylinder in a uniform stream, from the source sheet alone | **exact**: `4e-15`, and the net force is zero (d'Alembert) to `1e-16` |
| `spinning_cylinder` | the attached sheet of a cylinder spinning in a fluid at rest is uniform and equal to its surface speed | `0.9 %` |
| `diffusion` | the diffusion drift of a Gaussian vorticity patch is `2 nu r / sigma^2` | `37 %` (the kernel estimate is a one-parameter fit: the second-neighbour core is what sets it) |
| `cylinder_drag` | `Cd` of the cylinder at `Re = 40` against `1.3375` | **not reached**: measured below (see the note) |
| `strouhal` | shedding frequency at `Re = 200` against `0.1968` | see the measured value |

The drag is the honest gap. The wall creates vorticity at Lighthill's rate -- which sets the *flux*, and that flux
is what a vortex method's separation and wake come from -- but the tangential velocity at the wall is left free, so
the flow slips over the body instead of being held back by it, and the `no_slip` check reports it (the wall slip
stays at the potential-flow value). Measured on this checkout, against `1.3375`: `Cd = 1.01` at 24 panels (`t = 0.35`), `0.84` at 32 (`t = 0.6`) and
`0.55` at 96 (`t = 1.9`) -- and *worse* with more panels, which is the signature of a vorticity budget that is not
converging rather than of a discretisation error: the diffusion drift has magnitude `nu/eps`, so refining the core
radius refines the *drift* too and the boundary layer is spread faster than viscosity would spread it. The mean lift at zero incidence is zero to `5e-5`, which is
what a symmetric body must give, and the impermeability and stability checks pass throughout. The next step is written down: a
coupled wall system in which the attached vortex sheet is solved for no slip and only the part of it that diffuses
away is shed (the Koumoutsakos–Leonard formulation); the pieces for it are in `boundary.py` (`VortexSheet`, whose
spinning-cylinder check passes) and the obstacle is the representation of a wall sheet by a row of blobs, which is
what `Wall` (kept in the tests as `VortexSheet` + `SourceSheet`) was measuring.

## What to expect of a run

- **Cost.** `O(N^2)` in the domains plus `O(N N_panels)`: a few thousand domains is a minute of laptop time per
  unit of time at `Re = 40`; `Re = 4e4` at the resolution above runs `dt ~ 2e-3` and is minutes to tens of minutes.
- **Stability.** `dt` is the tighter of `Re eps^2` and `eps / U` (`step_size`), the drift is clamped to half a core
  radius per step (`drift_limit`), and the wall flux is capped at `4 U^2` (`flux_limit`) so that a sharp edge cannot
  ask the wall for vorticity no domain could carry. The checks report the largest domain step against its core.
- **Resolution.** `max_domains` bounds the number of domains; when it is reached the core radius floor grows, the
  domains merge, and the run continues coarser -- and says so (`history["eps_floor"]`, and the `resolution` check).
