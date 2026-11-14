# Granular DEM: volume-based contact, breakage, mills and wear

`pinneapple_simulation.granular` is a discrete element solver for **convex polyhedra and spheres**, built on one idea
that changes what the contact law can do: the force is a function of the **volume two bodies share**, not of a
penetration depth.

That is the method the GPU DEM line of work (BLAZE-DEM / Govender et al.) is built on, and it exists because a
penetration depth is not a good state variable for an angular particle. A corner entering a face and two faces lying
on each other share very different volumes at the same "depth", so a depth-based law makes the same material feel
stiff or soft depending on which feature happens to touch. A volume-based law gives one stiffness per unit of shared
volume, which is what makes angular particles usable for a mill, a tablet or an ore without rounding them into
spheres.

```python
from pinneapple_simulation.granular import World, Material, box

world = World(gravity=(0.0, -9.81, 0.0))
world.add(box(0.02, 0.02, 0.02, density=2500.0), position=(0.0, 0.03, 0.0), material=Material(density=2500.0))
world.finalize()
world.run(steps=4000, dt=2.0e-5)
```

## What the layers do

| module | what it holds |
|---|---|
| `shapes` | convex polyhedra with exact volume, centroid, surface area and inertia (divergence theorem over the face loops), spheres, and the shape factories (`box`, `regular_prism`, `faceted_sphere`) |
| `overlap` | the separating axis theorem for polytopes, and the **exact** shared volume and its centroid by clipping one body's face loops against the other's planes |
| `contacts` | materials, the volume and Hertz laws, Cundall-Strack tangential friction with a persistent spring, rolling resistance, rigid walls (planes) and finite obstacles (lifters) |
| `world` | bodies, the neighbour search, the vectorised contact kernels, semi-implicit integration, and the measurements (kinetic energy, momentum, wall torque, per-patch impulse) |
| `bonds` | bonded elements: beams with axial, shear, bending and torsional stiffness that fail at a strength, the fragment decomposition and `P80` |
| `equipment` | a rotating drum with lifters, its charge, and the two numbers a plant asks for: power draw and the product curve |
| `wear` | the impulse wear law on liner patches, and its calibration (the least-squares `k` and its `R^2`) against a measured profile |
| `coupling` | fluid drag (Schiller-Naumann, Di Felice voidage) as a force source -- the particle side of a DEM-CFD coupling |
| `lattice` | crystal lattice -> lattice constant -> elastic constants -> polycrystal moduli -> Wulff particle shape ("atoms to material") |

## What is checked, and how

Every claim in this package has an independent answer behind it in the tests:

| statement | check |
|---|---|
| the shared volume is exact | cube-cube overlap against the analytic product; a rotated pair against Monte Carlo (3 %) |
| the geometry is exact | box volume, centroid, area and inertia against `m a^2 / 6`; a 24-gon prism against the polygon formula |
| a body at rest sits where the force balance puts it | measured penetration against `m g / k_v` (2 %) for a sphere and for a box |
| the step conserves what it must | momentum to 1e-12 and total (orbital + spin) angular momentum in an off-centre collision |
| the contact never creates energy | kinetic energy is non-increasing in a collision with restitution 0 |
| restitution is what it says | measured bounce ratio is monotone in the configured one, within -0.05 / +0.25 (the dashpot is viscous, so this is nominal) |
| friction holds a body on a slope up to a limit | a sphere stays below `atan(mu_roll)` and rolls above it |
| a bond fails at its strength | a 2 GPa bond survives an impact that a 1 kPa bond does not, and the fragments keep the mass |
| the mill draws power | `power = torque * omega` exactly, power rises with the speed, mass is conserved, the charge stays inside |
| the wear law is recoverable | the least-squares `k` recovers the coefficient the synthetic scan was made with, `R^2 = 1` |
| fluid drag lands on the same terminal velocity as the closed form | against `pinneapple_simulation.numerical_solvers.particles.terminal_velocity` (2 %) |
| a central-force crystal obeys the Cauchy relation and Born stability | `C12 = C44` to 1e-4 and the Born inequalities for fcc (stable) and sc/bcc (unstable), as they must be for Lennard-Jones |

## Speed

Measured on one core of an M-series laptop, a mill with six lifters running at 30 rpm:

| charge | step | wall time per step |
|---|---|---|
| 57 bodies | 8.0e-5 s | 4.7 ms |
| 125 bodies | 8.0e-5 s | 3.6 ms |
| 394 bodies | 6.4e-5 s | 7.4 ms |

That is a few hundred bodies and roughly 0.02 s of simulated time per wall second. It is enough for a process
picture and for validating the machinery, and it is **not** a GPU solver: the kernels are plain NumPy, and the
neighbour search and the contact kernels are the two places a CUDA port would replace. The arrays are already in
the flat form a GPU upload wants.

## Limits worth knowing before trusting a number

* **A single polyhedron in face contact with a plane is one contact point.** The friction force at that point
  produces a moment about the body's centre that nothing balances, because the normal pressure cannot shift across
  the patch in a single-point model. A block that has to tip or rest on a slope is a **bonded cluster of bodies**
  (which is why `bonds` exists), not one polyhedron; a sphere on a slope is fine (rolling resistance sets the
  holding angle).
* **Restitution is nominal.** The normal dashpot is viscous and capped at the elastic force; the recovered
  coefficient is monotone but not exact (see above).
* **The coupling is one-way.** The fluid pushes the particles; the particles do not push the fluid back yet, so a
  dense bed will not slow the carrier flow.
* **Wall faceting is a discretisation.** A barrel is 48 planes, which sits 0.2 % inside the true circle.
* **Timesteps come from the stiffness.** `World.recommended_timestep` and `RotatingDrum.steady_timestep` bound the
  step from the Rayleigh time, the contact stiffness and the stiffest bond spring. Never take a step larger than
  they report: a bonded assembly step is set by the bonds, and a too-large step heats the assembly silently.

## Run it

```bash
python examples/granular_dem/01_mill_power_and_wear.py
pytest tests/test_granular_geometry.py tests/test_granular_dem.py tests/test_granular_process.py -q
```

The example prints the power sweep (10, 40 and 70 rpm), the liner wear with its calibrated coefficient, and the
fragments of a drop-shatter test.

## References

* Cundall, P. A. & Strack, O. D. L. (1979). A discrete numerical model for granular assemblies. *Geotechnique* 29:47-65.
* Potyondy, D. O. & Cundall, P. A. (2004). A bonded-particle model for rock. *Int. J. Rock Mech. Min. Sci.* 41:1329-1364.
* Ai, J., Chen, J.-F., Rotter, J. M. & Ooi, J. Y. (2011). Assessment of rolling resistance models in discrete element simulations. *Powder Technology* 206:205-211.
* Cleary, P. W. (1998). Predicting charge motion, power draw, segregation and wear in ball mills using discrete element methods. *Minerals Engineering* 11:1061-1080.
* Di Felice, R. (1994). The voidage function for fluid-particle interaction systems. *Int. J. Multiphase Flow* 20:153-159.
* Schiller, L. & Naumann, A. (1935). Über die grundlegenden Berechnungen bei der Schwerkraftaufbereitung. *VDI Zeitschrift* 77:318-320.
* Born, M. (1940). On the stability of crystal lattices I. *Math. Proc. Cambridge Philos. Soc.* 36:160-172.
* Born, M. & Huang, K. (1954). *Dynamical Theory of Crystal Lattices*. Oxford University Press.
* Wulff, G. (1901). Zur Frage der Geschwindigkeit des Wachsthums und der Auflösung der Krystallflächen. *Z. Kristallogr.* 34:449-530.
* Hill, R. (1952). The elastic behaviour of a crystalline aggregate. *Proc. Phys. Soc. A* 65:349-354.
