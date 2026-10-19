# Meshless Method Lab

**Meshless or mesh-based: measure where each wins on your problem.** Day 26 of the PINNeAPPle apps program; live at
<https://meshless.pinneapple.org> (port 8109).

**Pain.** Generating and maintaining good meshes is hard on complex or moving domains. Meshless methods promise to
skip it, but they bring their own costs (dense or ill-conditioned systems, shape parameters, stencil choices) and
neural solvers bring training budgets. Which one is right for this domain, this equation and this tolerance is an
empirical question: this lab runs the comparison instead of assuming an answer.

| You give it | You get |
|---|---|
| An equation: Poisson, Helmholtz or heat with Dirichlet conditions, from a manufactured solution (any expression in x, y, t) or from a source, boundary and initial value; a domain: square, disk, L-shape, annulus, plate with a hole, any polygon with circular holes, or an uploaded point cloud (CSV: x, y, boundary); node spacings; methods and their settings; a tolerance | For every method and spacing: relative L2 and max error at 2,000 independent evaluation points (and at the nodes), setup and solve time, 1-norm condition number, noise amplification, non-zeros; the observed order of convergence, condition-number growth, the Kansa shape-parameter sweep, error maps; **where each method wins**: accuracy at the finest spacing, cheapest to reach the tolerance, fastest convergence, best conditioned, least sensitive to perturbed data |

**Methods** (`pinneapple_simulation.meshless`): Kansa (global multiquadric collocation; dense, at most 4,000 nodes),
RBF-FD (polyharmonic spline r³ + polynomials up to degree 2–6 on the nearest nodes; sparse), finite differences on a
Cartesian grid embedded in the domain with the Shortley–Weller formula at the curved boundary, linear finite elements
on the Delaunay mesh of the same nodes (cells outside removed), and a PINN (registry MLP, autograd Laplacian) on the
same nodes within a wall-clock budget (`train_within_budget`). Heat problems march with Crank–Nicolson; without an
exact solution the reference is linear FEM at a third of the finest spacing, with its own estimated error.
R-OAMS is not in the library, so it is not offered.

**Outputs and integrations.** `StudyResult` contract (table of every run, convergence orders, winners, checks with
their values, limits), the solutions of every method at the evaluation points (CSV), the full study (JSON); REST API
with a job queue; point clouds from any pre-processor as CSV/Parquet.

## Validation (`tests/test_meshless.py`, `tests/test_app_meshless_lab.py`)

| Case | Result |
|---|---|
| L-shape, smooth u = sin(πx)cos(2y) + xy, h = 0.1 → 0.025 | observed order FD 2.01, FEM 1.96, RBF-FD 4.36, Kansa 3.8 (stalls at 1e-6 with condition 1e20); FEM best conditioned (475), RBF-FD most accurate (7.4e-7) |
| L-shape, corner solution r^(2/3) cos(2ψ/3) | every method falls to order ≈ 1.3 (the singularity); FEM most accurate (1.9e-3); Kansa gets worse with refinement |
| Plate with a hole, heat, e^(-t) sin(πx) sin(πy) + x | RBF-FD 5e-7; FD and FEM ≈ 6e-4; time-stepped Kansa unstable at h ≤ 0.05 |
| Disk, (1 − r²)/4 against the fine-FEM reference mode | the reference and its estimated error agree with the exact solution |

The PINN numbers depend on its budget (3 s in the examples: 0.37 relative L2 on the smooth L-shape).

## Run

```bash
pip install -e . -r apps/meshless_lab/requirements.txt          # from the repository root
cd apps/meshless_lab && uvicorn meshless_lab.api:app --port 8109
```

Environment: `MESHLESS_USER` / `MESHLESS_PASSWORD` (HTTP Basic login), `MESHLESS_MAX_JOBS` (default 2),
`MESHLESS_MAX_NODES` (per level, default 20,000), `MESHLESS_MAX_PINN_S` (default 30), `MESHLESS_DATA_DIR`,
`MESHLESS_ADMIN_TOKEN`, `MESHLESS_FREE_RUNS` / `MESHLESS_FREE_UNKNOWNS` / `MESHLESS_FREE_REPORTS` (40 / 2,000,000 /
100 per caller and month). Docker: `docker build -f apps/meshless_lab/Dockerfile -t meshless-lab .`; deployment in
[`apps/deploy`](../deploy) (service `meshless`).

## API

`POST /api/studies` (multipart `spec` JSON and optional `cloud` file) queues a study; `GET /api/jobs/{id}` gives the
progress and the result; `GET /api/jobs/{id}/files/{study_result.json|solutions.csv|study.json}`; `POST
/api/examples/{lshape|corner|heat|helmholtz}`; `POST /api/domain` (node preview); `POST /api/expression` (checks a
problem). Interactive documentation at `/docs`.

## Limits

- 2-D, scalar, linear, Dirichlet only: linear elasticity and Neumann / Robin conditions are not implemented yet.
- Manufactured solutions are smooth unless you make them otherwise: they favour high-order methods (try the corner).
- Errors at the evaluation points use each method's own interpolant (FD and FEM: P1 between nodes).
- Node sets are a lattice plus boundary nodes at the same spacing; cost is wall time on this server.
- Condition numbers are 1-norm estimates of unequilibrated systems (FD rows near the boundary raise them); the noise
  amplification measures the effect on the solution.

Monetization: runs, problem size (unknowns solved) and reports, with private projects and API keys for teams.
