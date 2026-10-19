# Engineering GUI Agent

**guiagent.pinneapple.org** · port 8115 · day 27 extra · library: `pinneapple_tools.automation`
([guide](../../docs/guides/engineering_gui_agent.md))

![Engineering GUI Agent: a run waiting for approval before the solver](../../docs/assets/apps/d27-guiagent-gate.jpg)

**Pain.** A routine engineering study means clicking through CAD, a CFD pre-processor, a solver and a
post-processor: change a parameter, regenerate, mesh, solve, export, check. It is slow, easy to get subtly wrong (a
boundary condition that is not what you set, a mesh in the wrong units), and afterwards nobody can say exactly what
was done.

**What it does.** You give it a task: a sentence ("obstacle height 55 mm at 20 mm/s") or a structured workflow. It
runs the task in the engineering software inside its container and hands back the files plus the evidence of every
step. Think of it as Selenium for computational engineering. The application's API is always used first (FreeCAD
Python, CfdOF Python, the OpenFOAM command line, `pvpython`). GUI steps run in FreeCAD's real GUI on a virtual
display, are observed with xdotool and are screenshotted. Every step is checked against the files it produced. The run
stops on the first failure, and it stops before the solver until a person approves.

## Input

- A task sentence: rule-based by default, or read by a local LLM (Ollama) if one is configured. It picks a
  **committed** workflow and sets its inputs; units are converted and bounds checked.
- Or a workflow and its inputs: `{"workflow": "channel_obstacle", "inputs": {"obstacle_height_mm": 55}}`.
- Self-hosted mode also accepts your own workflows: a zip with `workflow.yaml`, your FreeCAD project and macros.

## Output

For each step, the record shows:

- its state and duration;
- the commands, with their logs;
- the actions taken, including compatibility edits;
- the produced files, with SHA-256;
- images, each labelled with what it is (virtual-display screenshot, 3D-view crop, application export, chart from a log);
- the checks.

The run as a whole delivers:

- the acceptance criteria, each with the measured value and the threshold;
- an HTML evidence report (print it to PDF from the browser);
- `study_result.json`, a `pinneapple_data.contracts.StudyResult` (the contract the apps share) for the Evidence
  Report app (day 30);
- a zip of everything: the modified FreeCAD document, the CfdOF/OpenFOAM case, the velocity field as CSV, images,
  logs.

## The committed workflow: channel with an obstacle

| # | Step | Mode |
|---|---|---|
| 1 | Open the parametric FreeCAD project (spreadsheet aliases) | GUI (FreeCAD on Xvfb) |
| 2 | Run the macro that sets the obstacle height; check the new volume against the parameters | GUI |
| 3 | Build the CfdOF case: laminar steady air, velocity inlet, pressure outlet, gmsh mesh | GUI (CfdOF needs FreeCADGui) |
| 4 | Mesh (gmsh, gmshToFoam), then checkMesh; check the mesh is in metres | API |
| 5 | Solve with simpleFoam. **Asks for confirmation** (expensive) | API |
| 6 | Export U and p of every cell to CSV and \|U\| on the mid-plane to PNG with ParaView | API |
| 7 | Criteria: converged; outflow = inflow within 1 %; inflow = prescribed velocity × area within 1 %; pressure drop below 0.01 Pa (an illustrative budget); velocity exported | API |

Measured on this image at 20 mm/s:

| Obstacle | Iterations | Pressure drop | Decision |
|---|---|---|---|
| 25 mm | 119 | 1.9 mPa | accepted |
| 55 mm | 162 | 3.5 mPa | accepted |
| 80 mm | 223 | 14.1 mPa (14.3 in the Docker run, 212 iterations) | rejected |

A run takes 30-45 s (about 42 s in the Docker image with its 2 GB limit, using 0.9 GB after the run). While building the example, the agent's checks caught a mesh left in millimetres, a
diverging initialisation and an inlet that delivered 6 % less flow than prescribed. All three now come from real runs,
are handled, and are recorded in the evidence (see the guide).

## Integrations

- **Tools companies already use:** FreeCAD, CfdOF, OpenFOAM, ParaView. A Playwright adapter covers web UIs. A
  pywinauto adapter interface is documented for Windows tools (Ansys Workbench, Abaqus/CAE); it does not run in this
  Linux container.
- **PINNeAPPle:** the criteria use `pinneapple_check`'s finite-volume reader, and the logs are read by
  `pinneapple_data.simulation_metadata`. Results go out as a StudyResult for the Evidence Report app, and the run
  folder is plain files.
- **Simulation Workflow Runner (day 27):** the runner gains a step kind `guiagent`
  (`pinneapple_tools.automation.runner_step`). One runner step runs a committed workflow with given inputs and
  approvals, and its verdict and metrics flow downstream. The public runner image has no FreeCAD, so the step kind
  shows what is missing there; a self-hosted runner with the applications runs it.
- **Shared runtime (`apps/_shared`):** runs execute in the shared job queue, belong to the caller (IP, login or
  `X-API-Key`), and are metered in `runs`, `compute_seconds` and `storage_mb` (`GET /api/usage`).

**Monetization: jobs, compute time, storage.** These are the metered units, with free monthly limits per caller
(`GUIAGENT_FREE_RUNS` 20, `GUIAGENT_FREE_COMPUTE_SECONDS` 1800, `GUIAGENT_FREE_STORAGE_MB` 500). Paid tiers would add:

- more concurrent jobs, and longer or bigger cases (meshes, iterations);
- run storage and retention;
- self-hosted deployments, with your own workflows, macros and applications (SALOME, Windows tools).

## Limits

- One application chain and one committed workflow. SALOME (scriptable geometry and meshing) is the next adapter, and
  ParaView-only post-processing workflows are the cheapest to add; neither is implemented yet.
- Laminar, steady, one mesh (about 19 000 tetrahedra), and no mesh-independence study. gmsh runs several threads, so
  the cell count varies by a few hundred between runs.
- OpenFOAM v1912 (Ubuntu package) with CfdOF 1.37.3, which targets newer OpenFOAM. The agent makes two recorded edits
  for v1912 and does not use potential-flow initialisation.
- GUI steps are driven through FreeCAD's Python inside the GUI. xdotool finds and observes the windows; pixel clicking
  is available in the library but not needed here. There is no live remote desktop (noVNC): you see screenshots per
  step.
- The public demo runs one job at a time (others wait in the queue) and only committed workflows. Uploads and macros
  are for self-hosted mode only.

## Run

```bash
docker build -f apps/guiagent/Dockerfile -t guiagent .          # from the repository root; about 5.4 GB uncompressed, 1.8 GB compressed
docker run -p 8115:8115 guiagent                                 # http://localhost:8115, API at /docs
```

Without Docker (FreeCAD, CfdOF, OpenFOAM, ParaView, Xvfb and xdotool installed; see the guide for the variables):
`pip install pinneapple -r apps/guiagent/requirements.txt`, then `cd apps/guiagent && uvicorn guiagent.api:app --port
8115`. Environment variables: `GUIAGENT_USER`/`GUIAGENT_PASSWORD`, `GUIAGENT_MODE`, `GUIAGENT_MAX_JOBS`,
`GUIAGENT_KEEP_RUNS`, `GUIAGENT_FREE_*`, `GUIAGENT_ADMIN_TOKEN`, `GUIAGENT_DATA_DIR`,
`GUIAGENT_OLLAMA_URL`/`GUIAGENT_OLLAMA_MODEL` (the runtime's variables are listed in `apps/_shared/README.md`).

## API

| Endpoint | |
|---|---|
| `GET /api/workflows` | committed workflows: inputs and bounds, steps and whether each application is installed, criteria, limits |
| `POST /api/interpret` | a task to a workflow and inputs; runs nothing |
| `POST /api/runs` | start a run (`workflow` + `inputs`, or `task`) |
| `GET /api/runs/{id}` | live record |
| `POST /api/runs/{id}/approve`, `/decline` | answer the confirmation a step waits for |
| `GET /api/runs/{id}/report`, `/study_result`, `/bundle`, `/files/{path}` | the evidence |
| `POST /api/workflows` | self-hosted only: add a workflow (zip) |
| `GET /api/jobs/{id}`, `/api/usage`, `/api/keys` | the shared runtime: queue jobs, use this month, API keys |
