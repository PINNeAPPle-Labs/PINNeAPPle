# Engineering Model Lineage

**How was this result produced? The digital thread from CAD to prediction.**

Day 11 of the PINNeAPPle 30-app program. A prediction is only as trustworthy as the chain behind it: which geometry
revision, which mesh, which solver version, which simulations trained the model. Drop a project folder (or a zip, or a
lineage JSON) and get that chain as a graph you can question and export.

```
CAD-001 (rev A) → MESH-014 (1.2M cells) → SIM-382 (OpenFOAM v2312) → DATA-19 → MODEL-07 → PRED-883
```

Every artifact carries **file, version, software, parameters, timestamp, sha256, owner and origin**.

- **Auto-detection** from the files: OpenFOAM cases (blockMeshDict → polyMesh → run, solver and version read from the
  log, dates, turbulence model and settings), Gmsh `.geo` → `.msh`, CalculiX `.inp` → `.frd`/`.dat`, and datasets and
  reports exported by the other PINNeAPPle apps (linked through the source hashes in their metadata).
- **Declared lineage**: a `lineage.json` in the folder (or uploaded alone) is merged with what was detected, and every
  declared hash is checked against the uploaded file.
- **Checks**: a file that changed since it was recorded (hash mismatch), an output older than its input (stale), an
  input that is not in the lineage, cycles, a dataset or model that mixes revisions of the same item, incomplete
  provenance.
- **Questions**: which geometry revision produced this; which simulations is it built on, with which solver; if this
  changes, what has to be redone.
- **Exports**: the lineage JSON (`pinneapple.lineage/1`), **W3C PROV-JSON**, and a Markdown record.

```json
{"project": "bracket", "artifacts": [
  {"id": "CAD-001", "kind": "geometry", "version": "A", "file": "bracket_revA.step", "hash": "<sha256>", "owner": "design"},
  {"id": "MESH-014", "kind": "mesh", "inputs": ["CAD-001"], "software": "snappyHexMesh", "version": "v2312"},
  {"id": "SIM-382", "kind": "simulation", "inputs": ["MESH-014"], "software": "OpenFOAM simpleFoam", "version": "v2312"}]}
```

The minimal chain `{"geometry": ..., "mesh": ..., "solver": "OpenFOAM v2312", "simulation": ..., "model": ..., "result": ...}`
is accepted too (`solver` becomes the simulation's software and version).

## Examples

| Example | Verdict | Why |
|---|---|---|
| OpenFOAM pitzDaily study (3 real runs, an Interop Hub dataset, a Comparator report, the team's lineage.json) | FAIL | the fine case's `blockMeshDict` no longer matches the hash recorded before its cells were doubled |
| PINNeAPPle PINN plate example | PASS | 7 artifacts, every hash verified against the repository files |
| Illustrative digital thread (fictitious IDs) | WARNING | the prediction is older than the model it came from; the dataset mixes two CAD and mesh revisions |
| Simple chain | PASS | the minimal input |

## Run

```bash
pip install -e . -r apps/model_lineage/requirements.txt        # from the repository root
cd apps/model_lineage && uvicorn model_lineage.api:app --port 8090
```

Environment: `LIN_USER` / `LIN_PASSWORD` (HTTP Basic login), `LIN_MAX_MB` (default 300), `LIN_MAX_HEAVY` (default 2).
Docker: `docker build -f apps/model_lineage/Dockerfile -t model-lineage .`; deployment with the other apps in
[`apps/deploy`](../deploy) (service `lineage`). Tests: `tests/test_model_lineage.py`.

## API

`POST /api/lineage` (multipart `files`: a folder's files, a zip, or a lineage JSON) returns nodes, edges, checks, the
verdict, the lineage JSON, PROV-JSON and Markdown. Library: `from pinneapple_data.lineage import Lineage, detect_project`.
