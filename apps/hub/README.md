# PINNeAPPle Hub

The catalogue of ready-made tools, trained models, geometries, datasets, flows and examples of PINNeAPPle: search,
see an item with its recorded run, copy its code, download its files, open it in Flow Studio, run a zoo model in
the browser or on the server, and compose your own tool from existing blocks. Docs: [docs/apps/hub.md](../../docs/apps/hub.md).

Everything shown comes from the generated index of `pinneapple_hub.library` (`data/web/hub/index.json`); this
folder is only routes (`hub/api.py`) and the page (`hub/static`).

## Run

```bash
pip install -e . fastapi "uvicorn[standard]" python-multipart
cd apps/hub
PYTHONPATH=../..:../_shared:. uvicorn hub.api:app --port 8116      # http://127.0.0.1:8116
```

With Docker, from the repository root:

```bash
docker build -f apps/hub/Dockerfile -t pinneapple-hub .
docker run -p 8116:8116 -v hub_data:/data -e HUB_FLOW_STUDIO_URL=https://flow.example.org pinneapple-hub
```

On the server: the `hub` service of `apps/deploy` (`HUB_DOMAIN=hub.pinneapple.org`, see
[apps/deploy/README.md](../deploy/README.md#pinneapple-hub-hub)).

## API

| Route | What |
|---|---|
| `GET /api/config`, `GET /api/home` | counts, facets, seal colours, featured items, categories |
| `GET /api/search?q=&kind=&domain=&quality=&subkind=&limit=&offset=` | ranked cards |
| `GET /api/items/{id}` | the whole item (code, recorded run, results, files, links, related) |
| `GET /files/{path}` | a file the index points to (only under `data/web`, `data/geometries`, `lab/runs`, `examples`, ...) |
| `POST /api/models/{name}/predict` | `{"inputs": {...}}` -> outputs, validity range, query checks, verdict |
| `GET /api/compose/blocks`, `POST /api/compose` | blocks; a composed tool's flow JSON, Python and issues |
| `POST /api/compose/validate` | a job that runs the composed tool's Python in the sandbox (`GET /api/jobs/{id}`) |
| `GET /api/compose/download/{py,json}?spec=` | the composed tool as a file |
| `GET/POST /api/mine`, `DELETE /api/mine/{id}` | My Hub |

Plus the shared runtime routes (jobs, usage, API keys) of `apps/_shared`.

## Environment

All optional (runtime and security variables: [apps/_shared/README.md](../_shared/README.md)).

| Variable | Default | Meaning |
|---|---|---|
| `HUB_USER`, `HUB_PASSWORD_HASH` | unset | HTTP Basic login |
| `HUB_FLOW_STUDIO_URL` | `http://127.0.0.1:8765` | where "Open in Flow Studio" goes |
| `HUB_SANDBOX_S` | 60 | time limit of a validation run |
| `HUB_SANDBOX_MEMORY_MB` | 6144 | address-space limit of a validation run (PyTorch reserves much of it) |
| `HUB_MAX_JOBS` | 2 | validation runs at once |
| `HUB_FREE_VALIDATIONS`, `HUB_FREE_INFERENCES` | 200, 5000 | free use per caller and month |
| `HUB_INDEX` | `data/web/hub/index.json` | another index file |

## Tests

```bash
pytest tests/test_hub_library.py tests/test_hub_app.py tests/test_hub_e2e.py -q
```
