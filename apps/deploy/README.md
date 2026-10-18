# Deploy the apps on one server

HeatSink Sizer, PCB Hotspot, Engineering Data Health, Engineering Data Standardizer, Simulation Metadata, the Form
Compiler, Inverse Heat Lab, Simulation Preflight, Mesh Quality, Simulation Comparator, Engineering Model Lineage and the
Simulation Interoperability Hub, the
Aircraft Design Optimizer, the PINNeAPPle Observatory, the Inverse Model Studio, the Parameter Identifiability Lab, the Model Calibration Lab, the Engineering Evidence Report, the Physics Validation Lab and the PINNeAPPle Lab catalogue on a single Linux VM (e.g. Hetzner Cloud), each on its own
subdomain, behind one Caddy that obtains and renews Let's Encrypt certificates automatically.

```
                       ┌──────────── server ────────────┐
heatsink.example.org ─▶│ Caddy :80/:443 ─▶ heatsink:8080 │
pcb.example.org      ─▶│                ─▶ pcb:8081      │
                       └─────────────────────────────────┘
```

Only Caddy publishes ports. The apps are reachable only through it.

**Security.** The containers run as an unprivileged user (uid 10001) with a read-only root filesystem, no Linux
capabilities and no privilege escalation; Caddy adds TLS 1.2+, security headers, a body limit per site and a
rotated access log; each app adds its own login lockout, rate limits, CSP and audit trail. Set logins as hashes
(`<PREFIX>_PASSWORD_HASH`, from `python -m pinneapple_security hash-password`). Host hardening (SSH keys only,
firewall, fail2ban, automatic updates, Docker daemon, backups) and the read-only check `sudo ./check_server.sh`:
[docs/security/server.md](../../docs/security/server.md); credential rotation and incidents:
[docs/security/runbook.md](../../docs/security/runbook.md). **Upgrading** a server whose volumes were written by
the earlier root containers: run the one-time `chown` in docs/security/server.md before `docker compose up -d`.

> [!IMPORTANT]
> **Upgrading a server that already runs the apps? Run this once, before `docker compose up -d` with the new
> images.** The containers now run as an unprivileged user (uid 10001) instead of root, so the files the old root
> containers wrote into the named volumes must change owner, or the apps that keep state there (jobs, uploads,
> API keys, usage, audit trails) can not write to them:
>
> ```bash
> cd apps/deploy
> docker compose down                       # stop the old containers first
> for v in $(docker volume ls -q | grep -E '^pinneapple-apps_(.*_data|obs_cache)$' | grep -v caddy_data); do
>   docker run --rm -v "$v":/v alpine chown -R 10001:0 /v
> done
> docker compose build --pull && docker compose up -d
> ```
>
> Caddy's volumes (`caddy_data`, `caddy_config`) stay as they are. A fresh server needs nothing. Details and the
> rest of the host hardening: [docs/security/server.md](../../docs/security/server.md).

## 1. Server

- **Size:** 4 vCPU / 8 GB RAM is comfortable for the apps at demo traffic (e.g. Hetzner CX32 or CPX31). 2 vCPU / 4 GB works
  for demos if you lower `*_MAX_*` to 1 and `*_MEM_LIMIT` to `1500m`.
- **OS:** Ubuntu 24.04. Add your SSH key when creating it.
- **Firewall:** allow inbound TCP 22, 80 and 443, plus UDP 443 for HTTP/3.

## 2. DNS

At your DNS provider, create two **A** records (and AAAA records if you use IPv6) pointing at the server's IP:

| Name | Type | Value |
|---|---|---|
| `heatsink` | A | `<server IPv4>` |
| `pcb` | A | `<server IPv4>` |
| `datahealth` | A | `<server IPv4>` |
| `standardizer` | A | `<server IPv4>` |
| `simmeta` | A | `<server IPv4>` |
| `udr` | A | `<server IPv4>` |
| `inverse` | A | `<server IPv4>` |
| `preflight` | A | `<server IPv4>` |
| `mesh` | A | `<server IPv4>` |
| `compare` | A | `<server IPv4>` |
| `lineage` | A | `<server IPv4>` |
| `interop` | A | `<server IPv4>` |
| `aero` | A | `<server IPv4>` |
| `lab` | A | `<server IPv4>` |
| `sky` | A | `<server IPv4>` |
| `playground` | A | `<server IPv4>` |
| `surrogate` | A | `<server IPv4>` |
| `arena` | A | `<server IPv4>` |
| `models` | A | `<server IPv4>` |
| `rom` | A | `<server IPv4>` |
| `meshless` | A | `<server IPv4>` |
| `runner` | A | `<server IPv4>` |
| `campaign` | A | `<server IPv4>` |
| `uncertainty` | A | `<server IPv4>` |
| `sensitivity` | A | `<server IPv4>` |
| `validate` | A | `<server IPv4>` |
| `evidence` | A | `<server IPv4>` |
| `calibration` | A | `<server IPv4>` |
| `identifiability` | A | `<server IPv4>` |
| `twin` | A | `<server IPv4>` |
| `anomaly` | A | `<server IPv4>` |
| `control` | A | `<server IPv4>` |
| `inverse-studio` | A | `<server IPv4>` |
| `guiagent` | A | `<server IPv4>` |
| `optimize` | A | `<server IPv4>` |
| `design` | A | `<server IPv4>` |

If the zone is on Cloudflare, set both records to **DNS only** (grey cloud) for the first start, so
Let's Encrypt can reach Caddy. Check propagation with `dig +short heatsink.example.org`.

## 3. Docker

```bash
ssh root@<server IP>
curl -fsSL https://get.docker.com | sh
```

## 4. Code and configuration

```bash
git clone https://github.com/PINNeAPPle-Labs/PINNeAPPle.git /opt/pinneapple
cd /opt/pinneapple/apps/deploy
cp .env.example .env
nano .env        # domains, ACME e-mail, logins (use long passwords)
```

| Variable | Meaning |
|---|---|
| `HSS_DOMAIN`, `PCB_DOMAIN`, `EDH_DOMAIN`, `EDS_DOMAIN`, `SMD_DOMAIN`, `UDR_DOMAIN`, `IHL_DOMAIN`, `PFL_DOMAIN`, `MQA_DOMAIN`, `CMP_DOMAIN`, `LIN_DOMAIN`, `IOP_DOMAIN`, `ADO_DOMAIN`, `OBS_DOMAIN`, `PLG_DOMAIN`, `INVS_DOMAIN`, `IDENT_DOMAIN`, `CALIB_DOMAIN`, `EVR_DOMAIN`, `VAL_DOMAIN`, `SEN_DOMAIN`, `UQL_DOMAIN`, `MDL_DOMAIN`, `ARN_DOMAIN`, `SUR_DOMAIN`, `EDO_DOMAIN`, `PDO_DOMAIN` | Public hostnames (must match the DNS records) |
| `ACME_EMAIL` | Let's Encrypt account / expiry notices |
| `<APP>_USER` / `<APP>_PASSWORD` (HSS, PCB, EDH, EDS, SMD, UDR, IHL, PFL, MQA, CMP, LIN, IOP, ADO, OBS, UQL, SEN, VAL, EVR, CALIB, IDENT, INVS, ROM, MESHLESS, SUR, ARN, MDL, TWIN, ANOMALY, CONTROL, EDO, PDO) | HTTP Basic login per app. If either is empty, that app is public. `/health` is always public. |
| `HSS_MAX_SIZING`, `PCB_MAX_HEAVY`, `EDH/EDS/SMD/UDR/PFL/MQA/CMP/LIN/IOP/ADO/UQL/SEN/VAL/EVR_MAX_HEAVY` | Concurrent heavy runs per app. Any excess gets HTTP 429. |
| `HSS_MEM_LIMIT`, `PCB_MEM_LIMIT` | Container memory caps |
| `CALIB_MAX_JOBS`, `CALIB_FREE_<UNIT>`, `CALIB_ADMIN_TOKEN`, `CALIB_MEM_LIMIT` | Model Calibration Lab, on the shared runtime (job queue, API keys, usage metering; see `apps/_shared/README.md`): jobs running at once over all workers (the rest wait in the queue), free use per caller and month, the admin token for API keys, and the container memory cap (default 1g). Its state lives on the `calibration_data` volume. |
| `IDENT_MAX_JOBS`, `IDENT_FREE_<UNIT>`, `IDENT_ADMIN_TOKEN`, `IDENT_MEM_LIMIT` | Parameter Identifiability Lab, on the shared runtime (job queue, API keys, usage metering; see `apps/_shared/README.md`): jobs running at once over all workers (the rest wait in the queue), free use per caller and month, the admin token for API keys, and the container memory cap (default 1g). Its state lives on the `identifiability_data` volume. |
| `INVS_MAX_JOBS`, `INVS_FREE_<UNIT>`, `INVS_ADMIN_TOKEN`, `INVS_MEM_LIMIT` | Inverse Model Studio, on the shared runtime (job queue, API keys, usage metering; see `apps/_shared/README.md`): jobs running at once over all workers (the rest wait in the queue), free use per caller and month, the admin token for API keys, and the container memory cap (default 1g). Its state lives on the `inverse-studio_data` volume. |
| `IHL_MAX_JOBS` | Inverse Heat Lab: trainings running at the same time (default 2, each uses one CPU core). Any excess gets HTTP 429. The app runs one worker because jobs live in its memory. |
| `SUR_MAX_JOBS`, `SUR_MAX_BUDGET_S`, `SUR_MAX_SEEDS`, `SUR_FREE_*`, `SUR_ADMIN_TOKEN`, `SUR_REQUIRE_API_KEY` | Surrogate Studio: trainings at once (queue), largest budget and seed count per study, free use per caller and month, API-key management (see `apps/_shared/README.md`). |
| `ARN_MAX_JOBS`, `ARN_MAX_BUDGET_S`, `ARN_MAX_SEEDS`, `ARN_FREE_BENCHMARKS`, `ARN_ADMIN_TOKEN`, `ARN_REQUIRE_API_KEY` | Benchmark Arena: private benchmarks at once (queue), largest budget and seed count, free benchmarks per caller and month, API-key management (see `apps/_shared/README.md`). |
| `MDL_ADMIN_TOKEN`, `MDL_REQUIRE_API_KEY`, `MDL_MAX_ROWS`, `MDL_FREE_*` | Model Gateway: admin token for publishing and key management, mandatory API keys, rows per call, free use per caller and month. |
| `UDR_FORM_PDF` | Path inside the `udr` container to your copy of the fillable Form U-DR-1 (put the file in `apps/deploy/forms/`). Optional: users can upload it instead. |
| `UDR_OLLAMA_MODEL` | Optional. Enables local-LLM extraction in the Form Compiler with this Ollama model (pull it first). Every value it returns is checked against the document text; documents never leave your network. |
| `UDR_OLLAMA_URL` | Ollama server: `http://ollama:11434` (the optional `ollama` service, `--profile llm`) or `http://host.docker.internal:11434` (Ollama on the host). |
| `UDR_OLLAMA_TIMEOUT` | Seconds per LLM request (default 600; local models on CPU are slow). |
| `GUIAGENT_DOMAIN`, `GUIAGENT_USER` / `GUIAGENT_PASSWORD`, `GUIAGENT_MEM_LIMIT` | Engineering GUI Agent (optional heavy image): hostname, HTTP Basic login, memory cap (default 2g) |
| `GUIAGENT_MODE` | `public` (default): only the committed workflows run. `selfhosted`: `POST /api/workflows` also adds your own workflows, FreeCAD projects and macros. Macros are code, so never enable this on a public server. |
| `GUIAGENT_MAX_JOBS`, `GUIAGENT_KEEP_RUNS` | Runs at the same time (default 1; further runs wait in the shared job queue) and runs kept on disk (default 40; the oldest are deleted first) |
| `GUIAGENT_FREE_RUNS`, `GUIAGENT_FREE_COMPUTE_SECONDS`, `GUIAGENT_FREE_STORAGE_MB`, `GUIAGENT_ADMIN_TOKEN` | Free use per caller and month (defaults 20 runs, 1800 s, 500 MB; over it: HTTP 429) and the admin token for API keys (see `apps/_shared/README.md`) |
| `GUIAGENT_OLLAMA_MODEL`, `GUIAGENT_OLLAMA_URL` | Optional local LLM that reads free-form tasks. Without it, tasks are read by rules, and the demo works either way. |

## 5. Start

```bash
docker compose up -d --build     # first build takes a few minutes (PyTorch CPU wheels)
docker compose ps
docker compose logs -f caddy     # wait for "certificate obtained successfully"
```

Open `https://heatsink.example.org` and `https://pcb.example.org`.

Optional local LLM for the Form Compiler (documents stay on the server; a GPU helps, CPU works but is slow):

```bash
docker compose --profile llm up -d ollama
docker compose exec ollama ollama pull <model>   # any instruction model with structured-output support
# set UDR_OLLAMA_MODEL=<model> in .env, then
docker compose up -d udr
```

## Server that already has a reverse proxy

If ports 80/443 are already taken by another proxy (e.g. a Caddy serving other sites), don't start a second
Caddy. Start only the apps, attached to that proxy's Docker network, and add two site blocks to its config.

```bash
docker inspect <proxy container> --format '{{range $k, $v := .NetworkSettings.Networks}}{{$k}} {{end}}'
PROXY_NETWORK=<that network> docker compose -f docker-compose.yml -f docker-compose.existing-proxy.yml \
  up -d --build heatsink pcb datahealth standardizer simmeta udr inverse preflight mesh compare lineage interop aero lab observatory playground evidence validate sensitivity uncertainty runner campaign meshless rom inverse-studio identifiability calibration models arena surrogate twin anomaly control optimize design
```

Caddyfile blocks for the existing proxy (then `caddy validate` and `caddy reload` inside its container):

```
heatsink.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-heatsink:8080 {
		transport http {
			read_timeout 120s
		}
	}
}
pcb.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-pcb:8081 {
		transport http {
			read_timeout 120s
		}
	}
}

datahealth.example.org {
	encode zstd gzip
	request_body {
		max_size 120MB
	}
	reverse_proxy pinneapple-datahealth:8082 {
		transport http {
			read_timeout 300s
		}
	}
}
standardizer.example.org {
	encode zstd gzip
	request_body {
		max_size 120MB
	}
	reverse_proxy pinneapple-standardizer:8083 {
		transport http {
			read_timeout 300s
		}
	}
}
simmeta.example.org {
	encode zstd gzip
	request_body {
		max_size 120MB
	}
	reverse_proxy pinneapple-simmeta:8084 {
		transport http {
			read_timeout 300s
		}
	}
}
udr.example.org {
	encode zstd gzip
	request_body {
		max_size 50MB
	}
	reverse_proxy pinneapple-udr:8085 {
		transport http {
			read_timeout 600s
		}
	}
}
inverse.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-inverse:8086 {
		transport http {
			read_timeout 120s
		}
	}
}
preflight.example.org {
	encode zstd gzip
	request_body {
		max_size 220MB
	}
	reverse_proxy pinneapple-preflight:8087 {
		transport http {
			read_timeout 180s
		}
	}
}
mesh.example.org {
	encode zstd gzip
	request_body {
		max_size 220MB
	}
	reverse_proxy pinneapple-mesh:8088 {
		transport http {
			read_timeout 300s
		}
	}
}
compare.example.org {
	encode zstd gzip
	request_body {
		max_size 320MB
	}
	reverse_proxy pinneapple-compare:8089 {
		transport http {
			read_timeout 300s
		}
	}
}
lineage.example.org {
	encode zstd gzip
	request_body {
		max_size 320MB
	}
	reverse_proxy pinneapple-lineage:8090 {
		transport http {
			read_timeout 180s
		}
	}
}
interop.example.org {
	encode zstd gzip
	request_body {
		max_size 520MB
	}
	reverse_proxy pinneapple-interop:8091 {
		transport http {
			read_timeout 300s
		}
	}
}
aero.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-aero:8092 {
		transport http {
			read_timeout 120s
		}
	}
}
lab.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-lab:8093
}
observatory.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-observatory:8094 {
		transport http {
			read_timeout 180s
		}
	}
}
playground.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-playground:8095
}
surrogate.example.org {
	encode zstd gzip
	request_body {
		max_size 110MB
	}
	reverse_proxy pinneapple-surrogate:8097 {
		transport http {
			read_timeout 120s
		}
	}
}
arena.example.org {
	encode zstd gzip
	request_body {
		max_size 110MB
	}
	reverse_proxy pinneapple-arena:8105 {
		transport http {
			read_timeout 120s
		}
	}
}
models.example.org {
	encode zstd gzip
	request_body {
		max_size 110MB
	}
	reverse_proxy pinneapple-models:8114 {
		transport http {
			read_timeout 60s
		}
	}
}
runner.example.org {
	encode zstd gzip
	request_body {
		max_size 25MB
	}
	reverse_proxy pinneapple-runner:8110 {
		transport http {
			read_timeout 120s
		}
	}
}
campaign.example.org {
	encode zstd gzip
	request_body {
		max_size 110MB
	}
	reverse_proxy pinneapple-campaign:8096 {
		transport http {
			read_timeout 300s
		}
	}
}
uncertainty.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-uncertainty:8098 {
		transport http {
			read_timeout 180s
		}
	}
}
sensitivity.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-sensitivity:8099 {
		transport http {
			read_timeout 300s
		}
	}
}
validate.example.org {
	encode zstd gzip
	request_body {
		max_size 220MB
	}
	reverse_proxy pinneapple-validate:8107 {
		transport http {
			read_timeout 180s
		}
	}
}
evidence.example.org {
	encode zstd gzip
	request_body {
		max_size 220MB
	}
	reverse_proxy pinneapple-evidence:8113 {
		transport http {
			read_timeout 180s
		}
	}
}
calibration.example.org {
	encode zstd gzip
	request_body {
		max_size 60MB
	}
	reverse_proxy pinneapple-calibration:8101 {
		transport http {
			read_timeout 120s
		}
	}
}
identifiability.example.org {
	encode zstd gzip
	request_body {
		max_size 60MB
	}
	reverse_proxy pinneapple-identifiability:8108 {
		transport http {
			read_timeout 120s
		}
	}
}
inverse-studio.example.org {
	encode zstd gzip
	request_body {
		max_size 60MB
	}
	reverse_proxy pinneapple-inverse-studio:8104 {
		transport http {
			read_timeout 120s
		}
	}
}
guiagent.example.org {
	encode zstd gzip
	reverse_proxy pinneapple-guiagent:8115 {
		transport http {
			read_timeout 120s
		}
	}
}
rom.example.org {
	encode zstd gzip
	request_body {
		max_size 220MB
	}
	reverse_proxy pinneapple-rom:8102 {
		transport http {
			read_timeout 180s
		}
	}
}
meshless.example.org {
	encode zstd gzip
	request_body {
		max_size 25MB
	}
	reverse_proxy pinneapple-meshless:8109 {
		transport http {
			read_timeout 180s
		}
	}
}
twin.example.org {
	encode zstd gzip
	request_body {
		max_size 60MB
	}
	reverse_proxy pinneapple-twin:8103 {
		transport http {
			read_timeout 120s
		}
	}
}
anomaly.example.org {
	encode zstd gzip
	request_body {
		max_size 120MB
	}
	reverse_proxy pinneapple-anomaly:8111 {
		transport http {
			read_timeout 180s
		}
	}
}
control.example.org {
	encode zstd gzip
	request_body {
		max_size 20MB
	}
	reverse_proxy pinneapple-control:8112 {
		transport http {
			read_timeout 180s
		}
	}
}
optimize.example.org {
	encode zstd gzip
	request_body {
		max_size 25MB
	}
	reverse_proxy pinneapple-optimize:8100 {
		transport http {
			read_timeout 300s
		}
	}
}
design.example.org {
	encode zstd gzip
	request_body {
		max_size 25MB
	}
	reverse_proxy pinneapple-design:8106 {
		transport http {
			read_timeout 300s
		}
	}
}
```

## PINNeAPPle Observatory (`observatory`)

Needs outbound HTTPS to `celestrak.org` for satellite TLEs (cached in the `obs_cache` volume, refreshed every 6 h; without
it the satellites layer is empty and everything else works). The JPL DE421 ephemeris ships in the image (`skyfield-data`).
The browser's location and the phone's motion sensors require HTTPS, which Caddy provides. At start-up the app runs its
validation suite and trains the default Mars surrogate in the background (~1 min of one CPU core).

## PINNeAPPle Playground (`playground`)

A static site: the zoo models run in the visitor's browser (ONNX), so the container only serves files (Caddy,
128 MB). There is no login and nothing to keep. Build and run it alone with
`docker build -f apps/playground/Dockerfile -t pinneapple-playground . && docker run -p 8095:8095 pinneapple-playground`.
The same build can go to GitHub Pages instead: set the repository variable `PLAYGROUND_PAGES=true` and the docs
workflow publishes it under `<docs site>/playground/` (see `apps/playground/README.md`).

## Simulation Workflow Runner (`runner`, day 27)

Runs simulation campaigns as DAGs of steps (`pinneapple_core.workflows`) through the shared job queue. The public
image runs **only the built-in templates** (Python slab heating; the CalculiX beam and OpenFOAM cavity show as
"needs solver" because the image has neither solver), as an unprivileged user, with case values checked against each
template's parameter ranges. The Slurm executor only writes sbatch scripts there. Runs, step files, the queue, API
keys and usage live in the `runner_data` volume (keep it; delete old runs under `/data/runner/workflows/runs/` to
free space).

Self-hosted (your own machine or cluster head node, behind your own login): set `RUNNER_SELF_HOSTED=1` to accept
custom workflows (commands, scripts, Docker steps, Slurm submission with `dry_run: false`). That runs whatever the
caller sends, so never set it on a public server. Install OpenFOAM / CalculiX in the image (or use the Docker
executor) for the solver templates. Limits: `RUNNER_MAX_JOBS` runs at once, `RUNNER_MAX_CASES` cases per run,
`RUNNER_FREE_JOBS_RUN` / `RUNNER_FREE_COMPUTE_SECONDS` / `RUNNER_FREE_STORAGE_MB` per principal and month
(`/api/usage`); API keys via `/api/keys` with `RUNNER_ADMIN_TOKEN` (see `apps/_shared/README.md`). The "View
result" panel of a CalculiX or OpenFOAM step builds a glTF once and keeps it under `/data/runner/workflows/views/`
(decimated above `RUNNER_VIEW_MAX_FACES`, 250,000 triangles); it is a cache and can be deleted at any time. It needs a
runner with the solvers, so it does not appear on the public image.

## Simulation Campaign Planner (`campaign`)

Day 13, port 8096. It plans campaigns as jobs in the shared runtime queue (`apps/_shared`). Jobs, the files of
each campaign, API keys, usage and uploads live in the `campaign_data` volume; keep it. The airfoil preset reads
the 436 OpenFOAM runs shipped inside the library, so the app needs no network. It writes the solver cases but does
not run them, so neither OpenFOAM nor CalculiX is installed in the image. `CAMPAIGN_FREE_CAMPAIGNS` /
`CAMPAIGN_FREE_CASES_PLANNED` set the monthly free use per API key, login or IP (`/api/usage`). Keys are created by
the login user, or with `X-Admin-Token: $CAMPAIGN_ADMIN_TOKEN` (`POST /api/keys`).

## Engineering Uncertainty Lab (`uncertainty`)

Day 15, on the shared runtime (`apps/_shared/README.md`): studies are jobs in the queue (`UQL_MAX_JOBS` at once,
default 2; `GET /api/jobs/{id}`), uploads go through `/api/uploads`, and use is metered per caller and month in
`studies`, `scenarios` (cases screened), `propagation_samples` and `hf_runs` (`UQL_FREE_<UNIT>`; 429 past the limit).
Keys: `UQL_ADMIN_TOKEN` or the app login manage `/api/keys`. Jobs, uploads, keys and usage live in the
`uncertainty_data` volume. Study size is capped by `UQL_MAX_CASES`, `UQL_MAX_PROPAGATION`, `UQL_MAX_HF` (heat-sink
engine runs, about 0.4 s each) and `UQL_MAX_MB`; the synchronous "Try with example" runs are capped by
`UQL_MAX_HEAVY`. The zoo models and their reference datasets ship in the image; no outbound network is needed. ONNX
uploads run in onnxruntime (CPU). Build and run it alone with
`docker build -f apps/uncertainty_lab/Dockerfile -t uncertainty-lab . && docker run -p 8098:8098 uncertainty-lab`.

## Engineering Sensitivity Lab (`sensitivity`)

Day 16, on the shared runtime (`apps/_shared/README.md`): studies are jobs in the queue (`SEN_MAX_JOBS` at once,
default 2; `GET /api/jobs/{id}`), uploads go through `/api/uploads`, and use is metered per caller and month in
`studies`, `evaluations` and `parameters` (`SEN_FREE_STUDIES`, `SEN_FREE_EVALUATIONS`, `SEN_FREE_PARAMETERS`; 429 past
the limit). Keys: `SEN_ADMIN_TOKEN` or the app login manage `/api/keys`. Jobs, uploads, keys and usage live in the
`sensitivity_data` volume. Study size is capped by `SEN_MAX_EVALS` (model evaluations; the UI shows the count before
running), `SEN_MAX_PARAMS` and `SEN_MAX_MB`; the synchronous "Try with example" runs are capped by `SEN_MAX_HEAVY`.
The zoo models and datasets ship in the image; no outbound network is needed. Build and run it alone with
`docker build -f apps/sensitivity_lab/Dockerfile -t sensitivity-lab . && docker run -p 8099:8099 sensitivity-lab`.

## Physics Validation Lab (`validate`)

Runs `pinneapple check` on uploaded predictions, results or a zoo model and reports numerical verification,
validation against data and physical consistency separately, with an evidence record per check. Uploaded Python
models and checkpoints are refused (they would run uploaded code); zoo models ship in the image. Each validation
runs in a temporary folder that is deleted afterwards. Only the glTF of a 3-D view is kept (the prediction, reference
and error on the mesh of a 3-D result with a reference): the newest `VAL_VIEWS_KEPT` (300) files in the app's data
directory (`VAL_DATA_DIR`, default `/data/validate` when `/data` is writable, else the user's home), each decimated
above `VAL_VIEW_MAX_FACES` (250,000) triangles. They are a cache: a lost file only means running the validation again.
`VAL_MAX_HEAVY` validations run at once per worker (the excess gets HTTP 429), `VAL_MAX_MB` caps an upload. 1 GB of
memory is enough: the examples take under a second, the 3-D one about two.

## Engineering Evidence Report (`evidence`)

Builds a report (HTML, PDF) and an evidence package (zip with a SHA-256 manifest) from the outputs of a study:
Validation Lab reports, evidence ledgers, `pinneapple check` JSON, Simulation Comparator results, StudyResult /
ModelRef / DatasetSpec JSON, metrics tables, any file. A test is printed as passed only when its own evidence record
shows it ran. Nothing is stored. The example study (FEM vs PINN on a Poisson problem plus a Validation Lab run) is
computed on the first request (about 5 s, one CPU core) and then reused. `EVR_MAX_HEAVY` reports at
once per worker (the excess gets HTTP 429), `EVR_MAX_MB` caps an upload. 1 GB of memory is enough. API keys and
usage metering come from the shared runtime (`evidence_data` volume): `EVR_FREE_REPORTS` and `EVR_FREE_DOWNLOADS`
(PDF or package) per caller and month, default 30 each; `EVR_ADMIN_TOKEN` for `/api/keys`.

## Model Calibration Lab (`calibration`)

Fits model parameters to measurements, with Jacobian and bootstrap intervals only where the data determine them,
before/after comparison, validation on held-out measurements and other estimators ranked by held-out error.
Calibrations run in the shared job queue (`CALIB_MAX_JOBS` at once over both workers; the rest wait), with usage
metered per caller and month (`CALIB_FREE_CALIBRATIONS`, `CALIB_FREE_METHOD_RUNS`, `CALIB_FREE_PINN_RUNS`; `unlimited`
lifts a limit). Jobs, API keys (`CALIB_ADMIN_TOKEN`) and usage live in SQLite on the `calibration_data` volume.
`CALIB_IDENTIFIABILITY_URL` sets the link to the Parameter Identifiability Lab. The examples take 1 to 9 s; a PINN
comparison trains for about a minute on one core. 1 GB of memory is enough.

## Parameter Identifiability Lab (`identifiability`)

Checks, before an experiment or a calibration, which parameters a set of measurements can determine: Fisher
information rank and conditioning, local sensitivity, profile likelihood and an MCMC posterior, then the D-optimal
extra measurements. Analyses run in the shared job queue (`IDENT_MAX_JOBS` at once over both workers; the rest wait in
the queue), with usage metered per caller and month (`IDENT_FREE_ANALYSES`, `IDENT_FREE_PARAMETERS`,
`IDENT_FREE_SCENARIOS`; `unlimited` lifts a limit). Jobs, API keys (`IDENT_ADMIN_TOKEN` to manage them without the
login) and usage are kept in SQLite on the `identifiability_data` volume. The examples take under 3 s; 1 GB of memory
is enough.

## Inverse Model Studio (`inverse-studio`)

A separate app from the Inverse Heat Lab (`inverse`, which is unchanged). It formulates an inverse problem once and
solves it with least squares, adjoint gradient, autodiff, Bayesian regularisation and the Inverse Heat Lab's PINNs on
the same data, ranked by held-out error, with reconstructed fields and the identifiability diagnosis. Problems run in
the shared job queue (`INVS_MAX_JOBS` at once over both workers; the rest wait), metered per caller and month
(`INVS_FREE_PROBLEMS`, `INVS_FREE_METHOD_RUNS`, `INVS_FREE_PINN_RUNS`). Jobs, API keys (`INVS_ADMIN_TOKEN`) and usage
live in SQLite on the `inverse-studio_data` volume. `INVS_HEAT_LAB_URL` sets the link to the Inverse Heat Lab. The
classical methods take seconds; a PINN trains for 25 s (fin) to 95 s (plate) on one core, so keep `INVS_MAX_JOBS` at
or below the cores you can spare. 1 GB of memory is enough.

## Engineering GUI Agent (`guiagent`)

A **separate, optional heavy image** (about 5.4 GB uncompressed, 1.8 GB compressed). It contains:

- FreeCAD 1.0.2: the AppImage, extracted;
- the CfdOF 1.37.3 workbench;
- OpenFOAM v1912 (the Ubuntu package, as elsewhere in the repository);
- ParaView 5.11 with `pvpython`;
- Xvfb and xdotool.

It has the compose profile `guiagent`, so a plain `docker compose up -d` does not build or start it. Name it instead:

```bash
docker compose up -d --build guiagent                     # with this project's Caddy
PROXY_NETWORK=<network> docker compose -f docker-compose.yml -f docker-compose.existing-proxy.yml \
  up -d --build --no-deps guiagent                         # behind an existing proxy (pinneapple-guiagent:8115)
```

The service runs one uvicorn worker with `mem_limit` 2g and one run at a time (`GUIAGENT_MAX_JOBS=1`; later runs
wait in the shared job queue of `apps/_shared`, which also meters runs, compute seconds and storage per caller). A full run of
the example took 42 s in the image (2 GB limit, 0.9 GB in use after the run). Runs and their evidence, the job queue, API keys and usage are kept in
the `guiagent_data` volume (`/data/guiagent`); the oldest runs are deleted beyond `GUIAGENT_KEEP_RUNS`. On a public server, keep `GUIAGENT_MODE=public`: only the committed
workflows run, with their inputs inside the workflow's bounds, and no uploads are accepted. Building the image
downloads the FreeCAD AppImage from GitHub (about 800 MB) and CfdOF from GitHub.

## Reduced Order Model Lab (`rom`)

Builds POD + RBF / GPR, DMD, Operator Inference and an equal-budget neural surrogate from uploaded snapshots and scores them on held-out times or parameters. Studies run in the shared job queue: `ROM_MAX_JOBS` at once (default 2; one CPU core each for a few seconds to a minute), `ROM_MAX_BUDGET_S` caps the neural budget a study may ask for, `ROM_MAX_MB` an upload. Jobs, uploads and usage live on the `rom_data` volume; free use per caller and month is set by `ROM_FREE_STUDIES`, `ROM_FREE_EXPORTS`, `ROM_FREE_QUERIES` (30 / 100 / 20,000 by default). 1 GB of memory is enough for snapshots up to ~400,000 values (`ROM_MAX_VALUES`).

## Meshless Method Lab (`meshless`)

Refinement studies of Kansa, RBF-FD, embedded finite differences, linear finite elements and a PINN on the same 2-D problem. Studies run in the shared job queue: `MESHLESS_MAX_JOBS` at once (default 2; one CPU core each, seconds to a minute), `MESHLESS_MAX_NODES` caps one refinement level (default 20,000; Kansa stops at 4,000 in the library), `MESHLESS_MAX_PINN_S` the PINN budget. Jobs and usage live on the `meshless_data` volume; free use per caller and month: `MESHLESS_FREE_RUNS` (40), `MESHLESS_FREE_UNKNOWNS` (2,000,000 nodes summed over levels), `MESHLESS_FREE_REPORTS` (100 downloads). 1 GB of memory is enough at the default limits.

## Surrogate Studio (`surrogate`)

Trains surrogate models from uploaded simulation results (or the two built-in examples) and serves the chosen model
at `/v1/models/<name>/predict`, the Model Gateway's API. Studies run in the shared job queue (`SUR_MAX_JOBS` at once,
the rest wait in the queue); `SUR_MAX_BUDGET_S` and `SUR_MAX_SEEDS` cap the compute one study can ask for. Uploads,
job results, hosted models, API keys and usage live on the `surrogate_data` volume: keep it across rebuilds. One
uvicorn worker in a 1 GB container (torch ~400 MB idle, a study adds ~300 MB); the examples finish in about 10 s.
Free use per caller and month: `SUR_FREE_TRAININGS` (30), `SUR_FREE_INFERENCES` (20000 rows), `SUR_FREE_HOSTED_MODELS`
(10); `unlimited` lifts a limit. Hosted models are visible to everyone who can reach the server: set the login or
`SUR_REQUIRE_API_KEY=1` for private use.

## Physics AI Benchmark Arena (`arena`)

Shows rankings of physics-AI methods per data regime from benchmark runs stored in `apps/benchmark_arena/results/`
(written by `run_public_benchmarks.py`, with their date, hardware and dataset hash), and runs private benchmarks on
uploaded data with the same protocol in the shared job queue (`ARN_MAX_JOBS`, default 1: a benchmark keeps one core
busy). Refresh the public results on the deployment hardware with
`docker compose exec arena python run_public_benchmarks.py` followed by a restart (the files live in the image; copy
them back into the repository to keep them). `ARN_MAX_BUDGET_S` × methods × seeds × training sets bounds one
benchmark's time. Uploads, private results, keys and usage live on the `arena_data` volume.

## Physics AI Model Gateway (`models`)

Serves physical models behind one API (`/v1/models/<name>/predict`), with versions, stages, an OOD policy per version
(warn or refuse), a run log and latency / count / error metrics. On a fresh `models_data` volume it publishes three zoo
models. Publishing needs the login or `MDL_ADMIN_TOKEN`; API-key holders may publish ONNX graphs. A Surrogate Studio
model is published by its ModelRef (the gateway forwards to the studio and enforces the bounds itself) or as a
`model.zip` (admin only: loading it unpickles the model). Keep the volume: it holds the models, the run log, keys and
usage. Free use per caller and month: `MDL_FREE_INFERENCES` (50000 rows), `MDL_FREE_HOSTED_MODELS` (5).

## Digital Twin Runtime (`twin`)

Runs a validated model next to live data: EnKF state estimation, forecasts, alerts, drift monitoring and
revalidation (day 20). One worker by design: live twins, their threads and the embedded MQTT broker live in its
memory, and a restart stops them (the history stays in the `twin_data` volume, SQLite by default). `TWIN_MAX_ASSETS`
caps live twins, `TWIN_MAX_JOBS` replays and series flows running at once (the shared job queue). `TWIN_DB_URL`
takes a `postgresql://` URL when `psycopg[binary]` is added to the image. The embedded broker listens on
127.0.0.1 only; to accept external publishers set `TWIN_MQTT_HOST=0.0.0.0`, `TWIN_MQTT_PORT=1883` and publish the
port (`ports: ["1883:1883"]` in an override), or set `TWIN_MQTT_URL=mqtt://user:pass@broker:1883` for your own
broker. OPC UA (`TWIN_OPCUA_URL`, `TWIN_OPCUA_NODES`) is experimental and needs `asyncua` in the image. 1 GB of
memory is enough (about 400 MB idle).

## Physics Anomaly Detector (`anomaly`)

Finds abnormal behaviour in sensor time series with statistics, AR + CUSUM, conservation balances over the columns
and physical-model residuals, and explains each event (day 28). Analyses and benchmarks run in the shared job queue
(`ANOMALY_MAX_JOBS` at once); uploads, jobs and usage live in the `anomaly_data` volume. `ANOMALY_MAX_ROWS` caps a
table (default 500,000 rows). 1 GB of memory is enough: the example and a benchmark seed take a few seconds.

## Physics-Based Control Lab (`control`)

Compares model predictive control (linear, nonlinear with CasADi/IPOPT, and on a reduced-order surrogate) with PID and
rule-based baselines on the same simulated plant, and recommends actions (day 29). **Simulation and recommendation
only: there is no command path to real equipment.** Comparisons run in the shared job queue (`CONTROL_MAX_JOBS` at
once, about 15 s for the default study); jobs and usage live in the `control_data` volume. 1 GB of memory is
enough.

## Engineering Design Optimizer (`optimize`)

Black-box design optimisation (day 17): NSGA-II, pymoo, SciPy, Bayesian optimisation on a surrogate, a library
model or a GP trained on uploaded data, with candidates re-evaluated by a high-fidelity reference. Each run is a job
of the shared runtime (`apps/_shared/README.md`): state, uploads and API keys are in the `optimize_data` volume.
`EDO_MAX_JOBS` runs at once (2 by default; more wait in the queue). Metered per caller and month:
`EDO_FREE_EVALUATIONS` (200000), `EDO_FREE_CAMPAIGNS` (100), `EDO_FREE_REFERENCE_RUNS` (500).
`EDO_SOLVERS` (JSON) lists external solvers the operator allows as references, e.g.
`{"cfd": {"command": ["/opt/solver/run.sh", "{inputs}", "{outputs}"], "outputs": ["t_source_c"], "timeout": 900}}`;
users can only pick from this list, never send a command. 1 GB is enough (one worker, torch CPU).

## Physics Design Optimizer (`design`)

Gradient-aware design optimisation (day 23): autograd through differentiable surrogates, SLSQP, NSGA-II,
differential evolution and Bayesian optimisation compared on the same budget, candidates verified with the heat-sink
FVM engine, STL/STEP/CSV exports. Each study is a job of the shared runtime (`apps/_shared/README.md`): state and
API keys are in the `design_data` volume; `PDO_MAX_JOBS` studies run at once (2 by default; more wait in the queue).
Metered per caller and month: `PDO_FREE_EVALUATIONS` (200000), `PDO_FREE_CAMPAIGNS` (100), `PDO_FREE_SOLVER_RUNS`
(500). A study is capped at 40000 evaluations (150 per run with Bayesian optimisation). 1 GB is enough.

## PINNeAPPle Lab catalogue (`lab`)

The `lab` service runs `python -m pinneapple_lab serve` (standard library HTTP server, port 8093) on the `lab_data`
volume, which starts as a copy of the repository's `lab/` folder. It serves the catalogue page, every run's files,
a JSON API (`/api/catalog`, `/api/runs`, `/api/runs/<id>`, `/api/datasets`) and dataset downloads
(`/download/<experiment>/<dataset>.npz`). New runs appear on reload, so sweeps can run in the container:

```bash
docker compose exec lab python -m pinneapple_lab sweep repo_results -g source=burgers_pinn,lbm_strouhal
docker compose exec lab python -m pinneapple_lab examples --run --match use_cases   # examples and use cases
```

The **experiments page** (`/experiments`) lists every experiment: each Lab item with its A-F seal, trust card,
measured metrics, next steps and stamped figures, and every `examples/*` folder from its `experiment.json` (shown as
"not curated" with the command that curates it until the Lab grades it). JSON: `/api/experiments` (filters
`letter`, `domain`, `method`, `ready`, `q`, `sort`), `/api/experiment?key=`, `/api/coverage`; media: `/stamped/...`,
`/seal.png`, `/repo/...` (only files a manifest lists). The image already holds the repository at `/srv`
(`PINNEAPPLE_REPO`), so the example figures and movies come with it; Lab figures come from the `lab_data` volume.

To deploy it on a server that already runs the `lab` service:

```bash
cd PINNeAPPle && git pull                                   # main, with the experiments page
cd apps/deploy && docker compose build lab && docker compose up -d lab
docker compose exec lab python -m pinneapple_lab reindex    # the volume's runs into the index
docker compose exec lab python -m pinneapple_lab curate     # A-F letters and next steps for the volume's runs
curl -fsS -u "$LAB_USER:$LAB_PASSWORD" https://$LAB_DOMAIN/api/coverage | head -c 300   # check
```

The `lab_data` volume was seeded from the image the first time and is kept across rebuilds, so the `curate` step is
what brings the new letters and next steps to the runs already in it. A static copy for GitHub Pages:
`python -m pinneapple_lab site --out site` (`site/experiments/index.html` with stamped figures, seals and movies up
to 10 MB).

Set `LAB_USER` / `LAB_PASSWORD` in `.env` to require a login (`/health` stays open). Without Docker:
`pip install pinneapple pillow` and `python -m pinneapple_lab serve --host 0.0.0.0 --root /path/to/lab`.

## Operations

| Task | Command (in `apps/deploy`) |
|---|---|
| Update to the latest code | `git pull && docker compose up -d --build` |
| Logs | `docker compose logs -f heatsink` (or `pcb`, `caddy`, `guiagent`) |
| Change a password | edit `.env`, then `docker compose up -d` |
| Restart | `docker compose restart` |
| Stop | `docker compose down` (certificates stay in the `caddy_data` volume) |
| Reclaim disk after many builds | `docker image prune -f` |

The containers restart automatically after a reboot (`restart: unless-stopped`).

## Notes

- **Per-app deploy folders.** `apps/heatsink_sizer/deploy` and `apps/pcb_hotspot/deploy` each start their own
  Caddy on ports 80/443. Use this folder *instead of* them, not alongside them.
- **More sites.** To serve a landing page (e.g. the apex domain) from the same server, add a site block to
  `Caddyfile`, for instance `example.org { root * /srv/site; file_server }`, and mount the folder into the
  `caddy` service.
- **Cloudflare proxy.** After the certificates are issued you can switch the records to proxied (orange cloud).
  If you do, set SSL/TLS mode to **Full (strict)**.
- **Tested.** `docker compose config` and `caddy validate` pass. Routing was also checked end to end with
  Caddy's internal CA in place of Let's Encrypt: both hosts work, the login is enforced, `/health` is open,
  responses are gzip-compressed, HTTP redirects to HTTPS, and `/api/solve` runs through the proxy.
