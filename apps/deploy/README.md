# Deploy the apps on one server

HeatSink Sizer, PCB Hotspot, Engineering Data Health, Engineering Data Standardizer and Simulation Metadata on a single Linux VM (e.g. Hetzner Cloud), each on its own
subdomain, behind one Caddy that obtains and renews Let's Encrypt certificates automatically.

```
                       ┌──────────── server ────────────┐
heatsink.example.org ─▶│ Caddy :80/:443 ─▶ heatsink:8080 │
pcb.example.org      ─▶│                ─▶ pcb:8081      │
                       └─────────────────────────────────┘
```

Only Caddy publishes ports. The apps are reachable only through it.

## 1. Server

- **Size:** 4 vCPU / 8 GB RAM is comfortable for the five apps at demo traffic (e.g. Hetzner CX32 or CPX31). 2 vCPU / 4 GB works
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
| `HSS_DOMAIN`, `PCB_DOMAIN`, `EDH_DOMAIN`, `EDS_DOMAIN`, `SMD_DOMAIN`, `UDR_DOMAIN`, `IHL_DOMAIN` | Public hostnames (must match the DNS records) |
| `ACME_EMAIL` | Let's Encrypt account / expiry notices |
| `<APP>_USER` / `<APP>_PASSWORD` (HSS, PCB, EDH, EDS, SMD, UDR, IHL) | HTTP Basic login per app. If either is empty, that app is public. `/health` is always public. |
| `HSS_MAX_SIZING`, `PCB_MAX_HEAVY`, `EDH/EDS/SMD/UDR_MAX_HEAVY` | Concurrent heavy runs per app. Any excess gets HTTP 429. |
| `HSS_MEM_LIMIT`, `PCB_MEM_LIMIT` | Container memory caps |
| `IHL_MAX_JOBS` | Inverse Heat Lab: trainings running at the same time (default 2, each uses one CPU core). Any excess gets HTTP 429. The app runs one worker because jobs live in its memory. |
| `UDR_FORM_PDF` | Path inside the `udr` container to your copy of the fillable Form U-DR-1 (put the file in `apps/deploy/forms/`). Optional: users can upload it instead. |
| `UDR_OLLAMA_MODEL` | Optional. Enables local-LLM extraction in the Form Compiler with this Ollama model (pull it first). Every value it returns is checked against the document text; documents never leave your network. |
| `UDR_OLLAMA_URL` | Ollama server: `http://ollama:11434` (the optional `ollama` service, `--profile llm`) or `http://host.docker.internal:11434` (Ollama on the host). |
| `UDR_OLLAMA_TIMEOUT` | Seconds per LLM request (default 600; local models on CPU are slow). |

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
  up -d --build heatsink pcb datahealth standardizer simmeta udr inverse
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
```

## Operations

| Task | Command (in `apps/deploy`) |
|---|---|
| Update to the latest code | `git pull && docker compose up -d --build` |
| Logs | `docker compose logs -f heatsink` (or `pcb`, `caddy`) |
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
