# Operator runbook: PINNeAPPle apps

Commands run in `apps/deploy` on the server unless stated. `<PREFIX>` is the app's prefix (`HSS`, `MDL`,
`CAMPAIGN`, ...; the list is in `.env.example`), `<svc>` its Compose service (`heatsink`, `models`, `campaign`, ...).
Settings are described in `apps/_shared/README.md` and the host in `docs/security/server.md`.

## Upgrade to the hardened containers (once)

> [!IMPORTANT]
> The first upgrade to the non-root images (PR "Harden the deployment") needs a one-time ownership change of the
> app volumes, **before** the new images start; otherwise apps that keep state on a volume can not write to it.

```bash
cd apps/deploy
docker compose down
for v in $(docker volume ls -q | grep -E '^pinneapple-apps_(.*_data|obs_cache)$' | grep -v caddy_data); do
  docker run --rm -v "$v":/v alpine chown -R 10001:0 /v
done
docker compose build --pull && docker compose up -d
sudo ./check_server.sh                     # then every service healthy in `docker compose ps`
```

Also set login hashes (`<PREFIX>_PASSWORD_HASH`, below) and, if the Model Gateway hosted pickled Surrogate Studio
models, decide on `MDL_ALLOW_PICKLE_UPLOAD` (off by default now).

## Rotate credentials

**An app login password**

```bash
python -m pinneapple_security hash-password            # prompts twice, prints $scrypt$... (or $argon2id$...)
```

Put the hash in `.env` as `<PREFIX>_PASSWORD_HASH=...`, writing each `$` as `$$`, and clear `<PREFIX>_PASSWORD`.
Then `docker compose up -d <svc>`: the container is recreated with the new value and the old password stops working.

**An admin token** (`<PREFIX>_ADMIN_TOKEN`): generate one with
`python -c "import secrets; print(secrets.token_urlsafe(32))"`, put it in `.env`, run `docker compose up -d <svc>`,
and update every client that uses it.

**API keys** (callers of `/api` and `/v1`):

```bash
curl -s -H "X-Admin-Token: $TOKEN" https://<host>/api/keys                    # list: id, name, created, revoked
curl -s -X POST -H "X-Admin-Token: $TOKEN" -H "Content-Type: application/json" \
     -d '{"name": "ci-2026-10"}' https://<host>/api/keys                       # new key, shown once
curl -s -X DELETE -H "X-Admin-Token: $TOKEN" https://<host>/api/keys/<id>       # revoke the old one
```

Keys are stored as SHA-256 digests. A revoked key gets 401 at once, in every worker.

**The audit HMAC key** (`<PREFIX>_AUDIT_KEY`): a new key starts a new chain. Move the old `audit.jsonl` aside first
(`docker compose exec <svc> mv /data/<prefix>/audit.jsonl /data/<prefix>/audit-<date>.jsonl`), keep the old key
with that archive so it can still be verified, then set the new key and recreate the container.

**A credential found in the repository or a log**: treat it as public. Rotate it as above, revoke what it unlocks,
and check the audit trail and access log for its use since it was exposed.

## Add or remove a user

Single login per app: `<PREFIX>_USER` + `<PREFIX>_PASSWORD_HASH`. Several logins: a users file, mounted read-only.

```bash
mkdir -p secrets && chmod 700 secrets
printf 'ana:%s\n' "$(python -m pinneapple_security hash-password)" >> secrets/models.users
chmod 600 secrets/models.users
```

In `docker-compose.yml` (or an override file) for the service:

```yaml
    environment:
      MDL_USERS_FILE: /run/secrets/users
      MDL_ADMIN_USERS: ana                 # who may manage keys and publish models
    volumes:
      - ./secrets/models.users:/run/secrets/users:ro
```

`docker compose up -d models`. To remove someone, delete their line and recreate the container. The container
runs as uid 10001: make the file readable to it (`chown 10001 secrets/models.users`).

**A user is locked out** after too many failed logins (429 with `Retry-After`): the lock ends by itself after
`<PREFIX>_LOGIN_LOCK_S` (15 minutes), or at once with `docker compose restart <svc>` (lock state is in memory).
If fail2ban banned the address: `fail2ban-client set pinneapple-apps unbanip <ip>` (or `pinneapple-caddy`).

## Read the audit log

Each auth or admin event is one JSON line on the container's stderr and, when the app has a data directory, one
record in a hash-chained trail:

```bash
docker compose logs --no-log-prefix <svc> | grep '"event"' | jq -c 'select(.outcome != "ok")'
docker compose exec <svc> python - <<'EOF'
from pinneapple_security.audit import AuditLog
log = AuditLog("/data/<prefix>/audit.jsonl")
print(log.verify())                                        # {'ok': True, 'n': ..., 'head': ...}
for r in log.query(action="login_failed")[-20:]:
    print(r["time"], r["actor"], r["data"].get("ip"), r["target"])
EOF
```

Events: `login_failed`, `login_locked`, `csrf_blocked`, `rate_limited`, `api_key_lockout`, `admin_denied`,
`api_key_created`, `api_key_revoked`. `verify()` names the first record that was changed, removed or reordered.
The Caddy access log (`/data/access.log` in the `caddy_data` volume) has every request with its status; secrets
in headers are redacted by Caddy.

## Respond to an incident

1. **Contain.** Ban the source (`fail2ban-client set pinneapple-caddy banip <ip>`), or take the app off line
   (`docker compose stop <svc>`). For a suspected compromise of a container, stop it but do not remove it.
2. **Preserve evidence** before changing anything:
   ```bash
   ts=$(date +%F-%H%M); mkdir -p /root/incident-$ts
   docker compose logs --timestamps > /root/incident-$ts/compose.log
   docker inspect $(docker compose ps -q) > /root/incident-$ts/inspect.json
   docker run --rm -v pinneapple-apps_<svc>_data:/v:ro -v /root/incident-$ts:/out alpine tar czf /out/data.tgz -C /v .
   cp /var/lib/docker/volumes/pinneapple-apps_caddy_data/_data/access.log* /root/incident-$ts/
   ```
3. **Assess.** Verify the audit trails (above), list API keys and admin events, compare running images with the
   built ones (`docker compose images`), and check `docker diff <container>` (should be empty: read-only root).
4. **Eradicate and recover.** Rotate every credential of the affected app (and shared ones), revoke all its API
   keys, rebuild from a clean checkout (`docker compose build --pull --no-cache <svc>`), restore data from the
   last backup taken before the incident if it was altered, and start again.
5. **Follow up.** Write down the timeline, the cause and the fix; report vulnerabilities in PINNeAPPle itself as
   described in `SECURITY.md`.

## Update dependencies

- **Dependabot** (`.github/dependabot.yml`) opens weekly pull requests for Python packages, the Docker base images
  (pinned by digest where pinned) and GitHub Actions. The `Security` workflow runs on each: bandit, pip-audit,
  gitleaks, hadolint, Trivy on the base images, the security tests and a CycloneDX SBOM.
- **On the server**, after merging: `git pull && docker compose build --pull && docker compose up -d`, then
  `sudo ./check_server.sh` and a look at `docker compose ps` (every service `healthy`).
- **A published vulnerability** in a dependency: `pip-audit -r <(docker compose exec -T <svc> pip freeze)` shows
  whether an image is affected; raise the floor in `pyproject.toml` or the app's `requirements.txt`, rebuild,
  redeploy.
- **SBOM** of an image: the `Security` workflow uploads `sbom-*.cdx.json` artifacts; locally,
  `docker compose exec <svc> python -m pinneapple_security sbom -o /tmp/sbom.json`.
