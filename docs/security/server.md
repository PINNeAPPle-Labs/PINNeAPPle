# Server hardening: the host that runs the PINNeAPPle apps

The apps run as Docker containers (`apps/deploy/docker-compose.yml`) behind one Caddy: either the project's own
(ports 80/443 on the host) or an existing Caddy on a shared Docker network (`docker-compose.existing-proxy.yml`,
e.g. `PROXY_NETWORK=pinneapple_physai_net`). This page covers what the repository configures and what the operator
sets up on the host. `apps/deploy/check_server.sh` checks the host part read-only and prints PASS / WARN / FAIL.

```
internet ─▶ 80/443 ─▶ Caddy (TLS, headers, body limits, access log) ─▶ app containers (no host ports)
                         │                                                  │
                    fail2ban (DOCKER-USER chain) ◀── access log + app auth events (docker logs)
```

## What the repository sets

| Layer | Setting | Where |
|---|---|---|
| Image | runs as uid 10001 (`USER 10001:0`), `HOME=/tmp`, `/data` owned by that user | every `apps/*/Dockerfile`, `pinneapple_lab/deploy/Dockerfile` |
| Container | `read_only: true` root filesystem, `/tmp` as tmpfs, `cap_drop: [ALL]`, `no-new-privileges`, `pids_limit`, `cpus`, `mem_limit`, rotated json-file logs | `x-app-hardening` in `apps/deploy/docker-compose.yml` |
| Network | only Caddy publishes ports; the apps `expose` theirs on the Compose network (and on the proxy network in existing-proxy mode) | `docker-compose.yml`, `docker-compose.existing-proxy.yml` |
| Proxy | TLS 1.2/1.3, HSTS, nosniff, frame and referrer policy, `Permissions-Policy`, no `Server` header, a request body limit on every site, rotated JSON access log | `apps/deploy/Caddyfile` (`(common)` snippet) |
| App | CSP, rate limits, login lockout, hashed passwords, CSRF origin check, body limit, audit trail | `pinneapple_security.web` via `apps/_shared/appkit.py` (see `apps/_shared/README.md`) |

The GUI agent (FreeCAD/X11) and Ollama keep a writable root filesystem (`x-base-hardening`): their images write
outside `/data` and `/tmp`. Everything else is read-only.

### Upgrading a server whose volumes were written by root (run once, before starting the new images)

Before this change the containers ran as root, so existing named volumes hold root-owned files that uid 10001 can
not write. Once, before starting the new images:

```bash
cd apps/deploy
docker compose down
for v in $(docker volume ls -q | grep -E '^pinneapple-apps_(.*_data|obs_cache)$' | grep -v caddy_data); do
  docker run --rm -v "$v":/v alpine chown -R 10001:0 /v
done
```

The Caddy volumes (`caddy_data`, `caddy_config`) stay as they are: the official Caddy image runs as root, with
only `NET_BIND_SERVICE`.

## 1. SSH: keys only

`/etc/ssh/sshd_config.d/10-hardening.conf`:

```
PasswordAuthentication no
KbdInteractiveAuthentication no
PermitRootLogin prohibit-password
PubkeyAuthentication yes
MaxAuthTries 3
X11Forwarding no
AllowAgentForwarding no
```

Then `sshd -t && systemctl reload ssh`. Keep a second session open until a new login with the key works.

## 2. Firewall: SSH, 80 and 443 only

```bash
ufw default deny incoming
ufw default allow outgoing
ufw allow 22/tcp            # or your SSH port
ufw allow 80/tcp
ufw allow 443/tcp
ufw allow 443/udp           # HTTP/3
ufw enable
```

Docker publishes ports through its own iptables chains, before ufw sees the packets: a container started with
`-p 8080:8080` is reachable from the internet whatever ufw says. That is why only Caddy publishes ports; the check
script fails on any other container with a published port. fail2ban bans go into the `DOCKER-USER` chain, which
Docker evaluates first, so a ban also reaches the Caddy container.

## 3. fail2ban for SSH, Caddy and the app logins

```bash
apt install fail2ban
cp apps/deploy/fail2ban/filter.d/*.conf /etc/fail2ban/filter.d/
cp apps/deploy/fail2ban/jail.d/pinneapple.local /etc/fail2ban/jail.d/
fail2ban-regex /var/lib/docker/volumes/pinneapple-apps_caddy_data/_data/access.log pinneapple-caddy
systemctl restart fail2ban && fail2ban-client status
```

- `pinneapple-caddy`: 20 answers 401/429 to one client IP in 10 minutes ban it for an hour (Caddy's JSON access log;
  for an existing proxy, set `logpath` to its log).
- `pinneapple-apps`: 10 failed logins, lockouts, refused cross-site posts or invalid API keys logged by the apps
  (`docker logs`, one JSON line per event).
- `sshd`: 4 failures in 10 minutes.

The apps lock a client out on their own as well (`<PREFIX>_LOGIN_MAX_FAILURES`, default 10 in 15 minutes); fail2ban
takes the client off the network for every app at once.

## 4. Automatic security updates

```bash
apt install unattended-upgrades
dpkg-reconfigure -plow unattended-upgrades      # writes /etc/apt/apt.conf.d/20auto-upgrades
```

Reboot when `/var/run/reboot-required` exists (the check script warns). Images: rebuild weekly
(`docker compose build --pull && docker compose up -d`) so the Debian and Caddy base images bring their fixes; the
CI job `security.yml` scans the base images with Trivy.

## 5. Docker daemon

`/etc/docker/daemon.json`:

```json
{
  "no-new-privileges": true,
  "live-restore": true,
  "icc": false,
  "userland-proxy": false,
  "log-driver": "json-file",
  "log-opts": {"max-size": "10m", "max-file": "5"}
}
```

`systemctl restart docker` (with `live-restore` the containers keep running). The socket `/var/run/docker.sock`
is root-equivalent: never mount it into an app container, and keep the `docker` group empty except for
administrators. User-namespace remapping (`"userns-remap": "default"`) adds another layer but changes volume
ownership; turn it on only for a fresh server.

## 6. Caddy

The project's Caddyfile already sets TLS, headers, limits and logs. For an **existing** Caddy on
`pinneapple_physai_net`, add the same snippet to its Caddyfile and import it in each PINNeAPPle site:

```
(pinneapple) {
	encode zstd gzip
	tls {
		protocols tls1.2 tls1.3
	}
	header {
		Strict-Transport-Security "max-age=31536000; includeSubDomains"
		X-Content-Type-Options "nosniff"
		X-Frame-Options "DENY"
		Referrer-Policy "strict-origin-when-cross-origin"
		Permissions-Policy "camera=(), microphone=(), payment=(), usb=(), geolocation=(self)"
		-Server
	}
	log {
		output file /data/access.log {
			roll_size 50MiB
			roll_keep 10
		}
		format json
	}
}

heatsink.example.org {
	import pinneapple
	request_body {
		max_size 1MB
	}
	reverse_proxy pinneapple-heatsink:8080
}
```

The body limit per app is in `apps/deploy/Caddyfile`. Optional: a login at the proxy in front of an app that has
none of its own, with a bcrypt hash from `caddy hash-password`:

```
	basic_auth {
		demo $2a$14$...
	}
```

Rate limiting at the proxy needs a Caddy built with a rate-limit module (e.g. `xcaddy build --with
github.com/mholt/caddy-ratelimit`); the stock image has none. The apps enforce their own limits
(`<PREFIX>_RATE_LIMIT`), per worker process.

## 7. Secrets

- `.env` holds the logins and tokens: `chmod 600 apps/deploy/.env`, owned by root, never committed
  (`apps/deploy/.gitignore`). `.env.example` is the template and holds placeholders only.
- Store password **hashes** (`<PREFIX>_PASSWORD_HASH`, `python -m pinneapple_security hash-password`), not passwords.
  In `.env`, write every `$` of a hash as `$$`.
- Admin tokens (`<PREFIX>_ADMIN_TOKEN`) and the audit HMAC key (`<PREFIX>_AUDIT_KEY`): 32+ random bytes,
  `python -c "import secrets; print(secrets.token_urlsafe(32))"`.
- Rotation: docs/security/runbook.md.

## 8. Backups and a restore test

What holds state: the named volumes `pinneapple-apps_*_data` (jobs, uploads, API keys, usage, audit trails),
`pinneapple-apps_obs_cache`, `pinneapple-apps_caddy_data` (certificates, access log) and `apps/deploy/.env`.

```bash
#!/usr/bin/env bash
# /usr/local/sbin/pinneapple-backup, daily from a systemd timer or cron
set -euo pipefail
dest=/var/backups/pinneapple/$(date +%F)
mkdir -p "$dest"
for v in $(docker volume ls -q | grep '^pinneapple-apps_'); do
  docker run --rm -v "$v":/v:ro -v "$dest":/out alpine tar czf "/out/$v.tgz" -C /v .
done
install -m 600 /srv/pinneapple/apps/deploy/.env "$dest/env"
find /var/backups/pinneapple -maxdepth 1 -type d -mtime +14 -exec rm -rf {} +
touch /var/backups/pinneapple/last-ok           # read by check_server.sh
```

Copy the backups off the server (restic, borg or rclone to object storage, encrypted). **Restore test**, once a
quarter on a scratch VM: restore one `_data` volume, start that app, open a job from before the backup, and check the audit trail (`python -c "from pinneapple_security.audit import AuditLog; print(AuditLog('audit.jsonl').verify())"` must
report `'ok': True`).

## 9. Log retention

| Log | Retention | Setting |
|---|---|---|
| Container stdout/stderr (app auth events) | 5 × 10 MB per container | compose `logging`, `daemon.json` |
| Caddy access log | 10 × 50 MB, at most 30 days | Caddyfile `roll_keep`, `roll_keep_for` |
| systemd journal (SSH, fail2ban) | `SystemMaxUse=1G`, `MaxRetentionSec=90day` | `/etc/systemd/journald.conf.d/retention.conf` |
| App audit trails | kept with the volume; archive yearly | `audit.jsonl` in each app's data directory |

## 10. Check

```bash
sudo apps/deploy/check_server.sh
```

It reads sshd's effective settings (`sshd -T`), ufw/nftables, listening ports, fail2ban jails, unattended upgrades,
NTP, journald, `daemon.json`, every running app container (user, read-only root, capabilities,
no-new-privileges, published ports) and the backup stamp, and changes nothing. Exit code 1 when a check fails.
