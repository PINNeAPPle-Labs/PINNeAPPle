#!/usr/bin/env bash
# Read-only security check of the server that runs the PINNeAPPle apps (docs/security/server.md).
# Changes nothing; prints PASS / WARN / FAIL per check and exits 1 when any check fails.
#
#   sudo apps/deploy/check_server.sh            # as root, so it can read sshd, firewall and Docker settings
#
# Optional environment:
#   SSH_PORT=22                  the SSH port the firewall should allow
#   PROXY_CONTAINER=caddy        name fragment of the only container allowed to publish ports
#   APP_PREFIX=pinneapple        name fragment of the app containers (must run as non-root, no published ports)
#   BACKUP_STAMP=/var/backups/pinneapple/last-ok   file touched by a successful backup; must be < BACKUP_MAX_AGE_H old
#   BACKUP_MAX_AGE_H=36
set -u
SSH_PORT=${SSH_PORT:-22}
PROXY_CONTAINER=${PROXY_CONTAINER:-caddy}
APP_PREFIX=${APP_PREFIX:-pinneapple}
BACKUP_STAMP=${BACKUP_STAMP:-/var/backups/pinneapple/last-ok}
BACKUP_MAX_AGE_H=${BACKUP_MAX_AGE_H:-36}
pass=0; warn=0; fail=0

ok()   { printf 'PASS  %s\n' "$1"; pass=$((pass + 1)); }
bad()  { printf 'FAIL  %s\n' "$1"; fail=$((fail + 1)); }
meh()  { printf 'WARN  %s\n' "$1"; warn=$((warn + 1)); }
have() { command -v "$1" >/dev/null 2>&1; }

[ "$(id -u)" -eq 0 ] || meh "not running as root: some settings can not be read"

# ------------------------------------------------------------------------------------------------ SSH
if have sshd; then
  cfg=$(sshd -T 2>/dev/null || true)
  if [ -z "$cfg" ]; then
    meh "sshd -T gave no output (run as root)"
  else
    v() { printf '%s\n' "$cfg" | awk -v k="$1" '$1 == k {print $2; exit}'; }
    [ "$(v passwordauthentication)" = "no" ] && ok "SSH: password logins off" || bad "SSH: PasswordAuthentication is not 'no'"
    [ "$(v kbdinteractiveauthentication)" = "no" ] && ok "SSH: keyboard-interactive off" \
      || bad "SSH: KbdInteractiveAuthentication is not 'no'"
    case "$(v permitrootlogin)" in
      no|prohibit-password|without-password) ok "SSH: root login only with a key (or never)";;
      *) bad "SSH: PermitRootLogin allows passwords";;
    esac
    [ "$(v pubkeyauthentication)" = "yes" ] && ok "SSH: key logins on" || bad "SSH: PubkeyAuthentication is off"
    mt=$(v maxauthtries); [ -n "$mt" ] && [ "$mt" -le 4 ] && ok "SSH: MaxAuthTries $mt" || meh "SSH: MaxAuthTries ${mt:-?} (4 or less advised)"
  fi
else
  meh "sshd not found"
fi

# ------------------------------------------------------------------------------------------- firewall
if have ufw && ufw status 2>/dev/null | grep -q "Status: active"; then
  ok "firewall: ufw active"
  rules=$(ufw status 2>/dev/null | awk 'NR > 4 && /ALLOW/ {print $1}' | sed 's#/.*##' | sort -u | tr '\n' ' ')
  extra=$(for r in $rules; do case "$r" in "$SSH_PORT"|80|443|OpenSSH|"Nginx"|"WWW"*) ;; *) printf '%s ' "$r";; esac; done)
  [ -z "$extra" ] && ok "firewall: only SSH, 80 and 443 allowed" || bad "firewall: also allows $extra"
  ufw status verbose 2>/dev/null | grep -q "Default: deny (incoming)" && ok "firewall: incoming denied by default" \
    || bad "firewall: default incoming policy is not deny"
elif have nft && nft list ruleset 2>/dev/null | grep -q "hook input"; then
  if nft list ruleset 2>/dev/null | grep -Eq "hook input.*policy drop"; then
    ok "firewall: nftables input chain drops by default"
  else
    bad "firewall: nftables input chain does not drop by default"
  fi
else
  bad "firewall: neither ufw nor an nftables input chain is active"
fi
if have docker && have iptables; then
  iptables -S DOCKER-USER 2>/dev/null | grep -q -- "-j" && ok "firewall: DOCKER-USER chain has rules (Docker ports filtered)" \
    || meh "firewall: DOCKER-USER chain is empty (ports Docker publishes bypass ufw; only Caddy should publish)"
fi

# ------------------------------------------------------------------------------------ listening ports
if have ss; then
  public=$(ss -Htlnu 2>/dev/null | awk '{print $5}' | grep -Ev '^(127\.|\[::1\]|::1)' | sed 's#.*:##' | sort -un | tr '\n' ' ')
  extra=$(for p in $public; do case "$p" in "$SSH_PORT"|80|443|53) ;; *) printf '%s ' "$p";; esac; done)
  [ -z "$extra" ] && ok "listening publicly only on SSH, 80, 443 (ports: ${public:-none})" \
    || bad "also listening publicly on: $extra"
else
  meh "ss not found"
fi

# ------------------------------------------------------------------------------------------- fail2ban
if have fail2ban-client && fail2ban-client ping >/dev/null 2>&1; then
  ok "fail2ban running"
  jails=$(fail2ban-client status 2>/dev/null | sed -n 's/.*Jail list:\s*//p')
  case "$jails" in *sshd*) ok "fail2ban: sshd jail";; *) bad "fail2ban: no sshd jail";; esac
  case "$jails" in *pinneapple*) ok "fail2ban: pinneapple jail(s): $jails";; *) bad "fail2ban: no pinneapple jail (apps/deploy/fail2ban)";; esac
else
  bad "fail2ban not running"
fi

# ---------------------------------------------------------------------------------- automatic updates
if [ -f /etc/apt/apt.conf.d/20auto-upgrades ] && grep -q 'Unattended-Upgrade "1"' /etc/apt/apt.conf.d/20auto-upgrades; then
  ok "unattended security upgrades on"
elif have dnf && systemctl is-enabled dnf-automatic.timer >/dev/null 2>&1; then
  ok "dnf-automatic on"
else
  bad "automatic security updates are not enabled"
fi
[ -f /var/run/reboot-required ] && meh "a reboot is pending (kernel or libc update)" || ok "no reboot pending"

# ----------------------------------------------------------------------------------------- time, logs
if have timedatectl; then
  timedatectl show -p NTPSynchronized --value 2>/dev/null | grep -q yes && ok "clock synchronised (NTP)" || meh "clock not NTP-synchronised"
fi
if [ -f /etc/systemd/journald.conf ] || [ -d /etc/systemd/journald.conf.d ]; then
  grep -rqsE '^\s*SystemMaxUse=' /etc/systemd/journald.conf /etc/systemd/journald.conf.d/ \
    && ok "journald size limit set" || meh "journald has no SystemMaxUse (log retention unbounded by size)"
fi

# ---------------------------------------------------------------------------------------------- Docker
if have docker && docker info >/dev/null 2>&1; then
  dj=/etc/docker/daemon.json
  if [ -f "$dj" ]; then
    grep -q '"no-new-privileges"\s*:\s*true' "$dj" && ok "docker: no-new-privileges by default" || meh "docker: daemon.json lacks \"no-new-privileges\": true"
    grep -q '"live-restore"\s*:\s*true' "$dj" && ok "docker: live-restore" || meh "docker: daemon.json lacks \"live-restore\": true"
    grep -q '"max-size"' "$dj" && ok "docker: log rotation (max-size)" || bad "docker: no default log rotation in daemon.json"
    grep -q '"icc"\s*:\s*false' "$dj" && ok "docker: inter-container traffic on the default bridge off" || meh "docker: \"icc\": false not set"
  else
    bad "docker: no /etc/docker/daemon.json (docs/security/server.md has one)"
  fi
  [ -S /var/run/docker.sock ] && [ "$(stat -c %a /var/run/docker.sock)" -le 660 ] && ok "docker: socket not world-accessible" \
    || bad "docker: /var/run/docker.sock is world-accessible"
  while read -r name ports; do
    case "$name" in *"$PROXY_CONTAINER"*) continue;; esac
    if printf '%s' "$ports" | grep -q -- '->'; then bad "docker: $name publishes ports: $ports"; fi
  done < <(docker ps --format '{{.Names}} {{.Ports}}')
  napps=0
  while read -r name; do
    napps=$((napps + 1))
    u=$(docker inspect -f '{{.Config.User}}' "$name")
    ro=$(docker inspect -f '{{.HostConfig.ReadonlyRootfs}}' "$name")
    caps=$(docker inspect -f '{{.HostConfig.CapDrop}}' "$name")
    nnp=$(docker inspect -f '{{.HostConfig.SecurityOpt}}' "$name")
    case "$u" in ""|0|0:*|root|root:*) bad "docker: $name runs as root";; *) ok "docker: $name runs as $u";; esac
    [ "$ro" = "true" ] || meh "docker: $name root filesystem is writable"
    case "$caps" in *ALL*) ;; *) bad "docker: $name keeps Linux capabilities ($caps)";; esac
    case "$nnp" in *no-new-privileges*) ;; *) bad "docker: $name allows privilege escalation";; esac
    [ "$(docker inspect -f '{{.HostConfig.Privileged}}' "$name")" = "false" ] || bad "docker: $name is privileged"
  done < <(docker ps --format '{{.Names}}' | grep -i -- "$APP_PREFIX" | grep -v -- "$PROXY_CONTAINER")
  [ "$napps" -gt 0 ] || meh "docker: no running container matches '$APP_PREFIX'"
else
  meh "docker not reachable (not installed, or not root)"
fi

# --------------------------------------------------------------------------------------------- backups
if [ -f "$BACKUP_STAMP" ]; then
  age_h=$(( ( $(date +%s) - $(stat -c %Y "$BACKUP_STAMP") ) / 3600 ))
  [ "$age_h" -le "$BACKUP_MAX_AGE_H" ] && ok "backup: last success ${age_h} h ago" || bad "backup: last success ${age_h} h ago (> $BACKUP_MAX_AGE_H h)"
else
  bad "backup: no stamp at $BACKUP_STAMP (docs/security/server.md, Backups)"
fi

printf '\n%d passed, %d warnings, %d failed\n' "$pass" "$warn" "$fail"
[ "$fail" -eq 0 ]
