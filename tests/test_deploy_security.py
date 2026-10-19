"""Server deployment (apps/deploy, the app Dockerfiles): non-root containers, read-only root filesystems, no Linux
capabilities, no privilege escalation, limits, no app port on the host, Caddy with security headers and a body
limit on every site, no secrets in the templates."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

ROOT = Path(__file__).resolve().parents[1]
DEPLOY = ROOT / "apps" / "deploy"
COMPOSE = yaml.safe_load((DEPLOY / "docker-compose.yml").read_text())
SERVICES = COMPOSE["services"]
WRITES_OUTSIDE_DATA = {"guiagent", "ollama"}          # FreeCAD/X11 stack; Ollama's own image layout


def test_only_caddy_publishes_ports():
    published = {name for name, s in SERVICES.items() if s.get("ports")}
    assert published == {"caddy"}
    proxy = yaml.safe_load((DEPLOY / "docker-compose.existing-proxy.yml").read_text())
    assert not any(s.get("ports") for s in proxy["services"].values())


@pytest.mark.parametrize("name", sorted(SERVICES))
def test_service_hardening(name):
    s = SERVICES[name]
    assert s.get("cap_drop") == ["ALL"], name
    assert "no-new-privileges:true" in s.get("security_opt", []), name
    assert s.get("pids_limit"), name
    assert s.get("logging", {}).get("options", {}).get("max-size"), name
    if name in ("caddy", "ollama"):
        assert set(s.get("cap_add", [])) <= {"NET_BIND_SERVICE"}
        return
    if name not in WRITES_OUTSIDE_DATA:
        assert s.get("read_only") is True, name
        assert any(str(t).startswith("/tmp") for t in s.get("tmpfs", [])), name
        assert str(s.get("user", "")).split(":")[0] not in ("", "0", "root"), name
    assert s.get("mem_limit"), name


DOCKERFILES = sorted(ROOT.glob("apps/*/Dockerfile")) + [ROOT / "pinneapple_lab" / "deploy" / "Dockerfile"]


@pytest.mark.parametrize("path", DOCKERFILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_images_run_as_non_root(path):
    users = re.findall(r"^USER\s+(\S+)", path.read_text(), re.M)
    assert users and users[-1].split(":")[0] not in ("root", "0"), path
    assert not re.search(r"^ADD\s+https?://", path.read_text(), re.M)          # no unverified remote downloads


def _sites(caddyfile: str) -> dict[str, str]:
    return {m.group(1): m.group(2) for m in re.finditer(r"^\{\$(\w+)\} \{(.*?)^\}", caddyfile, re.M | re.S)}


def test_caddy_headers_limits_and_logs():
    text = (DEPLOY / "Caddyfile").read_text()
    common = text[text.index("(common) {"):text.index("{$HSS_DOMAIN}")]
    for needle in ("Strict-Transport-Security", "X-Content-Type-Options", "X-Frame-Options", "Referrer-Policy",
                   "Permissions-Policy", "-Server", "protocols tls1.2 tls1.3", "log {", "roll_keep"):
        assert needle in common, needle
    sites = _sites(text)
    assert len(sites) >= 35
    for name, body in sites.items():
        assert "import common" in body, name
        assert "max_size" in body, f"{name} has no request body limit"


def test_env_template_holds_no_real_secrets():
    env = (DEPLOY / ".env.example").read_text()
    for line in env.splitlines():
        if re.match(r"^\w+_(PASSWORD|PASSWORD_HASH|ADMIN_TOKEN|TOKEN|SECRET|AUDIT_KEY)=", line):
            value = line.split("=", 1)[1]
            assert value in ("", "change-me"), line
    assert ".env" in (DEPLOY / ".gitignore").read_text().split()
