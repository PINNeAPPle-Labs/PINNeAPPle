"""pinneapple_security web hardening: passwords, safe formula evaluation, zip extraction, rate limits, lockout,
client IP behind a proxy, CSRF origin check, CORS allow-list, CSP hashes, audit trail under concurrent writers."""
from __future__ import annotations

import io
import json
import multiprocessing as mp
import zipfile

import numpy as np
import pytest

from pinneapple_data import expressions
from pinneapple_security import archives, audit, passwords, web


# ------------------------------------------------------------------------------------------------- passwords
def test_scrypt_hash_roundtrip_and_salted():
    h1, h2 = passwords.hash_password("s3cret", "scrypt"), passwords.hash_password("s3cret", "scrypt")
    assert h1 != h2 and h1.startswith("$scrypt$") and "s3cret" not in h1
    assert passwords.verify_password("s3cret", h1) and not passwords.verify_password("wrong", h1)
    assert passwords.is_hash(h1) and not passwords.is_hash("s3cret")
    assert not passwords.needs_rehash(h1)


def test_verify_rejects_garbage_and_absurd_parameters():
    assert not passwords.verify_password("x", None)
    assert not passwords.verify_password("x", "plain")
    assert not passwords.verify_password("x", "$scrypt$n=1073741824,r=8,p=1$AAAA$AAAA")   # DoS parameters
    assert not passwords.verify_password("x", "$scrypt$broken")


def test_argon2_when_installed():
    pytest.importorskip("argon2")
    h = passwords.hash_password("pw", "argon2")
    assert h.startswith("$argon2id$") and passwords.verify_password("pw", h) and not passwords.verify_password("no", h)


# ------------------------------------------------- user formulas (pinneapple_data.expressions)
FUNCS = {"abs": np.abs, "sqrt": np.sqrt}


@pytest.mark.parametrize("expr,expected", [("Cd > 0", True), ("abs(u) <= 1", True), ("a == b * 2", True),
                                           ("sqrt(a) <= 2 and not Cd < 0", True), ("u ** 2", 0.25)])
def test_expressions_accept_formulas(expr, expected):
    env = {"Cd": 0.3, "u": -0.5, "a": 4.0, "b": 2.0}
    assert expressions.evaluate(expr, env, FUNCS) == expected


ESCAPES = ["().__class__", "Cd.__class__.__mro__", "[c for c in ()]", "(lambda: 1)()", "x[0]", "open('f')",
           "__import__('os')", "abs(x=1)", "abs(*x)", "a := 1", "a" * 3000]


@pytest.mark.parametrize("expr", ESCAPES)
def test_expressions_refuse_escapes(expr):
    with pytest.raises(expressions.ExpressionError):
        expressions.evaluate(expr, {"Cd": 1.0, "a": 1.0, "x": 1.0}, FUNCS)


def test_constraint_and_boundary_evaluators_use_the_interpreter():
    """The dataset constraints (benchmark arena, audit) and declared boundary values (Validation Lab via
    pinneapple check) refuse what eval() with emptied builtins would have run."""
    from pinneapple_check.checks.results import evaluate_expr
    from pinneapple_data.audit import constraint_mask
    cols = {"Cd": np.array([0.1, -0.2])}
    assert constraint_mask("Cd > 0", cols).tolist() == [True, False]
    assert np.allclose(evaluate_expr("sin(pi*x)", {"x": np.array([0.5])}, 1), 1.0)
    for bad in ("().__class__.__base__.__subclasses__()", "Cd.__class__"):
        with pytest.raises(Exception, match="not allowed|unknown name|may be called"):
            constraint_mask(bad, cols)
        with pytest.raises(ValueError):
            evaluate_expr(bad, {"x": np.zeros(1)}, 1)
    assert expressions.evaluate("9 ** 9 ** 9", {}, {}) is not None    # numpy powers overflow, never stall


def test_meshless_expression_guard():
    pytest.importorskip("sympy")
    from pinneapple_simulation.meshless.problems import _expr
    _expr("sin(pi*x)*y^2 + exp(-t)", time=True)
    for bad in ("x.__class__", "Symbol('z')", "x if y else t"):
        with pytest.raises(ValueError):
            _expr(bad, time=True)


# -------------------------------------------------------------------------------------------------- archives
def _zip(members: dict[str, bytes], symlink: str | None = None) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
        if symlink:
            info = zipfile.ZipInfo(symlink)
            info.external_attr = (0o120777 << 16)
            z.writestr(info, "/etc/passwd")
    return buf.getvalue()


def test_safe_extract_writes_inside_dest(tmp_path):
    files = archives.safe_extract_zip(_zip({"case/system/controlDict": b"x", "a.txt": b"y"}), tmp_path / "out")
    assert sorted(p.name for p in files) == ["a.txt", "controlDict"]
    assert all(p.is_relative_to((tmp_path / "out").resolve()) for p in files)


@pytest.mark.parametrize("name", ["../evil.txt", "a/../../evil.txt", "/etc/cron.d/x", "C:/windows/x", "..\\evil"])
def test_safe_extract_refuses_traversal(tmp_path, name):
    with pytest.raises(archives.UnsafeArchive):
        archives.safe_extract_zip(_zip({name: b"x"}), tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()


def test_safe_extract_refuses_links_and_bombs(tmp_path):
    with pytest.raises(archives.UnsafeArchive, match="link"):
        archives.safe_extract_zip(_zip({"a": b"x"}, symlink="link"), tmp_path / "o1")
    bomb = _zip({"zeros.bin": b"\0" * (50 << 20)})               # 50 MB of zeros, ~50 kB compressed
    with pytest.raises(archives.UnsafeArchive, match="ratio"):
        archives.safe_extract_zip(bomb, tmp_path / "o2")
    with pytest.raises(archives.UnsafeArchive, match="expands"):
        archives.read_members(bomb, archives.ArchiveLimits(max_total_bytes=1 << 20, max_ratio=1e9))
    with pytest.raises(archives.UnsafeArchive, match="members"):
        archives.read_members(_zip({f"f{i}": b"" for i in range(30)}), archives.ArchiveLimits(max_members=10))


# ---------------------------------------------------------------------------------------------- rate limits
class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_token_bucket_burst_then_refill():
    clock = Clock()
    rule = web.RateRule("write", 3, 60)
    rl = web.RateLimiter([rule], clock=clock)
    assert [rl.hit("ip:1", rule) for _ in range(3)] == [0.0, 0.0, 0.0]
    wait = rl.hit("ip:1", rule)
    assert 19 < wait <= 20.0                                    # one token per 20 s
    assert rl.hit("ip:2", rule) == 0.0                          # budgets are per key
    clock.t += 21
    assert rl.hit("ip:1", rule) == 0.0


def test_rate_rules_match_method_and_path_and_parse():
    rule = web.RateRule("w", 1, 1, frozenset({"POST"}), ("/api/",))
    assert rule.matches("POST", "/api/run") and not rule.matches("GET", "/api/run") and not rule.matches("POST", "/x")
    assert web.parse_rate("30/min") == (30, 60) and web.parse_rate("5/s") == (5, 1) and web.parse_rate("10/90") == (10, 90)
    assert web.parse_rate("off") is None and web.parse_rate(None, (1, 2)) == (1, 2)


def test_rate_limiter_memory_is_bounded():
    rule = web.RateRule("r", 10, 1)
    rl = web.RateLimiter([rule], max_keys=100)
    for i in range(1000):
        rl.hit(f"ip:{i}", rule)
    assert len(rl._buckets) == 100


def test_login_lockout_per_ip_and_user():
    clock = Clock()
    th = web.LoginThrottle(max_failures=3, window=60, lock_seconds=300, clock=clock)
    assert not th.failure("1.1.1.1", "admin") and not th.failure("1.1.1.1", "admin")
    assert th.failure("1.1.1.1", "admin")                        # third failure starts the lockout
    assert th.locked_for("1.1.1.1") == 300 and th.locked_for("9.9.9.9", "admin") == 300   # user locked from anywhere
    assert th.locked_for("9.9.9.9", "other") == 0
    clock.t += 301
    assert th.locked_for("1.1.1.1", "admin") == 0


# ------------------------------------------------------------------------------------------------- client IP
def test_forwarded_for_only_from_trusted_proxy():
    assert web.client_ip("172.18.0.3", "203.0.113.7") == "203.0.113.7"              # Caddy on the Docker network
    assert web.client_ip("203.0.113.50", "10.0.0.1") == "203.0.113.50"              # direct client: header ignored
    assert web.client_ip("172.18.0.3", "198.51.100.1, 203.0.113.7") == "203.0.113.7"  # right-most untrusted hop
    assert web.client_ip("172.18.0.3", "203.0.113.7, 10.0.0.9") == "203.0.113.7"
    assert web.client_ip("testclient", "203.0.113.7") == "testclient"
    assert web.client_ip("192.0.2.1", "203.0.113.7", trusted=["192.0.2.0/24"]) == "203.0.113.7"


# ------------------------------------------------------------------------------------------------ CSRF / CORS
def test_same_origin_check():
    ok = web.same_origin_ok
    assert ok("GET", {"origin": "https://evil.example", "sec-fetch-site": "cross-site"})
    assert not ok("POST", {"origin": "https://evil.example", "host": "app.example", "sec-fetch-site": "cross-site"})
    assert not ok("POST", {"origin": "https://evil.example", "host": "app.example"})        # older browser
    assert ok("POST", {"origin": "https://app.example", "host": "app.example", "sec-fetch-site": "same-origin"})
    assert ok("POST", {"origin": "https://app.example", "x-forwarded-host": "app.example", "host": "heatsink:8080"})
    assert ok("POST", {})                                                                  # curl / API client
    assert not ok("POST", {"origin": "null", "host": "app.example"})
    assert ok("POST", {"origin": "https://tool.example", "host": "app.example", "sec-fetch-site": "cross-site"},
              allowed_origins=["https://tool.example"])


def test_cors_allow_list_validation():
    assert web.check_cors_origins(["https://a.example", "http://localhost:3000/"], True) == [
        "https://a.example", "http://localhost:3000"]
    with pytest.raises(ValueError, match="credentials"):
        web.check_cors_origins(["*"], True)
    with pytest.raises(ValueError):
        web.check_cors_origins(["https://a.example/path"], False)
    with pytest.raises(ValueError):
        web.check_cors_origins(["a.example"], False)


# ------------------------------------------------------------------------------------------------------- CSP
def test_inline_hashes_and_csp():
    page = ('<script type="importmap">{"imports": {}}</script><script src="/a.js"></script>'
            '<form onsubmit="return false"></form>')
    scripts, handlers = web.inline_hashes(page)
    assert len(scripts) == 1 and len(handlers) == 1
    csp = web.build_csp(scripts, handlers)
    assert "'unsafe-inline'" not in csp.split("script-src", 1)[1].split(";")[0]
    assert "'unsafe-hashes'" in csp and "frame-ancestors 'none'" in csp and "object-src 'none'" in csp


# ----------------------------------------------------------------------------------------------- audit trail
def _append_many(path: str, who: str, n: int) -> None:
    log = audit.AuditLog(path)
    for i in range(n):
        log.append("login_ok", who, target=str(i))


def test_audit_chain_survives_concurrent_processes(tmp_path):
    """Two server workers writing the same trail must not fork the hash chain."""
    path = str(tmp_path / "audit.jsonl")
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_append_many, args=(path, f"user:{k}", 40)) for k in range(3)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(60)
    v = audit.AuditLog(path).verify()
    assert v["ok"] and v["n"] == 120, v


def test_auth_audit_writes_log_and_trail(tmp_path, caplog):
    a = web.AuthAudit("tst", path=str(tmp_path / "a.jsonl"))
    with caplog.at_level("INFO", logger="pinneapple.security"):
        a.event("login_failed", "user:bob", ip="203.0.113.7", outcome="denied")
    rec = json.loads(caplog.records[-1].message)
    assert rec["event"] == "login_failed" and rec["ip"] == "203.0.113.7"
    assert audit.AuditLog(str(tmp_path / "a.jsonl")).verify()["n"] == 1


def test_cae_upload_expand_and_gunzip_limits():
    """The shared CAE upload reader (Interop Hub, Mesh Quality, Model Lineage, Comparator, Validation Lab)."""
    import gzip

    from pinneapple_data.cae.upload import expand
    fs = expand([("case.zip", _zip({"c/system/controlDict": b"x", "c/processor0/p": b"y", "c/__MACOSX/z": b""}))])
    assert fs == {"system/controlDict": b"x"}
    with pytest.raises(ValueError, match="refused"):
        expand([("case.zip", _zip({"../x": b"x"}))])
    with pytest.raises(ValueError, match="refused"):
        expand([("big.zip", _zip({"zeros": b"\0" * (3 << 20)}))], max_unzipped=1 << 20)
    assert archives.gunzip(gzip.compress(b"abc" * 10)) == b"abc" * 10
    with pytest.raises(archives.UnsafeArchive):
        archives.gunzip(gzip.compress(b"\0" * (5 << 20)), max_bytes=1 << 20)


@pytest.mark.parametrize("target", ["/etc/hostname", "../outside.npz"])
def test_validation_lab_config_confined_to_uploads(tmp_path, target):
    """An uploaded pinneapple-check.toml can not point the Validation Lab at server files."""
    from pinneapple_veriphysics.validation_lab import run_validation
    root = tmp_path / "upload"
    root.mkdir()
    (tmp_path / "outside.npz").write_bytes(b"")
    (root / "pinneapple-check.toml").write_text(f'[[result]]\npath = "{target}"\nbounds = {{ T = [0, 1] }}\n')
    with pytest.raises(ValueError, match="outside the uploaded files"):
        run_validation(root)
