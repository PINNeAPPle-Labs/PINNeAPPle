"""Find credentials committed to code, notebooks, configs and solver cases before they are shared or published.

Detectors: known token formats (cloud, Git hosting, chat, payment and AI-API keys), private key blocks, connection
strings with an embedded password, ``password = "..."``-style assignments, and long high-entropy strings (Shannon
entropy, Bitbucket/truffleHog-style heuristic). Findings carry the file, line and a masked excerpt — the secret itself
is never printed or returned in full.
"""
from __future__ import annotations

import math
import os
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass

__all__ = ["SecretFinding", "scan_text", "scan_path", "shannon_entropy", "RULES"]

RULES: dict[str, re.Pattern] = {
    "aws_access_key_id": re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\b(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{60,})\b"),
    "gitlab_token": re.compile(r"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    "slack_token": re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "stripe_key": re.compile(r"\b(?:sk|rk)_(?:live|test)_[0-9A-Za-z]{16,}\b"),
    "openai_style_key": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{32,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    "private_key_block": re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----"),
    "url_with_password": re.compile(r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:[^\s/@]{3,}@[^\s/]+", re.I),
    "password_assignment": re.compile(r"(?i)\b(?:password|passwd|pwd|secret|api[_-]?key|token|access[_-]?key)\b\s*[:=]\s*['\"]([^'\"\s]{6,})['\"]"),
}
_PLACEHOLDER = re.compile(r"(?i)(x{4,}|\*{4,}|changeme|example|placeholder|your[_-]|<[^>]+>|\$\{|\{\{|dummy|redacted|test)")
_TOKEN = re.compile(r"[A-Za-z0-9+/=_-]{32,}")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", "dist", "build"}
TEXT_EXT = {".py", ".ipynb", ".json", ".yaml", ".yml", ".toml", ".cfg", ".ini", ".env", ".txt", ".md", ".sh", ".js",
            ".ts", ".tsx", ".sql", ".csv", ".xml", ".properties", ".conf", ""}


def shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in (s.count(ch) for ch in set(s)))


def _mask(s: str) -> str:
    return s[:4] + "…" + s[-2:] if len(s) > 8 else "…"


@dataclass(frozen=True)
class SecretFinding:
    rule: str
    path: str
    line: int
    excerpt: str            # masked


def scan_text(text: str, path: str = "<text>", entropy_threshold: float = 4.5) -> list[SecretFinding]:
    out: list[SecretFinding] = []
    for i, line in enumerate(text.splitlines(), start=1):
        hit = set()
        for rule, pat in RULES.items():
            for m in pat.finditer(line):
                val = m.group(1) if m.groups() else m.group()
                if rule in ("password_assignment", "url_with_password") and _PLACEHOLDER.search(val):
                    continue
                hit.add(m.span())
                out.append(SecretFinding(rule, path, i, _mask(val)))
        for m in _TOKEN.finditer(line):
            if any(a <= m.start() < b for a, b in hit) or _PLACEHOLDER.search(m.group()):
                continue
            tok = m.group()
            if re.fullmatch(r"[0-9a-f]+", tok):          # hex digests (sha256, commit ids) are not secrets by themselves
                continue
            if shannon_entropy(tok) >= entropy_threshold:
                out.append(SecretFinding("high_entropy_string", path, i, _mask(tok)))
    return out


def _files(root: str, exts: Iterable[str] | None) -> Iterator[str]:
    exts = set(exts) if exts is not None else TEXT_EXT
    if os.path.isfile(root):
        yield root
        return
    for dirpath, dirnames, names in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for n in names:
            if os.path.splitext(n)[1].lower() in exts or n.startswith(".env"):
                yield os.path.join(dirpath, n)


def scan_path(root: str, exts: Iterable[str] | None = None, max_bytes: int = 5_000_000,
              entropy_threshold: float = 4.5) -> list[SecretFinding]:
    out: list[SecretFinding] = []
    for p in _files(root, exts):
        try:
            if os.path.getsize(p) > max_bytes:
                continue
            with open(p, "rb") as f:
                raw = f.read()
            if b"\x00" in raw[:4096]:
                continue
            out += scan_text(raw.decode("utf-8", errors="replace"), os.path.relpath(p, root) if os.path.isdir(root) else p,
                             entropy_threshold)
        except OSError:
            continue
    return out
