"""The tools the agents use, confined to one workspace folder.

Files: ``list_dir``, ``read_file``, ``write_file``, ``edit_file`` (exact string replacement), ``search`` (regex over
files), ``glob``. Commands: ``run`` (a shell command with a timeout, run in the workspace: git, tests, scripts).
Background jobs: ``run(background=true)``, ``job_output``, ``job_stop``. Internet: ``web_search`` and ``fetch_url``
(see :mod:`.web`); off with ``Workspace(web=False)``. GitHub: ``github`` (REST calls with ``GITHUB_TOKEN``). Showing
work: ``show_file`` puts an image, HTML page, PDF or text file in the chat. A path that resolves outside
the workspace is refused, symlinks included. Whether a command needs your approval is the
:class:`ApprovalPolicy`'s call: ``"ask"`` asks for every command and every write, ``"auto"`` lets read-only and test
commands through and asks for the rest, ``"yolo"`` asks for nothing.
"""
from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

__all__ = ["Workspace", "ApprovalPolicy", "ToolError", "TOOL_SPECS", "SAFE_COMMANDS"]

MAX_OUTPUT = 12_000                    # characters of tool output handed back to the model
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", ".mypy_cache", ".pytest_cache", ".ruff_cache",
             "dist", "build", ".tox", ".idea"}

# Command prefixes "auto" runs without asking: they read, test or lint, and change nothing outside caches.
SAFE_COMMANDS = ("ls", "pwd", "cat", "head", "tail", "wc", "grep", "rg", "find", "tree", "which", "echo",
                 "git status", "git diff", "git log", "git show", "git branch", "pytest", "python -m pytest",
                 "ruff check", "python -m ruff check", "mypy", "python --version", "pip list", "pip show",
                 "npm test", "npm run test", "npm run lint", "npm run build")
_SHELL_META = re.compile(r"[;&|`$<>]|\b(rm|mv|sudo|curl|wget|chmod|chown|dd)\b")


class ToolError(Exception):
    pass


def _clip(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    return f"{text[:half]}\n... [{len(text) - limit} characters cut] ...\n{text[-half:]}"


@dataclass
class ApprovalPolicy:
    mode: str = "auto"                 # "ask" | "auto" | "yolo"

    def needs_approval(self, tool: str, args: dict) -> bool:
        if self.mode == "yolo":
            return False
        if tool in ("write_file", "edit_file"):
            return self.mode == "ask"
        if tool == "github":
            return str(args.get("method", "GET")).upper() != "GET"
        if tool == "job_stop":
            return False
        if tool != "run":
            return False
        if self.mode == "ask":
            return True
        cmd = " ".join(str(args.get("command", "")).split())
        return _SHELL_META.search(cmd) is not None or not any(cmd == p or cmd.startswith(p + " ")
                                                               for p in SAFE_COMMANDS)


class Workspace:
    """A folder the agents may read and change. ``approve(tool, args) -> bool`` is asked before guarded actions."""

    def __init__(self, root: str | os.PathLike, policy: ApprovalPolicy | None = None,
                 approve: Callable[[str, dict], bool] | None = None, run_timeout: float = 600, web: bool = True):
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise ToolError(f"workspace {self.root} is not a folder")
        self.policy = policy or ApprovalPolicy()
        self.approve = approve or (lambda tool, args: False)
        self.run_timeout = run_timeout
        self.web = web
        self.jobs: dict[str, dict] = {}
        self.shown: list[dict] = []                 # files passed to show_file, for the UI

    # -- paths ---------------------------------------------------------------------------------------------------
    def path(self, rel: str) -> Path:
        p = (self.root / (rel or ".")).resolve()
        if p != self.root and self.root not in p.parents:
            raise ToolError(f"{rel!r} is outside the workspace {self.root}")
        return p

    def rel(self, p: Path) -> str:
        return str(p.relative_to(self.root)) or "."

    # -- dispatch ------------------------------------------------------------------------------------------------
    def call(self, name: str, args: dict) -> str:
        fn = getattr(self, f"t_{name}", None)
        if name in ("web_search", "fetch_url") and not self.web:
            raise ToolError("internet access is off for this session")
        if fn is None:
            raise ToolError(f"unknown tool {name!r}")
        if self.policy.needs_approval(name, args) and not self.approve(name, args):
            return f"DENIED: the user did not approve this {name}. Do something else or ask the user."
        try:
            return _clip(fn(**args))
        except TypeError as e:
            raise ToolError(f"bad arguments for {name}: {e}") from e

    # -- tools ---------------------------------------------------------------------------------------------------
    def t_list_dir(self, path: str = ".") -> str:
        p = self.path(path)
        if not p.is_dir():
            raise ToolError(f"{path} is not a folder")
        rows = []
        for c in sorted(p.iterdir(), key=lambda c: (not c.is_dir(), c.name)):
            if c.name in SKIP_DIRS:
                continue
            rows.append(c.name + "/" if c.is_dir() else f"{c.name}  ({c.stat().st_size} B)")
        return "\n".join(rows) or "(empty)"

    def t_read_file(self, path: str, offset: int = 1, limit: int = 400) -> str:
        p = self.path(path)
        if not p.is_file():
            raise ToolError(f"{path} is not a file")
        lines = p.read_text(errors="replace").splitlines()
        start = max(1, int(offset))
        chunk = lines[start - 1:start - 1 + int(limit)]
        body = "\n".join(f"{i:6d}\t{line}" for i, line in enumerate(chunk, start))
        more = len(lines) - (start - 1 + len(chunk))
        return body + (f"\n... {more} more lines (read again with offset={start + len(chunk)})" if more > 0 else "")

    def t_write_file(self, path: str, content: str) -> str:
        p = self.path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content)
        return f"wrote {self.rel(p)} ({len(content.splitlines())} lines)"

    def t_edit_file(self, path: str, old: str, new: str, replace_all: bool = False) -> str:
        p = self.path(path)
        if not p.is_file():
            raise ToolError(f"{path} is not a file")
        text = p.read_text()
        n = text.count(old)
        if n == 0:
            raise ToolError("the old text was not found; read the file again and copy the text exactly")
        if n > 1 and not replace_all:
            raise ToolError(f"the old text occurs {n} times; add surrounding lines or set replace_all")
        p.write_text(text.replace(old, new) if replace_all else text.replace(old, new, 1))
        return f"edited {self.rel(p)} ({n if replace_all else 1} replacement)"

    def _files(self, base: Path):
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
            for f in filenames:
                yield Path(dirpath) / f

    def t_glob(self, pattern: str, path: str = ".") -> str:
        base = self.path(path)
        hits = [self.rel(f) for f in self._files(base) if fnmatch.fnmatch(self.rel(f), pattern)
                or fnmatch.fnmatch(f.name, pattern)]
        return "\n".join(sorted(hits)[:500]) or "(no match)"

    def t_search(self, pattern: str, path: str = ".", file_glob: str = "*", max_hits: int = 200) -> str:
        try:
            rx = re.compile(pattern)
        except re.error as e:
            raise ToolError(f"bad regex: {e}") from e
        base, hits = self.path(path), []
        files = [base] if base.is_file() else self._files(base)
        for f in files:
            if not fnmatch.fnmatch(f.name, file_glob):
                continue
            try:
                for i, line in enumerate(f.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"{self.rel(f)}:{i}: {line.strip()[:240]}")
                        if len(hits) >= max_hits:
                            return "\n".join(hits) + f"\n... stopped at {max_hits} hits"
            except (OSError, UnicodeError):
                continue
        return "\n".join(hits) or "(no match)"

    def t_run(self, command: str, timeout: float | None = None, background: bool = False) -> str:
        if background:
            return self._start_job(command)
        try:
            proc = subprocess.run(command, shell=True, cwd=self.root, capture_output=True, text=True,
                                  timeout=min(float(timeout or self.run_timeout), self.run_timeout))
        except subprocess.TimeoutExpired as e:
            out = (e.stdout or "") + (e.stderr or "") if isinstance(e.stdout, str) else ""
            return f"TIMEOUT after {e.timeout:.0f} s\n{out}"
        out = (proc.stdout or "") + (("\n[stderr]\n" + proc.stderr) if proc.stderr else "")
        return f"exit code {proc.returncode}\n{out}"

    # -- background jobs -----------------------------------------------------------------------------------------
    def _start_job(self, command: str) -> str:
        import tempfile
        import uuid

        jid = uuid.uuid4().hex[:8]
        log = Path(tempfile.gettempdir()) / f"pinneapple-wb-job-{jid}.log"
        fh = open(log, "w")                          # noqa: SIM115 - kept open for the child's lifetime
        proc = subprocess.Popen(command, shell=True, cwd=self.root, stdout=fh, stderr=subprocess.STDOUT, text=True)
        self.jobs[jid] = {"command": command, "proc": proc, "log": log, "fh": fh}
        return f"started job {jid} (pid {proc.pid}); read it with job_output, stop it with job_stop"

    def t_job_output(self, job_id: str, tail: int = 200) -> str:
        job = self.jobs.get(job_id)
        if job is None:
            raise ToolError(f"no job {job_id!r}; jobs: {sorted(self.jobs)}")
        job["fh"].flush()
        code = job["proc"].poll()
        lines = job["log"].read_text(errors="replace").splitlines()[-int(tail):]
        state = "running" if code is None else f"finished, exit code {code}"
        return f"job {job_id} ({job['command']}): {state}\n" + "\n".join(lines)

    def t_job_stop(self, job_id: str) -> str:
        job = self.jobs.get(job_id)
        if job is None:
            raise ToolError(f"no job {job_id!r}")
        if job["proc"].poll() is None:
            job["proc"].terminate()
            try:
                job["proc"].wait(10)
            except subprocess.TimeoutExpired:
                job["proc"].kill()
        return f"job {job_id} stopped (exit code {job['proc'].poll()})"

    def close(self) -> None:
        for jid in list(self.jobs):
            self.t_job_stop(jid)
            self.jobs[jid]["fh"].close()

    # -- showing files to the user ---------------------------------------------------------------------------
    def t_show_file(self, path: str, caption: str = "") -> str:
        p = self.path(path)
        if not p.is_file():
            raise ToolError(f"{path} is not a file")
        self.shown.append({"path": self.rel(p), "caption": caption, "size": p.stat().st_size})
        return f"shown to the user: {self.rel(p)}"

    # -- GitHub --------------------------------------------------------------------------------------------------
    def t_github(self, path: str, method: str = "GET", body: dict | None = None) -> str:
        """A GitHub REST call, e.g. path='repos/OWNER/REPO/pulls?state=open'."""
        import json as _json

        import requests

        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            raise ToolError("set GITHUB_TOKEN (a fine-grained personal access token) to use the GitHub tool")
        url = path if path.startswith("https://") else "https://api.github.com/" + path.lstrip("/")
        r = requests.request(method.upper(), url, json=body, timeout=60,
                             headers={"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                                      "X-GitHub-Api-Version": "2022-11-28"})
        try:
            data = _json.dumps(r.json(), indent=1)
        except ValueError:
            data = r.text
        return f"HTTP {r.status_code}\n{data}"

    def t_web_search(self, query: str, max_results: int = 8) -> str:
        from .web import web_search

        hits = web_search(query, max_results=int(max_results))
        return "\n\n".join(f"{i}. {h['title']}\n   {h['url']}\n   {h['snippet']}" for i, h in enumerate(hits, 1)) \
            or "(no results)"

    def t_fetch_url(self, url: str, max_chars: int = MAX_OUTPUT) -> str:
        from .web import fetch_text

        return fetch_text(url, max_chars=int(max_chars))


def _spec(name: str, description: str, props: dict, required: list[str]) -> dict:
    return {"type": "function", "function": {"name": name, "description": description,
                                             "parameters": {"type": "object", "properties": props,
                                                            "required": required}}}


_S, _I, _B = {"type": "string"}, {"type": "integer"}, {"type": "boolean"}
TOOL_SPECS = {
    "list_dir": _spec("list_dir", "List a folder of the workspace.", {"path": _S}, []),
    "read_file": _spec("read_file", "Read a text file with line numbers (offset is 1-based).",
                       {"path": _S, "offset": _I, "limit": _I}, ["path"]),
    "write_file": _spec("write_file", "Create or overwrite a file with the full content.",
                        {"path": _S, "content": _S}, ["path", "content"]),
    "edit_file": _spec("edit_file", "Replace an exact, unique piece of text in a file. Read the file first.",
                       {"path": _S, "old": _S, "new": _S, "replace_all": _B}, ["path", "old", "new"]),
    "glob": _spec("glob", "Find files by name pattern, e.g. '**/*.py' or 'test_*.py'.",
                  {"pattern": _S, "path": _S}, ["pattern"]),
    "search": _spec("search", "Search file contents with a regular expression; returns file:line: text.",
                    {"pattern": _S, "path": _S, "file_glob": _S}, ["pattern"]),
    "web_search": _spec("web_search", "Search the internet; returns titles, URLs and snippets.",
                        {"query": _S, "max_results": _I}, ["query"]),
    "fetch_url": _spec("fetch_url", "Download a web page or raw file and return its readable text.",
                       {"url": _S, "max_chars": _I}, ["url"]),
    "run": _spec("run", "Run a shell command in the workspace (tests, linters, git, python scripts). Set "
                 "background=true for servers or long jobs, then use job_output.",
                 {"command": _S, "timeout": _I, "background": _B}, ["command"]),
    "job_output": _spec("job_output", "State and last lines of output of a background job.",
                        {"job_id": _S, "tail": _I}, ["job_id"]),
    "job_stop": _spec("job_stop", "Stop a background job.", {"job_id": _S}, ["job_id"]),
    "show_file": _spec("show_file", "Show a file of the workspace to the user in the chat (image, HTML, PDF, text): "
                       "figures, reports, screenshots.", {"path": _S, "caption": _S}, ["path"]),
    "github": _spec("github", "Call the GitHub REST API: path like 'repos/OWNER/REPO/pulls', method GET/POST/PATCH/"
                    "PUT/DELETE, optional JSON body. Writes need the user's approval.",
                    {"path": _S, "method": _S, "body": {"type": "object"}}, ["path"]),
}
