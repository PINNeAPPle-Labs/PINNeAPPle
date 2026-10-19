"""Settings from the environment (and an optional ``.env`` file): LLM provider, database URL, folders, server.

==================================  ==========================================================================
``PINNEAPPLE_FLOW_HOME``            folder for the SQLite store and run folders (default ``~/.pinneapple_flow``)
``PINNEAPPLE_FLOW_DB_URL``          SQLAlchemy URL (``postgresql+psycopg://...``); default SQLite in HOME
``PINNEAPPLE_FLOW_DATA``            the only folder data-ingest nodes may read files from (default HOME/data)
``PINNEAPPLE_FLOW_HOST`` / ``_PORT``  server bind address (default ``127.0.0.1:8765``)
``PINNEAPPLE_FLOW_LLM_PROVIDER``    ``none`` | ``ollama`` | ``openai`` | ``openai_compatible`` | ``anthropic``
``PINNEAPPLE_FLOW_LLM_MODEL``       model name (e.g. ``llama3.1``, ``gpt-4o-mini``)
``PINNEAPPLE_FLOW_LLM_BASE_URL``    OpenAI-compatible endpoint (vLLM, LM Studio, llama.cpp, a hosted model)
``PINNEAPPLE_FLOW_LLM_API_KEY``     key for that endpoint (``OPENAI_API_KEY`` / ``ANTHROPIC_API_KEY`` also read)
``OLLAMA_HOST``                     Ollama URL (default ``http://127.0.0.1:11434``)
==================================  ==========================================================================

When the server listens on anything but localhost, the security of :mod:`appkit` applies with prefix ``FLOW``
(``FLOW_USER`` / ``FLOW_PASSWORD_HASH``, rate limits, CSRF, headers); see the app README.
"""
from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path

from pinneapple_core.workflows import StepFailed


def load_dotenv(path: str | Path = ".env", *, override: bool = False) -> dict[str, str]:
    """Read ``KEY=value`` lines (``#`` comments, optional quotes) into ``os.environ``; returns what was read."""
    p = Path(path)
    out: dict[str, str] = {}
    if not p.is_file():
        return out
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip().removeprefix("export ").strip(), v.strip().strip("'\"")
        if not k.replace("_", "").isalnum():
            continue
        out[k] = v
        if override or k not in os.environ:
            os.environ[k] = v
    return out


def home() -> Path:
    return Path(os.environ.get("PINNEAPPLE_FLOW_HOME", Path.home() / ".pinneapple_flow"))


def data_root() -> Path:
    return Path(os.environ.get("PINNEAPPLE_FLOW_DATA", home() / "data")).resolve()


def safe_data_path(name: str) -> Path:
    """A path inside the data folder (no absolute paths elsewhere, no ``..`` escapes)."""
    if not name:
        raise StepFailed("no file given")
    root = data_root()
    p = (root / name).resolve()
    if root != p and root not in p.parents:
        raise StepFailed(f"files are read from the data folder {root} only (set PINNEAPPLE_FLOW_DATA)")
    if not p.exists():
        raise StepFailed(f"{name} not found in the data folder {root}")
    return p


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    db_url: str = ""
    llm_provider: str = "none"
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""

    @classmethod
    def from_env(cls) -> Settings:
        e = os.environ.get
        provider = e("PINNEAPPLE_FLOW_LLM_PROVIDER", "none").strip().lower() or "none"
        key = e("PINNEAPPLE_FLOW_LLM_API_KEY", "")
        if not key and provider == "anthropic":
            key = e("ANTHROPIC_API_KEY", "")
        if not key and provider in ("openai", "openai_compatible"):
            key = e("OPENAI_API_KEY", "")
        return cls(host=e("PINNEAPPLE_FLOW_HOST", "127.0.0.1"), port=int(e("PINNEAPPLE_FLOW_PORT", "8765")),
                   db_url=e("PINNEAPPLE_FLOW_DB_URL", ""), llm_provider=provider, llm_model=e(
                       "PINNEAPPLE_FLOW_LLM_MODEL", ""), llm_base_url=e("PINNEAPPLE_FLOW_LLM_BASE_URL", ""),
                   llm_api_key=key)

    def public(self) -> dict:
        """Settings safe to show in a GUI (the key is masked, the DB password redacted)."""
        from .store import redact
        d = asdict(self)
        d["llm_api_key"] = "set" if self.llm_api_key else ""
        d["db_url"] = redact(self.db_url) if self.db_url else "sqlite (default)"
        return d
