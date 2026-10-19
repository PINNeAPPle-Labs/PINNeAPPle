"""The agent definitions: one Markdown file per agent, with a small front matter (name, title, description, tools)
and the system prompt as the body. ``prompts/_common.md`` is appended to every agent.

The built-in agents live in ``pinneapple_llm/workbench/prompts/``. Your own versions override them by name: put
``<name>.md`` in ``~/.pinneapple/workbench/agents/`` (all projects) or ``<workspace>/.pinneapple/agents/`` (one
project). A new name adds a new agent; the lead can delegate to it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["Agent", "load_agents", "parse_agent", "BUILTIN_DIR", "USER_DIR"]

BUILTIN_DIR = Path(__file__).with_name("prompts")
USER_DIR = Path.home() / ".pinneapple" / "workbench" / "agents"


@dataclass
class Agent:
    name: str
    title: str
    description: str
    tools: list[str]
    prompt: str
    source: str = ""
    extra: dict = field(default_factory=dict)

    def system_prompt(self, common: str = "") -> str:
        return self.prompt.strip() + ("\n\n" + common.strip() if common.strip() else "")


def parse_agent(text: str, source: str = "") -> Agent:
    meta, body = {}, text
    if text.startswith("---"):
        _, head, body = text.split("---", 2)
        for line in head.strip().splitlines():
            if ":" in line:
                k, v = line.split(":", 1)
                meta[k.strip()] = v.strip()
    name = meta.pop("name", Path(source).stem if source else "")
    if not name:
        raise ValueError(f"agent file {source or '<text>'} has no name")
    tools = [t.strip() for t in meta.pop("tools", "").split(",") if t.strip()]
    return Agent(name=name, title=meta.pop("title", name.title()), description=meta.pop("description", ""),
                 tools=tools, prompt=body.strip(), source=source, extra=meta)


def load_agents(workspace: str | Path | None = None, user_dir: Path | None = USER_DIR) -> tuple[dict[str, Agent], str]:
    """All agents by name (built-in, then the user folder, then the workspace folder), and the common rules."""
    dirs = [BUILTIN_DIR] + ([user_dir] if user_dir else [])
    if workspace:
        dirs.append(Path(workspace) / ".pinneapple" / "agents")
    agents, common = {}, ""
    for d in dirs:
        if not d or not d.is_dir():
            continue
        for f in sorted(d.glob("*.md")):
            if f.name == "_common.md":
                common = f.read_text()
                continue
            a = parse_agent(f.read_text(), str(f))
            agents[a.name] = a
    return agents, common
