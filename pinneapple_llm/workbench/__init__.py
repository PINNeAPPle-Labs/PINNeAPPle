"""PINNeAPPle Agent Workbench: a team of coding and engineering agents running on a local model.

A lead agent talks to you, plans, and hands parts of the work to specialists (explorer, researcher, coder, tester,
reviewer, physics engineer, reporter). They read and change your project, run commands, search the internet, follow
background jobs, call GitHub, start parallel sessions and show you files, with approvals for anything risky. Every
conversation and every model call is stored (SQLite or PostgreSQL) so you can rate turns and fine-tune the model on
your own work.

Start it with ``python -m pinneapple_llm.workbench serve --workspace <project>``; ``doctor`` checks the setup. See
``docs/guides/agent_workbench.md``.
"""
from .agents import Agent, load_agents
from .backend import OllamaBackend, OpenAIBackend, Reply, ScriptedBackend, ToolCall, make_backend
from .models import detect_machine, recommend
from .runtime import Session, SessionSettings
from .store import Store
from .tools import ApprovalPolicy, Workspace

__all__ = ["Agent", "load_agents", "OllamaBackend", "OpenAIBackend", "ScriptedBackend", "Reply", "ToolCall",
           "make_backend", "detect_machine", "recommend", "Session", "SessionSettings", "Store", "ApprovalPolicy",
           "Workspace"]
