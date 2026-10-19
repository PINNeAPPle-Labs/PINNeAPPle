"""The agent loop: the lead talks to you, calls tools, keeps a plan and delegates parts of the work to the specialist
agents, each running its own loop with its own prompt and tools.

Every step is an event (``user``, ``message``, ``tool_call``, ``tool_result``, ``plan``, ``file``,
``approval_request``, ``question``, ``delegate_start`` / ``delegate_end``, ``error``, ``done``) passed to ``on_event``
and stored, so the UI shows the work live and can replay it. Approvals and questions block the loop until
:meth:`Session.answer` is called. Every model call goes to ``on_model_call`` with its exact input and reply: the
fine-tuning data.

Besides the workspace tools, the runtime provides ``todo`` (the plan), ``delegate`` (a specialist agent),
``ask_user`` (a question, optionally with choices) and, when the server wires them through ``hooks``,
``schedule`` (a message back into this session later), ``start_session`` / ``send_to_session`` /
``list_sessions`` / ``session_status`` (other sessions working in parallel).
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from .agents import Agent, load_agents
from .backend import Backend, Reply
from .tools import TOOL_SPECS, ApprovalPolicy, ToolError, Workspace

__all__ = ["Session", "SessionSettings", "Cancelled"]

_RUNTIME_SPECS = {
    "todo": {"type": "function", "function": {
        "name": "todo", "description": "Write or update your plan: the full list of items with their status.",
        "parameters": {"type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {"text": {"type": "string"},
                                             "status": {"type": "string", "enum": ["pending", "in_progress",
                                                                                   "done"]}},
            "required": ["text", "status"]}}}, "required": ["items"]}}},
    "delegate": {"type": "function", "function": {
        "name": "delegate", "description": "Hand a self-contained task to a specialist agent and get its report.",
        "parameters": {"type": "object", "properties": {"agent": {"type": "string"}, "task": {"type": "string"}},
                       "required": ["agent", "task"]}}},
    "ask_user": {"type": "function", "function": {
        "name": "ask_user", "description": "Ask the user one precise question only they can answer, and wait. "
                                           "Offer choices when there are a few clear options.",
        "parameters": {"type": "object", "properties": {"question": {"type": "string"},
                                                        "options": {"type": "array", "items": {"type": "string"}}},
                       "required": ["question"]}}},
    "schedule": {"type": "function", "function": {
        "name": "schedule", "description": "Send a message back into this session after some minutes, e.g. to "
                                           "check on a long job or a CI run.",
        "parameters": {"type": "object", "properties": {"minutes": {"type": "number"}, "message": {"type": "string"}},
                       "required": ["minutes", "message"]}}},
    "start_session": {"type": "function", "function": {
        "name": "start_session", "description": "Start another session (its own lead and agents) on a task that can "
                                                "run in parallel; returns its id.",
        "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "task": {"type": "string"},
                                                        "workspace": {"type": "string"}},
                       "required": ["title", "task"]}}},
    "send_to_session": {"type": "function", "function": {
        "name": "send_to_session", "description": "Send a message to another session (steer it, add work).",
        "parameters": {"type": "object", "properties": {"session_id": {"type": "string"},
                                                        "message": {"type": "string"}},
                       "required": ["session_id", "message"]}}},
    "list_sessions": {"type": "function", "function": {
        "name": "list_sessions", "description": "List the sessions with their state.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    "session_status": {"type": "function", "function": {
        "name": "session_status", "description": "Plan, state and the last messages of another session.",
        "parameters": {"type": "object", "properties": {"session_id": {"type": "string"}},
                       "required": ["session_id"]}}},
}
_HOOK_TOOLS = ("schedule", "start_session", "send_to_session", "list_sessions", "session_status")


class Cancelled(Exception):
    pass


@dataclass
class SessionSettings:
    max_steps: int = 40                 # model calls per agent per turn
    max_depth: int = 2                  # lead -> specialist -> (specialist)
    context_chars: int = 90_000         # history size that triggers a compaction (about 4 characters per token)
    keep_recent: int = 12               # messages kept verbatim when compacting
    language: str = ""                  # e.g. "Português": the lead answers in it


@dataclass
class _Pending:
    kind: str
    event: threading.Event = field(default_factory=threading.Event)
    answer: object = None


class Session:
    def __init__(self, backend: Backend, workspace: str, *, policy: ApprovalPolicy | None = None,
                 settings: SessionSettings | None = None, on_event: Callable[[dict], None] | None = None,
                 agents: dict[str, Agent] | None = None, common: str | None = None,
                 history: list[dict] | None = None, on_model_call: Callable[[dict], None] | None = None,
                 hooks: dict[str, Callable[..., str]] | None = None, web: bool = True):
        self.backend, self.settings = backend, settings or SessionSettings()
        self.on_event = on_event or (lambda e: None)
        loaded, loaded_common = load_agents(workspace)
        self.agents = agents if agents is not None else loaded
        self.common = loaded_common if common is None else common
        self.workspace = Workspace(workspace, policy or ApprovalPolicy(), approve=self._approve, web=web)
        self.on_model_call = on_model_call or (lambda c: None)
        self.hooks = hooks or {}
        self.turn_id = ""
        self.history: list[dict] = history or []
        self.plan: list[dict] = []
        self._pending: dict[str, _Pending] = {}
        self._cancel = threading.Event()
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "model_calls": 0, "seconds": 0.0}

    # -- control from the UI -------------------------------------------------------------------------------------
    def cancel(self) -> None:
        self._cancel.set()
        for p in self._pending.values():
            p.answer = None
            p.event.set()

    def answer(self, request_id: str, value) -> bool:
        p = self._pending.get(request_id)
        if not p:
            return False
        p.answer = value
        p.event.set()
        return True

    def _wait(self, kind: str, payload: dict):
        rid = uuid.uuid4().hex[:10]
        p = self._pending[rid] = _Pending(kind)
        self._emit({"type": kind, "id": rid, **payload})
        while not p.event.wait(0.5):
            if self._cancel.is_set():
                break
        self._pending.pop(rid, None)
        if self._cancel.is_set():
            raise Cancelled()
        return p.answer

    def close(self) -> None:
        self.cancel()
        self.workspace.close()

    def _approve(self, tool: str, args: dict) -> bool:
        return bool(self._wait("approval_request", {"tool": tool, "args": args}))

    def _emit(self, event: dict) -> None:
        self.on_event(event)

    # -- the loop ------------------------------------------------------------------------------------------------
    def send(self, text: str) -> str:
        """Run one user turn with the lead; returns its final reply."""
        self._cancel.clear()
        self.turn_id = uuid.uuid4().hex[:12]
        if not self.history:
            self.history.append({"role": "system", "content": self._system(self.agents["lead"], depth=0)})
        self.history.append({"role": "user", "content": text})
        self._emit({"type": "user", "content": text, "turn_id": self.turn_id})
        try:
            out = self._loop(self.agents["lead"], self.history, depth=0)
        except Cancelled:
            out = "(stopped)"
            self._emit({"type": "cancelled"})
        self._emit({"type": "done", "turn_id": self.turn_id, "usage": dict(self.usage)})
        return out

    def _system(self, agent: Agent, depth: int) -> str:
        names = ", ".join(f"{a.name} ({a.description})" for a in self.agents.values() if a.name != "lead")
        extra = [f"Workspace: {self.workspace.root}", f"Date: {time.strftime('%Y-%m-%d')}"]
        if "delegate" in agent.tools and depth < self.settings.max_depth:
            extra.append(f"Agents you can delegate to: {names}")
        if self.settings.language and agent.name in ("lead", "reporter"):
            extra.append(f"Answer the user in {self.settings.language}.")
        return agent.system_prompt(self.common) + "\n\n" + "\n".join(extra)

    def _specs(self, agent: Agent, depth: int) -> list[dict]:
        specs = []
        for t in agent.tools:
            if t == "delegate" and depth >= self.settings.max_depth:
                continue
            if t in _HOOK_TOOLS and t not in self.hooks:
                continue
            spec = TOOL_SPECS.get(t) or _RUNTIME_SPECS.get(t)
            if spec:
                specs.append(spec)
        return specs

    def _loop(self, agent: Agent, messages: list[dict], depth: int) -> str:
        specs = self._specs(agent, depth)
        names = {s["function"]["name"] for s in specs}
        run_id = uuid.uuid4().hex[:12]
        for _ in range(self.settings.max_steps):
            if self._cancel.is_set():
                raise Cancelled()
            self._compact(messages, agent)
            reply = self._chat(messages, specs, agent, depth, run_id)
            if not reply.tool_calls:
                messages.append({"role": "assistant", "content": reply.content})
                self._emit({"type": "message", "agent": agent.name, "depth": depth, "content": reply.content})
                return reply.content
            messages.append({"role": "assistant", "content": reply.content, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in reply.tool_calls]})
            if reply.content.strip():
                self._emit({"type": "thinking", "agent": agent.name, "depth": depth, "content": reply.content})
            for call in reply.tool_calls:
                self._emit({"type": "tool_call", "agent": agent.name, "depth": depth, "id": call.id,
                            "tool": call.name, "args": call.arguments})
                ok, result = self._run_tool(call.name, call.arguments, names, depth)
                self._emit({"type": "tool_result", "agent": agent.name, "depth": depth, "id": call.id,
                            "tool": call.name, "ok": ok, "result": result})
                messages.append({"role": "tool", "tool_call_id": call.id, "name": call.name, "content": result})
        note = f"(stopped after {self.settings.max_steps} steps without a final answer)"
        self._emit({"type": "message", "agent": agent.name, "depth": depth, "content": note})
        return note

    def _chat(self, messages, specs, agent, depth, run_id="") -> Reply:
        t0 = time.time()
        snapshot = [dict(m) for m in messages]
        try:
            reply = self.backend.chat(messages, specs)
        except Exception as e:  # noqa: BLE001 - surfaced to the UI, then re-raised
            self._emit({"type": "error", "agent": agent.name, "depth": depth, "content": f"{type(e).__name__}: {e}"})
            raise
        dt = time.time() - t0
        self.usage["model_calls"] += 1
        self.usage["seconds"] = round(self.usage["seconds"] + dt, 2)
        for k in ("prompt_tokens", "completion_tokens"):
            self.usage[k] += int(reply.usage.get(k) or 0)
        self.on_model_call({
            "turn_id": self.turn_id, "run_id": run_id, "agent": agent.name, "model": self.backend.model,
            "messages": snapshot, "tools": specs, "seconds": round(dt, 3),
            "prompt_tokens": reply.usage.get("prompt_tokens"), "completion_tokens": reply.usage.get("completion_tokens"),
            "reply": {"role": "assistant", "content": reply.content, "tool_calls": [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": json.dumps(c.arguments)}}
                for c in reply.tool_calls]}})
        return reply

    def _run_tool(self, name: str, args: dict, allowed: set, depth: int) -> tuple[bool, str]:
        if name not in allowed:
            return False, f"ERROR: tool {name!r} is not available to this agent. Use one of: {sorted(allowed)}"
        try:
            if name == "todo":
                self.plan = [{"text": str(i.get("text", "")), "status": i.get("status", "pending")}
                             for i in args.get("items", [])]
                self._emit({"type": "plan", "items": self.plan})
                return True, "plan updated"
            if name == "ask_user":
                ans = self._wait("question", {"question": args.get("question", ""),
                                              "options": [str(o) for o in args.get("options") or []]})
                return True, f"The user answered: {ans}"
            if name == "delegate":
                return True, self._delegate(str(args.get("agent", "")), str(args.get("task", "")), depth)
            if name in _HOOK_TOOLS:
                return True, str(self.hooks[name](**args))
            n_shown = len(self.workspace.shown)
            out = self.workspace.call(name, args)
            for f in self.workspace.shown[n_shown:]:
                self._emit({"type": "file", **f})
            return True, out
        except Cancelled:
            raise
        except (ToolError, OSError, ValueError, TypeError, KeyError) as e:
            return False, f"ERROR: {e}"

    def _delegate(self, name: str, task: str, depth: int) -> str:
        agent = self.agents.get(name)
        if agent is None or name == "lead":
            return f"ERROR: no agent named {name!r}. Available: {sorted(a for a in self.agents if a != 'lead')}"
        self._emit({"type": "delegate_start", "agent": name, "depth": depth + 1, "task": task})
        msgs = [{"role": "system", "content": self._system(agent, depth + 1)}, {"role": "user", "content": task}]
        out = self._loop(agent, msgs, depth + 1)
        self._emit({"type": "delegate_end", "agent": name, "depth": depth + 1, "content": out})
        return f"Report from {name}:\n{out}"

    def _compact(self, messages: list[dict], agent: Agent) -> None:
        """Summarise the older part of a long history so it fits the model's context."""
        size = sum(len(m.get("content") or "") + len(json.dumps(m.get("tool_calls") or "")) for m in messages)
        keep = self.settings.keep_recent
        if size <= self.settings.context_chars or len(messages) <= keep + 2:
            return
        cut = len(messages) - keep
        while cut < len(messages) and messages[cut]["role"] == "tool":      # never split a call from its result
            cut += 1
        old = messages[1:cut]
        if not old:
            return
        transcript = "\n".join(f"[{m['role']}] {(m.get('content') or '')[:2000]}"
                               + (f" {json.dumps(m['tool_calls'])[:500]}" if m.get("tool_calls") else "")
                               for m in old)
        prompt = [{"role": "system", "content": "Summarise this work log for the same agent to continue. Keep: the "
                   "user's requests and decisions, files changed, commands run and their results, open problems, "
                   "the plan. Be factual and complete; drop chatter."},
                  {"role": "user", "content": transcript}]
        summary = self.backend.chat(prompt, []).content
        messages[1:cut] = [{"role": "user", "content": f"(Summary of earlier work)\n{summary}"}]
        self._emit({"type": "compacted", "agent": agent.name, "removed": len(old)})
