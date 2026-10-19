"""Chat-with-tools backends: Ollama, any OpenAI-compatible server (LM Studio, llama.cpp, vLLM, MLX), and a scripted
backend for tests.

Every backend takes OpenAI-style messages plus JSON-schema tool specs and returns one :class:`Reply`: the text and the
tool calls the model asked for. Models without native tool calling still work: the tools are described in the system
prompt and a JSON object in the reply is read as the call (``tool_mode="json"``).
"""
from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ToolCall", "Reply", "Backend", "OllamaBackend", "OpenAIBackend", "ScriptedBackend", "make_backend",
           "parse_json_tool_call"]


@dataclass
class ToolCall:
    name: str
    arguments: dict
    id: str = field(default_factory=lambda: "call_" + uuid.uuid4().hex[:12])


@dataclass
class Reply:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict = field(default_factory=dict)       # prompt_tokens, completion_tokens when the server reports them


class Backend:
    model: str = ""
    tool_mode: str = "native"                        # "native" or "json"

    def chat(self, messages: list[dict], tools: list[dict]) -> Reply:  # pragma: no cover - interface
        raise NotImplementedError

    def models(self) -> list[str]:
        return []


_JSON_TOOL_HELP = """
You can call tools. To call one, reply with ONLY a JSON object, no prose and no code fences:
{"tool": "<name>", "arguments": {...}}
When you are finished, reply in plain text (no JSON object). Tools:
""".strip()


def _json_tools_prompt(tools: list[dict]) -> str:
    lines = [_JSON_TOOL_HELP]
    for t in tools:
        f = t["function"]
        lines.append(f"- {f['name']}: {f['description']} Arguments (JSON schema): {json.dumps(f['parameters'])}")
    return "\n".join(lines)


def _first_json_object(text: str) -> dict | None:
    text = re.sub(r"```(?:json)?", "", text)
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            c = text[i]
            if in_str:
                esc = (c == "\\") and not esc
                if c == '"' and not esc:
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    return None


def parse_json_tool_call(text: str, tool_names: Iterable[str]) -> ToolCall | None:
    """Read ``{"tool": ..., "arguments": {...}}`` (or ``name``/``args``) from a reply; None when there is none."""
    obj = _first_json_object(text or "")
    if not isinstance(obj, dict):
        return None
    name = obj.get("tool") or obj.get("name") or obj.get("tool_name")
    args = obj.get("arguments", obj.get("args", obj.get("tool_args", {})))
    if name in set(tool_names) and isinstance(args, dict):
        return ToolCall(name=name, arguments=args)
    return None


def _with_json_tools(messages: list[dict], tools: list[dict]) -> list[dict]:
    """Messages for a model without native tools: the tool list joins the system prompt, tool results become user
    turns."""
    out, extra = [], _json_tools_prompt(tools) if tools else ""
    for m in messages:
        if m["role"] == "system":
            out.append({"role": "system", "content": m["content"] + ("\n\n" + extra if extra else "")})
        elif m["role"] == "tool":
            out.append({"role": "user", "content": f"Result of {m.get('name', 'tool')}:\n{m['content']}"})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            calls = [{"tool": c["function"]["name"], "arguments": _args(c["function"]["arguments"])}
                     for c in m["tool_calls"]]
            out.append({"role": "assistant", "content": (m.get("content") or "") + "\n" + json.dumps(calls[0])})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


def _args(a: Any) -> dict:
    if isinstance(a, dict):
        return a
    try:
        v = json.loads(a or "{}")
        return v if isinstance(v, dict) else {}
    except json.JSONDecodeError:
        return {}


def _json_reply(content: str, tools: list[dict], usage: dict) -> Reply:
    call = parse_json_tool_call(content, [t["function"]["name"] for t in tools])
    return Reply(content="" if call else content, tool_calls=[call] if call else [], usage=usage)


class OllamaBackend(Backend):
    """``/api/chat`` of a local Ollama server (``ollama serve``)."""

    def __init__(self, model: str, host: str = "http://127.0.0.1:11434", *, num_ctx: int = 32768,
                 temperature: float = 0.2, tool_mode: str = "native", timeout: float = 900):
        self.model, self.host, self.num_ctx = model, host.rstrip("/"), num_ctx
        self.temperature, self.tool_mode, self.timeout = temperature, tool_mode, timeout

    def chat(self, messages, tools):
        import requests

        native = self.tool_mode == "native" and bool(tools)
        msgs = [self._native_msg(m) for m in messages] if native or not tools else _with_json_tools(messages, tools)
        body = {"model": self.model, "messages": msgs, "stream": False,
                "options": {"num_ctx": self.num_ctx, "temperature": self.temperature}}
        if native:
            body["tools"] = tools
        try:
            r = requests.post(f"{self.host}/api/chat", json=body, timeout=self.timeout)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"no Ollama server at {self.host}: run `ollama serve` (or open the Ollama app)") from e
        if r.status_code == 400 and native and "does not support tools" in r.text:
            self.tool_mode = "json"                       # the model has no tool template: fall back for good
            return self.chat(messages, tools)
        r.raise_for_status()
        data = r.json()
        msg = data.get("message", {})
        usage = {"prompt_tokens": data.get("prompt_eval_count"), "completion_tokens": data.get("eval_count")}
        if not native:
            return _json_reply(msg.get("content", ""), tools, usage) if tools else Reply(msg.get("content", ""),
                                                                                       usage=usage)
        calls = [ToolCall(name=c["function"]["name"], arguments=_args(c["function"].get("arguments")))
                 for c in msg.get("tool_calls") or []]
        if not calls:                                     # some models still answer with a JSON call in the text
            return _json_reply(msg.get("content", ""), tools, usage)
        return Reply(content=msg.get("content", ""), tool_calls=calls, usage=usage)

    @staticmethod
    def _native_msg(m: dict) -> dict:
        out = {"role": m["role"], "content": m.get("content") or ""}
        if m.get("tool_calls"):
            out["tool_calls"] = [{"function": {"name": c["function"]["name"],
                                               "arguments": _args(c["function"]["arguments"])}}
                                 for c in m["tool_calls"]]
        if m["role"] == "tool" and m.get("name"):
            out["tool_name"] = m["name"]
        return out

    def models(self):
        import requests

        try:
            r = requests.get(f"{self.host}/api/tags", timeout=5)
            r.raise_for_status()
            return sorted(m["name"] for m in r.json().get("models", []))
        except requests.exceptions.RequestException:
            return []


class OpenAIBackend(Backend):
    """``/v1/chat/completions`` of any OpenAI-compatible server: LM Studio, llama.cpp ``llama-server``, vLLM,
    ``mlx_lm.server``, or a hosted endpoint."""

    def __init__(self, model: str, base_url: str = "http://127.0.0.1:1234/v1", api_key: str = "local", *,
                 temperature: float = 0.2, tool_mode: str = "native", timeout: float = 900):
        self.model, self.base_url, self.api_key = model, base_url.rstrip("/"), api_key
        self.temperature, self.tool_mode, self.timeout = temperature, tool_mode, timeout

    def chat(self, messages, tools):
        import requests

        native = self.tool_mode == "native" and bool(tools)
        body = {"model": self.model, "temperature": self.temperature,
                "messages": messages if native or not tools else _with_json_tools(messages, tools)}
        if native:
            body["tools"] = tools
        r = requests.post(f"{self.base_url}/chat/completions", json=body, timeout=self.timeout,
                          headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        data = r.json()
        msg = data["choices"][0]["message"]
        usage = data.get("usage") or {}
        if not native:
            return _json_reply(msg.get("content") or "", tools, usage) if tools else Reply(msg.get("content") or "",
                                                                                         usage=usage)
        calls = [ToolCall(name=c["function"]["name"], arguments=_args(c["function"].get("arguments")),
                          id=c.get("id") or "call_" + uuid.uuid4().hex[:12])
                 for c in msg.get("tool_calls") or []]
        if not calls:
            return _json_reply(msg.get("content") or "", tools, usage)
        return Reply(content=msg.get("content") or "", tool_calls=calls, usage=usage)

    def models(self):
        import requests

        try:
            r = requests.get(f"{self.base_url}/models", timeout=5, headers={"Authorization": f"Bearer {self.api_key}"})
            r.raise_for_status()
            return sorted(m["id"] for m in r.json().get("data", []))
        except requests.exceptions.RequestException:
            return []


class ScriptedBackend(Backend):
    """Replays a list of replies, or calls ``fn(messages, tools) -> Reply``: for tests and demos without a model."""

    def __init__(self, script: list[Reply] | Callable[[list[dict], list[dict]], Reply], model: str = "scripted"):
        self.script, self.model, self.calls = script, model, []

    def chat(self, messages, tools):
        self.calls.append([dict(m) for m in messages])
        if callable(self.script):
            return self.script(messages, tools)
        if not self.script:
            return Reply(content="(script finished)")
        return self.script.pop(0)

    def models(self):
        return [self.model]


def make_backend(kind: str = "ollama", model: str = "", **kw) -> Backend:
    if kind == "ollama":
        return OllamaBackend(model, **{k: v for k, v in kw.items() if k in ("host", "num_ctx", "temperature",
                                                                             "tool_mode", "timeout")})
    if kind in ("openai", "lmstudio", "llamacpp", "vllm", "mlx"):
        return OpenAIBackend(model, **{k: v for k, v in kw.items() if k in ("base_url", "api_key", "temperature",
                                                                             "tool_mode", "timeout")})
    raise ValueError(f"unknown backend {kind!r}: use 'ollama' or 'openai' (any OpenAI-compatible server)")
