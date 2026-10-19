"""PINNeAPPle Agent Workbench: tools, approvals, the agent loop with delegation, the database and the training
export, the web app, the built-in eval and the web helpers. No model is needed: a scripted backend plays the model."""
import json
import threading
import time
from pathlib import Path

import pytest

from pinneapple_llm.workbench import (
    ApprovalPolicy,
    Reply,
    ScriptedBackend,
    Session,
    Store,
    ToolCall,
    Workspace,
)
from pinneapple_llm.workbench.agents import load_agents, parse_agent
from pinneapple_llm.workbench.backend import OllamaBackend, parse_json_tool_call
from pinneapple_llm.workbench.finetune import TrainConfig, export_dataset, mlx_train_command, redact
from pinneapple_llm.workbench.models import recommend
from pinneapple_llm.workbench.tools import ToolError
from pinneapple_llm.workbench.web import html_to_text, parse_duckduckgo


def call(name, **args):
    return ToolCall(name=name, arguments=args)


# -- tools and approvals -------------------------------------------------------------------------------------------
def test_workspace_confines_paths_and_edits_exactly(tmp_path):
    ws = Workspace(tmp_path, ApprovalPolicy("yolo"))
    ws.call("write_file", {"path": "pkg/a.py", "content": "x = 1\ny = 2\nx = 1\n"})
    assert "x = 1" in ws.call("read_file", {"path": "pkg/a.py"})
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.call("read_file", {"path": "../secret.txt"})
    (tmp_path / "link").symlink_to("/etc")
    with pytest.raises(ToolError, match="outside the workspace"):
        ws.call("list_dir", {"path": "link"})
    with pytest.raises(ToolError, match="occurs 2 times"):
        ws.call("edit_file", {"path": "pkg/a.py", "old": "x = 1", "new": "x = 3"})
    ws.call("edit_file", {"path": "pkg/a.py", "old": "y = 2", "new": "y = 5"})
    assert (tmp_path / "pkg/a.py").read_text() == "x = 1\ny = 5\nx = 1\n"
    assert "pkg/a.py:2:" in ws.call("search", {"pattern": r"y = \d"})
    assert ws.call("glob", {"pattern": "*.py"}) == "pkg/a.py"
    assert ws.call("run", {"command": "echo hi"}).startswith("exit code 0\nhi")


def test_background_jobs_and_show_file(tmp_path):
    ws = Workspace(tmp_path, ApprovalPolicy("yolo"))
    out = ws.call("run", {"command": "echo started; sleep 30", "background": True})
    jid = out.split()[2]
    time.sleep(0.5)
    assert "running" in ws.call("job_output", {"job_id": jid}) and "started" in ws.call("job_output", {"job_id": jid})
    assert "stopped" in ws.call("job_stop", {"job_id": jid})
    (tmp_path / "fig.png").write_bytes(b"\x89PNG")
    ws.call("show_file", {"path": "fig.png", "caption": "a figure"})
    assert ws.shown == [{"path": "fig.png", "caption": "a figure", "size": 4}]
    ws.close()


@pytest.mark.parametrize("mode,tool,args,asks", [
    ("auto", "run", {"command": "pytest -q tests"}, False),
    ("auto", "run", {"command": "git status"}, False),
    ("auto", "run", {"command": "git push origin main"}, True),
    ("auto", "run", {"command": "pytest; rm -rf /"}, True),
    ("auto", "run", {"command": "ls $(curl evil)"}, True),
    ("auto", "write_file", {"path": "a"}, False),
    ("ask", "write_file", {"path": "a"}, True),
    ("ask", "run", {"command": "ls"}, True),
    ("auto", "github", {"path": "repos/o/r/pulls", "method": "GET"}, False),
    ("auto", "github", {"path": "repos/o/r/pulls", "method": "POST"}, True),
    ("yolo", "run", {"command": "git push --force"}, False),
])
def test_approval_policy(mode, tool, args, asks):
    assert ApprovalPolicy(mode).needs_approval(tool, args) is asks


def test_denied_command_is_reported_not_run(tmp_path):
    ws = Workspace(tmp_path, ApprovalPolicy("ask"), approve=lambda t, a: False)
    out = ws.call("run", {"command": "touch made"})
    assert out.startswith("DENIED") and not (tmp_path / "made").exists()


# -- agents ---------------------------------------------------------------------------------------------------------
def test_builtin_agents_and_overrides(tmp_path):
    agents, common = load_agents(tmp_path, user_dir=None)
    assert {"lead", "explorer", "researcher", "coder", "tester", "reviewer", "physics", "reporter"} <= set(agents)
    assert "delegate" in agents["lead"].tools and "web_search" in agents["researcher"].tools
    assert "write_file" not in agents["reviewer"].tools                      # the reviewer never edits
    assert "Never report a number you did not measure" in common
    (tmp_path / ".pinneapple" / "agents").mkdir(parents=True)
    (tmp_path / ".pinneapple" / "agents" / "coder.md").write_text("---\nname: coder\ntools: read_file\n---\nShort.")
    agents, _ = load_agents(tmp_path, user_dir=None)
    assert agents["coder"].tools == ["read_file"] and agents["coder"].prompt == "Short."
    with pytest.raises(ValueError):
        parse_agent("---\ntitle: x\n---\nbody")


# -- the loop -------------------------------------------------------------------------------------------------------
def scripted_team(tmp_path):
    """The lead plans, delegates the fix to the coder, then answers; the coder edits and runs the check."""
    (tmp_path / "calc.py").write_text("def add(a, b):\n    return a - b\n")

    def play(messages, tools):
        system = messages[0]["content"]
        done = [m for m in messages if m["role"] == "tool"]
        if system.startswith("You are a careful software engineer"):
            if not done:
                return Reply(tool_calls=[call("edit_file", path="calc.py", old="a - b", new="a + b")])
            if len(done) == 1:
                return Reply(tool_calls=[call("run", command="python -c 'from calc import add; print(add(2, 3))'")])
            return Reply(content=f"Fixed add; check printed {done[-1]['content'].split()[-1]}.")
        if not done:
            return Reply(content="Planning.", tool_calls=[call("todo", items=[
                {"text": "fix add", "status": "in_progress"}])])
        if len(done) == 1:
            return Reply(tool_calls=[call("delegate", agent="coder", task="Fix add in calc.py and check it.")])
        return Reply(content="Done: " + done[-1]["content"].splitlines()[-1], usage={"prompt_tokens": 10})

    return ScriptedBackend(play)


def test_lead_delegates_and_everything_is_recorded(tmp_path):
    events, calls = [], []
    sess = Session(scripted_team(tmp_path), str(tmp_path), policy=ApprovalPolicy("yolo"),
                   on_event=events.append, on_model_call=calls.append)
    out = sess.send("add() is wrong, fix it")
    assert (tmp_path / "calc.py").read_text().endswith("a + b\n")
    assert out == "Done: Fixed add; check printed 5."
    kinds = [e["type"] for e in events]
    assert kinds[0] == "user" and kinds[-1] == "done"
    assert {"plan", "delegate_start", "delegate_end", "tool_call", "tool_result", "message"} <= set(kinds)
    assert [e for e in events if e["type"] == "delegate_start"][0]["depth"] == 1
    assert sess.plan == [{"text": "fix add", "status": "in_progress"}]
    assert {c["agent"] for c in calls} == {"lead", "coder"} and len(calls) == 6
    assert all(c["turn_id"] == sess.turn_id for c in calls)
    assert calls[-1]["reply"]["content"].startswith("Done") and sess.usage["prompt_tokens"] == 10
    assert len({c["run_id"] for c in calls}) == 2                            # one run for the lead, one for the coder


def test_approval_blocks_until_answered(tmp_path):
    events = []
    script = [Reply(tool_calls=[call("run", command="touch made_by_agent")]), Reply(content="ok")]
    sess = Session(ScriptedBackend(script), str(tmp_path), policy=ApprovalPolicy("ask"), on_event=events.append)
    th = threading.Thread(target=sess.send, args=("make a file",))
    th.start()
    for _ in range(100):
        req = [e for e in events if e["type"] == "approval_request"]
        if req:
            break
        time.sleep(0.02)
    assert req and req[0]["args"]["command"] == "touch made_by_agent"
    assert not (tmp_path / "made_by_agent").exists()
    assert sess.answer(req[0]["id"], True)
    th.join(5)
    assert (tmp_path / "made_by_agent").exists()


def test_unknown_tool_and_cancel(tmp_path):
    events = []
    script = [Reply(tool_calls=[call("delete_everything")]), Reply(content="sorry")]
    sess = Session(ScriptedBackend(script), str(tmp_path), on_event=events.append)
    assert sess.send("go") == "sorry"
    res = [e for e in events if e["type"] == "tool_result"][0]
    assert not res["ok"] and "not available" in res["result"]

    sess2 = Session(ScriptedBackend([Reply(tool_calls=[call("ask_user", question="which?", options=["a", "b"])])]),
                    str(tmp_path), on_event=events.append)
    th = threading.Thread(target=sess2.send, args=("ask me",))
    th.start()
    time.sleep(0.3)
    sess2.cancel()
    th.join(5)
    assert not th.is_alive() and events[-1]["type"] == "done"


def test_history_is_compacted(tmp_path):
    from pinneapple_llm.workbench.runtime import SessionSettings

    steps = []

    def play(messages, tools):
        if not tools:
            return Reply(content="SUMMARY of the work")
        steps.append(1)
        return Reply(tool_calls=[call("list_dir")]) if len(steps) < 20 else Reply(content="end")

    sess = Session(ScriptedBackend(play), str(tmp_path), settings=SessionSettings(context_chars=400, keep_recent=4))
    assert sess.send("x" * 300) == "end"
    assert any("SUMMARY of the work" in (m.get("content") or "") for m in sess.history)


# -- backends -------------------------------------------------------------------------------------------------------
def test_json_tool_calls_from_text():
    names = ["read_file", "run"]
    c = parse_json_tool_call('Sure.\n```json\n{"tool": "run", "arguments": {"command": "ls {x}"}}\n```', names)
    assert c.name == "run" and c.arguments == {"command": "ls {x}"}
    assert parse_json_tool_call('{"tool": "nope", "arguments": {}}', names) is None
    assert parse_json_tool_call("just text", names) is None


class _Resp:
    def __init__(self, code, data):
        self.status_code, self._data, self.text = code, data, json.dumps(data)

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


def test_ollama_native_tools_and_fallback(monkeypatch):
    import requests

    sent = []

    def post(url, json=None, timeout=None):
        sent.append(json)
        if len(sent) == 1:
            return _Resp(200, {"message": {"content": "", "tool_calls": [
                {"function": {"name": "run", "arguments": {"command": "ls"}}}]}, "prompt_eval_count": 7,
                "eval_count": 3})
        if "tools" in json:
            return _Resp(400, {"error": "model does not support tools"})
        return _Resp(200, {"message": {"content": '{"tool": "run", "arguments": {"command": "pwd"}}'}})

    monkeypatch.setattr(requests, "post", post)
    from pinneapple_llm.workbench.tools import TOOL_SPECS

    be = OllamaBackend("m")
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    r = be.chat(msgs, [TOOL_SPECS["run"]])
    assert r.tool_calls[0].arguments == {"command": "ls"} and r.usage == {"prompt_tokens": 7, "completion_tokens": 3}
    r = be.chat(msgs, [TOOL_SPECS["run"]])                       # 400 -> JSON mode from now on
    assert be.tool_mode == "json" and r.tool_calls[0].arguments == {"command": "pwd"}
    assert "Tools:" in sent[-1]["messages"][0]["content"]


def test_recommendations_by_memory():
    assert recommend(128).model == "gpt-oss:120b"
    assert recommend(36).model == "qwen3-coder:30b"
    assert recommend(16).model == "qwen3:8b"
    assert recommend(8).model == "qwen3:4b"


# -- database and fine-tuning data -----------------------------------------------------------------------------------
def test_store_feedback_and_dataset_export(tmp_path):
    store = Store(tmp_path / "wb.sqlite")
    calls = []
    sid = store.create(str(tmp_path), {"policy": "auto"})
    for _ in range(2):
        sess = Session(scripted_team(tmp_path), str(tmp_path), policy=ApprovalPolicy("yolo"),
                       on_model_call=lambda c: store.add_model_call(sid, c))
        sess.send("fix it, token ghp_" + "a" * 30)
        calls.append(sess.turn_id)
    store.set_feedback(sid, calls[0], +1)
    store.set_feedback(sid, calls[1], -1, "wrong")
    assert store.stats() == {"engine": "sqlite", "sessions": 1, "model_calls": 12, "rated_turns": 2, "good_turns": 1}

    info = export_dataset(store, tmp_path / "ds", val_fraction=0.5)
    assert info["dropped_by_rating"] == 6 and info["train"] + info["valid"] == 2      # lead + coder runs of turn 1
    rows = [json.loads(line) for f in ("train.jsonl", "valid.jsonl") for line in (tmp_path / "ds" / f).open()]
    for ex in rows:
        assert ex["messages"][-1]["role"] == "assistant" and not ex["messages"][-1].get("tool_calls")
        assert ex["tools"] and all(m["role"] in ("system", "user", "assistant", "tool") for m in ex["messages"])
        assert "ghp_" not in json.dumps(ex)                                           # secrets masked
    lead = [ex for ex in rows if "lead engineer" in ex["messages"][0]["content"]][0]
    assert any(m.get("tool_calls") for m in lead["messages"]) and any(m["role"] == "tool" for m in lead["messages"])
    assert export_dataset(store, tmp_path / "ds2", mode="calls")["train"] >= 5


def test_redact():
    assert redact("key sk-" + "x" * 30) == "key <redacted>"
    assert redact("password = hunter2hunter") == "password = <redacted>"
    assert redact("nothing here") == "nothing here"


def test_mlx_training_command():
    cmd = mlx_train_command("data", TrainConfig(model="mlx-community/Qwen3-4B-4bit", iters=10))
    assert cmd[1:3] == ["-m", "mlx_lm.lora"] and "--train" in cmd and "--mask-prompt" in cmd
    assert cmd[cmd.index("--model") + 1] == "mlx-community/Qwen3-4B-4bit" and cmd[cmd.index("--iters") + 1] == "10"


# -- the web app ----------------------------------------------------------------------------------------------------
def test_web_app_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    import pinneapple_llm.workbench.server as server

    monkeypatch.setattr(server, "USER_DIR", tmp_path / "user_agents")
    ws = tmp_path / "proj"
    ws.mkdir()
    (ws / "plot.png").write_bytes(b"\x89PNG")
    script = lambda m, t: (Reply(tool_calls=[call("show_file", path="plot.png", caption="result")])  # noqa: E731
                           if not any(x["role"] == "tool" for x in m) else Reply(content="Here it is."))
    app = server.create_app(workspace=str(ws), db=tmp_path / "wb.sqlite", token="s3cret",
                            backend_obj=ScriptedBackend(script))
    c = TestClient(app)
    assert c.get("/api/info").status_code == 401
    h = {"Authorization": "Bearer s3cret"}
    info = c.get("/api/info", headers=h).json()
    assert info["recommended"]["model"] and {a["name"] for a in info["agents"]} >= {"lead", "coder"}
    sid = c.post("/api/sessions", json={"workspace": str(ws)}, headers=h).json()["id"]
    assert c.post(f"/api/sessions/{sid}/messages", json={"text": "show me the plot"}, headers=h).json()["ok"]
    for _ in range(100):
        ev = c.get(f"/api/sessions/{sid}/events?stream=false", headers=h).json()
        if ev and ev[-1]["type"] == "done":
            break
        time.sleep(0.05)
    assert [e["type"] for e in ev if e["type"] in ("file", "message")] == ["file", "message"]
    assert c.get(f"/api/sessions/{sid}/file?path=plot.png", headers=h).content == b"\x89PNG"
    assert c.get(f"/api/sessions/{sid}/file?path=../wb.sqlite", headers=h).status_code == 404
    assert c.get("/api/sessions", headers=h).json()[0]["title"] == "show me the plot"
    turn = ev[-1]["turn_id"]
    assert c.post(f"/api/sessions/{sid}/feedback", json={"turn_id": turn, "rating": 1}, headers=h).json()["ok"]
    assert c.get("/api/training", headers=h).json()["good_turns"] == 1
    exp = c.post("/api/training/export", json={"out_dir": str(tmp_path / "ds")}, headers=h).json()
    assert exp["train"] + exp["valid"] == 1
    text = c.get("/api/agents/coder", headers=h).json()["text"]
    assert c.put("/api/agents/coder", json={"text": text.replace("careful", "very careful")},
                 headers=h).json()["ok"]
    assert "very careful" in (tmp_path / "user_agents" / "coder.md").read_text()
    assert c.put("/api/agents/coder", json={"text": "---\nname: other\n---\nx"}, headers=h).status_code == 400
    assert c.get("/", headers=h).status_code == 200 and "Agent Workbench" in c.get("/").text


def test_parallel_session_tools(tmp_path):
    pytest.importorskip("fastapi")
    import pinneapple_llm.workbench.server as server

    def play(m, t):
        if m[-1]["role"] == "user" and "side task" in m[-1]["content"]:
            return Reply(content="side done")
        if not any(x["role"] == "tool" for x in m):
            return Reply(tool_calls=[call("start_session", title="side", task="do the side task")])
        return Reply(content="started")

    app = server.create_app(workspace=str(tmp_path), db=tmp_path / "wb.sqlite", backend_obj=ScriptedBackend(play))
    mgr = app.state.manager
    sid = mgr.store.create(str(tmp_path), {"policy": "auto"})
    mgr.send(sid, "split the work")
    mgr.busy[sid].join(5)
    others = [s for s in mgr.store.list() if s["id"] != sid]
    assert len(others) == 1 and others[0]["title"] == "side"
    mgr.busy[others[0]["id"]].join(5)
    assert "side done" in mgr._hooks(sid)["session_status"](others[0]["id"])


# -- eval and web helpers -------------------------------------------------------------------------------------------
def test_eval_checks_the_outcome():
    from pinneapple_llm.workbench.evaluate import run_eval

    def solver(m, t):
        done = [x for x in m if x["role"] == "tool"]
        if not done:
            return Reply(tool_calls=[call("edit_file", path="stats.py", old="/ len(xs) - 1", new="/ (len(xs) - 1)")])
        if len(done) == 1:
            return Reply(tool_calls=[call("run", command="python -m pytest -q")])
        return Reply(content="fixed")

    passed = run_eval(ScriptedBackend(solver), ["fix_failing_test"])[0]
    assert passed.passed and passed.model_calls == 3
    lazy = run_eval(ScriptedBackend(lambda m, t: Reply(content="looks fine")), ["fix_failing_test", "locate_code"])
    assert [r.passed for r in lazy] == [False, False]


def test_web_helpers():
    page = ('<a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.org%2Fdoc">'
            'Example <b>Doc</b></a><a class="result__snippet" href="x">The <b>snippet</b> text</a>'
            '<a class="result__a" href="https://second.org">Second</a>')
    hits = parse_duckduckgo(page)
    assert hits[0] == {"title": "Example Doc", "url": "https://example.org/doc", "snippet": "The snippet text"}
    assert hits[1]["url"] == "https://second.org"
    assert html_to_text("<html><head><title>x</title></head><body><script>bad()</script><p>Hello</p>"
                        "<p>World &amp; more</p></body></html>") == "Hello\n\nWorld & more"


def test_cli_dry_run_and_vscode(tmp_path, monkeypatch, capsys):
    from pinneapple_llm.workbench.__main__ import main

    assert main(["finetune", "train", "--data", str(tmp_path), "--dry-run", "--base", "mlx-community/Qwen3-4B-4bit"]) == 0
    assert "mlx_lm.lora" in capsys.readouterr().out
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    assert main(["vscode"]) == 0
    ext = tmp_path / ".vscode" / "extensions" / "pinneapple.agent-workbench-0.1.0"
    pkg = json.loads((ext / "package.json").read_text())
    assert pkg["main"] == "./extension.js" and (ext / "extension.js").is_file()
