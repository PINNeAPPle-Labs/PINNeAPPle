"""``python -m pinneapple_llm.workbench <command>`` (also ``pinneapple-workbench``).

  serve      start the web app (default command): the agents work in --workspace
  doctor     check the machine, the model server and the model, and print what to fix
  pull       download the recommended model (or --model) with Ollama
  run        one task in the terminal, no browser: run "fix the failing test"
  eval       measure a model on the built-in tasks
  finetune   export the dataset from the database, train a LoRA adapter, serve it
  vscode     install the VS Code extension (opens the workbench beside your editor)
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import webbrowser
from pathlib import Path


def _backend(args):
    from .backend import make_backend
    from .models import recommend

    rec = recommend()
    kw = {"num_ctx": getattr(args, "num_ctx", None) or rec.context}
    if args.backend == "ollama" and args.host:
        kw["host"] = args.host
    if args.base_url:
        kw["base_url"] = args.base_url
    return make_backend(args.backend, args.model or rec.model, **kw)


def _common(p):
    p.add_argument("--backend", default="ollama", help="ollama, or openai for any OpenAI-compatible server "
                   "(LM Studio, llama.cpp, mlx_lm.server, vLLM)")
    p.add_argument("--model", default="", help="model name; default: the recommendation for this machine")
    p.add_argument("--host", default="", help="Ollama URL (default http://127.0.0.1:11434)")
    p.add_argument("--base-url", default="", help="OpenAI-compatible URL, e.g. http://127.0.0.1:8080/v1")
    p.add_argument("--num-ctx", type=int, default=0, help="context window to request from Ollama")


def cmd_serve(args):
    import uvicorn

    from .server import create_app

    app = create_app(backend=args.backend, model=args.model, host=args.host, base_url=args.base_url,
                     workspace=args.workspace, db=args.db, token=args.token, language=args.language)
    url = f"http://{args.bind}:{args.port}/" + (f"?token={args.token}" if args.token else "")
    print(f"PINNeAPPle Agent Workbench on {url}  (workspace {Path(args.workspace).resolve()}, database {args.db})")
    if not args.no_browser:
        webbrowser.open(url)
    uvicorn.run(app, host=args.bind, port=args.port, log_level="warning")


def cmd_doctor(args):
    from .models import detect_machine, recommend

    m, rec, ok = detect_machine(), recommend(), True
    print(f"machine   {m.system} {m.arch}, {m.memory_gb} GB{' (Apple Silicon: Metal GPU)' if m.apple_silicon else ''}")
    print(f"advice    {rec.model} (~{rec.approx_size_gb} GB, context {rec.context}): {rec.note}")
    if args.backend == "ollama":
        if shutil.which("ollama") is None:
            ok = False
            print("ollama    NOT FOUND: install it from https://ollama.com (Mac: the app, or `brew install ollama`)")
        else:
            print(f"ollama    {shutil.which('ollama')}")
    be = _backend(args)
    have = be.models()
    if not have:
        ok = False
        print("server    not reachable: start it with `ollama serve` (or open the Ollama app)"
              if args.backend == "ollama" else f"server    not reachable at {args.base_url}")
    else:
        print(f"server    up, {len(have)} models")
        if be.model not in have and f"{be.model}:latest" not in have:
            ok = False
            print(f"model     {be.model} not installed: python -m pinneapple_llm.workbench pull --model {be.model}")
        else:
            print(f"model     {be.model} installed")
    for mod, why in (("fastapi", "the web app"), ("uvicorn", "the web app"), ("requests", "the model server"),
                     ("psycopg", "only for a PostgreSQL database"), ("mlx_lm", "only for fine-tuning on a Mac")):
        try:
            __import__(mod)
            print(f"python    {mod}: ok")
        except ImportError:
            print(f"python    {mod}: missing ({why})")
            ok = ok and mod in ("psycopg", "mlx_lm")
    print("\nready: python -m pinneapple_llm.workbench serve --workspace <your project>" if ok else
          "\nfix the items above, then run doctor again")
    return 0 if ok else 1


def cmd_pull(args):
    from .models import recommend

    model = args.model or recommend().model
    if shutil.which("ollama") is None:
        sys.exit("ollama not found: install it from https://ollama.com")
    print(f"ollama pull {model}")
    return subprocess.call(["ollama", "pull", model])


def _print_event(e):
    pad = "  " * (e.get("depth") or 0)
    t = e["type"]
    if t == "message":
        print(f"\n{pad}[{e['agent']}] {e['content']}\n")
    elif t == "tool_call":
        print(f"{pad}> {e['agent']}: {e['tool']} {json.dumps(e['args'])[:200]}")
    elif t == "tool_result" and not e["ok"]:
        print(f"{pad}  ! {e['result'][:300]}")
    elif t == "delegate_start":
        print(f"{pad}-> {e['agent']}: {e['task'][:200]}")
    elif t == "plan":
        print(pad + "plan: " + "; ".join(f"[{i['status']}] {i['text']}" for i in e["items"]))
    elif t == "error":
        print(f"{pad}ERROR {e['content']}")


def cmd_run(args):
    from .runtime import Session, SessionSettings
    from .store import Store
    from .tools import ApprovalPolicy

    store = Store(args.db)
    sid = store.create(str(Path(args.workspace).resolve()), {"policy": args.policy, "model": args.model}, args.task[:80])

    def on_event(e):
        store.add_event(sid, e)
        _print_event(e)
        if e["type"] == "approval_request":
            ans = input(f"approve {e['tool']} {json.dumps(e['args'])[:300]}? [y/N] ").strip().lower()
            sess.answer(e["id"], ans in ("y", "yes", "s", "sim"))
        elif e["type"] == "question":
            sess.answer(e["id"], input(f"{e['question']} {e.get('options') or ''}\n> "))

    sess = Session(_backend(args), args.workspace, policy=ApprovalPolicy(args.policy),
                   settings=SessionSettings(language=args.language), on_event=on_event,
                   on_model_call=lambda c: store.add_model_call(sid, c))
    sess.send(args.task)
    store.update(sid, history=sess.history)
    sess.close()
    u = sess.usage
    print(f"\n{u['model_calls']} model calls, {u['prompt_tokens']} tokens in, {u['completion_tokens']} out, "
          f"{u['seconds']} s; saved as session {sid}")


def cmd_eval(args):
    from .evaluate import as_dicts, format_results, run_eval

    be = _backend(args)
    res = run_eval(be, args.task or None, on_event=_print_event if args.verbose else None)
    print(format_results(res, be.model))
    if args.json:
        Path(args.json).write_text(json.dumps({"model": be.model, "results": as_dicts(res)}, indent=1))
    return 0


def cmd_finetune(args):
    from .finetune import (
        TrainConfig,
        export_dataset,
        mlx_train_command,
        serve_mlx_command,
        train_mlx,
        train_peft,
    )
    from .store import Store

    if args.action == "export":
        print(json.dumps(export_dataset(Store(args.db), args.data, min_rating=args.min_rating, mode=args.mode),
                         indent=1))
        return 0
    cfg = TrainConfig(adapter_path=args.adapter, iters=args.iters, learning_rate=args.lr)
    if args.base:
        cfg.model = args.base
    if args.action == "train":
        if args.engine == "peft":
            print(train_peft(args.data, cfg))
        else:
            print(" ".join(mlx_train_command(args.data, cfg)))
            if not args.dry_run:
                print(train_mlx(args.data, cfg))
        return 0
    cmd = serve_mlx_command(cfg, args.port)
    print(" ".join(cmd))
    print(f"then: python -m pinneapple_llm.workbench serve --backend openai --base-url "
          f"http://127.0.0.1:{args.port}/v1 --model {cfg.model}")
    return 0 if args.dry_run else subprocess.call(cmd)


def cmd_vscode(args):
    src = Path(__file__).with_name("vscode")
    dest = Path.home() / ".vscode" / "extensions" / "pinneapple.agent-workbench-0.1.0"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(src, dest)
    print(f"installed the VS Code extension in {dest}\nrestart VS Code, then run the command "
          f"'PINNeAPPle: Open Agent Workbench' (Cmd+Shift+P)")
    return 0


def main(argv=None):
    from .store import DEFAULT_DB

    ap = argparse.ArgumentParser(prog="pinneapple-workbench", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    p = sub.add_parser("serve", help="start the web app")
    _common(p)
    p.add_argument("--workspace", default=".", help="default project folder for new sessions")
    p.add_argument("--db", default=DEFAULT_DB, help="SQLite path or postgresql:// URL")
    p.add_argument("--port", type=int, default=8770)
    p.add_argument("--bind", default="127.0.0.1", help="keep 127.0.0.1 unless you set --token")
    p.add_argument("--token", default="", help="require this token on every request")
    p.add_argument("--language", default="", help='e.g. "Português": the lead answers in it')
    p.add_argument("--no-browser", action="store_true")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("doctor", help="check the setup")
    _common(p)
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("pull", help="download the model with Ollama")
    p.add_argument("--model", default="")
    p.set_defaults(fn=cmd_pull)

    p = sub.add_parser("run", help="one task in the terminal")
    _common(p)
    p.add_argument("task")
    p.add_argument("--workspace", default=".")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--policy", default="auto", choices=["ask", "auto", "yolo"])
    p.add_argument("--language", default="")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("eval", help="measure a model on the built-in tasks")
    _common(p)
    p.add_argument("--task", action="append", help="run only this task (repeatable)")
    p.add_argument("--json", default="", help="also write the results to this file")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("finetune", help="dataset, LoRA training and serving")
    p.add_argument("action", choices=["export", "train", "serve"])
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--data", default=str(Path.home() / ".pinneapple" / "workbench" / "dataset"))
    p.add_argument("--min-rating", type=int, default=0, help="0: drop bad turns; 1: only good turns")
    p.add_argument("--mode", default="trajectories", choices=["trajectories", "calls"])
    p.add_argument("--engine", default="mlx", choices=["mlx", "peft"])
    p.add_argument("--base", default="", help="base model (Hugging Face / MLX repo)")
    p.add_argument("--adapter", default=str(Path.home() / ".pinneapple" / "workbench" / "adapters" / "latest"))
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--dry-run", action="store_true", help="print the command only")
    p.set_defaults(fn=cmd_finetune)

    p = sub.add_parser("vscode", help="install the VS Code extension")
    p.set_defaults(fn=cmd_vscode)

    args = ap.parse_args(argv)
    if not getattr(args, "fn", None):
        args = ap.parse_args(["serve", *(argv or sys.argv[1:])])
    return args.fn(args) or 0


if __name__ == "__main__":
    sys.exit(main())
