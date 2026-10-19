# Agent Workbench: coding and engineering agents on a local model

`pinneapple_llm.workbench` is a team of agents that works in your projects the way a hosted coding assistant
does, but with a model running on your own machine. A **lead** agent talks to you, reads the code, keeps a plan,
makes changes, runs the tests and hands parts of the work to specialists. Every conversation is stored in a
database so you can rate it and fine-tune the model on your own work.

It runs as a web app (or inside VS Code) started by one command. Nothing leaves your machine except internet
searches, when you allow them.

![A session: the plan, tool calls, a question, an approval, a delegated review and a figure](../assets/workbench/session.png)

*The screenshots on this page come from the browser test, driven by a scripted model rather than a real one.*

## What the agents can do

| Tool | What it does | Approval (default `auto` policy) |
|---|---|---|
| `list_dir`, `read_file`, `glob`, `search` | browse and search the project | never |
| `write_file`, `edit_file` | create files; replace an exact piece of text | `ask` policy only |
| `run` | shell commands in the project: tests, linters, git, scripts | read-only and test commands run; anything else asks |
| `run(background=true)`, `job_output`, `job_stop` | servers, trainings, long simulations | as `run` |
| `web_search`, `fetch_url` | search the internet, read pages and raw files | never (can be turned off per session) |
| `github` | GitHub REST API: pull requests, issues, CI (needs `GITHUB_TOKEN`) | writes ask |
| `show_file` | put a figure, screenshot, HTML report, PDF or video in the chat | never |
| `todo` | the plan shown beside the chat | never |
| `delegate` | hand a self-contained task to a specialist | never |
| `ask_user` | a question to you, with choices | it waits for you |
| `start_session`, `send_to_session`, `list_sessions`, `session_status` | parallel sessions working on independent parts | never |
| `schedule` | a message back into the session later (check a CI run or a long job) | never |

The paths are confined to the session's folder (symlinks included). Commands with `;`, `|`, `&&`, `$( )`, `rm`,
`sudo`, `curl` and similar always ask under `auto`. `yolo` asks for nothing: use it only in a throw-away folder.

The agents are Markdown files in `pinneapple_llm/workbench/prompts/`:

| Agent | Role |
|---|---|
| `lead` | understands the request, plans, edits, runs checks, delegates, has the change reviewed, reports |
| `explorer` | finds where things live and explains how they work (read-only) |
| `researcher` | internet and documentation search, other repositories, with sources |
| `coder` | implements a change and checks it |
| `tester` | writes and runs tests, reproduces failures |
| `reviewer` | critical review of the diff: bugs, weak tests, unmeasured numbers, misplaced logic |
| `physics` | PINNeAPPle problems, solvers and validation (`pp.solve`, `pp.compare`, `pp.trust_card`) |
| `reporter` | status reports with "with me" / "with you" checklists |

All of them follow `_common.md`: look before changing, minimal changes in the local style, run the checks, never
report an unmeasured number, never weaken a test. Edit any of them in the app (**Agents**). Your version is saved in
`~/.pinneapple/workbench/agents/`. A project can also carry its own in `<project>/.pinneapple/agents/`, and a new
file adds a new agent the lead can delegate to.

## Install and start (Mac with Apple Silicon)

```bash
brew install ollama            # or the app from https://ollama.com; it uses the Mac's GPU (Metal)
ollama serve &                 # the Ollama app does this for you
pip install "pinneapple[workbench]"

python -m pinneapple_llm.workbench doctor       # memory, Ollama, model, Python packages: what to fix
python -m pinneapple_llm.workbench pull         # downloads the model recommended for your memory
python -m pinneapple_llm.workbench serve --workspace ~/code/my-project --language Português
```

The browser opens on `http://127.0.0.1:8770`. **New session** picks the folder, the model and the approval policy.
`pinneapple-workbench` is the same command.

On Linux with an NVIDIA GPU the steps are the same; Ollama uses CUDA.

### Which model

`doctor` picks from the machine's memory (`pinneapple_llm.workbench.models.RECOMMENDATIONS`):

| Memory | Default model | Size | Notes |
|---|---|---|---|
| 96 GB or more | `gpt-oss:120b` | ~65 GB | MoE with 5B active parameters: fast for its size |
| 32–64 GB | `qwen3-coder:30b` | ~19 GB | MoE coding model with native tool calls; 3B active, fast on Apple Silicon |
| 24 GB | `gpt-oss:20b` | ~14 GB | good tool use for its size |
| 16 GB | `qwen3:8b` | ~5 GB | shorter, simpler tasks |
| less | `qwen3:4b` | ~3 GB | single-file edits and questions |

Any other model works with `--model`. Models without native tool calls fall back to JSON tool calls in the text
automatically. Other servers work through their OpenAI-compatible endpoint:

```bash
# LM Studio, llama.cpp llama-server, vLLM, or mlx_lm.server (often the fastest on a Mac)
python -m pinneapple_llm.workbench serve --backend openai --base-url http://127.0.0.1:1234/v1 --model <name>
```

!!! warning "Quality"
    No model that fits on a laptop matches the largest hosted models on long, multi-file work. Expect the
    local team to do well on focused tasks (a bug, a function with its test, a question about the code, a script) and
    to need more steering on large refactors. Measure the model you choose on your machine: `eval` (below) runs
    the same tasks for any model and reports pass/fail, time and tokens.

## Measure a model

```bash
python -m pinneapple_llm.workbench eval --model qwen3-coder:30b --json eval-qwen.json
python -m pinneapple_llm.workbench eval --model gpt-oss:20b
```

Each task builds a small project in a temporary folder and checks the outcome mechanically:

| Task | Pass condition |
|---|---|
| `fix_failing_test` | the suite passes, and the test is unchanged |
| `add_feature_with_test` | the new function is correct on hidden inputs, and has a test |
| `locate_code` | the answer names the right file and function |

## In the terminal

```bash
python -m pinneapple_llm.workbench run "the heat solver test fails since the last commit; find out why and fix it" \
    --workspace ~/code/my-project
```

Approvals and questions are asked in the terminal. The session is saved like the ones in the app.

## Inside VS Code

```bash
python -m pinneapple_llm.workbench vscode      # copies the extension into ~/.vscode/extensions
```

Restart VS Code. Then:

- **PINNeAPPle: Open Agent Workbench** (Cmd+Shift+P) opens the app beside your editor and starts the server for the
  open folder if it is not running.
- **Ask the agents about the selection** (right click on selected code) starts a session with that code, its file
  and lines.

The interpreter, port and extra `serve` arguments are in the settings, under *PINNeAPPle Agent Workbench*.

## Internet search

`web_search` uses, in order: your SearXNG instance (`PINNEAPPLE_SEARXNG_URL`, private, no key), the Brave Search API
(`BRAVE_API_KEY`), the Tavily API (`TAVILY_API_KEY`), or DuckDuckGo's HTML endpoint (no key). `fetch_url` reads a page
as text and GitHub `blob` links as raw files. The researcher agent clones repositories it needs to study into
`.research/` inside the workspace.

## The database

By default everything goes to `~/.pinneapple/workbench/sessions.sqlite`. To use a PostgreSQL server you already
run:

```bash
pip install "psycopg[binary]"
python -m pinneapple_llm.workbench serve --db postgresql://user:password@host:5432/workbench
# or: export PINNEAPPLE_WORKBENCH_DB=postgresql://...
```

| Table | Content |
|---|---|
| `sessions` | folder, settings, the lead's message history |
| `events` | everything shown in the chat, in order (replayed when you reopen a session) |
| `model_calls` | every request to the model: input messages, tool specs, reply, agent, time, tokens |
| `feedback` | your 👍 / 👎 on a turn, with an optional note |

## Fine-tune the model on your sessions

Rate turns with 👍 / 👎 as you work. **Training data** in the app shows what was recorded and exports the dataset:

![Training data panel](../assets/workbench/training.png)

From the terminal:

```bash
# 1. dataset: one example per agent run (the full trajectory with tool calls), 👎 turns left out,
#    secrets (tokens, keys, passwords) masked. --min-rating 1 keeps only 👍 turns.
python -m pinneapple_llm.workbench finetune export --data ~/wb-data

# 2. LoRA training on the Mac (pip install mlx-lm), on the same model family you use in Ollama
python -m pinneapple_llm.workbench finetune train --data ~/wb-data \
    --base mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit --iters 600

# 3. serve base + adapter, and point the workbench at it
python -m pinneapple_llm.workbench finetune serve --base mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit
python -m pinneapple_llm.workbench serve --backend openai --base-url http://127.0.0.1:8080/v1 \
    --model mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit

# 4. compare with the base model on the same tasks
python -m pinneapple_llm.workbench eval --backend openai --base-url http://127.0.0.1:8080/v1 --model <same>
```

On NVIDIA, `--engine peft` trains with TRL + PEFT instead (`pip install "pinneapple[finetune]" trl`).
`--dry-run` prints the training command without running it. Check the exact MLX repository name on
huggingface.co/mlx-community before a long download.

A few hundred good trajectories of your own kind of work help most with your conventions, tools and project
structure. They do not add general capability the base model lacks.

## From Python

```python
from pinneapple_llm.workbench import ApprovalPolicy, Session, Store, make_backend

store = Store()                                   # or Store("postgresql://...")
sid = store.create("/path/to/project", {"policy": "auto"})
session = Session(make_backend("ollama", "qwen3-coder:30b"), "/path/to/project",
                  policy=ApprovalPolicy("auto"),
                  on_event=lambda e: store.add_event(sid, e),
                  on_model_call=lambda c: store.add_model_call(sid, c))
print(session.send("Where is the boundary condition applied in the heat solver?"))
```

## Safety

- The server listens on 127.0.0.1. If you bind it elsewhere, set `--token`, and keep in mind the agents run commands
  as your user.
- Prefer `auto` or `ask` in real projects, and commit before large changes so `git diff` shows what the agents did.
- Files shown in the chat are served only from the session's folder.
