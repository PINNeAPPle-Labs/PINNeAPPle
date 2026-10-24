Not every engineering codebase can be sent to a hosted coding assistant. We wanted the same workflow (read the code, plan, edit, run the tests, get it reviewed) on a model that runs on our own machine.

PINNeAPPle now has an Agent Workbench (`pinneapple_llm.workbench`). One command starts a web app; the same sessions open from VS Code or the terminal. It talks to Ollama or any OpenAI-compatible server (LM Studio, llama.cpp, vLLM, mlx_lm.server).

What is in it, counted from the code:
- 8 agents, each a Markdown prompt you can edit: a lead that plans and delegates, plus explorer, researcher, coder, tester, reviewer, physics (solvers and validation) and reporter.
- 3 approval policies. Under the default one, 29 read-only and test command prefixes run on their own; chained commands, rm, sudo, curl and similar always ask. Paths cannot leave the project folder.
- An eval command with 3 tasks checked mechanically: fix a failing test without touching the test, add a function that is correct on hidden inputs and has a test, name the right file and function.
- Every conversation is stored in SQLite or PostgreSQL. You rate turns, and the fine-tuning export drops the bad ones and masks secrets with 7 patterns (tokens, API keys, private keys, passwords).

The test suite (scripted model, no real LLM): 30 passed in my run.

What I am not posting: model scores. There was no local model in my test container. The guide says it plainly: laptop-sized models do well on focused tasks and need more steering on large refactors. Run eval on your own machine and model before trusting it.

Guide: https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/guides/agent_workbench.md

#LocalLLM #CodingAgents #Ollama #VSCode #ScientificComputing #OpenSource #PINNeAPPle

---
## Media

1. `media/agent-workbench-local/01_team.png` — The lead agent, its 7 specialists, the three front ends and the local model server.
2. `media/agent-workbench-local/02_session.png` — A session from the browser test (scripted model): plan, tool calls, a question, an approval, a delegated review.
3. `media/agent-workbench-local/03_approvals.png` — The three approval policies and what always asks under the default one.
4. `media/agent-workbench-local/04_eval.png` — The three eval tasks and their pass conditions; no scores shown.
5. `media/agent-workbench-local/05_training.png` — The Training data panel and secret masking from a run of `finetune.redact`.
6. `media/agent-workbench-local/06_try.png` — Commands to install and start, stated limits, test result, guide link.

## Sources

Repo paths relative to the root (main; PR #853, merge beec2b27). "My run" = commands run on 2026-10-24 in this container, outputs in `scratchpad/fp/work_awb/`.

- 8 agents (lead + 7 specialists), names and roles -> `pinneapple_llm/workbench/prompts/*.md` (8 files besides `_common.md`); guide table "The agents are Markdown files".
- Web app / VS Code / terminal; Ollama and OpenAI-compatible servers (LM Studio, llama.cpp, vLLM, mlx_lm.server) -> `docs/guides/agent_workbench.md` sections "Install and start", "Which model", "In the terminal", "Inside VS Code".
- 3 approval policies (ask / auto / yolo) -> `pinneapple_llm/workbench/tools.py` `ApprovalPolicy.mode` and `needs_approval`.
- 29 command prefixes allowed under auto -> `len(SAFE_COMMANDS)` = 29, `pinneapple_llm/workbench/tools.py` line 30.
- Always-ask characters/words (; & | ` $ < > rm mv sudo curl wget chmod chown dd) -> `tools.py` `_SHELL_META`, line 34.
- Paths confined to the folder, symlinks included -> `tools.py` module docstring; guide "What the agents can do".
- 3 eval tasks and pass conditions -> `pinneapple_llm/workbench/evaluate.py` `TASKS` (line 85); guide "Measure a model" table.
- SQLite default / PostgreSQL -> guide "The database".
- 7 secret patterns, masking examples on image 5 -> `pinneapple_llm/workbench/finetune.py` `_SECRETS` (line 40), `len` = 7; my run of `redact()` on synthetic strings.
- Bad (down-rated) turns dropped in export -> `finetune.py` docstring lines 5-7; guide "Fine-tune the model on your sessions".
- 30 passed -> my run `python -m pytest tests/test_llm_workbench.py` (`work_awb/pytest.txt`: "30 passed in 7.22s", with fastapi 0.162.0 installed; without fastapi: 28 passed, 2 skipped).
- Scripted model in screenshots -> `docs/guides/agent_workbench.md` line 13; screenshots `docs/assets/workbench/session.png`, `training.png`.
- Limits (laptop models, large refactors) -> guide "Which model", warning "Quality" (line 92).
