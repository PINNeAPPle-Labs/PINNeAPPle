# PINNeAPPle from any coding agent: the MCP server

`pinneapple-mcp` is a [Model Context Protocol](https://modelcontextprotocol.io) server (stdio, built on the official
MCP Python SDK, 1.x or 2.x). Any MCP client, such as Gemini CLI, Claude Code, Cursor, Codex or Antigravity, can then
solve, benchmark and validate physics problems with PINNeAPPle, and every result comes back with its physics checks.
For agents that read files instead of calling tools, the repository's [`AGENTS.md`](https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/AGENTS.md)
covers the same ground as a guide.

```bash
pip install "pinneapple[mcp]"
pinneapple-mcp --help            # or: python -m pinneapple_mcp
```

## Tools

| Tool | What it does | Runs as |
|---|---|---|
| `capabilities` | version, device, optional APIs present (benchmark, zoo, datasets, geometries), limits, working directory | call |
| `list_methods` | solve methods by kind, and the method catalog with each item's validation status | call |
| `describe_method` | options of a solve method (`pinn`, `fem`, ...) or code, references and evidence of a catalog item (`S2`, ...) | call |
| `list_presets` | problem presets | call |
| `solve` | a problem (preset or JSON box problem) by one method; metrics against the reference, physics checks, trust card, `prediction.npz`, `field.png` | job |
| `trust_card` | physics checks of any field (inline values or a `.npz` / `.npy` / `.csv` file): finite values, PDE residual, boundary conditions, conservation, maximum principle, reference error, applicability (OOD); decision APPROVED / REVIEW / REJECT | call |
| `design` | design from a plain-language goal: `out_of_scope` (with what would be needed) or `needs_input` (the questions) at once, otherwise the design package: verdict with probability, design, margins, full-solver verification, operating window, bill of materials, files ([guide](design_from_intent.md)) | call / job |
| `benchmark` | several methods with the same training budget (epochs, collocation points, seeds), scored on the same grid against one reference; `table.md` and a trust card per run | job |
| `zoo_list`, `zoo_predict`, `zoo_validate` | pre-trained, verified models of `pp.zoo` and the datasets of `pp.datasets`; predictions with the rows outside the validity range flagged; the model's trust report | call |
| `job_status`, `list_jobs`, `cancel_job` | follow, list and stop jobs | call |

Every input is described by a JSON schema (from the typed signatures), and every tool returns a JSON object.
`solve` and `benchmark` return a `job_id` at once: poll `job_status` (or pass `wait_seconds`). Results are written
under the working directory, and their paths are in the result.

A box problem in JSON (expressions use the coordinate names, `pi`, `e` and NumPy functions; they are parsed, never
`eval`-ed):

```json
{"name": "plate", "domain": {"x": [0, 1], "y": [0, 1]}, "fields": ["T"], "pde": "laplace",
 "bcs": [{"kind": "dirichlet", "field": "T", "on": "y_max", "value": "sin(pi*x)"},
         {"kind": "dirichlet", "field": "T", "on": "y_min", "value": 0},
         {"kind": "dirichlet", "field": "T", "on": "x_boundary", "value": 0}],
 "exact": "sin(pi*x)*sinh(pi*y)/sinh(pi)"}
```

or a preset: `{"preset": "burgers_1d", "preset_kwargs": {"nu": 0.0032}}`.

## Safe defaults

| Setting | Default | Change with |
|---|---|---|
| working directory | `./pinneapple_mcp_work` | `--workdir`, `PINNEAPPLE_MCP_WORKDIR` |
| wall time per job | 600 s (the job process is killed) | `--max-seconds`, `PINNEAPPLE_MCP_MAX_SECONDS` |
| memory per job | 8192 MB of address space | `--max-memory-mb`, `PINNEAPPLE_MCP_MAX_MEMORY_MB` |
| concurrent jobs | 2 | `--max-jobs`, `PINNEAPPLE_MCP_MAX_JOBS` |
| network inside jobs | off (non-local connections raise) | `--allow-network`, `PINNEAPPLE_MCP_ALLOW_NETWORK=1` |
| files `trust_card` may read | the working directory and the directory the server started in, up to 200 MB | `--roots`, `PINNEAPPLE_MCP_ROOTS`, `PINNEAPPLE_MCP_MAX_FILE_MB` |
| training epochs, checked grid points | 20000, 250000 | `PINNEAPPLE_MCP_MAX_EPOCHS`, `PINNEAPPLE_MCP_MAX_POINTS` |

## Plug it in

The command is the same everywhere: `pinneapple-mcp` (add `--workdir /path/to/results` to choose where results go).
If the console script is not on the agent's `PATH`, use `python -m pinneapple_mcp` with the interpreter of the
environment where PINNeAPPle is installed.

### Gemini CLI

`~/.gemini/settings.json` (or `.gemini/settings.json` in the project):

```json
{
  "mcpServers": {
    "pinneapple": {"command": "pinneapple-mcp", "args": ["--workdir", "pinneapple_results"], "timeout": 600000}
  }
}
```

Then `/mcp` in Gemini CLI lists the tools.

### Claude Code

```bash
claude mcp add pinneapple -- pinneapple-mcp --workdir pinneapple_results
```

or, for everyone working in a repository, a `.mcp.json` at its root (this repository ships one):

```json
{"mcpServers": {"pinneapple": {"command": "python", "args": ["-m", "pinneapple_mcp", "--workdir", "pinneapple_mcp_work"]}}}
```

### Cursor

`~/.cursor/mcp.json` (global) or `.cursor/mcp.json` (project):

```json
{"mcpServers": {"pinneapple": {"command": "pinneapple-mcp", "args": ["--workdir", "pinneapple_results"]}}}
```

### Codex, Antigravity and other clients

Any client that launches stdio MCP servers takes the same command. For Codex, in `~/.codex/config.toml`:

```toml
[mcp_servers.pinneapple]
command = "pinneapple-mcp"
args = ["--workdir", "pinneapple_results"]
```

### A Colab or Jupyter cell

The tools are plain Python functions, so a notebook can call them directly:

```python
!pip install -q "pinneapple[mcp]"
from pinneapple_mcp import tools
st = tools.solve({"preset": "burgers_1d"}, "pinn", {"epochs": 2000}, wait_seconds=600)
print(st["status"], st["result"]["decision"], st["result"]["metrics"])
```

or drive the real server through an MCP client in the same process (MCP SDK 2.x; with 1.x, use
`mcp.shared.memory.create_connected_server_and_client_session(server._mcp_server)`):

```python
from mcp import Client
from pinneapple_mcp import build_server

async with Client(build_server()) as client:          # top-level await works in Colab / Jupyter
    print([t.name for t in (await client.list_tools()).tools])
    r = await client.call_tool("list_methods", {"category": "solver", "validated_only": True})
    print(r.structured_content["catalog_counts"])
```

## What an agent should do with a result

Read `decision` and `trust_cards` before anything else. `REJECT` means a critical check failed (PDE residual, a
boundary condition or the reference): report the failing checks, do not report the numbers as a result. `REVIEW`
means a non-critical check failed or too few checks could run (for example no reference and no boundary data): say
which. The checks are listed with their values and limits in the card, and in `trust_card_<field>.md` next to the
prediction.
