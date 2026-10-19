"""Fine-tune the local model on your own workbench sessions.

1. ``export_dataset(store, "data/")`` writes ``train.jsonl`` and ``valid.jsonl`` in the chat format with tools that
   ``mlx_lm`` and TRL read (``{"messages": [...], "tools": [...]}``). By default one example is one agent run: the
   full trajectory of model calls, tool calls and tool results up to the agent's final answer. Turns you rated -1 are
   left out, and with ``min_rating=1`` only turns you rated +1 are kept. Tokens and keys that look like secrets are
   masked.
2. ``train_mlx(...)`` (Apple Silicon, ``pip install mlx-lm``) or ``train_peft(...)`` (NVIDIA, ``pip install
   "pinneapple[finetune]" trl``) trains a LoRA adapter on that data.
3. ``serve_mlx_command(...)`` serves the base model with the adapter on an OpenAI-compatible port; start the workbench
   with ``--backend openai --base-url http://127.0.0.1:8080/v1`` to use it, and ``evaluate`` to compare it with the
   base model on the same tasks.

The base model must be the same family as the one that produced the data (its chat template decides how tool calls
are written). Use the Hugging Face / MLX version of the model you run in Ollama.
"""
from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .store import Store

__all__ = ["export_dataset", "redact", "MLX_BASE_MODELS", "TrainConfig", "train_mlx", "train_peft",
           "serve_mlx_command", "mlx_train_command"]

# Hugging Face repos of the MLX conversions of the recommended Ollama models (check the exact name on
# huggingface.co/mlx-community before a long download; quantised repos change names over time).
MLX_BASE_MODELS = {
    "qwen3-coder:30b": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
    "qwen3:8b": "mlx-community/Qwen3-8B-4bit",
    "qwen3:4b": "mlx-community/Qwen3-4B-4bit",
}

_SECRETS = [
    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"), re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"), re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"xox[baprs]-[A-Za-z0-9\-]{10,}"), re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END"),
    re.compile(r"(?i)(password|passwd|secret|api[_-]?key|token)(\s*[=:]\s*)(['\"]?)[^\s'\"]{6,}"),
]


def redact(text: str) -> str:
    for rx in _SECRETS:
        text = rx.sub(lambda m: (m.group(1) + m.group(2) + m.group(3) + "<redacted>") if m.re.groups >= 3
                      else "<redacted>", text)
    return text


def _clean(m: dict) -> dict:
    out = {"role": m["role"], "content": redact(m.get("content") or "")}
    if m.get("tool_calls"):
        out["tool_calls"] = [{"id": c.get("id", ""), "type": "function",
                              "function": {"name": c["function"]["name"],
                                           "arguments": redact(c["function"]["arguments"]
                                                               if isinstance(c["function"]["arguments"], str)
                                                               else json.dumps(c["function"]["arguments"]))}}
                             for c in m["tool_calls"]]
    if m["role"] == "tool":
        out["tool_call_id"] = m.get("tool_call_id", "")
        if m.get("name"):
            out["name"] = m["name"]
    return out


def export_dataset(store: Store, out_dir: str | Path, *, min_rating: int = 0, mode: str = "trajectories",
                   agents: list[str] | None = None, val_fraction: float = 0.1, seed: int = 0) -> dict:
    """Write ``train.jsonl`` / ``valid.jsonl``; returns the counts.

    ``min_rating``: 0 keeps unrated and +1 turns (drops -1), 1 keeps only +1 turns. ``mode``: ``"trajectories"`` (one
    example per agent run, the last call plus its reply) or ``"calls"`` (one example per model call)."""
    if mode not in ("trajectories", "calls"):
        raise ValueError("mode must be 'trajectories' or 'calls'")
    fb = store.feedback()
    calls = [c for c in store.model_calls() if c["ok"] and (not agents or c["agent"] in agents)]
    kept, dropped = [], 0
    for c in calls:
        rating = fb.get((c["session_id"], c["turn_id"]), {}).get("rating", 0)
        if rating < 0 or rating < min_rating:
            dropped += 1
            continue
        kept.append(c)
    if mode == "trajectories":
        last: dict[str, dict] = {}
        for c in kept:                                   # calls are in time order: the last one per run wins
            last[c["run_id"] or c["id"]] = c
        kept = [c for c in last.values() if not c["reply"].get("tool_calls")]   # finished runs only
    examples = []
    for c in kept:
        msgs = [_clean(m) for m in c["messages"]] + [_clean(c["reply"])]
        ex = {"messages": msgs}
        if c["tools"]:
            ex["tools"] = c["tools"]
        examples.append(ex)
    random.Random(seed).shuffle(examples)
    n_val = max(1, int(round(len(examples) * val_fraction))) if len(examples) >= 2 else 0
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("valid.jsonl", examples[:n_val]), ("train.jsonl", examples[n_val:])):
        with open(out / name, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"train": len(examples) - n_val, "valid": n_val, "dropped_by_rating": dropped, "mode": mode,
            "dir": str(out)}


@dataclass
class TrainConfig:
    model: str = MLX_BASE_MODELS["qwen3-coder:30b"]
    adapter_path: str = "adapters/workbench"
    iters: int = 600
    batch_size: int = 1
    num_layers: int = 16                 # transformer blocks that get LoRA (MLX); -1 for all
    learning_rate: float = 1e-5
    max_seq_length: int = 8192
    lora_rank: int = 16                  # PEFT
    lora_alpha: int = 32                 # PEFT
    epochs: float = 2.0                  # PEFT


def mlx_train_command(data_dir: str | Path, cfg: TrainConfig) -> list[str]:
    return [sys.executable, "-m", "mlx_lm.lora", "--model", cfg.model, "--train", "--data", str(data_dir),
            "--adapter-path", cfg.adapter_path, "--iters", str(cfg.iters), "--batch-size", str(cfg.batch_size),
            "--num-layers", str(cfg.num_layers), "--learning-rate", str(cfg.learning_rate),
            "--max-seq-length", str(cfg.max_seq_length), "--mask-prompt", "--grad-checkpoint"]


def train_mlx(data_dir: str | Path, cfg: TrainConfig | None = None, *, dry_run: bool = False) -> str:
    """LoRA fine-tune with ``mlx_lm`` on Apple Silicon; returns the adapter folder. ``--mask-prompt`` trains on the
    final assistant turn of each example."""
    cfg = cfg or TrainConfig()
    cmd = mlx_train_command(data_dir, cfg)
    if dry_run:
        return " ".join(cmd)
    try:
        import mlx_lm  # noqa: F401
    except ImportError as e:
        raise ImportError("training on Apple Silicon needs mlx-lm: pip install mlx-lm") from e
    subprocess.run(cmd, check=True)
    return cfg.adapter_path


def serve_mlx_command(cfg: TrainConfig | None = None, port: int = 8080) -> list[str]:
    cfg = cfg or TrainConfig()
    return [sys.executable, "-m", "mlx_lm.server", "--model", cfg.model, "--adapter-path", cfg.adapter_path,
            "--port", str(port)]


def train_peft(data_dir: str | Path, cfg: TrainConfig) -> str:
    """LoRA fine-tune with Hugging Face TRL + PEFT (CUDA). ``cfg.model`` is a Hugging Face model id; returns the
    adapter folder."""
    try:
        from datasets import load_dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as e:
        raise ImportError('training with PEFT needs: pip install "pinneapple[finetune]" trl datasets') from e
    data = load_dataset("json", data_files={"train": str(Path(data_dir) / "train.jsonl"),
                                            "validation": str(Path(data_dir) / "valid.jsonl")})
    trainer = SFTTrainer(
        model=cfg.model,
        train_dataset=data["train"], eval_dataset=data["validation"],
        peft_config=LoraConfig(r=cfg.lora_rank, lora_alpha=cfg.lora_alpha, lora_dropout=0.05,
                               target_modules="all-linear", task_type="CAUSAL_LM"),
        args=SFTConfig(output_dir=cfg.adapter_path, num_train_epochs=cfg.epochs, learning_rate=cfg.learning_rate,
                       per_device_train_batch_size=cfg.batch_size, gradient_accumulation_steps=8,
                       max_length=cfg.max_seq_length, assistant_only_loss=True, gradient_checkpointing=True,
                       bf16=True, logging_steps=10, eval_strategy="epoch", save_strategy="epoch"),
    )
    trainer.train()
    trainer.save_model(cfg.adapter_path)
    return cfg.adapter_path
