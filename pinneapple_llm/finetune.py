"""LoRA fine-tuning of a local causal LM on data logged via
``conversation_store.ConversationStore``.

Scope, stated plainly: this fine-tunes a local Hugging-Face-format causal
LM's weights (via ``transformers``+``peft``, optional deps -- ``pip
install "pinneapple[finetune]"``). It does **not** fine-tune an Ollama
model directly -- Ollama itself does not do full/LoRA weight fine-tuning
(its ``Modelfile`` mechanism customises a system prompt/imports an
already-trained adapter, it doesn't train one). The realistic local
pipeline is: (1) fine-tune a HF-format base model here, on your logged
conversations; (2) if you want to *serve* the result via Ollama
afterwards, convert it to GGUF with ``llama.cpp``'s own conversion tooling
and ``ollama create`` from it -- both are separate, well-documented steps
outside this module's scope (GGUF conversion is its own significant tool,
not something to silently wrap).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Optional

from .conversation_store import ConversationStore


@dataclass
class FinetuneConfig:
    base_model: str = "meta-llama/Llama-3.2-1B"
    output_dir: str = "./pinneapple_finetuned"
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    learning_rate: float = 2e-4
    num_train_epochs: float = 3.0
    per_device_train_batch_size: int = 2
    max_seq_length: int = 1024
    seed: int = 42


def prepare_dataset(store: ConversationStore, out_jsonl: str, *, module: Optional[str] = None) -> str:
    """Export logged (prompt, response) pairs to a JSONL file, ready for
    :func:`finetune_lora`. Thin wrapper over ``ConversationStore
    .export_jsonl`` -- kept as a separate step (not folded into
    ``finetune_lora``) so the dataset can be inspected/filtered/versioned
    before spending any GPU time on it."""
    return store.export_jsonl(out_jsonl, module=module)


def tokenize_supervised_example(tokenizer, prompt: str, response: str, max_length: int) -> dict:
    """Build causal-LM labels with the instruction masked to -100.

    Returns unpadded examples so the collator can pad inputs and labels
    separately. Invalid or entirely truncated responses are rejected.
    """
    if not isinstance(prompt, str) or not isinstance(response, str):
        raise TypeError("prompt and response must be strings")
    if not prompt.strip() or not response.strip():
        raise ValueError("prompt and response must be nonempty")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")
    prefix = prompt + "\n\n"
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None:
        answer_ids = answer_ids + [eos_id]
    # Keep the full answer whenever possible. Long instructions are trimmed
    # from the left, never silently dropping all supervision.
    if len(answer_ids) >= max_length:
        answer_ids = answer_ids[:max_length]
        prefix_ids = []
    else:
        prefix_ids = prefix_ids[-(max_length - len(answer_ids)):]
    ids = prefix_ids + answer_ids
    return {
        "input_ids": ids,
        "attention_mask": [1] * len(ids),
        "labels": [-100] * len(prefix_ids) + answer_ids,
    }


def collate_supervised_batch(features: list[dict], pad_token_id: int) -> dict:
    """Right-pad inputs with the tokenizer pad ID and labels with -100."""
    import torch

    longest = max(len(item["input_ids"]) for item in features)
    return {
        "input_ids": torch.tensor([
            item["input_ids"] + [pad_token_id] * (longest - len(item["input_ids"]))
            for item in features
        ], dtype=torch.long),
        "attention_mask": torch.tensor([
            item["attention_mask"] + [0] * (longest - len(item["attention_mask"]))
            for item in features
        ], dtype=torch.long),
        "labels": torch.tensor([
            item["labels"] + [-100] * (longest - len(item["labels"]))
            for item in features
        ], dtype=torch.long),
    }


def finetune_lora(dataset_jsonl: str, cfg: FinetuneConfig,
                  validation_jsonl: Optional[str] = None):
    """Fine-tune a Hugging Face causal LM with response-only LoRA SFT.

    When provided, validation data is evaluated by Trainer and metrics are
    persisted; the validation split is never used to optimize weights.
    """
    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model
        from transformers import (
            AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments,
        )
    except ImportError as e:
        raise ImportError(
            "pinneapple_llm.finetune requires the 'finetune' extra: "
            'pip install "pinneapple[finetune]" (transformers, peft, datasets).'
        ) from e

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer needs a padding or EOS token")
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(cfg.base_model)
    model = get_peft_model(model, LoraConfig(
        r=cfg.lora_r, lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout, bias="none", task_type="CAUSAL_LM",
    ))

    def _load(path: str):
        dataset = load_dataset("json", data_files=path, split="train")
        if not len(dataset):
            raise ValueError(f"Empty fine-tuning dataset: {path}")
        def _encode(example):
            return tokenize_supervised_example(
                tokenizer, example["prompt"], example["response"], cfg.max_seq_length
            )
        return dataset.map(_encode, remove_columns=dataset.column_names)

    training = _load(dataset_jsonl)
    validation = _load(validation_jsonl) if validation_jsonl else None
    args = TrainingArguments(
        output_dir=cfg.output_dir,
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        learning_rate=cfg.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=training,
        eval_dataset=validation,
        data_collator=lambda features: collate_supervised_batch(
            features, tokenizer.pad_token_id
        ),
    )
    train_result = trainer.train()
    metrics = {"train": train_result.metrics}
    if validation is not None:
        metrics["validation"] = trainer.evaluate()
    os.makedirs(cfg.output_dir, exist_ok=True)
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)
    with open(os.path.join(cfg.output_dir, "pinneapple_finetune_config.json"), "w") as f:
        json.dump(cfg.__dict__, f, indent=2)
    with open(os.path.join(cfg.output_dir, "pinneapple_finetune_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return cfg.output_dir
