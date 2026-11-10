"""Offline source-code SFT dataset builder and local LoRA training entry point.

Source extraction is deterministic, traceable to files, and never calls an LLM.
Code/docstrings are not independently validated statements about physical truth.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PACKAGES = ("pinneapple", "pinneapple_core", "pinneapple_physics", "pinneapple_llm")
SECRET = re.compile(r"(sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)")
SKIP = {".git", ".venv", "venv", "build", "dist", "__pycache__", "node_modules"}


@dataclass(frozen=True)
class CodeDatasetConfig:
    root: str = "."
    output_dir: str = "artifacts/llm_code_dataset"
    packages: tuple[str, ...] = DEFAULT_PACKAGES
    validation_fraction: float = 0.1
    max_source_chars: int = 5000
    min_doc_chars: int = 20
    seed: int = 42


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    args = node.args
    positional = args.posonlyargs + args.args
    default_offset = len(positional) - len(args.defaults)
    names = []
    for i, arg in enumerate(positional):
        value = arg.arg
        if i >= default_offset:
            value += "=" + ast.unparse(args.defaults[i - default_offset])
        names.append(value)
        if args.posonlyargs and i == len(args.posonlyargs) - 1:
            names.append("/")
    if args.vararg:
        names.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        names.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        names.append(arg.arg + ("=" + ast.unparse(default) if default is not None else ""))
    if args.kwarg:
        names.append("**" + args.kwarg.arg)
    return ("async " if isinstance(node, ast.AsyncFunctionDef) else "") + node.name + "(" + ", ".join(names) + ")"


def _sources(root: Path, packages: tuple[str, ...]):
    for package in sorted(set(packages)):
        if not package.isidentifier() or package.startswith("_"):
            raise ValueError(f"Unsafe package selector: {package!r}")
        directory = root / package
        if not directory.is_dir() or not (directory / "__init__.py").is_file():
            continue
        for path in sorted(directory.rglob("*.py")):
            if not any(p in SKIP or p.startswith(".") for p in path.relative_to(directory).parts):
                yield path


def _examples(path: Path, root: Path, cfg: CodeDatasetConfig):
    source_file = path.read_text(encoding="utf-8", errors="replace")
    if SECRET.search(source_file):
        return
    try:
        tree = ast.parse(source_file, filename=str(path))
    except (SyntaxError, ValueError):
        return
    relative = path.relative_to(root).as_posix()
    module = relative[:-3].replace("/", ".")
    lines = source_file.splitlines(keepends=True)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name.startswith("_"):
            continue
        doc = (ast.get_docstring(node) or "").strip()
        if len(doc) < cfg.min_doc_chars or SECRET.search(doc):
            continue
        code = "".join(lines[node.lineno - 1:node.end_lineno]).rstrip()
        if len(code) > cfg.max_source_chars:
            continue
        symbol = module + "." + node.name
        signature = _signature(node)
        prompt = ("Explain this existing PINNeAPPle Python API. Describe only facts supported by its "
                  f"source and docstring.\nAPI: {symbol}\nSignature: {signature}")
        response = (f"API: {symbol}\nSignature: {signature}\n"
                    f"Documented behavior:\n{doc}\n\nSource:\n{code}")
        sample_id = hashlib.sha256((relative + ":" + symbol + ":" + code).encode()).hexdigest()
        yield {
            "id": sample_id, "prompt": prompt, "response": response,
            "source_path": relative, "symbol": symbol, "start_line": node.lineno,
            "end_line": node.end_lineno, "group": relative,
            "source_sha256": hashlib.sha256(source_file.encode()).hexdigest()
        }


def _revision(root: Path) -> str | None:
    try:
        run = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5, check=True)
        return run.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _split(group: str, cfg: CodeDatasetConfig) -> str:
    raw = hashlib.sha256(f"{cfg.seed}:{group}".encode()).digest()
    number = int.from_bytes(raw[:8], "big") / 2**64
    return "validation" if number < cfg.validation_fraction else "train"


def generate_code_dataset(cfg: CodeDatasetConfig = CodeDatasetConfig()) -> dict:
    """Produce train/validation JSONL and a checksum/provenance manifest.

    All symbols from the same file stay in one split to reduce leakage.
    """
    if not 0 <= cfg.validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    if cfg.max_source_chars < 1 or cfg.min_doc_chars < 0:
        raise ValueError("Invalid size limits")
    root = Path(cfg.root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    sources = list(_sources(root, cfg.packages))
    records = [row for path in sources for row in _examples(path, root, cfg)]
    records.sort(key=lambda x: (x["source_path"], x["symbol"], x["start_line"]))
    records = list({row["id"]: row for row in records}.values())
    if not records:
        raise ValueError("No documented, eligible public code symbols found")
    splits = {"train": [], "validation": []}
    for record in records:
        splits[_split(record["group"], cfg)].append(record)
    output = Path(cfg.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for split, rows in splits.items():
        path = output / f"{split}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        checksums[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        "schema_version": 1, "generator": "pinneapple_llm.code_training",
        "git_revision": _revision(root), "config": asdict(cfg),
        "files_scanned": len(sources), "counts": {k: len(v) for k, v in splits.items()},
        "sha256": checksums,
        "limitations": [
            "Docstrings and source are not independent execution or physics evidence",
            "Validation is split by file rather than independent repository",
            "Review third-party code licenses before dataset redistribution"
        ]
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8"
    )
    return manifest


def finetune_code_dataset(dataset_dir: str, *, base_model: str, output_dir: str,
                          epochs: float = 3, learning_rate: float = 2e-4) -> str:
    """Verify dataset integrity, then reuse PINNeAPPle's existing LoRA trainer.

    GPU/CPU allocation and base-model license/access are the caller's responsibility.
    The existing trainer currently fits only the train split; evaluation is separate.
    """
    from .finetune import FinetuneConfig, finetune_lora

    directory = Path(dataset_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported dataset schema")
    if not manifest["counts"]["train"]:
        raise ValueError("Empty training split")
    for split in ("train", "validation"):
        data_file = directory / f"{split}.jsonl"
        digest = hashlib.sha256(data_file.read_bytes()).hexdigest()
        if digest != manifest["sha256"][split]:
            raise ValueError(f"Dataset checksum mismatch: {split}")
    config = FinetuneConfig(
        base_model=base_model, output_dir=output_dir,
        num_train_epochs=epochs, learning_rate=learning_rate
    )
    validation_path = str(directory / "validation.jsonl") if manifest["counts"]["validation"] else None
    return finetune_lora(str(directory / "train.jsonl"), config, validation_jsonl=validation_path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PINNeAPPle code dataset and LoRA pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Create a reproducible training dataset")
    generate.add_argument("--root", default=".")
    generate.add_argument("--output-dir", default="artifacts/llm_code_dataset")
    generate.add_argument("--package", action="append", dest="packages")
    generate.add_argument("--validation-fraction", type=float, default=0.1)
    generate.add_argument("--seed", type=int, default=42)
    train = commands.add_parser("train", help="Fine-tune a local HF causal LM using LoRA")
    train.add_argument("--dataset-dir", default="artifacts/llm_code_dataset")
    train.add_argument("--base-model", required=True)
    train.add_argument("--output-dir", default="artifacts/llm_code_adapter")
    train.add_argument("--epochs", type=float, default=3)
    train.add_argument("--learning-rate", type=float, default=2e-4)
    ingest = commands.add_parser("ingest-evidence", help="Import reviewed, hashed execution examples")
    ingest.add_argument("--input", required=True)
    ingest.add_argument("--output", required=True)
    ingest.add_argument("--root", default=".")
    args = parser.parse_args(argv)
    if args.command == "generate":
        cfg = CodeDatasetConfig(
            root=args.root, output_dir=args.output_dir,
            packages=tuple(args.packages) if args.packages else DEFAULT_PACKAGES,
            validation_fraction=args.validation_fraction, seed=args.seed
        )
        print(json.dumps(generate_code_dataset(cfg), indent=2))
    elif args.command == "ingest-evidence":
        from .verified_examples import import_verified_examples
        print(json.dumps(import_verified_examples(args.input, args.output, root=args.root), indent=2))
    else:
        print(finetune_code_dataset(
            args.dataset_dir, base_model=args.base_model, output_dir=args.output_dir,
            epochs=args.epochs, learning_rate=args.learning_rate
        ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
