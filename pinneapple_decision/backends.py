"""Scoring backends: turn (state, choice, options in a given order) into scores.

A backend returns one real-valued score (a logit / log-probability) per option,
**in the order the options were shown**. It never generates text and never
parses free-form answers. The decider turns scores into a distribution and
applies L0 debiasing (``debias.py``).

* :class:`RuleBasedBackend` -- deterministic, offline, no LLM. Scores come from
  the selectors' rules (``selectors.py``), which read the problem and the
  evidence. Order-invariant by construction and has no label prior, so the
  decider skips the prior correction for it.
* :class:`LogProbLLMBackend` -- any causal LM that exposes next-token
  log-probabilities. The options are listed with letters (A, B, ...) and the
  score of each option is the log-probability of its letter as the next token
  after ``Answer:``. Position and label-prior biases are expected; that is what
  L0 corrects. The existing ``pinneapple_llm._dispatch.call_llm`` returns text
  only (Anthropic / OpenAI chat / Ollama), so it is not used here; plug a
  log-prob function in directly or use :meth:`LogProbLLMBackend.from_transformers`.
"""
from __future__ import annotations

import json
import string
from typing import Callable, Dict, List, Optional, Protocol, Sequence, runtime_checkable

from .schema import DecisionState, PhysicsChoice

LETTERS = string.ascii_uppercase


@runtime_checkable
class ScoringBackend(Protocol):
    name: str
    #: True when the scores carry a label prior that a neutral state can estimate.
    has_label_prior: bool

    def score(self, state: DecisionState, choice: PhysicsChoice, options: Sequence[str]) -> Sequence[float]:
        ...

    def rationale(self, state: DecisionState, choice: PhysicsChoice, options: Sequence[str]) -> List[str]:
        ...


class RuleBasedBackend:
    """Deterministic backend over per-choice rule functions (see ``selectors.py``)."""

    name = "rules"
    has_label_prior = False

    def __init__(self, rules: Optional[Dict[str, Callable]] = None):
        if rules is None:
            from .selectors import SELECTORS

            rules = {k: s.rules for k, s in SELECTORS.items()}
        self.rules = dict(rules)

    def _rules_for(self, choice: PhysicsChoice) -> Callable:
        try:
            return self.rules[choice.name]
        except KeyError:
            raise KeyError(f"RuleBasedBackend has no rules for choice {choice.name!r}; "
                           f"known: {sorted(self.rules)}") from None

    def score(self, state: DecisionState, choice: PhysicsChoice, options: Sequence[str]) -> List[float]:
        scores, _ = self._rules_for(choice)(state, list(options))
        return [float(scores[o]) for o in options]

    def rationale(self, state: DecisionState, choice: PhysicsChoice, options: Sequence[str]) -> List[str]:
        return list(self._rules_for(choice)(state, list(options))[1])


DEFAULT_TEMPLATE = (
    "You choose the next physics-AI experiment. You do not predict physical results.\n"
    "State (problem and evidence so far):\n{state}\n\n"
    "Question: {question}\n"
    "Options:\n{options}\n"
    "Reply with the letter of the best option.\n"
    "Answer:"
)

# prompt, candidate labels -> log-probability of each label as the next token
LogProbFn = Callable[[str, Sequence[str]], Sequence[float]]


class LogProbLLMBackend:
    """Scores options by next-token log-probability of their letter label."""

    has_label_prior = True

    def __init__(self, logprob_fn: LogProbFn, *, template: str = DEFAULT_TEMPLATE, name: str = "llm-logprob"):
        self.logprob_fn = logprob_fn
        self.template = template
        self.name = name

    def render(self, state: DecisionState, choice: PhysicsChoice, options: Sequence[str]) -> str:
        if len(options) > len(LETTERS):
            raise ValueError(f"at most {len(LETTERS)} options")
        lines = []
        for letter, opt in zip(LETTERS, options):
            desc = choice.info(opt).description
            lines.append(f"{letter}. {opt}" + (f" -- {desc}" if desc else ""))
        return self.template.format(
            state=json.dumps(state.summary(), indent=1, sort_keys=True, default=str),
            question=choice.question,
            options="\n".join(lines),
        )

    def score(self, state: DecisionState, choice: PhysicsChoice, options: Sequence[str]) -> List[float]:
        prompt = self.render(state, choice, options)
        return [float(x) for x in self.logprob_fn(prompt, list(LETTERS[: len(options)]))]

    def rationale(self, state: DecisionState, choice: PhysicsChoice, options: Sequence[str]) -> List[str]:
        return [f"{self.name}: next-token log-probability of each option letter (no text generated)"]

    @classmethod
    def from_transformers(cls, model_id: str, *, device: Optional[str] = None, **kwargs) -> "LogProbLLMBackend":
        """Build a backend on a local Hugging Face causal LM (optional dependency).

        One forward per prompt; the score of label ``X`` is the log-softmax of the
        last-position logits at the token id of ``" X"`` (falling back to ``"X"``).
        Check the model's own license before using it for client work.
        """
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError("LogProbLLMBackend.from_transformers needs `transformers` and `torch`") from e

        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id)
        if device:
            model = model.to(device)
        model.eval()

        def label_id(label: str) -> int:
            for text in (" " + label, label):
                ids = tok.encode(text, add_special_tokens=False)
                if len(ids) == 1:
                    return ids[0]
            raise ValueError(f"label {label!r} is not a single token for {model_id}")

        def logprob_fn(prompt: str, labels: Sequence[str]) -> List[float]:  # pragma: no cover - needs a model
            enc = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                logits = model(**enc).logits[0, -1]
            lp = torch.log_softmax(logits.float(), dim=-1)
            return [float(lp[label_id(lbl)]) for lbl in labels]

        return cls(logprob_fn, name=f"hf:{model_id}", **kwargs)


__all__ = ["DEFAULT_TEMPLATE", "LogProbLLMBackend", "RuleBasedBackend", "ScoringBackend"]
