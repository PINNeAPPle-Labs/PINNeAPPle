"""Declarative decision tree walked by :class:`~pinneapple_decision.loop.DecisionLoop`.

The default tree (:func:`physics_ai_tree`)::

    has_analytical_solution?  --yes-->  analytical_baseline (execute) --> check_baseline (validate)
            |                                                    passed -> accept
            no                                                   failed -> has_reference_data?
            v
    has_reference_data?  --yes-->  surrogate_model   (choice: data-driven surrogate)  --+
            |                                                                           |
            no ----------------->  physics_model     (choice: physics-informed model) --+--> train (execute)
                                                                                           |
                                                          validate  <----------------------+
                                                          passed -> accept
                                                          failed -> next_experiment (choice: training strategy)
                                                                    --> train  (back into the loop)

Node kinds:

* :class:`ChoiceNode` -- a :class:`PhysicsChoice`. If the node has a ``fact`` and
  the (adapted) problem carries that fact as a boolean, the answer is read from
  the problem: a :class:`Decision` with ``level=DecisionLevel.FACT`` and
  ``backend="fact:<key>"``, nothing is scored. Otherwise the decider (rules or
  LLM) answers, and the step is marked ``resolved_by="decider"`` with the reason.
* :class:`ExecuteNode` -- executes the most recent choice through
  ``DecisionLoop.execute`` (constraints re-checked). The executor receives the
  plan built along the path in ``problem["experiment"]``
  (``approach``, ``model``, ``training_strategy``...). Counts against
  ``max_experiments``.
* :class:`ValidateNode` -- ``DecisionLoop.verify``; the edge follows
  ``verification.passed``.
* :class:`TerminalNode` -- ends the walk with an outcome.

Every choice (fact or decider) is logged in the loop's ``EvidenceStore`` with
``decision.diagnostics["tree"]`` = ``{tree, node, resolved_by, fact,
fact_status, ...}``; execution and verification fill the same record, so the
store holds the full path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Union

from .constraints import NoFeasibleOptionError
from .schema import Decision, DecisionLevel, DecisionState, Evidence, ExecutionResult, PhysicsChoice, Verification

ChoiceFactory = Union[PhysicsChoice, Callable[[], PhysicsChoice]]


@dataclass
class ChoiceNode:
    id: str
    choice: ChoiceFactory
    edges: Dict[str, str]  # option -> next node id; "*" = any other option
    fact: Optional[str] = None
    fact_options: Dict[bool, str] = field(default_factory=lambda: {True: "yes", False: "no"})
    plan_key: Optional[str] = None  # store the selected option in the plan under this key
    plan_updates: Dict[str, Any] = field(default_factory=dict)

    def make_choice(self) -> PhysicsChoice:
        return self.choice if isinstance(self.choice, PhysicsChoice) else self.choice()

    def next(self, selected: str) -> str:
        try:
            return self.edges[selected] if selected in self.edges else self.edges["*"]
        except KeyError:
            raise KeyError(f"tree node {self.id!r} has no edge for option {selected!r}") from None


@dataclass
class ExecuteNode:
    id: str
    next: str
    plan_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidateNode:
    id: str
    on_pass: str
    on_fail: str
    plan_updates: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TerminalNode:
    id: str
    outcome: str
    plan_updates: Dict[str, Any] = field(default_factory=dict)


Node = Union[ChoiceNode, ExecuteNode, ValidateNode, TerminalNode]


@dataclass
class DecisionTree:
    name: str
    root: str
    nodes: Dict[str, Node]

    def __post_init__(self) -> None:
        if self.root not in self.nodes:
            raise ValueError(f"root {self.root!r} is not a node")
        for node in self.nodes.values():
            targets: List[str] = []
            if isinstance(node, ChoiceNode):
                targets = list(node.edges.values())
            elif isinstance(node, ExecuteNode):
                targets = [node.next]
            elif isinstance(node, ValidateNode):
                targets = [node.on_pass, node.on_fail]
            for t in targets:
                if t not in self.nodes:
                    raise ValueError(f"node {node.id!r} points to unknown node {t!r}")


@dataclass
class TreeStep:
    node: str
    kind: str  # "choice" | "execute" | "validate" | "terminal"
    selected: Optional[str] = None
    resolved_by: Optional[str] = None  # "fact" | "decider" (choice nodes)
    decision_id: Optional[str] = None
    passed: Optional[bool] = None
    note: str = ""


@dataclass
class TreeRun:
    tree: str
    outcome: str  # "accepted" | "budget_exhausted" | "no_feasible_option" | "max_steps" | terminal outcome
    path: List[TreeStep]
    plan: Dict[str, Any]
    experiments: int
    evidence: List[Evidence]
    facts: Dict[str, Any] = field(default_factory=dict)

    def decided_by_decider(self) -> List[str]:
        return [s.node for s in self.path if s.resolved_by == "decider"]

    def decided_by_fact(self) -> List[str]:
        return [s.node for s in self.path if s.resolved_by == "fact"]


# --------------------------------------------------------------------------- default tree

#: Data-driven surrogates (trained on reference simulations/measurements).
SURROGATE_MODELS = ("deeponet", "fno", "mesh_graph_net", "pino")
#: Physics-informed models trainable without reference data (PDE residual loss).
PHYSICS_INFORMED_MODELS = ("pinn", "xpinn", "pino", "inverse_pinn")


def physics_ai_tree() -> DecisionTree:
    """The tree proposed for the physics-AI workflow (see module docstring)."""
    from .selectors import ANALYTICAL_SOLUTION, REFERENCE_DATA, ModelSelector, TrainingStrategySelector

    nodes: List[Node] = [
        ChoiceNode("has_analytical_solution", ANALYTICAL_SOLUTION.choice,
                   {"yes": "analytical_baseline", "no": "has_reference_data"},
                   fact="has_analytical_solution"),
        ExecuteNode("analytical_baseline", next="check_baseline",
                    plan_updates={"approach": "analytical_baseline"}),
        ValidateNode("check_baseline", on_pass="accept", on_fail="has_reference_data"),
        ChoiceNode("has_reference_data", REFERENCE_DATA.choice,
                   {"yes": "surrogate_model", "no": "physics_model"}, fact="has_reference_data"),
        ChoiceNode("surrogate_model", lambda: ModelSelector.choice(options=SURROGATE_MODELS),
                   {"*": "train"}, plan_key="model", plan_updates={"approach": "data_driven_surrogate"}),
        ChoiceNode("physics_model", lambda: ModelSelector.choice(options=PHYSICS_INFORMED_MODELS),
                   {"*": "train"}, plan_key="model", plan_updates={"approach": "physics_informed"}),
        ExecuteNode("train", next="validate"),
        ValidateNode("validate", on_pass="accept", on_fail="next_experiment"),
        ChoiceNode("next_experiment", TrainingStrategySelector.choice, {"*": "train"},
                   plan_key="training_strategy"),
        TerminalNode("accept", outcome="accepted"),
    ]
    return DecisionTree("physics_ai", "has_analytical_solution", {n.id: n for n in nodes})


# --------------------------------------------------------------------------- walk

def _fact_decision(choice: PhysicsChoice, node: ChoiceNode, value: bool, status: str, rule: str) -> Decision:
    selected = node.fact_options[value]
    return Decision(
        choice=choice.name, question=choice.question, selected=selected,
        probabilities={o: (1.0 if o == selected else 0.0) for o in choice.options},
        level=DecisionLevel.FACT, requires_validation=False, backend=f"fact:{node.fact}",
        rationale=[f"{node.fact}={value} ({status}{': ' + rule if rule else ''}); answered from the problem, "
                   "not scored"],
    )


def run_tree(loop: Any, problem: Any, tree: Optional[DecisionTree] = None, *, max_experiments: int = 3,
             max_steps: int = 200) -> TreeRun:
    """Walk ``tree`` with ``loop`` (a :class:`DecisionLoop`). See the module docstring."""
    tree = tree or physics_ai_tree()
    base = DecisionState.from_problem(problem)
    facts = base.facts
    plan: Dict[str, Any] = {}
    path: List[TreeStep] = []
    touched: List[str] = []
    last_decision: Optional[Decision] = None
    last_result: Optional[ExecutionResult] = None
    last_verification: Optional[Verification] = None
    experiments = 0
    node_id = tree.root
    outcome = "max_steps"

    for _ in range(max_steps):
        node = tree.nodes[node_id]
        plan.update(node.plan_updates)

        if isinstance(node, TerminalNode):
            path.append(TreeStep(node.id, "terminal", note=node.outcome))
            outcome = node.outcome
            break

        if isinstance(node, ChoiceNode):
            choice = node.make_choice()
            value = base.problem.get(node.fact) if node.fact else None
            tree_diag: Dict[str, Any] = {"tree": tree.name, "node": node.id, "fact": node.fact}
            if node.fact and isinstance(value, bool):
                status = "inferred" if node.fact in facts.get("inferred", {}) else "given"
                rule = facts.get("inferred", {}).get(node.fact, {}).get("rule", "")
                decision = _fact_decision(choice, node, value, status, rule)
                loop._issued[decision.id] = (choice, base)
                tree_diag.update(resolved_by="fact", fact_status=status, fact_value=value)
            else:
                state = DecisionState(problem=dict(base.problem), facts=facts, previous_result=last_result,
                                      verification=last_verification, history=loop.store.history())
                try:
                    decision = loop.decide_choice(state, choice)
                except NoFeasibleOptionError as e:
                    path.append(TreeStep(node.id, "choice", note=str(e)))
                    outcome = "no_feasible_option"
                    break
                tree_diag["resolved_by"] = "decider"
                if node.fact:
                    tree_diag.update(fact_status="unknown",
                                     note=f"fact {node.fact!r} is unknown for this problem; asked the decider "
                                          f"({loop.decider.backend.name})")
            decision.diagnostics["tree"] = tree_diag
            loop.store.add(Evidence(decision=decision))
            touched.append(decision.id)
            last_decision = decision
            if node.plan_key:
                plan[node.plan_key] = decision.selected
            path.append(TreeStep(node.id, "choice", decision.selected, tree_diag["resolved_by"], decision.id,
                                 note=tree_diag.get("note", "")))
            node_id = node.next(decision.selected)
            continue

        if isinstance(node, ExecuteNode):
            if experiments >= max_experiments:
                path.append(TreeStep(node.id, "execute",
                                     note=f"budget exhausted: {experiments}/{max_experiments} experiments run"))
                outcome = "budget_exhausted"
                break
            if last_decision is None:
                raise RuntimeError(f"execute node {node.id!r} reached before any choice")
            experiments += 1
            last_result = loop.execute(last_decision, problem={**base.problem, "experiment": dict(plan)})
            last_verification = None
            path.append(TreeStep(node.id, "execute", last_decision.selected, decision_id=last_decision.id,
                                 note=f"experiment {experiments}/{max_experiments}: {dict(plan)}"))
            node_id = node.next
            continue

        if isinstance(node, ValidateNode):
            if last_result is None:
                raise RuntimeError(f"validate node {node.id!r} reached before any execution")
            last_verification = loop.verify(last_result)
            path.append(TreeStep(node.id, "validate", decision_id=last_result.decision_id,
                                 passed=last_verification.passed, note=last_verification.source))
            node_id = node.on_pass if last_verification.passed else node.on_fail
            continue

        raise TypeError(f"unknown node type {type(node).__name__}")

    loop.store.save()
    evidence = [e for e in (loop.store.find(i) for i in touched) if e is not None]
    return TreeRun(tree.name, outcome, path, plan, experiments, evidence, facts)


__all__ = [
    "ChoiceNode",
    "DecisionTree",
    "ExecuteNode",
    "PHYSICS_INFORMED_MODELS",
    "SURROGATE_MODELS",
    "TerminalNode",
    "TreeRun",
    "TreeStep",
    "ValidateNode",
    "physics_ai_tree",
    "run_tree",
]
