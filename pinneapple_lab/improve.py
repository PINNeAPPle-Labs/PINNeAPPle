"""``python -m pinneapple_lab improve <item> [--run]``: the suggested actions for an item's next letter, and, on
request, the registered lab commands among them run in-process, then a re-curation that reports the new letter.

Only actions the registry marks runnable (``pinneapple_lab.next_steps.ACTIONS``: rerun, refine sweep, seed sweep,
``pinneapple check``) are ever executed, and only with ``run=True``; every other action (a check to add to the
experiment, a comparison to code, a human review) is printed for a person to do. Nothing else is run.
"""
from __future__ import annotations

import os
from typing import Any

RUNNABLE = {"lab_run": "cli:pinneapple_lab run", "lab_sweep_refine": "cli:pinneapple_lab sweep",
            "lab_sweep_seeds": "cli:pinneapple_lab sweep", "physics_check_cli": "cli:pinneapple check"}


def runnable_actions(gaps) -> list[Any]:
    """The registered, runnable actions of ``gaps``, each once, in gap order."""
    out, seen = [], set()
    for g in gaps:
        for a in g.actions:
            if a.runnable and RUNNABLE.get(a.id) == a.target and a.argv and a.command not in seen:
                seen.add(a.command)
                out.append(a)
    return out


def _execute(action, store) -> int:
    argv = list(action.argv)
    if action.target == "cli:pinneapple check":
        parts = argv[1].split("/")[1:]                     # "<lab>/runs/exp/id" -> this store's run folder
        argv[1] = os.path.join(store.root, *parts)
        from pinneapple_check.cli import main as check_main
        try:
            return int(check_main(argv) or 0)
        except SystemExit as e:                             # argparse-style exits
            return int(e.code or 0)
    from .__main__ import main as lab_main
    return int(lab_main(["--root", store.root, *argv]) or 0)


def improve(item: str, store, *, run: bool = False, lang: str = "en", log=print) -> int:
    """Print the item's gaps and actions; with ``run=True`` run the registered lab commands and re-curate.
    Returns 0, or 1 when a command failed."""
    from .curation import curate, next_steps
    from .next_steps import explain
    from .seal import load_curation
    cur = load_curation(store)
    before = cur["items"][item]["tier"]
    gaps = next_steps(item, cur)
    log(explain(gaps, lang, f"{item}: {'nível' if lang == 'pt' else 'tier'} {before}"))
    acts = runnable_actions(gaps)
    pt = lang == "pt"
    if not acts:
        log("\n" + ("Nenhum comando de laboratório registrado para executar; as ações acima são manuais." if pt else
                    "No registered lab command to run; the actions above are for a person to do."))
        return 0
    if not run:
        log("\n" + ("Simulação. Com --run, executaria:" if pt else "Dry run. With --run it would run:"))
        for a in acts:
            log(f"  {a.command}")
        return 0
    failed = 0
    for a in acts:
        log(f"\n$ {a.command}")
        failed += _execute(a, store) != 0
    after = curate(store)["items"].get(item, {}).get("tier", before)
    log(f"\n{item}: {before} -> {after}")
    return 1 if failed else 0
