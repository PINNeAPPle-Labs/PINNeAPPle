"""Compare the Bayesian loop with random sampling at the same budget (#102).

    python scripts/benchmark_bayesopt.py                       # the table the guide quotes
    python scripts/benchmark_bayesopt.py --seeds 10 --budget 30

Writes the table to stdout and, with ``--markdown``, to a file the documentation embeds.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pinneapple_analysis.bayesopt import benchmark_bayesopt  # noqa: E402

DEFAULT_CASES = (("branin", 24), ("hartmann3", 30))


def table(reports) -> str:
    lines = ["| problema | orçamento | método | atingiu o alvo | chamadas até o alvo (mediana) | melhor valor (mediana) |",
             "|---|---|---|---|---|---|",]
    for r in reports:
        for name, key in (("Bayes", "bo"), ("aleatório", "random")):
            s = r[key]
            calls = "—" if s["median_evals"] is None else f"{s['median_evals']:.0f}"
            lines.append(f"| {r['problem']} | {r['budget']} | {name} | {s['reached_rate']:.0%} | {calls} | "
                         f"{s['median_best']:.4g} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="benchmark_bayesopt", description=__doc__.splitlines()[0])
    ap.add_argument("--seeds", type=int, default=5, help="how many seeds each method is measured on")
    ap.add_argument("--budget", type=int, default=None, help="override the budget of every case")
    ap.add_argument("--markdown", type=Path, default=None, help="also write the table here")
    a = ap.parse_args(argv)
    reports = [benchmark_bayesopt(name, budget=a.budget or budget, seeds=tuple(range(a.seeds)))
               for name, budget in DEFAULT_CASES]
    text = table(reports)
    print(text)
    for r in reports:
        print(f"\n{r['problem']}: fewer calls for Bayes = {r['fewer_calls']}")
    if a.markdown:
        a.markdown.parent.mkdir(parents=True, exist_ok=True)
        a.markdown.write_text(text + "\n")
        print(f"\nwritten to {a.markdown}")
    return 0 if all(r["fewer_calls"] for r in reports) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
