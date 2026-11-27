"""Check the environment you are in against the supported-versions matrix, and run the smoke (roadmap I6, #93).

    python scripts/check_matrix.py                # in the matrix? then the smoke; exit code says it
    python scripts/check_matrix.py --json         # the same, machine-readable
    python scripts/check_matrix.py --no-smoke     # only the matrix verdict (milliseconds, no torch import)
    python scripts/check_matrix.py --record       # write this passing run into pinneapple_stability/support_matrix.json
    python scripts/check_matrix.py --markdown     # regenerate docs/support_matrix.md from the recorded runs
    python scripts/check_matrix.py --check-docs   # fail when the page and the file disagree (what the test does)

The matrix itself — Python range, torch floor and lines, operating systems, cells — is declared in
``pinneapple_stability/support_matrix.json``; ``pinneapple_stability.support`` reads it and says what an environment
is. This script is the half that imports torch and the library: the five steps below, seconds each, chosen because
they are the first thing a torch or numpy change breaks.

Exit code: ``0`` in the matrix and every step passed; ``2`` the steps passed but the environment is outside the
matrix (unsupported Python, unsupported torch, no torch); ``1`` a step failed.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinneapple_stability import support  # noqa: E402

PAGE = os.path.join(ROOT, "docs", "support_matrix.md")


# ── the smoke steps ──────────────────────────────────────────────────────────────────────────────────────
def _imports() -> str:
    import numpy
    import scipy
    import torch

    return f"numpy {numpy.__version__}, scipy {scipy.__version__}, torch {torch.__version__}"


def _facade() -> str:
    import pinneapple as pp

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        pp.info()
    out = buf.getvalue()
    assert "import failed" not in out, "pp.info() reports a package that does not import"
    assert "Packages:" in out, "pp.info() printed no package list"
    version = str(pp.__version__)
    assert support.version_line(version), f"pp.__version__ does not parse: {version!r}"
    return f"pinneapple {version}"


def _preset() -> str:
    import pinneapple as pp

    spec = pp.get_preset("poisson_2d")
    kind = spec.pde.kind
    assert kind, "the preset has no PDE kind"
    assert tuple(spec.fields), "the preset has no fields"
    return f"poisson_2d: pde {kind!r}, fields {tuple(spec.fields)}"


def _stability() -> str:
    from pinneapple_stability import stability

    got = stability("pinneapple_physics.pinn_solver")
    assert got == "core", f"pinneapple_physics.pinn_solver is classified {got!r}, not 'core'"
    return "pinneapple_physics.pinn_solver is core"


def _autograd() -> str:
    import torch

    import pinneapple as pp

    model = pp.build_model("vanilla_pinn", in_dim=2, out_dim=1)
    x = torch.rand(16, 2, requires_grad=True)
    out = model(x)
    grad = torch.autograd.grad(out.y.sum(), x, create_graph=True)[0]
    assert tuple(grad.shape) == (16, 2), f"the gradient in the inputs has shape {tuple(grad.shape)}"
    loss = out.y.pow(2).mean() + grad.pow(2).mean()
    loss.backward()
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    opt.step()
    pred = model.predict(x.detach())
    assert tuple(pred.shape) == (16, 1), f"predict returned {tuple(pred.shape)}"
    return f"gradient {tuple(grad.shape)}, loss {loss.detach().item():.3e}, predict {tuple(pred.shape)}"


STEPS = {
    "imports": _imports,
    "facade": _facade,
    "preset": _preset,
    "stability": _stability,
    "autograd": _autograd,
}


def smoke(only: list[str] | None = None) -> list[dict]:
    """Run the declared steps, in the order of the matrix file, and time each one."""
    out = []
    for step in support.matrix()["smoke"]:
        if only and step["id"] not in only:
            continue
        started = time.perf_counter()
        try:
            detail = STEPS[step["id"]]()
            ok = True
        except Exception as exc:  # noqa: BLE001 - a failed step is the result, not a crash
            detail, ok = f"{type(exc).__name__}: {exc}", False
        out.append({"id": step["id"], "ok": ok, "detail": detail,
                    "seconds": round(time.perf_counter() - started, 2)})
    return out


# ── the command line ────────────────────────────────────────────────────────────────────────────────────
def _print_report(verdict: support.Verdict, results: list[dict]) -> None:
    env = verdict.env
    print(f"environment : python {env.get('python')} · torch {env.get('torch') or 'not installed'} · "
          f"{env.get('os')} {env.get('arch') or ''}".rstrip())
    print(f"matrix      : python {support.matrix()['python']['min']}-{support.matrix()['python']['max']} · "
          f"torch >= {support.matrix()['torch']['floor']} · {', '.join(support.matrix()['os'])}")
    for reason in verdict.reasons:
        print(f"verdict     : {reason}")
    if results:
        print()
        for r in results:
            print(f"  [{'ok' if r['ok'] else 'FAIL'}] {r['id']:<10} {r['seconds']:>5.2f}s  {r['detail']}")
        failed = [r["id"] for r in results if not r["ok"]]
        total = sum(r["seconds"] for r in results)
        print(f"\n{len(results) - len(failed)}/{len(results)} steps passed in {total:.1f}s"
              + (f"; failed: {', '.join(failed)}" if failed else ""))
    print(f"\n{verdict.status.upper()}"
          + (f" — {verdict.cell.label()}" if verdict.cell else ""))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true", help="print the verdict and the steps as JSON")
    ap.add_argument("--no-smoke", action="store_true", help="only the matrix verdict: no torch import")
    ap.add_argument("--only", nargs="*", help="run only these steps (default: every step in the matrix file)")
    ap.add_argument("--record", action="store_true", help="write a fully passing run into support_matrix.json")
    ap.add_argument("--note", default="", help="with --record: a sentence on the machine, printed on the page")
    ap.add_argument("--markdown", action="store_true", help="regenerate docs/support_matrix.md and stop")
    ap.add_argument("--check-docs", action="store_true", help="fail when docs/support_matrix.md is out of date")
    ap.add_argument("--python", dest="py", help="check these versions instead of the current ones")
    ap.add_argument("--torch", help="the torch version to classify (default: the installed one)")
    ap.add_argument("--os", dest="osname", help="the operating system to classify (linux, macos, windows)")
    a = ap.parse_args(argv)

    if a.markdown:
        with open(PAGE, "w", encoding="utf-8") as f:
            f.write(support.render_markdown())
        print(f"wrote {os.path.relpath(PAGE, ROOT)}")
        return 0
    if a.check_docs:
        try:
            with open(PAGE, encoding="utf-8") as f:
                current = f.read()
        except FileNotFoundError:
            current = ""
        if current != support.render_markdown():
            print("docs/support_matrix.md is out of date: run python scripts/check_matrix.py --markdown")
            return 1
        print("docs/support_matrix.md matches support_matrix.json")
        return 0

    env = support.probe(torch_version=a.torch)
    if a.py:
        env["python"] = a.py
    if a.osname:
        env["os"] = support.os_name(a.osname)
    verdict = support.classify(env)
    results = [] if a.no_smoke else smoke(a.only)

    if a.json:
        print(json.dumps({"verdict": verdict.as_dict(), "steps": results}, indent=2, ensure_ascii=False))
    else:
        _print_report(verdict, results)

    failed = [r for r in results if not r["ok"]]
    if a.record:
        if failed:
            print(f"not recorded: {len(failed)} step(s) failed")
        else:
            run = support.record(env, results, note=a.note)
            print(f"recorded {run['os']} python {support.version_line(run['python'])} "
                  f"torch {support.version_line(run['torch'])} on {run['at']}"
                  f" — run scripts/check_matrix.py --markdown to update the page")
    if failed:
        return 1
    return 0 if verdict.in_matrix else 2


if __name__ == "__main__":
    sys.exit(main())
