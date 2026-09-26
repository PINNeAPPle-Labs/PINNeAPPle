"""Compute the validation status of every item in ``pinneapple_catalog.methods`` from the test suites.

The unit of evidence is a single test function (``def test_...`` block, plus the file's
imports). A test exercises an item when the block (or the imports) contains one of the
item's probes; it is a *reference* test when the same block also compares against an
independent reference: closed-form / analytical / exact solution, manufactured solution,
or a published / tabulated value. Working per function (not per file) avoids crediting an
item with a reference comparison that belongs to another test in the same file.

    python scripts/build_method_status.py        # writes pinneapple_catalog/method_status.json

Status per item: ``validated`` (>= 1 reference test), ``tested`` (tests but no reference
test) or ``untested``. The JSON keeps the evidence (test file names) so every status can
be audited by hand.
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pinneapple_catalog.methods import METHODS  # noqa: E402

# PINNeAPPle-CFD's tests cover the pipe-flow equations (E45-E50) and P9; the sibling checkout is
# the default, PINNEAPPLE_CFD_TESTS points elsewhere (e.g. when running from a separate worktree).
CFD_TESTS = os.environ.get("PINNEAPPLE_CFD_TESTS", os.path.join(os.path.dirname(ROOT), "PINNeAPPle-CFD", "tests"))
TEST_DIRS = [os.path.join(ROOT, "tests"), CFD_TESTS]
REFERENCE = re.compile(
    r"analytic(?:al)?[ _](?:solution|profile|value|result|reference|formula|limit)|closed[- _]form|"
    r"exact[ _](?:solution|value|profile|traveling|travelling)|\bexact\s*=|\bdef exact\b|u_exact|_exact\b|manufactured|\bMMS\b|"
    r"published (?:value|result|table|number|figure)|tabulated|Ghia|Hansen|Incropera|Schlichting|"
    r"Howarth|reference solution|literature value",
    re.IGNORECASE,
)


def _test_files():
    for d in TEST_DIRS:
        for dirpath, _, names in os.walk(d):
            for n in names:
                if n.startswith("test_") and n.endswith(".py"):
                    path = os.path.join(dirpath, n)
                    with open(path, encoding="utf-8", errors="ignore") as f:
                        rel = os.path.relpath(path, d)
                        repo = "PINNeAPPle/tests" if d == TEST_DIRS[0] else "PINNeAPPle-CFD/tests"
                        yield f"{repo}/{rel}", f.read()


def _own_body(block: str) -> str:
    """Cut the block where the test's indentation ends (module-level lines after it belong to no test)."""
    lines = block.splitlines()
    indent = len(lines[0]) - len(lines[0].lstrip())
    kept = [lines[0]]
    for line in lines[1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent and not line.lstrip().startswith((")", "]", "}")):
            break
        kept.append(line)
    return "\n".join(kept)


# Manual review of the automatic result (2026-09-24). Each entry says why the scan was wrong.
OVERRIDES = {
    "S21": ("tested", "the only 'reference' hit is an OPC-UA live-stream test; no state-estimation "
                      "result is compared with an independent reference"),
    "S4": ("tested", "the hit compares gradient backends with each other (consistency), "
                     "not the spectral solver with an independent reference"),
}


def _uses_import(header: str, body: str, probes) -> bool:
    """The probe is only in the imports: count the test when it uses a name imported on a probe line."""
    for line in header.splitlines():
        if line.lstrip().startswith(("import ", "from ")) and any(p in line for p in probes):
            names = re.findall(r"import\s+(.+)", line)
            for n in (names[0].replace("(", "").replace(")", "").split(",") if names else []):
                n = n.split(" as ")[-1].strip()
                if n and re.search(rf"\b{re.escape(n)}\b", body):
                    return True
    return False


def main() -> None:
    files = list(_test_files())
    blocks = []  # (file, test name, header+body text)
    for name, text in files:
        parts = re.split(r"(?m)^(?=(?:    )?(?:async )?def test_)", text)
        header, tests = parts[0], parts[1:]
        for body in tests:
            test_name = re.match(r"\s*(?:async )?def (test_\w+)", body).group(1)
            blocks.append((name, test_name, header, _own_body(body)))
    out = {}
    for m in METHODS:
        ref, other = set(), set()
        for name, test_name, header, body in blocks:
            if any(p in body for p in m.probes) or (any(p in header for p in m.probes) and _uses_import(header, body, m.probes)):
                (ref if REFERENCE.search(body) else other).add(f"{name}::{test_name}")
        ref, other = sorted(ref), sorted(other - ref)
        status = "validated" if ref else ("tested" if other else "untested")
        entry = {"status": status, "reference_tests": ref, "other_tests": other}
        if m.id in OVERRIDES:
            entry["status"], entry["override"] = OVERRIDES[m.id]
        out[m.id] = entry
    payload = {
        "generated_on": _dt.date.today().isoformat(),
        "rule": "validated = a test exercising the item compares against an independent reference "
                "(closed form, manufactured solution, published/tabulated value)",
        "test_files_scanned": len(files),
        "test_functions_scanned": len(blocks),
        "methods": out,
    }
    dest = os.path.join(ROOT, "pinneapple_catalog", "method_status.json")
    with open(dest, "w") as f:
        json.dump(payload, f, indent=1, sort_keys=True)
    counts = {}
    for v in out.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    print(f"{len(files)} test files scanned -> {dest}")
    print(counts)


if __name__ == "__main__":
    main()


def write_markdown(status_path: str | None = None) -> str:
    """Render docs/dev/CATALOGO_METODOS.md (Portuguese, for the team) from the catalog + status JSON."""
    from pinneapple_catalog.methods import list_methods, method_status

    status = method_status()
    label = {"validated": "✅ validado", "tested": "🟡 testado, sem referência", "untested": "⚪ sem teste"}
    titles = {"solver": "A. Solvers (S)", "training": "B. Métodos de treino (T)",
              "equation": "C. Equações físicas (E)", "problem": "D. Problemas físicos (P)"}
    lines = [
        "# Catálogo de métodos do PINNeAPPle",
        "",
        "> Gerado por `scripts/build_method_status.py` a partir de `pinneapple_catalog/methods.py` e dos testes. "
        "Não editar à mão: rode o script de novo.",
        "",
        "**Regra (D5 de `PEDIDOS_2026-09-24.md`):** ✅ = existe teste que compara o método com uma referência "
        "independente (solução fechada, solução manufaturada, valor publicado ou tabelado), na mesma função de "
        "teste. 🟡 = há teste, mas nenhum compara com referência. ⚪ = nenhum teste encontrado. "
        "Só ✅ entra na lista oficial (`validated_methods()`). O resto fica **pendente: paper + benchmark**; "
        "nenhum código foi apagado.",
        "",
        "Referências: \"no código\" = já citadas no docstring do módulo; \"adicionadas\" = fonte canônica "
        "incluída no catálogo porque o código não citava nada. Todo arXiv foi conferido contra o título em 2026-09-24.",
        "",
    ]
    counts = {}
    for m in list_methods():
        s = status.get(m.id, {}).get("status", "untested")
        counts.setdefault(m.category, {}).setdefault(s, 0)
        counts[m.category][s] += 1
    lines += ["| Categoria | ✅ | 🟡 | ⚪ |", "|---|---|---|---|"]
    for cat, t in titles.items():
        c = counts.get(cat, {})
        lines.append(f"| {t} | {c.get('validated', 0)} | {c.get('tested', 0)} | {c.get('untested', 0)} |")
    for cat, title in titles.items():
        lines += ["", f"## {title}", ""]
        for m in list_methods(cat):
            st = status.get(m.id, {})
            lines.append(f"### {m.id} — {m.name}")
            lines.append(f"- **Estado:** {label[st.get('status', 'untested')]}")
            if st.get("override"):
                lines.append(f"- **Revisão manual:** {st['override']}")
            ev = st.get("reference_tests") or st.get("other_tests") or []
            if ev:
                more = f" (+{len(ev) - 3})" if len(ev) > 3 else ""
                lines.append(f"- **Evidência:** " + "; ".join(f"`{e}`" for e in ev[:3]) + more)
            lines.append(f"- **Código:** " + ", ".join(f"`{c}`" for c in m.code))
            if m.equations:
                lines.append(f"- **Equações:** {', '.join(m.equations)}")
            if m.refs_in_code:
                lines.append("- **Referências no código:** " + " · ".join(m.refs_in_code))
            if m.refs_added:
                lines.append("- **Referências adicionadas:** " + " · ".join(m.refs_added))
            if not m.references and m.category != "problem":
                lines.append("- **Referências:** nenhuma fonte encontrada — pendente de paper")
            lines.append("")
    dest = os.path.join(ROOT, "docs", "dev", "CATALOGO_METODOS.md")
    with open(dest, "w") as f:
        f.write("\n".join(lines))
    return dest


if __name__ == "__main__" and "--markdown" in sys.argv:
    print(write_markdown())
