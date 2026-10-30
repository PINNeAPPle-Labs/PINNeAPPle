"""Confere se docs/pt/_mapa/prosa/<pacote>.json cobre todo o inventário daquele pacote.

Uso: python docs/pt/_mapa/checar_prosa.py pinneapple_core [pinneapple_x ...]
     python docs/pt/_mapa/checar_prosa.py pinneapple_physics:astro,closed_form   # só esses submódulos
Lista módulos e símbolos públicos sem texto em português, e chaves que não existem no inventário.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXCLUDE_RE = re.compile(r"vectron|shinagawa|\bCSN\b", re.I)
REQUIRED = ("objetivo", "submodulos", "modulos", "simbolos", "exemplos", "limitacoes")


def expected(inv: dict, pkg: str, part: str | None = None):
    mods, syms = set(), set()
    for name, rec in inv["modules"].items():
        if name == pkg or name.startswith(pkg + "."):
            if part and not (name == part or name.startswith(part + ".")):
                continue
            mods.add(name)
            for c in rec["classes"] + rec["functions"]:
                syms.add(f"{name}:{c['name']}")
    return mods, syms


def check(inv: dict, pkg: str) -> list[str]:
    errs = []
    subs = None
    if ":" in pkg:
        pkg, raw = pkg.split(":", 1)
        subs = set(raw.split(","))
    files = sorted(HERE.glob(f"prosa/{pkg}.json")) + sorted(HERE.glob(f"prosa/{pkg}--*.json"))
    if not files:
        return [f"{pkg}: sem arquivo de prosa"]
    mods_txt, syms_txt, top = {}, {}, {}
    for f in files:
        d = json.loads(f.read_text(encoding="utf-8"))
        mods_txt.update(d.get("modulos", {}))
        syms_txt.update(d.get("simbolos", {}))
        for k in REQUIRED:
            if d.get(k):
                top[k] = True
        raw = f.read_text(encoding="utf-8")
        if EXCLUDE_RE.search(raw):
            errs.append(f"{f.name}: contém termo de cliente proibido")
    for k in REQUIRED:
        if subs is None and k not in top and k not in ("exemplos",):
            errs.append(f"{pkg}: campo '{k}' vazio")
    mods, syms = expected(inv, pkg)
    if subs is not None:
        keep = lambda n: (n.split(":")[0].split(".") + [""])[1] in subs
        mods = {m for m in mods if keep(m)}
        syms = {x for x in syms if keep(x)}
    for m in sorted(mods - {k for k, v in mods_txt.items() if v and v.strip()}):
        errs.append(f"módulo sem texto: {m}")
    for s in sorted(syms - {k for k, v in syms_txt.items() if v and v.strip()}):
        errs.append(f"símbolo sem texto: {s}")
    all_mods, all_syms = expected(inv, pkg)
    for k in sorted(set(mods_txt) - all_mods):
        errs.append(f"módulo inexistente no inventário: {k}")
    for k in sorted(set(syms_txt) - all_syms):
        errs.append(f"símbolo inexistente no inventário: {k}")
    return errs


def main(argv):
    inv = json.loads((HERE / "inventario.json").read_text(encoding="utf-8"))
    pkgs = argv or sorted(inv["packages"])
    bad = 0
    for pkg in pkgs:
        errs = check(inv, pkg)
        bad += len(errs)
        print(f"{pkg}: {'OK' if not errs else f'{len(errs)} pendência(s)'}")
        for e in errs[:200]:
            print("  ", e)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
