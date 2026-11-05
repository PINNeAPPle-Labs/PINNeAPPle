"""The Portuguese prose of the map of the library (``docs/pt/_mapa/prosa/*.json`` and ``glossario.json``).

Each package file holds the package's purpose (``objetivo``), its limitations (``limitacoes``), worked examples
(``exemplos``), and one text per submodule, module and symbol (``"module:Name"``); ``_apps_exemplos.json`` holds the
apps, examples, reproductions, scripts, benchmarks and workflows, with the gallery's titles in Portuguese
(``titulos``, by gallery id) and, under ``en``, the English title or description of an entry whose own source has
none in English (``en.titulos``, ``en.descricoes``). The glossary carries each term in both languages (``termo`` /
``definicao`` and ``term`` / ``definition``). The text is shown as written: nothing is
summarised, nothing is generated. A symbol without prose has none (``""``), and :func:`unmatched` lists prose whose
module or symbol is no longer in the code.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .inventory import EXCLUDE_RE, repo_root

PROSE_DIR = Path("docs") / "pt" / "_mapa" / "prosa"
GLOSSARY = Path("docs") / "pt" / "_mapa" / "glossario.json"


@dataclass
class Prose:
    packages: dict[str, dict[str, Any]] = field(default_factory=dict)   # objetivo, limitacoes, exemplos, submodulos
    modules: dict[str, str] = field(default_factory=dict)
    symbols: dict[str, str] = field(default_factory=dict)               # "module:Name" -> text
    folders: dict[str, dict[str, str]] = field(default_factory=dict)    # apps / examples / scripts / ... -> {name: text}
    reproductions: list[dict[str, Any]] = field(default_factory=list)
    glossary: list[dict[str, str]] = field(default_factory=list)        # [{"termo", "definicao", "term", "definition"}]
    titles: dict[str, str] = field(default_factory=dict)                # gallery id -> title in Portuguese
    en_titles: dict[str, str] = field(default_factory=dict)             # gallery id -> English title (source in PT)
    en_descriptions: dict[str, str] = field(default_factory=dict)       # gallery id -> English description

    def module(self, name: str) -> str:
        return self.modules.get(name, "")

    def symbol(self, module: str, name: str) -> str:
        return self.symbols.get(f"{module}:{name}", "")

    def package(self, name: str) -> dict[str, Any]:
        return self.packages.get(name, {})


def _clean(v: Any) -> Any:
    """Drop client names from the text (the same rule as the inventory)."""
    if isinstance(v, str):
        return "\n".join(ln for ln in v.splitlines() if not EXCLUDE_RE.search(ln))
    if isinstance(v, list):
        return [_clean(x) for x in v]
    if isinstance(v, dict):
        return {k: _clean(x) for k, x in v.items() if not EXCLUDE_RE.search(str(k))}
    return v


def load(root: Path | None = None) -> Prose:
    """The prose of the checkout at ``root``; empty when ``docs/pt/_mapa`` is not there."""
    root = Path(root or repo_root())
    out = Prose()
    d = root / PROSE_DIR
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        data = _clean(json.loads(p.read_text(encoding="utf-8")))
        if p.name == "_apps_exemplos.json":
            for k, v in data.items():
                if k == "reproducoes":
                    out.reproductions = list(v)
                elif k == "titulos":
                    out.titles = dict(v)
                elif k == "en":
                    out.en_titles = dict(v.get("titulos") or {})
                    out.en_descriptions = dict(v.get("descricoes") or {})
                elif isinstance(v, dict):
                    out.folders[k] = dict(v)
            continue
        pkg = data.get("pacote") or re.sub(r"--parte\d+$", "", p.stem)
        info = out.packages.setdefault(pkg, {"objetivo": "", "limitacoes": [], "exemplos": [], "submodulos": {}})
        if data.get("objetivo") and not info["objetivo"]:
            info["objetivo"] = data["objetivo"]
        info["limitacoes"] += [x for x in data.get("limitacoes") or [] if x not in info["limitacoes"]]
        info["exemplos"] += list(data.get("exemplos") or [])
        info["submodulos"].update(data.get("submodulos") or {})
        out.modules.update(data.get("modulos") or {})
        out.symbols.update(data.get("simbolos") or {})
    g = root / GLOSSARY
    if g.is_file():
        out.glossary = _clean(json.loads(g.read_text(encoding="utf-8")))
    return out


def unmatched(prose: Prose, inventory: dict[str, Any]) -> dict[str, list[str]]:
    """Prose keys whose module or symbol is not in ``inventory`` (code that moved or exists only on another branch)."""
    mods = inventory["modules"]
    names = {f"{m}:{x['name']}" for m, r in mods.items() for x in r["classes"] + r["functions"]}
    names |= {f"{m}:{c['name']}.{meth}" for m, r in mods.items() for c in r["classes"] for meth in c["methods"]}
    return {"modules": sorted(k for k in prose.modules if k not in mods),
            "symbols": sorted(k for k in prose.symbols if k not in names)}
