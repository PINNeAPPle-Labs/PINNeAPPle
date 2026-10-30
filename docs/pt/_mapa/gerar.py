"""Gera o mapa completo do PINNeAPPle em português a partir do inventário.

Entradas:
- ``docs/pt/_mapa/inventario.json`` (de ``inventario.py``): todos os números e assinaturas;
- ``docs/pt/_mapa/prosa/*.json``: o texto em português escrito lendo o código
  (objetivo de cada pacote, módulo e símbolo; exemplos; limitações);
- ``docs/pt/_mapa/visao_geral.md`` e ``docs/pt/_mapa/glossario.json``.

Saídas:
- ``docs/pt/mapa_completo.md``: volume 0 (visão geral, arquitetura, sumário dos volumes);
- ``docs/pt/mapa_completo/vNN_*.md``: um volume por camada, mais apps/exemplos e índice.

Nenhum número é escrito à mão: todos saem do inventário.

Uso: python docs/pt/_mapa/gerar.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OUT_MAIN = ROOT / "docs" / "pt" / "mapa_completo.md"
OUT_DIR = ROOT / "docs" / "pt" / "mapa_completo"
EXCLUDE_RE = re.compile(r"vectron|shinagawa|\bCSN\b", re.I)

VOLUMES = [
    ("v01_nucleo", "Volume 1 — Núcleo, porta de entrada e catálogo",
     "A porta de entrada `pp.*`, os primitivos do núcleo, o catálogo de métodos, o registro de artefatos e os shims "
     "de compatibilidade.",
     ["pinneapple", "pinneapple_core", "pinneapple_catalog", "pinneapple_registry", "pinneapple_orchestration",
      "pinneapple_models", "pinneapple_train", "pinneapple_solvers"]),
    ("v02_fisica", "Volume 2 — Física: problemas, equações e ciência",
     "Como um problema físico é descrito e resolvido (`PhysicalProblem`, `pp.solve`), as equações, os presets "
     "industriais, o desenho de problemas, os experimentos por área, a condensação e as consultas físicas.",
     ["pinneapple_physics", "pinneapple_presets", "pinneapple_problemdesign", "pinneapple_science",
      "pinneapple_condense", "pinneapple_query"]),
    ("v03_simulacao", "Volume 3 — Simulação numérica",
     "Solvers clássicos (FDM, FEM, FVM, espectral, SPH, LBM), pontes para solvers externos, eletromagnetismo, "
     "voo, multicorpos, veículos e rovers.",
     ["pinneapple_simulation"]),
    ("v04_aprendizado", "Volume 4 — Aprendizado de máquina e dados",
     "Arquiteturas e treino de redes (PINN, operadores neurais, grafos), dados e amostragem, adaptação, "
     "modelos de mundo, percepção, computação quântica e a arena de benchmarks.",
     ["pinneapple_neural", "pinneapple_data", "pinneapple_pdb", "pinneapple_adaptation", "pinneapple_arena",
      "pinneapple_worldmodel", "pinneapple_perception", "pinneapple_quantum"]),
    ("v05_confianca", "Volume 5 — Confiança, validação e decisão",
     "A camada que confere cada resultado contra a física: validação, incerteza, inversão, o linter "
     "`pinneapple check`, registros de evidência, segurança, decisão e a cadeia de porquês.",
     ["pinneapple_analysis", "pinneapple_check", "pinneapple_veriphysics", "pinneapple_security",
      "pinneapple_decision", "pinneapple_reason"]),
    ("v06_projeto_sistemas", "Volume 6 — Projeto e sistemas",
     "Geometria, otimização de projeto, projeto a partir de intenção, sistemas dinâmicos, séries temporais, "
     "co-simulação e gêmeos digitais.",
     ["pinneapple_design", "pinneapple_systems", "pinneapple_twin3d"]),
    ("v07_ferramentas_produtos", "Volume 7 — Ferramentas, orquestração e produtos",
     "Ferramentas (benchmarks, exportação, visualização, automação de GUIs de engenharia, NVIDIA), o Flow, o Hub, "
     "o Lab, os LLMs, o servidor MCP, o laboratório autônomo e o app web.",
     ["pinneapple_tools", "pinneapple_blender", "pinneapple_flow", "pinneapple_hub", "pinneapple_lab",
      "pinneapple_llm", "pinneapple_mcp", "pinneapple_autolab", "pinneapple_app"]),
]
VOL_APPS = ("v08_apps_exemplos_ci", "Volume 8 — Apps, exemplos, reproduções, scripts e CI")
VOL_INDEX = ("v09_indice_glossario_apendice", "Volume 9 — Índice alfabético, glossário e apêndice numérico")

FLAG_PT = {"experimental": "experimental", "placeholder": "placeholder/stub citado no docstring",
           "deprecated": "shim ou obsoleto", "raises_NotImplementedError": "levanta `NotImplementedError` em algum caminho"}


# ---------------------------------------------------------------- utilidades
def esc(s: str) -> str:
    """Escapa texto para uma célula de tabela Markdown."""
    s = (s or "").replace("\n", " ").replace("|", "\\|").strip()
    return s


def code(s: str, limit: int = 90) -> str:
    s = (s or "").replace("\n", " ")
    if len(s) > limit:
        s = s[: limit - 1] + "…)"
    s = s.replace("|", "\\|")
    tick = "``" if "`" in s else "`"
    return f"{tick}{s}{tick}"


def anchor(text: str) -> str:
    a = re.sub(r"[^\w\- ]", "", text.lower(), flags=re.U).strip().replace(" ", "-")
    return a


def safe(text: str) -> str:
    lines = [ln for ln in (text or "").splitlines() if not EXCLUDE_RE.search(ln)]
    return "\n".join(lines)


def n(x) -> str:
    """Número com separador de milhar à brasileira."""
    return f"{x:,}".replace(",", ".")


# ---------------------------------------------------------------- carga
def load():
    inv = json.loads((HERE / "inventario.json").read_text(encoding="utf-8"))
    prosa = defaultdict(lambda: {"objetivo": [], "submodulos": {}, "modulos": {}, "simbolos": {},
                                 "exemplos": [], "limitacoes": []})
    for f in sorted((HERE / "prosa").glob("pinneapple*.json")):
        pkg = f.stem.split("--")[0]
        d = json.loads(f.read_text(encoding="utf-8"))
        p = prosa[pkg]
        if d.get("objetivo"):
            p["objetivo"].append((len(d["objetivo"]), d["objetivo"], f.stem))
        p["submodulos"].update(d.get("submodulos", {}))
        p["modulos"].update(d.get("modulos", {}))
        p["simbolos"].update(d.get("simbolos", {}))
        p["exemplos"].extend(d.get("exemplos", []) or [])
        for lim in d.get("limitacoes", []) or []:
            if lim not in p["limitacoes"]:
                p["limitacoes"].append(lim)
    apps = {}
    if (HERE / "prosa" / "_apps_exemplos.json").exists():
        apps = json.loads((HERE / "prosa" / "_apps_exemplos.json").read_text(encoding="utf-8"))
    glos = json.loads((HERE / "glossario.json").read_text(encoding="utf-8"))
    visao = (HERE / "visao_geral.md").read_text(encoding="utf-8")
    return inv, prosa, apps, glos, visao


def package_objective(pkg: str, p: dict, leads: dict) -> str:
    """O objetivo escrito pela tarefa líder do pacote vem primeiro; partes vêm depois."""
    if not p["objetivo"]:
        return ""
    lead_stem = leads.get(pkg)
    items = sorted(p["objetivo"], key=lambda t: (t[2] != lead_stem, -t[0]))
    main = items[0][1]
    rest = [t[1] for t in items[1:]]
    out = main.strip()
    if rest:
        out += "\n\n" + "\n\n".join(r.strip() for r in rest)
    return out


# ---------------------------------------------------------------- capítulos
def methods_for_package(inv: dict, pkg: str) -> list:
    out = []
    for mid, m in sorted(inv["method_status"]["methods"].items(),
                         key=lambda kv: (kv[0][0], int(re.sub(r"\D", "", kv[0]) or 0))):
        if any(c.startswith(pkg + "/") or c.startswith(pkg + ".") for c in m["code"]):
            out.append((mid, m))
    return out


def chapter(inv, prosa, pkg, num, leads) -> list[str]:
    P = inv["packages"][pkg]
    p = prosa.get(pkg, {"objetivo": [], "submodulos": {}, "modulos": {}, "simbolos": {}, "exemplos": [],
                        "limitacoes": []})
    L = [f"## {num}. `{pkg}`", ""]
    obj = package_objective(pkg, p, leads)
    L += ["### Objetivo", "", safe(obj) or "_Sem texto de objetivo._", ""]
    L += ["### Números do inventário", "",
          "| módulos | linhas | classes públicas | funções públicas | testes em arquivos que importam o pacote | "
          "módulos sem teste direto |",
          "|---:|---:|---:|---:|---:|---:|",
          f"| {n(P['modules'])} | {n(P['lines'])} | {n(P['classes'])} | {n(P['functions'])} | "
          f"{n(P['tests_in_files_importing'])} ({n(P['test_files_importing'])} arquivos) | "
          f"{n(P['modules_without_tests'])} |", ""]
    if P.get("tests_dir"):
        L += [f"Pasta de testes dedicada: `{P['tests_dir']}` ({n(P['tests_in_tests_dir'])} funções de teste).", ""]

    mods = sorted((name, rec) for name, rec in inv["modules"].items() if name == pkg or name.startswith(pkg + "."))
    by_sub = defaultdict(list)
    for name, rec in mods:
        by_sub[(name.split(".") + [""])[1]].append((name, rec))
    subs = [s for s in sorted(by_sub) if s]
    if subs:
        L += ["### Submódulos", "", "| submódulo | o que faz | módulos | classes + funções | testes* |",
              "|---|---|---:|---:|---:|"]
        for s in subs:
            ms = by_sub[s]
            nsym = sum(len(r["classes"]) + len(r["functions"]) for _, r in ms)
            ntest = max((r["tests"] for _, r in ms), default=0)
            L.append(f"| `{s}` | {esc(safe(p['submodulos'].get(s, '')))} | {len(ms)} | {nsym} | {ntest} |")
        L += ["", "\\* maior número de testes que importam um módulo do submódulo.", ""]

    L += ["### Módulos e API pública", ""]
    for name, rec in mods:
        title = f"#### `{name}`"
        L += [title, ""]
        txt = safe(p["modulos"].get(name, "")) or safe(rec.get("doc1", ""))
        meta = [f"`{rec['path']}`", f"{n(rec['lines'])} linhas",
                f"testes: {rec['tests']}" if rec["tests"] else "**sem teste direto**"]
        flags = [FLAG_PT[f] for f in rec.get("flags", []) if f in FLAG_PT]
        if flags:
            meta.append("palavras-chave no código (heurística; o texto acima diz o status real): " + "; ".join(flags))
        L += [txt or "_Módulo sem docstring e sem texto._", "", "<small>" + " · ".join(meta) + "</small>", ""]
        syms = [("classe", c) for c in rec["classes"]] + [("função", f) for f in rec["functions"]]
        if syms:
            L += ["| nome | tipo | assinatura | o que faz |", "|---|---|---|---|"]
            for kind, s in syms:
                desc = safe(p["simbolos"].get(f"{name}:{s['name']}", "")) or safe(s.get("doc1", ""))
                L.append(f"| `{s['name']}` | {kind} | {code(s['sig'])} | {esc(desc)} |")
            L.append("")
    ex = [e for e in p["exemplos"] if e.get("codigo") and not EXCLUDE_RE.search(json.dumps(e))]
    L += ["### Exemplos de uso reais do repositório", ""]
    if ex:
        for e in ex[:6]:
            L += [f"**{esc(e.get('titulo', ''))}** — `{e.get('caminho', '')}`", "", "```python",
                  e["codigo"].rstrip(), "```", ""]
    else:
        L += ["_Nenhum exemplo encontrado no repositório para este pacote._", ""]
    L += ["### Testes e status de validação", ""]
    L += [f"- Módulos com pelo menos um teste que os importa: "
          f"{n(P['modules'] - P['modules_without_tests'])} de {n(P['modules'])}.",
          f"- Funções de teste em arquivos que importam algum módulo do pacote: "
          f"{n(P['tests_in_files_importing'])}.", ""]
    ms = methods_for_package(inv, pkg)
    if ms:
        L += ["Itens do catálogo de métodos (`pinneapple_catalog/method_status.json`) cujo código está neste pacote:",
              "", "| id | método | status | testes contra referência | outros testes |", "|---|---|---|---:|---:|"]
        for mid, m in ms:
            L.append(f"| {mid} | {esc(m['name'])} | {m['status']} | {m['reference_tests']} | {m['other_tests']} |")
        L.append("")
    L += ["### Limitações conhecidas", ""]
    lims = [safe(x) for x in p["limitacoes"] if safe(x).strip()]
    L += [f"- {x}" for x in lims] if lims else ["- _Nenhuma registrada._"]
    L.append("")
    return L


def volume_header(title: str, intro: str, inv) -> list[str]:
    return [f"# {title}", "",
            f"> Parte do [Mapa completo do PINNeAPPle](../mapa_completo.md). Gerado por `docs/pt/_mapa/gerar.py` a "
            f"partir do inventário do commit `{inv['generated_from_commit'][:10]}`.", "", intro, ""]


def apps_volume(inv, apps) -> list[str]:
    title = VOL_APPS[1]
    L = volume_header(title, "O que cada app, exemplo, reprodução de paper, script, benchmark e workflow de CI faz. "
                             "A coluna *testes* conta funções de teste em arquivos que citam o caminho.", inv)

    def table(key, label, items):
        txt = apps.get(key, {}) if isinstance(apps.get(key, {}), dict) else {}
        out = [f"## {label}", "", f"{len(items)} itens.", "", "| item | o que faz | arquivos .py | testes |",
               "|---|---|---:|---:|"]
        for r in items:
            d = safe(txt.get(r["name"], "")) or safe(r.get("title", "") + (": " if r.get("title") and r.get("summary")
                                                                            else "") + r.get("summary", ""))
            out.append(f"| `{r['path']}` | {esc(d)} | {r['py_files']} | {r['tests']} |")
        return out + [""]

    L += table("apps", "1. Apps (`apps/`)", inv["apps"])
    L += table("examples", "2. Exemplos (`examples/`)", inv["examples"])
    L += ["## 3. Reproduções de papers", ""]
    reps = apps.get("reproducoes", []) or []
    if reps:
        for r in reps:
            if EXCLUDE_RE.search(json.dumps(r, ensure_ascii=False)):
                continue
            L += [f"### `{r.get('caminho', '')}`", "", f"- **Paper:** {safe(r.get('paper', ''))}",
                  f"- **Resultado gravado:** {safe(r.get('resultado', ''))}", ""]
    else:
        L += ["_Sem texto._", ""]
    L += table("scripts", "4. Scripts (`scripts/`)", inv["scripts"])
    L += table("benchmarks", "5. Benchmarks (`benchmarks/`)", inv["benchmarks"])
    L += ["## 6. Integração contínua (`.github/workflows/`)", "",
          "| workflow | nome | gatilhos | jobs | o que faz |", "|---|---|---|---|---|"]
    wtxt = apps.get("workflows", {})
    for w in inv["workflows"]:
        fname = Path(w["file"]).name
        trig = ", ".join(w["triggers"]) + (f" (cron `{'`, `'.join(w['cron'])}`)" if w["cron"] else "")
        L.append(f"| `{fname}` | {esc(w['name'])} | {esc(trig)} | {esc(', '.join(w['jobs']))} | "
                 f"{esc(safe(wtxt.get(fname, wtxt.get(w['file'], ''))))} |")
    L += ["", "## 7. Documentação existente (`docs/`)", "",
          f"{len(inv['docs'])} arquivos Markdown. Lista com o título de cada um:", "",
          "| arquivo | título |", "|---|---|"]
    for d in inv["docs"]:
        L.append(f"| `{d['path']}` | {esc(d['title'])} |")
    return L + [""]


def index_volume(inv, prosa, glos) -> list[str]:
    L = volume_header(VOL_INDEX[1], "Índice de todas as classes e funções públicas de nível de módulo, o glossário e "
                                    "os números do inventário.", inv)
    entries = []
    for name, rec in inv["modules"].items():
        for c in rec["classes"]:
            entries.append((c["name"], "C", name))
        for f in rec["functions"]:
            entries.append((f["name"], "f", name))
    entries.sort(key=lambda e: (e[0].lower(), e[0], e[2]))
    L += ["## 1. Índice alfabético", "",
          f"{n(len(entries))} entradas. **C** = classe, **f** = função. O caminho é o módulo onde o nome é definido "
          "(ver o capítulo do pacote no volume indicado).", ""]
    vol_of = {pkg: v[0] for v in VOLUMES for pkg in v[3]}
    letter = None
    for name, kind, mod in entries:
        first = name[0].upper()
        if not first.isalpha():
            first = "#"
        if first != letter:
            letter = first
            L += ["", f"### {letter}", ""]
        pkg = mod.split(".")[0]
        L.append(f"- `{name}` {kind} — `{mod}` ({vol_of.get(pkg, '')[:3]})")
    L += ["", "## 2. Glossário", "", "| termo | significado |", "|---|---|"]
    for t in sorted(glos, key=lambda g: g["termo"].lower()):
        L.append(f"| **{esc(t['termo'])}** | {esc(t['definicao'])} |")
    L += ["", "## 3. Apêndice: números do inventário", ""]
    T = inv["totals"]
    rows = [("pacotes `pinneapple*`", T["packages"]), ("módulos Python nesses pacotes", T["modules"]),
            ("linhas de código nesses módulos", T["lines"]), ("classes públicas", T["public_classes"]),
            ("funções públicas de nível de módulo", T["public_functions"]),
            ("métodos públicos de classes", T["public_methods"]),
            ("módulos importados por pelo menos um arquivo de teste", T["modules_with_tests"]),
            ("arquivos de teste (`tests/**/test*.py`)", T["test_files"]),
            ("funções de teste (`test*`, inclusive em classes `Test*`)", T["test_functions"]),
            ("nomes expostos em `pp.*`", T["pp_names"]), ("extras do `pyproject.toml`", T["extras"]),
            ("apps em `apps/`", T["apps"]), ("entradas em `examples/`", T["examples"]),
            ("entradas em `scripts/`", T["scripts"]), ("arquivos Markdown em `docs/`", T["docs_md"]),
            ("workflows de CI", T["workflows"]), ("métodos no catálogo", T["methods"])]
    L += ["| item | valor |", "|---|---:|"] + [f"| {a} | {n(b)} |" for a, b in rows]
    L += ["", "Status dos métodos do catálogo (regra: " + esc(inv["method_status"]["rule"]) + "; arquivo gerado em "
          + str(inv["method_status"]["generated_on"]) + "):", "", "| status | métodos |", "|---|---:|"]
    L += [f"| {k} | {v} |" for k, v in T["methods_by_status"].items()]
    L += ["", "### 3.1 Por pacote", "",
          "| pacote | módulos | linhas | classes | funções | testes* | módulos sem teste |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for pkg, P in inv["packages"].items():
        L.append(f"| `{pkg}` | {n(P['modules'])} | {n(P['lines'])} | {n(P['classes'])} | {n(P['functions'])} | "
                 f"{n(P['tests_in_files_importing'])} | {n(P['modules_without_tests'])} |")
    L += ["", "\\* funções de teste em arquivos que importam algum módulo do pacote (um teste pode contar em mais de "
          "um pacote).", "", "### 3.2 Extras do `pyproject.toml`", "",
          f"Projeto `{inv['project']['name']}` versão `{inv['project']['version']}`, Python "
          f"`{inv['project']['requires_python']}`. Dependências obrigatórias: "
          + ", ".join(f"`{d}`" for d in inv["project"]["dependencies"]) + ".", "",
          "| extra | pacotes |", "|---|---|"]
    for k, v in inv["extras"].items():
        L.append(f"| `{k}` | {esc(', '.join(v))} |")
    L += ["", "### 3.3 Catálogo completo de métodos", "", "| id | método | status | testes contra referência | "
          "outros testes | código |", "|---|---|---|---:|---:|---|"]
    for mid, m in sorted(inv["method_status"]["methods"].items(),
                         key=lambda kv: (kv[0][0], int(re.sub(r"\D", "", kv[0]) or 0))):
        L.append(f"| {mid} | {esc(m['name'])} | {m['status']} | {m['reference_tests']} | {m['other_tests']} | "
                 f"{esc(', '.join(m['code'][:3]))} |")
    L += ["", "### 3.4 Como os números foram obtidos", "",
          "- Tudo vem de `docs/pt/_mapa/inventario.py`, que lê o código com `ast` (nada é importado ou executado).",
          "- *Público* = nome de nível de módulo que não começa com `_`. Métodos de classes são contados à parte.",
          "- *Testes de um módulo* = funções `test*` em arquivos de `tests/` que importam aquele módulo "
          "(`import`, `from … import`, `importlib.import_module`, `pytest.importorskip`). É uma medida de alcance, "
          "não de cobertura de linhas.",
          "- O status de validação dos métodos é o de `pinneapple_catalog/method_status.json`, gerado por "
          "`scripts/build_method_status.py`.",
          "- Itens ligados a clientes foram excluídos do inventário e do texto.", ""]
    return L


def pp_table(inv, prosa) -> list[str]:
    t = inv["top_api"]
    rows = {}
    for k, v in t["eager"].items():
        rows[k] = ("import direto", v)
    for k, v in t["lazy"].items():
        rows[k] = ("preguiçoso (`_LAZY_ATTRS`)", v)
    for k, v in t["submodules"].items():
        rows.setdefault(k, ("submódulo preguiçoso (`_SUBMODULES`)", v))
    for k in t["defined_here"]:
        rows.setdefault(k, ("definido em `pinneapple/__init__.py`", "pinneapple." + k))
    allsym = {}
    for pk in prosa.values():
        allsym.update(pk["simbolos"])
        allsym.update({k: v for k, v in pk["modulos"].items()})
    L = ["### A porta de entrada `pp.*`", "",
         f"`import pinneapple as pp` expõe {n(len(rows))} nomes (versão `{t['version']}`). A tabela diz de onde cada "
         "um vem e o que faz.", "", "| nome | como é carregado | origem | o que faz |", "|---|---|---|---|"]
    for k, (how, src) in sorted(rows.items(), key=lambda kv: kv[0].lower()):
        mod, _, attr = src.rpartition(".")
        desc = allsym.get(f"{mod}:{attr}") or allsym.get(src) or ""
        if not desc:
            rec = inv["modules"].get(mod)
            if rec:
                for s in rec["classes"] + rec["functions"]:
                    if s["name"] == attr:
                        desc = s["doc1"]
            elif src in inv["modules"]:
                desc = inv["modules"][src]["doc1"]
        L.append(f"| `pp.{k}` | {how} | `{src}` | {esc(safe(desc))} |")
    return L + [""]


def main():
    inv, prosa, apps, glos, visao = load()
    leads = {}
    specs_file = HERE / "leads.json"
    if specs_file.exists():
        leads = json.loads(specs_file.read_text())
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    T = inv["totals"]
    # volume 0
    L = ["# Mapa completo do PINNeAPPle", "",
         f"Documento de referência, em português, de todo o repositório: {n(T['packages'])} pacotes, "
         f"{n(T['modules'])} módulos, {n(T['public_classes'])} classes e {n(T['public_functions'])} funções públicas. "
         f"Gerado a partir do inventário automático do commit `{inv['generated_from_commit'][:10]}`.", "",
         "Como o texto passa de 150 páginas, ele está dividido em volumes por camada. Este arquivo é o volume 0: "
         "visão geral, arquitetura e sumário.", "", "## Volumes", "", "| volume | conteúdo | pacotes |",
         "|---|---|---|"]
    for key, title, intro, pkgs in VOLUMES:
        L.append(f"| [{title}](mapa_completo/{key}.md) | {esc(intro)} | "
                 + ", ".join(f"`{p}`" for p in pkgs) + " |")
    L.append(f"| [{VOL_APPS[1]}](mapa_completo/{VOL_APPS[0]}.md) | {len(inv['apps'])} apps, "
             f"{len(inv['examples'])} entradas de exemplos, reproduções de papers, scripts, benchmarks e "
             f"{len(inv['workflows'])} workflows de CI. | — |")
    L.append(f"| [{VOL_INDEX[1]}](mapa_completo/{VOL_INDEX[0]}.md) | Índice de todas as classes e funções públicas, "
             "glossário e números. | — |")
    L += ["", visao.replace("{{PP_TABLE}}", "\n".join(pp_table(inv, prosa))), ""]
    L += ["## Como este documento foi feito", "",
          "1. `docs/pt/_mapa/inventario.py` percorre o repositório por AST e grava `inventario.json`.",
          "2. O texto em português de cada pacote, módulo e símbolo foi escrito lendo o código e está em "
          "`docs/pt/_mapa/prosa/*.json`; `docs/pt/_mapa/checar_prosa.py` confere que nada ficou sem texto.",
          "3. `docs/pt/_mapa/gerar.py` monta os volumes. Quando falta texto em português para um símbolo, a tabela "
          "mostra a primeira linha da docstring original (em inglês).",
          "4. `docs/pt/_mapa/pdf.js` gera o PDF.", ""]
    OUT_MAIN.write_text("\n".join(L), encoding="utf-8")
    for key, title, intro, pkgs in VOLUMES:
        V = volume_header(title, intro, inv)
        V += ["## Sumário", ""] + [f"{i}. [`{p}`](#{i}-{anchor(p)})" for i, p in enumerate(pkgs, 1)] + [""]
        for i, pkg in enumerate(pkgs, 1):
            if pkg in inv["packages"]:
                V += chapter(inv, prosa, pkg, i, leads)
        (OUT_DIR / f"{key}.md").write_text("\n".join(V), encoding="utf-8")
    (OUT_DIR / f"{VOL_APPS[0]}.md").write_text("\n".join(apps_volume(inv, apps)), encoding="utf-8")
    (OUT_DIR / f"{VOL_INDEX[0]}.md").write_text("\n".join(index_volume(inv, prosa, glos)), encoding="utf-8")
    allp = {p for v in VOLUMES for p in v[3]}
    missing = set(inv["packages"]) - allp
    if missing:
        raise SystemExit(f"pacotes fora de qualquer volume: {sorted(missing)}")
    files = [OUT_MAIN] + sorted(OUT_DIR.glob("*.md"))
    for f in files:
        txt = f.read_text(encoding="utf-8")
        if EXCLUDE_RE.search(txt):
            raise SystemExit(f"termo de cliente em {f}")
        heads = sum(1 for ln in txt.splitlines() if ln.startswith("#"))
        print(f"{f.relative_to(ROOT)}: {len(txt.splitlines())} linhas, {heads} títulos")


if __name__ == "__main__":
    main()
