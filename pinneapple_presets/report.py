"""Solve, validate and document one preset: results, figures, 3-D stills, surrogate, inversion, optimisation,
design variants, README (data sheet + measured results) and the Lab manifest, in
``examples/industrial_presets/<name>/``.

    python -m pinneapple_presets report water_hammer_valve_closure      # or: report all
"""
from __future__ import annotations

import json
import os
import time

from .core import Preset, call_quiet, environment
from .registry import RESULTS_DIR, load

__all__ = ["write_report", "readme", "write_catalogue"]


def _rel(path: str, base: str) -> str:
    return os.path.relpath(path, base).replace(os.sep, "/")


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def write_report(preset: Preset | str, out_dir: str | None = None, *, surrogate: str = "load",
                 three_d: bool = True, lab: bool = True) -> dict:
    """``surrogate``: "build" (run the DOE and retrain), "load" (use the stored model; build if missing) or
    "skip". Returns the summary written to ``results/summary.json``."""
    p = load(preset) if isinstance(preset, str) else preset
    root = out_dir or os.path.join(RESULTS_DIR, p.name)
    res = os.path.join(root, "results")
    os.makedirs(res, exist_ok=True)
    t0 = time.perf_counter()
    r = p.run()
    r.save(os.path.join(res, "result.json"))
    figs = [_rel(f, root) for f in p.figures(r, res)]
    val = p.validate()
    with open(os.path.join(res, "validation.json"), "w", encoding="utf-8") as fh:
        json.dump(val, fh, indent=1, default=float)
    summary = {"preset": p.name, "decision": r.decision, "outputs": r.outputs,
               "trust": {"decision": r.trust.decisao, "score": r.trust.score, "coverage": r.trust.cobertura,
                         "checks": [{"name": c.name, "status": c.status, "value": c.value, "limit": c.limit,
                                     "source": c.source, "reason": c.reason} for c in r.trust.checks]},
               "validation": val, "figures": figs, "meta": r.meta}
    if p.surrogate_spec is not None and surrogate != "skip":
        from .surrogate import build_surrogate, load_surrogate
        sur = load_surrogate(p.name) if surrogate == "load" else None
        if sur is None:
            sur = build_surrogate(p)
        rs = call_quiet(p.run, fidelity="surrogate")
        summary["surrogate"] = {"model": sur.meta["model"], "inputs": sur.meta["inputs"],
                                "outputs": sur.meta["outputs"], "bounds": sur.meta["bounds"],
                                "n_train": sur.meta["n_train"], "n_test": sur.meta["n_test"],
                                "holdout": sur.meta["holdout"], "timing_s": sur.meta["timing_s"],
                                "default_case": {o: {"surrogate": rs.outputs[o], "std": rs.outputs[o + "_std"],
                                                     "solver": r.outputs[o]} for o in sur.meta["outputs"]}}
        summary["figures"] += [_rel(f, root) for f in _parity(p, sur, res)]
    inv = p.invert()
    if inv is not None:
        summary["inversion"] = inv
    opt = p.optimize()
    if opt is not None:
        summary["optimization"] = opt
    var = p.variants()
    if var:
        summary["variants"] = compare_variants(p, var, res, root, three_d=three_d)
        summary["figures"] += summary["variants"].pop("_figures", [])
    if three_d and p.has_3d:
        from .viz3d import stills
        sc = p.scene(r)
        modes = [k for s in sc.surfaces for k in s.fields] or ["real"]
        modes = list(dict.fromkeys(modes))[:3]
        shots = stills(sc, os.path.join(res, "3d"), p.name, modes)
        summary["figures_3d"] = [_rel(v, root) for k, v in shots.items() if k != "viewer" and v]
    summary["wall_time_total_s"] = time.perf_counter() - t0
    summary["environment"] = environment()
    from pinneapple_core.jsonable import to_jsonable
    summary = to_jsonable(summary)
    with open(os.path.join(res, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(root, "README.md"), "w", encoding="utf-8") as fh:
        fh.write(readme(p, summary))
    with open(os.path.join(root, ".gitignore"), "w", encoding="utf-8") as fh:
        fh.write("*_viewer.html\n*.glb\n*.usda\n")
    if lab:
        _manifest(p, summary, root)
    return summary


def compare_variants(p: Preset, variants: dict, res: str, root: str, three_d: bool = True) -> dict:
    """Run every variant with the same pipeline; table of the key outputs, a bar chart and a 3-D gallery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = {}
    results = {}
    for name, over in variants.items():
        rr = call_quiet(p.run, **over)
        results[name] = rr
        rows[name] = {"params": over, "verdict": rr.decision.get("verdict"), "trust": rr.trust.decisao,
                      "outputs": {k: v for k, v in rr.outputs.items() if isinstance(v, (int, float))}}
    keys = getattr(p, "variant_outputs", None) or list(next(iter(rows.values()))["outputs"])[:3]
    figs = []
    fig, axs = plt.subplots(1, len(keys), figsize=(3.6 * len(keys), 3.2))
    axs = [axs] if len(keys) == 1 else axs
    unit = {o.name: o.unit for o in p.outputs}
    for ax, k in zip(axs, keys, strict=False):
        vals = [rows[n]["outputs"].get(k, float("nan")) for n in rows]
        cols = ["#2a9d8f" if rows[n]["verdict"] in ("OK", "PASS") else "#e76f51" for n in rows]
        ax.bar(range(len(vals)), vals, color=cols)
        ax.set_xticks(range(len(vals)), list(rows), rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{k} [{unit.get(k, '')}]", fontsize=9)
    fig.suptitle("Variantes de projeto (verde = atende, vermelho = não atende)", fontsize=9)
    fig.tight_layout()
    path = os.path.join(res, "variants.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    figs.append(_rel(path, root))
    gallery = []
    if three_d and p.has_3d and getattr(p, "variant_gallery", True):
        from .viz3d import stills
        for name, rr in results.items():
            sc = p.scene(rr)
            if sc is None:
                continue
            sc.animation = None
            modes = [k for s in sc.surfaces for k in s.fields][:1] or ["real"]
            shots = stills(sc, os.path.join(res, "3d", "variants"), f"{p.name}_{_slug(name)}", modes, size=(800, 520))
            os.remove(shots.pop("viewer"))                    # the gallery keeps stills only
            gallery += [_rel(v, root) for v in shots.values() if v]
    return {"rows": rows, "compared": keys, "gallery_3d": gallery, "_figures": figs}


def _slug(name: str) -> str:
    import re
    import unicodedata
    s = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_").lower()


def _parity(p: Preset, sur, res: str) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    X, Y = np.asarray(sur.test["X"]), np.asarray(sur.test["Y"])
    pr = sur.predict(X)
    outs = sur.meta["outputs"]
    fig, axs = plt.subplots(1, len(outs), figsize=(3.6 * len(outs), 3.4))
    axs = [axs] if len(outs) == 1 else axs
    for j, (ax, o) in enumerate(zip(axs, outs, strict=False)):
        m, s = pr[o]
        ax.errorbar(Y[:, j], m, yerr=2 * s, fmt="o", ms=3, lw=0.8)
        lo, hi = min(Y[:, j].min(), m.min()), max(Y[:, j].max(), m.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
        e = sur.meta["holdout"]["per_output"][o]
        ax.set(xlabel=f"solver {o}", ylabel=f"surrogate {o}",
               title=f"hold-out n={len(Y)}: máx {100 * e['max_rel_err']:.2f} %", )
        ax.title.set_fontsize(9)
    fig.tight_layout()
    path = os.path.join(res, "surrogate_parity.png")
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return [path]


def _manifest(p: Preset, s: dict, root: str) -> None:
    from pinneapple_lab import experiment
    with experiment(f"examples/industrial_presets/{p.name}", title=p.title, description=p.question, domain=p.sector,
                    methods=list(p.library), modules=["pinneapple_presets"], out_dir=root, register=False,
                    references=[r["source"] for r in s["validation"]["references"]]) as ex:
        ex.param(**{k: v for k, v in s["meta"].items() if k == "wall_time_s"})
        for row in s["validation"]["references"]:
            ex.log_metric(f"rel_error[{row['case']}]", row["rel_error"], "-",
                          f"|computed - reference| / |reference|, {row['kind']}: {row['source']}")
            ex.check(row["case"], row["pass"], value=row["computed"], reference=row["reference"], rtol=row["tol"])
        if "surrogate" in s:
            ex.log_metric("surrogate_holdout_max_rel_err", s["surrogate"]["holdout"]["max_rel_err"], "-",
                          f"{s['surrogate']['n_test']} held-out solver runs (Latin hypercube)")
        if "inversion" in s and "rel_error" in s["inversion"]:
            ex.log_metric("inversion_rel_err", s["inversion"]["rel_error"], "-", s["inversion"].get("how", ""))
        for f in s["figures"] + s.get("figures_3d", []):
            ex.add_file(os.path.join(root, f), "figures")
        ex.add_file(os.path.join(root, "results", "summary.json"), "data")


def readme(p: Preset, s: dict) -> str:
    c = p.card()
    L = [f"# {p.title}", "", f"**Setor:** {p.sector}  ", f"**Preset:** `{p.name}`", "",
         f"**Problema.** {p.problem}", "", f"**Pergunta de engenharia.** {p.question}", "",
         "## Uso", "", "```python", "import pinneapple as pp", "", f'p = pp.presets.load("{p.name}")',
         "r = p.run()                      # valores padrão abaixo; troque qualquer entrada: p.run(nome=valor)",
         "print(r)                         # saídas, decisão, trust report",
         "p.validate()                     # casos de referência com o erro medido"]
    if p.surrogate_spec:
        L.append('p.run(fidelity="surrogate")     # surrogate treinado (com desvio-padrão e aviso fora do domínio)')
    L += ["```", "", f"CLI: `python -m pinneapple_presets run {p.name} nome=valor`  ",
          f"Flow Studio: nó `preset.run` com `preset = {p.name}`.", "", "## Entradas (SI) e fonte de cada valor padrão",
          "", "| entrada | unidade | padrão | domínio | fonte |", "|---|---|---|---|---|"]
    for q in c["inputs"]:
        dom = ", ".join(map(str, q["choices"])) if q.get("choices") else f"{_fmt(q['lo'])} – {_fmt(q['hi'])}"
        L.append(f"| `{q['name']}` | {q['unit']} | {_fmt(q['default'])} | {dom} | {q['source']} |")
    L += ["", "## Saídas (caso padrão, rodado aqui)", "", "| saída | valor | unidade | significado |",
          "|---|---|---|---|"]
    for o in p.outputs:
        v = s["outputs"].get(o.name)
        L.append(f"| `{o.name}` | {_fmt(v)} | {o.unit} | {o.help} |")
    d = s["decision"]
    L += ["", f"**Decisão:** {d.get('verdict')}", ""] + [f"- {x}" for x in d.get("reasons", [])]
    L += ["", f"**Trust report:** {s['trust']['decision']} (score {_fmt(s['trust']['score'])}, cobertura "
          f"{_fmt(s['trust']['coverage'])})", "", "| checagem | status | valor | limite | fonte |", "|---|---|---|---|---|"]
    for ch in s["trust"]["checks"]:
        L.append(f"| {ch['name']} | {ch['status']} | {_fmt(ch['value'])} | {_fmt(ch['limit'])} | {ch['source']} |")
    L += ["", "## Validação", "", "| caso | saída | calculado | referência | erro relativo | tolerância | tipo | fonte |",
          "|---|---|---|---|---|---|---|---|"]
    for row in s["validation"]["references"]:
        L.append(f"| {row['case']} | `{row['output']}` | {_fmt(row['computed'])} {row['unit']} | "
                 f"{_fmt(row['reference'])} {row['unit']} | {100 * row['rel_error']:.3g} % | {100 * row['tol']:.3g} % | "
                 f"{row['kind']} | {row['source']} |")
    cv = s["validation"].get("convergence")
    if cv:
        L += ["", f"Convergência ({cv['parameter']} = {cv['values']}): variação relativa entre os dois últimos níveis "
              f"{cv['rel_change_last']:.2e} ({'passa' if cv['pass'] else 'NÃO passa'})."]
    if "surrogate" in s:
        su = s["surrogate"]
        L += ["", "## Surrogate", "", f"{su['model']}, entradas {su['inputs']} no domínio "
              f"{su['bounds']}; DOE Latin hypercube com {su['n_train']} rodadas do solver para treino e "
              f"{su['n_test']} independentes para hold-out.", "", "| saída | erro rel. máx (hold-out) | médio | R² | "
              "cobertura ±2σ |", "|---|---|---|---|---|"]
        for o, e in su["holdout"]["per_output"].items():
            L.append(f"| `{o}` | {100 * e['max_rel_err']:.3g} % | {100 * e['mean_rel_err']:.3g} % | {e['r2']:.4f} | "
                     f"{100 * e['coverage_2sigma']:.0f} % |")
        L += ["", "Fora do domínio de treino o resultado sai marcado (aviso + checagem do trust report em "
              "*contradicts*), nunca extrapolado em silêncio."]
    if "inversion" in s:
        iv = s["inversion"]
        L += ["", "## Inversão", "", iv.get("description", ""), ""]
        for k in ("true", "recovered", "ci95", "rel_error", "noise", "how"):
            if k in iv:
                L.append(f"- **{k}:** {_fmt(iv[k]) if not isinstance(iv[k], (list, tuple)) else [_fmt(x) for x in iv[k]]}")
    if "optimization" in s:
        L += ["", "## Otimização", ""] + [f"- **{k}:** {_fmt(v)}" for k, v in s["optimization"].items()
                                          if not isinstance(v, (dict, list))]
    if "variants" in s:
        va = s["variants"]
        L += ["", "## Variantes de projeto", "", "| variante | parâmetros | " + " | ".join(va["compared"]) +
              " | decisão |", "|---|---|" + "---|" * len(va["compared"]) + "---|"]
        for n, row in va["rows"].items():
            L.append(f"| {n} | `{row['params']}` | " + " | ".join(_fmt(row["outputs"].get(k)) for k in va["compared"])
                     + f" | {row['verdict']} |")
        if va.get("gallery_3d"):
            L += ["", " ".join(f"![{os.path.basename(g)}]({g})" for g in va["gallery_3d"])]
    L += ["", "## Figuras", ""] + [f"![{os.path.basename(f)}]({f})" for f in s["figures"] + s.get("figures_3d", [])]
    if p.has_3d:
        L += ["", f"Visualizador 3-D interativo (arquivo único, abre do disco): regenere com "
              f"`python -m pinneapple_presets report {p.name}` → `results/3d/{p.name}_viewer.html`."]
    caps = c["capabilities"]
    L += ["", "## Capacidades", "", "| capacidade | tem? | por quê não |", "|---|---|---|"]
    for k, lab in (("3d", "visualização 3-D"), ("surrogate", "surrogate"), ("inversion", "inversão"),
                   ("optimization", "otimização"), ("variants", "variação de projeto"), ("reason", "cadeia de porquês")):
        L.append(f"| {lab} | {'sim' if caps[k] else 'não'} | {'' if caps[k] else p.missing.get(k, '')} |")
    L += ["", "## Pipeline", ""] + [f"1. {x}" for x in p.pipeline]
    L += ["", f"**Domínio de validade.** {p.validity}", ""]
    if p.limitations:
        L += ["**Limitações.**", ""] + [f"- {x}" for x in p.limitations] + [""]
    if p.reason_case:
        L += [f"**Por que cada elemento do projeto existe:** `pinneapple_reason.run_case(\"{p.reason_case}\")`.", ""]
    m = s["meta"]
    L += ["## Proveniência", "", f"Rodado em {m.get('date_utc')} · solver {m.get('wall_time_s', 0):.3g} s · "
          f"relatório completo {s.get('wall_time_total_s', 0):.3g} s · {m['hardware']['cpu']} ({m['hardware']['cores']} "
          f"núcleos) · " + ", ".join(f"{k} {v}" for k, v in m["versions"].items()), ""]
    return "\n".join(L)


def write_catalogue(root: str | None = None) -> str:
    """``examples/industrial_presets/README.md``: one row per preset from its stored summary (measured values only)
    and the folder's Lab manifest."""
    from .registry import names
    root = root or RESULTS_DIR
    L = ["# Presets industriais", "", "Problemas de engenharia já montados, resolvidos e validados com o PINNeAPPle. "
         "Cada pasta tem a ficha (entradas SI com a fonte de cada valor padrão), os resultados rodados aqui, a "
         "validação com o erro medido e as figuras. Regenere com `python -m pinneapple_presets report all`.", "",
         "| preset | setor | decisão do caso padrão | validação (pior erro / tolerância) | surrogate (erro hold-out máx) | "
         "inversão (erro) | 3-D | otimização | variantes |", "|---|---|---|---|---|---|---|---|---|"]
    for n in names():
        f = os.path.join(root, n, "results", "summary.json")
        if not os.path.isfile(f):
            continue
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        p = load(n)
        refs = s["validation"]["references"]
        worst = max(refs, key=lambda r: r["rel_error"] / max(r["tol"], 1e-12)) if refs else None
        val = (f"{'passa' if s['validation']['all_pass'] else 'FALHA'}: {100 * worst['rel_error']:.2g} % / "
               f"{100 * worst['tol']:.2g} % ({len(refs)} casos)") if worst else "-"
        sur = (f"{100 * s['surrogate']['holdout']['max_rel_err']:.3g} %" if "surrogate" in s else
               "não (" + p.missing.get("surrogate", "") + ")")
        inv = f"{100 * s['inversion']['rel_error']:.2g} %" if "inversion" in s else "-"
        L.append(f"| [{p.title}]({n}/README.md) | {p.sector} | {s['decision']['verdict']} | {val} | {sur} | {inv} | "
                 f"{'sim' if s.get('figures_3d') else 'não'} | {'sim' if 'optimization' in s else 'não'} | "
                 f"{len(s.get('variants', {}).get('rows', {}))} |")
    L += ["", "Uso: `pp.presets.load(nome).run(**params)`, `python -m pinneapple_presets run nome k=v`, nó "
          "`preset.run` no Flow Studio. Índice para o Hub: `pinneapple_presets/index.json`.", ""]
    path = os.path.join(root, "README.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    from pinneapple_lab import experiment
    with experiment("examples/industrial_presets", title="Presets industriais", description="catálogo de casos de engenharia "
                    "prontos, resolvidos e validados", domain="engineering", modules=["pinneapple_presets"],
                    out_dir=root, register=False) as ex:
        for n in names():
            f = os.path.join(root, n, "results", "summary.json")
            if os.path.isfile(f):
                with open(f, encoding="utf-8") as fh:
                    ok = json.load(fh)["validation"]["all_pass"]
                ex.check(f"{n}: validation", ok)
    return path
