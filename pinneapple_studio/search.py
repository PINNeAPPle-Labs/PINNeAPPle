"""Search the catalog in Portuguese or English: BM25 over the name, module, docstring, Portuguese prose and tags,
with accents folded, names split (``heat_sink`` / ``HeatSink`` -> ``heat sink``) and Portuguese <-> English
synonyms (a fixed table plus the English terms in parentheses in the glossary).

    idx = Index(load())            # catalog.load()
    idx.search("dissipador de calor com aletas", category="executable", area="thermal")
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any

from .taxonomy import fold

SYNONYMS = {
    "calor": "heat", "termica": "thermal", "termico": "thermal", "temperatura": "temperature", "fluido": "fluid",
    "escoamento": "flow", "estrutura": "structure", "estrutural": "structural", "incerteza": "uncertainty",
    "otimizacao": "optimization", "otimizar": "optimize", "inverso": "inverse", "malha": "mesh",
    "geometria": "geometry", "treinar": "train", "treino": "training", "validar": "validate",
    "validacao": "validation", "verificar": "verify", "verificacao": "verification", "previsao": "forecast",
    "prever": "predict", "tensao": "stress", "deformacao": "strain", "contorno": "boundary", "equacao": "equation",
    "sensibilidade": "sensitivity", "gemeo": "twin", "dados": "data", "modelo": "model", "rede": "network",
    "antena": "antenna", "eletromagnetico": "electromagnetic", "eletromagnetismo": "electromagnetics",
    "onda": "wave", "difusao": "diffusion", "viga": "beam", "placa": "plate", "tubo": "pipe", "bomba": "pump",
    "valvula": "valve", "aleta": "fin", "aletas": "fins", "dissipador": "heatsink", "resfriamento": "cooling",
    "pressao": "pressure", "velocidade": "velocity", "arrasto": "drag", "sustentacao": "lift", "asa": "wing",
    "aerofolio": "airfoil", "foguete": "rocket", "orbita": "orbit", "satelite": "satellite", "solo": "soil",
    "rio": "river", "clima": "climate", "serie": "series", "temporal": "time", "solucao": "solution",
    "resolver": "solve", "simular": "simulate", "simulacao": "simulation", "projeto": "design",
    "projetar": "design", "material": "material", "fadiga": "fatigue", "vibracao": "vibration",
    "acustica": "acoustics", "som": "sound", "reacao": "reaction", "quimica": "chemistry", "medicao": "measurement",
    "sensor": "sensor", "relatorio": "report", "exportar": "export", "grafico": "plot", "figura": "figure",
    "campo": "field", "residuo": "residual", "conservacao": "conservation", "referencia": "reference",
    "erro": "error", "custo": "cost", "superficie": "surface", "volume": "volume", "veiculo": "vehicle",
    "aeronave": "aircraft", "voo": "flight", "motor": "motor", "maquina": "machine", "eletrica": "electric",
    "corrente": "current", "campo_magnetico": "magnetic", "magnetico": "magnetic", "luz": "light",
    "otica": "optics", "quantica": "quantum", "proteina": "protein", "molecula": "molecule",
    "farmaco": "drug", "dose": "dose", "membrana": "membrane", "liberacao": "release", "robo": "robot",
    "controle": "control", "trajetoria": "trajectory", "bola": "ball", "futebol": "football",
}
STOP = {"de", "da", "do", "das", "dos", "e", "em", "um", "uma", "o", "a", "os", "as", "com", "para", "por", "no",
        "na", "nos", "nas", "que", "se", "the", "of", "and", "in", "to", "for", "with", "on", "an", "is", "by", "or",
        "como", "ao", "aos", "eu", "quero", "preciso"}
FIELDS = (("name", 3.0), ("module", 1.0), ("doc1", 2.0), ("pt", 1.5), ("doc", 0.5), ("tags", 1.0))


def tokens(text: str) -> list[str]:
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text or "")
    return [t for t in re.split(r"[^a-z0-9]+", fold(text)) if len(t) > 1 and t not in STOP]


def expand(words: list[str], extra: dict[str, str] | None = None) -> list[str]:
    syn = {**SYNONYMS, **(extra or {})}
    back = defaultdict(list)
    for k, v in syn.items():
        back[v].append(k)
    out = list(words)
    for w in words:
        if w in syn:
            out += tokens(syn[w])
        out += back.get(w, [])
    return out


def glossary_synonyms(glossary: list[dict[str, str]]) -> dict[str, str]:
    """``{portuguese word: english word}`` from glossary terms such as ``Condição de contorno (BC)``."""
    out = {}
    for g in glossary or []:
        m = re.match(r"(.+?)\s*\((.+)\)\s*$", g.get("termo", ""))
        if m:
            pt, en = tokens(m.group(1)), tokens(m.group(2))
            if len(pt) == 1 and len(en) == 1:
                out[pt[0]] = en[0]
    return out


class Index:
    def __init__(self, catalog: dict[str, Any], k1: float = 1.2, b: float = 0.75):
        self.catalog = catalog
        self.tools = catalog["tools"]
        self.extra = glossary_synonyms(catalog.get("glossary", []))
        self.k1, self.b = k1, b
        self.docs: dict[str, Counter] = {}
        df: Counter = Counter()
        for tid, t in self.tools.items():
            c: Counter = Counter()
            for field, w in FIELDS:
                if field == "tags":
                    text = " ".join(sum((list(t[k]) for k in ("tasks", "areas", "products")), []))
                elif field == "doc":
                    text = (t.get("doc") or "")[:800]
                elif field == "module":
                    text = t["module"].replace(".", " ")
                else:
                    text = t.get(field) or ""
                for tok in tokens(text):
                    c[tok] += w
            self.docs[tid] = c
            df.update(c.keys())
        self.n = max(1, len(self.docs))
        self.idf = {t: math.log(1 + (self.n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.avg = sum(sum(c.values()) for c in self.docs.values()) / self.n

    def search(self, query: str = "", *, category: str | None = None, status: str | None = None,
               task: str | None = None, area: str | None = None, product: str | None = None,
               package: str | None = None, kind: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Tools matching every filter, ranked by BM25 on the query (all of them, by name, without one)."""
        words = expand(tokens(query), self.extra)
        out = []
        for tid, t in self.tools.items():
            if (category and t["category"] != category) or (status and t["status"] != status) or \
                    (task and task not in t["tasks"]) or (area and area not in t["areas"]) or \
                    (product and product not in t["products"]) or (package and t["package"] != package) or \
                    (kind and t["kind"] != kind):
                continue
            score = 0.0
            if words:
                c = self.docs[tid]
                ln = sum(c.values())
                for w in set(words):
                    f = c.get(w, 0.0)
                    if f:
                        score += self.idf.get(w, 0.0) * f * (self.k1 + 1) / (
                            f + self.k1 * (1 - self.b + self.b * ln / self.avg))
                if score <= 0:
                    continue
            out.append((score, tid))
        out.sort(key=lambda x: (-x[0], x[1]))
        return [{"id": tid, "score": round(s, 3), "name": self.tools[tid]["name"],
                 "module": self.tools[tid]["module"], "category": self.tools[tid]["category"],
                 "status": self.tools[tid]["status"], "doc1": self.tools[tid]["doc1"],
                 "pt": self.tools[tid]["pt"][:300]} for s, tid in out[:limit]]

    def facets(self, ids: list[str] | None = None) -> dict[str, dict[str, int]]:
        """Counts per category, status, task, area, product and package (of ``ids``, or of every tool)."""
        sel = [self.tools[i] for i in ids] if ids is not None else list(self.tools.values())
        out: dict[str, Counter] = {k: Counter() for k in ("category", "status", "task", "area", "product", "package")}
        for t in sel:
            out["category"][t["category"]] += 1
            out["status"][t["status"]] += 1
            out["package"][t["package"]] += 1
            for k, f in (("task", "tasks"), ("area", "areas"), ("product", "products")):
                out[k].update(t[f].keys())
        return {k: dict(v.most_common()) for k, v in out.items()}
