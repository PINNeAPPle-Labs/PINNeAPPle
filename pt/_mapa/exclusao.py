"""Padrão de exclusão do mapa: nomes que nunca podem aparecer no texto gerado (dados de terceiros).

O padrão não fica no repositório. Ele vem, nesta ordem, de:

1. a variável de ambiente ``PINNEAPPLE_MAPA_EXCLUDE`` (uma expressão regular);
2. o arquivo local ``docs/pt/_mapa/exclude.local.txt``, ignorado pelo git (uma expressão por linha; linhas vazias
   e começadas por ``#`` não contam).

Sem nenhum dos dois, nada é excluído.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

LOCAL = Path(__file__).resolve().parent / "exclude.local.txt"
NEVER = re.compile(r"(?!x)x")            # casa com nada


def padrao() -> re.Pattern:
    env = os.environ.get("PINNEAPPLE_MAPA_EXCLUDE", "").strip()
    partes = [env] if env else []
    if not partes and LOCAL.is_file():
        partes = [ln.strip() for ln in LOCAL.read_text(encoding="utf-8").splitlines()
                  if ln.strip() and not ln.lstrip().startswith("#")]
    return re.compile("|".join(f"(?:{p})" for p in partes), re.I) if partes else NEVER
