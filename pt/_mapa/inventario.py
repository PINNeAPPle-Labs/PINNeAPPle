"""Inventário automático do repositório PINNeAPPle (base do docs/pt/mapa_completo).

A lógica mora em :mod:`pinneapple_studio.inventory` (análise estática por ``ast``: nada é importado, nada é
executado), que o PINNeAPPle Studio também usa. Este script só grava o JSON que ``gerar.py`` e ``checar_prosa.py``
leem.

Uso::

    python docs/pt/_mapa/inventario.py            # escreve docs/pt/_mapa/inventario.json
    python docs/pt/_mapa/inventario.py --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from pinneapple_studio.inventory import build  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=str(Path(__file__).with_name("inventario.json")))
    args = ap.parse_args(argv)
    inv = build(ROOT)
    Path(args.out).write_text(json.dumps(inv, ensure_ascii=False, indent=1, sort_keys=False), encoding="utf-8")
    print(json.dumps(inv["totals"], indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
