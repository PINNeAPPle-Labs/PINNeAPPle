- `docs/pt`: the complete map of the library in Portuguese (overview plus ten volumes) and its sources in
  `docs/pt/_mapa`: `inventario.py` (static AST inventory of packages, modules, public symbols, tests, catalog, apps
  and examples), the Portuguese prose per module and symbol (`prosa/*.json`), the glossary and the generators. The
  inventory JSON and the PDF are generated (`python docs/pt/_mapa/inventario.py`, `python docs/pt/_mapa/pdf.py`)
  and git-ignored.
