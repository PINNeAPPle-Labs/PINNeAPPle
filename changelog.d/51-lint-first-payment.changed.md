- Primeira descida da dívida de lint, paga na primeira vez que o gate novo pegou alguém (#51): o PR #2344 trouxe
  `studio/solid.py` e `studio/manufacture.py` com anotações `Optional[...]`/`List[...]` e uma variável morta
  (`before = len(self.scene.surfaces)`, medida e nunca usada). Os dois arquivos foram modernizados e o baseline baixou
  de **21 339 para 21 324 achados** — a primeira vez que `docs/dev/lint_debt.json` encolhe, que é o que a issue pede
  além do registro. O teto novo é o medido agora: quem devolver os 15 achados reprova.
