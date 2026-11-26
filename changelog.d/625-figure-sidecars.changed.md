- `scripts/reference/build.py` deixa de apagar as legendas de figura (#67, #625): o build limpava a pasta `figures/`
  de tudo que não estivesse no índice, e os sidecars `.json` que o `run_examples.py` escreve — a legenda que a figura
  carrega — nunca estão no índice. Foi assim que o PR #2344 apagou **quinze** deles logo depois de entrarem, e
  `tests/test_reference_figure_captions.py` ficou vermelho em `main` por cerca de duas horas sem ninguém ver, porque o
  teste mora no shard `root`, que não bloqueia PR. A poda agora vem em `prune_figures`, remove só imagem órfã, e tem
  teste próprio (`tests/test_reference_figures.py`). Os quinze sidecars voltaram e a referência de módulos foi
  regenerada com os símbolos públicos que os merges de hoje trouxeram.
