- `pinneapple_lab.quality`: o **checklist de nove itens e o selo de qualidade de um experimento** (#923). Lê o
  `experiment.json` e a pasta e responde, item a item, o que um leitor industrial precisa ver sem abrir o README:
  referência ou linha de base (e a fonte dela), recorte reservado ou extrapolação, repetições com intervalo, orçamento
  igual entre os métodos comparados, ablação, convergência de malha ou passo de tempo, proveniência e o comando que
  reproduz, veredito calculado dos resultados com as perdas mantidas, e a lista explícita do que **não** foi medido.
  Cada item é `met`, `missing`, `not applicable` (nada é comparado, nenhum solver, nenhuma alegação de melhoria) ou
  `not assessed` (não há o que ler) -- com a evidência que o decidiu e como fechar. É checklist, não nota: a letra A-F
  continua sendo a da curadoria e o portão continua sendo `manifests --check`; o selo é uma contagem com o que falta.
  O `experiment.json` ganha um bloco opcional `quality` (formato `pinneapple.quality.declaration/1`) em que a pasta
  declara `not_applicable` com motivo e `not_measured`; manifesto antigo segue válido e só mostra o que dá para ler.
  `python -m pinneapple_lab quality` imprime o selo de cada pasta, `-o lab/QUALITY.md` grava a tabela e `--backlog
  docs/quality_backlog.md` lista as lacunas ordenadas por visibilidade do exemplo (as páginas e guias que o citam).
  Medido no repositório: 177 pastas de `examples/`, em 0,3 s sem a varredura de visibilidade; a tabela e o backlog
  estão em `lab/QUALITY.md` e `docs/quality_backlog.md`.
