Um relatório de engenharia de uma página por experimento (#919): `pinneapple_analysis.report.experiment_report(folder)`
devolve `html, pdf` de uma pasta com `experiment.json`, com o mesmo arranjo para todos os experimentos — cabeçalho
(título, pergunta, data, commit, hardware, tempo de parede e custo), 3–6 cartões de KPI com intervalo, referência e
veredito, tabela de vereditos lida das checagens que o run gravou, as figuras que o manifesto nomeia, incerteza e
validade (o checklist de nove itens de `pinneapple_lab.quality` e o cartão de seis estágios de
`pinneapple_lab.curation`) e proveniência com o comando de reprodução. Não é um terceiro formato: cada seção é a
leitura de algo que o repositório já calcula, e o cartão de confiança sai verbatim de `lab/curation.json` quando a
pasta está no Lab. Um número que nenhuma checagem nomeia aparece como "no check", nunca como aprovado — a associação
entre métrica e checagem só vale quando um nome contém as palavras do outro, porque `hypotheses_refuted` não é
evidenciado por uma tabela de `T1 not refuted`. Pasta sem resultado levanta `ReportError` dizendo o que falta e qual
comando roda o experimento. `python -m pinneapple_analysis.report examples/condensation_falsification` escreve
`index.html`, `report.pdf`, `report.json` e `assets/`; `--out DIR` com várias pastas ainda escreve o índice `DIR/index.html`.
