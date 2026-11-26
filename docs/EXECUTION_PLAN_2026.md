# Plano de execução integrado — PINNeAPPle + Engineering API (2026)

> **Documento espelhado nos dois repositórios.** Fonte de verdade operacional compartilhada para agentes (Claude Code, Codex etc.). Atualizado em 21/11/2026. O cronograma é uma **fila nominal**, não uma garantia de 91 implementações validadas por dia. Não fechar issues sem evidência.

## Objetivo e fontes

- Período: **23/11/2026 a 16/12/2026**, 18 dias de segunda a sexta (validar feriados locais).
- Biblioteca: [PINNeAPPle](https://github.com/PINNeAPPle-Labs/PINNeAPPle), **1.240 issues abertas** marcadas `escopo: 2026` (de 1.254, 14 já fechadas). Fonte: [docs/backlog.md](https://github.com/PINNeAPPle-Labs/PINNeAPPle/blob/main/docs/backlog.md).
- Produto: [pinneapple-engineering-api](https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api), **387 issues de 2026** nas famílias A/B/C/F/G/H/K/L; 106 fora da prioridade. Fonte: [docs/product/CHECKPOINT_ISSUES.md](https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api/blob/main/docs/product/CHECKPOINT_ISSUES.md).
- Total inicial: **1.627**; média **90,39/dia útil**; distribuição nominal: **69 da biblioteca + 22 da API** nos primeiros 17 dias, e **67 + 13** no último.
- Congelamento do escopo: estes números são snapshot de 21/11. Novas issues e fechamentos posteriores exigem reconciliação; não assumir que uma issue da fila permanece aberta.

## Como chamar o agente

No Claude Code (com acesso autenticado aos dois repositórios), diga:

> Leia `docs/EXECUTION_PLAN_2026.md` integralmente. Execute o lote de **AAAA-MM-DD**. Antes de codificar, consulte as issues reais nos dois repositórios, cheque estados, critérios de aceite e dependências, e produza uma triagem por S/M/L, P0/P1/P2/P3, bloqueios e automação. Proponha grupos de PRs por componente e comece pelas tarefas desbloqueadas e verificáveis. Implemente, teste, registre evidências e feche somente as issues realmente concluídas. Ao final, reporte concluídas, parciais, bloqueadas, PRs, testes e saldo. Não altere o cronograma sem documentar a razão.

Para rodar só a triagem: substitua “Execute” por “Planeje sem alterar código nem fechar issues”. Para continuar: “Retome o lote pendente e replaneje o saldo com base no estado atual do GitHub”.

## Regras operacionais

1. **GitHub é a verdade atual:** reconsultar cada issue, labels, PRs relacionados, status e comentários. A lista abaixo identifica o snapshot; não sobrescreve o estado real.
2. **Classificação:** complexidade S (mudança isolada), M (múltiplos arquivos/testes), L (arquitetura, solver, benchmark ou pesquisa); prioridade P0 bloqueante, P1 essencial, P2 expansão, P3 exploratório; dependência livre/interna/externa; automação alta/parcial/baixa. Preservar labels existentes e registrar inferências.
3. **Ordem:** corrigir bloqueios e contratos compartilhados antes de features dependentes; priorizar testes, CI, Physics IR, registry, adapters e interoperabilidade; paralelizar somente lotes independentes.
4. **Lotes:** (A) correções/testes/docs/CI; (B) infraestrutura e APIs compartilhadas; (C) modelos e conectores; (D) experimentos e validação numérica; (E) produto, Studio e demos. Um PR pode cobrir múltiplas issues somente se cada aceite estiver comprovado.
5. **Validação:** executar testes unitários/integrados pertinentes; benchmarks físicos exigem referência, unidades, conservação, erro, seeds, dados e artefatos reproduzíveis; registrar limitações e custos. Nunca simular resultado de teste.
6. **Fechamento:** só após código integrado, critérios de aceite satisfeitos e evidência/PR vinculados; issue bloqueada permanece aberta com motivo, dependência e próximo passo. Não fechar em massa por quota.
7. **Segurança e controle:** não fazer deploy, publicar artefatos externos, criar despesas, modificar credenciais ou alterar proteções sem autorização; respeitar CI e revisão. Evitar PRs gigantes.
8. **Replanejamento diário:** ao fim do dia atualizar concluídas por repo, pendências, blockers, velocidade real por tamanho e estimativa de término. Reordenar próximos dias sem perder rastreabilidade do snapshot.
9. **Feriados:** 08/12 pode ser feriado municipal em Natal/RN; confirmar calendário da equipe antes de assumir os 18 dias.
10. **Métrica correta:** distinguir issues analisadas, iniciadas, PRs abertos, PRs integrados e issues fechadas/validadas.

## Cadência

- Início: atualizar checkout e estado das issues; escolher as desbloqueadas, com prioridade para P0/P1 e dependências.
- Execução: agrupar mudanças por módulo, implementar incrementalmente, rodar testes e revisão.
- Fechamento: anexar evidências e links, fechar somente as concluídas, manter registro de bloqueios.
- Checkpoints acumulados nominais: 27/11 **455** (345 biblioteca + 110 API); 04/12 **910** (690 + 220); 11/12 **1.365** (1.035 + 330); 16/12 **1.627** (1.240 + 387). São metas aritméticas, não previsões.
- Primeira semana: calibrar throughput real e revisar plano, priorizando qualidade e entrega de valor sobre contagem.

## Cronograma nominal — IDs de issues

**Formato:** cada intervalo `a-b` inclui todos os IDs inteiros de a até b; os IDs são específicos de cada repositório. Para consultar: `https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/ID` e `https://github.com/PINNeAPPle-Labs/pinneapple-engineering-api/issues/ID`. Os lotes abaixo foram particionados sem repetição dentro de cada repo. **Não executar cegamente pela ordem**: resolver dependências e redistribuir trabalho quando necessário.

### Dia 1 — 23/11/2026 (91 issues)

- **PINNeAPPle (69):** 31,33-35,38,40,46,52,53,57-59,61,62,68,70,75,77,79,83,84,93,101-106,110,136,152,153,155,156,159,167,171,172,201,205,209,215,225,227,265,272,281,294,295,447,494,515,528,625,911,917,918,923,927,933,1474,1477-1481,1494,1871,1891
- **Engineering API (22):** 27-38,44,87,97-100,102-105

### Dia 2 — 24/11/2026 (91 issues)

- **PINNeAPPle (69):** 51,54,60,63,65,67,78,80,96,100,112,114,115,131,132,134,161,173,177,179,202,204,206,207,210-212,216,219,221,231,243,245,254,269,278,292,293,296,297,315,322-324,327,492,513,522,644,645,677,919,920,929,936,967,1475,1476,1498-1500,1502,1507,1508,1679,1868,1883,1890,1893
- **Engineering API (22):** 106,107,110-116,119-121,124-133

### Dia 3 — 25/11/2026 (91 issues)

- **PINNeAPPle (69):** 328-341,343,345-382,384-399
- **Engineering API (22):** 134,136-143,145-147,149,152-154,165,171-174,176

### Dia 4 — 26/11/2026 (91 issues)

- **PINNeAPPle (69):** 400,401,403,405-411,413-419,421-424,426-433,435-446,455-462,464-469,471-473,475,477,478,482,485,489,499-501,503,506
- **Engineering API (22):** 177-181,188,192,193,197,202,211,216-220,226-231

### Dia 5 — 27/11/2026 (91 issues)

- **PINNeAPPle (69):** 511,512,516,517,519,524,530,532,539-541,543-551,553,555,556,560-563,566-568,570-572,574-594,601,603,604,614,619,620,626,634,635,640,664-668
- **Engineering API (22):** 232,234-236,239-242,244-248,251,253-256,258-261

### Dia 6 — 30/11/2026 (91 issues)

- **PINNeAPPle (69):** 669-673,675,676,686-694,777,903,905,908,909,912,914,915,921,922,932,935,943,944,946-950,954,959,960,962-964,976-978,992,993,1029,1032,1036-1039,1047,1050-1055,1065,1068,1072,1080-1086
- **Engineering API (22):** 262-264,267-272,274,278,280-283,285,293,295,297-299,307

### Dia 7 — 01/12/2026 (91 issues)

- **PINNeAPPle (69):** 1087,1089,1090,1094-1105,1107-1112,1122,1130,1131,1133-1135,1138,1141-1146,1150,1151,1161,1163-1166,1170,1185,1194-1198,1202-1204,1207,1210-1219,1221,1227-1232
- **Engineering API (22):** 309,310,325,328-335,337-339,341-343,345,347-350

### Dia 8 — 02/12/2026 (91 issues)

- **PINNeAPPle (69):** 1233,1234,1238,1240,1249,1252,1253,1257,1263-1285,1287,1288,1306,1314,1328,1330,1331,1338,1359,1362,1375-1377,1382-1385,1387-1396,1398-1401,1410,1413-1418
- **Engineering API (22):** 353-359,361-364,384,387-389,391-397

### Dia 9 — 03/12/2026 (91 issues)

- **PINNeAPPle (69):** 1419-1437,1441,1443-1445,1447,1451-1454,1456-1458,1464,1472,1473,1485-1488,1490,1492,1495,1503,1505,1506,1516,1521-1523,1526-1546
- **Engineering API (22):** 399-420

### Dia 10 — 04/12/2026 (91 issues)

- **PINNeAPPle (69):** 1547-1557,1559-1579,1581,1582,1584,1588,1589,1592-1595,1597,1600,1604,1605,1611,1612,1614,1615,1617,1618,1620,1621,1628-1632,1637,1638,1641-1643,1646,1650,1652-1655
- **Engineering API (22):** 421-429,431,433-439,442-446

### Dia 11 — 07/12/2026 (91 issues)

- **PINNeAPPle (69):** 1656-1659,1673,1674,1676-1678,1684-1686,1688,1692-1697,1700-1706,1708,1710,1711,1713,1714,1716-1720,1723-1725,1732,1738,1742,1744-1748,1750,1751,1753-1755,1759-1765,1781,1786,1788,1790,1792,1794,1796-1799
- **Engineering API (22):** 447-468

### Dia 12 — 08/12/2026 (91 issues)

- **PINNeAPPle (69):** 1800,1802,1804-1806,1808-1812,1815-1818,1820-1845,1847-1849,1858,1859,1863,1865,1876-1878,1881,1886-1888,1892,1913,1923-1928,1931,1934-1939
- **Engineering API (22):** 469,470,473,500-503,506-512,515-522

### Dia 13 — 09/12/2026 (91 issues)

- **PINNeAPPle (69):** 1940-1963,1965-1969,1971-1986,1988-1990,1993-1999,2001-2014
- **Engineering API (22):** 523-543,545

### Dia 14 — 10/12/2026 (91 issues)

- **PINNeAPPle (69):** 2016-2049,2055-2063,2074-2089,2093-2102
- **Engineering API (22):** 546-551,553-564,566-569

### Dia 15 — 11/12/2026 (91 issues)

- **PINNeAPPle (69):** 56,69,2103-2131,2137-2151,2161-2181,2199,2200
- **Engineering API (22):** 571-574,576-582,585-595

### Dia 16 — 14/12/2026 (91 issues)

- **PINNeAPPle (69):** 71,74,107,111,133,162-165,168,169,174,176,180,185,203,208,213,214,218,222,224,230,236,244,258,264,267,268,273,282,283,285,287,312-314,474,486,490,493,518,613,616,639,907,913,924,931,934,1289-1291,1297,1310-1312,1325,1327,1493,1496,1497,1501,1504,1509,1514,1515,1860,1864
- **Engineering API (22):** 596-604,650-654,656-660,672-674

### Dia 17 — 15/12/2026 (91 issues)

- **PINNeAPPle (69):** 73,82,116,119,135,217,223,248,257,259,260,270,305,479,480,483,484,487,488,498,504,505,507,508,510,514,520,531,534,535,642,910,916,937,1304,1308,1309,1315,1329,1332,1333,1335,1336,1341,1342,1348,1350,1351,1353-1355,1357,1358,1364,1367,1368,1370-1372,1374,1511,1512,1518,1681,1687,1690,1869,1870,1879
- **Engineering API (22):** 675-686,696,697,702-709

### Dia 18 — 16/12/2026 (80 issues)

- **PINNeAPPle (67):** 86,87,117,118,166,175,184,220,229,246,249,288,298,476,481,491,495,502,509,521,529,533,615,618,636-638,641,643,647,1292-1296,1298,1299,1301,1302,1305,1307,1316,1317,1319,1321,1322,1324,1326,1334,1337,1339,1340,1343-1346,1363,1365,1366,1369,1517,1680,1682,1683,1867,1880,1894
- **Engineering API (13):** 710-722

## Modelo de relatório de execução

```markdown
Data: YYYY-MM-DD
Issues planejadas: biblioteca N / API N
Triadas: biblioteca N / API N
Concluídas com evidência: biblioteca N / API N
Parciais: [repo#issue, PR, progresso, próxima ação]
Bloqueadas: [repo#issue, dependência, responsável, próximo passo]
PRs: [URLs, status de CI/review]
Testes e benchmarks: [comandos, resultados reais, evidências]
Mudanças de escopo/ordem: [motivo e destino]
Saldo 2026 atualizado: biblioteca N / API N / total N
Projeção revisada: [dias úteis restantes, throughput observado]
```

## Manutenção do espelho

Este documento é intencionalmente idêntico nos dois repositórios. Qualquer revisão de cronograma, contagem, critérios ou instruções deve ser aplicada a **ambos** no mesmo ciclo; documentar divergências temporárias. Não editar os snapshots gerados de backlog manualmente.
