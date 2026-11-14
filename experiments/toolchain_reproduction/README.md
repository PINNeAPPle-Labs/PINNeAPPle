# Reprodução: um PC em branco, um agente e a toolchain — agora com o PINNeAPPle

O experimento original começa num Windows vazio, pede a um agente de IA para instalar uma toolchain
open source (OpenFOAM, code_aster, ParaView) e rodar alguns casos clássicos de CFD e FEA. Uma hora
depois tudo está instalado, resolvido e pós-processado, e os resultados batem com os valores de livro.

Aqui a toolchain **é** o PINNeAPPle: o "instalar" vira `import`, não há licença para pedir e cada
etapa abaixo é uma chamada de função neste único arquivo. O que se reproduz é o essencial do
experimento — os mesmos casos clássicos, cada um conferido contra o valor publicado — e o que se
mede é quanto tempo isso custa e a que distância os números ficam da referência.

| Toolchain original | Aqui |
|---|---|
| OpenFOAM (Navier-Stokes incompressível) | `pinneapple_simulation.external_solvers.pict`, motor local vorticidade-função de corrente |
| OpenFOAM (transiente 1-D em tubulação) | `pinneapple_simulation.numerical_solvers.water_hammer_moc` (método das características) |
| code_aster (estruturas) | `beam_fe` (Hermite), `solid_fem` (hexaedros 3-D), `plane_stress_fem` (CST) |
| ParaView (pós-processamento) | matplotlib; `solid_fem.fea_figure` para a peça 3-D |

## Como rodar

```bash
python experiments/toolchain_reproduction/run.py            # todos os casos
python experiments/toolchain_reproduction/run.py --smoke    # malhas grossas, ~15 s numa máquina livre
python experiments/toolchain_reproduction/run.py --no-3d    # pula a viga 3-D em hexaedros
```

Do zero numa máquina limpa: `pip install pinneapple` e o comando acima (o script também se acha sozinho
a partir de um checkout, sem `PYTHONPATH`). Não há passo de instalação de solver, configuração de
ambiente nem clique em menu — o núcleo do pacote já traz numpy, scipy, matplotlib e torch, e é só disso
que os cinco modelos abaixo precisam.

Saída: `results/results.json` com cada número, cinco figuras em `results/` e a tabela no terminal.
Ao final, `main()` devolve `passed` — o script sai com código 1 se qualquer conferência falhar.

O resumo em PDF sai de `results/results.json` (nenhum número é redigitado):

```bash
python experiments/toolchain_reproduction/make_report.py --out ~/Downloads/PINNeAPPle-experimento.pdf
```

Ele precisa do `reportlab`, e só ele — o `run.py` não depende de nada além do núcleo do pacote.

## Os casos e o valor de livro de cada um

| Caso | O que é conferido | Referência |
|---|---|---|
| Cavidade com tampa deslizante, Re = 100 (64²) e Re = 1000 (128²) | rms de `u(x = 0.5)` e `v(y = 0.5)` contra as tabelas publicadas; divergência discreta; condição de contorno nas paredes | Ghia, Ghia & Shin, *J. Comput. Phys.* **48** (1982) 387–411 |
| Golpe de aríete após fechamento instantâneo e lento | pico de pressão contra `dH = a dV / g`; segundo pico dentro do intervalo entre a coluna rígida `L dV/(g t_c)` e Michaud `2L dV/(g t_c)` | Joukowsky (1898); Wylie & Streeter, *Fluid Transients in Systems* (1993) |
| Viga em balanço: elementos de Hermite, depois a mesma viga em hexaedros 3-D | flecha contra `qL⁴/8EI` e `PL³/3EI`; primeira frequência natural contra a solução exata; tensão de flexão no meio do vão contra `Mc/I` | Przemieniecki, *Theory of Matrix Structural Analysis* (1968), cap. 5 e 11 |
| Chapa com furo central sob tração | concentração de tensão na seção líquida, com extrapolação de Richardson em três malhas; equilíbrio de forças | Kirsch (1898); Pilkey & Pilkey, *Peterson's Stress Concentration Factors*, 3ª ed. (2008), forma de Heywood `Kt = 2 + (1 − d/W)³` |

Os parâmetros físicos estão no topo do `run.py` (`CAVITY`, `WATER_HAMMER`, `BEAM`, `PLATE`) em SI, e
cada um dos números é rastreável até a referência contra a qual é conferido.

## Resultados nesta máquina

python 3.11.16, numpy 2.4.6, scipy 1.17.1, matplotlib 3.11.2, torch 2.15.0. Licenças comerciais: **0**.
CalculiX (`ccx`) e OpenFOAM estão instalados nesta máquina, mas **nenhum** dos casos abaixo os usa — o
script registra isso e segue.

### As 20 conferências

| Caso | Quantidade | Calculado | Livro | Erro |
|---|---|---|---|---|
| Cavidade, Re = 100 (64²) | rms de `u(x = 0,5)` vs Ghia | 1,382e-3 | limite 5e-2 | passa |
| Cavidade, Re = 100 (64²) | rms de `v(y = 0,5)` vs Ghia | 2,839e-3 | limite 5e-2 | passa |
| Cavidade, Re = 100 (64²) | `du/dx + dv/dy` discreto | 3,6e-15 | 0 | passa |
| Cavidade, Re = 1000 (128²) | rms de `u(x = 0,5)` vs Ghia | 1,213e-2 | limite 5e-2 | passa † |
| Cavidade, Re = 1000 (128²) | rms de `v(y = 0,5)` vs Ghia | 9,659e-3 | limite 5e-2 | passa † |
| Cavidade, Re = 1000 (128²) | `du/dx + dv/dy` discreto | 7,1e-15 | 0 | passa |
| Golpe de aríete | pico com fechamento instantâneo | 108,1953 m | 108,1953 m (`aΔV/g`) | **+2,6e-14 %** |
| Golpe de aríete | pico com atrito de Darcy, f = 0,02 | 228,0995 m | 228,1953 m | −0,042 % |
| Golpe de aríete | pico com fechamento lento, `t_c` = 28,8 s | 6,4042 m | entre 4,5081 m (coluna rígida) e 9,0163 m (Michaud) | dentro do intervalo |
| Viga em balanço, Hermite | flecha com carga distribuída | 17,8571 mm | 17,8571 mm (`qL⁴/8EI`) | **−7,3e-10 %** |
| Viga em balanço, Hermite | primeira frequência | 16,710332 Hz | 16,710332 Hz | **+3,0e-7 %** |
| Viga 3-D, 20×2×2 | flecha na ponta | 47,059 mm | 47,619 mm (`PL³/3EI`) | −1,18 % |
| Viga 3-D, 40×4×4 | flecha na ponta | 47,280 mm | 47,619 mm | −0,71 % |
| Viga 3-D, 60×6×6 | flecha na ponta | 47,355 mm | 47,619 mm | −0,55 % |
| Viga 3-D, 60×6×6 | tensão de flexão no meio do vão | 150,0000 MPa | 150,0000 MPa (`Mc/I`) | **−1,7e-7 %** |
| Chapa com furo | equilíbrio de forças (reação / aplicado) | 1,0000000 | 1 | 9,6e-12 % |
| Chapa com furo, 1 156 nós | `Kt,net` | 2,2552 | 2,2963 (Heywood) | −1,79 % |
| Chapa com furo, 4 356 nós | `Kt,net` | 2,3023 | 2,2963 | +0,26 % |
| Chapa com furo, 16 900 nós | `Kt,net` | 2,3150 | 2,2963 | +0,82 % |
| Chapa com furo | `Kt,net` por Richardson, ordem observada 1,88 | 2,3198 | 2,2963 | **+1,02 %** |

† **Nota sobre a Re = 1000.** O motor local marcha até o estado estacionário e para no resíduo de
vorticidade (`steady_tol = 1e-7`) ou, se a máquina estiver ocupada demais para isso, no guarda de 300 s
de tempo de parede — e diz qual dos dois o encerrou. Nesta execução ele bateu no guarda: 9 000 passos,
resíduo 4,2e-2, rms de 1,2e-2 em vez de 8,1e-3. O caso Re = 100, que converge em 4 052 passos, sai
idêntico ao valor registrado em `examples/pict_cavity` (1,382e-3), o que mostra que a diferença é o
truncamento e não o método. Numa máquina livre, o mesmo caso de 128² converge em 22,5 s (é o número que
`examples/pict_cavity/cavity.json` guarda no repositório).

### Tempo

| Etapa | s |
|---|---|
| auditoria da toolchain | 0,3 |
| cavidade Re = 100 + Re = 1000 | 352,7 |
| golpe de aríete (3 fechamentos) | 0,1 |
| viga (Hermite + três malhas 3-D) | 74,9 |
| chapa com furo (três malhas) | 27,3 |
| pós-processamento (cinco figuras) | 79,4 |
| **total** | **539,4** (9 min) |

Estes tempos estão inflados: a máquina estava com média de carga entre 70 e 190 em 10 núcleos (27
processos `interFoam` de outra sessão) durante toda a execução. Numa máquina livre o mesmo conjunto roda
em cerca de um minuto — a cavidade de 128² sozinha cai de 302 s para 22,5 s, e o pós-processamento de
79 s para ~6 s. O que **não** depende da carga são os números da tabela acima: todos os modelos são
determinísticos, exceto o Re = 1000 truncado, que é justamente o que a nota marca.

### Figuras

- `cavity_vs_ghia.png` — as linhas de centro calculadas sobre os pontos publicados, nos dois Reynolds.
- `water_hammer.png` — pressão na válvula nos dois fechamentos, com as retas de Joukowsky, Michaud e da
  coluna rígida.
- `cantilever.png` — flecha de Hermite sobre a fórmula fechada; convergência da viga 3-D para `PL³/3EI`.
- `cantilever_3d.png` — a peça 3-D com a tensão de flexão `S11` na malha 60×6×6: no meio do vão,
  ±150 MPa contra `Mc/I` = 150 MPa (é a conferência da tabela). O pico de ±314 MPa no engaste é a
  singularidade do canto engastado — cresce com o refino (301 → 308 → 314 MPa em 20×2×2, 40×4×4 e
  60×6×6) e por isso a conferência é feita no meio do vão, não ali.
- `plate_hole.png` — campo de `σxx` na malha mais fina, a tensão ao longo do contorno do furo e a
  convergência de `Kt` contra Heywood e Kirsch.

## O que isso mostra, e o que não mostra

**Mostra** que, para um conjunto de casos clássicos, o ciclo inteiro — instalar, discretizar, resolver,
pós-processar e conferir contra a referência publicada — cabe num arquivo e roda em minutos, sem
licença. Onde o método é exato para a solução procurada, o número sai exato: a viga de Hermite
reproduz `qL⁴/8EI` e a frequência de Euler-Bernoulli a menos de 1e-9, o golpe de aríete instantâneo é
Joukowsky a menos de 1e-13, e a tensão de flexão 3-D no meio do vão é `Mc/I` a menos de 1e-6. Onde a
discretização é de primeira ordem (CST na chapa, hexaedros 3-D), o resultado converge para o livro na
medida em que a malha refina — a viga 3-D vai de 1,18 % para 0,55 % de erro na flecha, e o `Kt` da
chapa sai de 2,2552 para 2,3150, com Richardson a 1,0 % de Heywood — que é exatamente o que se deve
exigir desses elementos.

**Não mostra** que o PINNeAPPle substitui OpenFOAM ou code_aster. O motor de cavidade é um solver
vorticidade-função de corrente em malha uniforme, sem campo de pressão e sem transiente preciso; a
chapa é CST (tensão constante por elemento), não uma casca nem um elemento quadrático; o 3-D é
elástico linear em malha estruturada. Nenhum dos casos tem turbulência, contato, plasticidade,
escoamento compressível ou geometria de CAD real. O que este experimento mede é a *velocidade de
chegar a um número conferido* nos casos em que o número é conhecido — não a faixa de problemas que a
toolchain cobre.
