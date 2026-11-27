O caso da placa plana do TMR passa a ser comparado com a **referência do CFL3D**, e não só com a lei de 1/7 potência:
as duas tabelas publicadas por modelo (`sa-cf_cfl3d.dat`, `sst-cf_cfl3d.dat`) entram no repositório com a procedência
e `tmr.read_reference` passa a mapear as colunas **pelo nome que o arquivo declara** — os arquivos por modelo põem
`cf` primeiro e `retheta` depois, o inverso dos arquivos de teoria, e um leitor por posição compararia `cf` com `cf`.
A abscissa da referência é `Re_theta`, então a solução é colocada nela pela espessura de momento integrada do próprio
campo (`tmr.momentum_thickness`), não por correlação.

A coluna de erro que a #265 pede também entra, contra as tabelas de **`cf` na própria estação** `x = 0,97008` em que o
CFL3D e o FUN3D publicaram cinco malhas aninhadas — e as dos códigos **incompressíveis** (FUN3D, SC/Tetra, GHOST), que
são a comparação próxima de um `simpleFoam`. `tmr_reference` lê esses arquivos, extrapola a família com o mesmo
`richardson_extrapolate` do caso (reproduzindo a ordem aparente e o erro extrapolado que a página do TMR declara) e
devolve o erro por código, com a lista de diferenças de implementação OpenFOAM × TMR. `tmr.study` passa a usá-la por
padrão; medido: kOmegaSST fica a −0,2 % do CFL3D e o SpalartAllmaras a +10,4 %, os dois na mesma malha e com as mesmas
funções de parede.
