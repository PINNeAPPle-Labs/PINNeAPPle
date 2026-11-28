# O que usar do PhysicsNeMo em vez de reescrever, e o que a Luminary publica de aberto

Levantamento de 2026-11-28, feito por verificação — cada linha de "existe" foi checada com `import` no ambiente
local, não copiada de documentação.

## 1. PhysicsNeMo: instalado e importável aqui

`physicsnemo 2.2.2` já está no `/opt/homebrew/bin/python3.11`. Modelos que **importam hoje**:

| módulo | classes | serve para |
|---|---|---|
| `physicsnemo.models.domino` | `DoMINO` | a arquitetura de aerodinâmica externa que a Luminary recomenda; opera em STL direto, prevê Cp e WSS na superfície e campos no volume |
| `physicsnemo.models.geotransolver` | `GeoTransolver`, `GALE`, `GALEStructuredMesh3D` | o transformer geométrico, o segundo que eles usam |
| `physicsnemo.models.transolver` | `Transolver` | a base do nosso `PointAttention` — vale comparar contra a implementação de referência |
| `physicsnemo.models.meshgraphnet` | `MeshGraphNet`, `HybridMeshGraphNet`, `BiStrideMeshGraphNet` | malha → campo; é o que o nosso `physicsnemo_bridge` já usa |
| `physicsnemo.models.fno` | `FNO`, `FNO1D/2D/3D/4DEncoder` | operador espectral, útil como baseline barato |
| `physicsnemo.models.graphcast`, `afno`, `pangu`, `fengwu`, `dlwp`, `dit`, `unet`, `mlp`, `rnn`, `srrn` | — | fora do nosso escopo (clima, difusão, séries) |

O que o `pinneapple_neural/physicsnemo_bridge/` já faz — e portanto **não se reescreve**:

- `active_learning_bridge.py`: o padrão "PINNeAPPle escolhe os pontos por incerteza, PhysicsNeMo retreina";
- `adapters.py`: conversão de dados/estado entre os dois;
- `field_operators.py`: operadores de campo compartilhados;
- `fno_compare.py`, `mgn_compare.py`: comparações já escritas contra FNO e MeshGraphNet.

**Conclusão prática:** DoMINO, GeoTransolver e Transolver entram por `import`, não por implementação. O que falta não é
código de arquitetura — é (a) o carregador do formato de dados deles, (b) o laço de treino com o nosso split e o nosso
card, e (c) GPU: DoMINO em CPU é proibitivo, em MPS não é suportado de forma confiável. Ou seja, o caminho é treinar
esses três onde houver GPU (nuvem pontual) e trazer os pesos para o nosso pipeline avaliar com os mesmos gates.

### O formato de dados deles é o do DoMINO

Cada amostra é um diretório:

```
sample_000001/
  merged_surfaces.stl          geometria (superfície, sem volume)
  merged_surfaces.vtp          campo na superfície (Cp, WSS por vértice)
  merged_volumes.vtu           campo no volume (só no dataset completo, não no -sample)
  params.json                  condições de operação
  forces.json                  forças e momentos integrados
  metadata.json                proveniência da simulação
```

O `.vtp`/`.vtu` são VTK com `numpy`/`meshio` (já temos `pinneapple_design/geometry/io/meshio_bridge.py`,
`trimesh_bridge.py`, `stl.py`). Um carregador de 1 arquivo põe qualquer dataset SHIFT dentro do nosso formato de
design space — é o que viabiliza o comparativo da §3.

## 2. O que a Luminary publica de aberto

Organização **`luminary-shift` no Hugging Face**, verificada pela API: **15 datasets, 0 modelos** (os pesos ficam na
plataforma deles, não são baixáveis).

| dataset | o que é |
|---|---|
| `WING`, `Wing-sample` | asa transônica (CRM), 3 000+ RANS no completo; o `-sample` traz poucas amostras |
| `SUV`, `SUV-sample` | aerodinâmica de SUV, >1 000 DES transientes |
| `Truck` | picape, >700 DDES em malhas de 122 M células |
| `SHIFT-Crash`, `SHIFT-Crash-sample` | impacto frontal, ~5 000 simulações |
| `Battery`, `Battery-sample` | placa fria de bateria, >1 300 CHT |
| `Pump`, `Pump-sample` | bomba |
| `CCA`, `CCA-sample` | aerodinâmica de caça |
| `Submarine`, `Submarine-sample` | hidrodinâmica |

**Licença: `cc-by-nc-4.0`** em todos. Consequência que precisa estar escrita antes de qualquer coisa ser treinada em
cima: **não comercial**. Serve para benchmark, pesquisa e comparação pública; não serve para entregar modelo a cliente.
Um modelo treinado nesses dados não pode ser vendido — e o card precisa registrar isso.

Não há problemas/geometrias soltos além dos que vêm dentro dos datasets (a geometria é o `merged_surfaces.stl` de cada
amostra), e não há código de solver publicado.

## 3. Como isso vira o comparativo que decide se o nosso LPM funciona

Mesmo dado, três modelos, mesmos gates:

1. **Entrada**: `Wing-sample` (ou `SUV-sample`) → carregador → o nosso formato de design space
   (`points`, `normals`, `query`, `params`, `target`), com split **por geometria** e manifesto com SHA-256.
2. **Modelos**: (a) o nosso LPM, (b) DoMINO do PhysicsNeMo, (c) GeoTransolver do PhysicsNeMo — os dois últimos pelo
   `physicsnemo_bridge`, treinados onde houver GPU.
3. **Prova**: os mesmos quatro gates do VeriPhysics (accuracy, conservation, boundary, residual), mais o baseline de
   campo constante, mais o OOD calibrado.
4. **Saída**: uma tabela única — nosso LPM × DoMINO × GeoTransolver × baseline — com o veredito de cada um por gate.

O que essa tabela responde: **se o nosso LPM, na mesma distribuição de dado que o deles usa, fica à altura das
arquiteturas deles.** Se ficar, o nosso pipeline é o produto e as famílias internas são a prova de reprodutibilidade. Se
não ficar, o número diz em que gate perde (provavelmente residual e conservação, não accuracy) — o que é informação
melhor do que "0,36 contra 0,95".

Ressalva honesta: os SHIFT são dados de DES/RANS de alta fidelidade no formato de superfície + volume, e o nosso modelo
foi desenhado para nuvem de pontos com campo na superfície. O comparativo mede o modelo **naquele** dado, não promete
que ele faz volume.
