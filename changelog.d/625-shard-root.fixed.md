- O shard `root` (574 arquivos) foi remedido em oito pedaços e cada vermelho recebeu caixa e motivo (#625). Três
  defeitos de biblioteca, todos com teste que os reproduz: `ServingReport.speedup` procurava as linhas `torch` e
  `onnxruntime` enquanto o benchmark escreve `torch (cpu)` e `onnxruntime (cpu)`, então toda medição de verdade
  reportava `None`; o medidor de `scripts/example_catalog.py` rodava o script declarado pelo nome base e transformava
  em "exemplo que falha" todo exemplo que guarda os scripts num subdiretório (`examples/cfd`); e o sandbox do Hub
  aplicava sessão e limites num `preexec_fn`, que falha quando o chamador tem threads — o Hub valida de dentro das
  suas. Lacunas de registro também fecharam: `node:ugs.storage_season` sem categoria nem corrida no Hub,
  `recipe:pinneapple_plugins__0` fora de `RECIPES`, dois exemplos fora do catálogo e 17 pastas de `examples/` sem
  título nem descrição em português. As falhas que são dependência opcional ausente (meshio, pyarrow, imageio,
  trimesh) passam a pular com o extra no motivo, e as que mediam uma API que já mudou (mensagem do `build`, cartão sem
  proveniência, sonda do `physicsnemo.sym`, BLAS invisível ao `threadpoolctl`) passam a medir o que existe.
