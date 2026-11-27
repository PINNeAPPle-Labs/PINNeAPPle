- O shard `analysis` deixa de ficar vermelho por dependência opcional ausente (#625): os dois arquivos de teste de
  geometria e de qualidade de malha importam `build_mesh`, que é trimesh, e o `trimesh` vive nos extras `geometry` e
  `cfd`. Sem o extra, 13 testes morriam no import — o que, no relatório do shard, é indistinguível de um classificador
  quebrado. Agora eles pulam com o motivo e a linha de instalação; com o extra instalado, rodam como antes.
