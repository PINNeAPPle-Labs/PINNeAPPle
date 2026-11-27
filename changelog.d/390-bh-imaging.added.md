- **Imagem de buracos negros e lentes gravitacionais fortes como problemas inversos** (#390):
  `pinneapple_physics.astro.eht_imaging` constrói o lado da medição em torno das formas fechadas que o
  `astro.lensing` já tinha. O arranjo de 2017 (oito estações, 230 GHz, corte de elevação) gera 327 visibilidades em
  21 pares até 7,98 Gλ com ruído térmico da equação do radiômetro; a lua crescente geométrica do EHT (diâmetro, largura
  radial, modulação `1 + a cos(φ − pa)`) tem visibilidades em forma de Bessel fechada, conferidas contra a transformada
  direta de uma imagem renderizada em 2×10⁻⁷, e o ajuste aceita qualquer prior gaussiano como linha de resíduo — ligar
  e desligar o prior é mudar um argumento. Medido com 20 realizações: o diâmetro de anel de 42 µas volta em
  **42,029 ± 0,017 µas** (dispersão por realização 0,074; intervalo 95 % conformal ±0,191) e o diâmetro de sombra
  2√27 GM/c²D das massa e distância publicadas é 39,69 µas, 0,77σ abaixo do anel publicado. O erro publicado de ±3 µas
  **não** se reproduz e não pode: ele é dominado pela dispersão entre métodos de análise, não por ruído térmico, e o
  módulo mostra os dois números em vez de inflar o ruído até coincidirem.
- O mesmo módulo mede o que o prior faz, e não só que ele existe: `prior_dependence_study` ajusta as mesmas
  visibilidades com o prior desligado e ligado e reporta, por característica, o deslocamento emparelhado com seu erro,
  `moved` (deslocamento sobre a incerteza da própria característica) e `shrink` (quanto de precisão o prior comprou).
  Na sensibilidade da campanha nenhuma das cinco características é movida mais de 5 % da sua incerteza
  (`moved ≤ 0,05`, `shrink = 1,00`); com um centésimo do tempo de integração o diâmetro sai de 41,8 ± 3,7 para
  40,6 ± 1,5 µas (`shrink = 2,5`, `moved = 0,31`) e passa a ser *prior-influenced*. Com amplitudes apenas — o caso da
  modelagem geométrica do EHT — o ângulo de posição da assimetria volta em **−90,1 ± 2,3°** onde a verdade é +90°: a
  amplitude de um anel assimétrico é invariante ao sinal da assimetria, o erro formal de 2,3° não mostra os 180° de
  erro e nenhum prior deste estudo resgata a direção. Um raio traçado exato do disco fino (`trace_thin_disc` +
  `disc_image`, reescalado para massa e distância publicadas) tem sombra de 19,3 µas contra 19,84 analíticos e anel de
  **69,4 µas**: o anel da imagem não é a sombra, o que é a razão de os dois serem reportados separados.
- `pinneapple_physics.astro.lens_model` (metade de lentes do #390) ganhou os testes que faltavam e o conserto do
  palpite inicial do ajuste: uma lente de grupo (θ_E ≈ 5,3″) espalha o mesmo fluxo por um campo maior, o palpite de
  fluxo saía da caixa e `fit_lens_image` recusava rodar. As três lentes declaradas voltam com θ_E, q, ângulo de posição
  e ambos os componentes de cisalhamento a 10⁻⁵, com resíduo de arredondamento — porque as imagens são **geradas**, não
  observadas, e a página diz isso. Os testes fixam a convergência positiva (κ_min = 0,178 / 0,187 / 0,181), a equação
  da lente resolvida a 10⁻⁶″ e a multiplicidade contra as cáusticas (4 imagens dentro, 2 fora; 4 → 2 ao cruzar
  0,78 θ_E; 1 imagem em 1,6 θ_E); e fixam também o limite da convenção: em q = 0,95 a cáustica tangencial alcança só
  0,051 θ_E, então `consistency_report` devolve `inner_counts = [2, 4]` e `multiplicity_ok = False` em vez de passar em
  silêncio. Nem as visibilidades do EHT nem as amostras de lentes de HST/Euclid são redistribuídas ou baixadas:
  `data_requirements()` e os dois carregadores nomeiam release, arquivo, colunas e tamanho e falham com
  `NotConfigured`. Documentação com as tabelas medidas e a separação reproduzido/declarado/sintético em
  `docs/core_concepts/black_hole_imaging_lensing.md`.
