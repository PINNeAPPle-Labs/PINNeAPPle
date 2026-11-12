# Válvula globo com gás real, 80 → 60 bar

O caso de referência do pipeline compressível: uma válvula globo de metano com o disco a 4 mm da sede, 80 bar
de estagnação na entrada, 60 bar na saída e paredes adiabáticas. É o caso em que o gás real importa — a
laminação isentálpica esfria o metano em ~7 K, e um gás ideal daria 0 K.

```bash
python examples/cfd_valvula_globo/run.py --rapido   # malha grossa e janela curta (≈10 min)
python examples/cfd_valvula_globo/run.py            # o caso do relatório (≈1 h nesta máquina)
```

Saída em `_out/`: os casos do OpenFOAM (um por estágio), `resumo.json` com as checagens, as figuras do
matplotlib, os renders 3-D (um por campo) e o vídeo. Requer OpenFOAM nativo; sem Blender/ffmpeg o estudo roda
e as figuras do matplotlib saem.

## O que este exemplo ensina

1. **O domínio não é um sólido de revolução.** O corpo tem o pescoço da haste num eixo perpendicular ao
   escoamento, e a malha hexagonal do `passage` revolve um perfil em torno do eixo do fluxo — ela não tem
   esse ramo. O domínio é construído como sólido (`solid3d`) e malhado em hexa-dominante com `snappyHexMesh`.
   Medido: tetraedros (71°/22° de não-ortogonalidade) e o dual poliédrico (67 faces invertidas) travam o
   solver de pressão em 1000 iterações; a malha hexa-dominante (45°/10,6°) converge em 3.
2. **A entrada é `totalPressure` com `totalTemperature`.** O par pressão total + temperatura *estática* vaza
   entalpia de estagnação (+1,93 % numa passagem adiabática); o par correto deixa +1,20 %.
3. **A rampa de pressão.** Um passo único de 80 bar sobre um domínio parado sub-expande o gás e a inversão
   T(h) do Peng-Robinson estoura. A pressão de estagnação sobe 62 → 70 → 80 bar, cada estágio aquecido pelo
   campo do anterior.
4. **A média no tempo é o resultado.** A janela de média decide o número: com 3,5 ms de tempo físico o
   desbalanceamento de massa é 2,1 %; com 27,5 ms e média no último quarto, 0,22 %. E o ΔT medido vai de
   −1,7 K a −7,15 K — contra −7,45 K da referência isentálpica.

## Os números medidos

| grandeza | valor |
|---|---|
| vazão | 2,042 kg/s (131,1 m³/h) |
| Kv | 7,95 (IEC 60534-2-1: 8,84; Cd 0,497) |
| Δp estático | 15,28 bar |
| ΔT | −7,15 K (referência isentálpica −7,45 K) |
| Mach máximo | 0,525 — não afogado |
| ρ na entrada | 56,12 kg/m³ (Peng-Robinson a 0,10 %; o gás ideal daria 48,4) |
| desbalanceamento de massa | 0,22 % (critério 0,1 %) |

O procedimento completo, com as checagens e as ressalvas, está em
`PINNeAPPle-CFD/casos/cfd08-valvula-globo/RELATORIO.md`; o relatório para o cliente, em
`RESPOSTA_AO_ALUNO.pdf` no mesmo diretório. Guia do pipeline: `docs/guides/compressible_passages.md`.
