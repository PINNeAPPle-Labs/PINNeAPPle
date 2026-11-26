`pp.viz.detail` gera os detalhes mecânicos que uma malha de solver não tem: parafuso sextavado com arruela
(`hex_bolt`), círculo de parafusos (`bolt_circle`), cordão de solda ao longo de uma polilinha (`weld_bead`) e anel de
costura de fundição (`seam_ring`). Devolvem `Surface` comuns — exportam para glTF/USD e aceitam qualquer material da
biblioteca. Com a curva de interseção do ramal, o cordão segue a sela em vez de um círculo.
