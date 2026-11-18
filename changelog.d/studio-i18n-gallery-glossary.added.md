- Studio: the gallery and the glossary in Portuguese and English. Every gallery entry has a Portuguese title
  (`title_pt`, from `titulos` in `docs/pt/_mapa/prosa/_apps_exemplos.json`) and a description in both languages
  (the ten that had none in Portuguese and the six whose only source was Portuguese now have both), and every
  glossary term has its English `term` and `definition`; `tests/test_studio_i18n.py` fails if one goes missing.
