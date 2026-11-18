- `pp.zoo.anchors` publishes the zoo as a named family instead of a catalogue: one anchor per engineering
  problem (`Anchor-Aero`, `Anchor-Thermal-Forced`, `Anchor-Thermal-Natural`, `Anchor-Contact`), each carrying the
  problem it answers, the simulation it replaces, the held-out metrics and validity range of its model, and the
  trust decision it was issued with. `anchors.catalogue()` renders the family page, `anchors.card(id)` one anchor
  and `anchors.public()` the JSON the site's `/modelos` page is built from. Every number is read from
  `manifest.json` and the model card, never restated, so the family cannot drift from what ships; `anchors.problems()`
  lists an anchor whose model is unapproved, has no held-out metric or has no validity range, and the tests fail on
  a non-empty list.
