# GovernedCard

**A model card that cannot be issued without a physics check of the model behind it.**

Imported from PINNeAPPle-apps (`physics_verified_model_cards`). The rule is `pinneapple_hub.verified_card`
(`certify_model`, `card_from_guardrail`): train a model of a registered preset (or load uploaded `state_dict`
weights), run `PhysicsGuardrail`, and build a `pinneapple_hub.ModelCard` whose `validation_metrics` are only that
run's numbers (numeric checks by name, pass/fail checks as `<name>_passed`), with `reference_source` naming what
ran on how many fresh points. A card that fails `ModelCard.validate()` is refused, the same rule `push_to_hub`
applies. Related apps: `apps/model_gateway` and `apps/hub` publish and serve models with their cards;
`apps/validation_lab` validates models in depth with an evidence ledger. GovernedCard is the narrow "train or load,
guardrail, card or refusal" service.

## Run

```bash
pip install -e . -r apps/governed_card/requirements.txt
cd apps/governed_card && uvicorn governed_card.api:app --port 8148
curl -F preset=laplace_2d -F name=laplace-card -F train_epochs=300 localhost:8148/v1/generate_verified_card
curl localhost:8148/v1/compliance_report/<id>          # the job: card, every check, value, threshold, time
curl -X POST localhost:8148/v1/audit_card -H 'Content-Type: application/json' -d @card.json
```

Logins `GCARD_USER` / `GCARD_PASSWORD_HASH`; `GCARD_MAX_EPOCHS` (3000), `GCARD_FREE_CARDS` (20), `GCARD_FREE_AUDITS`
(1000). The metering stub and in-memory card store of the source are the appkit job queue and meter (SQLite on the
data volume). Deployed as `gcard` in `apps/deploy`.

## Tests

`pytest tests/test_hub_verified_card.py tests/test_app_governed_card.py` (3 + 2 pass; the app's identically seeded
"card metrics == direct guardrail run" test with `PINNEAPPLE_SLOW=1`, the same check runs fast in the library test).
The source tests are ported: metrics reproduced bit for bit by an independent, identically seeded run; an empty
card refused by the audit; a well-formed card accepted.

## Known limitations

- `laplace_2d` has a tagged boundary: the library now refuses to train a tagged preset without geometry, so the
  card uses the preset's boundary fixture (`presets.tag_geometry.tag_solve_kwargs`). The source app trained it
  without boundary points; its quoted residual (0.153 after 250 epochs) is not comparable.
- A short training usually leaves the card "not trustworthy": the card is still issued, with the failing value in
  `validation_metrics`. Issued means checked, not passed.
- `/v1/audit_card` is schema-only: it cannot re-run a card's physics without the weights and preset kwargs.
  No signer identity or post-market monitoring plan (EU AI Act Annex IV) is recorded.
