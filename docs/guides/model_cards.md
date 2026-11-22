# Model cards generated from an experiment record

The hub refuses a model card whose `validation_metrics` are empty, and refuses one whose provenance cannot be
re-derived. What it could not do is *produce* a card: the numbers were typed in from a run somebody remembered. This
generates the card from the record the run wrote, and validates it in the same call.

```
pinneapple model-card runs/burgers_pinn -o card.json --markdown card.md
```

Exit codes: `0` the card passed, `1` it was generated and refused (the reasons are printed), `2` there is no record.
A refused card is still written when `--out` is given — the reasons are the useful part, and throwing the draft away
hides which field the record could not fill.

```python
from pinneapple_hub.experiment_card import generate_and_validate

generated = generate_and_validate("runs/burgers_pinn")
generated.ok                  # True when it passes the hub's validation
generated.problems            # every reason it would be refused
generated.to_markdown()       # the card, with the refusals appended when there are any
```

## What it fills in

| Field | Where it comes from |
|---|---|
| `validation_metrics` | the record's `result.json`, flattened to `metric.field` (`relative_l2.u`) |
| `reference_source` | the `reference` the run was scored against and how many evaluation points |
| `cost` | **measured** `wall_time_s` and `n_eval_points`, plus the training options (`epochs`, `n_collocation`, ...) — and `trainable_parameters` when a model is passed in |
| `validity_range` | the problem's domain bounds and the observed field ranges, plus what it was compared against |
| `known_limitations` | what the record does not cover: outside the bounds the error is not measured, one seed is not a distribution over runs, and no units means the archive's own convention |
| `provenance` | `pinneapple_data.provenance.build_provenance`: solver, seed, problem fingerprint, commit and library version from the record's environment |

## What it refuses to invent

A record with no reference gets no `validation_metrics`, so no card is built at all — `ValueError` naming that the
run had nothing to score against is better than a card with a made-up number. A card missing `cost` or
`validity_range` is refused by `validate(require_cost=True, require_validity=True)`, which is what the generated
path always asks for: how long the answer took and where it is valid are part of the claim, not decoration.

The provenance rule is the same one the hub already enforced (#39): a card whose numbers cannot be re-derived is a
claim, not a record.
