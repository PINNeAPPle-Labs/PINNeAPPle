- **The arena replay** (`pinneapple_arena.battle.replay`, issue #373): the animation half of the battle, drawn from the
  record and from nothing else. One frame per step of `benchmarks/battle/results.json`: the error curve of both models
  filling up against the published tolerance, each model's health bar (its physics residual as a share of the alarm it
  was calibrated to), the step at which the reference proves each one left its domain, and the roster that ends with
  who is still standing -- so a GIF cannot state a number the table does not. `python scripts/battle.py --gif` redraws
  `docs/assets/battle/replay.gif` without training anything, and `--run` now rewrites the replay along with the record
  and the page, so the picture cannot go stale behind the numbers. The page says in as many words what the replay does
  not carry: the record holds two PDE cases, not the 2020 weather round of #348.
