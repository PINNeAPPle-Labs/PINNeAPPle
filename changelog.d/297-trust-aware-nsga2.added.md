- **Trust-aware survivor selection for NSGA-II (#297, epic #343).** The verification-budget loop of
  `pinneapple_analysis.surrogate_loop` and the NSGA-II of `pinneapple_design.design_optimizer` are now connected through
  one decision: what a survivor rule does with a candidate the surrogate cannot vouch for. The decision is written, and
  measured, in `pinneapple_design.design_optimizer.trust_survivors`:

  * **the trust verdict is a constraint, not a third objective.** Measured on the 2-D airfoil front with the ensemble
    surrogate, appending the severity as a third objective leaves 79 % of the returned population unvouched and drops
    the honest share of the recommendations from 0,479 (plain NSGA-II) to 0,333 in the regime whose campaign covered
    only thick sections; as a constraint the same regime reaches 0,938.
  * **inside the channel, a separate band ahead of the measured violations.** `TrustVerdict` orders candidates
    `trusted`, `undetermined`, `untrusted`; each band is filled by the same `constrained_select` (Deb's
    constrained-domination rule, now in `survivors.py` and shared with `_nsga2_select`, so the two cannot drift).
    `trust_rule="penalty"` (the sum, which `pinneapple_design.aero.optimize` does today with `+0.5` for low trust) and
    `"remove"` are kept and measured, and the tests assert the failure mode that rules the sum out: a vouched design
    that breaks the lift floor does lose its slot to an unvouched one that does not.
  * **a candidate too uncertain to judge is kept, in the middle band.** It is never dropped — its interval narrows as
    soon as the verification answer is in the training set — and `SearchResult.front(vouched_only=True)` is what the
    caller sees, so an unvouched design is not handed over as a front point.
  * **off unless the caller declares the accuracy the decision needs** (`tolerance=None` is plain NSGA-II, and no
    test, app or workflow changes). With a surrogate that does not lie the channel is a pure cost: with the Gaussian
    process on the corner regime it buys nothing and costs 21 % of hypervolume.

  Headline measurement (`benchmark_trust_survivors`, three seeds, `mu = 16`, twelve generations, identical solver
  calls in every arm): the honest fraction of the verified final population goes from **0,542 to 0,750** in the
  covered regime and from **0,479 to 0,938** in the corner regime, per seed and not only on the mean, at a cost of
  **3,4 %** and **12,4 %** of hypervolume; the vouched hypervolume — the part of the front whose predictions held —
  does not fall (14,65→14,79 and 12,69→12,76 x 1e-5). The three mechanisms are within the seed noise of each other,
  which is reported rather than tuned away. Tests: `tests/test_trust_survivors.py`.
