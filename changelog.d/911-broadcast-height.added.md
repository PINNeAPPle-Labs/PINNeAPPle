The broadcast ball height is repaired instead of the physics (#911): `pinneapple_systems.sports.football.ball` gains
`HeightMap` (a monotone map `z_true = f(z_feed)`, as a global scale, a pitch-dependent camera scale or a spline),
`fit_height_map`, which fits it **jointly over every flight of a training split with gravity fixed at 9.81** by
alternating the per-flight drag fits with a monotone regression of the feed height on the fitted one and one step
driven by the flights' own apparent gravity, and `identifiability_of`, which applies the Cramér-Rao bound so a drag
coefficient or a spin is reported only where the data identify it. `height_recovery_test` is the known-answer test:
simulated flights distorted by a known `f` give that `f` back. `ball.fit_flight` accepts `dt`, `p0` and `max_nfev`
(per-flight fits in a joint iteration are three times cheaper at the 10 Hz of a feed). The study is
`height_study.height_correction` with `examples/sports/06_height_correction.py`, and the verdict of the football
experiment gained the corresponding rows.
