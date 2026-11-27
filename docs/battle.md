# Battle of the models: the scoreboard (#373)

Two models, one stream of conditions, and one question that a leaderboard's mean error cannot answer:
**where did each of them stop being usable?** The table below is generated from
`benchmarks/battle/results.json` by `python scripts/battle.py --docs` -- do not edit it by hand.

It is the measured half of the arena the issue asks for: the numbers, the cost and the exit, not the
animation. What each model is allowed to see, what it costs, what it gets wrong at the end of the stream
and the step at which it left the domain it was validated on are all here, with the criterion written
down below so the exit can be recomputed from the curve.

## The protocol

* **seeds**: 0, 1 -- every model is trained once per seed; the error and the cost are reported as the mean over them, and the exit as the first seed that left (see *ranking*)
* **budget**: 300 epochs, lr 0.001, batch 8, identical for both models; the size knobs are each family's own (FNO {'layers': 4, 'modes': 8, 'width': 32}, PINN {'hidden': [64, 64, 64], 'pde_weight': 0.001})
* **grid**: 129 points in x, 41 steps in t
* **window**: 50% of the stream, `t <= 0.5` -- both models are trained on those steps only and scored on all of them
* **exit, the truth column**: the model left its domain at the first step outside the window where the relative L2 against the reference is above `0.05` -- the same tolerance this library uses to call a field reproduced, so the number reads the same for a model that was excellent in its window and for one that was mediocre
* **exit, the model's own column**: the residual alarm fires at the first step outside the window where the model's residual exceeds `4.0` times the value it had at `last step inside the window` -- a reference-free monitor, calibrated on the state the model was in when it left the window, which is the only calibration available without an answer to compare against
* **residual**: second-order central differences in x and t on the field the model returns; the same operator for every model, never the training loss
* **ranking**: the first step past the window where the error crossed the tolerance, over the seeds (the minimum, not the mean: a seed that survived the stream has no exit, and averaging the end of the stream in as a stand-in would let a model that left in one seed outrank one that never left); ties go to the model that left in more seeds, then to the smaller final relative L2, then to the cheaper training, then to the model name
* **cost**: wall time with torch.set_num_threads(1), after a warm-up pass, per seed; information, not a ranking key

## The scoreboard

| rank | model | cost (train / predict) | final relative L2 | left its domain (first seed to leave) | its own residual alarm (first seed to fire) |
|---|---|---:|---:|---|---|
| 1 | pinn | 17.4 s / 10.6 ms | 0.783 | step 21 (t = 0.53) | never (the monitor did not fire) |
| 2 | fno | 23.4 s / 90.2 ms | 0.257 | step 22 (t = 0.55) | never (the monitor did not fire) |

Case `burgers`: burgers_1d. Window: t <= 0.5 (the steps the models were trained on); the rest of the stream is the battle. 41 steps, 21 of them inside the window.

| 1 | pinn | 24.2 s / 49.4 ms | 0.082 | step 30 (t = 0.75), in 1 of 2 seeds | never (the monitor did not fire) |
| 2 | fno | 43.5 s / 192.5 ms | 0.042 | never (survived the whole stream) | never (the monitor did not fire) |

Case `heat`: heat_1d. Window: t <= 0.5 (the steps the models were trained on); the rest of the stream is the battle. 41 steps, 21 of them inside the window.

A model that never left any seed is placed at the end of the stream -- surviving is a result, not a missing
value. The order is total and recomputable from the record, which is what lets
`python scripts/battle.py --check` refuse to publish it out of order.

## Case `burgers` -- burgers_1d

* equation: `u_t + u u_x = nu u_xx, nu = 0.01, u(x, 0) = -sin(pi x), u = 0 at x = -1 and x = 1`
* reference: closed form (Cole-Hopf by Gauss-Hermite quadrature), scored on the whole stream, the window included
* window: t <= 0.5
* operator floor: the reference's own residual peaks at 0.346 (the `t` edges excluded), so a model's residual near that number is the grid, not the model

### fno -- neural operator

* sees: the window's field (data-supervised neural operator)
* 37,121 parameters
* cost: 23.4 s to train [20.5, 26.2] over the seeds, 90.2 ms to predict the whole stream
* relative L2, worst inside the window: 0.025; mean outside it: 0.151; last step: 0.257 [0.207, 0.308]
* first left its domain at step 22 (t = 0.55) (threshold 0.050); the seed spread of that step is [22, 24] and the seeds read seed 0: step 22, seed 1: step 24
* its own residual alarm never fired (threshold 3.161) while the reference column says the model left its domain: a monitor that stays silent on a model that left is a finding, not a blank

### pinn -- physics-informed network

* sees: the window's field + the governing equation (PDE residual in the loss)
* 8,577 parameters
* cost: 17.4 s to train [17.1, 17.6] over the seeds, 10.6 ms to predict the whole stream
* relative L2, worst inside the window: 0.189; mean outside it: 0.450; last step: 0.783 [0.678, 0.889]
* first left its domain at step 21 (t = 0.53) (threshold 0.050); the seed spread of that step is [21, 21] and the seeds read seed 0: step 21, seed 1: step 21
* its own residual alarm never fired (threshold 5.288) while the reference column says the model left its domain: a monitor that stays silent on a model that left is a finding, not a blank

### Where they disagree and the reference decides

At step 40 (t = 1.00) the two fields are furthest apart: the distance between them is 1.085 of the reference's norm, and at that step the errors are fno 0.308, pinn 0.889. The reference favours **fno**.

## Case `heat` -- heat_1d

* equation: `alpha T_xx = T_t, alpha = 0.05, T(x, 0) = sin(pi x), T = 0 at x = 0 and x = 1`
* reference: closed form: exp(-alpha pi^2 t) sin(pi x)
* window: t <= 0.5
* operator floor: the reference's own residual peaks at 2.615e-05 (the `t` edges excluded), so a model's residual near that number is the grid, not the model

### fno -- neural operator

* sees: the window's field (data-supervised neural operator)
* 37,121 parameters
* cost: 43.5 s to train [34.5, 52.4] over the seeds, 192.5 ms to predict the whole stream
* relative L2, worst inside the window: 3.033e-03; mean outside it: 0.019; last step: 0.042 [0.035, 0.049]
* first left its domain at never (survived the whole stream) (threshold 0.050): every seed survived, and that is the result, not an empty cell
* its own residual alarm never fired (threshold 2.858)

### pinn -- physics-informed network

* sees: the window's field + the governing equation (PDE residual in the loss)
* 8,577 parameters
* cost: 24.2 s to train [23.8, 24.7] over the seeds, 49.4 ms to predict the whole stream
* relative L2, worst inside the window: 0.013; mean outside it: 0.044; last step: 0.082 [0.046, 0.118]
* first left its domain at step 30 (t = 0.75), in 1 of 2 seeds (threshold 0.050); the seed spread of that step is [30, 41] and the seeds read seed 0: step 30, seed 1: never
* its own residual alarm never fired (threshold 0.258) while the reference column says the model left its domain: a monitor that stays silent on a model that left is a finding, not a blank

### Where they disagree and the reference decides

At step 40 (t = 1.00) the two fields are furthest apart: the distance between them is 0.151 of the reference's norm, and at that step the errors are fno 0.049, pinn 0.118. The reference favours **fno**.

## Reproducing it

```bash
python scripts/battle.py --run      # train both models over the stream, rewrite the results and the page
python scripts/battle.py --docs     # regenerate this page from the results file (no training)
python scripts/battle.py --check    # the gate: declaration, record and page agree, order included
```

The cost is a property of the machine and is published as information; a reproduction compares the
errors and the exits, never the seconds.

The run is seeded per model **and** held to one CPU thread (`_one_thread`), which is what makes the
errors comparable and the exit reproducible at all: torch splits a reduction over its threads and the
split changes the rounding, so on four threads two runs of this same code put the FNO's exit on
Burgers at step 23 and then at 22. With the split fixed at one, two runs of this record reproduced
every error curve bit for bit and the same exit for every model and seed.
