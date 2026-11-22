# Reproducing a recorded run: reproduced, divergent or refused

A run recorded six months ago is not reproducible by re-running it. The package moved, the machine changed, and the
re-run prints a number that looks like the old one without saying whether it is the same run. `pp.experiments.reproduce`
answers that question for both kinds of record this library writes.

```python
import pinneapple as pp

result = pp.experiments.reproduce("runs/burgers_pinn")
result.outcome                       # 'reproduced' | 'divergent' | 'refused'
print(result.to_markdown())          # the ticket: the metrics compared, and what moved
```

## The three outcomes

| Outcome | Meaning |
|---|---|
| `reproduced` | same versions, same inputs, every metric within the recorded tolerance |
| `divergent` | it ran, but the environment or the inputs differ, or a metric left its tolerance — and the differences are listed |
| `refused` | it cannot be run faithfully: a dependency, an artifact or an input is missing. The refusal names it |

**A refusal is a result, not an exception.** A record whose dataset was deleted, whose method no longer exists or
whose fingerprint no longer matches its problem comes back as `refused` with the sentence explaining why. A re-run on
whatever happens to be present would be a different experiment wearing the same name.

## What is compared

- **Software** — `python`, `pinneapple`, `numpy`, `torch`, `cuda`, `commit`. A change here is the first thing that
  explains a different number, so it is listed first.
- **Hardware** — `platform`, `machine`, `cpu_count`, `processor`, `blas`. Reported apart from the software: another CPU
  is not the same run, but it is not the same failure as a version bump either, and the metrics are compared within
  tolerance rather than bit for bit.
- **Inputs** — the problem fingerprint and the number of evaluation points. If the problem was edited, the run is not
  the recorded one.
- **Metrics** — each one against its tolerance. A metric the record has and this run does not produce counts as **out
  of tolerance**: a silently missing number is a comparison that passed by comparing nothing.

A record that does not say what it ran on cannot be called a reproduction. It comes back `divergent` with that reason
written down, even when the numbers match.

## Tolerances

Default relative tolerance is `1e-6`. The record decides when it says so: a `tolerances` map in `result.json` overrides
both the default and the argument. `reproduce(..., exact=True)` asks for bit-identical numbers, which is only honest for
a deterministic method on the same machine.

```python
pp.experiments.reproduce("runs/burgers_pinn", exact=True)              # rtol = 0
pp.experiments.reproduce("runs/burgers_pinn", rtol=1e-4)               # looser than the default
pp.experiments.reproduce("runs/burgers_pinn", tolerances={"relative_l2.u": 1e-2})
```

## The two kinds of record

**An experiment** — the directory `Experiment.save` and `ExperimentResult.save` write:

```python
exp = pp.Experiment("burgers_1d", method="pinn", options={"epochs": 2000}, reference="analytic", seed=3)
exp.save("runs/burgers_pinn")            # experiment.json
exp.run().save("runs/burgers_pinn")      # result.json (plus model.pt when there is one)
pp.experiments.reproduce("runs/burgers_pinn")
```

**A workflow run** — a flow document with what every node reported, plus the machine it ran on:

```python
import pinneapple_flow as pf

flow = pf.Flow.from_dict(document)
result = pf.run(flow, workdir="runs/heat")
pf.reproducibility.save_run("runs/heat", flow, result)     # flow.json, run.json, environment.json
pp.experiments.reproduce("runs/heat")
```

Each node's metrics are compared under `node.metric` names, so two nodes cannot hide behind the same metric name.
A flow that no longer runs — the archive was deleted, a tool was uninstalled — is `refused` with the failing node's own
sentence in the reason.

## Adding a kind

`pp.experiments.register_reproducer` takes a `Reproducer(kind, detect, prepare)`. `detect(directory)` says whether it
recognises the record; `prepare(directory)` returns the recorded metrics, environment and tolerances plus a callable
that runs it again and returns a `Measurement`. The comparison and the outcome are shared, so a new kind cannot invent
its own definition of "reproduced". Registering a kind that already exists with another meaning is refused.
