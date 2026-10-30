# Quickstart

Five minutes, four commands. If you want the version with thirteen worked recipes and the output
each one prints, read [GETTING_STARTED.md](GETTING_STARTED.md). If you want to know which path is
yours, read [START_HERE.md](START_HERE.md).

## Install

```bash
pip install pinneapple
```

Extras are opt-in. Two of them are bundles, named after the work they enable: `pinneapple[cfd]`
(geometry, meshing, 3-D viewing) and `pinneapple[enterprise]` (serving, signatures, the CI linter,
the hub, the MCP server). A research install is a few of the smaller ones:
`pip install "pinneapple[pinn,uq,viz3d,jax_cfd]"`. `pp.doctor("all")` lists every extra and which
ones are missing here.

```python
import pinneapple as pp

pp.info()                 # version, device, what is installed
pp.doctor("start")        # is this machine ready for the first week? fixes, one per line
```

## 1. Ask what to run

Describe the problem in a sentence, in English or Portuguese. You get a preset, a method that runs
without training anything, a model for later, and the terms the ranking matched.

```python
import pinneapple as pp

rec = pp.recommend("heat conduction in a heat sink cooled by a fan")
print(rec.explain())
```

## 2. Look before you run

```python
print(pp.card(rec.preset))     # coordinates, fields, conditions, method, and a program that runs
```

## 3. Solve it

```python
import numpy as np
import pinneapple as pp

sol = pp.solve("poisson_2d", method="reference")   # the classical solver: instant
print(sol.metadata())                       # what was solved, by which method, how long it took

X = np.array([[0.5, 0.5]])                  # a point in (x, y)
print(sol.predict(X))                       # one column per field
```

For the preset `recommend` gave you, `pp.card(rec.preset)` prints the same program with the right
coordinates, the right number of fields and a fallback: the card tries the cheapest method once, and
says whether it actually ran here. A partial answer is never returned in place of a full one, and a
preset that cannot run on this machine comes with the same physics written for a box you define.

## 4. Make it a project

```python
path = pp.new("my_study", template="plain", preset="poisson_2d")
```

```bash
cd my_study
python run.py
```

`case.yaml` is the only file you edit. `template="research"` adds metrics with their units, checks
against the classical answer and an experiment manifest; `template="cfd"` sweeps a parameter
through the reference solver into a CSV and a figure.

## Nothing here? Tell it in a sentence you would say out loud

```python
import pinneapple as pp

rec = pp.recommend("conducao de calor em uma aleta de resfriamento")
print(rec.preset, rec.method, rec.confidence)   # preset, method, how sure, and the matched terms
print(rec.alternatives)                          # the near misses, and what they differ in
```

## From the command line

```bash
pinneapple doctor research                  # check this machine for a research install
pinneapple recommend "pressure drop in a pipe"
pinneapple card pipe_flow_3d
pinneapple new my_study --template research --preset pipe_flow_3d
pinneapple check results/                   # the physics linter, for results and models
```

## Where to go next

| I want | read |
|---|---|
| worked recipes with their real output | [GETTING_STARTED.md](GETTING_STARTED.md) |
| to understand the five objects every run connects | [docs/getting_started/first_pinn.md](docs/getting_started/first_pinn.md) |
| my own PDE, geometry and data | [examples/quickstart/04_own_problem_and_data.py](examples/quickstart/04_own_problem_and_data.py) |
| to know which method family fits my problem | [docs/guides/choosing_a_method.md](docs/guides/choosing_a_method.md) |
| to record an experiment so it can be graded | [docs/guides/adding_an_experiment.md](docs/guides/adding_an_experiment.md) |
| the whole map of the library | [README.md](README.md) and [docs/index.md](docs/index.md) |
