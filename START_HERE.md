# Start here

One page to decide what to read next. It takes about two minutes, and nothing here needs a
background in physics-AI or in Python. Every command below runs as printed on a clean
`pip install pinneapple`.

```bash
pip install pinneapple
```

```python
import pinneapple as pp

report = pp.doctor("start", verbose=False)      # is this machine ready for the first week?
print("ready" if report.ok else [c.name for c in report.failures])
```

If the last line says `ready`, nothing else is missing. If it lists items, each one comes with the
command that fixes it: `pp.doctor("start")` prints them.

One thing worth knowing before you type a problem: **not every preset can run on every machine.**
Some need FEniCS or OpenFOAM (which `doctor` reports), and many describe boundaries as named regions
of a real geometry. So `pp.card(name)` tries the cheapest method once and tells you whether it ran
here, and `pp.recommend(...)` prefers a preset that does run. When the closest preset cannot run,
you still get a program: the same equation written for a box you define yourself, which trains
anywhere. No page here offers you a command that fails without saying why.

---

## Which of these is you?

| you are | go to | it takes |
|---|---|---|
| **an engineer with a problem, and you would rather not write Python** | [Path A](#path-a--i-have-a-problem-not-a-python-project) | 10 minutes |
| **starting in physics-AI and want to understand what these models do** | [Path B](#path-b--i-am-learning-physics-ai) | an afternoon |
| **you know the theory and want pipelines, comparisons and papers** | [Path C](#path-c--i-know-the-theory-i-want-pipelines-and-papers) | an hour to first result |
| **putting this behind a product or a service** | [Path D](#path-d--i-am-building-a-product-or-a-service) | an hour, then per case |

---

## Path A — I have a problem, not a Python project

You edit one file, `case.yaml`, and run one command. No Python.

```bash
pinneapple new my_case --template plain
cd my_case
```

Open `case.yaml`. It has three lines that matter:

```yaml
preset: cpu_heatsink_thermal     # what to solve
method: auto                     # auto | reference | analytic | pinn
params: {}                       # the physics: heat load, viscosity, Reynolds number
```

Do not know which preset? Delete `preset` and write the problem in a sentence, in English or
Portuguese. `run.py` picks the closest preset and prints why:

```yaml
problem: "aleta de resfriamento com carga termica de 80 W"
```

Then:

```bash
python run.py
```

It writes `results/field.png`, `results/report.md` in plain language, `results/solution.npz` and
`results/metadata.json`. Change `params`, run again, compare the two reports.

**When you outgrow it:** `pinneapple new my_case --template research` writes the same study with
metrics, units, checks and an experiment manifest, and `pinneapple card <preset>` tells you what
the preset accepts.

---

## Path B — I am learning physics-AI

Start from a problem whose answer is known, so every number can be checked.

```python
import numpy as np
import pinneapple as pp

print(pp.card("burgers_1d"))          # what it takes, what it gives back, how to run it

sol = pp.solve("burgers_1d", method="analytic")   # the exact answer: instant, no training
print(sol.metadata())
print(sol.predict(np.array([[0.0, 0.5]])))        # u at (x, t) = (0, 0.5)
```

Then train a network on the same problem and score it against that exact answer:

```python
import pinneapple as pp

# 200 epochs keeps this under half a minute; use 2000 for an answer worth quoting.
comparison = pp.compare("burgers_1d", methods=["pinn"], reference="analytic",
                        options={"pinn": {"epochs": 200}})
print(comparison)
```

The last column is the relative L2 error of the network against the exact solution. With 200
epochs it is around 3e-1: the lesson of the recipe is that the number is printed, not that it is
small. Run it again with 2000 and watch it fall.

The order matters and it is the habit this library is built around: **exact answer first, network
second, error against the exact answer always.**

Where to go next, in this order:

1. [`GETTING_STARTED.md`](GETTING_STARTED.md) — thirteen recipes, each with the output it prints.
2. [`docs/getting_started/first_pinn.md`](docs/getting_started/first_pinn.md) — the five objects every run connects.
3. [`docs/guides/choosing_a_method.md`](docs/guides/choosing_a_method.md) — why one family over another.
4. [`notebooks/colab/`](notebooks/colab/README.md) — the same recipes as notebooks, runnable in Colab.

---

## Path C — I know the theory, I want pipelines and papers

```bash
pinneapple doctor research          # PINNs, operators, uncertainty, JAX, figures
pinneapple new my_study --template research
cd my_study && python study.py
```

`study.py` solves, compares against the classical answer where one exists, records every metric
with its unit and how it was measured, and writes `experiment.json`, the manifest the Lab grades
from A to F. `python -m pinneapple_lab seal --explain experiment.json` says what the study lacks
for the next letter.

The APIs this path uses, in the order a study needs them:

| question | call |
|---|---|
| which preset is closest to mine, and why | `pp.recommend("...")` |
| what does it take and give back | `pp.card(name)` |
| solve by any method on the same description | `pp.solve(problem, method=...)` |
| two methods on the same points and budget | `pp.compare(...)` |
| can I trust this field | `pp.trust_card(...)` |
| validate a trained model | `pp.validate(model, problem, level="quick")` |
| record it so the Lab, the site and CI can read it | `with pinneapple_lab.experiment(...)` |
| lint results in CI | `pinneapple check <paths>` |

`pp.list_methods()` lists every backend; `pinneapple check --help` lists the physics checks.

---

## Path D — I am building a product or a service

```bash
pinneapple doctor enterprise       # serving, signatures, CI linter, hub, MCP server
```

What that profile checks, and what each part is for:

| need | in the library |
|---|---|
| an HTTP service in front of a trained model | `pinneapple_app` (FastAPI) and `pinneapple_tools.model_export` (ONNX) |
| provenance, signatures, audit trail | `pinneapple_security` |
| a physics gate in CI, so a wrong result cannot merge | `pinneapple check <paths>` |
| a model card with measured errors and a validity range | `pp.zoo.validate(name, inputs)` |
| an agent or an assistant calling the library | `pinneapple-mcp` (MCP server) |
| a decision record for a client deliverable | `pinneapple_veriphysics` |

The rule that makes this usable commercially: `pinneapple check` returns a non-zero exit code when
a result fails its physics checks, so the same command works on a laptop and in a pipeline.

---

## The four commands

| command | answers |
|---|---|
| `pinneapple doctor [profile]` | is this installation ready for `start`, `cfd`, `research`, `enterprise` or `all`? |
| `pinneapple recommend "<problem>"` | which preset, which method, which model, and why |
| `pinneapple card <preset>` | what a preset takes, what it gives back, the smallest program that uses it |
| `pinneapple new <name>` | a project of your own, at one of three levels |

They are also plain Python: `pp.doctor(...)`, `pp.recommend(...)`, `pp.card(...)`, `pp.new(...)`.

## Where everything is

| file | for |
|---|---|
| `START_HERE.md` | this page: which path is yours |
| [`GETTING_STARTED.md`](GETTING_STARTED.md) | thirteen recipes with the output they print |
| [`QUICKSTART.md`](QUICKSTART.md) | the five-minute version of this page |
| [`README.md`](README.md) | what the library is and why it exists |
| [`docs/`](docs/index.md) | the reference: concepts, guides, API |
| [`examples/`](examples/) | runnable scripts, by package |
| `templates/` | 35 single-file scripts, one capability each |

## Known limits

Said here rather than discovered later:

- **Eighteen presets name FEniCS as their solver**, among them `cpu_heatsink_thermal` and
  `plane_stress_2d`. Nine name OpenFOAM. Without them the card says so, and hands over the same
  physics written for a box you define.
- **Forty presets bind their boundaries to named geometry tags.** Thirty-two of them declare a
  canonical box or cylinder that the tags map onto unambiguously, and `pp.solve` sends it for you, so
  they train out of the box; the card prints which geometry it used. The other eight describe a real
  airfoil, car body or furnace, where inventing a box would produce a plausible but meaningless run:
  for those the card lists the tags and hands over the box version of the same equation.
- **The reference solver covers the fields it computes.** When it returns a partial field,
  `pp.solve` raises `MethodNotAvailable` instead of a wrong answer, which is what `pp.card` reports.
- **`pp.recommend` is keyword matching**, not a model. It shows the terms that matched so you can
  judge it, and returns nothing rather than guessing when nothing matches.
- **Two presets have no generated alternative** (`lane_emden_polytrope`, `shakura_sunyaev_accretion_disk`):
  their equation needs a setup the generated program does not write, so the card says what is
  missing instead of printing code that would fail.
