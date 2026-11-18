# Text to simulation, text to Physics AI

`pinneapple_intent` turns a sentence into a physics study that can be run, checked and replayed. It is the entry point
the library did not have: `pp.design` designs a part, `pp.recommend` suggests a preset, `pinneapple_flow.plan`
proposes a graph of library calls — this one compiles the request into a **typed, versioned contract**, runs the
fidelity ladder against real solvers, decides with the library's own trust rule, and writes the files that let
someone else reproduce the run without an agent.

```python
import pinneapple as pp

run = pp.study("a 12 mm steel wall with k = 200 W/mK, 300 degC inside and air at 25 degC outside with h = 12 W/m2K")
print(run.summary())
# Steady heat conduction through a wall, cooled by convection: cold_face_temperature = 572.953 K (APPROVED); 6/6 ...
run.files["report"]        # report.md, next to contract.json, run.json and trust.md
```

```python
pp.study("a heat sink for a 40 W chip")     # -> a contract with status needs_clarification and its questions
pp.study("a 20 mm pipe, 3 m long, carrying 0.5 L/min of water, viscosity 1 mPa*s")
pp.study("a cantilever beam 0.5 m long with E = 210 GPa, 20 mm deep, carrying 500 N at the tip")
pp.study("a cantilever beam 0.5 m long with E = 210 GPa carrying 500 N",
         answers={"height": "20 mm"})       # answer the question and run
```

```bash
pinneapple-intent families
python -m pinneapple_intent run "a 20 mm pipe, 3 m long, carrying 0.5 L/min of water, viscosity 1 mPa*s" --out runs/pipe
python -m pinneapple_intent replay runs/pipe --strict
```

## What it answers with

`compile_study` returns one of two objects, and never a guess:

| object | when |
|---|---|
| `StudyContract`, `status == "ready"` | every critical value is stated, answered, or defaulted with a source |
| `StudyContract`, `status == "needs_clarification"` | a critical value nobody gave: `contract.questions` lists what is missing, and the runner refuses to run it |
| `OutOfScope` | nothing covers the request. `kind` is `refused` (weapons), `needs` (with what would be required) or `no_template` (the families that do exist are listed) |

`text-to-science` is recognised and refused with what it would need: the literature stage (search, cite, compare
published results) is not part of this release, and saying so is better than pretending a simulation is a paper
reproduction.

## The contract

`StudyContract` (`pinneapple.study/1`) is the whole capability. It carries the sentence, the family, the equations,
the assumptions, the limitations, the references, the ladder, and every value with its provenance:

| provenance | meaning |
|---|---|
| `stated` | the user wrote it; `quote` is the text it was read from |
| `answered` | given in reply to a follow-up question |
| `defaulted` | the family filled it; `source` says from where (a datasheet, a convention, a fixed dimension) |
| `missing` | nobody gave it and there is no safe default — this is what becomes a question |

`contract.id` is the SHA-256 of the document without its timestamp, so two compilations of the same sentence share an
id and a run can be tied to the exact contract that produced it. `contract.write(path)` / `StudyContract.read(path)`
round-trip through JSON.

## The fidelity ladder

Every family runs the same shape of study, from cheap-and-exact to expensive-and-approximate:

| rung | what it is |
|---|---|
| `analytic` | the closed form, with its assumptions stated and its reference cited |
| `numerical` | a real solver in the library, on the case's own geometry |
| `convergence` | the same solver on a coarse and a fine grid, so the discretisation error is **measured**, not assumed |
| `surrogate` | `physics_ai` mode only: a sweep through the solver, a fitted surrogate, and its error on designs held out of the fit |

The grid spacing is derived from the case (`family.spacing(contract, rung)`), not fixed: a grid that does not resolve
the section is the most common reason a solver appears to "disagree" with a closed form it should reproduce.

## The checks and the decision

`family.compare(measurements)` produces the cross-rung checks — the solver against the closed form, the quantity of
interest under refinement, the surrogate on held-out designs — and each rung contributes its own (energy balance,
equilibrium of the discrete system, the Reynolds number against the model's range). They are turned into
`pinneapple_analysis.trust.trust_report.Check` objects and decided by `TrustReport`, so a study and a model deployment
answer to the same rule: a failed critical check is `REJECT`, a failed non-critical one is `REVIEW`, and a rung that
did not run lowers coverage instead of passing silently.

The checks that carry criticality: the solver against the closed form, the change under refinement, the slenderness
range Euler-Bernoulli applies to, and the Reynolds number where the laminar law holds. A study that asks for
5 L/min through a 20 mm pipe is **rejected**, because the answer it would compute is the answer to a different
problem.

## Replay

```python
from pinneapple_intent import replay

result = replay("runs/pipe")             # re-runs contract.json and compares every recorded number
print(result.summary(), result.differences, result.integrity)
```

The replay compares the contract's hash with the one the run recorded (catching a contract edited after the run),
re-runs the same ladder with the same seed, and reports every output that did not come back the same. It needs the
run folder and nothing else: no agent, no sentence, no network.

## The installed families

| family | quantity of interest | analytic rung | numerical rung |
|---|---|---|---|
| `heat_wall` | cold-face temperature [K] | series resistance (Incropera & DeWitt ch. 3) | `pinneapple_simulation.immersed.solve_heat` |
| `laminar_pipe` | pressure drop [Pa] | Hagen-Poiseuille | finite differences on the radial momentum balance |
| `cantilever_beam` | tip deflection [m] | Euler-Bernoulli, plane strain | `pinneapple_simulation.immersed.solve_elasticity` |

## Writing a family

A family is the unit a new domain plugs in with. Subclass `StudyFamily`, set the declaration, implement `evaluate`:

```python
from pinneapple_design.intent.spec import Slot
from pinneapple_intent import Measurement, StudyFamily, check, register
from pinneapple_intent.family import ANALYTIC, NUMERICAL


class MyCase(StudyFamily):
    name = "my_case"
    title = "What it answers"
    domain = "heat transfer"
    objective = "the quantity of interest"
    qoi = "my_quantity"
    qoi_unit = "K"
    qoi_label = "the quantity"
    keywords = {"my": 1.0, "words": 0.8}
    equations = ("the governing equation",)
    assumptions = ("what the closed form assumes",)
    limitations = ("what this study does not cover",)
    references = ("the source of the closed form",)
    slots = (Slot("my_input", "my input", unit="m", keywords=("input",), critical=True,
                  question="What is my input?"),)

    def analytic_label(self):
        return "the closed form"

    def evaluate(self, rung, contract, *, workdir=None, options=None):
        value = contract["my_input"] * 2.0
        return Measurement(rung=rung, outputs={self.qoi: value}, units={self.qoi: "K"},
                           checks=[check("the value is finite", value == value)])


register(MyCase())
```

Then `pp.compile_study("... words ...")` selects it and `pp.study(...)` runs its ladder. Two rules a family cannot
break, because the report is built from them:

- **Never invent a number.** A value comes from a closed form, a solver call, a cited reference, or the contract.
- **Never return a rung without its checks.** A number without the checks that make it a result is not a result.

## What this does not do yet

- `text-to-science` (a hypothesis or an article to a reproduced experiment) has no literature stage; the
  design and what remains is written down in [#1862](https://github.com/PINNeAPPle-Labs/PINNeAPPle/issues/1862).
- The surrogate rung fits a small MLP over one swept input; it is not a PINN or a neural operator trained on the
  field. Neural operators live in `pinneapple_neural` and are not wired into the ladder yet.
- Three families ship. The registry makes a fourth a module, not a rewrite, but the physics of each one is real work:
  every rung needs a closed form and a solver that can be checked against it.
