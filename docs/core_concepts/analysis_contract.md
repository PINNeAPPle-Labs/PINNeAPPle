# The analysis contract: what a calculation takes and how it words its finding

`pp.analysis` turns a function into an *analysis*: a calculation that declares what it takes, what it gives back, and
the one sentence that says what was found. The declaration is written once, next to the code that knows the number is
right, and it is what a form, a report, an app and an assistant all read instead of each re-deriving it.

```python
import pinneapple as pp
from pinneapple_core.analysis import InputSpec, OutputSpec, Severity, Status, analysis, tr

@analysis(
    "pipe.mean_velocity",
    label=tr("velocidade média", "mean velocity"),
    inputs={"Q": InputSpec(unit="m³/s", gt=0, label=tr("vazão", "flow rate")),
            "D": InputSpec(unit="m", gt=0, label=tr("diâmetro", "diameter"))},
    outputs={"U": OutputSpec(unit="m/s", label=tr("velocidade", "velocity"))},
    describe="U = {U:.3} m/s in a {D:.3} m pipe",
    example={"Q": 0.01, "D": 0.1},
)
def mean_velocity(Q, D):
    import math
    return {"U": 4 * Q / (math.pi * D * D), "D": D}   # what the sentence names travels with the result

pp.analyses()                                       # every registered analysis
pp.run_analysis("pipe.mean_velocity", Q=0.01, D=0.1)
pp.analysis_card("pipe.mean_velocity")              # inputs, units, validity range, outputs, example, references
pp.check_analysis("pipe.mean_velocity")             # [] when it answers junk with a reason
```

## Why declare a calculation at all

A bare function returns a number or raises. For something a person acts on, both are wrong some of the time: a NaN
propagates into a report, a blank field becomes a traceback on a screen, and a value outside the model's range comes
back looking exactly like a good one. The contract fixes the three cases in one place:

| The declaration | What it buys |
|---|---|
| `inputs=` with `unit`, `min`/`max`/`gt`/`lt`, `label` | the form, the API and the assistant all know the unit and the range the model holds in; a value outside it is refused with the number, the limit and the unit |
| `outputs=` with `unit` | an output the analysis forgot to return is an error, not a silently missing key |
| `describe=` | the finding is one sentence, in English or Portuguese, with the numbers in the reader's format |
| `maturity=` and `references=` | the result carries where the model comes from, so a finding is never separated from its source |
| `example=` | a valid call, which the card publishes and the kit runs |

## A status instead of an exception

Every call returns a `Result`; none of them raises at the caller:

| Status | When |
|---|---|
| `ok` | the calculation ran and produced an answer |
| `warning` / `bad` | it ran and the answer is worth a look / outside what is acceptable (the analysis says so with a `Finding`) |
| `not_applicable` | the model does not cover this input, and says why |
| `insufficient_data` | something the answer depends on is missing, NaN, not a number, or outside the model's range |
| `error` | the analysis itself failed; the sentence names the exception |

```python
pp.run_analysis("pipe.mean_velocity", Q=-1, D=0.1).status.value        # 'insufficient_data'
pp.run_analysis("pipe.mean_velocity", Q=-1, D=0.1).sentence            # 'flow rate (Q) must be greater than 0 m³/s, got -1'
pp.run_analysis("pipe.mean_velocity", Q=float("nan"), D=0.1).sentence  # 'flow rate (Q) must be finite, got NaN'
```

An analysis that cannot answer from what it was given raises `NotEnoughData` inside itself and the registry turns it
into `insufficient_data` with that message — "the window has 5 samples, 30 are needed" is a result a screen can show.
`strict=True` re-raises instead, which is for the analysis's own tests, not for anyone reading a result.

An analysis that judges the numbers returns a `Finding`, which carries its own status and severity:

```python
from pinneapple_core.analysis import Finding, Severity, Status

def against_a_limit(value, limit):
    ratio = value / limit
    if ratio > 1:
        return Finding(metrics={"ratio": ratio}, status=Status.BAD, severity=Severity.HIGH,
                       evidence={"value": value, "limit": limit})
    return Finding(metrics={"ratio": ratio})
```

## One sentence, in the reader's language

`describe` is a template over the metrics: `{dp:.4}` gives four decimals, `{ratio:%}` a percentage, `{name}` the
number as it is written. English and Portuguese differ in the decimal mark, so the numbers follow the language in
force — set with `use_language("pt")` or `PINNEAPPLE_LANG=pt`:

```python
from pinneapple_core.analysis import use_language

with use_language("pt"):
    print(pp.run_analysis("pipe.friction_factor", Re=45_000, rel_rough=4.5e-4).sentence)
    # f = 0,0228 em Re = 45.000 com ε/D = 0,00045 (turbulento)
```

A template naming a number the result does not carry prints an em dash, never a `KeyError`: the sentence a client
reads is the last place a library should fail.

## The kit: junk in, a result out

`pp.check_analysis` builds the hostile calls from the declaration alone — nothing at all, each input missing, NaN,
±infinity, text where a number belongs, one value below the minimum, one above the maximum, an empty array — and
reports any analysis that raises or that answers `ok` to something it could not have used. It is the test a new
analysis is not finished without:

```python
from pinneapple_core.analysis import check_all, check_analysis

assert check_analysis("pipe.friction_factor") == []
check_all()          # {analysis id: problems} for every analysis in the library; {} means all of them pass
```

```bash
python -m pinneapple_core.analysis.check            # exit 1 when an analysis fails the kit
```

## The analyses the library ships

Four, each wrapping a real closed-form model with no physics added — the wrapped function stays the source of the
number:

| Id | Model |
|---|---|
| `pipe.friction_factor` | Darcy friction factor, laminar / Colebrook, with the regime |
| `pipe.darcy_pressure_drop` | Darcy–Weisbach pressure drop over a length of pipe |
| `acoustics.helmholtz_frequency` | Helmholtz resonator tuning frequency |
| `thermal.fin_array_base_temperature` | base temperature of a fin array (Incropera & DeWitt) |

The registry loads them on first use, so `pp.analyses()` is the complete list whether or not anything was imported by
hand. A product adds its own with `@analysis(...)` in its own module, or with
`pp.register(my_analysis, kind="analysis")` — the same protocol as every other extensible object
([one registry for every kind](registry.md)).
