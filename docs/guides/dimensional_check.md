# Checking the dimensions of an equation

Roadmap item #205. A PDE whose terms do not share a dimension is not a modelling
subtlety: it is a unit mistake, and a network trained on it fits the residual of
a statement that cannot be true. It shows up later as an error floor nobody can
explain, usually after a day of tuning.

`pp.check_dimensions` reads the equation the way a reader does, assigns a
dimension to every term from the units the problem already declares, and says
which term broke it.

```python
import pinneapple as pp

report = pp.check_dimensions(
    "dT/dt = alpha*laplacian(T) + q/(rho*cp)",
    units={"T": "K", "t": "s", "x": "m", "alpha": "m^2/s",
           "q": "W/m^3", "rho": "kg/m^3", "cp": "J/(kg*K)"},
)
print(report)
# dT/dt -> K/s, alpha * laplacian(T) -> K/s, q / (rho * cp) -> K/s: consistent
```

Change one unit and the report says what does not fit:

```python
report = pp.check_dimensions("dT/dt = alpha*laplacian(T) + q/(rho*cp)",
                             units={..., "q": "K"})
print(report)
# dT/dt -> K/s, alpha * laplacian(T) -> K/s, q / (rho * cp) -> m·s^2·K^2/kg: inconsistent
report.problems
# ["'q / (rho * cp)' is m·s^2·K^2/kg, the other terms of the equation are K/s"]
```

`report.to_markdown()` is the same thing as a table, and
`report.raise_if_inconsistent()` refuses to go on with the report as the
message.

## On a problem, the units are already there

A `PhysicalProblem` that declares its units needs nothing repeated:

```python
problem = pp.PhysicalProblem.from_pde_spec(spec, units={
    "x": "m", "y": "m", "z": "m", "t": "s", "T": "K", "alpha": "m^2/s"})

problem.dimensions("dT/dt = alpha*laplacian(T)")     # a report
problem.validate(equations="dT/dt = alpha*laplacian(T)")   # [] — or the reasons
```

`validate(equations=…)` puts the dimensional problems in the same list as the
structural ones, so a problem can be refused once, for every reason it has.
That is the wiring the issue asks for: the check runs before training, next to
the checks that already refuse an incomplete problem.

## What it reads

| Written as | Means |
|---|---|
| `dT/dt`, `∂T/∂t` | time derivative: the field over the time coordinate |
| `d2T/dx2`, `∂²T/∂x²` | second spatial derivative |
| `laplacian(T)`, `lap(T)`, `∇²T` | field over a length squared |
| `grad(T)`, `div(u)`, `∇·u` | field over a length |
| `d_dt(T)`, `dt(T)`, `d_dx(T)`, `dx(T)`, `d2_dx2(T)` | the same, in call form |

A parenthesised sum is split into terms like any other, and `… = 0` is read as
"the expression vanishes": the zero is not a term that has to carry a unit.

## What it refuses

A name it does not know, a function it does not know, an exponent that is not a
whole number, and a spatial or time derivative with no coordinate declared. Each
of those is refused **by name**, because a checker that assumed the missing unit
would pass exactly the equation it was written to catch.

```python
pp.check_dimensions("dT/dt = alpha*laplacian(T) + source", units={...})
# ["unknown name 'source': give its unit, e.g. units={'source': 'K'}"]
```

`space=` and `time=` name the units behind the derivatives when the problem has
no coordinate to take them from; without one of those and without a coordinate
of that dimension, the checker refuses instead of assuming metres and seconds.

## What it does not do

It does not check that the equation is the right one, or that a term is
physically sensible: a source in W/m³ and a source in W are both consistent with
a temperature equation only if the unit says so. It checks one statement at a
time.
