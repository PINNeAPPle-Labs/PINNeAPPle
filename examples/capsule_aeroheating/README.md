# Capsule aeroheating: what the cheap model buys

The [capsule-aeroheating corpus](https://huggingface.co/datasets/bijective/capsule-aeroheating) gives two fidelities of
the same thing: 1.024 entry capsules with the laminar heat flux from a Newtonian panel method — an engineering model
that costs seconds — and 64 of those also with a second-order laminar Navier–Stokes solution from Eilmer, which cost
1.2 h to 5.4 h each on six cores. Every wall carries the same 2.361 nodes and node *i* is the same point of the heat
shield on every capsule and at both fidelities, so the two fidelities subtract node by node, with no interpolation.

That makes it the cleanest public multi-fidelity corpus this library reads, and it asks one question worth an
experiment: **how many CFD runs does the cheap model buy?**

## The answer

A kernel-ridge surrogate is fitted on N of the Navier–Stokes runs and scored on the ones it did not see, four ways:
the cheap model as it stands (no fit), a surrogate of the target from the six inputs alone, a learned correction to
the cheap model, and the same correction with the cheap model's value among its inputs.

Relative L2 against the Navier–Stokes value, median over 20 random splits:

| target | cheap model | correction, N=8 | correction, N=64 | design alone, N=64 |
|---|---|---|---|---|
| heat load | 5.85 % | 5.04 % | 3.55 % | 54.86 % |
| stagnation heat flux | 18.28 % | 17.87 % | 14.60 % | 26.85 % |
| **shoulder heat flux** | **27.43 %** | **16.41 %** | **7.01 %** | 25.55 % |

Three things follow, and they are the reason the corpus is worth keeping:

* **The shoulder is where the cheap model is structurally wrong** — it cannot see how the shoulder changes the flow
  over the face — and it is where eight Navier–Stokes runs buy the most: 27.4 % → 16.4 %, and 7.0 % with 64.
* **Where the cheap model is already good, eight runs buy little** (heat load: 5.85 % → 5.04 %). There is less left
  to correct, and the corpus says so instead of pretending a correction always pays.
* **The design alone never gets close** (26 % to 55 %): knowing the diameter, the nose radius and the Mach number does
  not tell you the heating. That is the justification for the panel method existing at all, measured rather than
  assumed.

## The same question asked of the whole field

A scalar per capsule is one view. The other is the field: 2.361 nodes per capsule, the same nodes on every capsule, so
the error is a *shape* as well as a size. Modelled as a few modes of the relative residual — each run's residual over
the magnitude of its own field, because the heating spans 175× across designs and the raw coefficients span that too —
with their amplitudes learned from the six inputs:

| | cheap field | + average correction | 3 modes, N=8 | 3 modes, N=64 |
|---|---|---|---|---|
| relative L2 per run | 17.91 % | **8.94 %** | 5.88 % | **3.57 %** |

The energy of the relative residual sits 81.0 % in one mode, 92.4 % in two and 97.4 % in three, which is why a handful
of runs is enough: **the correction is a shape, not a table**.

Two readings worth keeping apart. The *average* correction — one shape, no dependence on the design at all — already
halves the error (17.9 % → 8.9 %). That is not learning; it is the cheap model being wrong the same way everywhere.
What the design buys on top of it is the amplitude, and that is what the eight runs pay for (8.9 % → 5.9 %).

Pretraining on the cheap fidelity is not a strategy here: the cheap model is a function that can be evaluated at every
design, so a surrogate of it adds nothing but its own error. What the 1.024 cheap runs buy is the *shape of the
correction*, which is the `difference` row and the modes above.

## Run it

```bash
pip install bijective-gamut                       # the reader of the .gamut files
python examples/capsule_aeroheating/01_multifidelity.py          # downloads 77 MB once, then ~30 s
python examples/capsule_aeroheating/01_multifidelity.py --smoke  # 16 designs, 2 splits, seconds
```

Every number above is recorded in [`experiment.json`](experiment.json) with its unit and how it was measured — 20
metrics, 9 checks — and the two reports run from one command.

## What is here and what is not

The reader is `pinneapple_data.datasets.capsule_aeroheating` (pairing the fidelities, the wall as a bundle, the
fidelity pair of a field, the samples a field benchmark reads) and the four independent checks are in
`capsule_aeroheating_physics` (axial symmetry, the panel-method pressure recomputed from the geometry, the heat load
integrated over the faces, and the radiative equilibrium wall temperature). The regression curves are
`pinneapple_data.multifidelity`, which is not specific to this corpus: any pair of fidelities with a known value of the
cheap model at the expensive designs can use it.

What is not here: a pointwise model free to choose every node's correction. With 64 fields over 2.361 nodes it would
be fitting noise, and the modes above say how little there is to gain — the correction is 97 % three modes deep.

## Licence and citation

The data and the figures are CC BY 4.0; the reader (`bijective-gamut`) is Apache-2.0. Cite the dataset as

```bibtex
@misc{bijective_capsule_aeroheating_2026_v4,
  title        = {Capsule forebody aeroheating, Newtonian and Navier--Stokes {CFD}},
  author       = {{Bijective Systems}},
  year         = {2026},
  version      = {4},
  howpublished = {\url{https://huggingface.co/datasets/bijective/capsule-aeroheating}}
}
```
