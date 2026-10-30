# Design reports without an LLM: `Datasheet` and the self-planning study

A study that ends in a folder of JSON is not a deliverable. This guide documents the two modules that turn a
study into a datasheet, and the one that decides *what to study* before it runs:

| module | job |
|---|---|
| `pinneapple_tools.reporting.datasheet` | layout only: KPI strip, numbered sections, table-beside-figure rows, run plan, items to confirm, limitations. Renders HTML, prints to PDF. |
| `pinneapple_simulation.electromagnetics.reflector.study_report` | plans a compact-range study from a `Problem`, runs and caches every step, renders the datasheet |
| `...reflector.study_report_text` / `..._figures` | the narrative and the figures, both computed from the stored numbers |

Nothing in this chain calls a language model. Every sentence is a format string over a number that the solver
produced, and every step exists because a dimensionless number asked for it.

## The datasheet

```python
from pinneapple_tools.reporting.datasheet import KPI, Figure, Note, Section, Step, Table, Datasheet

d = Datasheet("3 GHz compact range", subtitle="Quiet zone by physical optics, validated",
              eyebrow="EM / reflector")

d.kpis([KPI("Focal length", "1.50", "m", "F/D = 0.62"),
        KPI("Fresnel number", "0.42", "", "a^2 / (lambda z)")])

d.note("The design does not deliver the 0.60 m quiet zone.", "bad", "Specification not met")

s = d.section("Model and resolution", lede="What the geometry is and at what resolution it is integrated.")
s.row(Table([["Focal length", "1.50 m"], ["Mesh", "18,400 triangles"]],
            header=["quantity", "value"], caption="**The design as modelled.**"),
      Figure("fig_side.png", "**Side section at x = 0.**"))
s.note("The quiet zone is in the near field.", "warn", "Fresnel number 0.42 < 2")

d.plan(done=[Step("Model", "the section is 12 wavelengths across", "done", "3 s")],
       queued=[Step("Bandwidth", "the range must cover 2-4 GHz", "queued", "")])
d.confirm(["Which tolerance, over which diameter?"])
d.limitations(["Physical optics: no fringe (PTD) edge current."])

d.build("report/cr.html", pdf="report/cr.pdf")   # HTML always; PDF when a browser exists
```

* `note(text, tone, title)` with `tone` in `info` (neutral), `warn` (amber) and `bad` (red). The verdict is a
  `bad` note, so a failed specification cannot be styled as a success.
* `plan(done=..., queued=...)` renders the run plan: what each step is, why it is there, what it cost, and
  where it stands. This is what makes the document auditable rather than assertive.
* Glyphs: write `$\\lambda$`, `$\\pm$`, `&deg;` freely. `build()` translates them after escaping, so a
  token never leaks into the PDF as literal text.
* `build()` prints through headless Chromium. `find_chrome()` probes `$PINNEAPPLE_CHROME`, `PATH`
  (`google-chrome`, `chromium`, `microsoft-edge`, `headless_shell`), the macOS app bundles, then Playwright's
  cache. With no browser it still writes the HTML.

## The study that plans itself

```python
from pinneapple_simulation.electromagnetics.reflector.study_report import Problem, plan, run, report

p = Problem(freq_hz=3e9, focal=1.5, section=1.2, offset=0.9, roll=0.15,
            feed_gain_dbi=12.0, qz_diameter=0.6, qz_z=3.0, band_hz=(2e9, 4e9))

for step in plan(p):
    print(step.key, "->", step.reason)

study = run(p, "results", parallel=8)   # every number cached in results/study.json
report(p, study, "report")              # report/compact_range.pdf
```

`plan()` reads the geometry and returns the steps it can justify, each with the number that put it there:

| step | when it is planned | the number that decides it |
|---|---|---|
| `model`, `validation` | always | the section in wavelengths; the integrator is checked against a closed form, PO against GO, the mesh against itself, before any number is read |
| `qz` | always | the quiet zone at the reference planes, cuts **and** the full disk |
| `spec` | always | usable diameter at three tolerance tiers, so the verdict does not hang on one threshold |
| `reference` | `reference=` given | published rows are compared row by row |
| `calibration` | `reference_geometry=` given | how close the module's edge family can get to the published extents and area, with the residual reported rather than hidden by a fit |
| `profile` | the edge is thin in wavelengths | the spread of every metric over the plausible profile family |
| `bandwidth` | `band_hz=` given | the edge depth in wavelengths across the band |
| `tolerance` | always | what a feed displacement of 0.25-3 lambda costs |
| `axial` | always | the quiet zone is a cylinder, not a plane |
| `sweep` | `qz_to_section > 0.6` or `Fresnel < 2` | whether any (feed gain, section size, rolled edge) in the family reaches the specification |
| `optimize` | same condition | a grid and a surrogate-driven search on the same solve budget |

`Problem.from_text()` parses a sentence into the same object:

```python
p = Problem.from_text("3 GHz, focal length 1.5 m, 1.2 m square reflector, 0.15 m rolled edge, "
                      "0.6 m quiet zone at 3 m, feed 12 dBi")
```

The command line does the whole thing in one call:

```bash
python -m pinneapple_simulation.electromagnetics.reflector.study_report \
    --freq 3 --focal 1.5 --section 1.2 --offset 0.9 --roll 0.15 --gain 12 --qz 0.6 --qz-distance 3 \
    --band 2 4 --parallel 8 --out cr_study
```

The lengths are metres unless ``--feet`` is given, and ``Problem.from_text`` accepts either unit in the
sentence itself.

## Rules the narrative follows

These are the ones that keep the report honest, and they are worth copying into a new study:

1. **The verdict comes from a criterion, not from a mood.** `usable >= qz_diameter` at the declared tolerance
   tier, printed with the percentage of the specification it represents.
2. **A disagreement with a reference is stated with its cause.** Where PO cannot represent the reference's
   edge, the difference is reported and the conclusions are drawn from differences between designs, which
   share the edge, never from absolute levels.
3. **Every absolute claim is quoted against the spread of the model family.** If a conclusion does not
   survive the profile-family band, it is not a conclusion.
4. **A metric with a different meaning is flagged as such.** The cross-polar level on the principal cuts is a
   feed property: with a symmetric reflector and an ideal `cos^q` feed it cancels by symmetry, so PO reporting
   -58 dB where a full-wave solve of a real horn reports -23 dB is measuring a different thing, not making an
   error. The report says so instead of claiming agreement.
5. **The run plan and the limitations are part of the document.** What was not modelled is a section, not a
   footnote.

## Worked examples

* A full study -- the reference geometry, the full-disk metrics, the 48-design sweep and the 30-solve
  equal-budget search -- is `examples/use_cases/antennas/compact_range_qz.py` for the model side; adding
  `run()` and `report()` around the same `Problem` produces the datasheet.
* The same datasheet applied to an audit, where an external solver's result table is read back, checked
  against the physical bound of the antenna class and written up without a new solve, is the pattern in
  `docs/guides/antenna_design.md`.

See also [Compact range quiet zone](compact_range.md) for the underlying model and
[Antenna design](antenna_design.md) for the Vivaldi side.
