# `calculix_contact/` — a real CalculiX run of a friction contact, and what the API's parser read from it

These files are the output of **CalculiX 2.23** (`~/ccx-env/bin/ccx`) on a deck written by the engineering API's
PrePoMax adapter (`app/prepomax`, commit `a63f5c5`): the sliding-block case — two 4 × 4 × 2 mm blocks of steel in
contact with Coulomb friction `mu = 0.3`, a prescribed pressure of 5 MPa on the top face and a tangential nodal load at
**half** the friction limit (12 N against 24 N, so the joint sticks and the friction cone is not reached).

| file | what it is |
|---|---|
| `fixture.inp` | the deck (`*NODE`, `*ELEMENT`, `*SURFACE INTERACTION`, `*FRICTION`, `*CONTACT PAIR`, `*CONTACT FILE`, `*NODE PRINT, TOTALS=YES`) |
| `fixture.frd` | the result file: 54 nodes, 16 elements, 4 increments, blocks `DISP`, `STRESS`, `CONTACT`, `ERROR` |
| `fixture.dat` | the printed results: the per-node `forces` of set `BFIX`, its **`total force`** per increment, and the `displacements` of set `STOP` |
| `fixture.sta` | the increment summary (4 convergent increments, no cutbacks) |
| `fixture.log` | the solver's output, ending in `Job finished` |
| `golden.json` | **what the API's own parser read from the four files above**, node by node and increment by increment |

`golden.json` is the point of the folder. `tests/pinneapple_data/test_cae_calculix_contact.py` compares the library's
readers (`pinneapple_data.cae`) against it, so the assertions say "the library reads what the parser in production
read", not "the library reads what it reads". The JSON was produced with the API's own module:

```python
from app.prepomax import results as R          # pinneapple-engineering-api @ a63f5c5, the parser before the library
status = R.read_status("fixture.sta")
{"coordinates": R.frd_nodes("fixture.frd"), "contact": R.contact("fixture.frd"),
 "contact_at_0_5": R.contact("fixture.frd", until=0.5),
 "displacement": R.displacement("fixture.frd"), "stress": R.stress("fixture.frd"),
 "totals": R.read_totals("fixture.dat"), "displacements": R.read_displacements("fixture.dat"),
 "status": status, "verdict": R.read_log("fixture.log", status, produced=True)}
```

## Why this file, and not a bigger one

`ccx` writes a value and its neighbour **without a separator** when the first one is wide, so rows of this file read
`-1        99-6.74101E-05 4.97508E-02-1.37441E-06 6.74101E+00 ...`: the node id butts against a negative value, and the
second value against the third. A reader that splits on whitespace finds five numbers where the file has six and every
offset after the mistake shifts — plausibly. Any replacement fixture must keep that property; the test asserts it.

The model is 2 × 2 × 2 cells per block so the file stays around 100 kB. The contact patch is then 3 × 3 nodes, which is
the smallest mesh that still shows a pressure profile; the app's own case refuses `malha < 3`, so this deck was written
directly from `app.prepomax.deck` / `mesh` rather than through the case library.

## Regenerating

```bash
cd <pinneapple-engineering-api checkout>
python - <<'PY'   # writes /tmp/fx/fixture.inp
from app.prepomax import deck as D
from app.prepomax.mesh import Grid, axis
a, hb, hs, mu, p, n = 4.0, 2.0, 2.0, 0.3, 5.0, 2
base = Grid(axis(0, a, n), axis(0, hb, n), axis(0, a, n))
slider = Grid(axis(0, a, n), axis(0, hs, n), axis(0, a, n), first_node=base.next_node,
              first_elem=base.next_elem, shift=lambda x: (0.0, hb, 0.0))
deck = D.Deck("contato com atrito: bloco deslizante sobre placa fixa (fixture)")
D.geometry(deck, [("BASE", base), ("SLIDER", slider)])
D.material(deck, "ACO", 210000.0, 0.3); D.section(deck, "BASE", "ACO"); D.section(deck, "SLIDER", "ACO")
top = slider.nodes_on("y", hb + hs)
D.node_set(deck, "BFIX", base.nodes_on("y", 0.0)); D.node_set(deck, "STOP", top); D.node_set(deck, "TOP", top)
D.surface(deck, "MASTER", base.faces("y", hb, side=-1)); D.surface(deck, "SLAVE", slider.faces("y", hb, side=+1))
D.surface(deck, "PRESS", slider.faces("y", hb + hs, side=-1))
D.interaction(deck, "INT", mu)
D.contact_pair(deck, "CONTACT", slave="SLAVE", master="MASTER", interaction_name="INT")
deck.block("*STEP, NLGEOM=YES", "*STATIC", "0.25, 1.0")
deck.block("*BOUNDARY", "BFIX, 1, 3, 0.0")
deck.block("*DLOAD", f"PRESS, P, {p:.6g}")
deck.block("*CLOAD", f"TOP, 1, {-0.5 * mu * p * a * a / len(top):.8g}")
D.outputs(deck, totals_set="BFIX", displacement_set="STOP", stress=True)
deck.block("*END STEP")
open("/tmp/fx/fixture.inp", "w").write(deck.text())
PY
cd /tmp/fx && ccx -i fixture > ccx.log 2>&1
```

then copy `fixture.{inp,frd,dat,sta}`, `ccx.log` (as `fixture.log`) and regenerate `golden.json` with the snippet above.
The numbers in `golden.json` are floats printed by `ccx` with seven significant digits; they are compared exactly
(`abs=1e-12`), which is the point: the two readers must agree digit for digit, not approximately.
