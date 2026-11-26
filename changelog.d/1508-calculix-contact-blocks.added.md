- `pinneapple_data.cae` reads what a structural contact analysis is judged by (#1508): the `.frd` block `CONTACT`
  (six components per node — `COPEN`, `CSLIP1/2`, `CPRESS`, `CSHEAR1/2`) through `contact()` and `read_blocks()`, the
  largest `|tau|/p` over the loaded nodes through `coulomb_ratio()` — the number the Coulomb cone compares — and the
  reaction totals of the `.dat` (`total force (fx,fy,fz) for set <NSET> and time <t>`) through `read_totals()`, with
  the per-increment history, so equilibrium and friction are measured without integrating a pressure field. `read_frd`
  now takes its blocks from the same reader, so the mesh view and the block view cannot drift apart.
  `pinneapple_data.cae.ccx` holds the run files (`.dat`, `.sta` and the solver log) and their verdict; every value is
  read by fixed column, which is the only way to read `ccx` output where a value and its neighbour are written with no
  separator. The tests read a real 2.23 run — `tests/fixtures/calculix_contact/` — and compare it node by node with
  what the engineering API's own parser read from the same bytes.
