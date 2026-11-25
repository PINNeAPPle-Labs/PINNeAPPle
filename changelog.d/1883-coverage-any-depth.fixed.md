The coverage table sweeps every module at any depth (#1883): `pinneapple_flow.coverage.modules` walked the first
level with `pkgutil.iter_modules`, so a module four levels down -- `pinneapple_physics.pinn_solver.compiler
.integral_kinds` is the case that exposed it -- was never a row, and the node-or-reason gate never asked it for
anything. The walk now recurses through subpackages (still skipping `_`-prefixed names, still without importing
what it visits), the table grows from 625 to 2542 rows, and each row says whether it owes a node or a stated
reason: the first-level rows and every module `api/core.json` lists at any depth do, and the 176 that had neither
are now covered (`unreached` is 0, against 176 before). The 1113 deeper helpers reached through their package are
reported as `internal` instead of demanding a reason each. `python -m pinneapple_flow coverage` and
`autocalls.generate()` filter on the same `owing` flag, and `autocalls.json` is regenerated from the wider sweep:
249 generated nodes and 77 recorded reasons become 305 and 189, which also drops the reasons the stale file still
carried for packages that have had a curated node since.
