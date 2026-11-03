# Industrial closed-loop flows

Reactor sizing (cooled packed bed and CSTR), computational lithography (ILT mask correction), wing planform design and the least-drag Ahmed body: each a model, an optimisation and the design re-verified by the model, with the measured run times and results.

Regenerate with `python -m pinneapple_flow example-flows gallery industrial` (runs every entry, quick and full, and rewrites these files from the runs).

| entry | flow | quick | full |
|---|---|---|---|
| [reactor_packed_bed](reactor_packed_bed/README.md) | `reactor_sizing_packed_bed` | done, 6.9 s | done, 16.2 s |
| [reactor_cstr](reactor_cstr/README.md) | `reactor_sizing_cstr` | done, 11.9 s | done, 27.3 s |
| [lithography_opc](lithography_opc/README.md) | `lithography_opc` | done, 32.7 s | done, 27.2 s |
| [wing_design](wing_design/README.md) | `wing_design` | done, 7.6 s | done, 17.8 s |
| [ahmed_design](ahmed_design/README.md) | `ahmed_body_end_to_end` | done, 184.5 s | done, 346.3 s |
