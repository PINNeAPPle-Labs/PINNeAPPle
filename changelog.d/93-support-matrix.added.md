- The supported versions matrix (#93): `pinneapple_stability/support_matrix.json` declares Python 3.10 to 3.13,
  torch 2.2 or newer (currently covering 2.15 and 2.14) and Linux, macOS and Windows, in seven cells;
  `scripts/check_matrix.py` checks the environment it is started in against it and runs five smoke steps
  (imports, the `pp` facade, a preset, the stability classes, a PINN forward/gradient/Adam step) in seconds.
  `--record` writes a passing run into the JSON, and `docs/support_matrix.md`, generated from it, marks a cell
  *executed* only where such a run exists and *declared* everywhere else. Five of the seven are executed today:
  Linux (Debian slim containers) on Python 3.11, 3.12 and 3.13, and macOS arm64 on Python 3.11 with torch 2.15 and
  2.14. The page says in the cell itself why the other two were never run: Linux on Python 3.10 stops at the install
  (no `numcodecs` wheel), and there is no Windows machine or runner here.
