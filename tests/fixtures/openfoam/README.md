# OpenFOAM fixtures

`cavity_binary/` — the lid-driven cavity of `examples/physics_check/pass/cases/cavity`, written with
`writeFormat binary;` so the reader is exercised on the binary layout OpenFOAM actually writes (`N\n(<raw>)\n)`,
`faces` as a compact offsets + nodes pair) and not only on ASCII.

Provenance, reproduced on OpenFOAM v2606 (native macOS build):

```sh
cp -R examples/physics_check/pass/cases/cavity /tmp/cavity
sed -i '' 's/writeFormat     ascii;/writeFormat     binary;/' /tmp/cavity/system/controlDict
openfoam -c 'cd /tmp/cavity && foamFormatConvert'
cp -R /tmp/cavity tests/fixtures/openfoam/cavity_binary
rm -rf tests/fixtures/openfoam/cavity_binary/4.9        # the two time directories the tests read: 0 and 5
```

400 cells, 882 points, 1 x 1 x 0.1 m, patches `movingWall`, `fixedWalls` and the `empty` patch `frontAndBack`;
`p` and `U` at t = 0 and t = 5 (`phi` is a `surfaceScalarField`, which the reader reports as skipped).
