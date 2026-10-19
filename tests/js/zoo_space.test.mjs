// The zoo demo Space (static) against Python: every precomputed sweep of every model, run again through the
// browser runtime (pinneapple_hub/zoo/web: onnx.js, zoo.js) from the files the Space serves. The predictions must
// match the PyTorch numbers of pinneapple_hub.zoo.demo.precompute and the verdict of the query checks must be the
// same at every point (a status may differ only where a check value sits at its limit, within float32 rounding).
//   SPACE_DIR=<output of pinneapple_hub.zoo.publish.build_space> node --test tests/js/zoo_space.test.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { test } from "node:test";
import { Session } from "../../pinneapple_hub/zoo/web/onnx.js";
import { evaluate } from "../../pinneapple_hub/zoo/web/zoo.js";

const DIR = process.env.SPACE_DIR;
const json = (p) => JSON.parse(readFileSync(join(DIR, p)));
const index = DIR ? json("index.json") : { models: [] };
const CHECK_TOL = 1e-3;                                            // as pinneapple_hub.zoo.export.CHECK_TOL

function run(session, rows) {
  const data = new Float64Array(rows.length * rows[0].length);
  rows.forEach((r, i) => data.set(r, i * r.length));
  const t = { dims: [rows.length, rows[0].length], data };
  return session.run({ x: t, query: t });
}
const row = (t, i) => { const w = t.dims[t.dims.length - 1]; return Array.from(t.data.subarray(i * w, (i + 1) * w)); };

for (const { name } of index.models) {
  test(`${name}: every sweep of the Space reproduced in the browser runtime`, () => {
    const pre = json(`models/${name}.sweeps.json`), meta = json(`models/${name}.json`);
    const session = new Session(new Uint8Array(readFileSync(join(DIR, "models", meta.onnx.file))));
    const names = pre.inputs.map((d) => d.name), base = names.map((k) => pre.default.inputs[k]);
    let points = 0, nearLimit = 0;
    for (const [vary, sw] of Object.entries(pre.sweeps)) {
      const rows = sw.x.map((x) => base.map((v, j) => (names[j] === vary ? x : v)));
      const out = run(session, rows);
      for (const [o, cols] of Object.entries(meta.onnx.outputs)) {
        if (!Array.isArray(cols)) continue;
        cols.forEach((k, j) => {
          // a spread is measured against the quantity it spreads around (as pinneapple_hub.zoo.export.output_scale)
          const ref = sw.pred[k], of = sw.pred[k.replace(/_std$/, "")] || ref;
          const span = Math.max(...of) - Math.min(...of);
          rows.forEach((_, i) => {
            const got = row(out[o], i)[j], tol = 1e-5 * Math.max(Math.abs(of[i]), span, 1e-9) + 1e-9;
            assert.ok(Math.abs(got - ref[i]) <= tol, `${name} ${vary} ${k}[${i}]: ${got} vs ${ref[i]}`);
          });
        });
      }
      rows.forEach((_, i) => {
        const v = row(out.checks, i), ev = evaluate(meta.checks, v, row(out.aux, i));
        points += 1;
        if (ev.verdict.status === sw.status[i]) return;
        // a different verdict is allowed only when a check that decides it sits at its limit
        const atLimit = meta.checks.checks.some((c, j) => (c.op === "in" ? c.limit : [c.limit])
          .some((l) => Math.abs(v[j] - l) <= CHECK_TOL * Math.max(Math.abs(l), 1)));
        assert.ok(atLimit, `${name} ${vary}[${i}] (x = ${sw.x[i]}): verdict ${ev.verdict.status} vs Python ${sw.status[i]}`);
        nearLimit += 1;
      });
    }
    assert.ok(points > 0 && nearLimit <= Math.ceil(0.01 * points), `${name}: ${nearLimit} of ${points} verdicts differ`);
    // the precomputed sweeps show where the model stops being trusted
    assert.ok(Object.values(pre.sweeps).some((sw) => sw.status.includes("fail")), `${name}: no failing stretch`);
  });
}
