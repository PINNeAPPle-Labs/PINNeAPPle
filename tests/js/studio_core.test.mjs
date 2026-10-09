// node --test tests/js/studio_core.test.mjs : the pure part of the studio viewer core (no browser, no three.js)
import { test } from "node:test";
import assert from "node:assert/strict";
import * as core from "../../pinneapple_tools/visualization/studio/web/studio-core.js";

test("jet ends: dark blue low, dark red high, clamped", () => {
  assert.deepEqual(core.jet(0).map((v) => Math.round(v * 255)), [0, 0, 143]);
  assert.deepEqual(core.jet(1).map((v) => Math.round(v * 255)), [128, 0, 0]);
  assert.deepEqual(core.jet(-3), core.jet(0));
  assert.deepEqual(core.jet(NaN), core.jet(0));
  assert.deepEqual(core.jet(0.5).map((v) => Math.round(v * 255)), [128, 255, 128]);
});

test("ticks and decimals", () => {
  assert.deepEqual(core.ticks(-0.5, 0.597), ["-0.50", "-0.23", "0.05", "0.32", "0.60"]);
  assert.deepEqual(core.ticks(0, 2e5), ["0", "5.00e+4", "1.00e+5", "1.50e+5", "2.00e+5"]);
  assert.equal(core.decimals(0, 1), 2);
  assert.equal(core.decimals(0, 1000), 0);
});

test("colour bar html has the title, the five ticks and the end words", () => {
  const h = core.colorbarHTML({ title: "Cp", lo: -1, hi: 1, loTxt: "suction", hiTxt: "stagnation" });
  for (const t of ["Cp", "-1.00", "-0.50", "0.00", "0.50", "1.00", "suction", "stagnation", "linear-gradient"]) assert.ok(h.includes(t), t);
});

test("auto range clips outliers, full keeps them, flat data gets a span", () => {
  const v = Array.from({ length: 1000 }, (_, i) => i / 999);
  v[0] = -100; v[999] = 100;
  const [lo, hi] = core.autoRange(v, "auto");
  assert.ok(lo > -1 && hi < 1);
  assert.deepEqual(core.autoRange(v, "full"), [-100, 100]);
  const [a, b] = core.autoRange([2, 2, 2]);
  assert.ok(a < 2 && b > 2);
  assert.deepEqual(core.autoRange([null, NaN, 1, 3], "full"), [1, 3]);
});

test("colour array is linear and spans the scale", () => {
  const c = core.colorArray([0, 1], 0, 1);
  assert.equal(c.length, 6);
  assert.ok(Math.abs(c[2] - Math.pow(143 / 255, 2.2)) < 1e-6 && Math.abs(c[3] - Math.pow(128 / 255, 2.2)) < 1e-6);
});

test("slice image: rows flipped, nulls transparent, odd mirror changes sign", () => {
  const im = core.sliceImage([[0, null], [1, 1]], 0, 1);
  assert.equal(im.width, 2); assert.equal(im.height, 2);
  assert.equal(im.data[4 * 3 + 3], 0);                       // row 0 is the bottom line of the image
  assert.equal(im.data[4 * 2 + 2], 143);                     // value 0 -> dark blue
  const m = core.sliceImage([[1]], -1, 1, "odd");
  assert.equal(m.width, 2);
  assert.equal(m.data[0], 0); assert.equal(m.data[2], 143);  // left half: -1 -> dark blue
  assert.equal(m.data[4], 128);                              // right half: +1 -> dark red
});

test("slice stacks group and order along the normal", () => {
  const s = (name, x, group) => ({ name, group, origin: [x, 0, 0], u: [0, 1, 0], v: [0, 0, 1] });
  const st = core.sliceStacks([s("b", 2, "wake"), s("a", 1, "wake"), s("mid", 0)]);
  assert.deepEqual(Object.keys(st).sort(), ["mid", "wake"]);
  assert.deepEqual(st.wake.map((x) => x.name), ["a", "b"]);
});

test("line thinning and repeat removal", () => {
  assert.deepEqual(core.thin(10, 1), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
  assert.deepEqual(core.thin(10, 0.3), [0, 3, 6]);
  assert.deepEqual(core.thin(10, 0), [0]);
  assert.deepEqual(core.thin(0, 1), []);
  const [P, S] = core.dropRepeats([[0, 0, 0], [1, 0, 0], [1, 0, 0]], [1, 2, 2]);
  assert.equal(P.length, 2); assert.deepEqual(S, [1, 2]);
});
