// Binary STL -> THREE.BufferGeometry (position + facet normals), for pages that fetch a part from their server.
// Usage: import { parseBinarySTL } from "/shared/stl.js"; const geom = parseBinarySTL(await res.arrayBuffer());
import * as THREE from "three";

export function parseBinarySTL(buffer) {
  const dv = new DataView(buffer);
  if (dv.byteLength < 84) throw new Error("not a binary STL");
  const n = dv.getUint32(80, true);
  if (dv.byteLength < 84 + 50 * n) throw new Error("truncated binary STL");
  const pos = new Float32Array(n * 9);
  const nor = new Float32Array(n * 9);
  for (let i = 0; i < n; i++) {
    const o = 84 + 50 * i;
    const nx = dv.getFloat32(o, true), ny = dv.getFloat32(o + 4, true), nz = dv.getFloat32(o + 8, true);
    for (let v = 0; v < 3; v++) {
      const p = o + 12 + 12 * v, k = 9 * i + 3 * v;
      pos[k] = dv.getFloat32(p, true);
      pos[k + 1] = dv.getFloat32(p + 4, true);
      pos[k + 2] = dv.getFloat32(p + 8, true);
      nor[k] = nx; nor[k + 1] = ny; nor[k + 2] = nz;
    }
  }
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  geom.setAttribute("normal", new THREE.BufferAttribute(nor, 3));
  return geom;
}
