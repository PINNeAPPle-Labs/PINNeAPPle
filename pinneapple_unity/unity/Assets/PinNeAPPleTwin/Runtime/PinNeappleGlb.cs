// Binary glTF (.glb) reading and writing for the PINNeAPPle twin files.
//
// Read:  scene geometry exported by PINNeAPPle (one mesh + node per part) and by Unity's exporter (node transforms,
//        TRS or matrix, merged per primitive into one part each).
// Write: the geometry half of `pinneapple_link.json` -- what the editor menu exports from a selection in Unity and
//        what `pinneapple_unity.emulate_unity_export` writes on the Python side, so the same files are produced on
//        both ends of the lane.
//
// No Unity types and no glTF package: it compiles with plain dotnet (see unity/tests), which is how the reader is
// checked against files the Python exporter writes.
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace PinNeAPPle
{
    public sealed class GlbMesh
    {
        public string Name = "part";
        public float[] Positions = new float[0];      // 3 per vertex
        public float[] Normals;                       // 3 per vertex, may be null
        public int[] Indices = new int[0];
        public float[] Color = { 0.75f, 0.77f, 0.80f, 1f };

        public int VertexCount { get { return Positions.Length / 3; } }
        public int TriangleCount { get { return Indices.Length / 3; } }
    }

    public static class Glb
    {
        const int MAGIC = 0x46546C67;                 // 'glTF'
        const int CHUNK_JSON = 0x4E4F534A;            // 'JSON'
        const int CHUNK_BIN = 0x004E4942;             // 'BIN\0'

        // ------------------------------------------------------------------ read
        public static List<GlbMesh> Read(string path)
        {
            return ReadBytes(File.ReadAllBytes(path));
        }

        public static List<GlbMesh> ReadBytes(byte[] bytes)
        {
            if (bytes.Length < 20) throw new InvalidDataException("not a glb: too short");
            if (ReadInt(bytes, 0) != MAGIC) throw new InvalidDataException("not a glb: no 'glTF' magic");
            int version = ReadInt(bytes, 4);
            if (version != 2) throw new InvalidDataException("glb version " + version + ", expected 2");
            int total = ReadInt(bytes, 8);
            if (total <= 0 || total > bytes.Length) total = bytes.Length;

            string json = null;
            byte[] bin = null;
            int off = 12;
            while (off + 8 <= total)
            {
                int len = ReadInt(bytes, off);
                int type = ReadInt(bytes, off + 4);
                off += 8;
                if (len < 0 || off + len > bytes.Length) break;
                if (type == CHUNK_JSON) json = Encoding.UTF8.GetString(bytes, off, len);
                else if (type == CHUNK_BIN) { bin = new byte[len]; Array.Copy(bytes, off, bin, 0, len); }
                off += len;                              // chunkLength already includes the 4-byte padding
            }
            if (json == null) throw new InvalidDataException("glb has no JSON chunk");
            return ReadDocument(MiniJson.Obj(MiniJson.Parse(json)), bin);
        }

        static List<GlbMesh> ReadDocument(Dictionary<string, object> gl, byte[] bin)
        {
            var accessors = ListOf(gl, "accessors");
            var views = ListOf(gl, "bufferViews");
            var meshes = ListOf(gl, "meshes");
            var nodes = ListOf(gl, "nodes");
            var materials = ListOf(gl, "materials");
            var scenes = ListOf(gl, "scenes");
            var outList = new List<GlbMesh>();

            List<int> roots = new List<int>();
            if (scenes.Count > 0)
            {
                var idx = MiniJson.Nums(scenes[0], "nodes");
                if (idx != null) foreach (double d in idx) roots.Add((int)d);
            }
            if (roots.Count == 0) for (int i = 0; i < nodes.Count; i++) roots.Add(i);

            var seen = new HashSet<int>();
            foreach (int r in roots) Walk(r, Identity(), nodes, meshes, materials, accessors, views, bin, outList, seen, 0);
            if (outList.Count == 0)
            {
                // a document with meshes but no reachable node: take the meshes as they are
                for (int i = 0; i < meshes.Count; i++)
                    outList.Add(ReadMesh(MiniJson.Obj(meshes[i]), "part" + i, Identity(), materials, accessors, views, bin));
            }
            return outList;
        }

        static void Walk(int nodeIndex, float[] parent, List<object> nodes, List<object> meshes, List<object> materials,
                         List<object> accessors, List<object> views, byte[] bin, List<GlbMesh> outList, HashSet<int> seen,
                         int depth)
        {
            if (nodeIndex < 0 || nodeIndex >= nodes.Count || depth > 64 || !seen.Add(nodeIndex)) return;
            var node = MiniJson.Obj(nodes[nodeIndex]);
            float[] world = Multiply(parent, Local(node));
            double meshIndex = MiniJson.Num(node, "mesh", -1);
            if (meshIndex >= 0 && (int)meshIndex < meshes.Count)
            {
                string name = MiniJson.Str(node, "name", "");
                var m = ReadMesh(MiniJson.Obj(meshes[(int)meshIndex]), name, world, materials, accessors, views, bin);
                if (name.Length == 0) m.Name = MiniJson.Str(meshes[(int)meshIndex], "name", m.Name);
                outList.Add(m);
            }
            var children = MiniJson.Nums(node, "children");
            if (children != null) foreach (double c in children) Walk((int)c, world, nodes, meshes, materials,
                                                                      accessors, views, bin, outList, seen, depth + 1);
        }

        static GlbMesh ReadMesh(Dictionary<string, object> mesh, string name, float[] world, List<object> materials,
                                List<object> accessors, List<object> views, byte[] bin)
        {
            var prims = MiniJson.Arr(mesh != null ? MiniJson.Get(mesh, "primitives") : null);
            var positions = new List<float>();
            var normals = new List<float>();
            var indices = new List<int>();
            bool hasNormals = true;
            float[] color = { 0.75f, 0.77f, 0.80f, 1f };
            if (prims != null)
            {
                foreach (object po in prims)
                {
                    var p = MiniJson.Obj(po);
                    var attrs = MiniJson.Obj(MiniJson.Get(p, "attributes"));
                    int posAcc = (int)MiniJson.Num(attrs, "POSITION", -1);
                    if (posAcc < 0) continue;
                    int baseVertex = positions.Count / 3;
                    float[] pos = ReadFloats(accessors, views, bin, posAcc, 3);
                    int vcount = pos.Length / 3;
                    for (int v = 0; v < vcount; v++)
                    {
                        float x, y, z;
                        TransformPoint(world, pos[v * 3], pos[v * 3 + 1], pos[v * 3 + 2], out x, out y, out z);
                        positions.Add(x); positions.Add(y); positions.Add(z);
                    }
                    int nrmAcc = (int)MiniJson.Num(attrs, "NORMAL", -1);
                    if (nrmAcc >= 0)
                    {
                        float[] nrm = ReadFloats(accessors, views, bin, nrmAcc, 3);
                        for (int v = 0; v < vcount; v++)
                        {
                            float x, y, z;
                            TransformDirection(world, nrm[v * 3], nrm[v * 3 + 1], nrm[v * 3 + 2], out x, out y, out z);
                            normals.Add(x); normals.Add(y); normals.Add(z);
                        }
                    }
                    else hasNormals = false;

                    int idxAcc = (int)MiniJson.Num(p, "indices", -1);
                    if (idxAcc >= 0)
                    {
                        int[] idx = ReadIndices(accessors, views, bin, idxAcc);
                        for (int k = 0; k < idx.Length; k++) indices.Add(idx[k] + baseVertex);
                    }
                    else
                    {
                        for (int k = 0; k < vcount; k++) indices.Add(baseVertex + k);
                    }
                    int matIndex = (int)MiniJson.Num(p, "material", -1);
                    if (matIndex >= 0 && matIndex < materials.Count)
                    {
                        var pbr = MiniJson.Obj(MiniJson.Get(materials[matIndex], "pbrMetallicRoughness"));
                        var c = MiniJson.Floats(pbr, "baseColorFactor");
                        if (c != null && c.Length >= 3) color = new[] { c[0], c[1], c[2], c.Length > 3 ? c[3] : 1f };
                    }
                }
            }
            var outMesh = new GlbMesh();
            outMesh.Name = name.Length > 0 ? name : "part";
            outMesh.Positions = positions.ToArray();
            outMesh.Normals = hasNormals && normals.Count == positions.Count ? normals.ToArray() : ComputeNormals(positions, indices);
            outMesh.Indices = indices.ToArray();
            outMesh.Color = color;
            return outMesh;
        }

        static List<object> ListOf(Dictionary<string, object> gl, string key)
        {
            var a = MiniJson.Arr(gl != null ? MiniJson.Get(gl, key) : null);
            return a ?? new List<object>();
        }

        static float[] ReadFloats(List<object> accessors, List<object> views, byte[] bin, int index, int ncomp)
        {
            var a = MiniJson.Obj(accessors[index]);
            int count = (int)MiniJson.Num(a, "count");
            int bvIndex = (int)MiniJson.Num(a, "bufferView", -1);
            if (bvIndex < 0 || bin == null) return new float[count * ncomp];
            var bv = MiniJson.Obj(views[bvIndex]);
            int baseOff = (int)MiniJson.Num(bv, "byteOffset") + (int)MiniJson.Num(a, "byteOffset");
            int stride = (int)MiniJson.Num(bv, "byteStride");
            int step = stride > 0 ? stride : 4 * ncomp;
            var outv = new float[count * ncomp];
            for (int i = 0; i < count; i++)
                for (int c = 0; c < ncomp; c++)
                {
                    int at = baseOff + i * step + c * 4;
                    if (at + 4 > bin.Length) return outv;
                    outv[i * ncomp + c] = ReadFloat(bin, at);
                }
            return outv;
        }

        static int[] ReadIndices(List<object> accessors, List<object> views, byte[] bin, int index)
        {
            var a = MiniJson.Obj(accessors[index]);
            int count = (int)MiniJson.Num(a, "count");
            int bvIndex = (int)MiniJson.Num(a, "bufferView", -1);
            int componentType = (int)MiniJson.Num(a, "componentType", 5125);
            int size = componentType == 5121 ? 1 : componentType == 5123 ? 2 : 4;
            var outv = new int[count];
            if (bvIndex < 0 || bin == null) return outv;
            var bv = MiniJson.Obj(views[bvIndex]);
            int baseOff = (int)MiniJson.Num(bv, "byteOffset") + (int)MiniJson.Num(a, "byteOffset");
            int stride = (int)MiniJson.Num(bv, "byteStride");
            int step = stride > 0 ? stride : size;
            for (int i = 0; i < count; i++)
            {
                int at = baseOff + i * step;
                if (at + size > bin.Length) return outv;
                outv[i] = size == 1 ? bin[at] : size == 2 ? bin[at] | (bin[at + 1] << 8)
                         : bin[at] | (bin[at + 1] << 8) | (bin[at + 2] << 16) | (bin[at + 3] << 24);
            }
            return outv;
        }

        // ------------------------------------------------------------------ write
        public static void Write(string path, IList<GlbMesh> meshes, string generator = "PINNeAPPle")
        {
            File.WriteAllBytes(path, WriteBytes(meshes, generator));
        }

        public static byte[] WriteBytes(IList<GlbMesh> meshes, string generator = "PINNeAPPle")
        {
            var bin = new MemoryStream();
            var sb = new StringBuilder();
            var views = new StringBuilder();
            var accessors = new StringBuilder();
            var nodes = new StringBuilder();
            var meshesJson = new StringBuilder();
            var materials = new StringBuilder();
            int nViews = 0, nAcc = 0;

            for (int m = 0; m < meshes.Count; m++)
            {
                var mesh = meshes[m];
                int vcount = mesh.Positions.Length / 3;
                if (vcount == 0) throw new ArgumentException("mesh " + m + " has no vertices");
                if (mesh.Indices.Length == 0) throw new ArgumentException("mesh " + m + " has no triangles");

                int posOff = (int)bin.Length;
                float minX = float.MaxValue, minY = float.MaxValue, minZ = float.MaxValue;
                float maxX = float.MinValue, maxY = float.MinValue, maxZ = float.MinValue;
                for (int v = 0; v < vcount; v++)
                {
                    float x = mesh.Positions[v * 3], y = mesh.Positions[v * 3 + 1], z = mesh.Positions[v * 3 + 2];
                    WriteFloat(bin, x); WriteFloat(bin, y); WriteFloat(bin, z);
                    if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
                    if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
                }
                Append(views, nViews, posOff, vcount * 12, 34962);
                Append(accessors, nAcc, nViews, vcount, 5126, "VEC3",
                       "[" + F(minX) + "," + F(minY) + "," + F(minZ) + "]", "[" + F(maxX) + "," + F(maxY) + "," + F(maxZ) + "]");
                int posAcc = nAcc; nViews++; nAcc++;

                int nrmAcc = -1;
                if (mesh.Normals != null && mesh.Normals.Length == mesh.Positions.Length)
                {
                    int nrmOff = (int)bin.Length;
                    for (int v = 0; v < vcount * 3; v++) WriteFloat(bin, mesh.Normals[v]);
                    Append(views, nViews, nrmOff, vcount * 12, 34962);
                    Append(accessors, nAcc, nViews, vcount, 5126, "VEC3", null, null);
                    nrmAcc = nAcc; nViews++; nAcc++;
                }
                int idxOff = (int)bin.Length;
                foreach (int i in mesh.Indices) WriteInt(bin, i);
                Append(views, nViews, idxOff, mesh.Indices.Length * 4, 34963);
                Append(accessors, nAcc, nViews, mesh.Indices.Length, 5125, "SCALAR", null, null);
                int idxAcc = nAcc; nViews++; nAcc++;

                var pbr = "{\"baseColorFactor\":[" + F(mesh.Color[0]) + "," + F(mesh.Color[1]) + "," + F(mesh.Color[2]) + ","
                          + F(mesh.Color.Length > 3 ? mesh.Color[3] : 1f) + "],\"metallicFactor\":0.1,\"roughnessFactor\":0.7}";
                if (m > 0) materials.Append(',');
                materials.Append("{\"name\":").Append(MiniJson.Quote("part" + m)).Append(",\"doubleSided\":true,\"pbrMetallicRoughness\":")
                         .Append(pbr).Append('}');

                if (m > 0) meshesJson.Append(',');
                meshesJson.Append("{\"name\":").Append(MiniJson.Quote(mesh.Name)).Append(",\"primitives\":[{\"attributes\":{\"POSITION\":")
                          .Append(posAcc).Append(nrmAcc >= 0 ? ",\"NORMAL\":" + nrmAcc : "")
                          .Append("},\"indices\":").Append(idxAcc).Append(",\"material\":").Append(m)
                          .Append(",\"mode\":4}]}");

                if (m > 0) nodes.Append(',');
                nodes.Append("{\"name\":").Append(MiniJson.Quote(mesh.Name)).Append(",\"mesh\":").Append(m).Append('}');
            }

            while (bin.Length % 4 != 0) bin.WriteByte(0);
            var binBytes = bin.ToArray();
            string json = "{\"asset\":{\"version\":\"2.0\",\"generator\":" + MiniJson.Quote(generator) + "},"
                          + "\"scene\":0,\"scenes\":[{\"nodes\":[" + NodeRange(meshes.Count) + "]}],"
                          + "\"nodes\":[" + nodes + "],\"meshes\":[" + meshesJson + "],"
                          + "\"materials\":[" + materials + "],\"accessors\":[" + accessors + "],"
                          + "\"bufferViews\":[" + views + "],"
                          + "\"buffers\":[{\"byteLength\":" + binBytes.Length + "}]}";
            var jsonBytes = Encoding.UTF8.GetBytes(json);
            int jsonPad = (4 - jsonBytes.Length % 4) % 4;
            int totalLen = 12 + 8 + jsonBytes.Length + jsonPad + 8 + binBytes.Length;

            var outBytes = new MemoryStream();
            WriteInt(outBytes, MAGIC);
            WriteInt(outBytes, 2);
            WriteInt(outBytes, totalLen);
            WriteInt(outBytes, jsonBytes.Length + jsonPad);
            WriteInt(outBytes, CHUNK_JSON);
            outBytes.Write(jsonBytes, 0, jsonBytes.Length);
            for (int i = 0; i < jsonPad; i++) outBytes.WriteByte(0x20);
            WriteInt(outBytes, binBytes.Length);
            WriteInt(outBytes, CHUNK_BIN);
            outBytes.Write(binBytes, 0, binBytes.Length);
            return outBytes.ToArray();
        }

        static string NodeRange(int n)
        {
            var sb = new StringBuilder();
            for (int i = 0; i < n; i++) { if (i > 0) sb.Append(','); sb.Append(i); }
            return sb.ToString();
        }

        static void Append(StringBuilder sb, int index, int offset, int length, int target)
        {
            if (index > 0) sb.Append(',');
            sb.Append("{\"buffer\":0,\"byteOffset\":").Append(offset).Append(",\"byteLength\":").Append(length)
              .Append(",\"target\":").Append(target).Append('}');
        }

        static void Append(StringBuilder sb, int index, int view, int count, int componentType, string type,
                           string min, string max)
        {
            if (index > 0) sb.Append(',');
            sb.Append("{\"bufferView\":").Append(view).Append(",\"componentType\":").Append(componentType)
              .Append(",\"count\":").Append(count).Append(",\"type\":\"").Append(type).Append('"');
            if (min != null) sb.Append(",\"min\":").Append(min);
            if (max != null) sb.Append(",\"max\":").Append(max);
            sb.Append('}');
        }

        static string F(float v) { return v.ToString("R", System.Globalization.CultureInfo.InvariantCulture); }

        // ------------------------------------------------------------------ maths
        static float[] Identity()
        {
            return new float[] { 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };   // row-major
        }

        static float[] Local(Dictionary<string, object> node)
        {
            if (node == null) return Identity();
            var matrix = MiniJson.Floats(node, "matrix");
            if (matrix != null && matrix.Length == 16)
            {
                var m = new float[16];                    // glTF is column-major: m[r][c] = matrix[c * 4 + r]
                for (int r = 0; r < 4; r++) for (int c = 0; c < 4; c++) m[r * 4 + c] = matrix[c * 4 + r];
                return m;
            }
            var t = MiniJson.Floats(node, "translation") ?? new float[] { 0, 0, 0 };
            var q = MiniJson.Floats(node, "rotation") ?? new float[] { 0, 0, 0, 1 };      // xyzw
            var s = MiniJson.Floats(node, "scale") ?? new float[] { 1, 1, 1 };
            float x = q[0], y = q[1], z = q[2], w = q[3];
            float xx = x * x, yy = y * y, zz = z * z, xy = x * y, xz = x * z, yz = y * z, wx = w * x, wy = w * y, wz = w * z;
            var m2 = new float[]
            {
                (1 - 2 * (yy + zz)) * s[0], 2 * (xy - wz) * s[0], 2 * (xz + wy) * s[0], t[0],
                2 * (xy + wz) * s[1], (1 - 2 * (xx + zz)) * s[1], 2 * (yz - wx) * s[1], t[1],
                2 * (xz - wy) * s[2], 2 * (yz + wx) * s[2], (1 - 2 * (xx + yy)) * s[2], t[2],
                0, 0, 0, 1,
            };
            return m2;
        }

        static float[] Multiply(float[] a, float[] b)
        {
            var o = new float[16];
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++)
                {
                    float sum = 0;
                    for (int k = 0; k < 4; k++) sum += a[r * 4 + k] * b[k * 4 + c];
                    o[r * 4 + c] = sum;
                }
            return o;
        }

        static void TransformPoint(float[] m, float x, float y, float z, out float ox, out float oy, out float oz)
        {
            ox = m[0] * x + m[1] * y + m[2] * z + m[3];
            oy = m[4] * x + m[5] * y + m[6] * z + m[7];
            oz = m[8] * x + m[9] * y + m[10] * z + m[11];
        }

        static void TransformDirection(float[] m, float x, float y, float z, out float ox, out float oy, out float oz)
        {
            ox = m[0] * x + m[1] * y + m[2] * z;
            oy = m[4] * x + m[5] * y + m[6] * z;
            oz = m[8] * x + m[9] * y + m[10] * z;
            float len = (float)Math.Sqrt(ox * ox + oy * oy + oz * oz);
            if (len > 1e-12f) { ox /= len; oy /= len; oz /= len; }
        }

        static float[] ComputeNormals(List<float> positions, List<int> indices)
        {
            int vcount = positions.Count / 3;
            var n = new float[positions.Count];
            for (int t = 0; t + 2 < indices.Count; t += 3)
            {
                int a = indices[t] * 3, b = indices[t + 1] * 3, c = indices[t + 2] * 3;
                if (a + 2 >= n.Length || b + 2 >= n.Length || c + 2 >= n.Length) continue;
                float ux = positions[b] - positions[a], uy = positions[b + 1] - positions[a + 1], uz = positions[b + 2] - positions[a + 2];
                float vx = positions[c] - positions[a], vy = positions[c + 1] - positions[a + 1], vz = positions[c + 2] - positions[a + 2];
                float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
                int[] at = { a, b, c };
                foreach (int i in at) { n[i] += nx; n[i + 1] += ny; n[i + 2] += nz; }
            }
            for (int v = 0; v < vcount; v++)
            {
                float len = (float)Math.Sqrt(n[v * 3] * n[v * 3] + n[v * 3 + 1] * n[v * 3 + 1] + n[v * 3 + 2] * n[v * 3 + 2]);
                if (len > 1e-12f) { n[v * 3] /= len; n[v * 3 + 1] /= len; n[v * 3 + 2] /= len; }
                else { n[v * 3 + 2] = 1f; }
            }
            return n;
        }

        // ------------------------------------------------------------------ little-endian io
        static int ReadInt(byte[] b, int at)
        {
            return b[at] | (b[at + 1] << 8) | (b[at + 2] << 16) | (b[at + 3] << 24);
        }

        static float ReadFloat(byte[] b, int at)
        {
            if (!BitConverter.IsLittleEndian)
            {
                var tmp = new byte[4];
                for (int i = 0; i < 4; i++) tmp[i] = b[at + 3 - i];
                return BitConverter.ToSingle(tmp, 0);
            }
            return BitConverter.ToSingle(b, at);
        }

        static void WriteFloat(Stream s, float v)
        {
            byte[] b = BitConverter.GetBytes(v);
            if (!BitConverter.IsLittleEndian) Array.Reverse(b);
            s.Write(b, 0, 4);
        }

        static void WriteInt(Stream s, int v)
        {
            byte[] b = BitConverter.GetBytes(v);
            if (!BitConverter.IsLittleEndian) Array.Reverse(b);
            s.Write(b, 0, 4);
        }
    }
}
