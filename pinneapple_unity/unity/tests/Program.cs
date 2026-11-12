// Data-layer check for the Unity package, runnable without Unity:
//
//     dotnet run --project pinneapple_unity/unity/tests -- <twin folder> [<unity link folder>]
//
// It reads a twin folder exported by `pinneapple_unity.export_for_unity` (scene.json + geometry.glb + fields.bin +
// pinneapple_unity.json), verifies it, and round-trips the geometry through the writer the editor menu uses. With a
// second argument it also reads a Unity-authored folder (geometry.glb + pinneapple_link.json) the way the editor's
// export writes it. Exit code 1 on any failure; every line it prints is what a test asserts on.
using System;
using System.Collections.Generic;
using System.IO;
using PinNeAPPle;

namespace PinNeAPPle.Tests
{
    public static class Program
    {
        static int _failed;

        public static int Main(string[] args)
        {
            if (args.Length < 1)
            {
                Console.WriteLine("usage: PinNeAppleDataTests <twin folder> [<unity link folder>]");
                return 2;
            }
            string folder = args[0];
            Console.WriteLine("twin folder: " + Path.GetFullPath(folder));

            TwinData data;
            try
            {
                data = TwinData.Load(folder);
            }
            catch (Exception exc)
            {
                Console.WriteLine("FAIL load: " + exc.Message);
                return 1;
            }
            Console.WriteLine("title: " + data.Title);
            Console.WriteLine("parts: " + data.Parts.Count + ", fields: " + data.Fields.Count + ", steps: " + data.StepCount);
            Console.WriteLine("up axis: " + data.UpAxis);
            Check(data.Parts.Count > 0, "the twin has parts");
            Check(data.StepCount > 0, "the twin has time steps");
            foreach (string p in data.Verify()) Fail(p);

            // every field is inside the range its manifest declares (what the legend will show)
            foreach (TwinField f in data.Fields)
            {
                float lo = float.MaxValue, hi = float.MinValue;
                for (int s = 0; s < f.Steps; s++)
                    for (int v = 0; v < f.Count; v += Math.Max(1, f.Count / 37))
                    {
                        float x = f.Value(s, v);
                        if (float.IsNaN(x) || float.IsInfinity(x)) { Fail("field " + f.Name + " is not finite"); break; }
                        if (x < lo) lo = x;
                        if (x > hi) hi = x;
                    }
                bool inside = lo >= f.Min - 1e-4f * Math.Max(1f, Math.Abs(f.Min)) &&
                              hi <= f.Max + 1e-4f * Math.Max(1f, Math.Abs(f.Max));
                Check(inside, "field " + f.Part + "/" + f.Name + " [" + f.Unit + "] sampled in [" + lo.ToString("G4") + ", "
                              + hi.ToString("G4") + "] within the declared [" + f.Min.ToString("G4") + ", " + f.Max.ToString("G4") + "]");
                Console.WriteLine("field " + f.Part + "/" + f.Name + " steps " + f.Steps + " count " + f.Count);
            }
            foreach (TwinSensor s in data.Sensors)
                Console.WriteLine("sensor " + s.Id + " " + (s.Series == null ? "(no series)" : s.Series.Length + " values")
                                  + (s.Envelope == null ? "" : " envelope [" + s.Envelope[0].ToString("G4") + ", "
                                  + s.Envelope[1].ToString("G4") + "]"));

            // geometry round trip: read -> write -> read (the editor menu writes what this reads)
            var meshes = new List<GlbMesh>();
            foreach (TwinPart p in data.Parts) if (p.Mesh != null) meshes.Add(p.Mesh);
            string tmp = Path.Combine(Path.GetTempPath(), "pinneapple_glb_roundtrip.glb");
            try
            {
                Glb.Write(tmp, meshes, "PinNeAppleDataTests");
                var again = Glb.Read(tmp);
                Check(again.Count == meshes.Count, "glb round trip keeps " + meshes.Count + " mesh(es) (read back " + again.Count + ")");
                for (int i = 0; i < Math.Min(again.Count, meshes.Count); i++)
                {
                    Check(again[i].VertexCount == meshes[i].VertexCount,
                          "mesh " + i + " keeps its " + meshes[i].VertexCount + " vertices");
                    Check(again[i].TriangleCount == meshes[i].TriangleCount,
                          "mesh " + i + " keeps its " + meshes[i].TriangleCount + " triangle(s)");
                    float worst = 0f;
                    for (int v = 0; v < Math.Min(again[i].Positions.Length, meshes[i].Positions.Length); v++)
                        worst = Math.Max(worst, Math.Abs(again[i].Positions[v] - meshes[i].Positions[v]));
                    Check(worst < 1e-6f, "mesh " + i + " positions survive the round trip (max delta " + worst.ToString("G3") + ")");
                }
            }
            finally
            {
                if (File.Exists(tmp)) File.Delete(tmp);
            }

            // a field that arrives at runtime (a bridge answer) joins the twin in memory; the view shows it with
            // `PinneappleTwinView.ApplyField`, which calls exactly this
            TwinPart first = null;
            foreach (TwinPart p in data.Parts) if (p.Mesh != null) { first = p; break; }
            if (first != null)
            {
                int n = first.Mesh.VertexCount;
                var computed = new float[data.StepCount * n];
                for (int k = 0; k < computed.Length; k++) computed[k] = 20f + k * 0.5f;
                TwinField runtime = data.AddField(first.Name, "computed_temperature", data.StepCount, n, "C", computed);
                Check(runtime.Computed && runtime.Steps == data.StepCount && runtime.Count == n,
                      "a runtime field keeps its shape (" + data.StepCount + " x " + n + ")");
                Check(Math.Abs(runtime.Value(0, 0) - 20f) < 1e-4f, "a runtime field reads its first value back");
                Check(Math.Abs(runtime.Value(1, 3) - computed[n + 3]) < 1e-4f, "a runtime field indexes (step, vertex)");
                float[] range = data.Range("computed_temperature");
                Check(range != null && Math.Abs(range[0] - 20f) < 1e-4f && Math.Abs(range[1] - (20f + (computed.Length - 1) * 0.5f)) < 1e-2f,
                      "the range of a runtime field comes from its values [" + range[0].ToString("G4") + ", "
                      + range[1].ToString("G4") + "]");
                Check(data.Unit("computed_temperature") == "C", "a runtime field keeps its unit");
                bool refused = false;
                try { data.AddField(first.Name, "bad", data.StepCount, n + 1, "C", computed); }
                catch (ArgumentException) { refused = true; }
                Check(refused, "a runtime field with the wrong vertex count is refused");
                bool refusedPart = false;
                try { data.AddField("no_such_part", "bad", data.StepCount, n, "C", computed); }
                catch (ArgumentException) { refusedPart = true; }
                Check(refusedPart, "a runtime field on a part that is not in the twin is refused");
                foreach (string p in data.Verify()) Fail("after AddField: " + p);
                data.Fields.Remove(runtime);
            }

            // colormap: the ends are the anchors, the middle is between them
            Rgb lo0 = Colormap.Sample("turbo", 0f), hi1 = Colormap.Sample("turbo", 1f);
            Check(Math.Abs(lo0.R - 0.18995f) < 1e-4f && Math.Abs(hi1.B - 0.01055f) < 1e-4f, "turbo maps 0 and 1 to its anchors");
            Rgb mid = Colormap.SampleValue("viridis", 0.5f, 0f, 1f);
            Check(mid.G > mid.B, "viridis at 0.5 is greener than it is blue");

            // JSON: what the bridge sends (nested arrays of objects and numbers)
            var doc = MiniJson.Obj(MiniJson.Parse("{\"a\":[{\"b\":1e-5,\"c\":[1,2,3]}],\"d\":\"x\\ny\"}"));
            var arr = MiniJson.Arr(MiniJson.Get(doc, "a"));
            Check(arr != null && arr.Count == 1 && Math.Abs(MiniJson.Num(arr[0], "b") - 1e-5) < 1e-12, "JSON reads nested objects and exponents");
            Check(MiniJson.Str(doc, "d") == "x\ny", "JSON unescapes strings");

            if (args.Length > 1)
            {
                string linkFolder = args[1];
                Console.WriteLine("unity link folder: " + Path.GetFullPath(linkFolder));
                var link = MiniJson.Obj(MiniJson.Parse(File.ReadAllText(Path.Combine(linkFolder, "pinneapple_link.json"))));
                var parts = MiniJson.Arr(MiniJson.Get(link, "parts"));
                var geometry = Glb.Read(Path.Combine(linkFolder, MiniJson.Str(link, "geometry", "scene.glb")));
                Check(parts != null && parts.Count == geometry.Count,
                      "the link names " + (parts == null ? 0 : parts.Count) + " part(s) and the glb has " + geometry.Count);
                var jobs = MiniJson.Arr(MiniJson.Get(link, "jobs"));
                Console.WriteLine("jobs in the link: " + (jobs == null ? 0 : jobs.Count));
                foreach (object j in jobs ?? new List<object>())
                    Console.WriteLine("  " + MiniJson.Str(j, "job") + " on " + MiniJson.Str(j, "part"));
            }

            Console.WriteLine(_failed == 0 ? "OK: the data layer reads the twin" : "FAILED: " + _failed + " check(s)");
            return _failed == 0 ? 0 : 1;
        }

        static void Check(bool ok, string what)
        {
            Console.WriteLine((ok ? "  ok   " : "  FAIL ") + what);
            if (!ok) _failed++;
        }

        static void Fail(string what) { Check(false, what); }
    }
}
