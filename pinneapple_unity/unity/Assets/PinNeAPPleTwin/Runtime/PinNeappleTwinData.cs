// Reads a PINNeAPPle twin folder: scene.json + geometry.glb + fields.bin (+ pinneapple_unity.json).
//
// The same three data files the web viewer reads (pinneapple-twin3d/1), parsed here without Unity types so the whole
// data layer builds and runs on the command line -- `dotnet run` against a folder exported by
// `pinneapple_unity.export_for_unity` is the check that Unity will see what Python wrote.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace PinNeAPPle
{
    public sealed class TwinPart
    {
        public string Name = "";
        public string Group = "";
        public float[] Color = { 0.75f, 0.77f, 0.80f };
        public int MeshIndex;
        public int Vertices;
        public int Triangles;
        public float[] Motion;        // Steps * 7 (position xyz + quaternion wxyz), null when the part is static
        public float[] Visibility;    // Steps (1 drawn, 0 hidden), null when always drawn
        public GlbMesh Mesh;
    }

    public sealed class TwinField
    {
        public string Part = "";
        public string Name = "";
        public string Unit = "";
        public int Steps = 1;
        public int Count;
        public int Offset;            // byte offset into fields.bin (0 for a field that arrived over the bridge)
        public float Min;
        public float Max;
        public bool Computed;         // true when the values came from a job (the bridge), not from fields.bin
        byte[] _bin;
        float[] _values;              // (Steps * Count) row-major, for a field computed at runtime

        internal void Bind(byte[] bin) { _bin = bin; }

        /// <summary>Attach values that did not come from fields.bin (a bridge answer) and take their range.</summary>
        internal void BindValues(float[] values, int steps, int count)
        {
            if (values == null || values.Length != steps * count)
                throw new ArgumentException("a field of " + steps + " x " + count + " needs " + steps * count
                                            + " values, got " + (values == null ? 0 : values.Length));
            _values = values;
            _bin = null;
            Steps = steps;
            Count = count;
            Computed = true;
            float lo = float.MaxValue, hi = float.MinValue;
            for (int i = 0; i < values.Length; i++)
            {
                float v = values[i];
                if (float.IsNaN(v) || float.IsInfinity(v)) continue;
                if (v < lo) lo = v;
                if (v > hi) hi = v;
            }
            Min = lo <= hi ? lo : 0f;
            Max = lo <= hi ? hi : 0f;
        }

        public float Value(int step, int vertex)
        {
            int index = step * Count + vertex;
            if (_values != null)
                return index >= 0 && index < _values.Length ? _values[index] : 0f;
            if (_bin == null) return 0f;
            int at = Offset + 4 * index;
            if (at < 0 || at + 4 > _bin.Length) return 0f;
            if (!BitConverter.IsLittleEndian)
            {
                var tmp = new byte[4];
                Array.Copy(_bin, at, tmp, 0, 4); Array.Reverse(tmp);
                return BitConverter.ToSingle(tmp, 0);
            }
            return BitConverter.ToSingle(_bin, at);
        }

        public void FillStep(int step, float[] dst)
        {
            int n = Math.Min(Count, dst.Length);
            for (int v = 0; v < n; v++) dst[v] = Value(step, v);
        }
    }

    public sealed class TwinSensor
    {
        public string Id = "";
        public string Label = "";
        public string Unit = "";
        public string Quantity = "";
        public float[] Position = new float[3];
        public float[] Series;
        public float[] Envelope;      // low, high (alarm band), null when the sensor has none
    }

    public sealed class TwinData
    {
        public string Title = "";
        public string Source = "";
        public string LengthUnit = "m";
        public string TimeUnit = "s";
        public string UpAxis = "z";
        public string Folder = "";
        public double[] Times = new double[0];
        public List<TwinPart> Parts = new List<TwinPart>();
        public List<TwinField> Fields = new List<TwinField>();
        public List<TwinSensor> Sensors = new List<TwinSensor>();

        public int StepCount { get { return Times.Length > 0 ? Times.Length : 1; } }

        public static TwinData Load(string folder)
        {
            var data = new TwinData();
            data.Folder = folder;
            string scenePath = Path.Combine(folder, "scene.json");
            if (!File.Exists(scenePath))
            {
                // an export root (the folder export_for_unity wrote) also works: its data sits under Assets/TwinData
                string nested = Path.Combine(folder, "Assets", "TwinData");
                if (File.Exists(Path.Combine(nested, "scene.json"))) folder = nested;
                scenePath = Path.Combine(folder, "scene.json");
            }
            data.Folder = folder;
            if (!File.Exists(scenePath)) throw new FileNotFoundException("no scene.json in " + folder, scenePath);
            var man = MiniJson.Obj(MiniJson.Parse(File.ReadAllText(scenePath)));
            string format = MiniJson.Str(man, "format");
            if (format != "pinneapple-twin3d/1")
                throw new InvalidDataException("scene.json format is '" + format + "', expected 'pinneapple-twin3d/1'");
            data.Title = MiniJson.Str(man, "title");
            data.Source = MiniJson.Str(man, "source");
            data.LengthUnit = MiniJson.Str(man, "length_unit", "m");
            data.TimeUnit = MiniJson.Str(man, "time_unit", "s");
            data.Times = MiniJson.Nums(man, "times") ?? new double[0];

            string geometry = Path.Combine(folder, MiniJson.Str(man, "geometry", "geometry.glb"));
            var meshes = Glb.Read(geometry);
            foreach (object o in MiniJson.Arr(MiniJson.Get(man, "parts")) ?? new List<object>())
            {
                var p = new TwinPart();
                p.Name = MiniJson.Str(o, "name");
                p.Group = MiniJson.Str(o, "group");
                p.Color = MiniJson.Floats(o, "color") ?? p.Color;
                p.MeshIndex = (int)MiniJson.Num(o, "mesh_index");
                p.Vertices = (int)MiniJson.Num(o, "vertices");
                p.Triangles = (int)MiniJson.Num(o, "triangles");
                if (p.MeshIndex >= 0 && p.MeshIndex < meshes.Count) p.Mesh = meshes[p.MeshIndex];
                data.Parts.Add(p);
            }

            string binPath = Path.Combine(folder, MiniJson.Str(man, "fields_file", "fields.bin"));
            byte[] bin = File.Exists(binPath) ? File.ReadAllBytes(binPath) : null;
            foreach (object o in MiniJson.Arr(MiniJson.Get(man, "fields")) ?? new List<object>())
            {
                var f = new TwinField();
                f.Part = MiniJson.Str(o, "part");
                f.Name = MiniJson.Str(o, "name");
                f.Unit = MiniJson.Str(o, "unit");
                f.Steps = Math.Max(1, (int)MiniJson.Num(o, "steps", 1));
                f.Count = (int)MiniJson.Num(o, "count");
                f.Offset = (int)MiniJson.Num(o, "offset");
                f.Min = (float)MiniJson.Num(o, "min");
                f.Max = (float)MiniJson.Num(o, "max");
                f.Bind(bin);
                data.Fields.Add(f);
            }
            foreach (object o in MiniJson.Arr(MiniJson.Get(man, "motions")) ?? new List<object>())
            {
                TwinPart part = data.Part(MiniJson.Str(o, "part"));
                int steps = (int)MiniJson.Num(o, "steps");
                if (part != null && bin != null)
                    part.Motion = ReadFloats(bin, (int)MiniJson.Num(o, "offset"), steps * 7 * part.Vertices);
            }
            foreach (object o in MiniJson.Arr(MiniJson.Get(man, "visibility")) ?? new List<object>())
            {
                TwinPart part = data.Part(MiniJson.Str(o, "part"));
                int steps = (int)MiniJson.Num(o, "steps");
                if (part != null && bin != null)
                    part.Visibility = ReadFloats(bin, (int)MiniJson.Num(o, "offset"), steps * part.Vertices);
            }
            foreach (object o in MiniJson.Arr(MiniJson.Get(man, "sensors")) ?? new List<object>())
            {
                var s = new TwinSensor();
                s.Id = MiniJson.Str(o, "id");
                s.Label = MiniJson.Str(o, "label", s.Id);
                s.Unit = MiniJson.Str(o, "unit");
                s.Quantity = MiniJson.Str(o, "quantity");
                s.Position = MiniJson.Floats(o, "position") ?? new float[3];
                s.Series = MiniJson.Floats(o, "series");
                s.Envelope = MiniJson.Floats(o, "envelope");
                data.Sensors.Add(s);
            }

            string unityPath = Path.Combine(folder, "pinneapple_unity.json");
            if (File.Exists(unityPath))
            {
                var meta = MiniJson.Obj(MiniJson.Parse(File.ReadAllText(unityPath)));
                data.UpAxis = MiniJson.Str(meta, "up_axis", "z");
            }
            return data;
        }

        static float[] ReadFloats(byte[] bin, int offset, int count)
        {
            var outv = new float[count];
            for (int i = 0; i < count; i++)
            {
                int at = offset + 4 * i;
                if (at < 0 || at + 4 > bin.Length) break;
                if (!BitConverter.IsLittleEndian)
                {
                    var tmp = new byte[4];
                    Array.Copy(bin, at, tmp, 0, 4); Array.Reverse(tmp);
                    outv[i] = BitConverter.ToSingle(tmp, 0);
                }
                else outv[i] = BitConverter.ToSingle(bin, at);
            }
            return outv;
        }

        public TwinPart Part(string name)
        {
            foreach (TwinPart p in Parts) if (p.Name == name) return p;
            return null;
        }

        public TwinField Field(string part, string name)
        {
            foreach (TwinField f in Fields) if (f.Part == part && f.Name == name) return f;
            return null;
        }

        /// <summary>Add (or replace) a field that was computed at runtime -- a bridge answer -- on a part.
        ///
        /// The values are ``steps * count`` floats, row-major, exactly the payload ``/v1/run`` returns; the range the
        /// legend uses comes from the values themselves, not from a manifest. Nothing is written to disk: this is the
        /// in-memory twin the view is showing.
        /// </summary>
        public TwinField AddField(string part, string name, int steps, int count, string unit, float[] values)
        {
            TwinPart p = Part(part);
            if (p == null) throw new ArgumentException("no part '" + part + "' in this twin");
            if (count != p.Vertices)
                throw new ArgumentException("field '" + name + "' has " + count + " values per step, part '" + part +
                                            "' has " + p.Vertices + " vertices");
            if (steps != StepCount && steps != 1)
                throw new ArgumentException("field '" + name + "' has " + steps + " steps, the twin " + StepCount);
            TwinField existing = Field(part, name);
            if (existing != null) Fields.Remove(existing);
            var f = new TwinField { Part = part, Name = name, Unit = unit ?? "" };
            f.BindValues(values, steps, count);
            Fields.Add(f);
            return f;
        }

        public List<string> FieldNames()
        {
            var names = new List<string>();
            foreach (TwinField f in Fields) if (!names.Contains(f.Name)) names.Add(f.Name);
            return names;
        }

        /// <summary>The unit of a field (the first part that carries it declares it), or "".</summary>
        public string Unit(string name)
        {
            foreach (TwinField f in Fields) if (f.Name == name) return f.Unit;
            return "";
        }

        /// <summary>The min/max of a field over every part and step (as written by the exporter), or null.</summary>
        public float[] Range(string name)
        {
            bool any = false;
            float lo = float.MaxValue, hi = float.MinValue;
            foreach (TwinField f in Fields)
            {
                if (f.Name != name) continue;
                any = true;
                if (f.Min < lo) lo = f.Min;
                if (f.Max > hi) hi = f.Max;
            }
            return any ? new[] { lo, hi } : null;
        }

        /// <summary>What does not add up in this folder (empty when it does).</summary>
        public List<string> Verify()
        {
            var problems = new List<string>();
            if (Parts.Count == 0) problems.Add("no parts in scene.json");
            int meshes = 0;
            foreach (TwinPart p in Parts)
            {
                if (p.Mesh == null) { problems.Add("part '" + p.Name + "' has no geometry (mesh_index " + p.MeshIndex + ")"); continue; }
                meshes++;
                if (p.Mesh.VertexCount != p.Vertices)
                    problems.Add("part '" + p.Name + "': glb has " + p.Mesh.VertexCount + " vertices, scene.json says " + p.Vertices);
            }
            foreach (TwinField f in Fields)
            {
                if (Part(f.Part) == null) problems.Add("field '" + f.Name + "' names an unknown part '" + f.Part + "'");
                if (f.Steps != StepCount && f.Steps != 1)
                    problems.Add("field '" + f.Part + "/" + f.Name + "' has " + f.Steps + " steps, the scene " + StepCount);
                if (f.Count > 0)
                    for (int s = 0; s < f.Steps; s += Math.Max(1, f.Steps - 1))
                        foreach (int v in new[] { 0, f.Count / 2, f.Count - 1 })
                            if (!IsFinite(f.Value(s, v)))
                            {
                                problems.Add("field '" + f.Part + "/" + f.Name + "' is not finite at step " + s);
                                s = f.Steps;                       // one report per field
                                break;
                            }
            }
            return problems;
        }

        static bool IsFinite(float v) { return !float.IsNaN(v) && !float.IsInfinity(v); }

        public override string ToString()
        {
            return Title + ": " + Parts.Count + " part(s), " + Fields.Count + " field(s), " + StepCount + " step(s)";
        }
    }
}
