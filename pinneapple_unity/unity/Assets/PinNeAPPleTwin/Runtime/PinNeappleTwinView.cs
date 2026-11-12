// The Unity face of a PINNeAPPle twin: load scene.json + geometry.glb + fields.bin, build one GameObject per part,
// colour the selected field per vertex over the scene's time steps, replay rigid motions and per-step visibility,
// mark the sensors, and draw the legend with the unit and the physical range of the field.
//
// Nothing here computes physics: the numbers come from PINNeAPPle (baked in fields.bin, or live from
// PinneappleBridgeClient). This component is the viewer, so what it shows is what the library verified.
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace PinNeAPPle.Unity
{
    [AddComponentMenu("PINNeAPPle/Twin View")]
    public class PinneappleTwinView : MonoBehaviour
    {
        [Header("Twin data")]
        [Tooltip("Folder holding scene.json, geometry.glb and fields.bin. Relative paths start at the project root.")]
        public string twinFolder = "Assets/TwinData";

        [Header("Field")]
        [Tooltip("Which field to colour. Empty: the first field of the twin.")]
        public string field = "";
        public string colormap = "turbo";
        public bool colorize = true;

        [Header("Playback")]
        public bool play = true;
        public float secondsPerStep = 1f;

        [Header("Look")]
        [Range(0f, 0.2f)] public float legendHeight = 0.06f;

        [Header("Camera")]
        [Tooltip("Frame the scene's main camera on the twin when it loads: a 0.4 m part in a default scene otherwise "
                 + "reads as a line at 10 m.")]
        public bool frameCamera = true;

        TwinData _data;
        Transform _root;
        readonly List<MeshRenderer> _renderers = new List<MeshRenderer>();
        readonly Dictionary<string, Mesh> _meshes = new Dictionary<string, Mesh>();
        // One Color[] per part, reused on every step: recolouring an animated field must not allocate a new array of
        // vertex colours 30 times a second (GC spikes on a large mesh), and the buffer's size never changes.
        readonly Dictionary<string, Color[]> _colors = new Dictionary<string, Color[]>();
        readonly List<GameObject> _sensorMarks = new List<GameObject>();
        float _lo, _hi;
        int _step = -1;
        GUIStyle _label, _value;
        Texture2D _legend;

        public TwinData Data { get { return _data; } }
        public int StepCount { get { return _data == null ? 1 : _data.StepCount; } }
        public int Step { get { return _step; } }
        public string[] FieldNames { get { return _data == null ? new string[0] : _data.FieldNames().ToArray(); } }

        void Start()
        {
            Load();
        }

        void Update()
        {
            if (_data == null || !play || StepCount <= 1) return;
            int k = (int)(Time.time / Mathf.Max(0.01f, secondsPerStep)) % StepCount;
            if (k != _step) ShowStep(k);
        }

        // ------------------------------------------------------------------ load and build
        public void Load()
        {
            string folder = twinFolder;
            if (!Path.IsPathRooted(folder))
            {
                string project = Directory.GetParent(Application.dataPath).FullName;
                folder = Path.Combine(project, folder);
            }
            try
            {
                _data = TwinData.Load(folder);
            }
            catch (System.Exception exc)
            {
                Debug.LogError("PINNeAPPle twin: " + exc.Message + "\n(folder: " + folder + ")", this);
                return;
            }
            List<string> problems = _data.Verify();
            foreach (string p in problems) Debug.LogWarning("PINNeAPPle twin: " + p, this);
            Debug.Log("PINNeAPPle twin loaded: " + _data + " from " + folder, this);
            Build();
        }

        void Build()
        {
            foreach (GameObject go in _sensorMarks) if (go != null) Destroy(go);
            _sensorMarks.Clear();
            _renderers.Clear();
            _meshes.Clear();
            _colors.Clear();
            if (_root != null) Destroy(_root.gameObject);

            _root = new GameObject("PINNeAPPle twin: " + _data.Title).transform;
            _root.SetParent(transform, false);
            // A twin and OpenUSD are Z-up, Unity and glTF are Y-up: the same rotation the exporter records.
            _root.localRotation = _data.UpAxis == "z" ? Quaternion.Euler(-90f, 0f, 0f) : Quaternion.identity;

            if (string.IsNullOrEmpty(field) && _data.FieldNames().Count > 0) field = _data.FieldNames()[0];
            float[] range = _data.Range(field);
            _lo = range != null ? range[0] : 0f;
            _hi = range != null ? range[1] : 1f;

            Shader shader = FieldShader();
            foreach (TwinPart part in _data.Parts)
            {
                if (part.Mesh == null) continue;
                GameObject go = new GameObject(part.Name);
                go.transform.SetParent(_root, false);
                Mesh mesh = ToMesh(part);
                _meshes[part.Name] = mesh;
                go.AddComponent<MeshFilter>().sharedMesh = mesh;
                var mr = go.AddComponent<MeshRenderer>();
                var mat = new Material(shader != null ? shader : FallbackShader());
                mat.name = "PINNeAPPle field (" + colormap + ")";
                if (mat.HasProperty("_Color")) mat.SetColor("_Color", Color.white);
                mr.sharedMaterial = mat;
                mr.shadowCastingMode = ShadowCastingMode.Off;
                _renderers.Add(mr);
            }
            AddSensorMarks();
            if (frameCamera) FrameMainCamera();
            ShowStep(0);
        }

        /// <summary>Put the main camera in a three-quarter view at the distance that fits the twin.</summary>
        void FrameMainCamera()
        {
            Camera cam = Camera.main;
            if (cam == null)
            {
                Debug.Log("PINNeAPPle twin: no Main Camera to frame (the parts are there, just not in view)", this);
                return;
            }
            var bounds = new Bounds();
            bool any = false;
            foreach (MeshRenderer r in _renderers)
            {
                if (r == null) continue;
                if (!any) { bounds = r.bounds; any = true; }
                else bounds.Encapsulate(r.bounds);
            }
            if (!any) return;
            float radius = Mathf.Max(bounds.extents.magnitude, 0.01f);
            float distance = radius / Mathf.Sin(cam.fieldOfView * 0.5f * Mathf.Deg2Rad) * 0.8f;
            Vector3 direction = new Vector3(1.0f, 1.15f, -1.6f).normalized;
            cam.transform.position = bounds.center + direction * distance;
            cam.transform.LookAt(bounds.center);
            cam.nearClipPlane = Mathf.Max(0.01f, distance * 0.02f);
            cam.farClipPlane = Mathf.Max(20f, distance * 20f);
            Debug.Log($"PINNeAPPle twin: camera framed at {distance:G3} m from {bounds.center}", this);
        }

        /// <summary>The field shader that matches the active render pipeline.
        ///
        /// Picking "whichever exists" is wrong: a project with the URP shader imported but the Built-in pipeline
        /// active silently draws nothing (a URP shader has no pass the Built-in pipeline can run, and the console
        /// stays clean). Ask the pipeline, not the project's file list.
        /// </summary>
        static Shader FieldShader()
        {
            RenderPipelineAsset pipeline = GraphicsSettings.currentRenderPipeline;
            bool urp = pipeline != null && pipeline.GetType().Name.ToLowerInvariant().Contains("universal");
            if (pipeline != null && !urp)
                Debug.LogWarning("PINNeAPPle: this project runs " + pipeline.GetType().Name +
                                 " (only URP and Built-in are supported) -- the parts are drawn with a plain unlit " +
                                 "colour, without the field");
            string wanted = urp ? "PINNeAPPle/Field URP" : "PINNeAPPle/Field";
            Shader s = Shader.Find(wanted);
            if (s == null)
                Debug.LogWarning("PINNeAPPle: '" + wanted + "' is missing for this project (" +
                                 (urp ? "URP" : "Built-in") + ") -- the parts are drawn with a plain unlit colour, " +
                                 "without the field");
            return s;
        }

        static Shader FallbackShader()
        {
            Shader s = Shader.Find("Universal Render Pipeline/Unlit");
            if (s == null) s = Shader.Find("Unlit/Color");
            if (s == null) s = Shader.Find("Sprites/Default");
            return s;
        }

        Mesh ToMesh(TwinPart part)
        {
            var mesh = new Mesh { name = "PINNeAPPle " + part.Name };
            GlbMesh g = part.Mesh;
            int n = g.VertexCount;
            if (n > 65000) mesh.indexFormat = IndexFormat.UInt32;
            var verts = new Vector3[n];
            for (int i = 0; i < n; i++)
                verts[i] = new Vector3(g.Positions[i * 3], g.Positions[i * 3 + 1], g.Positions[i * 3 + 2]);
            var normals = new Vector3[n];
            if (g.Normals != null && g.Normals.Length == g.Positions.Length)
                for (int i = 0; i < n; i++)
                    normals[i] = new Vector3(g.Normals[i * 3], g.Normals[i * 3 + 1], g.Normals[i * 3 + 2]);
            mesh.vertices = verts;
            if (g.Normals != null) mesh.normals = normals; else mesh.RecalculateNormals();
            mesh.triangles = g.Indices;
            var colors = new Color[n];
            FillColors(part, 0, g, colors);
            mesh.colors = colors;
            _colors[part.Name] = colors;
            mesh.RecalculateBounds();
            return mesh;
        }

        /// <summary>Write one part's colours at one step into ``dst`` -- reused across steps, so a step allocates
        /// nothing; only the upload to the mesh (``SetColors``) touches the graphics buffer.</summary>
        void FillColors(TwinPart part, int step, GlbMesh g, Color[] dst)
        {
            int n = Mathf.Min(dst.Length, g.VertexCount);
            Color baseColor = new Color(g.Color[0], g.Color[1], g.Color[2], g.Color.Length > 3 ? g.Color[3] : 1f);
            TwinField f = _data.Field(part.Name, field);
            if (!colorize || f == null)
            {
                for (int i = 0; i < n; i++) dst[i] = baseColor;
                return;
            }
            int k = f.Steps > 1 ? Mathf.Clamp(step, 0, f.Steps - 1) : 0;
            for (int i = 0; i < n; i++)
            {
                Rgb c = Colormap.SampleValue(colormap, f.Value(k, i), _lo, _hi);
                dst[i] = new Color(c.R, c.G, c.B, 1f);
            }
        }

        void AddSensorMarks()
        {
            if (_data.Sensors.Count == 0) return;
            float extent = 0f;
            foreach (TwinPart part in _data.Parts)
                if (part.Mesh != null)
                    for (int i = 0; i < part.Mesh.Positions.Length; i++)
                        extent = Mathf.Max(extent, Mathf.Abs(part.Mesh.Positions[i]));
            float size = Mathf.Max(0.004f, extent * 0.04f);
            foreach (TwinSensor s in _data.Sensors)
            {
                GameObject mark = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                mark.name = "sensor " + s.Id;
                var collider = mark.GetComponent<Collider>();
                if (collider != null) Destroy(collider);
                mark.transform.SetParent(_root, false);
                mark.transform.localPosition = new Vector3(s.Position[0], s.Position[1], s.Position[2]);
                mark.transform.localScale = Vector3.one * size;
                var mr = mark.GetComponent<MeshRenderer>();
                var mat = new Material(FallbackShader());
                mat.color = new Color(1f, 0.85f, 0.1f, 1f);
                mr.sharedMaterial = mat;
                _sensorMarks.Add(mark);
            }
        }

        // ------------------------------------------------------------------ playback
        public void ShowStep(int step)
        {
            if (_data == null) return;
            _step = ((step % StepCount) + StepCount) % StepCount;
            int i = 0;
            foreach (TwinPart part in _data.Parts)
            {
                if (part.Mesh == null) continue;
                Transform t = _root.Find(part.Name);
                if (t != null && part.Motion != null && part.Motion.Length >= (_step + 1) * 7)
                {
                    int at = _step * 7;
                    t.localPosition = new Vector3(part.Motion[at], part.Motion[at + 1], part.Motion[at + 2]);
                    t.localRotation = new Quaternion(part.Motion[at + 4], part.Motion[at + 5], part.Motion[at + 6],
                                                     part.Motion[at + 3]);
                }
                if (i < _renderers.Count)
                {
                    if (part.Visibility != null && part.Visibility.Length > _step * Mathf.Max(1, part.Vertices))
                        _renderers[i].enabled = part.Visibility[_step * Mathf.Max(1, part.Vertices)] > 0.5f;
                    Color[] colors;
                    if (_colors.TryGetValue(part.Name, out colors))
                    {
                        FillColors(part, _step, part.Mesh, colors);
                        _meshes[part.Name].SetColors(colors);
                    }
                }
                i++;
            }
        }

        public void SetField(string name)
        {
            field = name;
            float[] range = _data != null ? _data.Range(name) : null;
            _lo = range != null ? range[0] : 0f;
            _hi = range != null ? range[1] : 1f;
            ShowStep(_step < 0 ? 0 : _step);
        }

        /// <summary>Show a field computed at runtime -- a bridge answer -- on one part, and switch to it.
        ///
        /// ``values`` is ``steps * count`` floats, row-major, exactly what ``/v1/run`` returns; the legend picks up
        /// the range of the values themselves. The twin on disk is untouched: this field lives in memory for as long
        /// as the session does.
        /// </summary>
        public void ApplyField(string partName, string name, int steps, int count, float[] values, string unit)
        {
            if (_data == null)
            {
                Debug.LogError("PINNeAPPle: no twin loaded, cannot show the field '" + name + "'", this);
                return;
            }
            try
            {
                _data.AddField(partName, name, steps, count, unit, values);
            }
            catch (System.ArgumentException exc)
            {
                Debug.LogError("PINNeAPPle: " + exc.Message, this);
                return;
            }
            Debug.Log("PINNeAPPle: field '" + name + "' (" + unit + ", " + steps + " step(s) x " + count +
                      " vertices) applied to '" + partName + "'", this);
            SetField(name);
            ShowStep(_step < 0 ? 0 : _step);
        }

        // ------------------------------------------------------------------ legend and labels
        void OnGUI()
        {
            if (_data == null) return;
            if (_label == null)
            {
                _label = new GUIStyle(GUI.skin.label) { fontSize = 12 };
                _value = new GUIStyle(GUI.skin.label) { fontSize = 11, alignment = TextAnchor.UpperRight };
            }
            if (_legend == null || _legend.height > 1) _legend = BuildLegend();
            float w = Screen.width * 0.28f, h = Mathf.Max(28f, Screen.height * legendHeight * 0.5f);
            var box = new Rect(Screen.width - w - 16f, 16f, w, h);
            GUI.color = new Color(0f, 0f, 0f, 0.55f);
            GUI.DrawTexture(box, Texture2D.whiteTexture);
            GUI.color = Color.white;
            GUI.DrawTexture(new Rect(box.x + 8f, box.y + h - 14f, w - 16f, 8f), _legend);
            string unit = _data.Unit(field);
            GUI.Label(new Rect(box.x + 8f, box.y + 2f, w - 16f, 18f), "PINNeAPPle twin  |  " + _data.Title, _label);
            GUI.Label(new Rect(box.x + 8f, box.y + 18f, w - 16f, 18f),
                      (string.IsNullOrEmpty(field) ? "geometry" : field + (string.IsNullOrEmpty(unit) ? "" : " [" + unit + "]"))
                      + "  |  step " + (_step + 1) + "/" + StepCount, _label);
            GUI.Label(new Rect(box.x + 8f, box.y + h - 32f, (w - 16f) * 0.5f, 18f), _lo.ToString("G4"), _value);
            GUI.Label(new Rect(box.x + 8f + (w - 16f) * 0.5f, box.y + h - 32f, (w - 16f) * 0.5f, 18f),
                      _hi.ToString("G4"), _value);
            for (int i = 0; i < _data.Sensors.Count; i++)
            {
                TwinSensor s = _data.Sensors[i];
                Vector3 world = _sensorMarks.Count > i && _sensorMarks[i] != null
                    ? _sensorMarks[i].transform.position : Vector3.zero;
                Vector2 screen = Camera.main != null ? Camera.main.WorldToScreenPoint(world) : Vector2.zero;
                if (screen.x < 0 || screen.y < 0) continue;
                string text = s.Id;
                if (s.Series != null && s.Series.Length > 0)
                {
                    int k = Mathf.Clamp(_step, 0, s.Series.Length - 1);
                    text += "  " + s.Series[k].ToString("G4") + (string.IsNullOrEmpty(s.Unit) ? "" : " " + s.Unit);
                }
                bool alarm = false;
                if (s.Envelope != null && s.Envelope.Length >= 2 && s.Series != null && s.Series.Length > 0)
                {
                    int k = Mathf.Clamp(_step, 0, s.Series.Length - 1);
                    alarm = s.Series[k] < s.Envelope[0] || s.Series[k] > s.Envelope[1];
                }
                var rect = new Rect(screen.x + 8f, Screen.height - screen.y - 8f, 240f, 18f);
                GUI.color = alarm ? new Color(1f, 0.45f, 0.35f) : Color.white;
                GUI.Label(rect, text, _label);
                GUI.color = Color.white;
            }
        }

        Texture2D BuildLegend()
        {
            const int n = 256;
            var tex = new Texture2D(n, 1, TextureFormat.RGBA32, false) { wrapMode = TextureWrapMode.Clamp };
            var px = new Color[n];
            for (int i = 0; i < n; i++)
            {
                Rgb c = Colormap.Sample(colormap, i / (float)(n - 1));
                px[i] = new Color(c.R, c.G, c.B, 1f);
            }
            tex.SetPixels(px);
            tex.Apply();
            return tex;
        }
    }
}
