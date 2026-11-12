// Editor menu for the PINNeAPPle Unity package.
//
//   PINNeAPPle > Check Twin Data          re-read Assets/TwinData and report every inconsistency
//   PINNeAPPle > Open Twin Data Folder    open the twin folder in the OS file manager
//   PINNeAPPle > Export Selection as Scene Link
//                                         write Assets/TwinLink/scene.glb + pinneapple_link.json for the selected
//                                         objects -- the Unity -> PINNeAPPle -> OpenUSD lane, with the jobs template
//   PINNeAPPle > Copy Bridge Command      put `python -m pinneapple_unity serve` on the clipboard
using System.Collections.Generic;
using System.IO;
using System.Text;
using PinNeAPPle;
using UnityEditor;
using UnityEngine;

namespace PinNeAPPle.Unity.EditorTools
{
    public static class PinneappleMenu
    {
        const string DATA = "Assets/TwinData";
        const string LINK = "Assets/TwinLink";

        [MenuItem("PINNeAPPle/Check Twin Data")]
        public static void CheckTwinData()
        {
            string folder = ProjectPath(DATA);
            if (!Directory.Exists(folder))
            {
                Debug.LogWarning("PINNeAPPle: " + DATA + " does not exist. Export a twin with " +
                                 "pinneapple_unity.export_for_unity and copy its Assets folder here.");
                return;
            }
            TwinData data;
            try
            {
                data = TwinData.Load(folder);
            }
            catch (System.Exception exc)
            {
                Debug.LogError("PINNeAPPle: " + exc.Message);
                return;
            }
            List<string> problems = data.Verify();
            foreach (string p in problems) Debug.LogWarning("PINNeAPPle: " + p);
            Debug.Log("PINNeAPPle: " + data + (problems.Count == 0 ? " -- every file checks out" : " -- " + problems.Count + " problem(s)"));
        }

        [MenuItem("PINNeAPPle/Open Twin Data Folder")]
        public static void OpenTwinData()
        {
            string folder = ProjectPath(DATA);
            if (Directory.Exists(folder)) EditorUtility.RevealInFinder(folder);
            else Debug.LogWarning("PINNeAPPle: " + DATA + " does not exist");
        }

        [MenuItem("PINNeAPPle/Copy Bridge Command")]
        public static void CopyBridgeCommand()
        {
            GUIUtility.systemCopyBuffer = "python -m pinneapple_unity serve --port 8766";
            Debug.Log("PINNeAPPle: copied the bridge command to the clipboard");
        }

        [MenuItem("PINNeAPPle/Export Selection as Scene Link")]
        public static void ExportSelection()
        {
            var meshes = new List<GlbMesh>();
            var parts = new List<string>();
            foreach (GameObject go in Selection.gameObjects)
                Collect(go.transform, meshes, parts);
            if (meshes.Count == 0)
            {
                Debug.LogWarning("PINNeAPPle: select one or more objects with meshes");
                return;
            }
            Directory.CreateDirectory(ProjectPath(LINK));
            string glb = Path.Combine(ProjectPath(LINK), "scene.glb");
            Glb.Write(glb, meshes, "PINNeAPPle Unity exporter");
            string link = Path.Combine(ProjectPath(LINK), "pinneapple_link.json");
            File.WriteAllText(link, LinkJson(meshes, parts));
            AssetDatabase.Refresh();
            Debug.Log("PINNeAPPle: wrote " + glb + " (" + meshes.Count + " part(s)) and " + link +
                      ". Run `python -m pinneapple_unity ingest " + LINK + " --out out/twin` to compute the jobs " +
                      "and get an OpenUSD layer back.");
        }

        static void Collect(Transform t, List<GlbMesh> meshes, List<string> names)
        {
            var mf = t.GetComponent<MeshFilter>();
            if (mf != null && mf.sharedMesh != null)
            {
                Mesh src = mf.sharedMesh;
                var g = new GlbMesh { Name = t.name };
                Vector3[] verts = src.vertices;
                Vector3[] normals = src.normals;
                int[] tris = src.triangles;
                g.Positions = new float[verts.Length * 3];
                for (int i = 0; i < verts.Length; i++)
                {
                    Vector3 w = t.TransformPoint(verts[i]);
                    g.Positions[i * 3] = w.x; g.Positions[i * 3 + 1] = w.y; g.Positions[i * 3 + 2] = w.z;
                }
                if (normals.Length == verts.Length)
                {
                    g.Normals = new float[normals.Length * 3];
                    for (int i = 0; i < normals.Length; i++)
                    {
                        Vector3 n = t.TransformDirection(normals[i]).normalized;
                        g.Normals[i * 3] = n.x; g.Normals[i * 3 + 1] = n.y; g.Normals[i * 3 + 2] = n.z;
                    }
                }
                g.Indices = tris;
                var mr = t.GetComponent<MeshRenderer>();
                if (mr != null && mr.sharedMaterial != null)
                {
                    Material m = mr.sharedMaterial;
                    Color c = m.HasProperty("_BaseColor") ? m.GetColor("_BaseColor") : m.color;
                    g.Color = new[] { c.r, c.g, c.b, c.a };
                }
                meshes.Add(g);
                names.Add(t.name);
            }
            foreach (Transform child in t) Collect(child, meshes, names);
        }

        static string LinkJson(List<GlbMesh> meshes, List<string> names)
        {
            var sb = new StringBuilder();
            sb.Append("{\n \"protocol\": \"pinneapple-unity/1\",\n \"title\": \"Unity selection\",\n")
              .Append(" \"geometry\": \"scene.glb\",\n \"length_unit\": \"m\",\n \"time_unit\": \"s\",\n")
              .Append(" \"up_axis\": \"y\",\n \"times\": [0, 60, 300, 900, 3600],\n \"parts\": [\n");
            for (int i = 0; i < meshes.Count; i++)
            {
                float[] c = meshes[i].Color;
                sb.Append("  {\"name\": ").Append(MiniJson.Quote(names[i])).Append(", \"node\": ")
                  .Append(MiniJson.Quote(names[i])).Append(", \"group\": \"\", \"color\": [")
                  .Append(MiniJson.Number(c[0])).Append(", ").Append(MiniJson.Number(c[1])).Append(", ")
                  .Append(MiniJson.Number(c[2])).Append("], \"axis\": \"auto\"}")
                  .Append(i + 1 < meshes.Count ? ",\n" : "\n");
            }
            sb.Append(" ],\n \"sensors\": [],\n \"jobs\": [\n")
              .Append("  {\"job\": \"slab_transient\", \"part\": ").Append(MiniJson.Quote(names[0]))
              .Append(", \"params\": {\"alpha\": 1e-5, \"t_left\": 120, \"t_right\": 20, \"t0\": 20, \"sensor\": \"TC-01\"}}\n")
              .Append(" ],\n \"physics\": {\"gravity\": [0, -9.81, 0], \"parts\": {\n")
              .Append("  ").Append(MiniJson.Quote(names[0]))
              .Append(": {\"rigid_body\": false, \"collider\": true, \"material\": {\"static_friction\": 0.4, ")
              .Append("\"dynamic_friction\": 0.3, \"restitution\": 0.0}}}\n }}\n}\n");
            return sb.ToString();
        }

        static string ProjectPath(string relative)
        {
            return Path.Combine(Directory.GetParent(Application.dataPath).FullName, relative);
        }
    }
}
