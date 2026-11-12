// Client for the PINNeAPPle <-> Unity bridge (protocol pinneapple-unity/1, `python -m pinneapple_unity serve`).
//
// From Play mode: send the part's geometry and the scene's time steps, ask for a job, get the field back with its
// checks -- so a field the library computed and verified can be shown on the same meshes the twin came with, without
// re-exporting anything. Requests run in a coroutine: the frame never waits for the physics.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;

namespace PinNeAPPle.Unity
{
    public sealed class BridgeCheck
    {
        public string Name = "";
        public bool? Passed;
        public float? Value;
        public float? Threshold;
        public string Detail = "";

        public override string ToString()
        {
            string mark = Passed == null ? "n/a" : (Passed.Value ? "PASS" : "FAIL");
            return mark + " " + Name + (Value.HasValue ? " = " + Value.Value.ToString("G4") : "");
        }
    }

    public sealed class BridgeField
    {
        public string Name = "";
        public string Unit = "";
        public int Steps;
        public int Count;
        public float[] Values;          // Steps * Count, row-major (step, vertex)

        public float Value(int step, int vertex)
        {
            int at = step * Count + vertex;
            return Values != null && at >= 0 && at < Values.Length ? Values[at] : 0f;
        }
    }

    public sealed class BridgeResponse
    {
        public bool Ok;
        public string Error = "";
        public string Job = "";
        public string Part = "";
        public string Decision = "";
        public float Seconds;
        public List<BridgeCheck> Checks = new List<BridgeCheck>();
        public List<BridgeField> Fields = new List<BridgeField>();

        public BridgeField Field(string name)
        {
            foreach (BridgeField f in Fields) if (f.Name == name) return f;
            return null;
        }
    }

    [AddComponentMenu("PINNeAPPle/Bridge Client")]
    public class PinneappleBridgeClient : MonoBehaviour
    {
        [Header("Bridge")]
        [Tooltip("Where `python -m pinneapple_unity serve` is listening.")]
        public string url = "http://127.0.0.1:8766";
        [Tooltip("Sent as X-Pinneapple-Token when the bridge was started with --token.")]
        public string token = "";

        [Header("Request")]
        public PinneappleTwinView view;
        [Tooltip("The registered job: slab_transient, cantilever_stress, or your own (pinneapple_unity.register_job).")]
        public string job = "slab_transient";
        [Tooltip("Which part of the twin. Empty: the first part with geometry.")]
        public string part = "";
        [Tooltip("Job parameters as a JSON object.")]
        public string paramsJson = "{\"alpha\":1e-5,\"t_left\":120,\"t_right\":20,\"t0\":20}";
        public bool runOnStart = false;

        public bool LastOk { get; private set; }
        public string LastDecision { get; private set; }
        public BridgeResponse Last { get; private set; }

        void Start()
        {
            if (view == null) view = GetComponent<PinneappleTwinView>();
#if UNITY_2023_1_OR_NEWER || UNITY_6000_0_OR_NEWER
            if (view == null) view = FindFirstObjectByType<PinneappleTwinView>();
#else
            if (view == null) view = FindObjectOfType<PinneappleTwinView>();
#endif
            if (runOnStart) StartCoroutine(RunAndApply(response => { }));
        }

        /// <summary>Ask for a job and show its field(s) on the twin, in one call.</summary>
        public IEnumerator RunAndApply(Action<BridgeResponse> done)
        {
            yield return Run(response =>
            {
                if (!response.Ok || view == null) return;
                foreach (BridgeField f in response.Fields)
                    view.ApplyField(response.Part, f.Name, f.Steps, f.Count, f.Values, f.Unit);
            });
            if (done != null) done(Last);
        }

        /// <summary>Ask the bridge for one job and hand back the field with its checks.</summary>
        public IEnumerator Run(Action<BridgeResponse> done)
        {
            if (view == null || view.Data == null)
            {
                var bad = new BridgeResponse { Ok = false, Error = "no twin loaded: add a PinneappleTwinView" };
                Last = bad;
                if (done != null) done(bad);
                yield break;
            }
            TwinData data = view.Data;
            string partName = string.IsNullOrEmpty(part) ? FirstPart(data) : part;
            TwinPart p = data.Part(partName);
            if (p == null || p.Mesh == null)
            {
                var bad = new BridgeResponse { Ok = false, Error = "no part '" + partName + "' with geometry" };
                Last = bad;
                if (done != null) done(bad);
                yield break;
            }
            string body = BuildBody(data, p, partName);
            var request = new UnityWebRequest(url.TrimEnd('/') + "/v1/run", "POST");
            byte[] payload = Encoding.UTF8.GetBytes(body);
            request.uploadHandler = new UploadHandlerRaw(payload);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            if (!string.IsNullOrEmpty(token)) request.SetRequestHeader("X-Pinneapple-Token", token);
            request.timeout = 600;
            yield return request.SendWebRequest();

#if UNITY_2020_2_OR_NEWER
            bool failed = request.result != UnityWebRequest.Result.Success;
#else
            bool failed = request.isNetworkError || request.isHttpError;
#endif
            BridgeResponse response;
            if (failed)
                response = new BridgeResponse { Ok = false, Error = request.error + ": " + request.downloadHandler.text };
            else
                response = Parse(request.downloadHandler.text);

            Last = response;
            LastOk = response.Ok;
            LastDecision = response.Decision;
            foreach (BridgeCheck c in response.Checks) Debug.Log("PINNeAPPle bridge: " + c);
            if (!response.Ok) Debug.LogError("PINNeAPPle bridge: " + response.Error);
            else Debug.Log("PINNeAPPle bridge: " + response.Job + " on " + response.Part + " -> " + response.Decision +
                           " in " + response.Seconds.ToString("G3") + " s");
            if (done != null) done(response);
        }

        static string FirstPart(TwinData data)
        {
            foreach (TwinPart p in data.Parts) if (p.Mesh != null) return p.Name;
            return "";
        }

        string BuildBody(TwinData data, TwinPart part, string partName)
        {
            var sb = new StringBuilder(part.Mesh.VertexCount * 24 + 512);
            sb.Append("{\"protocol\":\"pinneapple-unity/1\",\"job\":").Append(MiniJson.Quote(job))
              .Append(",\"part\":").Append(MiniJson.Quote(partName))
              .Append(",\"encoding\":\"base64\",\"title\":").Append(MiniJson.Quote(data.Title))
              .Append(",\"length_unit\":").Append(MiniJson.Quote(data.LengthUnit))
              .Append(",\"up_axis\":").Append(MiniJson.Quote(data.UpAxis))   // the twin's own frame: no rotation
              .Append(",\"times\":[");
            for (int i = 0; i < data.Times.Length; i++)
            {
                if (i > 0) sb.Append(',');
                sb.Append(MiniJson.Number(data.Times[i]));
            }
            sb.Append("],\"vertices\":[");
            float[] pos = part.Mesh.Positions;
            for (int i = 0; i < pos.Length; i += 3)
            {
                if (i > 0) sb.Append(',');
                sb.Append('[').Append(MiniJson.Number(pos[i])).Append(',').Append(MiniJson.Number(pos[i + 1]))
                  .Append(',').Append(MiniJson.Number(pos[i + 2])).Append(']');
            }
            sb.Append("],\"params\":").Append(string.IsNullOrEmpty(paramsJson) ? "{}" : paramsJson).Append('}');
            return sb.ToString();
        }

        // ------------------------------------------------------------------ response
        public static BridgeResponse Parse(string text)
        {
            var r = new BridgeResponse();
            Dictionary<string, object> root;
            try
            {
                root = MiniJson.Obj(MiniJson.Parse(text));
            }
            catch (Exception exc)
            {
                r.Error = "unreadable response: " + exc.Message;
                return r;
            }
            if (root == null) { r.Error = "empty response"; return r; }
            r.Ok = MiniJson.Bool(root, "ok", true);
            r.Error = MiniJson.Str(root, "error");
            r.Job = MiniJson.Str(root, "job");
            r.Part = MiniJson.Str(root, "part");
            r.Decision = MiniJson.Str(root, "decision");
            r.Seconds = (float)MiniJson.Num(root, "seconds");
            var checks = MiniJson.Arr(MiniJson.Get(root, "checks"));
            if (checks != null)
                foreach (object co in checks)
                {
                    var c = new BridgeCheck();
                    c.Name = MiniJson.Str(co, "name");
                    object passed = MiniJson.Get(co, "passed");
                    c.Passed = passed is bool ? (bool)passed : (bool?)null;
                    c.Detail = MiniJson.Str(co, "detail");
                    object value = MiniJson.Get(co, "value");
                    c.Value = value == null ? (float?)null : (float)Convert.ToDouble(value);
                    object threshold = MiniJson.Get(co, "threshold");
                    c.Threshold = threshold == null ? (float?)null : (float)Convert.ToDouble(threshold);
                    r.Checks.Add(c);
                }
            var fields = MiniJson.Obj(MiniJson.Get(root, "fields"));
            if (fields != null)
                foreach (KeyValuePair<string, object> kv in fields)
                {
                    var f = new BridgeField();
                    f.Name = kv.Key;
                    f.Unit = MiniJson.Str(kv.Value, "unit");
                    f.Steps = Math.Max(1, (int)MiniJson.Num(kv.Value, "steps", 1));
                    f.Count = (int)MiniJson.Num(kv.Value, "count");
                    string encoding = MiniJson.Str(kv.Value, "encoding", "base64");
                    if (encoding == "base64")
                    {
                        string b64 = MiniJson.StrRaw(MiniJson.Get(kv.Value, "data"));
                        byte[] bytes = Convert.FromBase64String(b64);
                        f.Values = new float[bytes.Length / 4];
                        for (int i = 0; i < f.Values.Length; i++)
                        {
                            if (BitConverter.IsLittleEndian) f.Values[i] = BitConverter.ToSingle(bytes, i * 4);
                            else
                            {
                                var tmp = new byte[4];
                                Array.Copy(bytes, i * 4, tmp, 0, 4);
                                Array.Reverse(tmp);
                                f.Values[i] = BitConverter.ToSingle(tmp, 0);
                            }
                        }
                    }
                    else
                    {
                        var arr = MiniJson.Arr(MiniJson.Get(kv.Value, "values"));
                        var vals = new List<float>();
                        if (arr != null) foreach (object o in arr)
                        {
                            if (o is List<object>)
                                foreach (object x in (List<object>)o) vals.Add((float)Convert.ToDouble(x));
                            else vals.Add((float)Convert.ToDouble(o));
                        }
                        f.Values = vals.ToArray();
                    }
                    r.Fields.Add(f);
                }
            return r;
        }
    }
}
