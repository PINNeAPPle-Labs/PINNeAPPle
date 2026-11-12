// Minimal JSON reader and writer for the PINNeAPPle twin files (protocol pinneapple-unity/1).
//
// Why not JsonUtility: scene.json mixes arrays of numbers, arrays of objects and a union of per-field metadata, and
// a Unity package should not pull a third-party parser in for four files. This one is ~200 lines, allocation-light
// enough for a manifest, and has no dependency beyond the BCL -- so it also compiles and runs outside Unity, which is
// how the data layer is tested (see unity/tests).
//
// C# 9 (Unity 2021.3 LTS) and no Unity types: keep it that way.
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Text;

namespace PinNeAPPle
{
    public static class MiniJson
    {
        public static object Parse(string text)
        {
            int i = 0;
            object v = ParseValue(text, ref i);
            Skip(text, ref i);
            if (i < text.Length) throw new FormatException("trailing characters at " + i);
            return v;
        }

        // ---- typed accessors (all null-tolerant: a manifest may omit anything optional) ----
        public static Dictionary<string, object> Obj(object o) { return o as Dictionary<string, object>; }

        public static List<object> Arr(object o) { return o as List<object>; }

        public static object Get(object o, string key)
        {
            var d = o as Dictionary<string, object>;
            object v;
            return d != null && d.TryGetValue(key, out v) ? v : null;
        }

        public static string Str(object o, string key, string fallback = "")
        {
            object v = Get(o, key);
            return v == null ? fallback : Convert.ToString(v, CultureInfo.InvariantCulture);
        }

        public static string StrRaw(object v, string fallback = "")
        {
            return v == null ? fallback : Convert.ToString(v, CultureInfo.InvariantCulture);
        }

        public static double Num(object o, string key, double fallback = 0.0)
        {
            object v = Get(o, key);
            return v == null ? fallback : Convert.ToDouble(v, CultureInfo.InvariantCulture);
        }

        public static bool Bool(object o, string key, bool fallback = false)
        {
            object v = Get(o, key);
            return v is bool ? (bool)v : fallback;
        }

        public static float[] Floats(object o, string key)
        {
            var a = Arr(Get(o, key));
            if (a == null) return null;
            var outv = new float[a.Count];
            for (int k = 0; k < a.Count; k++) outv[k] = Convert.ToSingle(a[k], CultureInfo.InvariantCulture);
            return outv;
        }

        public static double[] Nums(object o, string key)
        {
            var a = Arr(Get(o, key));
            if (a == null) return null;
            var outv = new double[a.Count];
            for (int k = 0; k < a.Count; k++) outv[k] = Convert.ToDouble(a[k], CultureInfo.InvariantCulture);
            return outv;
        }

        // ---- writing ----
        public static string Quote(string s)
        {
            if (s == null) return "\"\"";
            var sb = new StringBuilder(s.Length + 2);
            sb.Append('"');
            foreach (char c in s)
            {
                switch (c)
                {
                    case '"': sb.Append("\\\""); break;
                    case '\\': sb.Append("\\\\"); break;
                    case '\n': sb.Append("\\n"); break;
                    case '\r': sb.Append("\\r"); break;
                    case '\t': sb.Append("\\t"); break;
                    default:
                        if (c < ' ') sb.Append("\\u").Append(((int)c).ToString("x4", CultureInfo.InvariantCulture));
                        else sb.Append(c);
                        break;
                }
            }
            sb.Append('"');
            return sb.ToString();
        }

        public static string Number(double v)
        {
            return v.ToString("R", CultureInfo.InvariantCulture);
        }

        // ---- the parser ----
        static void Skip(string s, ref int i)
        {
            while (i < s.Length)
            {
                char c = s[i];
                if (c == ' ' || c == '\t' || c == '\n' || c == '\r') { i++; continue; }
                break;
            }
        }

        static object ParseValue(string s, ref int i)
        {
            Skip(s, ref i);
            if (i >= s.Length) throw new FormatException("unexpected end of JSON");
            char c = s[i];
            switch (c)
            {
                case '{': return ParseObject(s, ref i);
                case '[': return ParseArray(s, ref i);
                case '"': return ParseString(s, ref i);
                case 't': Expect(s, ref i, "true"); return true;
                case 'f': Expect(s, ref i, "false"); return false;
                case 'n': Expect(s, ref i, "null"); return null;
                default: return ParseNumber(s, ref i);
            }
        }

        static void Expect(string s, ref int i, string word)
        {
            if (i + word.Length > s.Length || string.CompareOrdinal(s, i, word, 0, word.Length) != 0)
                throw new FormatException("expected '" + word + "' at " + i);
            i += word.Length;
        }

        static Dictionary<string, object> ParseObject(string s, ref int i)
        {
            var d = new Dictionary<string, object>();
            i++;                                        // '{'
            Skip(s, ref i);
            if (i < s.Length && s[i] == '}') { i++; return d; }
            while (true)
            {
                Skip(s, ref i);
                string key = ParseString(s, ref i);
                Skip(s, ref i);
                if (i >= s.Length || s[i] != ':') throw new FormatException("expected ':' at " + i);
                i++;
                d[key] = ParseValue(s, ref i);
                Skip(s, ref i);
                if (i >= s.Length) throw new FormatException("unterminated object");
                if (s[i] == ',') { i++; continue; }
                if (s[i] == '}') { i++; return d; }
                throw new FormatException("expected ',' or '}' at " + i);
            }
        }

        static List<object> ParseArray(string s, ref int i)
        {
            var a = new List<object>();
            i++;                                        // '['
            Skip(s, ref i);
            if (i < s.Length && s[i] == ']') { i++; return a; }
            while (true)
            {
                a.Add(ParseValue(s, ref i));
                Skip(s, ref i);
                if (i >= s.Length) throw new FormatException("unterminated array");
                if (s[i] == ',') { i++; continue; }
                if (s[i] == ']') { i++; return a; }
                throw new FormatException("expected ',' or ']' at " + i);
            }
        }

        static string ParseString(string s, ref int i)
        {
            if (i >= s.Length || s[i] != '"') throw new FormatException("expected a string at " + i);
            i++;
            var sb = new StringBuilder();
            while (true)
            {
                if (i >= s.Length) throw new FormatException("unterminated string");
                char c = s[i++];
                if (c == '"') return sb.ToString();
                if (c != '\\') { sb.Append(c); continue; }
                if (i >= s.Length) throw new FormatException("unterminated escape");
                char e = s[i++];
                switch (e)
                {
                    case '"': sb.Append('"'); break;
                    case '\\': sb.Append('\\'); break;
                    case '/': sb.Append('/'); break;
                    case 'b': sb.Append('\b'); break;
                    case 'f': sb.Append('\f'); break;
                    case 'n': sb.Append('\n'); break;
                    case 'r': sb.Append('\r'); break;
                    case 't': sb.Append('\t'); break;
                    case 'u':
                        if (i + 4 > s.Length) throw new FormatException("short \\u escape");
                        sb.Append((char)int.Parse(s.Substring(i, 4), NumberStyles.HexNumber, CultureInfo.InvariantCulture));
                        i += 4;
                        break;
                    default: throw new FormatException("bad escape '\\" + e + "'");
                }
            }
        }

        static double ParseNumber(string s, ref int i)
        {
            int start = i;
            if (i < s.Length && (s[i] == '-' || s[i] == '+')) i++;
            while (i < s.Length)
            {
                char c = s[i];
                if ((c >= '0' && c <= '9') || c == '.' || c == 'e' || c == 'E' || c == '+' || c == '-') { i++; continue; }
                break;
            }
            if (i == start) throw new FormatException("expected a number at " + start);
            return double.Parse(s.Substring(start, i - start), NumberStyles.Float, CultureInfo.InvariantCulture);
        }
    }
}
