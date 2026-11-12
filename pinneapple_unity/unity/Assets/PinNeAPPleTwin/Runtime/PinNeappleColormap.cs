// Colormaps for the twin's per-vertex field: turbo (default) and viridis, as 11-point anchors interpolated
// linearly. Values are an approximation of the published maps -- close enough to read a field, and stated as an
// approximation rather than a colourimetric match. Pure C# so the legend and the tests do not need Unity.
using System;

namespace PinNeAPPle
{
    public struct Rgb
    {
        public float R, G, B;
        public Rgb(float r, float g, float b) { R = r; G = g; B = b; }
    }

    public static class Colormap
    {
        public static readonly string[] Names = { "turbo", "viridis" };

        // turbo (Google Research, 2019), sampled at 0, 0.1, ... 1
        static readonly float[] TURBO = {
            0.18995f, 0.07176f, 0.23217f,  0.27616f, 0.30476f, 0.83399f,  0.10516f, 0.60537f, 0.97848f,
            0.03994f, 0.85205f, 0.79212f,  0.20273f, 0.98066f, 0.44368f,  0.61050f, 0.99140f, 0.16302f,
            0.88141f, 0.86376f, 0.13211f,  0.99718f, 0.60362f, 0.05100f,  0.96844f, 0.28008f, 0.02255f,
            0.78842f, 0.06318f, 0.02910f,  0.47960f, 0.01060f, 0.01055f,
        };

        // viridis (Smith & van der Walt, 2015), sampled at 0, 0.1, ... 1
        static readonly float[] VIRIDIS = {
            0.26700f, 0.00487f, 0.32942f,  0.28262f, 0.14090f, 0.45754f,  0.25438f, 0.26505f, 0.52973f,
            0.20744f, 0.37198f, 0.55269f,  0.16404f, 0.47096f, 0.55810f,  0.12809f, 0.56695f, 0.55096f,
            0.13469f, 0.65897f, 0.51787f,  0.26694f, 0.74851f, 0.44054f,  0.47801f, 0.82127f, 0.31826f,
            0.74109f, 0.87317f, 0.15011f,  0.99325f, 0.90616f, 0.14394f,
        };

        /// <summary>Colour of <paramref name="t"/> in [0, 1] under <paramref name="name"/> (turbo when unknown).</summary>
        public static Rgb Sample(string name, float t)
        {
            float[] table = string.Equals(name, "viridis", StringComparison.OrdinalIgnoreCase) ? VIRIDIS : TURBO;
            if (t < 0f) t = 0f;
            if (t > 1f) t = 1f;
            float x = t * 10f;
            int i = (int)Math.Floor(x);
            if (i >= 10) i = 9;
            float f = x - i;
            int a = i * 3, b = a + 3;
            return new Rgb(table[a] + (table[b] - table[a]) * f,
                           table[a + 1] + (table[b + 1] - table[a + 1]) * f,
                           table[a + 2] + (table[b + 2] - table[a + 2]) * f);
        }

        /// <summary>The colour of a value in a range, clamped to the ends (a NaN reads as the low end).</summary>
        public static Rgb SampleValue(string name, float value, float lo, float hi)
        {
            float span = hi - lo;
            if (span <= 0f || float.IsNaN(value)) return Sample(name, 0f);
            return Sample(name, (value - lo) / span);
        }
    }
}
