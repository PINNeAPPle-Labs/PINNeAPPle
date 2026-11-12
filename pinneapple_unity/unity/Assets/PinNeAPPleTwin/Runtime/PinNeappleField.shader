// Unlit vertex-colour shader for the twin field (Built-in Render Pipeline).
// The colour comes from Mesh.colors, written by PinneappleTwinView from the twin's field and the colormap.
// A light directional term keeps the form readable without needing a lit setup.
Shader "PINNeAPPle/Field"
{
    Properties
    {
        _Color ("Tint", Color) = (1,1,1,1)
        _Shade ("Shading", Range(0,1)) = 0.35
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" "Queue"="Geometry" }
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float3 normal : NORMAL;
                float4 color  : COLOR;
            };

            struct v2f
            {
                float4 pos   : SV_POSITION;
                float4 color : COLOR;
                float3 normalWS : TEXCOORD0;
            };

            float4 _Color;
            float _Shade;

            v2f vert (appdata v)
            {
                v2f o;
                o.pos = UnityObjectToClipPos(v.vertex);
                o.color = v.color * _Color;
                o.normalWS = UnityObjectToWorldNormal(v.normal);
                return o;
            }

            fixed4 frag (v2f i) : SV_Target
            {
                float3 n = normalize(i.normalWS);
                float light = 1.0 - _Shade * (1.0 - abs(n.z));
                return fixed4(i.color.rgb * light, i.color.a);
            }
            ENDCG
        }
    }
    Fallback "Unlit/Color"
}
