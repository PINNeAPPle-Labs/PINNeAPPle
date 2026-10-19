// Architecture preview: one column of layers per block group (a DeepONet's branch and trunk side by side), the
// parameter count of each layer and the total.
export function LayerGraph({ graph }: { graph: any }) {
  if (!graph || !graph.nodes) return null;
  const groups: string[] = graph.groups || [];
  const byGroup: Record<string, any[]> = {};
  for (const n of graph.nodes) (byGroup[n.group] ||= []).push(n);
  const colW = 210, rowH = 46, pad = 16;
  const rows = Math.max(...groups.map((g) => byGroup[g]?.length || 0), 1);
  const W = groups.length * colW + pad * 2, H = rows * rowH + 60;
  const fill = (t: string) =>
    t === "Linear" || t.startsWith("Conv") ? "#e0ecff" : t.includes("Spectral") ? "#ede4ff" : t.startsWith("POD") ? "#dcfce7" : t === "Dropout" || t === "LayerNorm" ? "#f1f5f9" : "#fff4dc";
  return (
    <div data-testid="layer-graph">
      <div className="text-xs text-slate-600 mb-1">
        <b>{graph.total_params?.toLocaleString()}</b> trainable parameters
        {graph.config?.kind ? <> · {graph.config.kind}</> : null}
        {graph.in_dim ? <> · in {graph.in_dim} → out {graph.out_dim}</> : null}
      </div>
      <svg width={W} height={H} className="max-w-full border border-slate-100 rounded bg-white">
        {groups.map((g, gi) => (
          <g key={g} transform={`translate(${pad + gi * colW},10)`}>
            <text x={colW / 2 - 10} y={12} textAnchor="middle" fontSize={11} fill="#64748b" fontWeight={600}>{g}</text>
            {(byGroup[g] || []).map((n, i) => (
              <g key={n.id} transform={`translate(0,${24 + i * rowH})`}>
                {i > 0 && <line x1={colW / 2 - 10} x2={colW / 2 - 10} y1={-rowH + 32} y2={0} stroke="#94a3b8" markerEnd="url(#arr)" />}
                <rect width={colW - 20} height={32} rx={6} fill={fill(n.type)} stroke="#cbd5e1" />
                <text x={8} y={13} fontSize={11} fill="#0f172a" fontWeight={600}>{n.type}</text>
                <text x={8} y={26} fontSize={9.5} fill="#475569">{(n.detail || "").slice(0, 34)}</text>
                <text x={colW - 28} y={13} fontSize={9.5} fill="#475569" textAnchor="end">{n.params ? n.params.toLocaleString() : ""}</text>
              </g>
            ))}
          </g>
        ))}
        <defs>
          <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8" />
          </marker>
        </defs>
      </svg>
    </div>
  );
}
