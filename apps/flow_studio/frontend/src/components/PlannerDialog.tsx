// Describe the problem, get proposals; each shows the planner that produced it and opens on the canvas.
import { useState } from "react";
import { api } from "../api";
import type { FlowDoc, PlanResult } from "../types";

export function PlannerDialog({ planners, onOpen, onClose }: { planners: string[]; onOpen: (f: FlowDoc) => void; onClose: () => void }) {
  const [text, setText] = useState("");
  const [backend, setBackend] = useState(planners.length > 1 ? "auto" : "heuristic");
  const [busy, setBusy] = useState(false);
  const [res, setRes] = useState<PlanResult | null>(null);
  const [err, setErr] = useState("");
  const go = async () => {
    setBusy(true);
    setErr("");
    try {
      setRes(await api.plan(text, backend));
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  };
  const det = res?.detected?.heuristic || res?.detected || {};
  return (
    <div className="fixed inset-0 bg-black/30 z-50 flex items-start justify-center pt-16" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-xl w-[760px] max-h-[80vh] overflow-y-auto p-5" onClick={(e) => e.stopPropagation()} data-testid="planner">
        <div className="flex justify-between items-center mb-2">
          <h2 className="text-lg font-semibold">Plan a flow from a description</h2>
          <button onClick={onClose} className="text-slate-400 text-xl">×</button>
        </div>
        <textarea
          className="w-full border border-slate-200 rounded p-2 text-sm" rows={4} value={text} data-testid="planner-text"
          placeholder="e.g. Transient heat in a 2D plate with its edges held at 0 °C; compare a PINN with finite differences and show it in 3-D"
          onChange={(e) => setText(e.target.value)}
        />
        <div className="flex items-center gap-2 mt-2">
          <select className="border rounded px-2 py-1 text-sm" value={backend} onChange={(e) => setBackend(e.target.value)}>
            <option value="heuristic">heuristic (offline)</option>
            <option value="llm" disabled={planners.length < 2}>LLM ({planners[1] || "not configured"})</option>
            <option value="auto">auto</option>
          </select>
          <button className="px-3 py-1.5 rounded bg-brand-500 text-white text-sm font-medium disabled:opacity-50" disabled={busy || !text.trim()} onClick={go} data-testid="planner-go">
            {busy ? "Planning…" : "Propose"}
          </button>
          {err && <span className="text-red-600 text-sm">{err}</span>}
        </div>
        {res && (
          <div className="mt-4 space-y-3">
            <div className="text-[12px] text-slate-600" data-testid="planner-label">
              Planner: <b>{res.planner}</b>{res.fallback_from ? ` (fallback from ${res.fallback_from})` : ""}
              {det.family !== undefined && (
                <> · understood: {det.family || "?"}{det.dimension ? `, ${det.dimension}-D` : ""}{det.steady ? ", steady" : ""}
                  {det.intents?.length ? `, wants ${det.intents.join(", ")}` : ""}
                  {det.coefficients && Object.keys(det.coefficients).length ? `, ${JSON.stringify(det.coefficients)}` : ""}</>
              )}
            </div>
            {res.warnings.map((w, i) => <div key={i} className="text-[12px] text-amber-800 bg-amber-50 rounded px-2 py-1">{w}</div>)}
            {res.proposals.map((p, i) => (
              <div key={i} className="border rounded-lg p-3 hover:border-brand-500" data-testid="proposal">
                <div className="flex justify-between items-start gap-3">
                  <div>
                    <div className="font-semibold text-sm">{p.title}</div>
                    <div className="text-[12px] text-slate-600 mt-1">{p.rationale}</div>
                    <div className="text-[11px] text-slate-400 mt-1">
                      {p.flow.nodes.length} nodes: {p.flow.nodes.map((n) => n.label || n.type).join(" → ")} · by {p.planner}
                    </div>
                    {p.repairs && p.repairs.length > 0 && <div className="text-[11px] text-amber-700 mt-1">repairs: {p.repairs.join("; ")}</div>}
                  </div>
                  <button className="px-2 py-1 rounded border border-brand-500 text-brand-700 text-sm whitespace-nowrap" onClick={() => onOpen(p.flow)} data-testid="proposal-open">
                    Open
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
