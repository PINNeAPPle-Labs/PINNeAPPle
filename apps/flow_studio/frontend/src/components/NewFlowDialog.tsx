// "New flow": a blank canvas, a description for the planner, or the guided wizard (problem -> physics -> geometry /
// data -> methods -> validation -> outputs, /api/wizard). Every proposal opens on the canvas, fully editable.
import { useEffect, useState } from "react";
import { api } from "../api";
import type { FlowDoc, PlanResult } from "../types";

type Step = { id: string; title: string; kind: string; help?: string; choices?: Record<string, string> };
type Cap = { id: string; title: string; domain: string; description: string; inputs: string[]; references: string[]; uncited: string };

export function NewFlowDialog({ onBlank, onDescribe, onOpen, onClose }: {
  onBlank: () => void; onDescribe: () => void; onOpen: (f: FlowDoc) => void; onClose: () => void;
}) {
  const [mode, setMode] = useState<"choose" | "wizard">("choose");
  const [steps, setSteps] = useState<Step[]>([]);
  const [caps, setCaps] = useState<Cap[]>([]);
  const [k, setK] = useState(0);
  const [answers, setAnswers] = useState<Record<string, any>>({ methods: [], validation: [], outputs: [] });
  const [res, setRes] = useState<PlanResult & { wizard?: { description: string } } | null>(null);
  const [err, setErr] = useState("");
  useEffect(() => {
    if (mode === "wizard" && !steps.length) api.capabilities().then((r) => { setSteps(r.steps); setCaps(r.capabilities); }).catch((e) => setErr(String(e)));
  }, [mode, steps.length]);
  const step = steps[k];
  const set = (id: string, v: any) => setAnswers((a) => ({ ...a, [id]: v }));
  const toggle = (id: string, v: string) => setAnswers((a) => ({ ...a, [id]: (a[id] || []).includes(v) ? a[id].filter((x: string) => x !== v) : [...(a[id] || []), v] }));
  const finish = async () => {
    setErr("");
    try { setRes(await api.wizard(answers)); } catch (e) { setErr(String(e)); }
  };
  const card = "flex-1 border rounded-lg p-4 text-left hover:border-brand-500 hover:bg-brand-50";
  return (
    <div className="fixed inset-0 bg-black/30 z-50 flex items-start justify-center pt-14" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-xl w-[820px] max-h-[84vh] overflow-y-auto p-5" onClick={(e) => e.stopPropagation()} data-testid="new-flow">
        <div className="flex justify-between items-center mb-3">
          <h2 className="text-lg font-semibold">New flow</h2>
          <button onClick={onClose} className="text-slate-400 text-xl">×</button>
        </div>
        {mode === "choose" && (
          <div className="flex gap-3">
            <button className={card} onClick={onBlank} data-testid="new-blank"><b>Blank canvas</b><div className="text-[12px] text-slate-600 mt-1">Start empty and drag nodes from the palette.</div></button>
            <button className={card} onClick={onDescribe} data-testid="new-describe"><b>Describe the problem</b><div className="text-[12px] text-slate-600 mt-1">The planner proposes flows from your text, each choice with its decision record and references.</div></button>
            <button className={card} onClick={() => setMode("wizard")} data-testid="new-wizard"><b>Guided wizard</b><div className="text-[12px] text-slate-600 mt-1">Problem → physics → geometry / data → methods → validation → outputs.</div></button>
          </div>
        )}
        {mode === "wizard" && step && !res && (
          <div data-testid="wizard">
            <div className="flex gap-1 mb-3 text-[11px]">
              {steps.map((s, i) => (
                <button key={s.id} onClick={() => setK(i)} className={`px-2 py-1 rounded ${i === k ? "bg-brand-500 text-white" : i < k ? "bg-brand-50 text-brand-700" : "bg-slate-100 text-slate-500"}`}>{i + 1}. {s.title}</button>
              ))}
            </div>
            <div className="font-semibold text-sm mb-1">{step.title}</div>
            {step.help && <div className="text-[12px] text-slate-500 mb-2">{step.help}</div>}
            {step.kind === "text+capability" && (
              <>
                <textarea className="w-full border rounded p-2 text-sm" rows={3} value={answers.problem || ""} data-testid="wizard-problem"
                  placeholder="e.g. Take-off distance of an A320 against Raymer's closed form" onChange={(e) => set("problem", e.target.value)} />
                <div className="text-[12px] text-slate-500 mt-2 mb-1">or start from a domain study:</div>
                <select className="border rounded px-2 py-1 text-sm w-full" value={answers.capability || ""} data-testid="wizard-capability" onChange={(e) => set("capability", e.target.value || undefined)}>
                  <option value="">(none: let the planner choose)</option>
                  {Array.from(new Set(caps.map((c) => c.domain))).map((d) => (
                    <optgroup key={d} label={d}>{caps.filter((c) => c.domain === d).map((c) => <option key={c.id} value={c.id}>{c.title}</option>)}</optgroup>
                  ))}
                </select>
              </>
            )}
            {step.kind === "choice" && (
              <div className="grid grid-cols-2 gap-1">
                {Object.entries(step.choices || {}).map(([v, label]) => (
                  <label key={v} className="text-sm flex gap-2 items-center"><input type="radio" name={step.id} checked={(answers[step.id] || "auto") === v} onChange={() => set(step.id, v)} />{label}</label>
                ))}
              </div>
            )}
            {step.kind === "multi" && (
              <div className="grid grid-cols-2 gap-1">
                {Object.entries(step.choices || {}).map(([v, label]) => (
                  <label key={v} className="text-sm flex gap-2 items-center"><input type="checkbox" checked={(answers[step.id] || []).includes(v)} onChange={() => toggle(step.id, v)} data-testid={`wizard-${step.id}-${v}`} />{label}</label>
                ))}
              </div>
            )}
            <div className="flex justify-between mt-4">
              <button className="btn" disabled={k === 0} onClick={() => setK(k - 1)}>Back</button>
              {k < steps.length - 1
                ? <button className="btn-primary" onClick={() => setK(k + 1)} data-testid="wizard-next">Next</button>
                : <button className="btn-primary" onClick={finish} data-testid="wizard-finish">Propose flows</button>}
            </div>
          </div>
        )}
        {err && <div className="text-red-600 text-sm mt-2">{err}</div>}
        {res && (
          <div className="space-y-3" data-testid="wizard-result">
            <div className="text-[12px] text-slate-600">Planner read: <i>{res.wizard?.description}</i></div>
            {res.warnings.map((w, i) => <div key={i} className="text-[12px] text-amber-800 bg-amber-50 rounded px-2 py-1">{w}</div>)}
            {res.proposals.map((p, i) => {
              const dec: any[] = (p.flow.meta as any)?.decisions || [];
              return (
                <div key={i} className="border rounded-lg p-3" data-testid="proposal">
                  <div className="flex justify-between gap-3">
                    <div>
                      <div className="font-semibold text-sm">{p.title}</div>
                      <div className="text-[12px] text-slate-600 mt-1">{p.rationale}</div>
                      <div className="text-[11px] text-slate-400 mt-1">{p.flow.nodes.length} nodes: {p.flow.nodes.map((n) => n.label || n.type).join(" → ")}</div>
                      <div className="text-[11px] text-slate-500 mt-1">
                        {dec.length} decisions; references: {Array.from(new Set(dec.flatMap((d) => (d.references || []).map((r: any) => r.key || r.title)))).join(", ") || "no reference (lower confidence)"}
                      </div>
                    </div>
                    <button className="px-2 py-1 rounded border border-brand-500 text-brand-700 text-sm h-fit" onClick={() => onOpen(p.flow)} data-testid="proposal-open">Open</button>
                  </div>
                </div>
              );
            })}
            <button className="btn" onClick={() => setRes(null)}>Back to the wizard</button>
          </div>
        )}
      </div>
    </div>
  );
}
