// Decision records of a node (or of the whole flow): who decided, why, the references (clickable), the evidence,
// the confidence, and what would raise it. User edits can get a note and a reference.
import { useState } from "react";
import type { Decision, Reference } from "../types";

const LABEL: Record<string, string> = { high: "bg-emerald-100 text-emerald-800", medium: "bg-amber-100 text-amber-800", low: "bg-red-100 text-red-800" };

function Ref({ r }: { r: Reference }) {
  const a = (r.authors || []).slice(0, 2).join(", ") + ((r.authors || []).length > 2 ? " et al." : "");
  const text = `${a}${r.year ? ` (${r.year})` : ""} ${r.title || r.key || ""}`.trim();
  return (
    <div className="text-[11px]">
      {r.link ? <a href={r.link} target="_blank" rel="noreferrer noopener" className="text-sky-700 underline">{text}</a> : <span>{text}</span>}
      {r.status && r.status !== "registry" && <span className="ml-1 text-amber-700">[{r.status}]</span>}
      {r.note && <span className="ml-1 text-slate-400">{r.note}</span>}
    </div>
  );
}

export function DecisionList({ decisions, onAnnotate, onRaise }: {
  decisions: Decision[]; onAnnotate?: (d: Decision, notes: string, ref: string) => void; onRaise?: (d: Decision, type: string) => void;
}) {
  const [edit, setEdit] = useState<string | null>(null);
  const [notes, setNotes] = useState("");
  const [ref, setRef] = useState("");
  if (!decisions.length) return <div className="text-[12px] text-slate-400">No decision recorded for this node yet.</div>;
  return (
    <div className="space-y-2" data-testid="decisions">
      {decisions.map((d) => (
        <div key={d.id} className={`rounded border p-2 ${d.no_reference ? "border-red-200 bg-red-50/40" : "border-slate-200"}`} data-testid="decision">
          <div className="flex justify-between gap-2">
            <div className="text-[12px] font-medium">{d.what}</div>
            <span className={`text-[10px] px-1.5 py-0.5 rounded whitespace-nowrap h-fit ${LABEL[d.label] || ""}`}>{d.confidence} {d.label}</span>
          </div>
          <div className="text-[10.5px] text-slate-500">
            {d.topic} · by {d.decided_by.kind === "rule" ? `rule ${d.decided_by.id}` : d.decided_by.kind === "llm" ? `LLM ${d.decided_by.id}` : "user"}
            {d.no_reference && <b className="ml-1 text-red-700" data-testid="no-reference">no reference</b>}
          </div>
          {d.rationale && <div className="text-[11px] text-slate-600 mt-0.5">{d.rationale}</div>}
          {d.references.map((r, i) => <Ref key={i} r={r} />)}
          {d.evidence.map((e, i) => <div key={i} className="text-[10.5px] text-slate-500">evidence ({e.kind}): {e.ref} — {e.detail}</div>)}
          {d.uncited_reason && <div className="text-[10.5px] italic text-slate-500">no literature reference: {d.uncited_reason}</div>}
          {d.suggestion && <div className="text-[10.5px] text-amber-800 mt-0.5">{d.suggestion}</div>}
          {onRaise && (d.raise_with || []).length > 0 && (
            <div className="flex flex-wrap gap-1 mt-0.5" data-testid="raise-with">
              {(d.raise_with || []).map((t) => (
                <button key={t} className="text-[10.5px] px-1.5 rounded border border-sky-200 text-sky-700" onClick={() => onRaise(d, t)}>+ {t}</button>
              ))}
            </div>
          )}
          {d.notes && <div className="text-[10.5px] text-slate-600">note: {d.notes}</div>}
          {d.decided_by.kind === "user" && onAnnotate && (edit === d.id ? (
            <div className="mt-1 space-y-1">
              <input className="w-full text-[11px] border rounded px-1" placeholder="note" value={notes} onChange={(e) => setNotes(e.target.value)} />
              <input className="w-full text-[11px] border rounded px-1" placeholder="reference: registry key or DOI" value={ref} onChange={(e) => setRef(e.target.value)} />
              <button className="text-[11px] text-sky-700" onClick={() => { onAnnotate(d, notes, ref); setEdit(null); }}>save</button>
            </div>
          ) : (
            <button className="text-[11px] text-sky-700 mt-1" onClick={() => { setEdit(d.id); setNotes(d.notes || ""); setRef(""); }}>add note / reference</button>
          ))}
        </div>
      ))}
    </div>
  );
}

export function DecisionBadge({ decisions }: { decisions?: Decision[] }) {
  if (!decisions || !decisions.length) return null;
  const noRef = decisions.filter((d) => d.no_reference).length;
  const low = Math.min(...decisions.map((d) => d.confidence));
  const cls = noRef ? "bg-red-100 text-red-700" : low < 0.5 ? "bg-amber-100 text-amber-800" : "bg-emerald-100 text-emerald-800";
  return (
    <span className={`text-[9.5px] px-1 rounded ${cls}`} title={`${decisions.length} decisions, ${noRef} without a reference, lowest confidence ${low}`} data-testid="decision-badge">
      {noRef ? `${noRef} no ref` : `${decisions.length} ref'd`}
    </span>
  );
}
