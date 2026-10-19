// The Lab seal of the items this run validated, as pinneapple_lab states it (letter, reason; for the lowest letters
// the failing check and how to raise it), and what the next letter needs with buttons that insert the nodes the Lab
// suggests. Nothing is graded here: everything comes from /api/lab (pinneapple_flow.lab_status).
import { useState } from "react";
import { api } from "../api";
import type { NextSteps, SealStatus } from "../types";

const COLOR: Record<string, string> = {
  A: "bg-emerald-600", B: "bg-amber-500", C: "bg-sky-600", D: "bg-slate-500", E: "bg-orange-600", F: "bg-red-700",
};

export function SealPanel({ seals, onInsert }: {
  seals: SealStatus[]; onInsert?: (nodes: { type: string; params?: Record<string, any> }[]) => void;
}) {
  const [steps, setSteps] = useState<Record<string, NextSteps | string>>({});
  if (!seals.length) return null;
  const load = (item: string) =>
    api.nextSteps(item).then((s) => setSteps((m) => ({ ...m, [item]: s }))).catch((e) => setSteps((m) => ({ ...m, [item]: String(e) })));
  return (
    <div className="space-y-2" data-testid="seal-panel">
      {seals.map((s) => {
        const st = steps[s.item];
        return (
          <div key={s.item} className="rounded border border-slate-200 p-2">
            <div className="flex items-center gap-2">
              <span className={`text-white font-bold rounded w-7 h-7 flex items-center justify-center ${s.letter ? COLOR[s.letter] || "bg-slate-400" : "bg-slate-300"}`}
                data-testid="seal-letter">{s.letter || "–"}</span>
              <div className="min-w-0">
                <div className="text-[12px] font-semibold">{s.letter ? `Lab seal ${s.letter}${s.name ? ` · ${s.name}` : ""}` : "not curated"}</div>
                <div className="text-[10.5px] text-slate-500 truncate" title={s.item}>{s.item} · scale {s.scale}{s.sealed ? "" : " · not sealed"}</div>
              </div>
            </div>
            {s.reason && <div className="text-[11px] text-slate-600 mt-1">{s.reason}</div>}
            {s.failing_check && <div className="text-[11px] text-red-700 mt-1">failing check: {s.failing_check}</div>}
            {s.hint && <div className="text-[11px] text-amber-800">how to raise it: {s.hint}</div>}
            {!st && <button className="text-[11px] text-sky-700 mt-1" onClick={() => load(s.item)} data-testid="seal-next-steps">what the next letter needs</button>}
            {typeof st === "string" && <div className="text-[11px] text-red-700">{st}</div>}
            {st && typeof st !== "string" && (
              <div className="mt-1 space-y-1">
                {st.message && <div className="text-[10.5px] text-slate-500 italic" data-testid="next-steps-message">{st.message}</div>}
                {st.gaps.length === 0 && !st.message && <div className="text-[11px] text-slate-500">nothing listed</div>}
                {st.gaps.map((g, i) => (
                  <div key={i} className="text-[11px] border-t border-slate-100 pt-1" data-testid="seal-gap">
                    <div>{g.to ? <b className="mr-1">{g.to}</b> : null}{g.gap}</div>
                    {g.why && g.why !== g.gap && <div className="text-slate-500">{g.why}</div>}
                    {(g.actions && g.actions.length ? g.actions : (g.action || g.command) ? [{ id: "", kind: "", target: "", command: g.command, text: g.action, nodes: g.nodes }] : []).map((a, j) => (
                      <div key={j} className="flex items-start gap-1 mt-0.5">
                        <span className="text-slate-500 flex-1" title={a.command}>{a.text || a.command}{a.kind === "cli" ? " (command line)" : ""}</span>
                        {a.nodes.length > 0 && onInsert && (
                          <button className="text-sky-700 whitespace-nowrap" onClick={() => onInsert(a.nodes)} data-testid="seal-insert">+ {a.nodes.map((n) => n.type).join(", ")}</button>
                        )}
                      </div>
                    ))}
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
