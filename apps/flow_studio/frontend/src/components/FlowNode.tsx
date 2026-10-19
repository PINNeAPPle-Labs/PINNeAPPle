import { Handle, Position, type NodeProps } from "@xyflow/react";
import { memo } from "react";
import { PORT_COLORS, STATUS_STYLE, type Decision, type NodeType, type Status } from "../types";
import { DecisionBadge } from "./DecisionsPanel";

export interface FlowNodeData extends Record<string, unknown> {
  spec: NodeType;
  label: string;
  params: Record<string, any>;
  status: Status;
  summary?: string;
  error?: string;
  decisions?: Decision[];
}

const CATEGORY_ACCENT: Record<string, string> = {
  Problem: "border-sky-400", Solvers: "border-emerald-500", Learning: "border-orange-500", Inverse: "border-pink-500",
  Data: "border-yellow-500", Validation: "border-violet-500", Visualization: "border-blue-500", Export: "border-slate-400",
  Analysis: "border-fuchsia-500", "Digital twin": "border-teal-500", "Time series": "border-amber-600", Systems: "border-cyan-600",
};

function FlowNodeView({ data, selected }: NodeProps) {
  const d = data as FlowNodeData;
  const s = d.spec;
  const rows = Math.max(s.inputs.length, s.outputs.length, 1);
  return (
    <div
      data-testid={`node-${d.label}`}
      className={`rounded-lg bg-white shadow-sm border-l-4 ${CATEGORY_ACCENT[s.category] || "border-slate-400"} border border-slate-200 min-w-[200px] max-w-[240px] ${selected ? "ring-2 ring-brand-500" : ""} ${!s.available ? "opacity-60" : ""}`}
    >
      <div className="px-3 pt-2 pb-1 flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-[13px] font-semibold text-slate-800 truncate">{d.label || s.title}</div>
          <div className="text-[10.5px] text-slate-400 truncate">{s.type}</div>
        </div>
        <div className="flex flex-col items-end gap-0.5">
          <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium whitespace-nowrap ${STATUS_STYLE[d.status] || STATUS_STYLE.idle}`} data-testid="node-status">
            {d.status}
          </span>
          <DecisionBadge decisions={d.decisions} />
        </div>
      </div>
      <div className="relative" style={{ height: rows * 20 + 6 }}>
        {s.inputs.map((p, i) => (
          <div key={p.name} className="absolute left-0 text-[10.5px] text-slate-500 pl-3" style={{ top: i * 20 + 2 }}>
            <Handle
              type="target" id={p.name} position={Position.Left}
              style={{ top: 9, background: PORT_COLORS[p.type] || "#999", width: 10, height: 10, border: p.required ? "2px solid white" : "2px dashed white" }}
              title={`${p.name}: ${p.type}${p.multiple ? " (several)" : ""}${p.required ? "" : " (optional)"}`}
            />
            {p.name}
          </div>
        ))}
        {s.outputs.map((p, i) => (
          <div key={p.name} className="absolute right-0 text-[10.5px] text-slate-500 pr-3 text-right" style={{ top: i * 20 + 2 }}>
            {p.name}
            <Handle
              type="source" id={p.name} position={Position.Right}
              style={{ top: 9, background: PORT_COLORS[p.type] || "#999", width: 10, height: 10, border: "2px solid white" }}
              title={`${p.name}: ${p.type}`}
            />
          </div>
        ))}
      </div>
      {(d.summary || d.error || !s.available) && (
        <div className={`px-3 pb-2 text-[10.5px] leading-tight ${d.error || !s.available ? "text-red-600" : "text-slate-500"}`}>
          {d.error ? d.error.slice(0, 140) : !s.available ? s.unavailable_reason.slice(0, 120) : d.summary?.slice(0, 140)}
        </div>
      )}
    </div>
  );
}

export const FlowNode = memo(FlowNodeView);
