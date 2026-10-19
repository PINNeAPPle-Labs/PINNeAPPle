// Shapes of the pinneapple-flow/1 document and the API answers.
export interface Port { name: string; type: string; required: boolean; multiple: boolean; help: string }
export interface ParamSpec {
  name: string; kind: string; default: any; help: string; choices: any[] | null;
  minimum: number | null; maximum: number | null; advanced: boolean; label: string;
}
export interface NodeType {
  type: string; title: string; category: string; description: string; calls: string;
  inputs: Port[]; outputs: Port[]; params: ParamSpec[]; available: boolean; unavailable_reason: string;
  tags?: string[]; palette?: PaletteEntry;
}
export interface PaletteEntry {
  section: string; title: string; summary: string; tags: string[]; package: string; curated: boolean; group: string; install: string;
}
export interface PaletteSection { id: string; title: string; glyph: string; description: string; collapsed: boolean; count: number; types: string[] }
export interface PaletteInfo { sections: PaletteSection[]; favourites: string[]; recent: string[] }
export interface Catalog { nodes: NodeType[]; port_types: Record<string, string>; conversions: [string, string][] }

export interface FlowNodeDoc { id: string; type: string; params: Record<string, any>; position: { x: number; y: number }; label: string }
export interface FlowEdgeDoc { id?: string; source: string; source_port: string; target: string; target_port: string }
export interface FlowDoc {
  schema: string; id: string; name: string; description: string;
  nodes: FlowNodeDoc[]; edges: FlowEdgeDoc[]; meta: Record<string, any>;
}

export interface Proposal { title: string; rationale: string; flow: FlowDoc; planner: string; repairs?: string[] }
export interface PlanResult {
  planner: string; proposals: Proposal[]; warnings: string[]; detected: Record<string, any>; fallback_from?: string;
}

export type Status = "idle" | "queued" | "running" | "done" | "degraded" | "not_run" | "failed" | "cached" | "cancelled";
export interface NodeBrief { status: Status; error?: string; seconds?: number; summary?: string; metrics?: Record<string, any>; warnings?: string[] }
export interface RunStatus { run_id: string; status: string; nodes: Record<string, NodeBrief>; error?: string }
export interface NodeResult {
  node: string; status: Status; error?: string; seconds?: number; log: string; outputs: Record<string, any>; lab?: SealStatus[];
  report: {
    metrics: Record<string, any>; figures: { path: string; name: string; caption: string }[];
    tables: { title: string; columns: string[]; rows: any[][] }[]; views: Record<string, any>;
    warnings: string[]; summary: string;
  };
}

export const PORT_COLORS: Record<string, string> = {
  geometry: "#0ea5e9", mesh: "#06b6d4", equations: "#8b5cf6", problem: "#7c3aed", solution: "#16a34a",
  model: "#ea580c", dataset: "#ca8a04", series: "#d97706", metrics: "#475569", params: "#db2777",
  architecture: "#9333ea", figure: "#2563eb", scene: "#0891b2", twin: "#059669", report: "#64748b", any: "#94a3b8",
  vehicle: "#b45309", soil: "#a16207", site: "#78716c", vehicle_run: "#0d9488",
};

export const STATUS_STYLE: Record<string, string> = {
  idle: "bg-slate-100 text-slate-500", queued: "bg-slate-200 text-slate-700", running: "bg-amber-100 text-amber-800 animate-pulse",
  done: "bg-emerald-100 text-emerald-800", cached: "bg-sky-100 text-sky-800", failed: "bg-red-100 text-red-800",
  cancelled: "bg-zinc-200 text-zinc-700", degraded: "bg-orange-100 text-orange-800", not_run: "bg-zinc-100 text-zinc-600 italic",
};

export interface Reference { key?: string; title?: string; authors?: string[]; year?: number; venue?: string; link?: string; status?: string; note?: string; supports?: string }
export interface Decision {
  id: string; topic: string; what: string; decided_by: { kind: string; id: string }; rationale: string; value: any;
  references: Reference[]; evidence: { kind: string; ref: string; detail: string }[]; node: string; notes: string;
  uncited_reason: string; confidence: number; label: string; no_reference: boolean; suggestion: string; raise_with?: string[];
}
export interface DecisionsResult { decisions: Decision[]; by_node: Record<string, Decision[]>; summary: { decisions: number; no_reference: number; by_label: Record<string, number>; references: string[] } }

export interface SealStatus {
  item: string; letter: string | null; name: string; scale: string; reason: string; failing_check: string; hint: string;
  sealed: boolean; source: string; curated?: string;
}
export interface GapAction { id: string; kind: string; target: string; command: string; text: string; nodes: { type: string; params: Record<string, any> }[] }
export interface Gap { gap: string; why?: string; to?: string; check?: string; action: string; command: string; actions?: GapAction[]; nodes: { type: string; params: Record<string, any> }[] }
export interface NextSteps { available: boolean; message: string; gaps: Gap[] }
