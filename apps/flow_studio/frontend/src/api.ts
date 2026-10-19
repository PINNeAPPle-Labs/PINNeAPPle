// Thin client of the Flow Studio API (apps/flow_studio/backend/app.py -> pinneapple_flow.service).
import type { Catalog, PaletteInfo, DecisionsResult, FlowDoc, NextSteps, PlanResult, RunStatus, NodeResult, SealStatus } from "./types";

async function call<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers || {}) },
  });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = j.detail || msg;
    } catch {
      /* not JSON */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

const post = <T,>(path: string, body: unknown) => call<T>(path, { method: "POST", body: JSON.stringify(body) });

export const api = {
  info: () => call<{ version: string; settings: Record<string, string>; planners: string[] }>("/api/info"),
  catalog: () => call<Catalog>("/api/catalog"),
  palette: () => call<PaletteInfo>("/api/palette"),
  paletteSearch: (q: string) => call<{ type: string; section: string }[]>(`/api/palette/search?q=${encodeURIComponent(q)}`),
  setPalettePrefs: (prefs: { favourites?: string[]; recent?: string[] }) =>
    call<{ favourites: string[]; recent: string[] }>("/api/palette/prefs", { method: "PUT", body: JSON.stringify(prefs) }),
  plan: (text: string, backend: string) => post<PlanResult>("/api/plan", { text, backend }),
  validate: (flow: FlowDoc) => post<{ valid: boolean; issues: { level: string; message: string; node: string; edge: string }[] }>("/api/validate", flow),
  previewEquations: (geometry: unknown, equations: unknown) =>
    post<{ ok: boolean; error?: string; system?: any; resolved?: any; geometry?: any }>("/api/preview/equations", { geometry, equations }),
  previewArchitecture: (kind: string, config: unknown, in_dim: number, out_dim: number) =>
    post<any>("/api/preview/architecture", { kind, config, in_dim, out_dim }),
  flows: () => call<{ id: string; name: string; latest_version: number; updated: number }[]>("/api/flows"),
  getFlow: (id: string) => call<FlowDoc>(`/api/flows/${encodeURIComponent(id)}`),
  saveFlow: (flow: FlowDoc, note = "") => post<{ id: string; version: number }>("/api/flows", { flow, note }),
  examples: () => call<{ name: string; title: string; description: string; nodes: number }[]>("/api/examples"),
  example: (name: string) => call<FlowDoc>(`/api/examples/${encodeURIComponent(name)}`),
  run: (flow: FlowDoc, from_node?: string) => post<{ run_id: string }>("/api/runs", { flow, from_node }),
  status: (runId: string) => call<RunStatus>(`/api/runs/${runId}`),
  cancel: (runId: string) => post<{ cancelled: boolean }>(`/api/runs/${runId}/cancel`, {}),
  decisions: (flow: FlowDoc, run_id?: string | null) => post<DecisionsResult>("/api/decisions", { flow, run_id }),
  references: () => call<any[]>("/api/references"),
  checkReference: (r: any) => post<any>("/api/references/check", r),
  customNodes: () => call<any[]>("/api/custom-nodes"),
  saveCustomNode: (spec: any) => post<any>("/api/custom-nodes", spec),
  libraryCalls: () => call<any[]>("/api/library-calls"),
  seal: (item: string) => call<SealStatus>(`/api/lab/seal?item=${encodeURIComponent(item)}`),
  nextSteps: (item: string) => call<NextSteps>(`/api/lab/next-steps?item=${encodeURIComponent(item)}`),
  insertNodes: (flow: FlowDoc, anchor: string | null, nodes: { type: string; params?: Record<string, any> }[]) =>
    post<{ flow: FlowDoc; unwired: string[]; valid: boolean; issues: any[] }>("/api/flows/insert-nodes", { flow, anchor, nodes }),
  node: (runId: string, nodeId: string) => call<NodeResult>(`/api/runs/${runId}/nodes/${encodeURIComponent(nodeId)}`),
};

export const artifactUrl = (path: string) => `/api/artifact?path=${encodeURIComponent(path)}`;
export const htmlArtifactUrl = (path: string) => `/api/artifact/html?path=${encodeURIComponent(path)}`;
