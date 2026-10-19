import {
  addEdge, Background, Controls, MiniMap, ReactFlow, ReactFlowProvider, useEdgesState, useNodesState, useReactFlow,
  type Connection, type Edge, type Node,
} from "@xyflow/react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api } from "./api";
import { FlowNode, type FlowNodeData } from "./components/FlowNode";
import { Palette } from "./components/Palette";
import { ParamPanel } from "./components/ParamPanel";
import { PlannerDialog } from "./components/PlannerDialog";
import { CustomNodeDialog } from "./components/CustomNodeDialog";
import { DecisionList } from "./components/DecisionsPanel";
import { ResultPanel } from "./components/ResultPanel";
import { SealPanel } from "./components/SealPanel";
import type { Catalog, Decision, DecisionsResult, FlowDoc, NodeResult, NodeType } from "./types";
import { PORT_COLORS } from "./types";

const nodeTypes = { flow: FlowNode };
const SCHEMA = "pinneapple-flow/1";

function compatibleFn(cat: Catalog) {
  const conv = new Set(cat.conversions.map(([a, b]) => `${a}>${b}`));
  return (s: string, t: string) => s === t || t === "any" || s === "any" || conv.has(`${s}>${t}`);
}

function Studio() {
  const [catalog, setCatalog] = useState<Catalog | null>(null);
  const [added, setAdded] = useState({ type: "", n: 0 });
  const [info, setInfo] = useState<{ version: string; planners: string[] } | null>(null);
  const [nodes, setNodes, onNodesChange] = useNodesState<Node>([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState<Edge>([]);
  const [meta, setMeta] = useState({ id: "untitled", name: "Untitled flow", description: "", meta: {} as Record<string, any> });
  const [selected, setSelected] = useState<string | null>(null);
  const [tab, setTab] = useState<"params" | "results" | "decisions">("params");
  const [decisions, setDecisions] = useState<DecisionsResult | null>(null);
  const [showDecisions, setShowDecisions] = useState(false);
  const [showCustom, setShowCustom] = useState(false);
  const [runId, setRunId] = useState<string | null>(null);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState<NodeResult | null>(null);
  const [issues, setIssues] = useState<{ level: string; message: string; node: string }[]>([]);
  const [toast, setToast] = useState("");
  const [showPlanner, setShowPlanner] = useState(false);
  const [lists, setLists] = useState<{ examples: any[]; flows: any[] } | null>(null);
  const wrapper = useRef<HTMLDivElement>(null);
  const rf = useReactFlow();
  const specs = useMemo(() => Object.fromEntries((catalog?.nodes || []).map((n) => [n.type, n])), [catalog]);
  const compatible = useMemo(() => (catalog ? compatibleFn(catalog) : () => false), [catalog]);

  useEffect(() => {
    api.catalog().then(setCatalog).catch((e) => setToast(String(e)));
    api.info().then(setInfo).catch(() => undefined);
  }, []);

  const say = (m: string) => { setToast(m); setTimeout(() => setToast(""), 4000); };

  // ---- document <-> canvas --------------------------------------------------------------------------------------
  const toDoc = useCallback((): FlowDoc => ({
    schema: SCHEMA, id: meta.id, name: meta.name, description: meta.description, meta: meta.meta,
    nodes: nodes.map((n) => {
      const d = n.data as FlowNodeData;
      return { id: n.id, type: d.spec.type, params: d.params, position: n.position, label: d.label };
    }),
    edges: edges.map((e) => ({ id: e.id, source: e.source, source_port: e.sourceHandle || "", target: e.target, target_port: e.targetHandle || "" })),
  }), [nodes, edges, meta]);

  const load = useCallback((doc: FlowDoc) => {
    if (!catalog) return;
    const unknown = doc.nodes.filter((n) => !specs[n.type]).map((n) => n.type);
    if (unknown.length) say(`unknown node types skipped: ${unknown.join(", ")}`);
    setNodes(doc.nodes.filter((n) => specs[n.type]).map((n) => ({
      id: n.id, type: "flow", position: n.position || { x: 0, y: 0 },
      data: { spec: specs[n.type], label: n.label || specs[n.type].title, params: n.params || {}, status: "idle" } as FlowNodeData,
    })));
    const typeOf = (id: string) => doc.nodes.find((n) => n.id === id)?.type || "";
    setEdges(doc.edges.map((e) => edgeOf(e.source, e.source_port, e.target, e.target_port,
      specs[typeOf(e.source)]?.outputs.find((o) => o.name === e.source_port)?.type)));
    setMeta({ id: doc.id || "untitled", name: doc.name || "flow", description: doc.description || "", meta: doc.meta || {} });
    setSelected(null);
    setRunId(null);
    setResult(null);
    setTimeout(fitReadable, 60);
  }, [catalog, specs]);

  // fit the flow, but never below a zoom where node titles and port labels stay readable; a flow wider than the
  // canvas then starts at its first column (pan or use the minimap for the rest)
  const fitReadable = useCallback(() => {
    const MIN = 0.8;
    rf.fitView({ padding: 0.08, minZoom: MIN, maxZoom: 1.1 });
    const vp = rf.getViewport();
    const b = rf.getNodesBounds(rf.getNodes());
    const w = wrapper.current?.clientWidth || 0;
    if (b.width * vp.zoom + 40 > w) rf.setViewport({ x: 24 - b.x * vp.zoom, y: vp.y, zoom: vp.zoom });
  }, [rf]);

  const edgeOf = (s: string, sp: string, t: string, tp: string, type?: string): Edge => {
    return { id: `${s}.${sp}->${t}.${tp}`, source: s, sourceHandle: sp, target: t, targetHandle: tp, style: { stroke: PORT_COLORS[type || ""] || "#94a3b8", strokeWidth: 2 } };
  };

  // ---- editing ------------------------------------------------------------------------------------------------
  const portType = (nodeId: string | null, handle: string | null | undefined, dir: "in" | "out") => {
    const n = nodes.find((x) => x.id === nodeId);
    const s = n && (n.data as FlowNodeData).spec;
    const p = s && (dir === "out" ? s.outputs : s.inputs).find((x) => x.name === handle);
    return p;
  };
  const isValidConnection = useCallback((c: Connection | Edge) => {
    const o = portType(c.source, c.sourceHandle, "out");
    const i = portType(c.target, c.targetHandle, "in");
    if (!o || !i || c.source === c.target) return false;
    if (!compatible(o.type, i.type)) return false;
    if (!i.multiple && edges.some((e) => e.target === c.target && e.targetHandle === c.targetHandle)) return false;
    return true;
  }, [nodes, edges, compatible]);
  const onConnect = useCallback((c: Connection) => {
    const o = portType(c.source, c.sourceHandle, "out");
    setEdges((eds) => addEdge({ ...c, id: `${c.source}.${c.sourceHandle}->${c.target}.${c.targetHandle}`, style: { stroke: PORT_COLORS[o?.type || ""] || "#94a3b8", strokeWidth: 2 } }, eds));
  }, [nodes]);

  const addNode = (type: string, position?: { x: number; y: number }) => {
    const s = specs[type];
    if (!s) return;
    let k = 1;
    const base = type.split(".").pop()!.replace(/[^a-z0-9_]/gi, "_");
    while (nodes.some((n) => n.id === `${base}${k}`)) k++;
    const id = `${base}${k}`;
    const box = wrapper.current?.getBoundingClientRect();
    const pos = position || rf.screenToFlowPosition({ x: (box?.left || 0) + (box?.width || 800) / 2, y: (box?.top || 0) + (box?.height || 400) / 2 });
    setNodes((ns) => [...ns, { id, type: "flow", position: pos, data: { spec: s, label: s.title, params: {}, status: "idle" } as FlowNodeData }]);
    setSelected(id);
    setTab("params");
    setAdded((a) => ({ type, n: a.n + 1 }));
  };
  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const type = e.dataTransfer.getData("application/pinneapple-node");
    if (type) addNode(type, rf.screenToFlowPosition({ x: e.clientX, y: e.clientY }));
  };
  // add registered nodes suggested by a seal gap or a weak decision, wired by port type in the library
  const insertNodes = async (anchor: string | null, items: { type: string; params?: Record<string, any> }[]) => {
    try {
      const keep = runId;
      const r = await api.insertNodes(toDoc(), anchor, items);
      load(r.flow);
      setRunId(keep);
      say(`added ${items.map((i) => i.type).join(", ")}${r.unwired.length ? ` · connect: ${r.unwired.join(", ")}` : " (wired)"}`);
    } catch (e) {
      say(String(e));
    }
  };

  const patchNode = (id: string, patch: Partial<FlowNodeData>) => {
    if (patch.params) {
      const old = (nodes.find((n) => n.id === id)?.data as FlowNodeData | undefined)?.params || {};
      const changed = Object.keys(patch.params).filter((k) => JSON.stringify(patch.params![k]) !== JSON.stringify(old[k]));
      if (changed.length) {
        setMeta((m) => {
          const edits = (m.meta.user_edits || []).filter((e: any) => !(e.node === id && changed.includes(e.param)));
          return { ...m, meta: { ...m.meta, user_edits: [...edits, ...changed.map((k) => ({ node: id, param: k, value: patch.params![k] }))] } };
        });
      }
    }
    setNodes((ns) => ns.map((n) => (n.id === id ? { ...n, data: { ...n.data, ...patch } } : n)));
  };
  const annotate = (d: Decision, notes: string, ref: string) => {
    const [, node, param] = d.id.split(":");
    setMeta((m) => ({ ...m, meta: { ...m.meta, user_edits: (m.meta.user_edits || []).map((e: any) => e.node === node && e.param === param
      ? { ...e, notes, references: ref ? [...(e.references || []), ref.startsWith("10.") ? { doi: ref, title: ref } : ref.startsWith("http") ? { url: ref, title: ref } : ref] : e.references }
      : e) } }));
  };

  // ---- validation -----------------------------------------------------------------------------------------------
  useEffect(() => {
    if (!catalog) return;
    const t = setTimeout(() => api.validate(toDoc()).then((r) => setIssues(r.issues)).catch(() => undefined), 400);
    return () => clearTimeout(t);
  }, [nodes, edges, catalog]);

  useEffect(() => {
    if (!catalog || running) return;
    const t = setTimeout(() => api.decisions(toDoc(), runId).then((r) => {
      setDecisions(r);
      setNodes((ns) => ns.map((n) => ({ ...n, data: { ...n.data, decisions: r.by_node[n.id] || [] } })));
    }).catch(() => undefined), 600);
    return () => clearTimeout(t);
  }, [edges, meta, catalog, runId, running, nodes.length, JSON.stringify(nodes.map((n) => (n.data as FlowNodeData).params))]);

  // ---- running --------------------------------------------------------------------------------------------------
  const run = async (fromNode?: string) => {
    try {
      const r = await api.run(toDoc(), fromNode);
      setRunId(r.run_id);
      setRunning(true);
      setNodes((ns) => ns.map((n) => ({ ...n, data: { ...n.data, status: "queued", error: undefined, summary: undefined } })));
      say(`run ${r.run_id} started`);
    } catch (e) {
      say(String(e));
    }
  };
  useEffect(() => {
    if (!runId || !running) return;
    const iv = setInterval(async () => {
      try {
        const st = await api.status(runId);
        setNodes((ns) => ns.map((n) => {
          const s = st.nodes[n.id];
          return s ? { ...n, data: { ...n.data, status: s.status || "queued", summary: s.summary, error: s.error || undefined } } : n;
        }));
        if (["done", "failed", "cancelled"].includes(st.status)) {
          setRunning(false);
          say(`run ${st.status}${st.error ? `: ${st.error}` : ""}`);
        }
      } catch {
        /* the next poll retries */
      }
    }, 700);
    return () => clearInterval(iv);
  }, [runId, running]);
  useEffect(() => {
    if (!runId || !selected) { setResult(null); return; }
    const st = (nodes.find((n) => n.id === selected)?.data as FlowNodeData | undefined)?.status;
    if (st && ["done", "failed", "cached", "degraded", "not_run"].includes(st)) api.node(runId, selected).then(setResult).catch(() => setResult(null));
  }, [selected, runId, nodes.find((n) => n.id === selected)?.data?.status]);

  // ---- files ----------------------------------------------------------------------------------------------------
  const exportJson = () => {
    const blob = new Blob([JSON.stringify(toDoc(), null, 1)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${meta.id || "flow"}.json`;
    a.click();
  };
  const importJson = (f: File) => f.text().then((t) => load(JSON.parse(t))).catch((e) => say(`import failed: ${e}`));
  const save = () => api.saveFlow(toDoc()).then((r) => say(`saved ${r.id} v${r.version}`)).catch((e) => say(String(e)));
  const openLists = async () => setLists({ examples: await api.examples(), flows: await api.flows() });

  const sel = nodes.find((n) => n.id === selected);
  const selData = sel?.data as FlowNodeData | undefined;
  const upstreamGeometry = useMemo(() => {
    if (!sel) return {};
    const e = edges.find((x) => x.target === sel.id && x.targetHandle === "geometry");
    const g = e && nodes.find((n) => n.id === e.source);
    return g ? (g.data as FlowNodeData).params : {};
  }, [sel, edges, nodes]);
  const errors = issues.filter((i) => i.level === "error");

  if (!catalog) return <div className="p-8 text-slate-500">Loading the node catalog… {toast}</div>;
  return (
    <div className="h-screen flex flex-col">
      <header className="h-12 shrink-0 flex items-center gap-2 px-3 bg-white border-b border-slate-200">
        <div className="font-bold text-[15px] mr-2">🍍 PINNeAPPle <span className="text-brand-600">Flow Studio</span></div>
        <input className="text-sm border border-slate-200 rounded px-2 py-1 w-56" value={meta.name} onChange={(e) => setMeta({ ...meta, name: e.target.value })} data-testid="flow-name" />
        <button className="btn-primary" onClick={() => setShowPlanner(true)} data-testid="open-planner">✨ Plan from text</button>
        <button className="btn" onClick={openLists} data-testid="open-lists">Open…</button>
        <button className="btn" onClick={save} data-testid="save">Save</button>
        <button className="btn" onClick={exportJson}>Export JSON</button>
        <label className="btn cursor-pointer">Import<input type="file" accept=".json" hidden onChange={(e) => e.target.files?.[0] && importJson(e.target.files[0])} /></label>
        <button className="btn" onClick={() => load({ schema: SCHEMA, id: "untitled", name: "Untitled flow", description: "", nodes: [], edges: [], meta: {} })}>New</button>
        <button className="btn" onClick={() => setShowDecisions(true)} data-testid="open-decisions">
          Decisions &amp; references{decisions?.summary.no_reference ? ` (${decisions.summary.no_reference} ⚠)` : ""}
        </button>
        <div className="flex-1" />
        <span className={`text-[12px] ${errors.length ? "text-red-600" : "text-emerald-700"}`} title={errors.map((e) => `${e.node}: ${e.message}`).join("\n")} data-testid="validity">
          {nodes.length === 0 ? "" : errors.length ? `${errors.length} issue(s)` : "✓ valid"}
        </span>
        {running ? (
          <button className="btn border-red-300 text-red-700" onClick={() => runId && api.cancel(runId)}>Stop</button>
        ) : (
          <button className="btn-primary" disabled={!nodes.length || errors.length > 0} onClick={() => run()} data-testid="run">▶ Run flow</button>
        )}
      </header>
      <div className="flex flex-1 min-h-0">
        <Palette nodes={catalog.nodes} onAdd={(t) => addNode(t)} onNewType={() => setShowCustom(true)}
          recentBump={added.type ? `${added.type}#${added.n}` : undefined} />
        <div className="flex-1 relative" ref={wrapper} onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = "move"; }} onDrop={onDrop} data-testid="canvas">
          <ReactFlow
            nodes={nodes} edges={edges} nodeTypes={nodeTypes} onNodesChange={onNodesChange} onEdgesChange={onEdgesChange}
            onConnect={onConnect} isValidConnection={isValidConnection} onNodeClick={(_, n) => { setSelected(n.id); setTab(runId ? "results" : "params"); }}
            onPaneClick={() => setSelected(null)} fitView fitViewOptions={{ padding: 0.08, minZoom: 0.8, maxZoom: 1.1 }} deleteKeyCode={["Delete", "Backspace"]} proOptions={{ hideAttribution: true }}
          >
            <Background gap={18} color="#e2e8f0" />
            <Controls />
            <MiniMap pannable zoomable nodeStrokeWidth={3} />
          </ReactFlow>
          {nodes.length === 0 && (
            <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
              <div className="text-center text-slate-400 text-sm">Describe a problem with <b>Plan from text</b>, open an example, or drag nodes from the palette.</div>
            </div>
          )}
          {meta.meta?.planner && <div className="absolute top-2 left-2 text-[11px] bg-white/90 border rounded px-2 py-1 text-slate-500" data-testid="planner-badge">planned by {meta.meta.planner}</div>}
        </div>
        <aside className="w-[380px] shrink-0 border-l border-slate-200 bg-white flex flex-col min-h-0" data-testid="side">
          {sel && selData ? (
            <>
              <div className="flex border-b text-[12.5px]">
                {(["params", "results", "decisions"] as const).map((t) => (
                  <button key={t} className={`flex-1 py-2 ${tab === t ? "border-b-2 border-brand-500 font-semibold" : "text-slate-500"}`} onClick={() => setTab(t)} data-testid={`tab-${t}`}>
                    {t === "params" ? "Parameters" : t === "results" ? "Results" : `Decisions${(selData.decisions || []).some((d) => d.no_reference) ? " ⚠" : ""}`}
                  </button>
                ))}
              </div>
              <div className="overflow-y-auto flex-1">
                {tab === "params" ? (
                  <ParamPanel
                    spec={selData.spec as NodeType} nodeId={sel.id} label={selData.label} params={selData.params}
                    upstreamGeometry={upstreamGeometry}
                    onChange={(p) => patchNode(sel.id, { params: p })} onLabel={(l) => patchNode(sel.id, { label: l })}
                    onRunFrom={() => run(sel.id)}
                    onDelete={() => { setNodes((ns) => ns.filter((n) => n.id !== sel.id)); setEdges((es) => es.filter((e) => e.source !== sel.id && e.target !== sel.id)); setSelected(null); }}
                  />
                ) : tab === "results" ? (
                  <>
                    {result?.lab && result.lab.length > 0 && (
                      <div className="px-3 pt-3"><SealPanel seals={result.lab} onInsert={(ns) => insertNodes(sel.id, ns)} /></div>
                    )}
                    <ResultPanel result={result} onAccept={(k) => patchNode(sel.id, { params: { ...selData.params, candidate: k } })} />
                  </>
                ) : (
                  <div className="p-3"><DecisionList decisions={selData.decisions || []} onAnnotate={annotate} onRaise={(d, t) => insertNodes(d.node || sel.id, [{ type: t }])} /></div>
                )}
              </div>
            </>
          ) : (
            <div className="p-4 text-[12.5px] text-slate-500 space-y-3 overflow-y-auto">
              <div className="font-semibold text-slate-700">{meta.name}</div>
              {meta.description && <div>{meta.description}</div>}
              <div>Select a node to edit its parameters or see its results. Connect outputs to inputs of a compatible type (colours show the types).</div>
              {errors.length > 0 && (
                <div className="space-y-1">{errors.slice(0, 12).map((e, i) => <div key={i} className="text-red-600 text-[11.5px]">{e.node || "flow"}: {e.message}</div>)}</div>
              )}
              <div className="text-[11px] text-slate-400">Flow Studio {info?.version} · planners: {info?.planners.join(", ")}</div>
            </div>
          )}
        </aside>
      </div>
      {showPlanner && <PlannerDialog planners={info?.planners || ["heuristic"]} onClose={() => setShowPlanner(false)} onOpen={(f) => { load(f); setShowPlanner(false); }} />}
      {lists && (
        <div className="fixed inset-0 bg-black/30 z-50 flex items-start justify-center pt-16" onClick={() => setLists(null)}>
          <div className="bg-white rounded-xl shadow-xl w-[620px] max-h-[75vh] overflow-y-auto p-5" onClick={(e) => e.stopPropagation()} data-testid="lists">
            <h2 className="font-semibold mb-2">Examples</h2>
            {lists.examples.map((x) => (
              <button key={x.name} className="block w-full text-left px-2 py-1.5 rounded hover:bg-brand-50 text-sm" data-testid={`example-${x.name}`}
                onClick={async () => { load(await api.example(x.name)); setLists(null); }}>
                <b>{x.title}</b> <span className="text-slate-500 text-[12px]">{x.description.slice(0, 110)}</span>
              </button>
            ))}
            <h2 className="font-semibold mt-4 mb-2">Saved flows</h2>
            {lists.flows.length === 0 && <div className="text-sm text-slate-400">none yet</div>}
            {lists.flows.map((x) => (
              <button key={x.id} className="block w-full text-left px-2 py-1.5 rounded hover:bg-brand-50 text-sm"
                onClick={async () => { load(await api.getFlow(x.id)); setLists(null); }}>
                <b>{x.name}</b> <span className="text-slate-500 text-[12px]">v{x.latest_version}</span>
              </button>
            ))}
          </div>
        </div>
      )}
      {showCustom && <CustomNodeDialog catalog={catalog.nodes} onClose={() => setShowCustom(false)}
        onSaved={() => api.catalog().then(setCatalog).then(() => say("node type saved"))} />}
      {showDecisions && decisions && (
        <div className="fixed inset-0 bg-black/30 z-50 flex items-start justify-center pt-12" onClick={() => setShowDecisions(false)}>
          <div className="bg-white rounded-xl shadow-xl w-[820px] max-h-[82vh] overflow-y-auto p-5" onClick={(e) => e.stopPropagation()} data-testid="decisions-report">
            <h2 className="font-semibold mb-1">Decisions and references</h2>
            <div className="text-[12px] text-slate-600 mb-3">
              {decisions.summary.decisions} decisions · {decisions.summary.no_reference} without a reference · high {decisions.summary.by_label.high}, medium {decisions.summary.by_label.medium}, low {decisions.summary.by_label.low}
            </div>
            {Object.entries(decisions.by_node).map(([nid, ds]) => (
              <div key={nid} className="mb-3">
                <div className="text-[12px] font-semibold mb-1">{nid || "flow"}</div>
                <DecisionList decisions={ds} onAnnotate={annotate} onRaise={(d, t) => { setShowDecisions(false); insertNodes(d.node || null, [{ type: t }]); }} />
              </div>
            ))}
          </div>
        </div>
      )}
      {toast && <div className="fixed bottom-4 left-1/2 -translate-x-1/2 bg-slate-900 text-white text-sm px-3 py-2 rounded shadow" data-testid="toast">{toast}</div>}
    </div>
  );
}

export default function App() {
  return (
    <ReactFlowProvider>
      <Studio />
    </ReactFlowProvider>
  );
}
