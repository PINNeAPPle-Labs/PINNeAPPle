// Side panel of the selected node: a form generated from the node's parameter schema, with dedicated editors for
// equations (live, safely parsed preview) and surrogate architectures (editable blocks + layer-graph preview).
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { NodeType, ParamSpec } from "../types";
import { LayerGraph } from "./LayerGraph";

const input = "w-full text-[12.5px] border border-slate-200 rounded px-2 py-1 bg-white";

function JsonField({ value, onChange, rows = 4 }: { value: any; onChange: (v: any) => void; rows?: number }) {
  const [text, setText] = useState(JSON.stringify(value ?? null, null, 1));
  const [err, setErr] = useState("");
  useEffect(() => setText(JSON.stringify(value ?? null, null, 1)), [JSON.stringify(value)]);
  return (
    <div>
      <textarea
        className={`${input} font-mono`} rows={rows} value={text}
        onChange={(e) => {
          setText(e.target.value);
          try {
            onChange(JSON.parse(e.target.value));
            setErr("");
          } catch {
            setErr("not valid JSON yet");
          }
        }}
      />
      {err && <div className="text-[10.5px] text-amber-700">{err}</div>}
    </div>
  );
}

function Field({ p, value, onChange }: { p: ParamSpec; value: any; onChange: (v: any) => void }) {
  const v = value === undefined ? p.default : value;
  switch (p.kind) {
    case "int":
    case "float":
      return (
        <input
          type="number" className={input} value={v ?? ""} step={p.kind === "int" ? 1 : "any"}
          min={p.minimum ?? undefined} max={p.maximum ?? undefined}
          onChange={(e) => onChange(e.target.value === "" ? null : p.kind === "int" ? parseInt(e.target.value) : parseFloat(e.target.value))}
        />
      );
    case "bool":
      return <input type="checkbox" checked={!!v} onChange={(e) => onChange(e.target.checked)} />;
    case "enum":
      return (
        <select className={input} value={String(v)} onChange={(e) => {
          const c = (p.choices || []).find((x) => String(x) === e.target.value);
          onChange(c);
        }}>
          {(p.choices || []).map((c) => <option key={String(c)} value={String(c)}>{String(c)}</option>)}
        </select>
      );
    case "text":
      return <textarea className={`${input} font-mono`} rows={5} value={v ?? ""} onChange={(e) => onChange(e.target.value)} />;
    case "equations":
      return (
        <textarea
          className={`${input} font-mono`} rows={Math.max(2, (v || []).length + 1)} value={(v || []).join("\n")}
          onChange={(e) => onChange(e.target.value.split("\n").filter((l) => l.trim() !== ""))}
          data-testid={`param-${p.name}`}
        />
      );
    case "list":
    case "dict":
      return <JsonField value={v} onChange={onChange} rows={p.name === "conditions" ? 6 : 3} />;
    default:
      return <input className={`${input} ${p.kind === "expression" ? "font-mono" : ""}`} value={v ?? ""} onChange={(e) => onChange(e.target.value)} />;
  }
}

function ConditionsEditor({ value, onChange }: { value: any[]; onChange: (v: any[]) => void }) {
  const rows = value || [];
  const set = (i: number, k: string, x: string) => onChange(rows.map((r, j) => (j === i ? { ...r, [k]: x } : r)));
  return (
    <div className="space-y-1">
      {rows.map((r, i) => (
        <div key={i} className="flex gap-1 items-center">
          <select className={`${input} !w-24`} value={r.kind} onChange={(e) => set(i, "kind", e.target.value)}>
            {["dirichlet", "neumann", "robin", "initial", "periodic"].map((k) => <option key={k}>{k}</option>)}
          </select>
          <input className={`${input} !w-20`} value={r.on} onChange={(e) => set(i, "on", e.target.value)} title="x_min, x_max, y_min, ..., boundary, t_min" />
          <input className={`${input} font-mono`} value={r.equation} onChange={(e) => set(i, "equation", e.target.value)} />
          <button className="text-slate-400 hover:text-red-600 px-1" onClick={() => onChange(rows.filter((_, j) => j !== i))}>×</button>
        </div>
      ))}
      <button className="text-[11.5px] text-brand-700" onClick={() => onChange([...rows, { kind: "dirichlet", on: "boundary", equation: "u = 0" }])}>
        + condition
      </button>
    </div>
  );
}

function EquationPreview({ geometry, params, onLoad }: { geometry: any; params: any; onLoad: (eq: any) => void }) {
  const [res, setRes] = useState<any>(null);
  const key = JSON.stringify([geometry, params]);
  useEffect(() => {
    const t = setTimeout(() => api.previewEquations(geometry, params).then(setRes).catch((e) => setRes({ ok: false, error: String(e) })), 350);
    return () => clearTimeout(t);
  }, [key]);
  if (!res) return null;
  if (res.error) return <div className="text-[11.5px] text-red-600 bg-red-50 rounded p-2" data-testid="eq-error">{res.error}</div>;
  const sys = res.system;
  return (
    <div className="bg-slate-50 rounded p-2 text-[11.5px] space-y-1" data-testid="eq-preview">
      <div>
        <b>Inferred:</b> {sys.classification.family} · {sys.classification.linear ? "linear" : "nonlinear"} ·{" "}
        {res.geometry.dimension}-D{sys.classification.time_dependent ? " + time" : ", steady"} · coordinates {res.geometry.coords.join(", ")}
      </div>
      {sys.pdes.map((e: any, i: number) => (
        <div key={i} className="font-mono text-slate-700">PDE: {e.text} <span className="text-slate-400">[{e.derivatives.join(", ")}]</span></div>
      ))}
      {sys.conditions.map((e: any, i: number) => (
        <div key={i} className="font-mono text-slate-600">{e.kind} @ {e.on}: {e.text}</div>
      ))}
      {sys.issues.length > 0 && <div className="text-amber-700">{sys.issues.join("; ")}</div>}
      {params.template !== "custom" && (
        <button className="text-brand-700 font-medium" onClick={() => onLoad(res.resolved)} data-testid="eq-load">
          Edit these equations (stop following the template)
        </button>
      )}
    </div>
  );
}

function ArchitectureEditor({ kind, value, onChange }: { kind: string; value: any; onChange: (v: any) => void }) {
  const [dims, setDims] = useState({ in_dim: 1, out_dim: 81 });
  const [graph, setGraph] = useState<any>(null);
  const [err, setErr] = useState("");
  const cfg = value || {};
  const blocks: any[] = cfg.blocks || [];
  const preview = () =>
    api.previewArchitecture(kind, cfg, dims.in_dim, dims.out_dim).then((g) => { setGraph(g); setErr(""); }).catch((e) => setErr(String(e)));
  useEffect(() => { preview(); }, [kind, JSON.stringify(cfg), dims.in_dim, dims.out_dim]);
  const setBlock = (i: number, k: string, x: any) => onChange({ ...cfg, blocks: blocks.map((b, j) => (j === i ? { ...b, [k]: x } : b)) });
  const usesBlocks = kind === "mlp" || kind === "pod_mlp";
  const effective = graph?.config || {};
  return (
    <div className="space-y-2" data-testid="arch-editor">
      {usesBlocks && (
        <div className="space-y-1">
          {(blocks.length ? blocks : effective.blocks || []).map((b: any, i: number) => (
            <div key={i} className="flex gap-1 items-center">
              <select className={`${input} !w-24`} value={b.type} onChange={(e) => setBlock(i, "type", e.target.value)}>
                {["dense", "residual", "dropout", "layernorm"].map((k) => <option key={k}>{k}</option>)}
              </select>
              {(b.type === "dense" || b.type === "residual") && (
                <>
                  <input className={`${input} !w-16`} type="number" value={b.width ?? 64} onChange={(e) => setBlock(i, "width", parseInt(e.target.value))} />
                  <select className={`${input} !w-20`} value={b.activation ?? "tanh"} onChange={(e) => setBlock(i, "activation", e.target.value)}>
                    {["tanh", "gelu", "relu", "silu", "sin"].map((k) => <option key={k}>{k}</option>)}
                  </select>
                </>
              )}
              {b.type === "dropout" && <input className={`${input} !w-16`} type="number" step="0.05" value={b.p ?? 0.1} onChange={(e) => setBlock(i, "p", parseFloat(e.target.value))} />}
              <button className="text-slate-400 hover:text-red-600 px-1" onClick={() => onChange({ ...cfg, blocks: (blocks.length ? blocks : effective.blocks).filter((_: any, j: number) => j !== i) })}>×</button>
            </div>
          ))}
          <button className="text-[11.5px] text-brand-700" data-testid="add-block"
            onClick={() => onChange({ ...cfg, blocks: [...(blocks.length ? blocks : effective.blocks || []), { type: "dense", width: 64, activation: "gelu" }] })}>
            + block
          </button>
        </div>
      )}
      {!usesBlocks && <JsonField value={cfg} onChange={onChange} rows={3} />}
      <div className="flex gap-2 items-center text-[11px] text-slate-500">
        preview sizes: in <input className={`${input} !w-14`} type="number" value={dims.in_dim} onChange={(e) => setDims({ ...dims, in_dim: parseInt(e.target.value) || 1 })} />
        out <input className={`${input} !w-16`} type="number" value={dims.out_dim} onChange={(e) => setDims({ ...dims, out_dim: parseInt(e.target.value) || 1 })} />
      </div>
      {err && <div className="text-[11px] text-red-600">{err}</div>}
      {graph && <LayerGraph graph={graph} />}
    </div>
  );
}

export function ParamPanel({
  spec, nodeId, label, params, upstreamGeometry, onChange, onLabel, onRunFrom, onDelete,
}: {
  spec: NodeType; nodeId: string; label: string; params: Record<string, any>; upstreamGeometry: any;
  onChange: (p: Record<string, any>) => void; onLabel: (l: string) => void; onRunFrom: () => void; onDelete: () => void;
}) {
  const [adv, setAdv] = useState(false);
  const set = (k: string, v: any) => onChange({ ...params, [k]: v });
  const visible = useMemo(() => spec.params.filter((p) => adv || !p.advanced), [spec, adv]);
  const isEq = spec.type === "equations.system";
  const isArch = spec.type === "surrogate.train" || spec.type === "surrogate.hpo";
  return (
    <div className="p-3 space-y-3 text-[12.5px]" data-testid="param-panel">
      <div>
        <input className="w-full font-semibold text-[14px] border-b border-transparent hover:border-slate-200 focus:border-brand-500 outline-none" value={label} onChange={(e) => onLabel(e.target.value)} />
        <div className="text-[11px] text-slate-400">{nodeId} · {spec.type}</div>
        <div className="text-[11.5px] text-slate-600 mt-1">{spec.description}</div>
        <div className="text-[10.5px] text-slate-400 mt-1 font-mono break-all">wraps {spec.calls}</div>
        {!spec.available && <div className="text-[11.5px] text-red-600 mt-1">Unavailable: {spec.unavailable_reason}</div>}
      </div>
      {isEq && <EquationPreview geometry={upstreamGeometry} params={params} onLoad={(eq) => onChange({ ...params, ...eq, follow_template: false, template: params.template })} />}
      {visible.map((p) => (
        <div key={p.name}>
          <label className="block text-[11px] font-semibold text-slate-600 mb-0.5" title={p.help}>
            {p.label || p.name.replace(/_/g, " ")}
            {p.advanced && <span className="ml-1 text-slate-400 font-normal">(advanced)</span>}
          </label>
          {isEq && p.name === "conditions" ? (
            <ConditionsEditor value={params.conditions ?? p.default} onChange={(v) => set("conditions", v)} />
          ) : isArch && p.name === "architecture" ? (
            <ArchitectureEditor kind={params.kind ?? spec.params.find((x) => x.name === "kind")?.default} value={params.architecture} onChange={(v) => set("architecture", v)} />
          ) : (
            <Field p={p} value={params[p.name]} onChange={(v) => set(p.name, v)} />
          )}
          {p.help && <div className="text-[10.5px] text-slate-400 mt-0.5">{p.help}</div>}
        </div>
      ))}
      {spec.params.some((p) => p.advanced) && (
        <button className="text-[11.5px] text-slate-500" onClick={() => setAdv(!adv)}>{adv ? "hide" : "show"} advanced</button>
      )}
      <div className="flex gap-2 pt-2 border-t">
        <button className="px-2 py-1 rounded bg-brand-500 text-white text-[12px] font-medium hover:bg-brand-600" onClick={onRunFrom} data-testid="run-from">
          Run from here
        </button>
        <button className="px-2 py-1 rounded border text-[12px] text-red-600 border-red-200 hover:bg-red-50" onClick={onDelete}>Delete node</button>
      </div>
    </div>
  );
}
