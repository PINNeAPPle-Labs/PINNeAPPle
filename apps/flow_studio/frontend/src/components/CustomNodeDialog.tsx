// Define a new node type from a registered library call: title, category, parameters fixed or hidden. Saved in the
// store; the backend accepts only allowlisted library calls (no code in the definition).
import { useEffect, useMemo, useState } from "react";
import { api } from "../api";
import type { NodeType } from "../types";

export function CustomNodeDialog({ catalog, onClose, onSaved }: { catalog: NodeType[]; onClose: () => void; onSaved: () => void }) {
  const [calls, setCalls] = useState<any[]>([]);
  const [base, setBase] = useState("");
  const [name, setName] = useState("");
  const [title, setTitle] = useState("");
  const [category, setCategory] = useState("Custom");
  const [values, setValues] = useState<Record<string, { default?: string; hidden: boolean }>>({});
  const [err, setErr] = useState("");
  useEffect(() => { api.libraryCalls().then(setCalls).catch((e) => setErr(String(e))); }, []);
  const spec = useMemo(() => catalog.find((n) => n.type === `lib.${base}`), [catalog, base]);
  const save = async () => {
    const params: Record<string, any> = {};
    for (const [k, v] of Object.entries(values)) {
      if (v.default === undefined || v.default === "") { if (v.hidden) setErr(`give ${k} a value to hide it`); continue; }
      let val: any = v.default;
      try { val = JSON.parse(v.default); } catch { /* a plain string */ }
      params[k] = { default: val, hidden: v.hidden };
    }
    try {
      await api.saveCustomNode({ type: `custom.${name}`, base, title, category, params });
      onSaved();
      onClose();
    } catch (e) {
      setErr(String(e));
    }
  };
  return (
    <div className="fixed inset-0 bg-black/30 z-50 flex items-start justify-center pt-12" onClick={onClose}>
      <div className="bg-white rounded-xl shadow-xl w-[640px] max-h-[82vh] overflow-y-auto p-5 space-y-2 text-sm" onClick={(e) => e.stopPropagation()} data-testid="custom-node">
        <h2 className="font-semibold">New node type from a library call</h2>
        <select className="w-full border rounded px-2 py-1" value={base} onChange={(e) => { setBase(e.target.value); setValues({}); }} data-testid="custom-base">
          <option value="">choose a registered library call…</option>
          {calls.map((c) => <option key={c.key} value={c.key}>{c.title} — {c.target}</option>)}
        </select>
        <div className="flex gap-2">
          <input className="border rounded px-2 py-1 flex-1" placeholder="type name (custom.<name>)" value={name} onChange={(e) => setName(e.target.value.toLowerCase().replace(/[^a-z0-9_]/g, "_"))} data-testid="custom-name" />
          <input className="border rounded px-2 py-1 flex-1" placeholder="title" value={title} onChange={(e) => setTitle(e.target.value)} />
          <input className="border rounded px-2 py-1 w-28" placeholder="category" value={category} onChange={(e) => setCategory(e.target.value)} />
        </div>
        {spec && spec.params.map((p) => (
          <div key={p.name} className="flex gap-2 items-center">
            <span className="w-40 text-[12px] font-mono">{p.name}</span>
            <input className="border rounded px-2 py-0.5 flex-1 text-[12px]" placeholder={`default ${JSON.stringify(p.default)}`}
              value={values[p.name]?.default ?? ""} onChange={(e) => setValues({ ...values, [p.name]: { ...(values[p.name] || { hidden: false }), default: e.target.value } })} />
            <label className="text-[11px]"><input type="checkbox" checked={values[p.name]?.hidden || false}
              onChange={(e) => setValues({ ...values, [p.name]: { ...(values[p.name] || {}), hidden: e.target.checked } })} /> fixed &amp; hidden</label>
          </div>
        ))}
        {err && <div className="text-red-600 text-[12px]">{err}</div>}
        <div className="flex justify-end gap-2">
          <button className="btn" onClick={onClose}>Cancel</button>
          <button className="btn-primary" disabled={!base || !name} onClick={save} data-testid="custom-save">Save node type</button>
        </div>
      </div>
    </div>
  );
}
