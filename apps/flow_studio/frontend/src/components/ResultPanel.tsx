// What a node produced in the last run: summary, warnings, metrics, tables, figures, the 3-D scene, the
// architecture, the parsed equations, the log.
import { artifactUrl, htmlArtifactUrl } from "../api";
import type { NodeResult } from "../types";
import { LayerGraph } from "./LayerGraph";
import { Viewer3D } from "./Viewer3D";

const fmt = (v: any): string => {
  if (v && typeof v === "object" && "value" in v) return fmt(v.value);
  if (typeof v === "number") return Math.abs(v) !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5) ? v.toExponential(3) : String(+v.toPrecision(5));
  if (v === null || v === undefined) return "—";
  return typeof v === "object" ? JSON.stringify(v) : String(v);
};

export function ResultPanel({ result, onAccept }: { result: NodeResult | null; onAccept?: (k: number) => void }) {
  if (!result) return <div className="p-3 text-[12px] text-slate-400">Run the flow, then click a node to see its results.</div>;
  const r = result.report || ({} as any);
  const views = r.views || {};
  return (
    <div className="p-3 space-y-3 text-[12.5px]" data-testid="result-panel">
      <div className="flex items-center justify-between">
        <span className="font-semibold">Results · {result.node}</span>
        <span className="text-[11px] text-slate-500">{result.status}{result.seconds != null ? ` · ${result.seconds.toFixed(2)} s` : ""}</span>
      </div>
      {result.error && <pre className="text-[11px] text-red-700 bg-red-50 p-2 rounded whitespace-pre-wrap">{result.error}</pre>}
      {r.summary && <div className="text-slate-700">{r.summary}</div>}
      {(r.warnings || []).map((w: string, i: number) => <div key={i} className="text-[11.5px] text-amber-800 bg-amber-50 rounded px-2 py-1">{w}</div>)}
      {r.metrics && Object.keys(r.metrics).length > 0 && (
        <table className="w-full text-[11.5px]" data-testid="metrics">
          <tbody>
            {Object.entries(r.metrics).map(([k, v]: [string, any]) => (
              <tr key={k} className="border-b border-slate-100" title={v && typeof v === "object" && v.how ? v.how : ""}>
                <td className="py-0.5 pr-2 text-slate-500">{k}</td>
                <td className="py-0.5 font-mono text-right">{fmt(v)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {views.candidates && (
        <div data-testid="candidates">
          <div className="font-semibold text-[11.5px] mb-1">Identified systems</div>
          {views.candidates.discovered && (
            <div className="text-[11px] bg-sky-50 rounded p-1.5 mb-1">from the data: <span className="font-mono">{views.candidates.discovered.pretty}</span> (R² {fmt(views.candidates.discovered.r2)})</div>
          )}
          {views.candidates.candidates.map((c: any, k: number) => (
            <div key={k} className={`border rounded p-2 mb-1 ${k === views.candidates.accepted ? "border-brand-500" : "border-slate-200"}`}>
              <div className="flex justify-between">
                <b className="text-[12px]">#{k} {c.family}</b>
                <span className="text-[11px]">confidence {c.confidence} · {c.source}</span>
              </div>
              {c.pdes.map((p: string) => <div key={p} className="font-mono text-[11px]">{p}</div>)}
              <div className="text-[10.5px] text-slate-500">{c.conservation_law}; {c.constitutive_relation}</div>
              <div className="text-[10.5px] text-slate-500">{c.reasoning.join("; ")}</div>
              {onAccept && k !== views.candidates.accepted && (
                <button className="text-[11px] text-sky-700" onClick={() => onAccept(k)}>accept this one</button>
              )}
            </div>
          ))}
        </div>
      )}
      {views.architecture && (
        <div><div className="font-semibold text-[11.5px] mb-1">Architecture</div><LayerGraph graph={views.architecture} /></div>
      )}
      {views.html && (
        <div data-testid="html-view">
          <div className="flex items-center justify-between mb-1">
            <div className="font-semibold text-[11.5px]">{views.html.title || "Dashboard"}</div>
            <a className="text-[11px] text-sky-700" href={htmlArtifactUrl(views.html.path)} target="_blank" rel="noopener">open full size ↗</a>
          </div>
          {/* the dashboard runs in its own opaque origin: scripts only, no access to the app */}
          <iframe src={htmlArtifactUrl(views.html.path)} sandbox="allow-scripts allow-downloads" title={views.html.title || "dashboard"}
            className="w-full rounded border border-slate-200 bg-black" style={{ height: 420 }} data-testid="html-frame" />
          {views.html.links && (
            <div className="flex flex-wrap gap-2 mt-1 text-[11px]" data-testid="html-links">
              {Object.entries(views.html.links as Record<string, string>).filter(([k]) => k !== "html").map(([k, p]) => (
                <a key={k} className="text-sky-700 underline" href={artifactUrl(p)} target="_blank" rel="noopener" download={k !== "png" && k !== "frame" ? true : undefined}>{k.toUpperCase()}</a>
              ))}
            </div>
          )}
        </div>
      )}
      {views.video && (
        <figure data-testid="video">
          <video src={artifactUrl(views.video.path)} poster={views.video.poster ? artifactUrl(views.video.poster) : undefined}
            controls loop muted playsInline className="w-full rounded border border-slate-100" />
          {views.video.caption && <figcaption className="text-[10.5px] text-slate-500">{views.video.caption}</figcaption>}
        </figure>
      )}
      {views.scene && <div><div className="font-semibold text-[11.5px] mb-1">3-D view</div><Viewer3D scene={views.scene} /></div>}
      {(r.figures || []).map((f: any) => (
        <figure key={f.path}>
          <img src={artifactUrl(f.path)} alt={f.caption || f.name} className="w-full rounded border border-slate-100" data-testid="figure" />
          {f.caption && <figcaption className="text-[10.5px] text-slate-500">{f.caption}</figcaption>}
        </figure>
      ))}
      {(r.tables || []).map((t: any, i: number) => (
        <div key={i} className="overflow-x-auto">
          <div className="font-semibold text-[11.5px] mb-1">{t.title}</div>
          <table className="text-[11px] min-w-full">
            <thead><tr>{t.columns.map((c: string) => <th key={c} className="text-left pr-2 text-slate-500 font-medium">{c}</th>)}</tr></thead>
            <tbody>{t.rows.map((row: any[], j: number) => <tr key={j} className="border-t border-slate-100">{row.map((c, k) => <td key={k} className="pr-2 font-mono">{fmt(c)}</td>)}</tr>)}</tbody>
          </table>
        </div>
      ))}
      {views.equations && (
        <div className="text-[11px]">
          <div className="font-semibold mb-1">Equations ({views.equations.classification?.family})</div>
          {views.equations.pdes.map((e: any, i: number) => <div key={i} className="font-mono">{e.text}</div>)}
        </div>
      )}
      {result.log && (
        <details>
          <summary className="text-[11.5px] text-slate-500 cursor-pointer">Log</summary>
          <pre className="text-[10.5px] bg-slate-900 text-slate-100 p-2 rounded max-h-64 overflow-auto whitespace-pre-wrap">{result.log}</pre>
        </details>
      )}
    </div>
  );
}
