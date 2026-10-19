import { Fragment, useEffect, useMemo, useRef, useState } from "react";
import { api } from "../api";
import { PORT_COLORS, type NodeType, type PaletteInfo, type PaletteSection } from "../types";

// The palette shows what pinneapple_flow.palette describes: sections in workflow order, human titles, one-line
// summaries, search ranked by the library (curated nodes above the generated library calls).

const OPEN_KEY = "flow-studio.palette.open";

function readOpen(): Record<string, boolean> {
  try {
    return JSON.parse(localStorage.getItem(OPEN_KEY) || "{}");
  } catch {
    return {};
  }
}

function Highlight({ text, q }: { text: string; q: string }) {
  const words = q.toLowerCase().split(/[^a-z0-9]+/).filter(Boolean);
  if (!words.length) return <>{text}</>;
  const re = new RegExp(`(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")).join("|")})`, "ig");
  return <>{text.split(re).map((p, i) => (i % 2 ? <mark key={i} className="bg-amber-100 text-inherit rounded-sm">{p}</mark> : <Fragment key={i}>{p}</Fragment>))}</>;
}

type Hover = { n: NodeType; top: number; left: number } | null;

function HoverCard({ h }: { h: Hover }) {
  if (!h) return null;
  const { n } = h;
  const p = n.palette;
  return (
    <div className="fixed z-50 w-80 rounded-lg border border-slate-200 bg-white p-3 shadow-xl text-[12px] text-slate-700 pointer-events-none"
      style={{ top: Math.min(h.top, window.innerHeight - 260), left: h.left }} data-testid="palette-tooltip">
      <div className="font-semibold text-slate-900">{p?.title || n.title}</div>
      <div className="text-[10.5px] text-slate-400 mb-1">{n.type}{p?.package ? ` · pinneapple_${p.package}` : ""}</div>
      {!n.available && (
        <div className="mb-2 rounded bg-amber-50 border border-amber-200 px-2 py-1 text-amber-800">
          <div className="font-medium">Unavailable here</div>
          <div>{n.unavailable_reason}</div>
          {p?.install && <div className="mt-0.5 font-mono text-[11px]">{p.install}</div>}
        </div>
      )}
      <div className="whitespace-pre-line line-clamp-[8]">{n.description}</div>
      {n.calls && <div className="mt-1 text-[10.5px] text-slate-400 break-all">wraps {n.calls}</div>}
    </div>
  );
}

function Item({ n, q, star, onStar, onAdd, sectionLabel, onHover }: {
  n: NodeType; q: string; star: boolean; onStar: () => void; onAdd: () => void; sectionLabel?: string; onHover: (h: Hover) => void;
}) {
  const p = n.palette;
  const color = PORT_COLORS[n.outputs[0]?.type || "any"] || "#94a3b8";
  const title = p?.title || n.title;
  const tip = [
    title, "", n.description, "", `wraps: ${n.calls}`,
    n.available ? "" : `\nunavailable: ${n.unavailable_reason}${p?.install ? `\nfix: ${p.install}` : ""}`,
  ].join("\n").trim();
  return (
    <div
      draggable={n.available}
      onDragStart={(e) => {
        e.dataTransfer.setData("application/pinneapple-node", n.type);
        e.dataTransfer.effectAllowed = "move";
        // drag preview: a small node card in the type's colour
        const g = document.createElement("div");
        g.textContent = title;
        g.style.cssText = `position:absolute;top:-1000px;left:-1000px;padding:6px 10px;border-radius:8px;font:600 12px system-ui;`
          + `background:white;border:2px solid ${color};box-shadow:0 4px 12px rgba(0,0,0,.15);color:#0f172a`;
        document.body.appendChild(g);
        e.dataTransfer.setDragImage(g, 12, 12);
        setTimeout(() => g.remove(), 0);
      }}
      onDoubleClick={() => n.available && onAdd()}
      aria-label={tip}
      onMouseEnter={(e) => { const r = e.currentTarget.getBoundingClientRect(); onHover({ n, top: r.top, left: r.right + 8 }); }}
      onMouseLeave={() => onHover(null)}
      onDragStartCapture={() => onHover(null)}
      className={`group relative flex gap-2 px-2 py-1.5 mb-1 rounded-md border select-none ${n.available
        ? "border-slate-200 bg-white hover:bg-brand-50 hover:border-brand-500 cursor-grab active:cursor-grabbing"
        : "border-dashed border-slate-200 bg-slate-50 opacity-60 cursor-not-allowed"}`}
      data-testid={`palette-${n.type}`}
      data-available={n.available ? "yes" : "no"}
    >
      <span className="mt-1 h-2.5 w-2.5 shrink-0 rounded-sm" style={{ background: n.available ? color : "#cbd5e1" }} />
      <div className="min-w-0 flex-1">
        <div className="text-[12.5px] font-medium text-slate-800 leading-tight"><Highlight text={title} q={q} /></div>
        {p?.summary && <div className="text-[10.5px] text-slate-500 leading-snug line-clamp-2"><Highlight text={p.summary} q={q} /></div>}
        {!n.available && <div className="text-[10px] text-amber-700 mt-0.5" data-testid="palette-unavailable">{p?.install || n.unavailable_reason}</div>}
        {sectionLabel && <div className="text-[9.5px] uppercase tracking-wide text-slate-400 mt-0.5" data-testid="palette-hit-section">{sectionLabel}{p && !p.curated ? " · library call" : ""}</div>}
      </div>
      <button
        className={`shrink-0 self-start text-[13px] leading-none ${star ? "text-amber-500" : "text-slate-300 opacity-0 group-hover:opacity-100"}`}
        title={star ? "remove from favourites" : "add to favourites"}
        onClick={(e) => { e.stopPropagation(); onStar(); }}
        data-testid={`palette-star-${n.type}`}
      >{star ? "★" : "☆"}</button>
    </div>
  );
}

function Header({ id, title, glyph, count, open, onToggle }: { id: string; title: string; glyph?: string; count: number; open: boolean; onToggle: () => void }) {
  return (
    <button className="w-full flex items-center gap-1.5 py-1 text-left text-[10.5px] uppercase tracking-wider font-semibold text-slate-500 hover:text-slate-800"
      onClick={onToggle} data-testid={`palette-section-toggle-${id}`} aria-expanded={open}>
      <span className="w-3 text-slate-400">{open ? "▾" : "▸"}</span>
      {glyph && <span className="w-3 text-center normal-case text-slate-400">{glyph}</span>}
      <span className="flex-1">{title}</span>
      <span className="rounded-full bg-slate-100 px-1.5 text-[10px] text-slate-500 normal-case" data-testid={`palette-count-${id}`}>{count}</span>
    </button>
  );
}

export function Palette({ nodes, onAdd, onNewType, recentBump }: {
  nodes: NodeType[]; onAdd: (t: string) => void; onNewType?: () => void; recentBump?: string;
}) {
  const [info, setInfo] = useState<PaletteInfo | null>(null);
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<{ type: string; section: string }[] | null>(null);
  const [open, setOpen] = useState<Record<string, boolean>>(readOpen);
  const byType = useMemo(() => Object.fromEntries(nodes.map((n) => [n.type, n])), [nodes]);
  const seq = useRef(0);
  const [hover, setHover] = useState<Hover>(null);

  useEffect(() => { api.palette().then(setInfo).catch(() => setInfo(null)); }, [nodes]);
  useEffect(() => {
    if (!q.trim()) { setHits(null); return; }
    const k = ++seq.current;
    const t = setTimeout(() => api.paletteSearch(q).then((h) => k === seq.current && setHits(h)).catch(() => setHits([])), 120);
    return () => clearTimeout(t);
  }, [q]);
  // a node was added on the canvas: it goes to the front of "Recent"
  useEffect(() => {
    if (!recentBump || !info) return;
    const recent = [recentBump.split("#")[0], ...info.recent.filter((t) => t !== recentBump.split("#")[0])].slice(0, 12);
    setInfo({ ...info, recent });
    api.setPalettePrefs({ recent }).catch(() => undefined);
  }, [recentBump]);

  const sections: PaletteSection[] = info?.sections || [];
  const title = Object.fromEntries(sections.map((s) => [s.id, s.title]));
  const isOpen = (id: string, dflt: boolean) => open[id] ?? dflt;
  const toggle = (id: string, dflt: boolean) => {
    const next = { ...open, [id]: !isOpen(id, dflt) };
    setOpen(next);
    try { localStorage.setItem(OPEN_KEY, JSON.stringify(next)); } catch { /* storage unavailable */ }
  };
  const favs = new Set(info?.favourites || []);
  const star = (t: string) => {
    if (!info) return;
    const favourites = favs.has(t) ? info.favourites.filter((x) => x !== t) : [...info.favourites, t];
    setInfo({ ...info, favourites });
    api.setPalettePrefs({ favourites }).catch(() => undefined);
  };
  const item = (t: string, label?: string) => {
    const n = byType[t];
    return n ? <Item key={t} n={n} q={q} star={favs.has(t)} onStar={() => star(t)} onAdd={() => onAdd(t)} sectionLabel={label} onHover={setHover} /> : null;
  };
  const listOf = (ts: string[]) => ts.filter((t) => byType[t]);

  return (
    <aside onMouseLeave={() => setHover(null)} className="w-64 shrink-0 border-r border-slate-200 bg-white flex flex-col" data-testid="palette">
      <div className="p-2 border-b border-slate-100">
        <input
          className="w-full text-sm border border-slate-200 rounded px-2 py-1"
          placeholder="Search nodes (title, description, package)…" value={q} onChange={(e) => setQ(e.target.value)}
          data-testid="palette-search"
        />
      </div>
      <div className="overflow-y-auto flex-1 px-2 py-1">
        {hits ? (
          <div data-testid="palette-results">
            <div className="text-[10.5px] text-slate-400 py-1">{hits.length ? `${hits.length} match${hits.length > 1 ? "es" : ""}` : "no match"}</div>
            {hits.map((h) => item(h.type, title[h.section] || h.section))}
          </div>
        ) : !info ? (
          <div className="text-xs text-slate-400 p-2">loading…</div>
        ) : (
          <>
            {(["favourites", "recent"] as const).map((k) => {
              const ts = listOf(info[k]);
              if (!ts.length) return null;
              const id = k;
              return (
                <div key={k} data-testid={`palette-${k}`} className="mb-1">
                  <Header id={id} title={k === "favourites" ? "Favourites" : "Recent"} glyph={k === "favourites" ? "★" : "↺"} count={ts.length}
                    open={isOpen(id, true)} onToggle={() => toggle(id, true)} />
                  {isOpen(id, true) && ts.map((t) => item(t))}
                </div>
              );
            })}
            {sections.map((s) => {
              const ts = listOf(s.types);
              if (!ts.length) return null;
              const o = isOpen(s.id, !s.collapsed);
              const groups: Record<string, string[]> = {};
              if (s.id === "library") for (const t of ts) (groups[byType[t].palette?.group || "other"] ||= []).push(t);
              return (
                <div key={s.id} data-testid={`palette-section-${s.id}`} data-open={o ? "yes" : "no"} className="mb-1" title={s.description}>
                  <Header id={s.id} title={s.title} glyph={s.glyph} count={ts.length} open={o} onToggle={() => toggle(s.id, !s.collapsed)} />
                  {o && (s.id !== "library" ? ts.map((t) => item(t)) : Object.entries(groups).map(([g, gts]) => {
                    const gid = `library.${g}`;
                    const go = isOpen(gid, false);
                    return (
                      <div key={g} className="pl-2" data-testid={`palette-group-${g}`}>
                        <Header id={gid} title={`pinneapple_${g}`} count={gts.length} open={go} onToggle={() => toggle(gid, false)} />
                        {go && gts.map((t) => item(t))}
                      </div>
                    );
                  }))}
                </div>
              );
            })}
          </>
        )}
      </div>
      <HoverCard h={hover} />
      <div className="p-2 text-[10.5px] text-slate-400 border-t">
        Drag onto the canvas, or double-click to add at the centre. ☆ marks a favourite.
        {onNewType && <button className="block mt-1 text-sky-700" onClick={onNewType} data-testid="new-node-type">+ new node type from a library call</button>}
      </div>
    </aside>
  );
}
