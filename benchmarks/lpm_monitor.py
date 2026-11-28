"""The LPM campaign on one screen: which design space is running, at what stage, and what it has measured.

    python -m benchmarks.lpm_monitor --out ~/lpm-campaign              # serve http://127.0.0.1:8130/
    python -m benchmarks.lpm_monitor --out ~/lpm-campaign --state      # the same state as JSON, and exit
    python -m benchmarks.lpm_monitor --out ~/lpm-campaign --detach ~/lpm-campaign/panel.log   # survives the shell

The campaign runner writes one file per thing it is doing — ``dataset/manifest.json`` when the data is there,
``dataset/progress.json`` while it is being built, ``model/progress.json`` while the fit runs, ``model/card.json``
when the model is measured, and ``runner.json`` for which family is running right now. This module reads those and
nothing else: it never imports the families to ask how they are, so what the page shows is what a run that died left
behind, which is the only state worth trusting at 3 a.m.

That is why the progress files exist at all (:mod:`pinneapple_neural.lpm_pipeline` writes them when asked): a fit is
four hours of silence otherwise, and a screen reading ``card.json`` alone would show a family as *pending* for the
whole four hours.
"""
from __future__ import annotations

import argparse
import http.server
import json
import os
import socketserver
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from benchmarks import _daemon, lpm_figures

__all__ = ["CAPTIONS", "STAGES", "state", "page", "serve", "main"]

#: the five stages the pipeline runs, in order (see docs/guides/lpm_pipeline.md)
STAGES = ("draw", "build", "write", "fit", "card")

SECTIONS = {
    "dataset": ("draw", "build", "write"),
    "fit": ("fit",),
    "card": ("card",),
}


def _load(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _written(path: Path) -> str | None:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(timespec="seconds")
    except OSError:
        return None


def _age(path: Path) -> float | None:
    """Seconds since the file was last written: how a screen tells a live run from one that died."""
    try:
        return round(time.time() - path.stat().st_mtime, 1)
    except OSError:
        return None


def catalogue() -> dict[str, dict[str, Any]]:
    """What each design space is, from the families themselves; a name-only fallback if they cannot be imported.

    Importing the families pulls torch, so it happens once, here, and never again — the page reads files only.
    """
    global _CATALOGUE
    if _CATALOGUE is not None:
        return _CATALOGUE
    with _CATALOGUE_LOCK:                               # a warmed thread and a first request can race here
        if _CATALOGUE is not None:
            return _CATALOGUE
        from benchmarks.lpm_campaign import RECIPES

        try:
            from benchmarks.lpm_campaign import families_by_name

            fams = families_by_name()
            _CATALOGUE = {
                n: {"title": f.title, "domain": f.domain, "dim": f.dim, "fields": list(f.field_names),
                    "units": list(f.field_units), "n_points": int(f.n_points),
                    "reference": f.reference.implements, "reference_name": f.reference.name,
                    "tolerance": float(f.tolerance), "recipes": RECIPES.get(n, {})}
                for n, f in fams.items()
            }
        except Exception:                               # noqa: BLE001 - a screen must open with a broken install
            _CATALOGUE = {n: {"title": n, "domain": "", "dim": 0, "fields": [], "units": [], "n_points": 0,
                              "reference": "?", "reference_name": "", "tolerance": 0.05, "recipes": r}
                          for n, r in RECIPES.items()}
        return _CATALOGUE


_CATALOGUE: dict[str, dict[str, Any]] | None = None
_CATALOGUE_LOCK = threading.Lock()


def _stage_of(manifest: dict | None, card: dict | None, ds_progress: dict | None,
              fit_progress: dict | None) -> tuple[str, str]:
    """(stage, state): the stage the family is in and whether it is running, done or waiting."""
    if card is not None:
        return "card", "done"
    if manifest is not None and fit_progress is None:
        return "fit", "starting"                          # the data is there; the fit has not written yet
    if fit_progress is not None:
        return "fit", ("done" if fit_progress.get("finished") else "running")
    if ds_progress is not None:
        return "dataset", ("done" if ds_progress.get("finished") else "running")
    return "pending", "waiting"


def family_state(name: str, out: Path) -> dict[str, Any]:
    """Everything known about one design space from the files under ``out/name``."""
    root = out / name
    dataset, model = root / "dataset", root / "model"
    manifest_p, card_p = dataset / "manifest.json", model / "card.json"
    ds_p, fit_p = dataset / "progress.json", model / "progress.json"
    manifest, card = _load(manifest_p), _load(card_p)
    ds_progress, fit_progress = _load(ds_p), _load(fit_p)
    stage, state = _stage_of(manifest, card, ds_progress, fit_progress)

    # a family that has run describes itself in its own manifest; only one that has not is looked up in the families,
    # which is what pulls torch — so a campaign in progress reads files alone
    manifest_family = (manifest or {}).get("family") or {}
    meta = {} if manifest_family else catalogue().get(name, {})
    row: dict[str, Any] = {
        "name": name,
        "title": manifest_family.get("title") or meta.get("title") or name,
        "domain": manifest_family.get("domain") or meta.get("domain") or "",
        "dim": int(manifest_family.get("dim") or meta.get("dim") or 0),
        "fields": list(manifest_family.get("field_names") or meta.get("fields") or []),
        "units": list(manifest_family.get("field_units") or meta.get("units") or []),
        "n_points": int(manifest_family.get("n_points") or meta.get("n_points") or 0),
        "reference": ((manifest_family.get("reference") or {}).get("implements") or meta.get("reference") or "?"),
        "reference_name": (manifest_family.get("reference") or {}).get("name") or meta.get("reference_name") or "",
        "stage": stage,
        "state": state,
        "stages": {s: False for s in STAGES},
        "designs": {"done": None, "total": None},
        "epochs": {"done": None, "total": None},
        "progress": None,
        "eta_s": None,
        "elapsed_s": None,
        "updated_at": None,
        "stale": False,
        "held_out": None,
        "baseline": None,
        "ood": None,
        "within_tolerance": None,
        "written": _written(card_p) or _written(manifest_p),
    }
    if manifest is not None:
        row["stages"].update({s: True for s in SECTIONS["dataset"]})
        row["designs"] = {"done": manifest.get("n_designs"), "total": manifest.get("n_designs")}
    if card is not None:
        row["stages"].update({s: True for s in SECTIONS["fit"] + SECTIONS["card"]})
        held = card.get("held_out") or {}
        row["held_out"] = held.get("relative_l2")
        row["baseline"] = held.get("constant_field_baseline_relative_l2")
        row["within_tolerance"] = held.get("within_tolerance")
        ood = card.get("ood") or {}
        row["ood"] = f"{ood.get('probes_flagged')}/{ood.get('probes_total')}"
        row["epochs"] = {"done": (card.get("training") or {}).get("epochs"),
                         "total": (card.get("training") or {}).get("epochs")}
        row["fit_seconds"] = (card.get("training") or {}).get("seconds")
        row["written"] = _written(card_p)
    # the live one wins over the finished one: while a second --force run is going, the card belongs to the previous
    live = fit_progress if (fit_progress is not None and not fit_progress.get("finished")) else (
        ds_progress if (ds_progress is not None and not ds_progress.get("finished")) else None)
    source = live or fit_progress or ds_progress
    if source is not None:
        done, total = source.get("done") or 0, source.get("total") or 0
        if source.get("kind") == "dataset":
            row["designs"] = {"done": done, "total": total}
            # a bar is lit when its stage is *finished*: the designs were drawn before the first record was written,
            # and build/write interleave per design until the manifest says they are done
            row["stages"]["draw"] = True
        else:
            row["epochs"] = {"done": done, "total": total}
            row["loss"] = source.get("loss")
            row["loss_first"] = source.get("loss_first")
        row["progress"] = (done / total) if total else None
        row["eta_s"] = source.get("eta_s")
        row["elapsed_s"] = source.get("elapsed_s")
        row["updated_at"] = source.get("updated_at")
    if live is not None:
        # a progress file older than a couple of minutes on a running stage is a stale file, not a slow solver
        age = _age(fit_p if live.get("kind") == "fit" else ds_p)
        row["stale"] = bool(age is not None and age > 180)
        row["age_s"] = age
    return row


def _log_tail(path: Path | None, lines: int = 14) -> list[str]:
    if path is None or not path.is_file():
        return []
    try:
        with path.open(encoding="utf-8", errors="replace") as f:
            return [ln.rstrip("\n") for ln in f.readlines()[-lines:]]
    except OSError:
        return []


def _pid_alive(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (OSError, TypeError, ValueError):
        return False
    return True


def state(out: str | Path, log: str | Path | None = None, lines: int = 14) -> dict[str, Any]:
    """The whole campaign as the files describe it: one row per design space, the runner, the failures, the log."""
    out = Path(out).expanduser()
    log_path = Path(log).expanduser() if log else out / "campaign.log"
    runner = _load(out / "runner.json") or {}
    campaign = _load(out / "campaign.json") or {}
    failures = _load(out / "failures.json") or {}
    names = list(catalogue())
    order = runner.get("families") or campaign.get("order") or names
    rows = [family_state(n, out) for n in order if n in names]
    current = runner.get("current") or (campaign.get("current") if campaign else None)
    alive = _pid_alive(runner.get("pid")) if runner.get("pid") else False
    if campaign and not alive:
        run_state = "finished"
    elif runner.get("stopped"):
        run_state = "stopped"
    elif alive:
        run_state = "running"
    else:
        run_state = "idle"
    finished = sum(1 for r in rows if r["state"] == "done")

    def _first(key: str, default: Any = None) -> Any:
        """The runner's value when it has one, the campaign's when it does not — ``0`` and ``""`` are values."""
        for source in (runner, campaign):
            if source.get(key) is not None:
                return source[key]
        return default

    estimate = runner.get("estimate_s")
    if estimate is None and campaign.get("estimates"):
        estimate = sum(campaign["estimates"].values())
    return {
        "schema": "pinneapple.lpm.monitor/1",
        "as_of": datetime.now(UTC).isoformat(timespec="seconds"),
        "out": str(out),
        "campaign": {
            "state": run_state,
            "scale": _first("scale"),
            "stage": _first("stage"),
            "seed": _first("seed"),
            "pid": runner.get("pid"),
            "current": current,
            "current_stage": runner.get("current_stage"),
            "started_at": _first("started_at"),
            "elapsed_s": runner.get("elapsed_s"),
            "estimate_s": estimate,
            "max_hours": runner.get("max_hours"),
            "threads": runner.get("threads"),
            "one_at_a_time": runner.get("one_at_a_time", True),
            "failures": sorted(failures),
            "done": finished,
            "total": len(rows),
        },
        "families": rows,
        "failures": {k: v.get("error") for k, v in failures.items()},
        "log": _log_tail(log_path, lines),
        "log_path": str(log_path),
    }


# ------------------------------------------------------------------------------------------------ the page

_CSS = """
:root{--bg:#1e1e1e;--panel:#262626;--line:#3a3a3a;--ink:#e8e8e8;--dim:#9a9a9a;--accent:#0081be;--warn:#a23a00;
--ok:#2e7d32;--wait:#4a4a4a}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:18px 22px 12px;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap;gap:18px;align-items:baseline}
h1{font-size:16px;margin:0;letter-spacing:.02em;font-weight:600}
.badge{font-size:12px;padding:2px 8px;border-radius:10px;background:var(--wait);color:#fff}
.badge.running{background:var(--accent)}.badge.finished{background:var(--ok)}
.badge.stopped,.badge.idle{background:var(--wait)}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.dim{color:var(--dim)}
main{padding:18px 22px 40px}
.now{border:1px solid var(--line);border-left:3px solid var(--accent);background:var(--panel);padding:12px 16px;
border-radius:6px;margin-bottom:18px}
.now .big{font-size:15px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--dim);font-weight:500;border-bottom:1px solid var(--line);padding:6px 8px;font-size:12px}
td{padding:7px 8px;border-bottom:1px solid #2e2e2e;vertical-align:top}
tr.done td{color:var(--dim)}
tr td.bad{color:#ff8a65;font-weight:600}
.pill{display:inline-block;padding:1px 7px;border-radius:9px;font-size:11px;background:var(--wait);color:#fff}
.pill.running{background:var(--accent)}.pill.done{background:var(--ok)}
.pill.starting{background:#5a5a5a}.pill.stale{background:var(--warn)}
.bar{height:6px;background:#333;border-radius:3px;overflow:hidden;min-width:80px;margin-top:4px}
.bar>i{display:block;height:100%;background:var(--accent)}
.stages{display:flex;gap:3px;margin-top:5px}
.stages span{width:14px;height:4px;border-radius:2px;background:#3a3a3a}
.stages span.on{background:var(--accent)}
.fail{border:1px solid var(--warn);background:#2a1a12;padding:10px 14px;border-radius:6px;margin-bottom:14px}
tr.row{cursor:pointer}
tr.row:hover td{background:#2b2b2b}
tr.sel td{background:#22303a}
.detail{margin-top:20px;border-top:1px solid var(--line);padding-top:14px}
.detail h2{font-size:14px;margin:0 0 12px;font-weight:600}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:14px}
figure{margin:0;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:8px;min-width:0}
figure img{width:100%;height:auto;display:block;border-radius:3px}
figure.missing img{display:none}
figure figcaption{font-size:11px;color:var(--dim);padding:6px 2px 0}
figure.wide{grid-column:span 2}
figure.full{grid-column:1/-1}
figure iframe{width:100%;height:520px;border:0;border-radius:3px;background:#f4f4f4;display:block}
figure.missing iframe{display:none}
figure.missing figcaption::after{content:" — no figures for this family yet (fitted before the view was written)"}
pre{background:#141414;border:1px solid var(--line);padding:10px;border-radius:6px;overflow:auto;font-size:12px;
max-height:260px;margin:0}
footer{padding:0 22px 30px;color:var(--dim);font-size:12px}
"""

_JS = """
let LAST = null, OPEN = null;
async function tick(){
  try{
    const r = await fetch('state.json',{cache:'no-store'});
    LAST = await r.json();
    paint();
  }catch(e){
    document.getElementById('asof').textContent = 'state unreadable — ' + e;
  }
}
function paint(){
  const s = LAST;
  document.getElementById('root').innerHTML = render(s);
  document.getElementById('asof').textContent = 'as of ' + s.as_of;
  document.getElementById('out').textContent = s.out;
}
function toggle(name){ OPEN = (OPEN === name ? null : name); if(LAST) paint(); }
function hms(v){ if(v==null) return '—';
  if(v<90) return Math.round(v)+'s'; if(v<5400) return Math.round(v/60)+'m'; return (v/3600).toFixed(1)+'h'; }
function esc(t){ return String(t==null?'':t).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function pill(r){
  const cls = r.stale ? 'stale' : r.state;
  const label = r.stale ? r.stage+' (stale?)' : (r.state==='done' ? 'done' : (r.state==='running' ? r.stage+'…' : r.state));
  return '<span class="pill '+cls+'">'+esc(label)+'</span>';
}
function stages(r){
  return '<div class="stages">'+['draw','build','write','fit','card']
    .map(s=>'<span class="'+(r.stages[s]?'on':'')+'" title="'+s+'"></span>').join('')+'</div>';
}
function bar(r){
  if(r.progress==null) return '';
  const pct = Math.max(0,Math.min(1,r.progress))*100;
  return '<div class="bar"><i style="width:'+pct.toFixed(1)+'%"></i></div>';
}
function row(r){
  const c = (r.state==='done' ? 'done ' : '') + (OPEN===r.name ? 'sel ' : '') + 'row';
  const ds = (r.designs.done!=null) ? r.designs.done+'/'+r.designs.total+' designs' : '';
  const ep = (r.epochs.done!=null) ? r.epochs.done+'/'+r.epochs.total+' ep' : '';
  const n = [ds, ep].filter(Boolean).join(' · ');
  const loss = (r.loss!=null) ? 'loss '+Number(r.loss).toPrecision(3) : '';
  const held = (r.held_out!=null) ? Number(r.held_out).toFixed(3) : '—';
  const base = (r.baseline!=null) ? Number(r.baseline).toFixed(3) : '—';
  // a model at or above the constant-field baseline has not learnt the design space: it ran, and it did not converge
  const bad = (r.held_out!=null && r.baseline!=null && r.held_out>=r.baseline);
  return '<tr class="'+c+'" onclick="toggle(\\''+esc(r.name)+'\\')"><td><b>'+esc(r.name)+'</b><div class="dim">'+esc(r.title)+'</div></td>'+
    '<td>'+r.dim+'D<div class="dim">'+esc(r.fields.join(', '))+'</div></td>'+
    '<td>'+esc(r.reference)+'<div class="dim">'+esc(r.reference_name)+'</div></td>'+
    '<td>'+pill(r)+stages(r)+'</td>'+
    '<td class="mono">'+esc(n)+(loss?'<div class="dim">'+esc(loss)+'</div>':'')+bar(r)+'</td>'+
    '<td class="mono">'+hms(r.eta_s)+'</td>'+
    '<td class="mono'+(bad?' bad':'')+'" title="'+(bad?'at or above the constant-field baseline: did not learn the space':'')+
      '">'+held+' <span class="dim">/ '+base+'</span></td>'+
    '<td class="mono dim">'+esc((r.written||'').slice(0,16).replace('T',' '))+'</td></tr>';
}
const CAPTIONS = __CAPTIONS__;
function detail(s){
  if(!OPEN) return '';
  const r = (s.families||[]).find(f=>f.name===OPEN);
  if(!r) return '';
  const head = '<h2>'+esc(r.name)+' <span class="dim">'+esc(r.title)+'</span></h2>';
  if(r.state!=='done')
    return '<section class="detail">'+head+'<p class="dim">The figures are drawn when the card exists — '
      + esc(r.stage)+' now. The row above already says how far it got.</p></section>';
  const t = encodeURIComponent(r.written||'');
  const fam = encodeURIComponent(r.name);
  const viewer = '<figure class="full"><iframe loading="lazy" src="figure/'+fam+'/viewer.html?t='+t+'" '
    + 'title="Studio 3-D viewer" onerror="this.parentNode.classList.add(\\'missing\\')"></iframe>'
    + '<figcaption>'+esc(CAPTIONS._viewer)+'</figcaption></figure>';
  const figs = Object.keys(CAPTIONS).filter(k=>k.charAt(0)!=='_').map(k =>
    '<figure class="'+(k==='geometry'||k==='prediction'?'wide':'')+'"><img loading="lazy" src="figure/'+fam+'/'+k+'.png?t='+t+'" alt="'+k+'" '
    + 'onerror="this.parentNode.classList.add(\\'missing\\')"><figcaption>'+esc(CAPTIONS[k])+'</figcaption></figure>'
  ).join('');
  return '<section class="detail">'+head+'<div class="figs">'+viewer+figs+'</div></section>';
}
function render(s){
  const c = s.campaign;
  const now = c.current ? ('<div class="now"><div class="big"><b>'+esc(c.current)+'</b> — '+esc(c.current_stage||'?')+
    (c.state==='running'?' <span class="pill running">running</span>':'')+'</div>'+
    '<div class="dim">'+esc(c.scale||'')+' scale, stage '+esc(c.stage||'')+', one design space at a time — '+
    (c.done||0)+' of '+(c.total||0)+' fitted'+(c.estimate_s?' · about '+hms(c.estimate_s)+' of work in the plan':'')+
    (c.max_hours?' · stops after '+c.max_hours+' h':'')+(c.threads?' · '+(c.threads||0)+' threads':'')+'</div></div>')
    : ('<div class="now"><div class="big">No runner reporting'+(c.state==='finished'?' — campaign finished':'')+
       '</div><div class="dim">runner.json is written by benchmarks.lpm_campaign while it runs</div></div>');
  const fails = Object.keys(s.failures||{}).length
    ? '<div class="fail"><b>Failed:</b> '+Object.entries(s.failures).map(([k,v])=>esc(k)+' — '+esc(v)).join('<br>')+'</div>' : '';
  return now+fails+
    '<table><thead><tr><th>design space</th><th>dim / predicts</th><th>reference</th><th>stage</th>'+
    '<th>progress</th><th>eta</th><th>held-out / baseline</th><th>written</th></tr></thead><tbody>'+
    (s.families||[]).map(row).join('')+'</tbody></table>'+detail(s);
}
tick(); setInterval(tick, 2000);
"""

_PAGE = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LPM campaign</title><style>__CSS__</style></head><body>
<header><h1>Large Physics Model — campaign</h1>
<span id="asof" class="dim mono">…</span><span class="dim" id="out"></span></header>
<main><div id="root">loading…</div></main>
<footer>draw → build → write → fit → card, per design space. Held-out is the relative L2 on bodies held out by
geometry; a model at or above its baseline has not learnt the space. The five bars are the stages.</footer>
<script>__JS__</script></body></html>"""


#: what each figure is, in the panel's words. Kept here and injected as JSON, so a caption can hold an apostrophe
#: without breaking the page it is written into
CAPTIONS = {
    "loss": "training loss, log scale — did it converge or stop early",
    "design": "the design space on its two widest parameters; a second colour is a body held out by geometry",
    "geometry": "the bodies behind the samples, as the model sees them (the point cloud)",
    "prediction": "reference (solid) against the model (dashed), one row per field, one column per body",
    "_viewer": ("the Studio 3-D viewer — drag to orbit, scroll to zoom, switch the field on the top bar; "
                "the samples stand side by side, each coloured by the model's prediction"),
}


def page() -> str:
    """The panel's HTML, self-contained: it polls ``state.json`` next to it and needs nothing external."""
    return (_PAGE.replace("__CSS__", _CSS).replace("__JS__", _JS)
            .replace("__CAPTIONS__", json.dumps(CAPTIONS, ensure_ascii=True).replace("</", "<\\/")))


def serve(out: str | Path, port: int = 8130, host: str = "127.0.0.1", log: str | Path | None = None,
          block: bool = True) -> socketserver.ThreadingTCPServer:
    """Serve the panel: ``/`` the page, ``/state.json`` the state read from the campaign's files at every request."""
    body = page().encode()
    # the families' own descriptions come from importing them, which pulls torch: warm that in the background so the
    # first request for state.json is not the one paying for it in front of a blank page
    threading.Thread(target=catalogue, daemon=True).start()

    class _H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):                                # noqa: N802 - http.server naming
            path = self.path.split("?")[0]
            try:
                if path in ("/", "/index.html"):
                    payload, ctype = body, "text/html; charset=utf-8"
                elif path == "/state.json":
                    payload, ctype = json.dumps(state(out, log), default=float).encode(), "application/json"
                elif path.startswith("/figure/"):
                    family, _, name = path[len("/figure/"):].partition("/")
                    if not family or not name:
                        self.send_error(404)
                        return
                    stem, _, suffix = name.rpartition(".")
                    target = lpm_figures.ensure(out, family, stem or name)
                    payload = target.read_bytes()
                    ctype = "text/html; charset=utf-8" if target.suffix == ".html" else "image/png"
                else:
                    self.send_error(404)
                    return
            except Exception as exc:                      # noqa: BLE001 - a panel that crashes is worse than a blank one
                # an <img> that 404s is drawn as missing; a 200 carrying an error would be a broken picture instead
                status = 404 if path.startswith("/figure/") else 500
                payload = json.dumps({"error": f"{type(exc).__name__}: {exc}"}).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *a):
            pass

    socketserver.ThreadingTCPServer.allow_reuse_address = True
    srv = socketserver.ThreadingTCPServer((host, port), _H)
    srv.daemon_threads = True
    if not block:
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        return srv
    print(f"LPM campaign panel: http://{host}:{port}/   (out: {Path(out).expanduser()})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        srv.shutdown()
    return srv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Follow an LPM campaign: one screen, every design space.")
    parser.add_argument("--out", type=Path, default=Path.home() / "lpm-campaign",
                        help="the campaign's output directory (default: ~/lpm-campaign)")
    parser.add_argument("--port", type=int, default=8130)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--log", type=Path, help="the runner's log (default: <out>/campaign.log)")
    parser.add_argument("--state", action="store_true", help="print the state as JSON and exit")
    parser.add_argument("--detach", type=Path, metavar="LOG",
                        help="serve in its own session, writing to LOG, and return immediately")
    parser.add_argument("--open", action="store_true", help="open the panel in the browser")
    a = parser.parse_args(argv)
    if a.state:
        print(json.dumps(state(a.out, a.log), indent=2, default=float))
        return 0
    if a.detach:
        _daemon.detach("benchmarks.lpm_monitor", a.detach, marker="LPM_MONITOR_DETACHED")
    if a.open:
        import webbrowser
        threading.Timer(0.5, lambda: webbrowser.open(f"http://{a.host}:{a.port}/")).start()
    serve(a.out, port=a.port, host=a.host, log=a.log)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
