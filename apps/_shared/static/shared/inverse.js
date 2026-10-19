// UI pieces shared by the identifiability, calibration and inverse-studio apps (no dependencies).
//   InvUI.api(path, opts)              fetch JSON; throws Error(detail) on a non-2xx answer
//   InvUI.pollJob(id, onTick)          polls /api/jobs/{id} (shared runtime) until done / failed / cancelled
//   InvUI.xy(el, spec)                 SVG plot on a real x axis: lines, points with error bars, bands
//   InvUI.heat(el, matrix, labels)     correlation matrix, -1 (blue) .. +1 (red)
//   InvUI.ProblemEditor(el, opts)      model + parameters + measurements (+ held-out measurements) editor,
//                                      tables uploaded through POST /api/table and mapped to columns
(function () {
  const css = `
  .inv-xy svg { display: block; width: 100%; overflow: visible; font: 11px system-ui, sans-serif; }
  .inv-heat svg { display: block; width: auto; max-width: 100%; height: auto; overflow: visible; font: 11px system-ui, sans-serif; }
  .inv-xy text, .inv-heat text { fill: #5d6b82; } .inv-xy .ax { stroke: #c9d1dd; } .inv-xy .grid { stroke: #eef1f5; }
  .inv-xy .yl { font-weight: 600; }
  .inv-legend { display: flex; flex-wrap: wrap; gap: 4px 14px; font-size: 12px; color: #5d6b82; margin-top: 4px; }
  .inv-legend i { display: inline-block; width: 14px; height: 3px; border-radius: 2px; margin-right: 5px; vertical-align: 3px; }
  .inv-legend i.pt { width: 8px; height: 8px; border-radius: 50%; vertical-align: 0; }
  .pe table { width: 100%; border-collapse: collapse; font-size: 12.5px; margin: 4px 0; }
  .pe th { text-align: left; font-weight: 600; color: var(--muted); font-size: 11.5px; padding: 3px 3px; }
  .pe td { padding: 2px 3px; } .pe td input, .pe td select { margin: 0; padding: 4px 5px; font-size: 12.5px; }
  .pe td input[type=checkbox] { width: auto; display: inline; }
  .pe td.x button { border: 0; background: none; cursor: pointer; color: var(--muted); font-size: 15px; }
  .pe .tw { max-height: 260px; overflow: auto; border: 1px solid var(--line); border-radius: 7px; }
  .pe .tools { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; align-items: center; }
  .pe .tools button, .pe .tools label.btn { padding: 4px 10px; font-size: 12.5px; }
  .pe label.btn { display: inline-block; margin: 0; border: 1px solid var(--line); border-radius: 7px; cursor: pointer;
    color: var(--ink); background: #fff; }
  .pe label.btn input { display: none; }
  .pe .map { border: 1px dashed var(--line); border-radius: 8px; padding: 8px; margin-top: 6px; background: #fafbfd; }
  .pe textarea { width: 100%; font: 12.5px var(--mono); border: 1px solid var(--line); border-radius: 7px; padding: 6px; }
  .pe .opt { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; }`;
  const st = document.createElement("style"); st.textContent = css; document.head.appendChild(st);

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  function fmt(v, d = 4) {
    if (v === null || v === undefined || (typeof v === "number" && !isFinite(v))) return "—";
    if (typeof v !== "number") return esc(v);
    const a = Math.abs(v);
    if (a !== 0 && (a < 1e-3 || a >= 1e6)) return v.toExponential(2);
    return String(+v.toPrecision(d));
  }

  async function api(path, opts = {}) {
    const o = { ...opts };
    if (o.json !== undefined) { o.method = o.method || "POST"; o.headers = { "Content-Type": "application/json" }; o.body = JSON.stringify(o.json); delete o.json; }
    const r = await fetch(path, o);
    const t = await r.text();
    let j; try { j = t ? JSON.parse(t) : null; } catch (e) { j = { detail: t }; }
    if (!r.ok) {
      let d = j && (j.detail ?? j.message ?? j.error);
      if (Array.isArray(d)) d = d.map((x) => `${(x.loc || []).slice(1).join(".")}: ${x.msg}`).join("; ");
      if (d && typeof d === "object") d = d.message || JSON.stringify(d);
      throw new Error(d || `HTTP ${r.status}`);
    }
    return j;
  }

  async function pollJob(id, onTick) {
    for (;;) {
      const j = await api(`/api/jobs/${id}`);
      onTick && onTick(j);
      if (["done", "failed", "cancelled"].includes(j.status)) return j;
      await new Promise((r) => setTimeout(r, 600));
    }
  }

  function nice(lo, hi, n) {
    if (!(hi > lo)) { hi = lo + 1; lo = lo - 1; }
    const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= raw) || raw;
    const out = [];
    for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }

  // spec: {height, xLabel, yLabel, xLog, yLog, series: [{name, x, y, color, mode: "line"|"points", err, dash, lo, hi}],
  //        vlines: [{x, label, color}], hlines: [{y, label, color}]}
  function xy(el, spec) {
    const W = Math.max(280, el.clientWidth || 560), H = spec.height || 240, L = 58, R = 14, T = 10, B = 34;
    const fx = spec.xLog ? (v) => Math.log10(v) : (v) => v, fy = spec.yLog ? (v) => Math.log10(v) : (v) => v;
    const xs = [], ys = [];
    for (const s of spec.series) {
      s.x.forEach((x, i) => {
        if (x === null || !isFinite(x) || (spec.xLog && x <= 0)) return;
        xs.push(fx(x));
        for (const v of [s.y && s.y[i], s.lo && s.lo[i], s.hi && s.hi[i]]) if (v !== null && v !== undefined && isFinite(v) && (!spec.yLog || v > 0)) ys.push(fy(v));
        if (s.err && s.y && isFinite(s.y[i])) { ys.push(fy(s.y[i] + s.err[i])); ys.push(fy(Math.max(s.y[i] - s.err[i], spec.yLog ? 1e-300 : -Infinity))); }
      });
    }
    for (const h of spec.hlines || []) ys.push(fy(h.y));
    if (!xs.length) { el.innerHTML = `<div class="hint">no data</div>`; return; }
    let x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = spec.yMin ?? Math.min(...ys), y1 = spec.yMax ?? Math.max(...ys);
    if (x1 === x0) { x0 -= 1; x1 += 1; }
    const py = (y1 - y0) * 0.07 || Math.abs(y0) * 0.1 || 1; y0 -= py; y1 += py;
    const px = (x1 - x0) * 0.03; x0 -= px; x1 += px;
    const X = (v) => L + ((fx(v) - x0) / (x1 - x0)) * (W - L - R), Y = (v) => T + (1 - (fy(v) - y0) / (y1 - y0)) * (H - T - B);
    let g = "";
    const tl = (lo, hi, log) => log ? nice(Math.floor(lo), Math.ceil(hi), Math.max(1, Math.min(6, Math.ceil(hi) - Math.floor(lo)))).map((e) => Math.pow(10, e)) : nice(lo, hi, 5);
    for (const v of tl(y0, y1, spec.yLog)) { const y = Y(v); if (y < T - 1 || y > H - B + 1) continue; g += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y}" y2="${y}"/><text x="${L - 6}" y="${y + 3.5}" text-anchor="end">${fmt(v, 3)}</text>`; }
    for (const v of tl(x0, x1, spec.xLog)) { const x = X(v); if (x < L - 1 || x > W - R + 1) continue; g += `<text x="${x}" y="${H - B + 14}" text-anchor="middle">${fmt(v, 3)}</text>`; }
    for (const h of spec.hlines || []) { g += `<line x1="${L}" x2="${W - R}" y1="${Y(h.y)}" y2="${Y(h.y)}" stroke="${h.color || "#5d6b82"}" stroke-dasharray="4 3"/>${h.label ? `<text x="${W - R}" y="${Y(h.y) - 4}" text-anchor="end" style="fill:${h.color || "#5d6b82"}">${esc(h.label)}</text>` : ""}`; }
    for (const v of spec.vlines || []) { g += `<line y1="${T}" y2="${H - B}" x1="${X(v.x)}" x2="${X(v.x)}" stroke="${v.color || "#5d6b82"}" stroke-dasharray="4 3"/>${v.label ? `<text x="${X(v.x) + 4}" y="${H - B - 6}" style="fill:${v.color || "#5d6b82"}">${esc(v.label)}</text>` : ""}`; }
    for (const s of spec.series) {
      const col = s.color || "#0b6bcb";
      const idx = s.x.map((_, i) => i).filter((i) => s.x[i] !== null && isFinite(s.x[i]));
      if (s.lo && s.hi) {
        const ok = idx.filter((i) => isFinite(s.lo[i]) && isFinite(s.hi[i]));
        if (ok.length) g += `<polygon points="${ok.map((i) => `${X(s.x[i])},${Y(s.hi[i])}`).join(" ")} ${ok.slice().reverse().map((i) => `${X(s.x[i])},${Y(s.lo[i])}`).join(" ")}" fill="${col}" fill-opacity=".15" stroke="none"/>`;
      }
      if (!s.y) continue;
      if (s.mode === "points") {
        for (const i of idx) {
          if (!isFinite(s.y[i])) continue;
          if (s.err) g += `<line x1="${X(s.x[i])}" x2="${X(s.x[i])}" y1="${Y(s.y[i] - s.err[i])}" y2="${Y(s.y[i] + s.err[i])}" stroke="${col}" stroke-width="1.2"/>`;
          g += `<circle cx="${X(s.x[i])}" cy="${Y(s.y[i])}" r="${s.r || 3.8}" fill="${s.fill || "#fff"}" stroke="${col}" stroke-width="1.6"><title>${esc(s.name || "")} ${fmt(s.x[i])}: ${fmt(s.y[i])}</title></circle>`;
        }
      } else {
        const ord = idx.slice().sort((a, b) => s.x[a] - s.x[b]).filter((i) => isFinite(s.y[i]) && (!spec.yLog || s.y[i] > 0));
        g += `<path d="${ord.map((i, k) => `${k ? "L" : "M"}${X(s.x[i]).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join("")}" fill="none" stroke="${col}" stroke-width="${s.width || 1.8}" ${s.dash ? `stroke-dasharray="${s.dash}"` : ""}/>`;
      }
    }
    g += `<line class="ax" x1="${L}" x2="${W - R}" y1="${H - B}" y2="${H - B}"/><line class="ax" x1="${L}" x2="${L}" y1="${T}" y2="${H - B}"/>`;
    if (spec.xLabel) g += `<text x="${L + (W - L - R) / 2}" y="${H - 4}" text-anchor="middle">${esc(spec.xLabel)}</text>`;
    if (spec.yLabel) g += `<text class="yl" x="12" y="${T + (H - T - B) / 2}" transform="rotate(-90 12 ${T + (H - T - B) / 2})" text-anchor="middle">${esc(spec.yLabel)}</text>`;
    el.classList.add("inv-xy");
    const named = spec.series.filter((s) => s.name);
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" height="${H}" role="img" aria-label="${esc(spec.yLabel || "plot")}">${g}</svg>` +
      (named.length > 1 ? `<div class="inv-legend">${named.map((s) => `<span><i class="${s.mode === "points" ? "pt" : ""}" style="background:${s.color || "#0b6bcb"}"></i>${esc(s.name)}</span>`).join("")}</div>` : "");
  }

  function heat(el, m, labels) {
    const n = labels.length, c = 46, L = 70, T = 8, W = L + n * c + 8, H = T + n * c + 52;
    const col = (v) => { if (v === null || !isFinite(v)) return "#eef1f5"; const a = Math.min(1, Math.abs(v)); return v >= 0 ? `rgba(180,35,24,${0.08 + 0.85 * a})` : `rgba(11,107,203,${0.08 + 0.85 * a})`; };
    let g = "";
    for (let i = 0; i < n; i++) {
      g += `<text x="${L - 6}" y="${T + i * c + c / 2 + 4}" text-anchor="end">${esc(labels[i])}</text>`;
      g += `<text x="${L + i * c + c / 2}" y="${T + n * c + 14}" text-anchor="middle">${esc(labels[i])}</text>`;
      for (let j = 0; j < n; j++) {
        const v = m[i][j];
        g += `<rect x="${L + j * c}" y="${T + i * c}" width="${c - 2}" height="${c - 2}" rx="4" fill="${col(v)}"><title>${esc(labels[i])} / ${esc(labels[j])}: ${fmt(v, 3)}</title></rect>`;
        g += `<text x="${L + j * c + c / 2 - 1}" y="${T + i * c + c / 2 + 3}" text-anchor="middle" style="fill:${Math.abs(v) > 0.6 ? "#fff" : "#172033"}">${v === null ? "—" : Math.abs(v) > 0.99 && Math.abs(v) < 1 ? (+v).toFixed(3) : (+v).toFixed(2)}</text>`;
      }
    }
    g += `<text x="${L}" y="${H - 8}">−1</text><rect x="${L + 18}" y="${H - 18}" width="60" height="10" fill="url(#ivg)"/><text x="${L + 84}" y="${H - 8}">+1</text>`;
    el.classList.add("inv-heat");
    el.innerHTML = `<svg viewBox="0 0 ${W} ${H}" width="${W}" style="max-width:100%"><defs><linearGradient id="ivg"><stop offset="0" stop-color="#0b6bcb"/><stop offset=".5" stop-color="#eef1f5"/><stop offset="1" stop-color="#b42318"/></linearGradient></defs>${g}</svg>`;
  }

  // ------------------------------------------------------------------ problem editor
  // opts: {meta (from /api/meta: models), values: "optional" | "required", validation: bool, title}
  function ProblemEditor(root, opts) {
    const st = { model: { type: "pin_fin", options: {} }, parameters: [], obs: [], val: [], inputs: ["x_mm"], outputs: ["T"] };
    const MODELS = opts.meta.models;
    const LABEL = { pin_fin: "Pin fin (analytic)", rod_fv: "Fin with h(x) zones (finite volumes, adjoint)", plate_2d: "Heat-spreader plate (2D finite volumes)",
      lumped_cooling: "Cooling curve (lumped capacitance)", arrhenius: "Reaction kinetics (Arrhenius)", expression: "My formula", table: "My simulated results (table)" };
    const OPTS = { pin_fin: [["d_mm", "Diameter (mm)"], ["length_mm", "Length (mm)"]], rod_fv: [["d_mm", "Diameter (mm)"], ["length_mm", "Length (mm)"], ["n_zones", "h zones"], ["n_cells", "Cells"]],
      plate_2d: [["nx", "Cells in x"], ["ny", "Cells in y"]], lumped_cooling: [["area_m2", "Area (m²)"]], arrhenius: [] };
    root.classList.add("pe");
    root.innerHTML = `
      <fieldset><legend>Model</legend>
        <label>Physical model <select data-k="mtype">${Object.keys(LABEL).filter((k) => MODELS[k] || k === "expression" || k === "table").map((k) => `<option value="${k}">${esc(LABEL[k])}</option>`).join("")}</select></label>
        <div class="hint" data-k="mdesc"></div>
        <div class="opt" data-k="mopts"></div>
        <div data-k="expr" hidden>
          <label>Formula per observable <span class="u">one per line: <code>T = T_air + (T0 - T_air)*exp(-h*A*t_s/C)</code></span>
            <textarea rows="3" data-k="exprs"></textarea></label>
          <label>Input columns <span class="u">comma-separated, e.g. t_s</span><input data-k="exprin"></label>
        </div>
        <div data-k="tab" hidden>
          <div class="tools"><label class="btn">Upload simulated runs<input type="file" data-k="tabfile" accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet,.json"></label><span class="hint" data-k="tabinfo"></span></div>
          <div data-k="tabmap"></div>
        </div>
      </fieldset>
      <fieldset><legend>Parameters</legend>
        <div class="tw"><table><thead><tr><th>Name</th><th>Start / nominal</th><th>Lower</th><th>Upper</th><th>Unit</th><th title="analyse in log scale">log</th><th title="known, not estimated">fixed</th><th></th></tr></thead><tbody data-k="ptab"></tbody></table></div>
        <div class="tools"><button type="button" class="ghost" data-k="padd">+ parameter</button>
          <label class="btn">From a ParameterSpace (JSON)<input type="file" data-k="pspace" accept=".json"></label></div>
      </fieldset>
      <fieldset><legend>${esc(opts.title || "Measurements")}</legend>
        <div data-k="obsed"></div>
      </fieldset>
      ${opts.validation ? `<fieldset><legend>Held-out measurements (never fitted)</legend><div data-k="valed"></div></fieldset>` : ""}`;
    const q = (k) => root.querySelector(`[data-k="${k}"]`);

    function obsEditor(host, key, label) {
      host.innerHTML = `<div class="tw"><table><thead data-k="${key}h"></thead><tbody data-k="${key}b"></tbody></table></div>
        <div class="tools"><button type="button" class="ghost" data-k="${key}add">+ row</button>
          <label class="btn">Upload ${label} (CSV, Excel, Parquet, JSON)<input type="file" data-k="${key}file" accept=".csv,.tsv,.txt,.xlsx,.xls,.parquet,.json"></label>
          <button type="button" class="ghost" data-k="${key}clear">Clear</button><span class="hint" data-k="${key}n"></span></div>
        <div data-k="${key}map"></div>`;
      host.querySelector(`[data-k="${key}add"]`).onclick = () => { st[key].push(blankRow()); renderRows(key); };
      host.querySelector(`[data-k="${key}clear"]`).onclick = () => { st[key] = []; renderRows(key); };
      host.querySelector(`[data-k="${key}file"]`).onchange = (e) => uploadTable(e.target.files[0], key, host.querySelector(`[data-k="${key}map"]`));
    }
    function blankRow() { const r = { observable: st.outputs[0], sigma: 0.5, value: "" }; st.inputs.forEach((k) => (r[k] = "")); return r; }
    function renderRows(key) {
      const host = root.querySelector(`[data-k="${key}b"]`), head = root.querySelector(`[data-k="${key}h"]`);
      if (!host) return;
      const need = opts.values === "required" || key === "val";
      head.innerHTML = `<tr>${st.inputs.map((k) => `<th>${esc(k)}</th>`).join("")}${st.outputs.length > 1 ? "<th>Observable</th>" : ""}<th>Value${need ? "" : ' <span class="u">(optional)</span>'}</th><th>σ</th><th></th></tr>`;
      host.innerHTML = st[key].map((r, i) => `<tr data-i="${i}">${st.inputs.map((k) => `<td><input data-f="${esc(k)}" value="${esc(r[k])}" inputmode="decimal"></td>`).join("")}${st.outputs.length > 1 ? `<td><select data-f="observable">${st.outputs.map((o) => `<option ${o === r.observable ? "selected" : ""}>${esc(o)}</option>`).join("")}</select></td>` : ""}<td><input data-f="value" value="${esc(r.value)}" inputmode="decimal"></td><td><input data-f="sigma" value="${esc(r.sigma)}" inputmode="decimal"></td><td class="x"><button type="button" title="remove">×</button></td></tr>`).join("");
      host.querySelectorAll("tr").forEach((tr) => {
        const i = +tr.dataset.i;
        tr.querySelectorAll("[data-f]").forEach((inp) => inp.addEventListener("change", () => { st[key][i][inp.dataset.f] = inp.value; }));
        tr.querySelector("button").onclick = () => { st[key].splice(i, 1); renderRows(key); };
      });
      const n = root.querySelector(`[data-k="${key}n"]`); if (n) n.textContent = `${st[key].length} rows`;
    }
    async function uploadTable(file, key, mapHost) {
      if (!file) return;
      const fd = new FormData(); fd.append("file", file);
      mapHost.innerHTML = `<div class="hint">reading ${esc(file.name)}…</div>`;
      try {
        const t = await api("/api/table", { method: "POST", body: fd });
        const opt = (sel) => `<option value="">—</option>` + t.columns.map((c) => `<option ${c === sel ? "selected" : ""}>${esc(c)}</option>`).join("");
        const guess = (k) => t.columns.find((c) => c.toLowerCase() === k.toLowerCase()) || t.columns.find((c) => c.toLowerCase().startsWith(k.toLowerCase().split("_")[0])) || "";
        mapHost.innerHTML = `<div class="map"><div class="hint">${esc(file.name)}: ${t.n_rows} rows (${esc(t.format)}). Map the columns:</div>
          <div class="opt">${st.inputs.map((k) => `<label>${esc(k)} <select data-m="${esc(k)}">${opt(guess(k))}</select></label>`).join("")}
          <label>Value <select data-m="value">${opt(t.columns.find((c) => !st.inputs.includes(c) && /temp|^t$|value|reading|meas|x$|conv/i.test(c)) || "")}</select></label>
          <label>σ column or number <input data-m="sigma" value="0.5"></label>
          ${st.outputs.length > 1 ? `<label>Observable column or name <input data-m="observable" value="${esc(st.outputs[0])}"></label>` : ""}</div>
          <div class="tools"><button type="button" class="primary">Use these rows</button></div></div>`;
        mapHost.querySelector("button").onclick = () => {
          const m = {}; mapHost.querySelectorAll("[data-m]").forEach((s) => (m[s.dataset.m] = s.value));
          const n = t.n_rows, rows = [];
          for (let i = 0; i < n; i++) {
            const r = { observable: st.outputs[0], value: m.value ? t.rows[m.value][i] : "", sigma: 0.5 };
            st.inputs.forEach((k) => (r[k] = m[k] ? t.rows[m[k]][i] : ""));
            r.sigma = t.columns.includes(m.sigma) ? t.rows[m.sigma][i] : m.sigma;
            if (m.observable !== undefined) r.observable = t.columns.includes(m.observable) ? t.rows[m.observable][i] : m.observable;
            rows.push(r);
          }
          st[key] = rows; renderRows(key); mapHost.innerHTML = "";
        };
      } catch (e) { mapHost.innerHTML = `<div class="err">${esc(e.message)}</div>`; }
    }
    function renderParams() {
      q("ptab").innerHTML = st.parameters.map((p, i) => `<tr data-i="${i}"><td><input data-f="name" value="${esc(p.name)}"></td><td><input data-f="value" value="${esc(p.value)}"></td><td><input data-f="lower" value="${esc(p.lower ?? "")}"></td><td><input data-f="upper" value="${esc(p.upper ?? "")}"></td><td><input data-f="unit" value="${esc(p.unit || "")}" style="width:70px"></td><td><input type="checkbox" data-f="log" ${p.log ? "checked" : ""}></td><td><input type="checkbox" data-f="fixed" ${p.fixed ? "checked" : ""}></td><td class="x"><button type="button" title="remove">×</button></td></tr>`).join("");
      q("ptab").querySelectorAll("tr").forEach((tr) => {
        const i = +tr.dataset.i;
        tr.querySelectorAll("[data-f]").forEach((inp) => inp.addEventListener("change", () => { st.parameters[i][inp.dataset.f] = inp.type === "checkbox" ? inp.checked : inp.value; }));
        tr.querySelector("button").onclick = () => { st.parameters.splice(i, 1); renderParams(); };
      });
    }
    function renderModelOpts() {
      const t = st.model.type;
      q("mtype").value = t;
      q("expr").hidden = t !== "expression"; q("tab").hidden = t !== "table";
      q("mdesc").textContent = MODELS[t] ? MODELS[t].description : t === "expression" ? "Your model as a formula of the parameters and inputs (numbers, + − × ÷ ^, exp, log, sqrt, sin, cos, tanh, …)." : "A table of simulated runs (one row per run and probe): parameter columns, input columns and the simulated output; interpolated between runs.";
      q("mopts").innerHTML = (OPTS[t] || []).map(([k, l]) => `<label>${esc(l)}<input data-o="${k}" value="${esc(st.model.options?.[k] ?? "")}" placeholder="default"></label>`).join("");
      q("mopts").querySelectorAll("[data-o]").forEach((inp) => inp.addEventListener("change", () => { st.model.options = st.model.options || {}; if (inp.value === "") delete st.model.options[inp.dataset.o]; else st.model.options[inp.dataset.o] = +inp.value; refreshShape(); }));
      if (t === "expression") { q("exprs").value = Object.entries(st.model.expressions || {}).map(([k, v]) => `${k} = ${v}`).join("\n"); q("exprin").value = (st.model.inputs || []).join(", "); }
    }
    function refreshShape() {
      const t = st.model.type;
      if (MODELS[t]) { st.inputs = MODELS[t].inputs; st.outputs = MODELS[t].outputs; }
      else if (t === "expression") { st.inputs = (st.model.inputs || []); st.outputs = Object.keys(st.model.expressions || { y: "" }); }
      else if (t === "table") { st.inputs = st.model.inputs || []; st.outputs = [st.model.output || "y"]; }
      if (t === "rod_fv") {
        const nz = +(st.model.options?.n_zones || 1), names = nz === 1 ? ["h"] : Array.from({ length: nz }, (_, j) => `h${j + 1}`);
        const have = st.parameters.map((p) => p.name);
        if (names.some((n) => !have.includes(n))) {
          const rest = st.parameters.filter((p) => !/^h\d*$/.test(p.name));
          st.parameters = names.map((n) => ({ name: n, value: 50, lower: 1, upper: 2000, unit: "W/m²K", log: true, fixed: false })).concat(rest);
          renderParams();
        }
      }
      renderRows("obs"); renderRows("val");
    }
    q("mtype").onchange = () => {
      const t = q("mtype").value; st.model = { type: t, options: {} };
      if (MODELS[t]) st.parameters = MODELS[t].parameters.map((p) => ({ ...p }));
      else if (t === "expression") { st.model.expressions = { T: "T_air + (T0 - T_air)*exp(-h*0.012*t_s/C)" }; st.model.inputs = ["t_s"]; st.parameters = [{ name: "h", value: 20, lower: 0.5, upper: 500, unit: "W/m²K", log: true }, { name: "C", value: 450, lower: 1, upper: 1e5, unit: "J/K", fixed: true, log: true }, { name: "T0", value: 90, lower: -50, upper: 1000, unit: "°C" }, { name: "T_air", value: 25, lower: -50, upper: 500, unit: "°C" }]; }
      else { st.model.parameters = []; st.model.inputs = []; st.model.output = ""; st.parameters = []; }
      st.obs = []; st.val = [];
      renderModelOpts(); refreshShape(); renderParams();
    };
    const readExpr = () => {
      const ex = {}; q("exprs").value.split("\n").forEach((l) => { const k = l.indexOf("="); if (k > 0) ex[l.slice(0, k).trim()] = l.slice(k + 1).trim(); });
      st.model.expressions = ex; st.model.inputs = q("exprin").value.split(",").map((s) => s.trim()).filter(Boolean); refreshShape();
    };
    q("exprs").addEventListener("change", readExpr); q("exprin").addEventListener("change", readExpr);
    q("tabfile").onchange = async (e) => {
      const f = e.target.files[0]; if (!f) return;
      const fd = new FormData(); fd.append("file", f);
      try {
        const t = await api("/api/table", { method: "POST", body: fd });
        q("tabinfo").textContent = `${f.name}: ${t.n_rows} runs × ${t.columns.length} columns`;
        const chk = (role) => t.columns.map((c) => `<label style="display:inline-block;margin-right:10px"><input type="checkbox" data-r="${role}" value="${esc(c)}" style="width:auto;display:inline"> ${esc(c)}</label>`).join("");
        q("tabmap").innerHTML = `<div class="map"><div class="hint">Parameter columns (to calibrate)</div>${chk("p")}<div class="hint">Input columns (probe position, time, …)</div>${chk("i")}
          <label>Simulated output column <select data-r="o">${t.columns.map((c) => `<option>${esc(c)}</option>`).join("")}</select></label>
          <div class="tools"><button type="button" class="primary">Use this table</button></div></div>`;
        q("tabmap").querySelector("button").onclick = () => {
          const ps = [...q("tabmap").querySelectorAll('[data-r="p"]:checked')].map((x) => x.value), ins = [...q("tabmap").querySelectorAll('[data-r="i"]:checked')].map((x) => x.value);
          const out = q("tabmap").querySelector('[data-r="o"]').value;
          st.model = { type: "table", rows: t.columns.length ? t.rows : {}, parameters: ps, inputs: ins, output: out };
          st.parameters = ps.map((p) => { const v = t.rows[p].filter((x) => x !== null).map(Number); const lo = Math.min(...v), hi = Math.max(...v); return { name: p, value: (lo + hi) / 2, lower: lo, upper: hi, unit: "", log: false }; });
          q("tabmap").innerHTML = `<div class="hint">${ps.length} parameters, ${ins.length} inputs, output ${esc(out)}.</div>`;
          refreshShape(); renderParams();
        };
      } catch (err) { q("tabinfo").textContent = err.message; }
    };
    q("padd").onclick = () => { st.parameters.push({ name: "p" + (st.parameters.length + 1), value: 1, lower: 0.01, upper: 100, unit: "", log: true, fixed: false }); renderParams(); };
    q("pspace").onchange = async (e) => {
      const f = e.target.files[0]; if (!f) return;
      try { st.parameters = await api("/api/parameter-space", { json: JSON.parse(await f.text()) }); renderParams(); }
      catch (err) { alert(err.message); }
    };
    obsEditor(q("obsed"), "obs", "measurements");
    if (opts.validation) obsEditor(q("valed"), "val", "held-out measurements");

    const num = (v) => {                       // a sensor export may write 94,1 for 94.1
      if (v === "" || v === null || v === undefined) return null;
      if (typeof v === "string" && /^\s*-?\d+,\d+\s*$/.test(v)) v = v.replace(",", ".");
      const x = +v; return isFinite(x) ? x : null;
    };
    function obsJSON(rows, needValues) {
      if (!rows.length) return null;
      const inputs = {}; st.inputs.forEach((k) => (inputs[k] = rows.map((r) => num(r[k]))));
      const vals = rows.map((r) => num(r.value));
      const anyVal = vals.some((v) => v !== null);
      if (needValues && vals.some((v) => v === null)) throw new Error("every measurement needs a value");
      return { inputs, sigma: rows.map((r) => num(r.sigma)), values: anyVal ? vals : null, observable: rows.map((r) => r.observable || st.outputs[0]) };
    }
    function modelSpec() {
      const m = st.model;
      if (m.type === "expression") return { type: "expression", expressions: m.expressions, inputs: m.inputs, parameters: st.parameters.map(paramJSON) };
      if (m.type === "table") {
        const n = m.rows[m.output]?.length || 0, rows = [];
        for (let i = 0; i < n; i++) { const r = {}; [...m.parameters, ...m.inputs, m.output].forEach((c) => (r[c] = m.rows[c][i])); rows.push(r); }
        return { type: "table", rows, parameters: m.parameters, inputs: m.inputs, output: m.output };
      }
      return { type: m.type, options: m.options || {} };
    }
    const paramJSON = (p) => { const o = { name: p.name, value: +p.value, lower: num(p.lower), upper: num(p.upper), unit: p.unit || "", log: !!p.log, fixed: !!p.fixed }; if (p.truth !== undefined && p.truth !== null) o.truth = +p.truth; return o; };
    return {
      get() {
        return { model: modelSpec(), parameters: st.parameters.map(paramJSON), observations: obsJSON(st.obs, opts.values === "required"),
          validation: opts.validation ? obsJSON(st.val, true) : undefined };
      },
      set(p) {
        st.model = JSON.parse(JSON.stringify(p.model_spec || p.model));
        if (!st.model.options) st.model.options = {};
        st.parameters = p.parameters.map((x) => ({ ...x }));
        refreshShape();
        const rows = (o) => !o ? [] : o.sigma.map((s, i) => { const r = { sigma: s, value: o.values ? o.values[i] : "", observable: o.observable ? o.observable[i] : st.outputs[0] }; st.inputs.forEach((k) => (r[k] = o.inputs[k] ? o.inputs[k][i] : "")); return r; });
        st.obs = rows(p.observations); st.val = rows(p.validation);
        renderModelOpts(); renderParams(); renderRows("obs"); renderRows("val");
      },
      state: st,
    };
  }

  window.InvUI = { api, pollJob, xy, heat, fmt, esc, ProblemEditor };
})();
