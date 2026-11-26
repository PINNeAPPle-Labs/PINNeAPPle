/* The run viewer: gates, stages, figures, and the model on a grid of its own space. No dependencies. */
(function () {
  "use strict";
  const D = window.PP_DATA || {};
  const $ = (id) => document.getElementById(id);
  const f4 = (v) => (v === null || v === undefined || Number.isNaN(v)) ? "—" :
    (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(3) : String(+v.toFixed(4)));

  /* ------------------------------------------------------------------ header, gates, stages, figures */
  const run = D.run || {};
  $("run-name").textContent = run.name || "run";
  const pill = $("run-status");
  pill.textContent = run.status || "—";
  pill.className = "pill " + (run.status || "");
  $("run-summary").textContent = run.summary || "";
  document.title = "pinneapple — " + (run.name || "run");

  const gates = D.gates || [];
  const table = $("gates");
  table.innerHTML = "<tr><th>gate</th><th>status</th><th>value</th><th>why</th></tr>" + gates.map((g) => {
    const why = (g.limitations && g.limitations[0]) || (g.evidence && g.evidence[0]) || (g.recommendations && g.recommendations[0]) || "";
    return `<tr><td class="gate">${g.gate}</td><td><span class="pill ${g.status}">${g.status}</span></td>` +
           `<td class="num">${g.value || ""}</td><td class="note">${why}</td></tr>`;
  }).join("");

  $("stages").innerHTML = (D.stages || []).map((s) =>
    `<li><span class="s">${s.stage}</span>${s.seconds !== null && s.seconds !== undefined ?
      `<span class="t">${(+s.seconds).toFixed(1)} s</span>` : ""}` +
    `${s.notes && s.notes[0] ? `<span class="t">· ${s.notes[0]}</span>` : ""}</li>`).join("");

  $("figures").innerHTML = (D.figures || []).map((f) =>
    `<figure><img src="${f.src}" alt="${f.name}"><figcaption>${f.caption}</figcaption></figure>`).join("");

  if ((D.scenes || []).length) {
    $("scene-card").hidden = false;
    $("scenes").innerHTML = D.scenes.map((s) => `<div class="hint">${s.title}</div>`).join("");
    const loader = document.createElement("script");
    loader.src = "scene.js";
    loader.onload = () => D.scenes.forEach((s) => {
      const frame = document.createElement("iframe");       /* file:// cannot frame file://: the page goes in as srcdoc */
      frame.srcdoc = window.PP_SCENE || "";
      $("scenes").appendChild(frame);
    });
    document.head.appendChild(loader);
  }

  /* ------------------------------------------------------------------------------- the explorer */
  const E = D.explore;
  if (!E || E.error || !E.magnitude) {
    $("explore-note").textContent = E && E.error ? "this run has no model to explore: " + E.error
      : "this run has no parameters to sweep.";
    $("map").style.display = "none";
    return;
  }
  $("explore-note").textContent = E.note || "";
  const nx = E.shape[0], ny = E.shape[1], m = (E.outputs || []).length;
  const mag = E.magnitude, vals = E.values;
  let sel = [Math.floor(nx / 2), Math.floor(ny / 2)];
  let scale = [Math.min(...mag.flat()), Math.max(...mag.flat())];
  if (!(scale[1] > scale[0])) { scale = [scale[0] - 1e-9, scale[0] + 1e-9]; }

  $("axis-x").textContent = `${E.axes[0]}: ${f4(E.x[0])} → ${f4(E.x[nx - 1])}`;
  $("axis-y").textContent = ny > 1 ? `${E.axes[1]}: ${f4(E.y[0])} → ${f4(E.y[ny - 1])}` : "";

  function colour(t) {                       /* viridis, near enough */
    t = Math.max(0, Math.min(1, t));
    const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    const p = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(p)), f = p - i;
    return `rgb(${stops[i].map((c, k) => Math.round(c + f * (stops[i + 1][k] - c))).join(",")})`;
  }

  const map = $("map"), ctx = map.getContext("2d");
  function plotLines() {                      /* one input: the outputs as curves of that input */
    const pad = 34, w = map.width - 2 * pad, h = map.height - 2 * pad;
    const series = [];
    for (let k = 0; k < m && k < 8; k++) series.push(vals[0].map((cell) => cell[k]));
    const all = series.flat(), lo = Math.min(...all), hi = Math.max(...all), span = (hi - lo) || 1;
    ctx.clearRect(0, 0, map.width, map.height);
    ctx.strokeStyle = "#1b212a"; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) {
      const y = pad + (h * g) / 4;
      ctx.beginPath(); ctx.moveTo(pad, y); ctx.lineTo(pad + w, y); ctx.stroke();
    }
    series.forEach((s, k) => {
      ctx.strokeStyle = `hsl(${200 - k * 26}, 70%, 58%)`; ctx.lineWidth = 2; ctx.beginPath();
      s.forEach((v, i) => {
        const x = pad + (w * i) / Math.max(1, s.length - 1), y = pad + h - (h * (v - lo)) / span;
        i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
      });
      ctx.stroke();
    });
    const mx = pad + (w * sel[0]) / Math.max(1, nx - 1);
    ctx.strokeStyle = "#f0f6fc"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(mx, pad - 10); ctx.lineTo(mx, pad + h + 6); ctx.stroke();
    ctx.fillStyle = "#8b949e"; ctx.font = "12px ui-monospace, monospace";
    ctx.fillText(`max ${f4(hi)}`, pad + 4, pad - 12);
    ctx.fillText(`min ${f4(lo)}`, pad + 4, pad + h + 20);
  }

  function plotStrip() {                      /* one input, a field: the field across that input */
    const vals0 = vals[0], rows = m, cols = nx;
    const flat = vals0.flat(), lo = Math.min(...flat), hi = Math.max(...flat), span = (hi - lo) || 1;
    const w = Math.floor(map.width / cols), h = Math.floor(map.height / rows);
    ctx.clearRect(0, 0, map.width, map.height);
    for (let i = 0; i < cols; i++) {
      for (let k = 0; k < rows; k++) {
        ctx.fillStyle = colour((vals0[i][k] - lo) / span);
        ctx.fillRect(i * w, map.height - (k + 1) * h, w + 1, h + 1);
      }
    }
    const mx = sel[0] * w + w / 2;
    ctx.strokeStyle = "#f0f6fc"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(mx, 0); ctx.lineTo(mx, map.height); ctx.stroke();
  }

  function drawMap() {
    if (ny === 1 && m > 8) { plotStrip(); return; }
    if (ny === 1) { plotLines(); return; }
    const w = Math.floor(map.width / nx), h = Math.floor(map.height / ny);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const t = (mag[j][i] - scale[0]) / (scale[1] - scale[0]);
        ctx.fillStyle = colour(t);
        ctx.fillRect(i * w, map.height - (j + 1) * h, w + 1, h + 1);
      }
    }
    const cx = sel[0] * w + w / 2, cy = map.height - (sel[1] + 1) * h + h / 2;
    ctx.strokeStyle = "#f0f6fc"; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(cx, 0); ctx.lineTo(cx, map.height); ctx.moveTo(0, cy); ctx.lineTo(map.width, cy); ctx.stroke();
    ctx.strokeStyle = "#0e1117"; ctx.lineWidth = 1; ctx.stroke();
    ctx.fillStyle = "#f0f6fc"; ctx.beginPath(); ctx.arc(cx, cy, 4, 0, 2 * Math.PI); ctx.fill();
  }

  function held() {
    const rows = (E.inputs || []).slice(2);
    $("held").innerHTML = rows.length
      ? "held at the middle of its range: " + rows.map((r) => `${r.name} = ${f4((r.low + r.high) / 2)}`).join(" · ")
      : "";
  }

  function drawCurve(cell) {
    const c = $("curve"), g = c.getContext("2d");
    g.clearRect(0, 0, c.width, c.height);
    const pad = 30, w = c.width - 2 * pad, h = c.height - 2 * pad;
    if (!cell || !cell.length) return;
    if (m === 1 && ny > 1) {                                 /* a physics-only model: the two cuts through the point */
      const cutX = vals[sel[1]].map((v) => v[0]);            /* u(x) at the chosen t */
      const cutY = vals.map((row) => row[sel[0]][0]);        /* u(t) at the chosen x */
      const all = cutX.concat(cutY), lo = Math.min(...all), hi = Math.max(...all), span = (hi - lo) || 1;
      const line = (series, colour) => {
        g.strokeStyle = colour; g.lineWidth = 2; g.beginPath();
        series.forEach((v, k) => {
          const x = pad + (w * k) / Math.max(1, series.length - 1), y = pad + h - (h * (v - lo)) / span;
          k ? g.lineTo(x, y) : g.moveTo(x, y);
        });
        g.stroke();
      };
      line(cutX, "#0081be"); line(cutY, "#a371f7");
      g.font = "12px ui-monospace, monospace"; g.fillStyle = "#0081be";
      g.fillText(`${E.outputs[0]}(${E.axes[0]}) at ${E.axes[1]} = ${f4(E.y[sel[1]])}`, pad, 16);
      g.fillStyle = "#a371f7";
      g.fillText(`${E.outputs[0]}(${E.axes[1]}) at ${E.axes[0]} = ${f4(E.x[sel[0]])}`, pad + 190, 16);
      $("curve-note").textContent = `the two cuts through the point: ${f4(lo)} … ${f4(hi)}`;
      return;
    }
    const lo = Math.min(...cell), hi = Math.max(...cell), span = (hi - lo) || 1;
    g.strokeStyle = "#0081be"; g.lineWidth = 2; g.beginPath();
    if (cell.length > 8) {                                   /* a field: one line over its grid */
      cell.forEach((v, k) => {
        const x = pad + (w * k) / (cell.length - 1), y = pad + h - (h * (v - lo)) / span;
        k ? g.lineTo(x, y) : g.moveTo(x, y);
      });
      g.stroke();
      $("curve-note").textContent = `the field at that point: ${cell.length} values, ${f4(lo)} … ${f4(hi)}`;
    } else {                                                 /* a handful of outputs: bars */
      const bw = w / cell.length;
      cell.forEach((v, k) => {
        const bh = h * (v - Math.min(0, lo)) / (hi - Math.min(0, lo) || 1);
        g.fillStyle = "#0081be";
        g.fillRect(pad + k * bw + bw * 0.2, pad + h - bh, bw * 0.6, bh);
        g.fillStyle = "#8b949e"; g.font = "11px ui-monospace, monospace"; g.textAlign = "center";
        g.fillText(E.outputs[k], pad + k * bw + bw / 2, c.height - 8);
      });
      $("curve-note").textContent = `${cell.length} output(s) at that point`;
    }
  }

  function update() {
    drawMap();
    held();
    const cell = vals[sel[1]][sel[0]];
    const shown = m <= 8 ? E.outputs.map((name, k) => [name, cell[k]]) : [["‖field‖", Math.hypot(...cell)]];
    $("values").innerHTML = shown.map(([k, v]) =>
      `<div class="cell"><div class="k">${k}</div><div class="v">${f4(v)}</div></div>`).join("") +
      `<div class="cell"><div class="k">${E.axes[0]}</div><div class="v">${f4(E.x[sel[0]])}</div></div>` +
      (ny > 1 ? `<div class="cell"><div class="k">${E.axes[1]}</div><div class="v">${f4(E.y[sel[1]])}</div></div>` : "");
    drawCurve(cell);
  }

  function pick(event) {
    const box = map.getBoundingClientRect();
    const fx = Math.min(1, Math.max(0, (event.clientX - box.left) / box.width));
    const fy = 1 - (event.clientY - box.top) / box.height;
    const i = ny === 1 ? Math.round(fx * (nx - 1)) : Math.round(fx * nx - 0.5);
    const j = ny === 1 ? 0 : Math.round(fy * ny - 0.5);
    sel = [Math.min(nx - 1, Math.max(0, i)), Math.min(ny - 1, Math.max(0, j))];
    update();
  }
  let dragging = false;
  map.addEventListener("mousedown", (e) => { dragging = true; pick(e); });
  map.addEventListener("mousemove", (e) => { if (dragging) pick(e); });
  window.addEventListener("mouseup", () => { dragging = false; });
  map.addEventListener("touchmove", (e) => { if (e.touches[0]) { pick(e.touches[0]); e.preventDefault(); } }, { passive: false });

  update();
})();
