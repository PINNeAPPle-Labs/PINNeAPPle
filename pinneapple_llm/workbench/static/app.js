// PINNeAPPle Agent Workbench: sessions, the live event feed, approvals, questions, files, plan and feedback.
"use strict";

const TOKEN = new URLSearchParams(location.search).get("token") || "";
const $ = (s) => document.querySelector(s);
let INFO = null, CURRENT = null, STREAM = null, LAST_SEQ = 0, BUSY = false;
const TOOL_CARDS = {};                       // tool call id -> <details>

async function api(path, opts = {}) {
  const sep = path.includes("?") ? "&" : "?";
  const r = await fetch(path + (TOKEN ? `${sep}token=${encodeURIComponent(TOKEN)}` : ""), {
    headers: { "Content-Type": "application/json" }, ...opts,
    body: opts.body ? JSON.stringify(opts.body) : undefined,
  });
  if (!r.ok) throw new Error((await r.text()) || r.statusText);
  return r.json();
}

function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// A small, safe Markdown subset: code blocks, inline code, headings, bold, italics, links, lists.
function md(text) {
  const blocks = String(text ?? "").split(/```/);
  return blocks.map((b, i) => {
    if (i % 2) {
      const nl = b.indexOf("\n");
      return `<pre><code>${esc(nl >= 0 ? b.slice(nl + 1) : b)}</code></pre>`;
    }
    let h = esc(b);
    h = h.replace(/`([^`]+)`/g, "<code>$1</code>")
         .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
         .replace(/(^|\W)\*([^*\n]+)\*(?=\W|$)/g, "$1<em>$2</em>")
         .replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>')
         .replace(/^#{1,4} (.*)$/gm, "<strong>$1</strong>")
         .replace(/^\s*[-*] (.*)$/gm, "• $1");
    return h.replace(/\n/g, "<br>");
  }).join("");
}

function add(el) {
  const feed = $("#feed");
  const stick = feed.scrollHeight - feed.scrollTop - feed.clientHeight < 80;
  feed.appendChild(el);
  if (stick) feed.scrollTop = feed.scrollHeight;
  return el;
}

function node(cls, html, depth = 0) {
  const d = document.createElement("div");
  d.className = `${cls} depth-${Math.min(depth || 0, 2)}`;
  d.innerHTML = html;
  return d;
}

function short(args) {
  const s = JSON.stringify(args ?? {});
  return s.length > 140 ? s.slice(0, 140) + "…" : s;
}

function fileUrl(path) {
  return `/api/sessions/${CURRENT}/file?path=${encodeURIComponent(path)}` + (TOKEN ? `&token=${encodeURIComponent(TOKEN)}` : "");
}

function renderEvent(e) {
  LAST_SEQ = Math.max(LAST_SEQ, e.seq || 0);
  const d = e.depth || 0;
  switch (e.type) {
    case "user":
      add(node("msg user", esc(e.content)));
      setBusy(true);
      break;
    case "message":
      add(node("msg agent", `<div class="who">${esc(e.agent)}</div>${md(e.content)}`, d));
      break;
    case "thinking":
      add(node("note", `${esc(e.agent)}: ${esc(e.content)}`, d));
      break;
    case "delegate_start":
      add(node("note", `→ <strong>${esc(e.agent)}</strong>: ${esc(e.task).slice(0, 400)}`, d - 1));
      break;
    case "delegate_end":
      add(node("note", `← ${esc(e.agent)} finished`, d - 1));
      break;
    case "tool_call": {
      if (e.tool === "todo") break;                      // the plan panel shows it
      const det = document.createElement("details");
      det.className = `tool depth-${Math.min(d, 2)}`;
      det.dataset.testid = "tool";
      det.innerHTML = `<summary>${esc(e.agent)} · <strong>${esc(e.tool)}</strong> ${esc(short(e.args))}` +
        `<span class="st">…</span></summary><pre>${esc(JSON.stringify(e.args, null, 1))}</pre>`;
      TOOL_CARDS[e.id] = det;
      add(det);
      break;
    }
    case "tool_result": {
      const det = TOOL_CARDS[e.id];
      if (det) {
        det.querySelector(".st").innerHTML = e.ok ? '<span class="ok">done</span>' : '<span class="bad">error</span>';
        const pre = document.createElement("pre");
        pre.textContent = e.result;
        det.appendChild(pre);
      }
      break;
    }
    case "plan":
      renderPlan(e.items);
      break;
    case "file": {
      const p = e.path.toLowerCase(), url = fileUrl(e.path);
      let inner = `<a href="${url}" target="_blank">${esc(e.path)}</a>`;
      if (/\.(png|jpe?g|gif|webp|svg)$/.test(p)) inner = `<img src="${url}" alt="${esc(e.path)}">`;
      else if (/\.(html?|pdf)$/.test(p)) inner = `<iframe src="${url}" sandbox="allow-scripts"></iframe>`;
      else if (/\.(mp4|webm)$/.test(p)) inner = `<video src="${url}" controls style="max-width:100%"></video>`;
      add(node("file", `${inner}<div class="note">${esc(e.caption || e.path)}</div>`));
      break;
    }
    case "approval_request":
      add(card(e, `<strong>Approve ${esc(e.tool)}?</strong><pre>${esc(e.args.command || JSON.stringify(e.args, null, 1))}</pre>`,
        [["Approve", true], ["Deny", false]]));
      break;
    case "question":
      add(card(e, `<strong>${esc(e.question)}</strong>`, (e.options || []).map((o) => [o, o]), true));
      break;
    case "error":
      add(node("err", esc(e.content), d));
      break;
    case "compacted":
      add(node("note", `(${esc(e.agent)}: older context summarised)`, d));
      break;
    case "cancelled":
      add(node("note", "stopped"));
      break;
    case "done":
      setBusy(false);
      add(rating(e));
      if (e.usage) $("#meta").textContent = usage(e.usage);
      break;
  }
}

function usage(u) {
  return `${u.model_calls} model calls · ${u.prompt_tokens} tokens in · ${u.completion_tokens} out · ${u.seconds} s`;
}

function card(e, html, buttons, free = false) {
  const c = node("card", html);
  c.dataset.testid = e.type;
  const row = document.createElement("div");
  row.className = "row";
  const send = async (value) => {
    try { await api(`/api/sessions/${CURRENT}/answer`, { method: "POST", body: { id: e.id, value } }); }
    catch (err) { /* already answered (replay) */ }
    c.classList.add("answered");
    row.querySelectorAll("button,input").forEach((b) => (b.disabled = true));
    const n = document.createElement("div");
    n.className = "note";
    n.textContent = value === true ? "approved" : value === false ? "denied" : `answered: ${value}`;
    c.appendChild(n);
  };
  for (const [label, value] of buttons) {
    const b = document.createElement("button");
    b.textContent = label;
    if (value === true) b.className = "primary";
    b.onclick = () => send(value);
    row.appendChild(b);
  }
  if (free) {
    const inp = document.createElement("input");
    inp.placeholder = "Your answer";
    inp.onkeydown = (k) => { if (k.key === "Enter") send(inp.value); };
    const b = document.createElement("button");
    b.textContent = "Answer";
    b.className = "primary";
    b.onclick = () => send(inp.value);
    row.append(inp, b);
  }
  c.appendChild(row);
  return c;
}

function rating(e) {
  const r = node("rate", "Was this turn good? (used to pick fine-tuning data)");
  for (const [label, v] of [["👍", 1], ["👎", -1]]) {
    const b = document.createElement("button");
    b.textContent = label;
    b.onclick = async () => {
      const note = v < 0 ? prompt("What went wrong? (optional)") || "" : "";
      await api(`/api/sessions/${CURRENT}/feedback`, { method: "POST", body: { turn_id: e.turn_id, rating: v, note } });
      r.querySelectorAll("button").forEach((x) => x.classList.toggle("on", x === b));
    };
    r.appendChild(b);
  }
  return r;
}

function renderPlan(items) {
  const box = $("#planbox"), ol = $("#plan");
  box.hidden = !items || !items.length;
  ol.innerHTML = (items || []).map((i) => `<li class="${esc(i.status)}">${esc(i.text)}</li>`).join("");
}

function setBusy(b) {
  BUSY = b;
  $("#stop").hidden = !b;
  $("#send").disabled = b;
}

async function loadSessions() {
  const list = await api("/api/sessions");
  const box = $("#sessions");
  box.innerHTML = "";
  for (const s of list) {
    if (s.id === CURRENT) $("#title").textContent = s.title;
    const d = document.createElement("div");
    d.className = "sess" + (s.id === CURRENT ? " active" : "");
    d.innerHTML = `${s.busy ? '<span class="dot"></span>' : ""}<span class="t" title="${esc(s.workspace)}">${esc(s.title)}</span><span class="x">✕</span>`;
    d.onclick = (ev) => {
      if (ev.target.classList.contains("x")) {
        if (confirm(`Delete "${s.title}"?`)) api(`/api/sessions/${s.id}`, { method: "DELETE" }).then(loadSessions);
        return;
      }
      open(s.id);
    };
    box.appendChild(d);
  }
}

async function open(id) {
  if (STREAM) STREAM.close();
  CURRENT = id;
  LAST_SEQ = 0;
  $("#feed").innerHTML = "";
  for (const k of Object.keys(TOOL_CARDS)) delete TOOL_CARDS[k];
  const s = await api(`/api/sessions/${id}`);
  $("#title").textContent = s.title;
  $("#meta").textContent = `${s.workspace} · ${s.settings.model} · approvals: ${s.settings.policy}`;
  renderPlan(s.plan);
  setBusy(s.busy);
  STREAM = new EventSource(`/api/sessions/${id}/events?after=0` + (TOKEN ? `&token=${encodeURIComponent(TOKEN)}` : ""));
  STREAM.onmessage = (m) => renderEvent(JSON.parse(m.data));
  loadSessions();
}

async function sendMessage() {
  const text = $("#input").value.trim();
  if (!text || BUSY) return;
  if (!CURRENT) { await newSession(); if (!CURRENT) return; }
  $("#input").value = "";
  try { await api(`/api/sessions/${CURRENT}/messages`, { method: "POST", body: { text } }); }
  catch (e) { add(node("err", esc(e.message))); }
  setTimeout(loadSessions, 300);
}

function newSession() {
  return new Promise((resolve) => {
    const dlg = $("#dlg-new");
    $("#ws").value = INFO.workspace;
    const sel = $("#model-select");
    const models = [...new Set([INFO.model, ...INFO.installed_models])];
    sel.innerHTML = models.map((m) => `<option ${m === INFO.model ? "selected" : ""}>${esc(m)}</option>`).join("");
    dlg.onclose = async () => {
      if (dlg.returnValue === "ok") {
        const f = new FormData($("#form-new"));
        try {
          const r = await api("/api/sessions", { method: "POST", body: {
            workspace: f.get("workspace"), model: f.get("model"), policy: f.get("policy"), web: f.get("web") === "on" } });
          await open(r.id);
        } catch (e) { alert(e.message); }
      }
      resolve();
    };
    dlg.showModal();
  });
}

async function showPanel(kind) {
  const p = $("#panel");
  if (kind === "machine") {
    const m = INFO.machine, r = INFO.recommended;
    p.innerHTML = `<h3>Model &amp; machine</h3><table>
      <tr><th>Machine</th><td>${esc(m.system)} ${esc(m.arch)}, ${m.memory_gb} GB${m.apple_silicon ? " (Apple Silicon)" : ""}</td></tr>
      <tr><th>Backend</th><td>${esc(INFO.backend)}</td></tr><tr><th>Default model</th><td>${esc(INFO.model)}</td></tr>
      <tr><th>Recommended here</th><td>${esc(r.model)} (~${r.approx_size_gb} GB, context ${r.context}): ${esc(r.note)}</td></tr>
      <tr><th>Installed</th><td>${esc(INFO.installed_models.join(", ") || "none found: is the server running?")}</td></tr></table>
      <p class="note">Measure a model on your machine: <code>python -m pinneapple_llm.workbench eval --model ${esc(INFO.model)}</code></p>`;
  } else if (kind === "training") {
    const t = await api("/api/training");
    p.innerHTML = `<h3>Training data</h3><table>
      <tr><th>Database</th><td>${esc(t.engine)}</td></tr><tr><th>Sessions</th><td>${t.sessions}</td></tr>
      <tr><th>Model calls recorded</th><td>${t.model_calls}</td></tr>
      <tr><th>Rated turns</th><td>${t.rated_turns} (${t.good_turns} good)</td></tr></table>
      <p><button id="export" class="primary">Export dataset (drops 👎 turns)</button></p><pre id="exp"></pre>
      <p class="note">Then train: <code>python -m pinneapple_llm.workbench finetune train --data ~/.pinneapple/workbench/dataset</code>
      (MLX on Apple Silicon) and compare with <code>eval</code>.</p>`;
    $("#export").onclick = async () => {
      $("#exp").textContent = JSON.stringify(await api("/api/training/export", { method: "POST", body: {} }), null, 1);
    };
  } else {
    p.innerHTML = `<h3>Agents</h3><p class="note">Edit a prompt to change how an agent works. Saved copies go to
      ~/.pinneapple/workbench/agents/ and override the built-in ones.</p>` +
      INFO.agents.map((a) => `<p><button data-agent="${esc(a.name)}">${esc(a.title)}</button>
        <span class="note">${esc(a.description)}<br>tools: ${esc(a.tools.join(", "))}</span></p>`).join("") +
      `<textarea id="agent-text" hidden></textarea><p><button id="agent-save" class="primary" hidden>Save</button></p>`;
    let current = null;
    p.querySelectorAll("[data-agent]").forEach((b) => (b.onclick = async () => {
      current = b.dataset.agent;
      const a = await api(`/api/agents/${current}`);
      $("#agent-text").hidden = $("#agent-save").hidden = false;
      $("#agent-text").value = a.text;
    }));
    $("#agent-save").onclick = async () => {
      try { await api(`/api/agents/${current}`, { method: "PUT", body: { text: $("#agent-text").value } }); alert("Saved"); }
      catch (e) { alert(e.message); }
    };
  }
  $("#dlg-panel").showModal();
}

$("#send").onclick = sendMessage;
$("#stop").onclick = () => CURRENT && api(`/api/sessions/${CURRENT}/cancel`, { method: "POST" });
$("#new-btn").onclick = newSession;
$("#panel-close").onclick = () => $("#dlg-panel").close();
document.querySelectorAll("nav [data-panel]").forEach((b) => (b.onclick = () => showPanel(b.dataset.panel)));
$("#input").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

(async () => {
  INFO = await api("/api/info");
  await loadSessions();
  const first = document.querySelector(".sess");
  const want = new URLSearchParams(location.search).get("session");
  if (want) open(want); else if (first) first.click();
  setInterval(loadSessions, 5000);
})();
