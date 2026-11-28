// "Why trust this?": the model card, the evidence behind it and the checks that ran, for the current mode.
import { h, icon, sig, renderVerdict } from "./ui.js";

const DRAWER = () => document.getElementById("trust");

export function openTrust(open = true) {
  const d = DRAWER(), scrim = document.getElementById("scrim"), btn = document.getElementById("trust-open");
  d.classList.toggle("open", open); d.setAttribute("aria-hidden", String(!open));
  scrim.hidden = !open; btn.setAttribute("aria-expanded", String(open));
  if (open) document.getElementById("trust-close").focus();
}

const pretty = (k) => k.replace(/_/g, " ");

function validityTable(vr, inputs) {
  const unit = Object.fromEntries((inputs || []).map((i) => [i.name, i.unit]));
  const rows = Object.entries(vr).map(([k, v]) => h("tr", {}, h("td", {}, pretty(k)),
    Array.isArray(v) ? h("td", { class: "n" }, `${sig(v[0], 4)} … ${sig(v[1], 4)} ${unit[k] && unit[k] !== "-" ? unit[k] : ""}`) : h("td", {}, String(v))));
  return h("table", { class: "t" }, h("tbody", {}, rows));
}

/** The stored trust report of the zoo gate: metrics vs acceptance limits and the physics checks. */
function reportTable(tr) {
  const rows = (tr.checagens || []).map((c) => h("tr", {},
    h("td", {}, c.name),
    h("td", { class: "n" }, c.value === null || c.value === undefined ? "–" : sig(c.value, 3)),
    h("td", { class: "n" }, c.limit === null || c.limit === undefined ? "" : `≤ ${sig(c.limit, 3)}`),
    h("td", {}, h("span", { class: `badge ${c.status === "supports" ? "pass" : "fail"}` }, c.status === "supports" ? "pass" : c.status))));
  return h("table", { class: "t" }, h("thead", {}, h("tr", {}, h("th", {}, "Check"), h("th", {}, "Value"), h("th", {}, "Limit"), h("th", {}, ""))), h("tbody", {}, rows));
}

export function renderModelTrust({ meta, evaluation, extra }) {
  const c = meta.card, tr = c.trust_report || {}, body = document.getElementById("trust-body");
  body.innerHTML = "";
  const live = h("div");
  renderVerdict(live, evaluation, "verdict-drawer");
  body.append(
    h("h4", {}, "This query"), live,
    h("p", { class: "hint" }, "These checks run in your browser on every change. Rule IDs match ", h("code", {}, "pinneapple check"), "."),
    h("h4", {}, "The model"),
    h("p", {}, h("b", {}, `${c.name} v${c.version}`), " · ", c.task),
    h("p", {}, c.description),
    h("p", {}, "Zoo gate decision: ", h("span", { class: `badge ${tr.decisao === "APPROVED" ? "pass" : "warn"}` }, tr.decisao || "n/a"),
      tr.score !== undefined ? ` score ${sig(tr.score, 3)}` : ""),
    h("h4", {}, "Validity range"), validityTable(c.validity_range, c.inputs),
    h("h4", {}, "Evidence: metrics against an independent reference, and the physics checks"),
    h("p", {}, c.reference_source),
    reportTable(tr),
    h("h4", {}, "Data provenance"),
    h("ul", {},
      h("li", {}, "Dataset: ", h("code", {}, c.data_lineage?.dataset || "–"), c.data_lineage?.split ? ` (split ${c.data_lineage.split})` : ""),
      h("li", {}, "Source: ", c.data_lineage?.source || "–"),
      c.training_config?.script ? h("li", {}, "Training script: ", h("code", {}, c.training_config.script)) : null,
      h("li", {}, "Licence: ", c.license)),
    h("h4", {}, "Known limitations"), h("ul", {}, (c.known_limitations || []).map((k) => h("li", {}, k))),
    extra || null,
    h("h4", {}, "What runs in your browser"),
    h("p", {}, `The zoo model exported by pp.zoo.export_web (ONNX, ${(meta.onnx.bytes / 1024).toFixed(0)} kB, opset ${meta.onnx.opset}): pre- and post-processing and these query checks are inside the graph. `,
      `Its outputs match the PyTorch model to ${meta.onnx.parity.toExponential(1)} (limit 1e-5) on the zoo's held-out and independent test sets.`),
    h("p", { class: "hint" }, "SHA-256 ", h("code", { style: "word-break:break-all" }, meta.onnx.sha256)),
    h("p", {}, h("a", { href: meta.hub, target: "_blank", rel: "noopener" }, "Model on the Hugging Face Hub"), " · ",
      h("a", { href: meta.code.adapter, target: "_blank", rel: "noopener" }, "adapter and physics checks"), " · ",
      h("a", { href: meta.code.checks, target: "_blank", rel: "noopener" }, "query checks"), " · ",
      h("a", { href: meta.code.export, target: "_blank", rel: "noopener" }, "ONNX export")),
  );
}

export function renderGeometryTrust(g) {
  const body = document.getElementById("trust-body");
  body.innerHTML = "";
  if (!g) { body.append(h("p", {}, "Pick a geometry to see where its dimensions come from.")); return; }
  const est = g.sources.filter((s) => s.estimate).length;
  body.append(
    h("h4", {}, "The geometry"),
    h("p", {}, h("b", {}, g.title), ` · ${g.category}`), h("p", {}, g.description),
    h("div", { class: `verdict ${est ? "warn" : "pass"}`, html: icon(est ? "warn" : "pass") },
      h("div", {}, h("b", {}, est ? `${est} of ${g.sources.length} dimensions are estimates` : "Every dimension has a published source"),
        h("span", {}, "Written from published dimensions; no third-party CAD file is copied."))),
    h("h4", {}, "Sources of the dimensions"),
    h("table", { class: "t" }, h("tbody", {}, g.sources.map((s) => h("tr", {}, h("td", {}, s.what), h("td", {}, s.ref, s.estimate ? h("span", { class: "badge warn", style: "margin-left:6px" }, "estimate") : null))))),
    h("h4", {}, "What it is for"), h("ul", {}, g.physics.map((p) => h("li", {}, p))),
    h("h4", {}, "Regenerate it"), h("pre", { class: "code" }, g.regenerate),
    h("p", {}, h("a", { href: g.code, target: "_blank", rel: "noopener" }, "Builder source"), " · ",
      h("a", { href: g.glb_hub, target: "_blank", rel: "noopener" }, "GLB on the Hub")),
  );
}
