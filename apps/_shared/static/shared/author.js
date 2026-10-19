// "About the author" button + dialog and footer credit, shared by the
// PINNeAPPle apps. Details live in /shared/author.json (one place to edit).
(function () {
  const css = `
  .author-btn { border: 1px solid var(--line); background: var(--panel); color: var(--ink); border-radius: 8px;
    padding: 7px 12px; font: inherit; font-size: 13px; font-weight: 600; cursor: pointer; display: inline-flex; gap: 6px; align-items: center; }
  .author-btn:hover { border-color: var(--brand); }
  dialog.author { border: 0; border-radius: 14px; padding: 0; width: min(440px, calc(100vw - 32px));
    box-shadow: 0 20px 60px #0004; color: var(--ink); }
  dialog.author::backdrop { background: #17203366; }
  .author .hd { display: flex; gap: 14px; align-items: center; padding: 20px 22px 8px; }
  .author .av { width: 52px; height: 52px; border-radius: 50%; background: var(--brand); color: #fff;
    display: grid; place-items: center; font-weight: 700; font-size: 19px; flex: none; }
  .author .nm { font-weight: 700; font-size: 17px; }
  .author .rl { color: var(--muted); font-size: 13px; }
  .author .bd { padding: 6px 22px 4px; font-size: 13.5px; line-height: 1.55; }
  .author .ctx { padding: 4px 22px 0; font-size: 12px; color: var(--muted); }
  .author .acts { display: grid; gap: 8px; padding: 16px 22px 20px; }
  .author .acts a, .author .acts button { display: flex; align-items: center; justify-content: center; gap: 8px;
    padding: 10px 14px; border-radius: 9px; font: inherit; font-weight: 600; font-size: 14px; text-decoration: none; cursor: pointer; }
  .author .primary { background: var(--brand); color: #fff; border: 0; }
  .author .li { background: #0a66c2; color: #fff; border: 0; }
  .author .sec { background: var(--panel); color: var(--ink); border: 1px solid var(--line); }
  .author .x { position: absolute; top: 10px; right: 12px; border: 0; background: none; font-size: 20px;
    color: var(--muted); cursor: pointer; }
  .foot .credit a { color: inherit; font-weight: 600; }
  @media print { .author-btn, dialog.author { display: none !important; } }`;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const MAIL = '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="m3 7 9 6 9-6"/></svg>';
  const LI = '<svg width="16" height="16" viewBox="0 0 24 24" fill="currentColor"><path d="M4.98 3.5a2.5 2.5 0 1 1 0 5 2.5 2.5 0 0 1 0-5zM3 9h4v12H3zM9 9h3.8v1.7h.05c.53-1 1.83-2.05 3.77-2.05 4.03 0 4.78 2.65 4.78 6.1V21h-4v-5.6c0-1.34-.03-3.06-1.86-3.06-1.87 0-2.15 1.46-2.15 2.96V21H9z"/></svg>';

  // Interface text in the page's language (<html lang>): English, or Portuguese for "pt*". Apps with a language
  // switcher dispatch an "app:lang" event on document and the button, dialog and credit are rebuilt.
  const TEXT = {
    en: { about: "About the author", close: "Close", builtOn: "{app} is built on PINNeAPPle, an open-source physics-AI toolkit.",
      email: "Email", linkedin: "Message me on LinkedIn", copy: "Copy email address", copied: "Copied ✓", builtBy: "Built by",
      subject: "{app} — contact", body: "Hi {first},\n\nI was using {app} ({origin}) and " },
    pt: { about: "Sobre o autor", close: "Fechar", builtOn: "{app} é construído sobre o PINNeAPPle, um kit de ferramentas de IA para física, de código aberto.",
      email: "E-mail", linkedin: "Fale comigo no LinkedIn", copy: "Copiar o e-mail", copied: "Copiado ✓", builtBy: "Feito por",
      subject: "{app} — contato", body: "Olá, {first},\n\nEu estava usando o {app} ({origin}) e " },
  };
  const fill = (s, v) => s.replace(/\{(\w+)\}/g, (m, k) => (v[k] ?? m));
  let parts = [];

  function build(a) {
    for (const el of parts) el.remove();
    parts = [];
    const lang = (document.documentElement.lang || "en").toLowerCase().startsWith("pt") ? "pt" : "en";
    const T = TEXT[lang], loc = { ...a, ...((a.i18n || {})[lang] || {}) };
    const app = document.title;
    const v = { app, first: a.name.split(" ")[0], origin: location.origin };
    const mailto = `mailto:${a.email}?subject=${encodeURIComponent(fill(T.subject, v))}&body=${encodeURIComponent(fill(T.body, v))}`;
    const initials = a.name.split(/\s+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();

    const dlg = document.createElement("dialog");
    dlg.className = "author";
    dlg.innerHTML = `<button class="x" aria-label="${esc(T.close)}">×</button>
      <div class="hd"><div class="av">${esc(initials)}</div>
        <div><div class="nm">${esc(a.name)}</div><div class="rl">${esc(loc.role)}</div></div></div>
      <div class="bd">${esc(loc.bio)}</div>
      <div class="ctx">${esc(fill(T.builtOn, v))}</div>
      <div class="acts">
        <a class="primary" href="${esc(mailto)}">${MAIL} ${esc(T.email)} ${esc(a.email)}</a>
        ${a.linkedin ? `<a class="li" href="${esc(a.linkedin)}" target="_blank" rel="noopener">${LI} ${esc(T.linkedin)}</a>` : ""}
        <button class="sec" data-copy>${esc(T.copy)}</button>
      </div>`;
    document.body.appendChild(dlg);
    parts.push(dlg);
    dlg.querySelector(".x").addEventListener("click", () => dlg.close());
    dlg.addEventListener("click", (e) => { if (e.target === dlg) dlg.close(); });
    const copy = dlg.querySelector("[data-copy]");
    copy.addEventListener("click", async () => {
      try { await navigator.clipboard.writeText(a.email); copy.textContent = T.copied; }
      catch { copy.textContent = a.email; }
      setTimeout(() => (copy.textContent = T.copy), 2000);
    });

    const btn = document.createElement("button");
    btn.className = "author-btn"; btn.type = "button";
    btn.innerHTML = `<span class="av-mini">👤</span> ${esc(T.about)}`;
    btn.addEventListener("click", () => dlg.showModal());
    const header = document.querySelector("header.top");
    if (header) { header.appendChild(btn); parts.push(btn); }

    const foot = document.querySelector("footer.foot");
    if (foot) {
      const c = document.createElement("div");
      c.className = "credit"; c.style.marginTop = "6px";
      c.innerHTML = `${esc(T.builtBy)} <a href="#" data-about>${esc(a.name)}</a> · <a href="${esc(mailto)}">${esc(a.email)}</a>` +
        (a.linkedin ? ` · <a href="${esc(a.linkedin)}" target="_blank" rel="noopener">LinkedIn</a>` : "");
      c.querySelector("[data-about]").addEventListener("click", (e) => { e.preventDefault(); dlg.showModal(); });
      foot.appendChild(c);
      parts.push(c);
    }
  }

  fetch("/shared/author.json").then((r) => r.json()).then((a) => {
    build(a);
    document.addEventListener("app:lang", () => build(a));
  }).catch(() => {});
})();
