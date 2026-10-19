// Opens the PINNeAPPle Agent Workbench in a VS Code panel, starting the server for the open folder when needed.
const vscode = require("vscode");
const cp = require("child_process");

let server = null;
let panel = null;

function cfg() {
  const c = vscode.workspace.getConfiguration("pinneappleWorkbench");
  return { python: c.get("python"), port: c.get("port"), extra: c.get("extraArgs") || [] };
}

function folder() {
  const f = vscode.workspace.workspaceFolders;
  return f && f.length ? f[0].uri.fsPath : process.env.HOME;
}

async function up(port) {
  try { return (await fetch(`http://127.0.0.1:${port}/api/info`)).ok; } catch (e) { return false; }
}

async function ensureServer(out) {
  const { python, port, extra } = cfg();
  if (await up(port)) return port;
  server = cp.spawn(python, ["-m", "pinneapple_llm.workbench", "serve", "--workspace", folder(), "--port",
                             String(port), "--no-browser", ...extra], { cwd: folder() });
  server.stdout.on("data", (d) => out.append(d.toString()));
  server.stderr.on("data", (d) => out.append(d.toString()));
  for (let i = 0; i < 60; i++) {
    if (await up(port)) return port;
    await new Promise((r) => setTimeout(r, 500));
  }
  throw new Error("the workbench did not start; see the 'PINNeAPPle Workbench' output panel");
}

async function open(context, out, sessionId) {
  const port = await ensureServer(out);
  const uri = await vscode.env.asExternalUri(vscode.Uri.parse(`http://127.0.0.1:${port}/`));
  const src = uri.toString(true) + (sessionId ? `?session=${sessionId}` : "");
  if (!panel) {
    panel = vscode.window.createWebviewPanel("pinneappleWorkbench", "Agent Workbench", vscode.ViewColumn.Beside,
                                             { enableScripts: true, retainContextWhenHidden: true });
    panel.onDidDispose(() => (panel = null), null, context.subscriptions);
  }
  panel.webview.html = `<!doctype html><html><head><meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; frame-src ${uri.scheme}://${uri.authority}; style-src 'unsafe-inline';">
    <style>html,body,iframe{margin:0;padding:0;border:0;width:100%;height:100vh}</style></head>
    <body><iframe src="${src}"></iframe></body></html>`;
  panel.reveal(vscode.ViewColumn.Beside);
}

function activate(context) {
  const out = vscode.window.createOutputChannel("PINNeAPPle Workbench");
  context.subscriptions.push(vscode.commands.registerCommand("pinneappleWorkbench.open", () =>
    open(context, out).catch((e) => vscode.window.showErrorMessage(e.message))));
  context.subscriptions.push(vscode.commands.registerCommand("pinneappleWorkbench.askSelection", async () => {
    const ed = vscode.window.activeTextEditor;
    if (!ed) return;
    const question = await vscode.window.showInputBox({ prompt: "What should the agents do with this code?" });
    if (!question) return;
    try {
      const port = await ensureServer(out);
      const rel = vscode.workspace.asRelativePath(ed.document.uri);
      const sel = ed.selection;
      const text = `${question}\n\nFile: ${rel}, lines ${sel.start.line + 1}-${sel.end.line + 1}:\n\`\`\`\n` +
                   `${ed.document.getText(sel)}\n\`\`\``;
      const base = `http://127.0.0.1:${port}/api/sessions`;
      const s = await (await fetch(base, { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify({ workspace: folder() }) })).json();
      await fetch(`${base}/${s.id}/messages`, { method: "POST", headers: { "Content-Type": "application/json" },
                                                body: JSON.stringify({ text }) });
      await open(context, out, s.id);
    } catch (e) { vscode.window.showErrorMessage(e.message); }
  }));
}

function deactivate() { if (server) server.kill(); }

module.exports = { activate, deactivate };
