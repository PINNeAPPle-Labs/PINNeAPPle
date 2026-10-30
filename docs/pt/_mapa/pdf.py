"""Gera o PDF do mapa completo (todos os volumes num arquivo só) e conta páginas e seções.

Caminho: Markdown -> HTML (pandoc) -> PDF (Chromium via Playwright do Node). O diagrama Mermaid é desenhado
no navegador com ``mermaid.min.js`` local (``npm install mermaid@10.9.1`` numa pasta qualquer).

Uso:
    python docs/pt/_mapa/pdf.py --mermaid /caminho/node_modules/mermaid/dist/mermaid.min.js
Saída: docs/pt/mapa_completo.pdf (e uma linha com páginas e seções).
"""
from __future__ import annotations

import argparse
import html
import os
import re
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
MAIN = ROOT / "docs" / "pt" / "mapa_completo.md"
VOLS = sorted((ROOT / "docs" / "pt" / "mapa_completo").glob("v*.md"))
OUT = ROOT / "docs" / "pt" / "mapa_completo.pdf"

CSS = """
@page { size: A4; margin: 14mm 12mm 16mm 12mm; }
html, body { max-width: none !important; margin: 0 !important; padding: 0 !important; }
body { font-family: 'DejaVu Sans', Arial, sans-serif; font-size: 9.5pt; line-height: 1.35; color: #111; }
h1 { font-size: 20pt; page-break-before: always; border-bottom: 2px solid #c8a200; padding-bottom: 4px; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 15pt; margin-top: 18px; border-bottom: 1px solid #ccc; }
h3 { font-size: 12pt; margin-top: 14px; }
h4 { font-size: 10pt; margin: 12px 0 4px; background: #f4f1e6; padding: 3px 6px; }
table { border-collapse: collapse; width: 100%; font-size: 7.6pt; margin: 4px 0 10px; }
th, td { border: 1px solid #ccc; padding: 2px 4px; vertical-align: top; overflow-wrap: anywhere; }
th { background: #eee; }
tr { page-break-inside: avoid; }
code { font-family: 'DejaVu Sans Mono', monospace; font-size: 0.92em; overflow-wrap: anywhere; }
pre { background: #f6f6f6; padding: 6px; font-size: 7.8pt; white-space: pre-wrap; overflow-wrap: anywhere; }
small { color: #555; }
blockquote { color: #555; border-left: 3px solid #ccc; margin: 6px 0; padding-left: 8px; }
ul { margin: 2px 0 8px; }
nav#TOC { font-size: 8.5pt; }
nav#TOC ul { list-style: none; padding-left: 14px; }
.mermaid { text-align: center; margin: 10px 0; }
"""

NODE = r"""
const { chromium } = require('playwright');
(async () => {
  const [src, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  await page.goto('file://' + src, { waitUntil: 'load', timeout: 0 });
  await page.waitForFunction(() => window.__mermaidDone === true, null, { timeout: 120000 });
  await page.pdf({ path: out, format: 'A4', printBackground: true, displayHeaderFooter: true,
    headerTemplate: '<div></div>',
    footerTemplate: '<div style="font-size:7pt;width:100%;text-align:center;color:#777">Mapa completo do PINNeAPPle · <span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: { top: '14mm', bottom: '16mm', left: '12mm', right: '12mm' }, timeout: 0 });
  await browser.close();
})();
"""


def count_headings(md: str) -> int:
    """Títulos Markdown fora de blocos de código."""
    n, fence = 0, False
    for ln in md.splitlines():
        if ln.startswith("```"):
            fence = not fence
        elif not fence and re.match(r"#{1,6} ", ln):
            n += 1
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mermaid", required=True)
    ap.add_argument("--node-path", default="", help="NODE_PATH para achar o playwright")
    args = ap.parse_args()
    # remove os links relativos entre volumes (no PDF tudo está num arquivo só)
    parts = []
    for f in [MAIN] + VOLS:
        txt = f.read_text(encoding="utf-8")
        txt = re.sub(r"\]\((?:\.\./)?(?:mapa_completo/)?(v\d\d_[a-z_]+|mapa_completo)\.md\)", "](#)", txt)
        parts.append(txt)
    with tempfile.TemporaryDirectory() as tmp:
        md = Path(tmp) / "all.md"
        md.write_text("\n\n".join(parts), encoding="utf-8")
        htmlf = Path(tmp) / "all.html"
        subprocess.run(["pandoc", str(md), "-f", "gfm+raw_html", "-t", "html5", "-s", "--toc", "--toc-depth=2",
                        "--metadata", "title=Mapa completo do PINNeAPPle", "--metadata", "lang=pt-BR",
                        "-o", str(htmlf)], check=True)
        page = htmlf.read_text(encoding="utf-8")
        page = re.sub(r'<pre class="mermaid"><code>(.*?)</code></pre>',
                      lambda m: f'<div class="mermaid">{m.group(1)}</div>', page, flags=re.S)
        mer = Path(args.mermaid).read_text(encoding="utf-8")
        inject = (f"<style>{CSS}</style><script>{mer}</script><script>"
                  "window.addEventListener('load', async () => {"
                  "mermaid.initialize({startOnLoad:false, theme:'neutral', flowchart:{htmlLabels:true}});"
                  "try { await mermaid.run({querySelector: '.mermaid'}); } catch (e) { console.log(e); }"
                  "window.__mermaidDone = true; });</script>")
        page = page.replace("</head>", inject + "</head>", 1)
        # o título "Mapa completo" do pandoc já é o h1 do volume 0
        page = re.sub(r'<header id="title-block-header">.*?</header>', "", page, flags=re.S)
        htmlf.write_text(page, encoding="utf-8")
        js = Path(tmp) / "print.js"
        js.write_text(NODE)
        env = dict(os.environ)
        if args.node_path:
            env["NODE_PATH"] = args.node_path
        subprocess.run(["node", str(js), str(htmlf), str(OUT)], check=True, env=env)
    info = subprocess.run(["pdfinfo", str(OUT)], capture_output=True, text=True).stdout
    pages = re.search(r"Pages:\s+(\d+)", info).group(1)
    heads = sum(count_headings(p) for p in parts)
    print(f"{OUT.relative_to(ROOT)}: {pages} páginas, {heads} seções, {OUT.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
