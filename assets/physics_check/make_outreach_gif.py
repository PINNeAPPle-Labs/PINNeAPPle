"""Render docs/assets/physics_check/pr_blocked.gif: a pull request blocked by a physics violation.

The text in every frame is the real output of ``pinneapple check`` on the failing cavity fixture
(``examples/physics_check/fail``, lid speed typed as 2 m/s), run in a scratch repository laid out like a user's.
The pull-request chrome around it is a simplified illustration, not a screenshot of a hosting service.

    pip install playwright pillow && python docs/assets/physics_check/make_outreach_gif.py
"""
from __future__ import annotations

import html
import io
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
W, H = 960, 560

CONFIG = """[[result]]
name = "cavity"
path = "cases/cavity"
physics = "incompressible-flow"
parameters = { nu = "0.01 m2/s" }
reference = { path = "reference/cavity.npz", fields = ["U"] }
[result.bc.movingWall]
U = [1, 0, 0]
[result.bc.fixedWalls]
U = [0, 0, 0]
"""


def real_output(work: Path) -> tuple[str, dict]:
    (work / ".git").mkdir(parents=True)
    shutil.copytree(ROOT / "examples/physics_check/fail/cases/cavity", work / "cases/cavity")
    (work / "reference").mkdir()
    shutil.copy(ROOT / "examples/physics_check/fail/reference/cavity.npz", work / "reference")
    (work / "pinneapple-check.toml").write_text(CONFIG)
    cmd = [sys.executable, "-m", "pinneapple_check", "--color", "never", "--json", str(work / "r.json")]
    text = subprocess.run(cmd, cwd=work, capture_output=True, text=True).stdout
    return text, json.loads((work / "r.json").read_text())


CSS = """
body{margin:0;font:14px -apple-system,Segoe UI,Helvetica,Arial,sans-serif;background:#f6f8fa;color:#1f2328}
.wrap{width:%dpx;height:%dpx;box-sizing:border-box;padding:22px 26px;overflow:hidden}
h1{font-size:20px;font-weight:600;margin:0 0 4px} .sub{color:#59636e;margin-bottom:14px}
.pill{display:inline-block;background:#8250df;color:#fff;border-radius:12px;padding:2px 10px;font-size:12px;margin-right:6px}
.box{background:#fff;border:1px solid #d1d9e0;border-radius:8px;margin-bottom:12px;overflow:hidden}
.row{display:flex;align-items:center;gap:10px;padding:9px 14px;border-top:1px solid #eef1f4}
.row:first-child{border-top:none} .x{color:#d1242f;font-weight:700} .ok{color:#1a7f37;font-weight:700}
.muted{color:#59636e} .bad{background:#ffebe9;border-color:#ffcecb}
.term{background:#0d1117;color:#e6edf3;font:13px/1.45 ui-monospace,SFMono-Regular,Menlo,monospace;
      border-radius:8px;padding:16px;white-space:pre-wrap;height:%dpx;box-sizing:border-box}
.r{color:#ff7b72;font-weight:700} .c{color:#79c0ff} .d{color:#8b949e} .b{font-weight:700}
.code{font:12.5px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace;white-space:pre}
.ln{display:inline-block;width:44px;color:#8c959f;text-align:right;padding-right:10px}
.add{background:#dafbe1} .del{background:#ffebe9} .ann{margin:4px 14px 8px 58px;border:1px solid #ffcecb;
      border-left:4px solid #d1242f;border-radius:6px;background:#fff;padding:10px 12px;font:13px/1.45 sans-serif;
      white-space:normal}
table{border-collapse:collapse;width:100%%} td,th{border:1px solid #d1d9e0;padding:6px 8px;text-align:left;font-size:13px}
th{background:#f6f8fa} .cap{position:absolute;bottom:10px;right:16px;color:#8c959f;font-size:11px}
""" % (W, H, H - 44)


def page(body: str) -> str:
    return f"<html><head><style>{CSS}</style></head><body><div class='wrap'>{body}</div>" \
           f"<div class='cap'>pinneapple check · real output, illustrative pull-request view</div></body></html>"


def header() -> str:
    return ("<h1>Raise lid speed in cavity study <span class='muted'>#42</span></h1>"
            "<div class='sub'><span class='pill' style='background:#1a7f37'>Open</span>"
            "wants to merge 1 commit into <code>main</code></div>")


def frames(text: str, rep: dict) -> list[str]:
    e = html.escape
    term = e(text)
    term = term.replace("[error]", "[<span class='r'>error</span>]")
    for rid in ("PC201", "PC701"):
        term = term.replace(f" {rid} ", f" <span class='r'>{rid}</span> ")
    term = term.replace("Found 2 errors", "<span class='r'>Found 2 errors</span>")
    f1 = page(f"<div class='term'><span class='d'>$</span> <span class='b'>pinneapple check</span>\n{term}"
              f"<span class='d'>$ echo $?</span>\n1</div>")
    bc = next(v for v in rep["violations"] if v["rule"] == "PC201")
    ref = next(v for v in rep["violations"] if v["rule"] == "PC701")
    checks = (header() + "<div class='box bad'>"
              "<div class='row'><span class='x'>✗</span><b>Some checks were not successful</b>"
              "<span class='muted'>1 failing and 2 successful checks</span></div>"
              f"<div class='row'><span class='x'>✗</span>Physics check / physics <span class='muted'>— "
              f"{len(rep['violations'])} physics violations (PC201, PC701)</span></div>"
              "<div class='row'><span class='ok'>✓</span>Tests / test <span class='muted'>— passed</span></div>"
              "<div class='row'><span class='ok'>✓</span>Docs / build <span class='muted'>— passed</span></div>"
              "</div><div class='box'><div class='row'><span class='x'>⊘</span><b>Merging is blocked</b>"
              "<span class='muted'>The required check “Physics check” failed.</span></div></div>")
    f2 = page(checks)
    U = (ROOT / "examples/physics_check/fail/cases/cavity/5/U").read_text().splitlines()
    line = bc["location"]["line"]
    ann = (f"<div class='ann'><b class='x'>✗ Check failure</b> &nbsp;<b>PC201</b> Boundary values differ "
           f"from the declared boundary condition<br>{e(bc['message'])}<br>"
           f"<span class='muted'>Fix: {e(bc['fix'])}</span></div>")
    rows = []
    for n in range(line - 4, line + 3):
        t = U[n - 1]
        if "uniform (2 0 0)" in t:
            old = t.replace("(2 0 0)", "(1 0 0)")
            rows.append(f"<div class='del'><span class='ln'>{n}</span>- {e(old)}</div>")
            rows.append(f"<div class='add'><span class='ln'>{n}</span>+ {e(t)}</div>")
            if n == line:
                rows.append(ann)
        else:
            rows.append(f"<div><span class='ln'>{n}</span>&nbsp; {e(t)}</div>")
        if n == line and "uniform (2 0 0)" not in t:
            rows.append(f"<div class='ann'><b class='x'>✗ Check failure</b> &nbsp;<b>PC201</b> Boundary values differ "
                        f"from the declared boundary condition<br>{e(bc['message'])}<br>"
                        f"<span class='muted'>Fix: {e(bc['fix'])}</span></div>")
    diff = (header() + "<div class='box'><div class='row'><b>cases/cavity/5/U</b>"
            "<span class='muted'>+1 −1</span></div><div class='code'>" + "".join(rows) + "</div></div>")
    f3 = page(diff)
    summ = (header() + "<div class='box'><div class='row'><b>❌ Physics check failed</b>"
            f"<span class='muted'>{rep['summary']['passed']} checks passed, 2 errors</span></div>"
            "<div style='padding:10px 14px'><table><tr><th>Target</th><th>Kind</th><th>Errors</th><th>Data</th>"
            "<th>Model</th><th>Physics</th><th>Benchmark</th></tr>"
            "<tr><td><code>cavity</code></td><td>OpenFOAM</td><td>2</td><td>✅</td><td>▫️</td><td>❌</td><td>❌</td></tr>"
            "</table><br><table><tr><th>Rule</th><th>Location</th><th>Message</th></tr>"
            f"<tr><td>PC201 ❌</td><td><code>cases/cavity/5/U:{line}</code></td><td>{e(bc['message'])}</td></tr>"
            f"<tr><td>PC701 ❌</td><td><code>pinneapple-check.toml:{ref['location']['line']}</code></td>"
            f"<td>{e(ref['message'])}</td></tr></table></div></div>")
    f4 = page(summ)
    return [f1, f2, f3, f4]


def main() -> None:
    from PIL import Image
    from playwright.sync_api import sync_playwright
    with tempfile.TemporaryDirectory() as d:
        text, rep = real_output(Path(d) / "repo")
    shots = []
    exe = "/opt/pw-browsers/chromium-1194/chrome-linux/chrome"
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=exe if Path(exe).exists() else None)
        pg = b.new_page(viewport={"width": W, "height": H}, device_scale_factor=1)
        for f in frames(text, rep):
            pg.set_content(f)
            shots.append(Image.open(io.BytesIO(pg.screenshot())).convert("RGB"))
        b.close()
    shots[2].save(OUT / "pr_annotation.png")
    pal = [s.convert("P", palette=Image.ADAPTIVE, colors=128) for s in shots]
    pal[0].save(OUT / "pr_blocked.gif", save_all=True, append_images=pal[1:], duration=[3500, 2500, 4500, 3500],
                loop=0, optimize=True)
    print("wrote", OUT / "pr_blocked.gif", "and", OUT / "pr_annotation.png")


if __name__ == "__main__":
    main()
