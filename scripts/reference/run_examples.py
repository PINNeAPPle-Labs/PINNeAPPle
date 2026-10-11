"""Run the curated examples of the module reference: each in a fresh interpreter, with matplotlib on Agg; keep its
printed output and save the figures it leaves open. ``python scripts/reference/run_examples.py [name ...]``."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from content import PACKAGES  # noqa: E402

WRAP = r'''
import os, sys, warnings
warnings.filterwarnings("ignore")
os.environ["MPLBACKEND"] = "Agg"
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams.update({"figure.dpi": 110, "savefig.bbox": "tight", "font.size": 9})
sys.path.insert(0, ROOT_)
exec(compile(open(CODE_).read(), "example", "exec"))
for k, n in enumerate(plt.get_fignums()):
    plt.figure(n).savefig(FIG_.format(k), dpi=110, facecolor="white")
'''


def run_one(key: str, code: str, out_dir: str, timeout: int = 600) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "example.py")
        open(src, "w").write(code)
        fig = os.path.join(out_dir, key + "_{}.png")
        wrap = os.path.join(tmp, "wrap.py")
        open(wrap, "w").write(WRAP.replace("ROOT_", repr(ROOT)).replace("CODE_", repr(src)).replace("FIG_", repr(fig)))
        t0 = time.time()
        p = subprocess.run([sys.executable, wrap], cwd=tmp, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, "PYTHONHASHSEED": "0", "CUDA_VISIBLE_DEVICES": ""})
        figs = sorted(f for f in os.listdir(out_dir) if f.startswith(key + "_") and f.endswith(".png"))
        out = p.stdout.strip()
        if len(out) > 2500:
            out = out[:2500] + "\n..."
        return {"ok": p.returncode == 0, "stdout": out, "stderr": p.stderr.strip()[-3000:],
                "figures": figs, "seconds": round(time.time() - t0, 1)}


def main(names: list[str]) -> None:
    out_dir = os.path.join(ROOT, "docs", "org", "reference", "figures")
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(HERE, "examples_output.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for pkg in PACKAGES:
        for k, ex in enumerate(pkg.get("examples", [])):
            key = f"{pkg['name']}__{k}"
            if names and pkg["name"] not in names and key not in names:
                continue
            for f in os.listdir(out_dir):
                if f.startswith(key + "_"):
                    os.remove(os.path.join(out_dir, f))
            r = run_one(key, ex["code"], out_dir)
            cache[key] = {**r, "code_hash": hash_code(ex["code"])}
            status = "ok " if r["ok"] else "FAIL"
            print(f"{status} {key:40s} {r['seconds']:6.1f}s figs={len(r['figures'])}")
            if not r["ok"]:
                print("   ", r["stderr"].splitlines()[-1] if r["stderr"] else "")
    json.dump(cache, open(cache_path, "w"), indent=1, sort_keys=True)


def hash_code(code: str) -> str:
    import hashlib
    return hashlib.sha256(code.encode()).hexdigest()[:16]


if __name__ == "__main__":
    main(sys.argv[1:])
