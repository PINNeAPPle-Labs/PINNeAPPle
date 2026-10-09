import json, subprocess, base64, re, sys
C=json.load(open("candidates.json"))
sel=[r for r in C if r.get("gfi",0)+r.get("hw",0)>0 and r.get("contributing") and (r.get("last_merge") or "")>="2026-07-01"
     and r["lang"] in ("Python","Jupyter Notebook") and r["prs"]<60 and r["org"] not in ("NOAA-GFDL",)]
sel.sort(key=lambda r:-(r["gfi"]+r["hw"]))
FILES=["CONTRIBUTING.md",".github/CONTRIBUTING.md","docs/CONTRIBUTING.md","CONTRIBUTING.rst",".github/pull_request_template.md",".github/PULL_REQUEST_TEMPLATE.md","pull_request_template.md",".github/PULL_REQUEST_TEMPLATE/pull_request_template.md","AI_POLICY.md","AGENTS.md",".github/copilot-instructions.md","README.md"]
pat={"AI":r"\bLLMs?\b|large language model|AI[- ](generated|assisted|policy|disclos|tool)|generative AI|artificial intelligence|copilot|chatgpt|disclos\w+ (the )?use of",
     "CLA":r"\bCLA\b|contributor license agreement","DCO":r"signed-off-by|sign-off|\bDCO\b|Developer Certificate"}
out=[]
for r in sel[:45]:
    hit={}
    for f in FILES:
        p=subprocess.run(["gh","api",f"repos/{r['name']}/contents/{f}","--jq",".content"],capture_output=True,text=True)
        if p.returncode or not p.stdout.strip(): continue
        t=base64.b64decode(p.stdout.strip()).decode("utf8","ignore")
        for k,v in pat.items():
            if re.search(v,t,re.I): hit.setdefault(k,[]).append(f)
    out.append((r["name"],r["gfi"],r["hw"],r["prs"],{k:v[:2] for k,v in hit.items()}))
for o in out: print(o)
json.dump(out,open("screen2.json","w"))
