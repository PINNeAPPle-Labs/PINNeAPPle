import json, subprocess, time
R=json.load(open("all_repos.json"))
C=[r for r in R if not r["fork"] and not r["archived"] and r["pushed"]>="2025-10-09" and r["lic"] and r["has_issues"] and r["issues"]>0]
print(len(C),"candidatos (inclui licenca propria)")
GF='["good first issue","good-first-issue","Good first issue","good first issues","beginner","first-timers-only","easy","starter"]'
HW='["help wanted","help-wanted","Help wanted","contributions welcome","PRs welcome"]'
def q(batch):
    parts=[]
    for i,r in enumerate(batch):
        o,n=r["name"].split("/")
        parts.append(f'''r{i}: repository(owner:"{o}",name:"{n}"){{ nameWithOwner
          gf: issues(states:OPEN,labels:{GF}){{totalCount}} hw: issues(states:OPEN,labels:{HW}){{totalCount}}
          prs: pullRequests(states:OPEN){{totalCount}}
          merged: pullRequests(states:MERGED,first:1,orderBy:{{field:UPDATED_AT,direction:DESC}}){{nodes{{mergedAt}}}}
          c1: object(expression:"HEAD:CONTRIBUTING.md"){{__typename}} c2: object(expression:"HEAD:.github/CONTRIBUTING.md"){{__typename}}
          c3: object(expression:"HEAD:docs/CONTRIBUTING.md"){{__typename}} c4: object(expression:"HEAD:CONTRIBUTING.rst"){{__typename}}
          cod: object(expression:"HEAD:CODE_OF_CONDUCT.md"){{__typename}} }}''')
    return "query{"+"\n".join(parts)+"}"
by={r["name"]:r for r in C}
for s in range(0,len(C),20):
    batch=C[s:s+20]
    for attempt in range(3):
        p=subprocess.run(["gh","api","graphql","-f",f"query={q(batch)}"],capture_output=True,text=True)
        if p.returncode==0: break
        time.sleep(5)
    if p.returncode!=0:
        try: d=json.loads(p.stdout)
        except Exception: print("fail batch",s,p.stderr[:120]); continue
    else: d=json.loads(p.stdout)
    for i,r in enumerate(batch):
        x=(d.get("data") or {}).get(f"r{i}")
        if not x: continue
        by[r["name"]].update({"gfi":x["gf"]["totalCount"],"hw":x["hw"]["totalCount"],"prs":x["prs"]["totalCount"],
            "last_merge":(x["merged"]["nodes"][0]["mergedAt"][:10] if x["merged"]["nodes"] else None),
            "contributing":any(x[k] for k in("c1","c2","c3","c4")),"coc":bool(x["cod"])})
    if (s//20)%10==0: print(s,flush=True)
json.dump(C,open("candidates.json","w"))
print(sum('gfi' in r for r in C),"enriquecidos")
