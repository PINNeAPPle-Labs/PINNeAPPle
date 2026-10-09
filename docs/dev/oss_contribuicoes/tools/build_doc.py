import json, csv, collections, datetime
ALL=json.load(open("all_repos.json")); C=json.load(open("candidates.json"))
import os
OUT=os.environ.get("OSS_OUT",".").rstrip("/")+"/"
for r in C:
    r.setdefault("gfi",0); r.setdefault("hw",0); r.setdefault("prs",0); r.setdefault("contributing",False); r.setdefault("last_merge",None)
    r["lic_note"]="licenca propria (ver LICENSE)" if r["lic"]=="NOASSERTION" else r["lic"]
    r["easy"]=r["gfi"]+r["hw"]
csvp=OUT+"oss_repos.csv"
with open(csvp,"w",newline="") as f:
    w=csv.writer(f); w.writerow(["categoria","org","repo","estrelas","linguagem","licenca","issues_abertas","good_first_issue","help_wanted","prs_abertos","ultimo_merge","CONTRIBUTING","ultimo_push","url","descricao"])
    for r in sorted(C,key=lambda r:(r["cat"],r["org"],-r["stars"])):
        w.writerow([r["cat"],r["org"],r["name"],r["stars"],r["lang"],r["lic"],r["issues"],r["gfi"],r["hw"],r["prs"],r["last_merge"],"sim" if r["contributing"] else "nao",r["pushed"],r["url"],r["desc"]])
tot=collections.Counter(r["org"] for r in ALL)
cand=collections.Counter(r["org"] for r in C)
L=[]
A=L.append
A("# Repositorios open source de ciencia, tecnologia e pesquisa\n")
A(f"Gerado em {datetime.date.today()} a partir da API do GitHub. Lista completa em `oss_repos.csv` (mesma pasta).\n")
A("## Como a lista foi feita\n")
A(f"1. Organizacoes procuradas: {len(tot)} orgs confirmadas no GitHub (empresas, NASA, NOAA, laboratorios do DOE, agencias, comunidades de simulacao e ML cientifico). Total de repositorios publicos nelas: **{len(ALL):,}**.")
A("2. Filtros de 'contribuivel': nao e fork, nao esta arquivado, houve push nos ultimos 12 meses, tem licenca reconhecida (SPDX; `NOASSERTION` entra marcado como licenca propria, ex.: NASA Open Source Agreement), tem aba de issues e ao menos 1 issue aberta.")
A(f"3. Resultado: **{len(C):,} repositorios** candidatos. Para cada um: issues `good first issue`/`help wanted`, PRs abertos, data do ultimo PR mesclado (proxy de quanto os mantenedores respondem) e se existe CONTRIBUTING.\n")
A("## Quem nao tem codigo aberto utilizavel\n")
A("- **PhysicsX** (empresa, physicsx.ai): nao ha organizacao publica no GitHub. A conta `PhysicsX` no GitHub e de outra pessoa (projetos de Jetson/Qt) e nao tem relacao. Nada a listar.")
A("- **CIA**: nao ha organizacao oficial com codigo no GitHub (`cia` nao existe; `cia-foundation` e um projeto de TempleOS sem relacao). A agencia publica ferramentas em outros canais, nao no GitHub. Nada a listar.")
A("- **Luminary Cloud**: 3 repositorios publicos (`tutorials` e forks do Modulus e do Open MPI). Pouco a contribuir; entra na lista pelo que existe.\n")
A("## Resumo por organizacao\n")
A("| Org | Repos publicos | Candidatos | Com good-first/help-wanted |\n|---|---|---|---|")
for o,n in sorted(tot.items(),key=lambda x:-cand.get(x[0],0)):
    e=sum(1 for r in C if r["org"]==o and r["easy"]>0)
    A(f"| {o} | {n} | {cand.get(o,0)} | {e} |")
A("\n## Melhores alvos (good first issue / help wanted + CONTRIBUTING + PRs sendo mesclados)\n")
top=[r for r in C if r["easy"]>0 and r["contributing"] and r["last_merge"] and r["last_merge"]>="2026-04-01"]
top.sort(key=lambda r:(-r["easy"],-r["stars"]))
A(f"{len(top)} repositorios. Mostrando os 120 primeiros.\n")
A("| Repo | Estrelas | Lang | Licenca | Issues | GFI | HW | PRs abertos | Ultimo merge | Descricao |\n|---|---|---|---|---|---|---|---|---|---|")
for r in top[:120]:
    A(f"| [{r['name']}]({r['url']}) | {r['stars']} | {r['lang'] or ''} | {r['lic_note']} | {r['issues']} | {r['gfi']} | {r['hw']} | {r['prs']} | {r['last_merge']} | {r['desc'][:90].replace('|','/')} |")
A("\n## Por categoria (repos com 30+ estrelas ou com issues para iniciantes)\n")
for cat in ["Empresas","NASA","NOAA","Labs/agencias","Comunidade cientifica"]:
    rows=[r for r in C if r["cat"]==cat and (r["stars"]>=30 or r["easy"]>0)]
    rows.sort(key=lambda r:(r["org"],-r["stars"]))
    A(f"\n### {cat} ({len(rows)} de {sum(1 for r in C if r['cat']==cat)} candidatos)\n")
    A("| Org | Repo | ★ | Lang | Licenca | Issues | GFI | HW | CONTRIB. | Descricao |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        A(f"| {r['org']} | [{r['name'].split('/')[1]}]({r['url']}) | {r['stars']} | {r['lang'] or ''} | {r['lic_note']} | {r['issues']} | {r['gfi']} | {r['hw']} | {'sim' if r['contributing'] else 'nao'} | {r['desc'][:80].replace('|','/')} |")
A("\n## Ja contribuidos antes (clones em `forkes/`)\n")
A("NVIDIA/physicsnemo, PolymathicAI/the_well, jax-md/jax-md, Ceyron/exponax. Ver tambem a anotacao de PRs abertos na memoria do projeto.")
open(OUT+"OSS_REPOS.md","w").write("\n".join(L))
print(len(L),"linhas;",len(top),"melhores alvos;",len(C),"candidatos")
