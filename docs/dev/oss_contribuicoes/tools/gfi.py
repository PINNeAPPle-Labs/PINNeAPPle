import subprocess, json, time
ORGS="nasa NASA-IMPACT NASA-AMMOS nasa-jpl NOAA-EMC NOAA-OWP NOAA-PSL NOAA-GSL noaa-oar-arl NOAA-ORR-ERD noaa-ocs-modeling ufs-community E3SM-Project NCAR Unidata lanl sandialabs llnl idaholab ORNL NatLabRockies pnnl esa ESA-PhiLab ecmwf ecmwf-lab ACCESS-NRI precice firedrakeproject FEniCS dealii mfem tum-pbs deepmodeling PolymathicAI pangeo-data openmm materialsproject su2code OPM jax-ml ansys google-research google-deepmind NVIDIA-Omniverse".split()
out=[]
for o in ORGS:
    for lab in ("good first issue","help wanted"):
        p=subprocess.run(["gh","search","issues","--owner",o,"--label",lab,"--state","open","--no-assignee","--limit","60","--json","repository,number,title,commentsCount,createdAt,url,labels"],capture_output=True,text=True)
        if p.returncode: print("ERR",o,lab,p.stderr[:80]); time.sleep(10); continue
        for i in json.loads(p.stdout):
            out.append({"repo":i["repository"]["nameWithOwner"],"n":i["number"],"title":i["title"],"c":i["commentsCount"],"created":i["createdAt"][:10],"labels":[l["name"] for l in i["labels"]],"url":i["url"]})
    time.sleep(2.2)
seen={};
for r in out: seen[r["url"]]=r
json.dump(list(seen.values()),open("gfi_issues.json","w"))
print(len(seen),"issues")
