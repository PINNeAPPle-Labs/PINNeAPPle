import json, subprocess, sys
ORGS = {
 "Empresas": ["NVIDIA","NVlabs","NVIDIA-Omniverse","luminarycloud","google-deepmind","google-research","ansys","altairengineering","Autodesk","rescale"],
 "NASA": ["nasa","NASA-AMMOS","NASA-IMPACT","nasa-gcn","NASA-SW-VnV","nasa-jpl"],
 "NOAA": ["NOAA-EMC","NOAA-GFDL","NOAA-OWP","NOAA-PSL","NOAA-GSL","noaa-oar-arl","NOAA-ORR-ERD","noaa-ocs-modeling","ufs-community"],
 "Labs/agencias": ["E3SM-Project","NCAR","Unidata","lanl","sandialabs","llnl","idaholab","ORNL","NatLabRockies","pnnl","esa","ESA-PhiLab","ecmwf","ecmwf-lab","ACCESS-NRI"],
 "Comunidade cientifica": ["su2code","precice","firedrakeproject","FEniCS","dealii","mfem","KratosMultiphysics","NGSolve","OPM","tum-pbs","deepmodeling","PolymathicAI","jax-ml","SciML","pangeo-data","openmm","materialsproject","opencae"],
}
out = []
for cat, orgs in ORGS.items():
    for o in orgs:
        p = subprocess.run(["gh","api",f"orgs/{o}/repos?type=public&per_page=100","--paginate"],capture_output=True,text=True)
        if p.returncode: print("ERR",o,p.stderr[:100]); continue
        txt = p.stdout.replace("][", ",")
        repos = json.loads(txt)
        for r in repos:
            out.append({"cat":cat,"org":o,"name":r["full_name"],"stars":r["stargazers_count"],"lang":r["language"],"lic":(r["license"] or {}).get("spdx_id"),
                        "pushed":r["pushed_at"][:10],"issues":r["open_issues_count"],"archived":r["archived"],"fork":r["fork"],"has_issues":r["has_issues"],
                        "desc":r["description"] or "","topics":r.get("topics",[]),"url":r["html_url"],"branch":r["default_branch"]})
        print(o,len(repos),flush=True)
json.dump(out,open("all_repos.json","w"))
print("total",len(out))
