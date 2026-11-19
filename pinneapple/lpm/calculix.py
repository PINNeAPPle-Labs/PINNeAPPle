"""CalculiX 3D C3D8 cantilever generation and nodal displacement extraction."""
from pathlib import Path
import re
import numpy as np
from .execution import checked

def build(root,d,c):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    nx,ny,nz=(int(c.numerical[k]) for k in ("nx","ny","nz"))
    if min(nx,ny,nz)<1:raise ValueError("Invalid element counts")
    L=float(d.get("length",1));W=float(d.get("width",.1));H=float(d.get("height",.1))
    E=float(d.get("young_modulus",2e11));nu=float(d.get("poisson",.3));F=float(d.get("force",-1000))
    if min(L,W,H,E)<=0 or not(-1<nu<.5):raise ValueError("Invalid physical parameters")
    def node(i,j,k):return 1+i+(nx+1)*(j+(ny+1)*k)
    lines=["*HEADING","Parametric linear elastic cantilever","*NODE"]
    xyz=[]
    for k in range(nz+1):
        for j in range(ny+1):
            for i in range(nx+1):
                pos=(i*L/nx,j*W/ny,k*H/nz);xyz.append(pos)
                lines.append(f"{node(i,j,k)}, {pos[0]:.14g}, {pos[1]:.14g}, {pos[2]:.14g}")
    lines.append("*ELEMENT, TYPE=C3D8, ELSET=SOLID")
    eid=0
    for k in range(nz):
        for j in range(ny):
            for i in range(nx):
                eid+=1
                ids=[node(i,j,k),node(i+1,j,k),node(i+1,j+1,k),node(i,j+1,k),node(i,j,k+1),node(i+1,j,k+1),node(i+1,j+1,k+1),node(i,j+1,k+1)]
                lines.append(f"{eid}, "+", ".join(map(str,ids)))
    fixed=[node(0,j,k) for k in range(nz+1) for j in range(ny+1)]
    tip=[node(nx,j,k) for k in range(nz+1) for j in range(ny+1)]
    lines+=["*NSET, NSET=FIXED",", ".join(map(str,fixed)),"*NSET, NSET=TIP",", ".join(map(str,tip)),"*NSET, NSET=ALL"]
    allnodes=list(range(1,len(xyz)+1))
    for off in range(0,len(allnodes),16):lines.append(", ".join(map(str,allnodes[off:off+16])))
    lines+=["*MATERIAL, NAME=ELASTIC","*ELASTIC",f"{E:.14g}, {nu:.14g}","*SOLID SECTION, ELSET=SOLID, MATERIAL=ELASTIC","","*STEP","*STATIC","*BOUNDARY","FIXED, 1, 3, 0","*CLOAD"]
    for idx in tip:lines.append(f"{idx}, 3, {F/len(tip):.14g}")
    lines+=["*NODE PRINT, NSET=ALL","U","*END STEP"]
    (root/"beam.inp").write_text("\n".join(lines)+"\n")
    return dict(length=L,width=W,height=H,young_modulus=E,poisson=nu,force=F,nx=nx,ny=ny,nz=nz,coords=np.asarray(xyz,dtype="float32"))

def parse_dat(path,count):
    text=Path(path).read_text(errors="replace")
    sections=list(re.finditer(r"displacements\s*\([^\n]*\)\s*for\s+set\s+ALL[^\n]*",text,re.I))
    if not sections:raise ValueError("Missing CalculiX ALL displacement section")
    chunk=text[sections[-1].end():];rows={}
    pat=re.compile(r"^\s*(\d+)\s+([-+\d.eEdD]+)\s+([-+\d.eEdD]+)\s+([-+\d.eEdD]+)\s*$",re.M)
    for m in pat.finditer(chunk):
        idx=int(m.group(1))
        if not 1<=idx<=count:continue
        rows[idx]=[float(v.replace("D","E").replace("d","e")) for v in m.groups()[1:]]
        if len(rows)==count:break
    if len(rows)!=count:raise ValueError(f"Only {len(rows)}/{count} displacements recovered")
    out=np.array([rows[i] for i in range(1,count+1)],dtype="float32")
    if not np.isfinite(out).all():raise ValueError("Nonfinite displacements")
    return out

def collect(root,meta):
    u=parse_dat(Path(root)/"beam.dat",len(meta["coords"]))
    xyz=meta["coords"].copy()/np.array([meta["length"],meta["width"],meta["height"]])
    n=len(xyz)
    x=np.column_stack([xyz,*[np.full(n,meta[k]) for k in ("length","width","height","young_modulus","poisson","force")]]).astype("float32")
    return dict(x=x,y=u,coords=xyz,meta={k:v for k,v in meta.items() if k!="coords"})

def run(root,d,c,execute=True):
    meta=build(root,d,c)
    if execute:
        checked(["ccx","-i","beam"],root,timeout=int(c.numerical.get("timeout",3600)))
        return collect(root,meta)
    meta.pop("coords")
    return meta
