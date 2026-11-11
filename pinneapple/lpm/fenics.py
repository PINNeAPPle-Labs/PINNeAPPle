"""Legacy dolfin/FEniCS 2019 cantilever adapter, executed in a subprocess."""
import json
from pathlib import Path
import numpy as np
from .execution import checked

SCRIPT = r'''
import sys,json
from pathlib import Path
import numpy as np
from dolfin import *
root=Path(sys.argv[1]);p=json.loads((root/"params.json").read_text())
L,W,H,E,nu,F=(float(p[k]) for k in ("length","width","height","young_modulus","poisson","force"))
nx,ny,nz=(int(p[k]) for k in ("nx","ny","nz"))
mesh=BoxMesh(Point(0,0,0),Point(L,W,H),nx,ny,nz)
V=VectorFunctionSpace(mesh,"P",1)
bc=DirichletBC(V,Constant((0.,0.,0.)),"near(x[0],0.0) && on_boundary")
mu=E/(2*(1+nu));lam=E*nu/((1+nu)*(1-2*nu))
def sigma(u):return 2*mu*sym(grad(u))+lam*tr(sym(grad(u)))*Identity(3)
u=TrialFunction(V);v=TestFunction(V)
facets=MeshFunction("size_t",mesh,mesh.topology().dim()-1,0)
class Tip(SubDomain):
 def inside(self,x,on_boundary):return on_boundary and near(x[0],L)
Tip().mark(facets,1)
ds=Measure("ds",domain=mesh,subdomain_data=facets)
traction=Constant((0.,0.,F/(W*H)))
uh=Function(V)
solve(inner(sigma(u),sym(grad(v)))*dx==dot(traction,v)*ds(1),uh,bc)
coords=[];values=[]
for k in range(nz+1):
 for j in range(ny+1):
  for i in range(nx+1):
   xyz=(i*L/nx,j*W/ny,k*H/nz)
   coords.append(xyz);values.append(uh(Point(*xyz)))
np.savez_compressed(str(root/"fenics_result.npz"),coords=np.asarray(coords),u=np.asarray(values))
'''

def run(root,d,c,execute=True):
    root=Path(root);root.mkdir(parents=True,exist_ok=True)
    nx,ny,nz=(int(c.numerical[k]) for k in ("nx","ny","nz"))
    p={**{k:float(d.get(k,v)) for k,v in {"length":1,"width":.1,"height":.1,"young_modulus":2e11,"poisson":.3,"force":-1000}.items()},"nx":nx,"ny":ny,"nz":nz}
    if min(p[k] for k in ("length","width","height","young_modulus"))<=0 or not(-1<p["poisson"]<.5):raise ValueError("Invalid physical parameters")
    (root/"params.json").write_text(json.dumps(p,indent=2))
    (root/"solve_fenics.py").write_text(SCRIPT)
    if not execute:return p
    checked([c.numerical.get("python","python3"),"solve_fenics.py",str(root.resolve())],root,timeout=int(c.numerical.get("timeout",3600)))
    with np.load(root/"fenics_result.npz") as arr:
        coords=arr["coords"].copy();u=arr["u"].copy()
    xyz=coords/np.array([p["length"],p["width"],p["height"]]);n=len(xyz)
    x=np.column_stack([xyz,*[np.full(n,p[k]) for k in ("length","width","height","young_modulus","poisson","force")]]).astype("float32")
    return dict(x=x,y=np.asarray(u,dtype="float32"),coords=xyz.astype("float32"),meta=p)
