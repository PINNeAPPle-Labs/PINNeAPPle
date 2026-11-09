"""Native OpenFOAM lid-driven cavity adapter: generate, execute, collect."""
from pathlib import Path
import re
import numpy as np
from .execution import checked

HEADER="FoamFile\n{ version 2.0; format ascii; class %s; object %s; }\n"

def _write(path,text):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text)

def case(root,d,c):
    root=Path(root);nx=int(c.numerical["nx"]);ny=int(c.numerical["ny"])
    if min(nx,ny)<4:raise ValueError("OpenFOAM cavity requires nx,ny >= 4")
    lx=float(d.get("width",1));ly=float(d.get("height",1));u=float(d.get("lid_speed",1));nu=float(d.get("nu",0.01))
    if min(lx,ly,nu)<=0 or u<0:raise ValueError("Invalid cavity parameters")
    dz=min(lx/nx,ly/ny)
    vertices=[(0,0,0),(lx,0,0),(lx,ly,0),(0,ly,0),(0,0,dz),(lx,0,dz),(lx,ly,dz),(0,ly,dz)]
    v="\n".join("(%g %g %g)"%p for p in vertices)
    _write(root/"system/blockMeshDict",HEADER%("dictionary","blockMeshDict")+f"""convertToMeters 1;
vertices ({v});
blocks (hex (0 1 2 3 4 5 6 7) ({nx} {ny} 1) simpleGrading (1 1 1));
edges ();
boundary (
 movingWall {{ type wall; faces ((3 7 6 2)); }}
 fixedWalls {{ type wall; faces ((0 4 7 3) (1 2 6 5) (0 1 5 4)); }}
 frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
mergePatchPairs ();
""")
    end=float(c.numerical.get("end_time",1.0));dt=float(c.numerical.get("delta_t",0.002))
    if end<=0 or dt<=0:raise ValueError("Time settings must be positive")
    _write(root/"system/controlDict",HEADER%("dictionary","controlDict")+f"""application icoFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime {end};
deltaT {dt};
writeControl adjustableRunTime;
writeInterval {end};
writeFormat ascii;
writePrecision 12;
runTimeModifiable false;
""")
    _write(root/"system/fvSchemes",HEADER%("dictionary","fvSchemes")+"""ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) Gauss linearUpwind grad(U); }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
    _write(root/"system/fvSolution",HEADER%("dictionary","fvSolution")+"""solvers {
 p { solver PCG; preconditioner DIC; tolerance 1e-08; relTol 0.05; }
 pFinal { $p; relTol 0; }
 U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-08; relTol 0; }
}
PISO { nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }
""")
    _write(root/"constant/transportProperties",HEADER%("dictionary","transportProperties")+f"nu [0 2 -1 0 0 0 0] {nu:.12g};\n")
    _write(root/"0/U",HEADER%("volVectorField","U")+f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField {{ movingWall {{ type fixedValue; value uniform ({u} 0 0); }} fixedWalls {{ type noSlip; }} frontAndBack {{ type empty; }} }}
""")
    _write(root/"0/p",HEADER%("volScalarField","p")+"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField { movingWall { type zeroGradient; } fixedWalls { type zeroGradient; } frontAndBack { type empty; } }
""")
    return dict(nx=nx,ny=ny,width=lx,height=ly,lid_speed=u,nu=nu,re=u*lx/nu)

def _field(path,n,dim):
    """Use PINNeAPPLe's native ASCII/binary OpenFOAM field parser."""
    from pinneapple_simulation.external_solvers.openfoam.field_reader import _read_internal_field
    values=_read_internal_field(str(path),n_cells_hint=n)
    out=values.detach().cpu().numpy().reshape(n,-1)
    if out.shape!=(n,dim) or not np.isfinite(out).all():
        raise ValueError(f"Invalid OpenFOAM field {path}: {out.shape}")
    return out

def collect(root,d,meta):
    root=Path(root);nx=meta["nx"];ny=meta["ny"];n=nx*ny
    times=[]
    for p in root.iterdir():
        if p.is_dir():
            try:
                if float(p.name)>0:times.append(p)
            except ValueError:pass
    if not times:raise RuntimeError("OpenFOAM produced no solution timestep")
    sol=max(times,key=lambda p:float(p.name))
    u=_field(sol/"U",n,3);p=_field(sol/"p",n,1)
    xx,yy=np.meshgrid((np.arange(nx)+0.5)/nx,(np.arange(ny)+0.5)/ny)
    coords=np.column_stack([xx.ravel(),yy.ravel()]).astype("float32")
    x=np.column_stack([coords,*[np.full(n,meta[k]) for k in ("width","height","lid_speed","nu")]]).astype("float32")
    y=np.column_stack([u[:,0],u[:,1],p[:,0]]).astype("float32")
    return dict(x=x,y=y,coords=coords,meta=meta,time=float(sol.name))

def run(root,d,c,execute=True):
    meta=case(root,d,c)
    if not execute:return meta
    timeout=int(c.numerical.get("timeout",3600))
    for cmd in ("blockMesh","checkMesh","icoFoam"):
        checked([cmd,"-case",str(Path(root).resolve())],root,timeout=timeout)
    return collect(root,d,meta)
