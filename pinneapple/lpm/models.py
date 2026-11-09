"""LPM model adapters: native PINNeAPPLe FNO plus campaign-specific point and graph models."""
import torch
from torch import nn

class PointMLP(nn.Module):
    def __init__(self,inp,out,width=96,depth=4):
        super().__init__()
        layers=[]
        for i in range(depth):
            layers.extend([nn.Linear(inp if i==0 else width,width),nn.GELU()])
        self.net=nn.Sequential(*layers,nn.Linear(width,out))
    def forward(self,x,shape=None):return self.net(x)

class SpectralConv2d(nn.Module):
    def __init__(self,cin,cout,modes=8):
        super().__init__();self.modes=modes
        self.wpos=nn.Parameter(torch.randn(cin,cout,modes,modes,2)*.02)
        self.wneg=nn.Parameter(torch.randn(cin,cout,modes,modes,2)*.02)
    def forward(self,x):
        b,c,h,w=x.shape;xf=torch.fft.rfft2(x,norm="ortho")
        n0=min(self.modes,h//2);n1=min(self.modes,w//2+1)
        out=torch.zeros(b,self.wpos.shape[1],h,w//2+1,device=x.device,dtype=xf.dtype)
        if n0 and n1:
            out[:,:,:n0,:n1]=torch.einsum("bixy,ioxy->boxy",xf[:,:,:n0,:n1],torch.view_as_complex(self.wpos[:,:,:n0,:n1].contiguous()))
            out[:,:,-n0:,:n1]=torch.einsum("bixy,ioxy->boxy",xf[:,:,-n0:,:n1],torch.view_as_complex(self.wneg[:,:,:n0,:n1].contiguous()))
        return torch.fft.irfft2(out,s=(h,w),norm="ortho")

class FNO2D(nn.Module):
    def __init__(self,inp,out,width=48,modes=8,depth=4):
        super().__init__()
        self.lift=nn.Conv2d(inp,width,1)
        self.spec=nn.ModuleList([SpectralConv2d(width,width,modes) for _ in range(depth)])
        self.local=nn.ModuleList([nn.Conv2d(width,width,1) for _ in range(depth)])
        self.head=nn.Sequential(nn.Conv2d(width,128,1),nn.GELU(),nn.Conv2d(128,out,1))
    def forward(self,x,shape):
        if shape is None:raise ValueError("FNO requires structured grid shape")
        ny,nx=shape
        z=self.lift(x.reshape(x.shape[0],ny,nx,x.shape[-1]).permute(0,3,1,2))
        for a,b in zip(self.spec,self.local):z=torch.nn.functional.gelu(a(z)+b(z))
        return self.head(z).permute(0,2,3,1).reshape(x.shape[0],-1,self.head[-1].out_channels)

class GraphModel(nn.Module):
    def __init__(self,inp,out,width=64,steps=3,k=6):
        super().__init__()
        self.k=k;self.steps=steps
        self.encoder=nn.Sequential(nn.Linear(inp,width),nn.GELU(),nn.Linear(width,width))
        self.message=nn.Sequential(nn.Linear(2*width+3,width),nn.GELU(),nn.Linear(width,width))
        self.update=nn.GRUCell(width,width)
        self.decoder=nn.Sequential(nn.Linear(width,width),nn.GELU(),nn.Linear(width,out))
    def forward(self,x,shape=None):
        B,N,_=x.shape;dim=2 if x.shape[-1]==6 else 3
        coords=x[...,:dim]
        neighbors=torch.cdist(coords,coords).topk(min(self.k+1,N),largest=False).indices[...,1:]
        h=self.encoder(x)
        for _ in range(self.steps):
            src=h.unsqueeze(1).expand(-1,N,-1,-1).gather(2,neighbors.unsqueeze(-1).expand(-1,-1,-1,h.shape[-1]))
            dst=h.unsqueeze(2).expand_as(src)
            src_coords=coords.unsqueeze(1).expand(-1,N,-1,-1).gather(2,neighbors.unsqueeze(-1).expand(-1,-1,-1,dim))
            delta=coords.unsqueeze(2)-src_coords
            if dim==2:delta=torch.nn.functional.pad(delta,(0,1))
            messages=self.message(torch.cat([src,dst,delta],dim=-1)).mean(dim=2)
            h=self.update(messages.reshape(-1,h.shape[-1]),h.reshape(-1,h.shape[-1])).reshape(B,N,-1)
        return self.decoder(h)

class NativeFNO2D(nn.Module):
    """Channel-layout adapter around PINNeAPPle's existing FNO2d."""
    def __init__(self,inp,out,opts):
        super().__init__()
        from pinneapple_neural.architectures.neural_operators.fno import FNO2d
        modes=int(opts.get("modes",8))
        self.core=FNO2d(in_channels=inp,out_channels=out,width=int(opts.get("width",48)),
                        modes1=modes,modes2=modes,layers=int(opts.get("depth",4)),
                        use_grid=False,padding=int(opts.get("padding",0)))
    def forward(self,x,shape=None):
        if shape is None:raise ValueError("FNO requires (ny,nx)")
        ny,nx=shape
        if x.shape[1]!=ny*nx:raise ValueError("FNO grid shape mismatch")
        inp=x.reshape(x.shape[0],ny,nx,x.shape[-1]).permute(0,3,1,2)
        result=self.core(inp)
        out=result.y if hasattr(result,"y") else result
        return out.permute(0,2,3,1).reshape(x.shape[0],ny*nx,-1)

class NativeMeshGraphNet(nn.Module):
    """Fixed structured-topology adapter around PINNeAPPle's MeshGraphNet."""
    def __init__(self,inp,out,opts):
        super().__init__()
        from pinneapple_neural.architectures.graphnn.mesh_graph_net import MeshGraphNet
        self.core=MeshGraphNet(node_in_dim=inp,out_dim=out,hidden_dim=int(opts.get("width",64)),
                               n_message_passing=int(opts.get("steps",3)))
        self.edge_index=None
    def forward(self,x,shape=None):
        from pinneapple_neural.architectures.graphnn.base import GraphBatch
        n=x.shape[1]
        if self.edge_index is None or self.edge_index.shape[1]!=2*(n-1):
            ids=torch.arange(n,device=x.device)
            self.edge_index=torch.stack([torch.cat([ids[:-1],ids[1:]]),
                                         torch.cat([ids[1:],ids[:-1]])])
        return self.core(GraphBatch(x=x,edge_index=self.edge_index.to(x.device))).y

def make_model(kind,inp,out,opts):
    if kind=="mlp":return PointMLP(inp,out,int(opts.get("width",96)),int(opts.get("depth",4)))
    if kind=="fno":return NativeFNO2D(inp,out,opts)
    if kind=="graph":return NativeMeshGraphNet(inp,out,opts)
    raise ValueError(f"Unknown LPM architecture: {kind}")
