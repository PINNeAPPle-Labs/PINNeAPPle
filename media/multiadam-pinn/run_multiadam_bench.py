"""Adam vs MultiAdam on two multi-term PINN losses (CPU, small). Same net, steps, points, seeds.

Run from the PINNeAPPle repo root:  python run_multiadam_bench.py [out.json]
Arms: adam (torch default betas 0.9/0.999), adam_b99 (Adam with MultiAdam's betas 0.99/0.99,
a control), multiadam (pinneapple_neural.trainer.multiadam.MultiAdam, betas 0.99/0.99). lr=1e-3 for all.
"""
import json, math, sys, time
import torch
from pinneapple_neural.trainer.multiadam import MultiAdam

torch.set_num_threads(4)
SEEDS = [0, 1, 2]
ARMS = ["adam", "adam_b99", "multiadam"]


def mlp(din, width, depth, dtype):
    layers, d = [], din
    for _ in range(depth):
        layers += [torch.nn.Linear(d, width), torch.nn.Tanh()]
        d = width
    layers.append(torch.nn.Linear(d, 1))
    return torch.nn.Sequential(*layers).to(dtype)


def make_opt(arm, net):
    if arm == "multiadam":
        return MultiAdam(net.parameters(), lr=1e-3)
    betas = (0.9, 0.999) if arm == "adam" else (0.99, 0.99)
    return torch.optim.Adam(net.parameters(), lr=1e-3, betas=betas)


def train(arm, seed, problem, steps, log_every, eval_every):
    torch.manual_seed(seed)
    net, terms_fn, rel_l2 = problem(seed)
    opt = make_opt(arm, net)
    hist = {"step": [], "terms": [], "eval_step": [], "rel_l2": []}
    t0 = time.time()
    for it in range(steps + 1):
        terms = terms_fn(net)
        if it % log_every == 0:
            hist["step"].append(it)
            hist["terms"].append([float(t.detach()) for t in terms])
        if it % eval_every == 0:
            hist["eval_step"].append(it)
            hist["rel_l2"].append(rel_l2(net))
        if it == steps:
            break
        if arm == "multiadam":
            opt.step(terms)
        else:
            opt.zero_grad()
            sum(terms).backward()
            opt.step()
    hist["final_rel_l2"] = hist["rel_l2"][-1]
    hist["final_terms"] = hist["terms"][-1]
    hist["seconds"] = time.time() - t0
    return hist


# ---------- Problem A: 1-D Poisson u'' = -pi^2 sin(pi x), u(+-1)=0, exact u = sin(pi x)
def poisson1d(w_bc):
    def build(seed):
        dt = torch.float64
        net = mlp(1, 20, 2, dt)
        x = torch.linspace(-1, 1, 64, dtype=dt).reshape(-1, 1).requires_grad_(True)
        xb = torch.tensor([[-1.0], [1.0]], dtype=dt)
        f = (-(math.pi ** 2) * torch.sin(math.pi * x)).detach()
        xt = torch.linspace(-1, 1, 501, dtype=dt).reshape(-1, 1)
        ref = torch.sin(math.pi * xt)

        def terms(net):
            u = net(x)
            du = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
            d2u = torch.autograd.grad(du.sum(), x, create_graph=True)[0]
            return [((d2u - f) ** 2).mean(), w_bc * (net(xb) ** 2).mean()]

        def rel(net):
            with torch.no_grad():
                return float(torch.linalg.norm(net(xt) - ref) / torch.linalg.norm(ref))
        return net, terms, rel
    return build


# ---------- Problem B: 2-D Helmholtz  lap u + k^2 u = q on [-1,1]^2, u=0 on boundary,
# exact u = sin(a1 pi x) sin(a2 pi y), a1=1, a2=4, k=1 (Wang, Teng & Perdikaris 2021 setup)
A1, A2, K = 1.0, 4.0, 1.0


def helm_exact(xy):
    return torch.sin(A1 * math.pi * xy[:, :1]) * torch.sin(A2 * math.pi * xy[:, 1:])


def helmholtz2d(seed):
    dt = torch.float32
    g = torch.Generator().manual_seed(1234)  # same collocation points for every arm and seed
    xr = (2 * torch.rand(1024, 2, generator=g) - 1).to(dt).requires_grad_(True)
    s = 2 * torch.rand(64, generator=g) - 1
    one = torch.ones(64)
    xb = torch.cat([torch.stack([s, -one], 1), torch.stack([s, one], 1),
                    torch.stack([-one, s], 1), torch.stack([one, s], 1)]).to(dt)
    with torch.no_grad():
        q = (-(A1 * math.pi) ** 2 - (A2 * math.pi) ** 2 + K ** 2) * helm_exact(xr)
    gx = torch.linspace(-1, 1, 101)
    X, Y = torch.meshgrid(gx, gx, indexing="ij")
    xt = torch.stack([X.flatten(), Y.flatten()], 1).to(dt)
    ref = helm_exact(xt)
    torch.manual_seed(seed)
    net = mlp(2, 50, 3, dt)

    def terms(net):
        u = net(xr)
        gu = torch.autograd.grad(u.sum(), xr, create_graph=True)[0]
        uxx = torch.autograd.grad(gu[:, 0].sum(), xr, create_graph=True)[0][:, :1]
        uyy = torch.autograd.grad(gu[:, 1].sum(), xr, create_graph=True)[0][:, 1:]
        return [((uxx + uyy + K ** 2 * u - q) ** 2).mean(), (net(xb) ** 2).mean()]

    def rel(net):
        with torch.no_grad():
            return float(torch.linalg.norm(net(xt) - ref) / torch.linalg.norm(ref))
    return net, terms, rel


def main(out):
    res = {"config": {"seeds": SEEDS, "arms": ARMS, "lr": 1e-3, "torch": torch.__version__,
                      "threads": torch.get_num_threads()}, "problems": {}}
    probs = {
        "poisson1d_wbc1": dict(build=poisson1d(1.0), steps=2000, log=20, ev=100,
                               desc="1-D Poisson, PDE+BC, boundary weight 1, MLP 1-20-20-1 tanh float64, 64 pts"),
        "poisson1d_wbc1e-3": dict(build=poisson1d(1e-3), steps=2000, log=20, ev=100,
                                  desc="same, boundary term weighted by 1e-3 (repo reference test)"),
        "helmholtz2d": dict(build=helmholtz2d, steps=3000, log=25, ev=250,
                            desc="2-D Helmholtz a1=1 a2=4 k=1, PDE+BC weight 1, MLP 2-50-50-50-1 tanh float32, 1024 interior + 256 boundary pts"),
    }
    t_all = time.time()
    for name, p in probs.items():
        res["problems"][name] = {"desc": p["desc"], "steps": p["steps"], "runs": {}}
        for arm in ARMS:
            for seed in SEEDS:
                h = train(arm, seed, p["build"], p["steps"], p["log"], p["ev"])
                res["problems"][name]["runs"][f"{arm}/s{seed}"] = h
                print(f"{name:18s} {arm:9s} s{seed}  relL2={h['final_rel_l2']:.4g}  "
                      f"terms={[f'{t:.3g}' for t in h['final_terms']]}  {h['seconds']:.1f}s", flush=True)
        summ = {}
        for arm in ARMS:
            v = [res["problems"][name]["runs"][f"{arm}/s{s}"]["final_rel_l2"] for s in SEEDS]
            summ[arm] = {"rel_l2_per_seed": v, "rel_l2_median": sorted(v)[1],
                         "rel_l2_min": min(v), "rel_l2_max": max(v)}
        res["problems"][name]["summary"] = summ
    res["total_seconds"] = time.time() - t_all
    json.dump(res, open(out, "w"))
    print("total", res["total_seconds"])


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "multiadam_bench.json")
