---
title: "NVIDIA Warp backend: warp.fem weak forms + zero-copy torch interop"
labels: [enhancement, gpu, solvers]
---

**Who has it:** NVIDIA Warp 1.18 (Apache-2.0): `@wp.kernel` with `wp.Tape` autodiff, `warp.fem` integrands on grid/tet/hex geometries, `wp.to_torch` via DLPack.

**What PINNeAPPle already has:** Warp only indirectly through the Newton bridge; ROADMAP §3 lists Warp as a candidate differentiable backend.

**Proposal:** `compute_backends.warp`: zero-copy tensors, a Poisson/elasticity example in warp.fem, differentiable particle step for `particle_dynamics`.

**Acceptance (rule D1 — independent reference):** warp.fem Poisson matches the manufactured solution with O(h^2) convergence; Warp vs torch gradients agree to 1e-6.

**Origin:** mapping 2026-10-12: NVIDIA Warp

---

**Estado (2026-11-04): entregue.** `pinneapple_tools/compute_backends/warp_backend.py` (extra `pinneapple[warp]`):

- zero-copy: `to_warp` / `to_torch` sobre DLPack, com um teste que escreve por um lado e lê pelo outro pelo mesmo
  `data_ptr`; tensor não contíguo é recusado em vez de copiado.
- `warp.fem`: Poisson com solução manufaturada `u = x³(1-x)³ y³(1-y)³` (escolhida por NÃO ser autofunção do
  operador discreto — a usual `sin(πx) sin(πy)` converge em duas iterações de CG e esconde erro de montagem).
  Ordem observada: Q1 1.97→2.00, Q2 2.91→2.99 em L2, 8→64 células por lado; resíduo do CG reportado.
- passo diferenciável para `particle_dynamics`: `WarpBackend.step` é a mesma aritmética de `RigidBody.step`,
  gravada em `wp.Tape`; `gradients_against_torch()` compara valor e gradiente (posição, velocidade, força, torque)
  com `torch.autograd` — 1e-16 relativo em 2-D e 3-D, contra o 1e-6 do critério de aceitação D1.
- três armadilhas silenciosas encontradas e documentadas no código: kernel que sobrescreve a própria entrada faz o
  Warp avaliar a adjunta no valor novo (5e-4 de erro relativo em `d loss / d omega`); loss calculada fora da fita
  deixa todos os gradientes em zero sem erro; quaternion do Warp é `(x, y, z, w)` e o de `RigidBodyState` é
  `(w, x, y, z)`.

Testes: `tests/test_warp_backend.py` (13 checks). Docs: `docs/core_concepts/warp_backend.md`. Não é um
`pp.set_backend("warp")` (Warp diferencia por fita explícita, não pelo `PhysicsBackend` de arrays), e os kernels de
passo são float64 apenas — as duas limitações estão escritas na docstring do módulo e na doc.
