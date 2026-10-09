"""Your first PINNeAPPle result in about a minute on a laptop CPU: solve Burgers' equation with a
physics-informed network, check it against the exact solution and plot both.

    pip install pinneapple
    python examples/first_example/first_example.py        # writes first_example.png
"""
import matplotlib.pyplot as plt
import numpy as np

import pinneapple as pp

prob = pp.PhysicalProblem.from_preset("burgers_1d", nu=0.01 / np.pi)   # u_t + u u_x = nu u_xx, x in [-1, 1]
pinn = pp.solve(prob, "pinn", epochs=1500, seed=0)                     # train the network (about 45 s)
exact = pp.solve(prob, "analytic")                                     # Cole-Hopf closed form

x = np.linspace(-1, 1, 201)
pts = np.stack([x, np.full_like(x, 0.75)], axis=1)                     # (x, t = 0.75): the steep front
u_pinn, u_exact = pinn.predict(pts)[:, 0], exact.predict(pts)[:, 0]
print(f"relative L2 error at t = 0.75: {pp.metrics.relative_l2(u_pinn, u_exact):.3f}")

plt.plot(x, u_exact, "k-", label="exact")
plt.plot(x, u_pinn, "r--", label="PINN")
plt.xlabel("x")
plt.ylabel("u(x, t = 0.75)")
plt.legend()
plt.savefig("first_example.png", dpi=120)
