"""A stand-in for a surrogate of the peak temperature of a slab with uniform heating q, thickness L and
conductivity k, cooled to T0 on both faces (exact: T0 + q L^2 / (8 k)). A unit slip divides by 1/k instead of k."""


def predict(X):
    q, L, k, T0 = X.T
    return (T0 + q * L ** 2 / (8 * (1e4 / k))).reshape(-1, 1)
