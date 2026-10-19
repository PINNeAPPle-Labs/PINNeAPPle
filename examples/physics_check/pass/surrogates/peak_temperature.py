"""A stand-in for a surrogate of the peak temperature of a slab with uniform heating q, thickness L and
conductivity k, cooled to T0 on both faces (exact: T0 + q L^2 / (8 k))."""


def predict(X):
    q, L, k, T0 = X.T
    return (T0 + q * L ** 2 / (8 * k)).reshape(-1, 1)
