`pinneapple_data.audit.near_duplicates` and `leakage` (and the fault-injection dataset checks built on them) find
exact copies on every machine. Distances came from `|a|^2 + |b|^2 - 2 a.b`, which put identical rows apart by
rounding noise that depends on the BLAS and CPU (up to 1.4 for 1000-wide rows with a large offset); when copies
made up half a set, the median distance was that noise too, so no copy was counted. The distance to the nearest
neighbour is now computed directly, on centred data.
