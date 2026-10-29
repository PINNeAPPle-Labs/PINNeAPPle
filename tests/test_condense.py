"""pinneapple_condense: exactness of the condensation against the reference solver, the hierarchy against the flat
solver, the reduced interfaces (convergence, energy balance, error estimate), Keff, the learned operator with its
fallback, and the Craig-Bampton transient (exact with every mode)."""
import numpy as np
import pytest

from pinneapple_condense import fv
from pinneapple_condense import homogenize as H
from pinneapple_condense.hierarchy import Hierarchy
from pinneapple_condense.package import Variant, reference_stack, uniform_package
from pinneapple_condense.substructure import (
    Condensed,
    energy_norm,
    errors,
    reference_observables,
    tmax_cells,
)

ST = reference_stack(n=4)          # 4 x 4 x 23 cells per block: small enough for CI


@pytest.fixture(scope="module")
def case():
    pkg = uniform_package((4, 3), ST, seed=1)
    pkg.variants.append(Variant(removed=("ubump",)))
    pkg.variants.append(Variant(inclusion_k=(("ubump", 0.2),), contact=(("die", "tim", 3.0),)))
    pkg.layout[0, :] = 1
    pkg.layout[2, 1] = 2
    T, info = fv.solve_package(pkg, method="direct")
    return pkg, T, reference_observables(pkg, T)


def test_reference_energy_balance_and_maximum_principle(case):
    pkg, T, ref = case
    assert abs((ref["q_top"].sum() + ref["q_bottom"].sum()) / pkg.power.sum() - 1) < 1e-8
    tmax, tmax_active = tmax_cells(T, ST.active_k)
    assert tmax == pytest.approx(tmax_active, abs=1e-12)          # all heat on the active row, power >= 0


def test_exact_condensation_equals_reference(case):
    pkg, T, ref = case
    c = Condensed(pkg, m=10 ** 6).prepare()
    st = c.solve()
    assert np.abs(c.field(st) - T).max() < 1e-7 * T.max()
    e = errors(c.observables(st), ref, pkg.power.sum())
    assert e["err_Tmax"] < 1e-7 and e["err_q_ubump"] < 1e-7 and abs(e["energy_balance"]) < 1e-8
    assert c.timings["distinct_blocks"] == 3                     # three variants, twelve blocks: three factorisations


def test_reduced_interfaces_converge_conserve_and_estimate(case):
    pkg, T, ref = case
    sysm = fv.assemble_package(pkg)
    out = {}
    for m in (2, 8):
        c = Condensed(pkg, m).prepare()
        st = c.solve()
        e = errors(c.observables(st), ref, pkg.power.sum())
        en = energy_norm(sysm.K, (c.field(st) - T).ravel())
        est = np.sqrt((c.estimate(st) ** 2).sum())
        assert abs(e["energy_balance"]) < 1e-8                   # the constant is in every face basis
        assert 0.3 < est / en < 3.0                              # estimator effectivity
        out[m] = (e["err_Tj_max"], en)
    assert out[8][0] < 0.2 * out[2][0] and out[8][1] < out[2][1]


def test_refinement_to_exact(case):
    pkg, T, ref = case
    c = Condensed(pkg, 2).prepare()
    st = c.solve()
    for _ in range(12):
        if c.face_flags.all():
            break
        c = c.refine(c.estimate(st), 0.9, mode="full").prepare()
        st = c.solve()
    assert c.face_flags.all() and np.abs(c.field(st) - T).max() < 1e-7 * T.max()


def test_hierarchy_matches_flat_and_path_query(case):
    pkg, T, ref = case
    flat = Condensed(pkg, 4).prepare()
    a = flat.observables(flat.solve())
    h = Hierarchy(pkg, 4).prepare()
    sol = h.solve()
    b = h.observables(sol)
    assert max(np.abs(a[k] - b[k]).max() for k in a) < 1e-8
    assert np.abs(h.block_ports(sol, 2, 1) - sol["tb"][2, 1]).max() < 1e-10
    assert np.abs(h.recover(sol, 2, 1) - flat.recover(flat.solve(), 2, 1)).max() < 1e-8
    from pinneapple_condense.hierarchy import estimate
    assert np.allclose(estimate(h, sol), flat.estimate(flat.solve()), rtol=1e-6, atol=1e-12)


def test_hierarchy_reuses_identical_subtrees():
    small = Hierarchy(uniform_package((8, 8), ST), 2).prepare()
    big = Hierarchy(uniform_package((32, 32), ST), 2).prepare()
    # 16 x more blocks, 4 more distinct merges (two levels of halving per doubling of the side)
    assert big.timings["distinct_merges"] - small.timings["distinct_merges"] == 4


def test_keff_formula_and_what_it_erases():
    assert H.effective_layer(1.0, 50.0, 0.0) == pytest.approx((1.0, 1.0))
    assert H.effective_layer(1.0, 50.0, 1.0) == pytest.approx((50.0, 50.0))
    pkg = uniform_package((4, 4), ST, seed=1)
    p = pkg.power.copy()
    p[5, 6] += 0.2 * p.sum()                                     # a hotspot smaller than the pitch
    pkg.power = p
    T, _ = fv.solve_package(pkg, method="direct")
    ref = reference_observables(pkg, T)
    Th, sysm, _ = H.solve_homogenized(pkg)
    cmp = H.compare(pkg, ref, H.homogenized_observables(pkg, Th, sysm))
    assert abs(cmp["energy_balance"]) < 1e-8
    assert cmp["err_Tmax"] < -0.1                                # the coarse model misses the peak
    assert cmp["intra_block_spread_max"] > 0.1


def test_learned_operator_with_fallback():
    from pinneapple_condense.learned import (
        LearnedBlocks,
        solve_adaptive,
        solve_exact,
        variant_field,
    )
    pytest.importorskip("sklearn")
    L = LearnedBlocks(ST, m=4, n_train=12).fit()
    theta = variant_field((4, 4), seed=2, defects=1)
    p = uniform_package((4, 4), ST, seed=3).power
    r = solve_adaptive(L, theta, p, n_audit=4)
    assert r.flagged_ood == 1 and r.exact_blocks.sum() >= 1     # the defect is outside the training box: exact
    o_ex, _ = solve_exact(L, theta, p)
    assert abs(r.obs["Tj"].max() - o_ex["Tj"].max()) < 0.05


def test_craig_bampton_transient_exact_with_every_mode():
    from pinneapple_condense.transient import CBTransient, reference_transient
    pkg = uniform_package((2, 2), ST, seed=1)
    powers = np.array([pkg.power] * 5)
    ref = reference_transient(pkg, powers, 2e-4)
    nface = ST.n * ST.nz
    out = CBTransient(pkg, m=nface, k=ST.n ** 2 * ST.nz, shifts=(0.0,)).prepare(2e-4).run(powers)
    assert np.abs(out["Tmax"] - ref["Tmax"]).max() < 1e-8
    assert np.abs(out["q_tim"] - ref["q_tim"]).max() < 1e-8 * np.abs(ref["q_tim"]).max()


def test_p_refinement_follows_the_estimate(case):
    pkg, T, ref = case
    K = fv.assemble_package(pkg).K
    c = Condensed(pkg, 2).prepare()
    errs = []
    for _ in range(7):
        st = c.solve()
        errs.append(energy_norm(K, (c.field(st) - T).ravel()))
        c = c.refine(c.estimate(st), 0.5).prepare()
    assert all(b < a for a, b in zip(errs, errs[1:], strict=False)) and errs[-1] < 0.8 * errs[0]
    assert c.face_m.max() >= 16 and (c.face_m == 2).any()                 # local: some faces never refined


def test_exact_correction_converges_to_reference(case):
    pkg, T, ref = case
    c = Condensed(pkg, 2).prepare()
    st = c.solve()
    r = c.correct(st, rtol=1e-10)
    o = c.observables_fine(r["U"], r["pb"])
    assert r["history"][-1] <= 1e-10 and r["iterations"] < 300
    assert abs(o["Tj"].max() - ref["Tj"].max()) < 1e-6


# --------------------------------------------------------------------------------- track 1b: meshless, 1c: prior
def _slab():
    from pinneapple_condense.package import BASE, Layer, Package, Stack
    st = Stack(layers=(Layer("die", 100e-6, 4, "silicon"), Layer("tim", 30e-6, 2, "silicon")), contact=(),
               active="die", h_top=2e4, h_bottom=1e3)
    return Package(st, np.zeros((1, 1), int), [BASE], np.full((8, 8), 1e-3 / 64))


@pytest.mark.parametrize("method", ["gfdm", "rbf_fd"])
def test_meshless_exact_on_a_quadratic_through_an_interface(method, monkeypatch):
    """Uniform source, Robin top and bottom: the exact field is quadratic in z, which degree-2 stencils reproduce;
    with the sides as mirror planes the collocation is stable under refinement (it was not with side nodes)."""
    from pinneapple_condense import meshless as ML
    pkg = _slab()
    q, k, H, hT, hB = 1e10, 148.0, 130e-6, 2e4, 1e3
    C1, C0 = np.linalg.solve([[-k, hB], [-k - hT * H, -hT]], [0.0, -hT * q * H ** 2 / (2 * k) - q * H])
    monkeypatch.setattr(ML.Geometry, "source", lambda self, X: np.full(len(X), q))
    for hf in (6, 12):
        cl = ML.point_cloud(pkg, h_xy=pkg.stack.pitch / hf)
        sol = ML.solve(pkg, cl, method=method, oversample=True)
        exact = -q / (2 * k) * cl.X[:, 2] ** 2 + C1 * cl.X[:, 2] + C0
        assert np.abs(sol.T - exact).max() < 1e-3 * exact.max()


def test_meshless_layers_with_contact_resistance_against_fv():
    from pinneapple_condense import meshless as ML
    from pinneapple_condense.package import BASE, Package, Stack
    L = {x.name: x for x in reference_stack().layers}
    st = Stack(layers=(L["die"], L["tim"], L["lid"]), contact=(("die", "tim", 5e-6), ("tim", "lid", 5e-6)),
               active="die", h_top=2e4, h_bottom=1e3)
    pkg = Package(st, np.zeros((1, 1), int), [BASE], np.full((8, 8), 1.6e-3 / 64))
    T, _ = fv.solve_package(pkg, method="direct")
    o = ML.observables(ML.solve(pkg, method="gfdm", h_xy=st.pitch / 6), pkg)
    assert abs(o["Tmax"] / T[:, :, st.active_k].max() - 1) < 0.01 and abs(o["energy_balance"]) < 0.01


def test_prior_and_exception_patches():
    from pinneapple_condense import exceptions as EX
    pkg = uniform_package((6, 6), ST, seed=1)
    p = pkg.power.copy()
    p[10, 12] += 0.2 * p.sum()                                   # a sub-pitch hotspot the prior misses
    pkg.power = p
    T, _ = fv.solve_package(pkg, method="direct")
    ref = reference_observables(pkg, T)
    full = EX.run(pkg, budget=1.0, halo=0)                         # every block solved: the reference itself
    assert np.abs(full["T"] - T).max() < 1e-6 * T.max()
    err = {}
    for sel in ("residual", "random"):
        r = EX.run(pkg, budget=0.1, halo=1, select=sel, seed=3)
        o = reference_observables(pkg, r["T"])
        assert abs((o["q_top"].sum() + o["q_bottom"].sum()) / pkg.power.sum() - 1) < 1e-9    # energy corrected
        err[sel] = abs(o["Tj"].max() - ref["Tj"].max())
    assert err["residual"] < err["random"]
    assert EX.run(pkg, budget=0.0)["eta"].argmax() == np.ravel_multi_index((10 // ST.n, 12 // ST.n), (6, 6))
