"""The native backend: one C ABI, the NumPy reference as the definition, and a refusal for every lie.

What is asserted here, and why each one is the test that matters:

* the C kernels and the NumPy reference compute the **same numbers** — a native layer whose "faster" path
  disagrees with the reference is a bug factory, so the comparison is element by element, not a statistic;
* the answer is **right**, not only self-consistent: the Jacobi sweep is run against a manufactured solution
  and lands on the discretisation's own error, not on zero and not on a solver artefact;
* a library built for another ABI, or one that kept the operation names and changed their meaning, is
  **refused by name** instead of called;
* a machine without a compiler, or a run with ``PINNEAPPLE_NATIVE=0``, gets the reference **and the reason**,
  so no result is ever attributed to a backend that did not run;
* the calls C refuses (unstable step, omega out of range, wrong output size) come back as a sentence with the
  limit in it.
"""
from __future__ import annotations

import subprocess
import sys

import numpy as np
import pytest

from pinneapple_core.native import (
    ABI_VERSION,
    AbiVersionMismatch,
    NativeKernels,
    clear_cache,
    load,
    status,
)
from pinneapple_core.native import (
    build as build_module,
)
from pinneapple_core.native.abi import NativeCallError, UnknownOperation, check_specs
from pinneapple_core.native.kernels import (
    heat_step_2d_reference,
    jacobi_2d_reference,
    laplacian_2d_reference,
)

pytest.importorskip("numpy")

needs_compiler = pytest.mark.skipif(
    build_module.compiler() is None, reason="no C compiler on this machine: the native backend cannot be built here"
)


@pytest.fixture()
def library():
    """The library this machine builds, or a skip with the reason the compiler was not there."""
    try:
        path = build_module.build()
    except Exception as exc:  # CompilerNotFound or BuildFailed
        pytest.skip(f"native backend unavailable: {exc}")
    loaded = load(path)
    assert loaded is not None
    return loaded


@pytest.fixture(autouse=True)
def _isolated_cache(monkeypatch, tmp_path):
    """Every test builds into (and loads from) its own cache, so the environment of one case cannot leak."""
    monkeypatch.setenv("PINNEAPPLE_NATIVE_CACHE", str(tmp_path / "cache"))
    clear_cache()
    yield
    clear_cache()


def grid(n: int = 33) -> tuple:
    """The unit square and the manufactured solution ``u = sin(pi x) sin(pi y)`` of ``-laplacian(u) = f``."""
    x = np.linspace(0.0, 1.0, n)
    xx, yy = np.meshgrid(x, x, indexing="ij")
    exact = np.sin(np.pi * xx) * np.sin(np.pi * yy)
    return exact, 2.0 * np.pi ** 2 * exact, 1.0 / (n - 1)


# -- the protocol -----------------------------------------------------------------------------------------


@needs_compiler
def test_the_library_reports_the_abi_this_loader_speaks(library):
    assert library.abi_version == ABI_VERSION
    assert library.build.strip(), "the library must say which compiler built it"


@needs_compiler
def test_the_advertised_signatures_are_the_ones_the_loader_expects(library):
    assert check_specs(library.served) is None, "the C listing and the Python specs have drifted apart"
    assert library.ops() == ["heat_step_2d", "jacobi_2d", "laplacian_2d"]


@needs_compiler
def test_the_build_is_cached_by_the_fingerprint_of_what_it_compiles(library):
    first = build_module.build()
    assert build_module.build() == first, "a second call must reuse the build instead of compiling again"
    manifest = first.parent / "build.json"
    assert manifest.exists()
    assert str(first) in manifest.read_text(encoding="utf-8")


# -- the kernels agree, element by element ----------------------------------------------------------------


@needs_compiler
@pytest.mark.parametrize("n", [9, 33, 65])
def test_laplacian_agrees_with_the_reference(library, n):
    u, _, h = grid(n)
    native = library.call("laplacian_2d", [u], (n, n, h), u.size).reshape(u.shape)
    reference = laplacian_2d_reference(u, h)
    assert np.max(np.abs(native - reference)) <= 1e-14
    assert np.max(np.abs(native - reference)) == 0.0  # measured: the same expression in the same order


@needs_compiler
def test_heat_step_agrees_with_the_reference(library):
    u, _, _ = grid(33)
    native = library.call("heat_step_2d", [u], (33, 33, 1.0, 1e-5, 50), u.size).reshape(u.shape)
    reference = heat_step_2d_reference(u, 1.0, 1e-5, 50)
    assert np.max(np.abs(native - reference)) <= 1e-12


@needs_compiler
@pytest.mark.parametrize("omega", [1.0, 2.0 / 3.0])
def test_jacobi_agrees_with_the_reference(library, omega):
    _, f, h = grid(33)
    # 400 sweeps is enough to exercise every code path: this test asks whether the two agree, and the one that
    # asks whether they are *right* runs long enough to converge.
    native = library.call("jacobi_2d", [np.zeros((33, 33)), f], (33, 33, h, 400, omega), 33 * 33)
    reference = jacobi_2d_reference(np.zeros((33, 33)), f, h, 400, omega)
    assert np.max(np.abs(native.reshape(33, 33) - reference)) <= 1e-12


@needs_compiler
def test_the_native_jacobi_lands_on_the_discretisation_error_not_on_zero():
    """The right answer, not just an agreeing one: the five-point stencil's own truncation error is
    ``pi^2 h^2 / 12`` for this manufactured solution, and that is where both backends land."""
    n = 33
    exact, f, h = grid(n)
    for backend in (NativeKernels(), NativeKernels(force_reference=True)):
        u = backend.jacobi_2d(np.zeros((n, n)), f, h=h, steps=4000, omega=1.0)
        error = float(np.max(np.abs(u - exact)))
        expected = float(np.pi ** 2 * h ** 2 / 12.0)
        assert error == pytest.approx(expected, rel=0.05), (
            f"{backend.backend}: measured {error:.3e}, the truncation estimate is {expected:.3e}"
        )


@needs_compiler
def test_the_heat_step_follows_the_analytic_decay_of_the_first_mode():
    """``u_t = laplacian(u)`` decays ``sin(pi x) sin(pi y)`` as ``exp(-2 pi^2 t)``; the difference is the
    explicit scheme's own truncation, so it shrinks with dt."""
    n = 33
    exact, _, _ = grid(n)
    alpha, dt, steps = 1.0, 1e-5, 50
    reference = heat_step_2d_reference(exact, alpha, dt, steps)
    analytic = exact * np.exp(-2.0 * np.pi ** 2 * alpha * dt * steps)
    error = float(np.max(np.abs(reference - analytic)))
    assert 0.0 < error < 1e-4, f"measured {error:.3e} against exp(-2 pi^2 t)"


# -- the refusals -----------------------------------------------------------------------------------------


def _compile(tmp_path, source: str) -> str:
    """Compile a C library in the test's own directory, or skip when this machine has no compiler."""
    compiler = build_module.compiler()
    if compiler is None:
        pytest.skip("no C compiler on this machine")
    path = tmp_path / ("libfake.dylib" if sys.platform == "darwin" else "libfake.so")
    result = subprocess.run(
        [compiler, "-O0", "-fPIC", "-shared", "-std=c11", "-o", str(path), "-x", "c", "-"],
        input=source, capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    return str(path)


FAKE_C = r"""
#include <stdint.h>
#include <string.h>
const char *pp_build(void) { return "fake backend built by a test"; }
int32_t pp_ops(char *buf, int32_t cap, int32_t *needed) {
    (void)buf; (void)cap;
    if (needed) *needed = 1;
    return 0;
}
int32_t pp_call(const char *op, const double *const *in, const int64_t *sizes, int32_t n_in,
                const double *params, int32_t n_params, double *out, int64_t out_cap,
                int64_t *out_size, char *err, int32_t err_cap) {
    (void)op; (void)in; (void)sizes; (void)n_in; (void)params; (void)n_params; (void)out; (void)out_cap;
    (void)out_size; (void)err; (void)err_cap;
    return 0;
}
"""


def _abi_version_function(version: int) -> str:
    return "int32_t pp_abi_version(void) { return " + str(version) + "; }\n"


def test_a_library_built_for_another_abi_is_refused(tmp_path):
    source = "#include <stdint.h>\n" + _abi_version_function(ABI_VERSION + 1) + FAKE_C
    path = _compile(tmp_path, source)
    with pytest.raises(AbiVersionMismatch) as excinfo:
        load(path)
    assert excinfo.value.found == ABI_VERSION + 1
    assert str(ABI_VERSION) in str(excinfo.value)


def test_a_library_that_renamed_its_operations_is_refused(tmp_path):
    """The ABI version matches, so only the listing can catch this: the names survived, the meaning did not."""
    source = (
        "#include <stdint.h>\n#include <string.h>\n"
        + _abi_version_function(ABI_VERSION)
        + r"""
const char *pp_build(void) { return "renamed"; }
int32_t pp_ops(char *buf, int32_t cap, int32_t *needed) {
    const char *listing = "laplacian_2d\t(u[nx*ny]; nx, ny) -> lap\n"
                          "heat_step_2d\t(u[nx*ny]; nx, ny) -> lap\n"
                          "jacobi_2d\t(u[nx*ny]; nx, ny) -> lap\n";
    int32_t n = (int32_t)strlen(listing) + 1;
    if (needed) *needed = n;
    if (!buf || cap <= 0) return -3;
    memcpy(buf, listing, (size_t)(n < cap ? n : cap));
    buf[(n < cap ? n : cap) - 1] = 0;
    return 0;
}
int32_t pp_call(const char *op, const double *const *in, const int64_t *sizes, int32_t n_in,
                const double *params, int32_t n_params, double *out, int64_t out_cap,
                int64_t *out_size, char *err, int32_t err_cap) {
    (void)op; (void)in; (void)sizes; (void)n_in; (void)params; (void)n_params; (void)out; (void)out_cap;
    (void)out_size; (void)err; (void)err_cap;
    return 0;
}
"""
    )
    path = _compile(tmp_path, source)
    with pytest.raises(NativeCallError) as excinfo:
        load(path)
    assert "signatures disagree" in str(excinfo.value)


@needs_compiler
def test_an_output_buffer_that_is_too_small_is_refused_by_the_library(library):
    u, _, h = grid(9)
    with pytest.raises(NativeCallError) as excinfo:
        library.call("laplacian_2d", [u], (9, 9, h), out_size=5)
    assert "out is smaller than nx*ny" in str(excinfo.value)


@needs_compiler
def test_an_operation_the_library_does_not_serve_is_named(library):
    with pytest.raises(UnknownOperation) as excinfo:
        library.call("solve_everything", [np.zeros((3, 3))], (), 9)
    assert "solve_everything" in str(excinfo.value)


@needs_compiler
def test_the_library_refuses_a_grid_that_is_not_whole(library):
    u = np.zeros((9, 9))
    with pytest.raises(NativeCallError) as excinfo:
        library.call("laplacian_2d", [u], (9.5, 9, 0.1), u.size)
    assert "must be an integer" in str(excinfo.value)


def test_an_unstable_step_is_refused_with_the_limit_in_the_message():
    u = np.zeros((33, 33))
    with pytest.raises(ValueError) as excinfo:
        heat_step_2d_reference(u, alpha=1.0, dt=1.0, steps=1)
    assert "limit" in str(excinfo.value) and "unstable" in str(excinfo.value)


def test_omega_above_one_is_refused_because_damped_jacobi_diverges_there():
    """Measured before it was a check: omega = 1.9 on a 33 x 33 grid overflowed to inf in 2000 sweeps."""
    u = np.zeros((33, 33))
    with pytest.raises(ValueError) as excinfo:
        jacobi_2d_reference(u, u, h=1.0 / 32, steps=10, omega=1.9)
    assert "(0, 1]" in str(excinfo.value)


# -- the honest fallback ----------------------------------------------------------------------------------


def test_without_a_compiler_the_reference_runs_and_says_why(monkeypatch):
    monkeypatch.setattr(build_module, "compiler", lambda: None)
    monkeypatch.delenv("PINNEAPPLE_NATIVE_LIBRARY", raising=False)
    kernels = NativeKernels()
    assert kernels.backend == "reference:numpy"
    reason = kernels.describe()["reason"]
    assert "no C compiler" in reason

    exact, f, h = grid(33)
    u = kernels.jacobi_2d(np.zeros((33, 33)), f, h=h, steps=200, omega=1.0)
    assert np.all(np.isfinite(u))
    assert np.max(np.abs(u - jacobi_2d_reference(np.zeros((33, 33)), f, h=h, steps=200, omega=1.0))) == 0.0


def test_the_opt_out_variable_turns_the_native_backend_off(monkeypatch, library):
    """Even with a library already built and loaded, PINNEAPPLE_NATIVE=0 means the reference runs."""
    monkeypatch.setenv("PINNEAPPLE_NATIVE", "0")
    clear_cache()
    assert status()["available"] is False
    assert "PINNEAPPLE_NATIVE=0" in status()["reason"]
    kernels = NativeKernels()
    assert kernels.backend == "reference:numpy"


@needs_compiler
def test_the_status_dictionary_says_where_the_library_is():
    info = status()
    assert info["available"] is True
    assert info["compiler"] and info["library"]
    assert info["library"].endswith((".dylib", ".so", ".dll"))


@needs_compiler
def test_the_dispatcher_reports_the_backend_that_produced_the_number(library):
    native = NativeKernels(library=library)
    reference = NativeKernels(force_reference=True)
    assert native.backend.startswith("native:")
    assert reference.backend == "reference:numpy"
    assert native.is_native and not reference.is_native
    exact, _, h = grid(17)
    assert np.allclose(native.laplacian_2d(exact, h), reference.laplacian_2d(exact, h), atol=1e-14)


def test_a_non_contiguous_or_float32_input_is_made_safe_before_crossing_the_boundary(library):
    """The classic silent corruption: C reads the buffer row-major whatever the stride says."""
    n = 17
    base, _, h = grid(n)
    transposed = np.asfortranarray(base).T  # same values, different layout
    native = NativeKernels(library=library).laplacian_2d(transposed, h)
    reference = laplacian_2d_reference(np.ascontiguousarray(transposed), h)
    assert np.max(np.abs(native - reference)) == 0.0

    float32 = base.astype(np.float32)
    native32 = NativeKernels(library=library).laplacian_2d(float32, h)
    assert np.max(np.abs(native32 - laplacian_2d_reference(float32, h))) == 0.0


def test_the_loader_returns_none_instead_of_raising_when_it_cannot_build(monkeypatch):
    monkeypatch.setattr(build_module, "compiler", lambda: None)
    monkeypatch.delenv("PINNEAPPLE_NATIVE_LIBRARY", raising=False)
    monkeypatch.delenv("PINNEAPPLE_NATIVE", raising=False)
    clear_cache()
    assert load() is None
    from pinneapple_core.native.loader import last_reason

    assert "no C compiler" in last_reason()
