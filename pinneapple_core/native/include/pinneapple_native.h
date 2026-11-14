/*
 * pinneapple_native.h -- the stable C ABI a native kernel backend exposes to PINNeAPPle.
 *
 * This header is the protocol, not an implementation detail: a backend in C, C++, Rust (extern "C"),
 * Fortran (iso_c_binding) or Zig implements these five functions and PINNeAPPle can call it. Python stays
 * the scientific interface; the native code is an execution backend behind a C ABI, so N languages do not
 * need N x N bridges.
 *
 * Contract:
 *   - All arrays are double, C-contiguous, in Fortran-free (row-major) order.
 *   - A 2-D array of shape (nx, ny) is passed flat with index i * ny + j.
 *   - Integer sizes travel as int64_t; problem sizes as double in `params` and must be exact integers.
 *   - Nothing is allocated by the library: the caller passes `out`, its capacity, and gets back `out_size`.
 *   - A non-zero return is an error, and `error` carries the sentence a human needs. Nothing is written to
 *     `out` on error.
 *   - Bumping PP_NATIVE_ABI_VERSION is a breaking change; a loader refuses a library whose version differs
 *     instead of calling it and hoping the structs line up.
 *
 * Build (the loader does this itself, see pinneapple_core/native/build.py):
 *   cc -O3 -fPIC -shared -std=c11 -o libpinneapple_native.dylib kernels.c -I../include
 */
#ifndef PINNEAPPLE_NATIVE_H
#define PINNEAPPLE_NATIVE_H

#include <stdint.h>

#define PP_NATIVE_ABI_VERSION 1
#define PP_NATIVE_ERROR_CAPACITY 512

/* Return codes. Everything below zero is a refusal; the message says which. */
#define PP_OK 0
#define PP_ERR_UNKNOWN_OP (-1)
#define PP_ERR_BAD_ARGS (-2)
#define PP_ERR_OUTPUT_SIZE (-3)
#define PP_ERR_INTERNAL (-4)

/*
 * The ABI version this library was compiled against. The loader compares it with its own before calling
 * anything else.
 */
int32_t pp_abi_version(void);

/*
 * One line naming the compiler and target the library was built with, so a report can say what ran.
 * Never NULL.
 */
const char *pp_build(void);

/*
 * Writes the operations this library serves, one per line, as "name<TAB>signature". `needed` receives the
 * length of the full listing (+1 for the NUL) whatever `cap` is; a caller that got a short buffer knows how
 * much to allocate. Returns PP_OK, or PP_ERR_OUTPUT_SIZE when `cap` was too small (the listing is still
 * truncated into `buf`).
 */
int32_t pp_ops(char *buf, int32_t cap, int32_t *needed);

/*
 * Calls one operation.
 *
 *   op             one of the names pp_ops() advertised
 *   inputs         n_inputs pointers, each to input_sizes[i] doubles
 *   input_sizes    length of each input
 *   n_inputs       how many inputs the operation takes
 *   params         n_params scalars (sizes, spacings, iteration counts), in the order the signature gives
 *   out            the caller's buffer, out_capacity doubles long
 *   out_capacity   how many doubles fit in `out`
 *   out_size       receives how many doubles were written
 *   error          receives the sentence for a non-zero return; may be NULL
 *   error_capacity how many bytes fit in `error`
 *
 * Returns PP_OK on success.
 */
int32_t pp_call(const char *op,
                const double *const *inputs, const int64_t *input_sizes, int32_t n_inputs,
                const double *params, int32_t n_params,
                double *out, int64_t out_capacity, int64_t *out_size,
                char *error, int32_t error_capacity);

#endif /* PINNEAPPLE_NATIVE_H */
