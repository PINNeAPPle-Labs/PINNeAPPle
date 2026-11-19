/*
 * kernels.c -- the reference implementation of the PINNeAPPle native kernel ABI (pinneapple_native.h).
 *
 * Three kernels of the same five-point stencil, which is where an interpreter's per-element cost stops
 * being academic: a 512 x 512 grid is a quarter of a million interior nodes per sweep.
 *
 *   laplacian_2d  one pass of the discrete Laplacian
 *   heat_step_2d  n explicit steps of u_t = alpha * laplacian(u)
 *   jacobi_2d     n sweeps of the Jacobi iteration for -laplacian(u) = f
 *
 * Every kernel uses zero Dirichlet data on the boundary nodes; the boundary of `out` is written as 0, and the
 * boundary of the input is never read. The Python side of the repository has the same three kernels written
 * in NumPy (pinneapple_core/native/kernels.py), and a test asserts the two agree.
 */
#include "pinneapple_native.h"

#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Written by build.py with -DPP_BUILD="..."; the default covers a hand build from this file. */
#ifndef PP_BUILD
#define PP_BUILD "unrecorded build"
#endif

int32_t pp_abi_version(void) { return PP_NATIVE_ABI_VERSION; }

const char *pp_build(void) { return PP_BUILD; }

static const char *const PP_OP_LISTING =
    "laplacian_2d\t(u[nx*ny]; nx, ny, h) -> lap[nx*ny]\n"
    "heat_step_2d\t(u[nx*ny]; nx, ny, alpha, dt, steps) -> u[nx*ny]\n"
    "jacobi_2d\t(u0[nx*ny], f[nx*ny]; nx, ny, h, steps, omega) -> u[nx*ny]\n";

int32_t pp_ops(char *buf, int32_t cap, int32_t *needed) {
    const size_t n = strlen(PP_OP_LISTING) + 1;
    if (needed != NULL) {
        *needed = (int32_t)n;
    }
    if (buf == NULL || cap <= 0) {
        return PP_ERR_OUTPUT_SIZE;
    }
    const size_t room = (size_t)cap;
    const size_t copy = n < room ? n : room;
    memcpy(buf, PP_OP_LISTING, copy);
    buf[copy - 1] = '\0';
    return copy < n ? PP_ERR_OUTPUT_SIZE : PP_OK;
}

static int32_t fail(char *error, int32_t error_capacity, int32_t code, const char *message) {
    if (error != NULL && error_capacity > 0) {
        const size_t n = strlen(message) + 1;
        const size_t copy = n < (size_t)error_capacity ? n : (size_t)error_capacity;
        memcpy(error, message, copy);
        error[copy - 1] = '\0';
    }
    return code;
}

/* A grid size arrives as a double and must be an exact integer >= 3 (a boundary ring plus one interior node). */
static int read_size(double value, int64_t *out, char *error, int32_t error_capacity, const char *name) {
    if (!(value >= 3.0) || value != floor(value) || value > 1e9) {
        char message[128];
        snprintf(message, sizeof message, "%s must be an integer >= 3, got %g", name, value);
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, message);
    }
    *out = (int64_t)value;
    return PP_OK;
}

/* Contiguous row-major grid indexed i * ny + j for i in [0, nx) and j in [0, ny). */
typedef struct {
    int64_t nx;
    int64_t ny;
} grid;

static int32_t read_grid(const double *params, int32_t n_params, grid *g, char *error, int32_t error_capacity) {
    if (params == NULL || n_params < 2) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "the first two params must be nx and ny");
    }
    int32_t status = read_size(params[0], &g->nx, error, error_capacity, "nx");
    if (status != PP_OK) {
        return status;
    }
    return read_size(params[1], &g->ny, error, error_capacity, "ny");
}

static int32_t check_input(const double *values, int64_t size, int64_t expected, int32_t index,
                           char *error, int32_t error_capacity) {
    if (values == NULL) {
        char message[128];
        snprintf(message, sizeof message, "input %d is NULL", index);
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, message);
    }
    if (size != expected) {
        char message[160];
        snprintf(message, sizeof message, "input %d has %lld values, the grid needs %lld", index, (long long)size,
                 (long long)expected);
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, message);
    }
    return PP_OK;
}

/* 5-point Laplacian of the interior, h on both axes (hx = hy = h keeps the ABI small; an anisotropic grid
 * is a second signature, not an extra pair of params that every caller would have to pass). */
static void laplacian_into(const double *u, double *lap, grid g, double h) {
    const double inv_h2 = 1.0 / (h * h);
    for (int64_t i = 0; i < g.nx; i++) {
        lap[i * g.ny + 0] = 0.0;
        lap[i * g.ny + (g.ny - 1)] = 0.0;
    }
    for (int64_t j = 0; j < g.ny; j++) {
        lap[0 * g.ny + j] = 0.0;
        lap[(g.nx - 1) * g.ny + j] = 0.0;
    }
    for (int64_t i = 1; i < g.nx - 1; i++) {
        const double *row = u + (i - 1) * g.ny;
        const double *mid = u + i * g.ny;
        const double *row_next = u + (i + 1) * g.ny;
        double *out = lap + i * g.ny;
        for (int64_t j = 1; j < g.ny - 1; j++) {
            const double nn = mid[j - 1] + mid[j + 1] + row[j] + row_next[j];
            out[j] = (nn - 4.0 * mid[j]) * inv_h2;
        }
    }
}

static int32_t op_laplacian_2d(const double *const *inputs, const int64_t *sizes, int32_t n_inputs,
                               const double *params, int32_t n_params, double *out, int64_t out_capacity,
                               int64_t *out_size, char *error, int32_t error_capacity) {
    grid g;
    int32_t status = read_grid(params, n_params, &g, error, error_capacity);
    if (status != PP_OK) {
        return status;
    }
    if (n_params < 3) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "params must be (nx, ny, h)");
    }
    const int64_t cells = g.nx * g.ny;
    if (n_inputs != 1) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "laplacian_2d takes exactly one input");
    }
    status = check_input(inputs[0], sizes[0], cells, 0, error, error_capacity);
    if (status != PP_OK) {
        return status;
    }
    if (out_capacity < cells) {
        return fail(error, error_capacity, PP_ERR_OUTPUT_SIZE, "out is smaller than nx*ny");
    }
    const double h = params[2];
    if (!(h > 0.0)) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "h must be positive");
    }
    laplacian_into(inputs[0], out, g, h);
    *out_size = cells;
    return PP_OK;
}

static int32_t op_heat_step_2d(const double *const *inputs, const int64_t *sizes, int32_t n_inputs,
                               const double *params, int32_t n_params, double *out, int64_t out_capacity,
                               int64_t *out_size, char *error, int32_t error_capacity) {
    grid g;
    int32_t status = read_grid(params, n_params, &g, error, error_capacity);
    if (status != PP_OK) {
        return status;
    }
    if (n_params < 5) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "params must be (nx, ny, alpha, dt, steps)");
    }
    const int64_t cells = g.nx * g.ny;
    if (n_inputs != 1) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "heat_step_2d takes exactly one input");
    }
    status = check_input(inputs[0], sizes[0], cells, 0, error, error_capacity);
    if (status != PP_OK) {
        return status;
    }
    if (out_capacity < cells) {
        return fail(error, error_capacity, PP_ERR_OUTPUT_SIZE, "out is smaller than nx*ny");
    }
    const double alpha = params[2];
    const double dt = params[3];
    const double steps_d = params[4];
    if (!(alpha > 0.0) || !(dt > 0.0) || !(steps_d >= 0.0) || steps_d != floor(steps_d)) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "alpha and dt must be positive and steps a whole number");
    }
    const int64_t steps = (int64_t)steps_d;
    /* h is taken as 1 / (nx - 1) on both axes: the unit square, which is the case every example uses. Making
     * it a parameter would change the signature without changing the arithmetic. */
    const double hx = 1.0 / (double)(g.nx - 1);
    const double hy = 1.0 / (double)(g.ny - 1);
    const double rx = alpha * dt / (hx * hx);
    const double ry = alpha * dt / (hy * hy);
    if (rx + ry > 0.5) {
        /* The explicit scheme is unstable there; returning a blowing-up field would look like a bug in us. */
        return fail(error, error_capacity, PP_ERR_BAD_ARGS,
                    "alpha*dt*(1/hx^2 + 1/hy^2) > 0.5: the explicit step would be unstable");
    }
    memcpy(out, inputs[0], (size_t)cells * sizeof(double));
    double *next = (double *)malloc((size_t)cells * sizeof(double));
    if (next == NULL) {
        return fail(error, error_capacity, PP_ERR_INTERNAL, "could not allocate the scratch buffer");
    }
    for (int64_t step = 0; step < steps; step++) {
        for (int64_t j = 0; j < g.ny; j++) {
            next[j] = 0.0;
            next[(g.nx - 1) * g.ny + j] = 0.0;
        }
        for (int64_t i = 1; i < g.nx - 1; i++) {
            next[i * g.ny] = 0.0;
            next[i * g.ny + (g.ny - 1)] = 0.0;
            for (int64_t j = 1; j < g.ny - 1; j++) {
                const int64_t k = i * g.ny + j;
                next[k] = out[k] + rx * (out[k - 1] - 2.0 * out[k] + out[k + 1]) +
                          ry * (out[k - g.ny] - 2.0 * out[k] + out[k + g.ny]);
            }
        }
        memcpy(out, next, (size_t)cells * sizeof(double));
    }
    free(next);
    *out_size = cells;
    return PP_OK;
}

static int32_t op_jacobi_2d(const double *const *inputs, const int64_t *sizes, int32_t n_inputs,
                            const double *params, int32_t n_params, double *out, int64_t out_capacity,
                            int64_t *out_size, char *error, int32_t error_capacity) {
    grid g;
    int32_t status = read_grid(params, n_params, &g, error, error_capacity);
    if (status != PP_OK) {
        return status;
    }
    if (n_params < 5) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "params must be (nx, ny, h, steps, omega)");
    }
    const int64_t cells = g.nx * g.ny;
    if (n_inputs != 2) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "jacobi_2d takes two inputs: u0 and f");
    }
    status = check_input(inputs[0], sizes[0], cells, 0, error, error_capacity);
    if (status != PP_OK) {
        return status;
    }
    status = check_input(inputs[1], sizes[1], cells, 1, error, error_capacity);
    if (status != PP_OK) {
        return status;
    }
    if (out_capacity < cells) {
        return fail(error, error_capacity, PP_ERR_OUTPUT_SIZE, "out is smaller than nx*ny");
    }
    const double h = params[2];
    const double steps_d = params[3];
    const double omega = params[4];
    if (!(h > 0.0) || !(steps_d >= 0.0) || steps_d != floor(steps_d) || !(omega > 0.0) || omega > 1.0) {
        /* Damped Jacobi is stable only for omega in (0, 1]: the iteration matrix has eigenvalues
         * 1 - omega*(1 - mu) with mu = (cos(i*pi*h) + cos(j*pi*h))/2 up to cos(pi*h), so omega > 1 can put
         * one of them below -1 and the field overflows instead of converging. Returning inf would look like
         * a bug in the kernel, so the range is part of the contract. */
        return fail(error, error_capacity, PP_ERR_BAD_ARGS,
                    "h must be positive, steps a whole number and omega in (0, 1] (damped Jacobi is unstable above 1)");
    }
    const int64_t steps = (int64_t)steps_d;
    const double h2 = h * h;
    const double *u0 = inputs[0];
    const double *f = inputs[1];
    memcpy(out, u0, (size_t)cells * sizeof(double));
    double *next = (double *)malloc((size_t)cells * sizeof(double));
    if (next == NULL) {
        return fail(error, error_capacity, PP_ERR_INTERNAL, "could not allocate the scratch buffer");
    }
    const double keep = 1.0 - omega;
    for (int64_t step = 0; step < steps; step++) {
        for (int64_t j = 0; j < g.ny; j++) {
            next[j] = 0.0;
            next[(g.nx - 1) * g.ny + j] = 0.0;
        }
        for (int64_t i = 1; i < g.nx - 1; i++) {
            next[i * g.ny] = 0.0;
            next[i * g.ny + (g.ny - 1)] = 0.0;
            for (int64_t j = 1; j < g.ny - 1; j++) {
                const int64_t k = i * g.ny + j;
                const double nn = out[k - 1] + out[k + 1] + out[k - g.ny] + out[k + g.ny];
                next[k] = keep * out[k] + omega * 0.25 * (nn + h2 * f[k]);
            }
        }
        memcpy(out, next, (size_t)cells * sizeof(double));
    }
    free(next);
    *out_size = cells;
    return PP_OK;
}

int32_t pp_call(const char *op,
                const double *const *inputs, const int64_t *input_sizes, int32_t n_inputs,
                const double *params, int32_t n_params,
                double *out, int64_t out_capacity, int64_t *out_size,
                char *error, int32_t error_capacity) {
    if (op == NULL) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "op is NULL");
    }
    if (out == NULL || out_size == NULL) {
        return fail(error, error_capacity, PP_ERR_BAD_ARGS, "out and out_size are required");
    }
    *out_size = 0;
    if (strcmp(op, "laplacian_2d") == 0) {
        return op_laplacian_2d(inputs, input_sizes, n_inputs, params, n_params, out, out_capacity, out_size,
                               error, error_capacity);
    }
    if (strcmp(op, "heat_step_2d") == 0) {
        return op_heat_step_2d(inputs, input_sizes, n_inputs, params, n_params, out, out_capacity, out_size,
                               error, error_capacity);
    }
    if (strcmp(op, "jacobi_2d") == 0) {
        return op_jacobi_2d(inputs, input_sizes, n_inputs, params, n_params, out, out_capacity, out_size,
                            error, error_capacity);
    }
    return fail(error, error_capacity, PP_ERR_UNKNOWN_OP, "unknown op: call pp_ops() for the list");
}
