/**
 * MSW mock server for frontend tests.
 *
 * Provides realistic stub responses for every API endpoint the Arena
 * frontend consumes.  Tests import `server` and can call
 * `server.use(http.get(...))` to override individual handlers per test.
 */
import { setupServer } from 'msw/node'
import { http, HttpResponse } from 'msw'

// ─── Shared stub data ─────────────────────────────────────────────────────────

export const stubLeaderboardEmpty = {
  total_experiments: 0,
  total_architectures: 0,
  problems_covered: 0,
  entries: [],
}

export const stubLeaderboardWithData = {
  total_experiments: 5,
  total_architectures: 3,
  problems_covered: 2,
  entries: [
    {
      rank: 1,
      architecture_id: 'vanilla_pinn',
      architecture_name: 'Vanilla PINN',
      family: 'PINN',
      n_experiments: 3,
      n_wins: 2,
      win_rate: 0.6667,
      avg_l2_error: 0.0042,
      avg_rmse: 0.0038,
      avg_r2_score: 0.997,
      avg_max_error: 0.012,
      avg_training_time_s: 12.4,
      convergence_rate: 1.0,
      problems_count: 2,
    },
    {
      rank: 2,
      architecture_id: 'fourier_pinn',
      architecture_name: 'Fourier PINN',
      family: 'PINN',
      n_experiments: 2,
      n_wins: 1,
      win_rate: 0.5,
      avg_l2_error: 0.0081,
      avg_rmse: 0.0076,
      avg_r2_score: 0.992,
      avg_max_error: 0.02,
      avg_training_time_s: 14.1,
      convergence_rate: 1.0,
      problems_count: 1,
    },
    {
      rank: 3,
      architecture_id: 'deeponet',
      architecture_name: 'DeepONet',
      family: 'NEURAL_OPERATOR',
      n_experiments: 1,
      n_wins: 0,
      win_rate: 0.0,
      avg_l2_error: 0.015,
      avg_rmse: 0.013,
      avg_r2_score: 0.98,
      avg_max_error: 0.04,
      avg_training_time_s: 18.0,
      convergence_rate: 1.0,
      problems_count: 1,
    },
  ],
}

export const stubExperimentsEmpty = {
  total: 0,
  experiments: [],
}

export const stubProblems = {
  total: 34,
  problems: [
    {
      id: 'burgers_1d',
      name: 'Burgers (1D)',
      description: '1D Burgers equation',
      domain: 'FLUID_MECHANICS',
      dimension: 'ONE_D',
      equations: ['u_t + u*u_x = nu*u_xx'],
      difficulty: 'BEGINNER',
      requires_geometry: false,
      tags: [],
    },
  ],
}

export const stubArchitectures = {
  total: 93,
  architectures: [
    {
      id: 'vanilla_pinn',
      name: 'Vanilla PINN',
      family: 'PINN',
      description: 'Standard physics-informed neural network',
      supports_physics_loss: true,
      supports_3d: true,
      inverse_only: false,
      best_for: ['burgers_1d'],
      parameters: {},
    },
  ],
}

// ─── Default handlers ─────────────────────────────────────────────────────────

export const handlers = [
  // Leaderboard — returns empty by default; individual tests override as needed
  http.get('/api/leaderboard', () =>
    HttpResponse.json(stubLeaderboardEmpty),
  ),

  // Experiments list — empty for guest
  http.get('/api/experiments', () =>
    HttpResponse.json(stubExperimentsEmpty),
  ),

  // Problems catalog
  http.get('/api/problems', () =>
    HttpResponse.json(stubProblems),
  ),

  // Architectures catalog
  http.get('/api/architectures', () =>
    HttpResponse.json(stubArchitectures),
  ),

  // Auth — current user (unauthenticated by default)
  http.get('/api/auth/me', () =>
    HttpResponse.json({ detail: 'Authentication required.' }, { status: 401 }),
  ),

  // Health
  http.get('/api/health', () =>
    HttpResponse.json({ status: 'ok', api_version: '1.0', torch_available: true }),
  ),
]

// ─── Server instance (exported for test setup/teardown) ───────────────────────

export const server = setupServer(...handlers)
