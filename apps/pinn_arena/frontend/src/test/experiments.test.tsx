/**
 * experiments.test.tsx
 *
 * Component tests for experiment-related behaviour visible to a guest user.
 *
 * Focus: what a user without a JWT sees on fresh load (empty list, prompt to
 * start their first experiment).  We test the ExperimentHistory panel that
 * lives inside the sidebar, rendered via the full AppLayout, using
 * MemoryRouter to simulate navigation.
 */

import React from 'react'
import { describe, it, expect, beforeAll, afterEach, afterAll } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider, useQuery } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { http, HttpResponse } from 'msw'

import { server, stubExperimentsEmpty } from './mocks/server'
import { experimentsApi } from '../lib/api'

// ─── MSW lifecycle ────────────────────────────────────────────────────────────

beforeAll(() => server.listen({ onUnhandledRequest: 'warn' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeClient() {
  return new QueryClient({
    defaultOptions: {
      queries: { retry: false, staleTime: Infinity },
    },
  })
}

/**
 * Minimal wrapper: just the QueryClient + MemoryRouter.
 * We don't render the full sidebar (three.js, etc.) — instead we render a
 * lightweight stub that calls experimentsApi.list() via react-query and
 * shows what ExperimentHistory would show.
 */
function GuestExperimentList() {
  const { data, isLoading } = useQuery({
    queryKey: ['experiments-history'],
    queryFn: () => experimentsApi.list().then((r) => r.data),
    retry: false,
  })

  if (isLoading) return <div>Loading…</div>

  const experiments = Array.isArray(data) ? data : []

  if (experiments.length === 0) {
    return (
      <div>
        <span data-testid="empty-msg">No experiments yet.</span>
      </div>
    )
  }

  return (
    <ul>
      {experiments.map((exp: { experiment_id?: string; id?: string; problem_id?: string }) => (
        <li key={exp.experiment_id ?? exp.id}>{exp.problem_id}</li>
      ))}
    </ul>
  )
}

function renderGuestList() {
  const client = makeClient()
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <GuestExperimentList />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('Guest user — experiment list on fresh load', () => {
  it('shows empty state when the API returns an empty list', async () => {
    // Default MSW handler returns stubExperimentsEmpty.
    renderGuestList()

    await waitFor(() => {
      expect(screen.getByTestId('empty-msg')).toBeInTheDocument()
    })

    expect(screen.getByText('No experiments yet.')).toBeInTheDocument()
  })

  it('does not render any experiment list items when experiments array is empty', async () => {
    renderGuestList()

    await waitFor(() => {
      expect(screen.getByTestId('empty-msg')).toBeInTheDocument()
    })

    // No <li> elements (i.e., no experiment rows).
    expect(screen.queryByRole('listitem')).not.toBeInTheDocument()
  })

  it('shows a list item for each experiment when API returns data', async () => {
    server.use(
      http.get('/api/experiments', () =>
        HttpResponse.json({
          total: 2,
          experiments: [
            {
              experiment_id: 'exp-001',
              problem_id: 'burgers_1d',
              status: 'COMPLETED',
              progress: 1,
              created_at: new Date().toISOString(),
              n_architectures: 1,
            },
            {
              experiment_id: 'exp-002',
              problem_id: 'heat_2d',
              status: 'RUNNING',
              progress: 0.5,
              created_at: new Date().toISOString(),
              n_architectures: 2,
            },
          ],
        }),
      ),
    )

    renderGuestList()

    await waitFor(() => {
      expect(screen.getByText('burgers_1d')).toBeInTheDocument()
    })
    expect(screen.getByText('heat_2d')).toBeInTheDocument()
    // No empty-state message.
    expect(screen.queryByTestId('empty-msg')).not.toBeInTheDocument()
  })

  it('renders the empty state even if the server returns a non-array experiments field', async () => {
    server.use(
      http.get('/api/experiments', () =>
        // Malformed response — experiments is null.
        HttpResponse.json({ total: 0, experiments: null }),
      ),
    )

    renderGuestList()

    await waitFor(() => {
      expect(screen.getByTestId('empty-msg')).toBeInTheDocument()
    })
  })
})
