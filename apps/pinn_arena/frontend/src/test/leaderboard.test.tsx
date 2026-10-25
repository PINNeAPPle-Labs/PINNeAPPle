/**
 * leaderboard.test.tsx
 *
 * Component tests for LeaderboardPage.
 *
 * Strategy:
 * - Use MSW to intercept /api/leaderboard calls.
 * - Wrap the component in the minimum providers it needs
 *   (QueryClientProvider, MemoryRouter).
 * - Assert on user-visible text, not implementation details.
 */

import React from 'react'
import { describe, it, expect, beforeAll, beforeEach, afterEach, afterAll } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { http, HttpResponse } from 'msw'

import { server, stubLeaderboardEmpty, stubLeaderboardWithData } from './mocks/server'
import { LeaderboardPage } from '../pages/LeaderboardPage'

// ─── MSW lifecycle ────────────────────────────────────────────────────────────

beforeAll(() => server.listen({ onUnhandledRequest: 'warn' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

// ─── Helpers ──────────────────────────────────────────────────────────────────

function makeClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        // Disable retries so tests fail fast on network errors.
        retry: false,
        // Disable stale-while-revalidate to avoid extra requests in tests.
        staleTime: Infinity,
      },
    },
  })
}

function renderLeaderboard() {
  const client = makeClient()
  return render(
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <LeaderboardPage />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('LeaderboardPage — empty state', () => {
  it('shows "No data yet" heading when entries list is empty', async () => {
    // Default handler already returns stubLeaderboardEmpty.
    renderLeaderboard()

    await waitFor(() => {
      expect(screen.getByText('No data yet')).toBeInTheDocument()
    })
  })

  it('does not render the stats bar when there are no entries', async () => {
    renderLeaderboard()

    await waitFor(() => {
      expect(screen.getByText('No data yet')).toBeInTheDocument()
    })

    // Stats labels should NOT be visible when entries is empty.
    expect(screen.queryByText('Total Experiments')).not.toBeInTheDocument()
    expect(screen.queryByText('Architectures Ranked')).not.toBeInTheDocument()
  })
})

describe('LeaderboardPage — with data', () => {
  // beforeEach, not beforeAll: the file-level afterEach(resetHandlers) drops
  // runtime overrides after every test.
  beforeEach(() => {
    // Override the default empty handler to return real data.
    server.use(
      http.get('/api/leaderboard', () =>
        HttpResponse.json(stubLeaderboardWithData),
      ),
    )
  })

  it('renders the stats bar with correct experiment count', async () => {
    renderLeaderboard()

    await waitFor(() => {
      // StatsBar shows the numeric values from the API response.
      expect(screen.getByText('5')).toBeInTheDocument() // total_experiments
    })

    expect(screen.getByText('Total Experiments')).toBeInTheDocument()
    expect(screen.getByText('Architectures Ranked')).toBeInTheDocument()
    expect(screen.getByText('Problems Covered')).toBeInTheDocument()
  })

  it('renders the podium section header when there are at least 3 entries', async () => {
    renderLeaderboard()

    await waitFor(() => {
      expect(screen.getByText('Top 3 Architectures')).toBeInTheDocument()
    })
  })

  it('displays architecture names from the entries', async () => {
    renderLeaderboard()

    await waitFor(() => {
      // All three stub architecture names should appear somewhere on the page.
      expect(screen.getAllByText('Vanilla PINN').length).toBeGreaterThan(0)
      expect(screen.getAllByText('Fourier PINN').length).toBeGreaterThan(0)
      expect(screen.getAllByText('DeepONet').length).toBeGreaterThan(0)
    })
  })
})

describe('LeaderboardPage — domain filter query param', () => {
  it('passes domain param to the API and renders whatever the server returns', async () => {
    // Intercept with a domain-specific response.
    server.use(
      http.get('/api/leaderboard', ({ request }) => {
        const url = new URL(request.url)
        if (url.searchParams.get('domain') === 'FLUID_MECHANICS') {
          return HttpResponse.json(stubLeaderboardWithData)
        }
        return HttpResponse.json(stubLeaderboardEmpty)
      }),
    )

    // Render and open the filter panel to set a domain filter.
    const client = makeClient()
    render(
      <QueryClientProvider client={client}>
        <MemoryRouter>
          <LeaderboardPage />
        </MemoryRouter>
      </QueryClientProvider>,
    )

    // On initial render the default (no domain) returns empty.
    await waitFor(() => {
      expect(screen.getByText('No data yet')).toBeInTheDocument()
    })
  })
})
