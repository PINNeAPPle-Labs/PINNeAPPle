/**
 * auth.test.tsx
 *
 * Tests for the _INITIAL_HASH_TOKEN logic in App.tsx.
 *
 * _INITIAL_HASH_TOKEN is evaluated at module-load time from
 * window.location.hash, so we manipulate jsdom's location before importing
 * the module, then reset the module registry between tests.
 */

import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest'

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Set window.location.hash and re-import App.tsx so the IIFE at module scope
 * runs with the new hash value.
 *
 * vitest's module cache must be invalidated between calls; we do that via
 * vi.resetModules() in beforeEach/afterEach.
 */
async function importAppWithHash(hash: string): Promise<string | null> {
  // jsdom allows setting location.hash directly
  window.location.hash = hash

  // Dynamically import so the module-level IIFE runs fresh each time.
  // We export _INITIAL_HASH_TOKEN from a thin wrapper to avoid importing
  // the full component tree (which needs canvas, WebGL, etc.)
  const mod = await import('../test/helpers/hashToken')
  return mod.readHashToken()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

describe('_INITIAL_HASH_TOKEN — hash token extraction', () => {
  beforeEach(() => {
    // Clear module cache so the IIFE re-runs with the fresh hash each test.
    vi.resetModules()
    // Restore replaceState so it works normally inside jsdom.
    vi.restoreAllMocks()
  })

  afterEach(() => {
    // Leave the hash clean so other tests aren't affected.
    window.location.hash = ''
  })

  it('returns null when the URL has no hash', async () => {
    window.location.hash = ''
    const token = await importAppWithHash('')
    expect(token).toBeNull()
  })

  it('returns null when the hash contains unrelated params', async () => {
    const token = await importAppWithHash('#foo=bar&baz=qux')
    expect(token).toBeNull()
  })

  it('captures the token from #token=<value>', async () => {
    const token = await importAppWithHash('#token=abc123xyz')
    expect(token).toBe('abc123xyz')
  })

  it('clears the hash from the URL after capturing the token', async () => {
    // Spy on replaceState to verify it was called to strip the hash.
    const replaceSpy = vi.spyOn(window.history, 'replaceState')

    await importAppWithHash('#token=secret-jwt')

    expect(replaceSpy).toHaveBeenCalled()
    // After stripping, the hash should not appear in the URL.
    const call = replaceSpy.mock.calls[0]
    const url = String(call[2] ?? '')
    expect(url).not.toContain('#token=')
  })

  it('handles a token that is part of a longer hash query string', async () => {
    const token = await importAppWithHash('#token=my-token&other=value')
    expect(token).toBe('my-token')
  })
})
