/**
 * Thin helper that re-implements the _INITIAL_HASH_TOKEN IIFE from App.tsx
 * so auth tests can exercise the logic without importing React, framer-motion,
 * three.js or any other heavy dep that would fail in jsdom.
 *
 * The function is intentionally not cached — callers should reset the module
 * registry (vi.resetModules()) between tests so each import re-runs the logic
 * against the current window.location.hash.
 */
export function readHashToken(): string | null {
  try {
    const hashParams = new URLSearchParams(window.location.hash.slice(1))
    const t = hashParams.get('token')
    if (t) {
      window.history.replaceState({}, '', window.location.pathname + window.location.search)
    }
    return t
  } catch {
    return null
  }
}
