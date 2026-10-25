import { create } from 'zustand'

// ─── Types ────────────────────────────────────────────────────────────────────

export interface AuthUser {
  github_id: number
  username: string
  name?: string
  email?: string
  avatar_url?: string
}

interface AuthStore {
  user: AuthUser | null
  token: string | null
  isLoading: boolean
  setUser: (user: AuthUser | null) => void
  setToken: (token: string | null) => void
  setLoading: (loading: boolean) => void
  logout: () => void
}

// ─── Store ────────────────────────────────────────────────────────────────────

const LOCAL_STORAGE_KEY = 'arena_token'

export const useAuthStore = create<AuthStore>((set) => ({
  user: null,
  token: localStorage.getItem(LOCAL_STORAGE_KEY),
  isLoading: true,

  setUser: (user) => set({ user }),

  setToken: (token) => {
    if (token) {
      localStorage.setItem(LOCAL_STORAGE_KEY, token)
    } else {
      localStorage.removeItem(LOCAL_STORAGE_KEY)
    }
    set({ token })
  },

  setLoading: (loading) => set({ isLoading: loading }),

  logout: () => {
    localStorage.removeItem(LOCAL_STORAGE_KEY)
    set({ user: null, token: null })
  },
}))
