import React, { useMemo, useEffect, useRef, useState, useCallback } from 'react'
import { BrowserRouter, Routes, Route, Navigate, NavLink, useLocation, useNavigate } from 'react-router-dom'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  FlaskConical, BarChart3, ChevronRight, Trophy,
  History, Clock, CheckCircle2, XCircle, Loader2, Plus,
  Github, LogOut, User, Menu, X, Sun, Moon, RefreshCw,
  Boxes, GitCompare, Gauge, Database,
} from 'lucide-react'
import clsx from 'clsx'
import { experimentsApi, authApi, externalCatalogApi } from './lib/api'
import { useAuthStore } from './store/authStore'
import { useExperimentStore } from './store/experimentStore'
import { ProblemsPage } from './pages/ProblemsPage'
import { ExperimentSetupPage } from './pages/ExperimentSetupPage'
import { TrainingPage } from './pages/TrainingPage'
import { ResultsPage } from './pages/ResultsPage'
import { LeaderboardPage } from './pages/LeaderboardPage'
import { ExternalCatalogPage } from './pages/external/ExternalCatalogPage'
import { ExternalModelDetailPage } from './pages/external/ExternalModelDetailPage'
import { ExternalComparePage } from './pages/external/ExternalComparePage'
import { ExternalBenchmarkPage } from './pages/external/ExternalBenchmarkPage'
import { ExternalDatasetsPage } from './pages/external/ExternalDatasetsPage'

// ─── Dark mode hook ───────────────────────────────────────────────────────────

function useDarkMode(): [boolean, () => void] {
  const [dark, setDark] = useState<boolean>(() => {
    try { return localStorage.getItem('theme') === 'dark' } catch { return false }
  })

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    try { localStorage.setItem('theme', dark ? 'dark' : 'light') } catch {}
  }, [dark])

  const toggle = useCallback(() => setDark((d) => !d), [])
  return [dark, toggle]
}

// ─── Types ────────────────────────────────────────────────────────────────────

interface ExperimentSummary {
  experiment_id: string
  id: string
  status: string
  progress: number
  created_at: string
  updated_at: string
  problem_id: string
  n_architectures: number
  winner?: string
  current_model?: string
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

const pageVariants = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -8 },
}

function formatProblemId(id: string): string {
  if (!id) return 'Experiment'
  return id
    .split('_')
    .map((w) => {
      if (w === '2d') return '2D'
      if (w === '1d') return '1D'
      if (w === '3d') return '3D'
      if (w === 'ns') return 'NS'
      return w.charAt(0).toUpperCase() + w.slice(1)
    })
    .join(' ')
}

function groupByDate(experiments: ExperimentSummary[]) {
  const now = new Date()
  const todayStart = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime()
  const yesterdayStart = todayStart - 86_400_000
  const lastWeekStart = todayStart - 7 * 86_400_000

  const today: ExperimentSummary[] = []
  const yesterday: ExperimentSummary[] = []
  const lastWeek: ExperimentSummary[] = []
  const older: ExperimentSummary[] = []

  for (const exp of experiments) {
    const t = exp.created_at ? new Date(exp.created_at).getTime() : 0
    if (t >= todayStart) today.push(exp)
    else if (t >= yesterdayStart) yesterday.push(exp)
    else if (t >= lastWeekStart) lastWeek.push(exp)
    else older.push(exp)
  }

  const groups: { label: string; items: ExperimentSummary[] }[] = []
  if (today.length) groups.push({ label: 'Today', items: today })
  if (yesterday.length) groups.push({ label: 'Yesterday', items: yesterday })
  if (lastWeek.length) groups.push({ label: 'Last 7 days', items: lastWeek })
  if (older.length) groups.push({ label: 'Older', items: older })
  return groups
}

// ─── ExperimentHistoryItem ────────────────────────────────────────────────────

function ExperimentHistoryItem({
  exp,
  onClick,
}: {
  exp: ExperimentSummary
  onClick: () => void
}) {
  const status = (exp.status ?? '').toLowerCase()

  const statusIcon = (() => {
    if (status === 'completed') return <CheckCircle2 size={11} className="text-success flex-shrink-0" />
    if (status === 'failed') return <XCircle size={11} className="text-red-500 flex-shrink-0" />
    if (status === 'running') return <Loader2 size={11} className="text-accent animate-spin flex-shrink-0" />
    return <Clock size={11} className="text-slate-400 flex-shrink-0" />
  })()

  const label = formatProblemId(exp.problem_id)

  return (
    <button
      onClick={onClick}
      title={label}
      className="w-full text-left px-2.5 py-2 rounded-lg hover:bg-surface-elevated transition-colors group"
    >
      <div className="flex items-center gap-1.5">
        {statusIcon}
        <span className="text-xs text-text-primary truncate flex-1 font-medium leading-tight">{label}</span>
        {exp.n_architectures > 0 && (
          <span className="text-[9px] text-text-muted bg-slate-100 rounded px-1 flex-shrink-0 leading-4">
            {exp.n_architectures}
          </span>
        )}
      </div>
      {status === 'running' && exp.current_model && (
        <div className="text-[9px] text-accent mt-0.5 ml-4 truncate">
          Training {exp.current_model}...
        </div>
      )}
    </button>
  )
}

// ─── ExperimentHistory ────────────────────────────────────────────────────────

function ExperimentHistory() {
  const navigate = useNavigate()
  const { reset } = useExperimentStore()
  const qc = useQueryClient()

  const { data: experiments, isLoading, isFetching } = useQuery<ExperimentSummary[]>({
    queryKey: ['experiments-history'],
    queryFn: () => experimentsApi.list().then((r) => r.data as unknown as ExperimentSummary[]),
    refetchInterval: 5_000,
    staleTime: 3_000,
  })

  const groups = useMemo(() => groupByDate(experiments ?? []), [experiments])

  const handleClick = (exp: ExperimentSummary) => {
    const status = (exp.status ?? '').toLowerCase()
    const id = exp.experiment_id ?? exp.id
    if (status === 'completed' || status === 'failed') {
      navigate(`/experiment/${id}/results`)
    } else {
      navigate(`/experiment/${id}/training`)
    }
  }

  return (
    <div className="flex flex-col flex-1 min-h-0 border-t border-border">
      {/* Section header */}
      <div className="px-3 pt-3 pb-1.5 flex items-center justify-between flex-shrink-0">
        <div className="flex items-center gap-1.5">
          <History size={11} className="text-text-muted" />
          <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">
            Experiments
          </span>
        </div>
        <button
          onClick={() => qc.invalidateQueries({ queryKey: ['experiments-history'] })}
          title="Refresh experiments"
          className="text-text-muted hover:text-accent transition-colors"
        >
          {isFetching
            ? <Loader2 size={10} className="animate-spin text-accent" />
            : <RefreshCw size={10} />}
        </button>
      </div>

      {/* Scrollable list */}
      <div className="flex-1 overflow-y-auto px-2 pb-2 space-y-0.5">
        {groups.length === 0 && !isLoading && (
          <div className="text-[10px] text-text-muted px-2 py-6 text-center">
            No experiments yet.
            <br />
            <button
              onClick={() => { reset(); navigate('/experiment/setup') }}
              className="text-accent hover:underline mt-1 inline-block"
            >
              Start your first one →
            </button>
          </div>
        )}

        {groups.map((group) => (
          <div key={group.label}>
            <div className="text-[9px] font-semibold text-text-muted uppercase tracking-wider px-2.5 pt-2 pb-1">
              {group.label}
            </div>
            {group.items.map((exp) => (
              <ExperimentHistoryItem
                key={exp.experiment_id ?? exp.id}
                exp={exp}
                onClick={() => handleClick(exp)}
              />
            ))}
          </div>
        ))}
      </div>
    </div>
  )
}

// ─── UserSection ──────────────────────────────────────────────────────────────

function UserSection() {
  const { user, logout } = useAuthStore()
  const [menuOpen, setMenuOpen] = React.useState(false)
  const menuRef = useRef<HTMLDivElement>(null)

  // Close menu when clicking outside
  useEffect(() => {
    if (!menuOpen) return
    const handler = (e: MouseEvent) => {
      if (menuRef.current && !menuRef.current.contains(e.target as Node)) {
        setMenuOpen(false)
      }
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [menuOpen])

  const handleLogout = () => {
    logout()
    setMenuOpen(false)
    authApi.logout().catch(() => {/* best-effort */})
  }

  if (!user) {
    return (
      <div className="px-3 pb-3 border-t border-border pt-3 flex-shrink-0">
        <a
          href="/api/auth/github/login"
          className="w-full flex items-center justify-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-text-secondary hover:text-text-primary hover:bg-surface-elevated border border-border transition-all duration-200"
        >
          <Github size={13} />
          Sign in with GitHub
        </a>
        <p className="text-[9px] text-text-muted text-center mt-2 leading-relaxed px-1">
          Experiments are stored only in this browser tab and will be lost on refresh.
          Sign in to save them permanently.
        </p>
      </div>
    )
  }

  return (
    <div className="px-3 pb-3 border-t border-border pt-3 flex-shrink-0 relative" ref={menuRef}>
      <button
        onClick={() => setMenuOpen((v) => !v)}
        className="w-full flex items-center gap-2 px-2.5 py-2 rounded-lg hover:bg-surface-elevated transition-colors text-left"
      >
        {user.avatar_url ? (
          <img
            src={user.avatar_url}
            alt={user.username}
            className="w-6 h-6 rounded-full flex-shrink-0"
          />
        ) : (
          <div className="w-6 h-6 rounded-full bg-accent/10 border border-accent/20 flex items-center justify-center flex-shrink-0">
            <User size={12} className="text-accent" />
          </div>
        )}
        <div className="flex-1 min-w-0">
          <div className="text-xs font-medium text-text-primary truncate">{user.username}</div>
          {user.name && (
            <div className="text-[9px] text-text-muted truncate">{user.name}</div>
          )}
        </div>
      </button>

      {/* Dropdown menu */}
      {menuOpen && (
        <div className="absolute bottom-full left-3 right-3 mb-1 bg-white border border-border rounded-lg shadow-lg overflow-hidden z-50">
          <div className="px-3 py-2 border-b border-border">
            <div className="text-[11px] font-medium text-text-primary">{user.username}</div>
            {user.email && (
              <div className="text-[10px] text-text-muted truncate">{user.email}</div>
            )}
          </div>
          <button
            onClick={handleLogout}
            className="w-full flex items-center gap-2 px-3 py-2 text-xs text-red-600 hover:bg-red-50 transition-colors"
          >
            <LogOut size={12} />
            Sign out
          </button>
        </div>
      )}
    </div>
  )
}

// ─── AnimatedRoutes ───────────────────────────────────────────────────────────

function AnimatedRoutes() {
  const location = useLocation()

  return (
    <AnimatePresence mode="wait">
      <Routes location={location} key={location.pathname}>
        <Route path="/" element={<Navigate to="/problems" replace />} />
        <Route
          path="/problems"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <ProblemsPage />
            </motion.div>
          }
        />
        <Route
          path="/experiment/setup"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <ExperimentSetupPage />
            </motion.div>
          }
        />
        <Route
          path="/experiment/:id/training"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <TrainingPage />
            </motion.div>
          }
        />
        <Route
          path="/experiment/:id/results"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <ResultsPage />
            </motion.div>
          }
        />
        <Route
          path="/leaderboard"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <LeaderboardPage />
            </motion.div>
          }
        />
        {/* External catalog — ~94 third-party physics-ML libraries, benchmarked
            alongside PINNeAPPle's own architectures above. */}
        <Route
          path="/external-catalog"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <ExternalCatalogPage />
            </motion.div>
          }
        />
        <Route
          path="/external-catalog/compare"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <ExternalComparePage />
            </motion.div>
          }
        />
        <Route
          path="/external-catalog/benchmark"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <ExternalBenchmarkPage />
            </motion.div>
          }
        />
        <Route
          path="/external-catalog/datasets"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <ExternalDatasetsPage />
            </motion.div>
          }
        />
        <Route
          path="/external-catalog/:slug"
          element={
            <motion.div {...pageVariants} transition={{ duration: 0.25 }} className="h-full">
              <ExternalModelDetailPage />
            </motion.div>
          }
        />
      </Routes>
    </AnimatePresence>
  )
}

// ─── Sidebar ──────────────────────────────────────────────────────────────────

interface SidebarProps {
  onClose?: () => void
  mobileOpen?: boolean
}

function Sidebar({ onClose, mobileOpen }: SidebarProps) {
  const navigate = useNavigate()
  const { data: catalogHealth } = useQuery({
    queryKey: ['external-catalog-health'],
    queryFn: () => externalCatalogApi.health().then((r) => r.data),
    staleTime: 5 * 60_000,
  })
  const { reset } = useExperimentStore()

  const navItems = [
    { to: '/problems', icon: <FlaskConical size={16} />, label: 'Problems' },
    { to: '/leaderboard', icon: <Trophy size={16} />, label: 'Leaderboard' },
  ]

  // External catalog — ~94 third-party physics-ML libraries, a clearly
  // separate section from PINNeAPPle's own architectures above.
  const externalNavItems = [
    { to: '/external-catalog', icon: <Boxes size={16} />, label: 'Catalog' },
    { to: '/external-catalog/compare', icon: <GitCompare size={16} />, label: 'Compare' },
    { to: '/external-catalog/benchmark', icon: <Gauge size={16} />, label: 'Benchmark' },
    { to: '/external-catalog/datasets', icon: <Database size={16} />, label: 'Datasets' },
  ]

  const handleNav = () => {
    if (onClose) onClose()
  }

  const startNewExperiment = () => {
    reset()
    navigate('/experiment/setup')
    handleNav()
  }

  const panel = (
    <aside className="w-60 flex-shrink-0 flex flex-col bg-white border-r border-border h-full">
      {/* Logo */}
      <div className="p-4 border-b border-border flex-shrink-0">
        <div className="flex items-center gap-2.5">
          <img
            src="/logo.svg"
            alt="PINNeAPPle Arena"
            className="w-8 h-8 flex-shrink-0"
          />
          <div className="flex-1 min-w-0">
            <div className="text-[13px] font-bold text-accent leading-none">PINNeAPPle</div>
            <div className="text-[11px] text-slate-700 font-semibold leading-none mt-0.5">Arena</div>
          </div>
          {/* Close button — mobile only */}
          {onClose && (
            <button
              onClick={onClose}
              className="md:hidden p-1 rounded-lg text-text-muted hover:bg-surface-elevated transition-colors"
            >
              <X size={16} />
            </button>
          )}
        </div>
        <p className="text-[9px] text-text-muted mt-2 leading-relaxed">
          Physics-Informed Neural Network Benchmarking
        </p>
      </div>

      {/* Top nav */}
      <nav className="p-3 space-y-0.5 flex-shrink-0">
        {navItems.map((item) => (
          <NavLink
            key={item.to}
            to={item.to}
            onClick={handleNav}
            className={({ isActive }) =>
              clsx(
                'flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200',
                isActive
                  ? 'bg-accent text-white'
                  : 'text-text-secondary hover:text-text-primary hover:bg-surface-elevated'
              )
            }
          >
            {item.icon}
            {item.label}
          </NavLink>
        ))}

        {/* New Experiment button */}
        <button
          onClick={startNewExperiment}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium text-text-secondary hover:text-text-primary hover:bg-surface-elevated transition-all duration-200"
        >
          <Plus size={16} />
          New Experiment
        </button>

        {/* External catalog — separate section: ~94 third-party physics-ML
            libraries benchmarked alongside PINNeAPPle's own architectures. */}
        <div className="pt-3 mt-1 border-t border-border">
          <div className="px-3 pb-1.5 text-[10px] font-semibold text-text-muted uppercase tracking-wider">
            External Catalog
          </div>
          {externalNavItems.map((item, i) => (
            <NavLink
              key={item.to}
              to={item.to}
              end={i === 0}
              onClick={handleNav}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium transition-all duration-200',
                  isActive
                    ? 'bg-accent text-white'
                    : 'text-text-secondary hover:text-text-primary hover:bg-surface-elevated'
                )
              }
            >
              {item.icon}
              {item.label}
            </NavLink>
          ))}
        </div>
      </nav>

      {/* Experiment History — takes all remaining vertical space */}
      <ExperimentHistory />

      {/* Backend status footer */}
      <div className="px-3 pt-3 pb-1 border-t border-border flex-shrink-0">
        <div className="flex items-center gap-2 text-[10px] text-text-muted">
          <div className="w-1.5 h-1.5 rounded-full bg-success animate-pulse flex-shrink-0" />
          <span>PINNeAPPle Backend</span>
        </div>
        <div className="text-[9px] text-text-muted mt-0.5 mb-2">93 Architectures
          {catalogHealth && <> · {catalogHealth.models_indexed} External Models · {catalogHealth.datasets_indexed} Dataset{catalogHealth.datasets_indexed !== 1 ? 's' : ''}</>}
        </div>
      </div>

      {/* User section — sign in / user profile */}
      <UserSection />
    </aside>
  )

  return (
    <>
      {/* Desktop: static sidebar */}
      <div className="hidden md:flex h-screen sticky top-0">
        {panel}
      </div>

      {/* Mobile: overlay drawer */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              key="backdrop"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="fixed inset-0 z-40 bg-black/40 md:hidden"
              onClick={onClose}
            />
            {/* Drawer */}
            <motion.div
              key="drawer"
              initial={{ x: '-100%' }}
              animate={{ x: 0 }}
              exit={{ x: '-100%' }}
              transition={{ type: 'spring', stiffness: 350, damping: 35 }}
              className="fixed inset-y-0 left-0 z-50 flex md:hidden"
            >
              {panel}
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  )
}

// ─── TopBar ───────────────────────────────────────────────────────────────────

interface TopBarProps {
  onMenuClick: () => void
  dark: boolean
  onToggleDark: () => void
}

function TopBar({ onMenuClick, dark, onToggleDark }: TopBarProps) {
  const location = useLocation()

  const breadcrumbs = (() => {
    const path = location.pathname
    if (path.startsWith('/leaderboard')) return [{ label: 'Leaderboard', to: '/leaderboard' }]
    if (path.startsWith('/problems')) return [{ label: 'Problems', to: '/problems' }]
    if (path.startsWith('/experiment/setup')) return [
      { label: 'Problems', to: '/problems' },
      { label: 'New Experiment', to: '/experiment/setup' },
    ]
    if (path.includes('/training')) return [
      { label: 'Problems', to: '/problems' },
      { label: 'Training', to: '#' },
    ]
    if (path.includes('/results')) return [
      { label: 'Problems', to: '/problems' },
      { label: 'Results', to: '#' },
    ]
    if (path.startsWith('/external-catalog/compare')) return [
      { label: 'External Catalog', to: '/external-catalog' },
      { label: 'Compare', to: '/external-catalog/compare' },
    ]
    if (path.startsWith('/external-catalog/benchmark')) return [
      { label: 'External Catalog', to: '/external-catalog' },
      { label: 'Benchmark', to: '/external-catalog/benchmark' },
    ]
    if (path.startsWith('/external-catalog/datasets')) return [
      { label: 'External Catalog', to: '/external-catalog' },
      { label: 'Datasets', to: '/external-catalog/datasets' },
    ]
    if (path.startsWith('/external-catalog/')) return [
      { label: 'External Catalog', to: '/external-catalog' },
      { label: 'Model', to: '#' },
    ]
    if (path.startsWith('/external-catalog')) return [
      { label: 'External Catalog', to: '/external-catalog' },
    ]
    return []
  })()

  return (
    <header className="h-11 border-b border-border bg-white flex items-center px-4 gap-2 flex-shrink-0 sticky top-0 z-10">
      {/* Hamburger — mobile only */}
      <button
        onClick={onMenuClick}
        className="md:hidden p-1.5 rounded-lg text-text-muted hover:bg-surface-elevated transition-colors flex-shrink-0"
        aria-label="Open menu"
      >
        <Menu size={18} />
      </button>

      {/* Logo — mobile only, shown beside hamburger */}
      <div className="md:hidden flex items-center gap-1.5 flex-shrink-0">
        <img src="/logo.svg" alt="PINNeAPPle Arena" className="w-5 h-5" />
        <span className="text-xs font-bold text-accent">Arena</span>
      </div>

      {/* Breadcrumbs — hidden on small mobile, visible from sm+ */}
      <div className="hidden sm:flex items-center gap-1.5 text-sm">
        {breadcrumbs.map((crumb, i) => (
          <React.Fragment key={crumb.label}>
            {i > 0 && <ChevronRight size={13} className="text-text-muted" />}
            <span className={i === breadcrumbs.length - 1 ? 'text-text-primary font-medium text-xs' : 'text-text-muted text-xs'}>
              {crumb.label}
            </span>
          </React.Fragment>
        ))}
      </div>

      <div className="ml-auto flex items-center gap-2">
        <div className="hidden sm:flex items-center gap-1.5 text-[11px] text-text-muted bg-surface-elevated border border-border rounded-lg px-2.5 py-1">
          <BarChart3 size={12} className="text-accent" />
          <span className="hidden md:inline">PINNeAPPle Arena v1.0</span>
          <span className="md:hidden">v1.0</span>
        </div>
        <button
          onClick={onToggleDark}
          title={dark ? 'Switch to light mode' : 'Switch to dark mode'}
          className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium text-text-secondary hover:text-text-primary bg-surface-elevated border border-border hover:border-border-strong transition-colors"
        >
          {dark ? <Sun size={13} className="text-amber-400" /> : <Moon size={13} className="text-slate-500" />}
          <span className="hidden sm:inline">{dark ? 'Light' : 'Dark'}</span>
        </button>
      </div>
    </header>
  )
}

// ─── AppLayout ────────────────────────────────────────────────────────────────

function AppLayout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [dark, toggleDark] = useDarkMode()
  const location = useLocation()

  // Close drawer on route change
  useEffect(() => { setSidebarOpen(false) }, [location.pathname])

  return (
    <div className="flex h-screen bg-background overflow-hidden">
      <Sidebar mobileOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopBar onMenuClick={() => setSidebarOpen(true)} dark={dark} onToggleDark={toggleDark} />
        <main className="flex-1 overflow-y-auto bg-background">
          <AnimatedRoutes />
        </main>
      </div>
    </div>
  )
}

// ─── AuthBootstrap ────────────────────────────────────────────────────────────
// Capture #token= BEFORE React/React Router renders — <Navigate> fires a
// history.replaceState via useLayoutEffect which clears the hash before any
// useEffect runs. Reading at module scope guarantees we get it.
const _INITIAL_HASH_TOKEN: string | null = (() => {
  try {
    const hashParams = new URLSearchParams(window.location.hash.slice(1))
    const t = hashParams.get('token')
    if (t) window.history.replaceState({}, '', window.location.pathname + window.location.search)
    return t
  } catch {
    return null
  }
})()

function AuthBootstrap({ children }: { children: React.ReactNode }) {
  const { token, setToken, setUser, setLoading } = useAuthStore()
  const hasBooted = useRef(false)

  useEffect(() => {
    if (hasBooted.current) return
    hasBooted.current = true

    const bootstrap = async () => {
      // 1. Use the token captured at module load (before React Router cleared the hash)
      const urlToken = _INITIAL_HASH_TOKEN

      if (urlToken) {
        setToken(urlToken)
      }

      // 2. Verify whatever token we have (from URL param or localStorage)
      const activeToken = urlToken ?? token
      if (!activeToken) {
        setLoading(false)
        return
      }

      try {
        const resp = await authApi.me()
        setUser(resp.data)
      } catch {
        // Token invalid / expired — clear it silently
        setToken(null)
        setUser(null)
      } finally {
        setLoading(false)
      }
    }

    bootstrap()
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  return <>{children}</>
}

// ─── App ──────────────────────────────────────────────────────────────────────

export default function App() {
  return (
    <BrowserRouter future={{ v7_startTransition: true, v7_relativeSplatPath: true }}>
      <AuthBootstrap>
        <AppLayout />
      </AuthBootstrap>
    </BrowserRouter>
  )
}
