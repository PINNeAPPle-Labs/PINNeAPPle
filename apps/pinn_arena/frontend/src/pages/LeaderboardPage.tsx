import React, { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Trophy, Medal, BarChart3, FlaskConical, ChevronDown, ChevronUp,
  Cpu, Target, CheckCircle2, Clock, TrendingDown, Star,
  Filter, X, ChevronRight, Zap, HardDrive, Activity,
  Radar as RadarIcon,
} from 'lucide-react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  ResponsiveContainer, Tooltip, Legend,
} from 'recharts'
import clsx from 'clsx'
import { leaderboardApi, problemsApi } from '../lib/api'
import type { LeaderboardEntry, LeaderboardFilters, PhysicsArea, Dimension, Difficulty, MultiObjectiveScores } from '../types'
import { Badge } from '../components/ui/Badge'
import type { BadgeVariant } from '../components/ui/Badge'
import { FAMILY_CONFIG } from '../components/ArchitectureCard'
import { DOMAIN_CONFIG, DIMENSION_LABELS, DIFFICULTY_CONFIG } from '../components/ProblemCard'
import { FilterChip } from '../components/ui/FilterChip'
import { PageLoading } from '../components/ui/LoadingSpinner'

// ─── Constants ────────────────────────────────────────────────────────────────

const DOMAINS: PhysicsArea[] = [
  'FLUID_MECHANICS', 'SOLID_MECHANICS', 'ELECTROMAGNETISM',
  'QUANTUM_MECHANICS', 'THERMODYNAMICS', 'WAVE_PHYSICS', 'NONLINEAR_DYNAMICS',
]
const DIMENSIONS: Dimension[] = ['ONE_D', 'TWO_D', 'THREE_D']
const DIFFICULTIES: Difficulty[] = ['BEGINNER', 'INTERMEDIATE', 'ADVANCED']

// ─── Helpers ─────────────────────────────────────────────────────────────────

function fmt(v: number | null | undefined, decimals = 4): string {
  if (v == null) return '—'
  if (v === 0) return '0'
  if (Math.abs(v) < 0.001 || Math.abs(v) >= 10_000) {
    return v.toExponential(2)
  }
  return v.toFixed(decimals)
}

function pct(v: number): string {
  return `${(v * 100).toFixed(0)}%`
}

// ─── PodiumCard ───────────────────────────────────────────────────────────────

const PODIUM_CONFIG = [
  {
    place: 1,
    label: '1st',
    medal: '🥇',
    gradient: 'from-amber-50 to-amber-100',
    border: 'border-amber-300',
    ring: 'ring-amber-200',
    badgeBg: 'bg-amber-500',
    textColor: 'text-amber-700',
    height: 'h-48 sm:h-56',
  },
  {
    place: 2,
    label: '2nd',
    medal: '🥈',
    gradient: 'from-slate-50 to-slate-100',
    border: 'border-slate-300',
    ring: 'ring-slate-200',
    badgeBg: 'bg-slate-400',
    textColor: 'text-slate-600',
    height: 'h-40 sm:h-48',
  },
  {
    place: 3,
    label: '3rd',
    medal: '🥉',
    gradient: 'from-orange-50 to-orange-100',
    border: 'border-orange-300',
    ring: 'ring-orange-200',
    badgeBg: 'bg-orange-500',
    textColor: 'text-orange-700',
    height: 'h-36 sm:h-44',
  },
] as const

function PodiumCard({ entry, cfg }: { entry: LeaderboardEntry; cfg: typeof PODIUM_CONFIG[number] }) {
  const family = FAMILY_CONFIG[entry.family as keyof typeof FAMILY_CONFIG]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: (cfg.place - 1) * 0.1, type: 'spring', stiffness: 300, damping: 28 }}
      className={clsx(
        'relative flex flex-col items-center justify-end rounded-2xl border-2 shadow-lg p-4 pt-6 text-center',
        `bg-gradient-to-b ${cfg.gradient}`,
        cfg.border,
        `ring-2 ${cfg.ring}`,
        cfg.height,
        'w-full',
      )}
    >
      {/* Medal */}
      <div className="absolute -top-5 flex flex-col items-center">
        <span className="text-3xl leading-none drop-shadow-sm">{cfg.medal}</span>
        <span className={clsx('text-[10px] font-bold mt-0.5', cfg.textColor)}>{cfg.label}</span>
      </div>

      {/* Family badge */}
      {family && (
        <Badge
          variant={family.variant}
          size="xs"
          style={{ color: family.color, borderColor: `${family.color}30`, backgroundColor: `${family.color}10` } as React.CSSProperties}
          className="mb-1.5"
        >
          {family.label}
        </Badge>
      )}

      {/* Name */}
      <h3 className={clsx('text-xs font-bold leading-tight mb-2 px-1', cfg.textColor)}>
        {entry.architecture_name}
      </h3>

      {/* Key metric */}
      <div className="w-full space-y-1">
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-slate-500">Avg L2 err</span>
          <span className="font-mono font-semibold text-slate-700">{fmt(entry.avg_l2_error)}</span>
        </div>
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-slate-500">Win rate</span>
          <span className="font-semibold text-slate-700">{pct(entry.win_rate)}</span>
        </div>
        <div className="flex items-center justify-between text-[10px]">
          <span className="text-slate-500">Experiments</span>
          <span className="font-semibold text-slate-700">{entry.n_experiments}</span>
        </div>
      </div>
    </motion.div>
  )
}

// ─── Podium ───────────────────────────────────────────────────────────────────

function Podium({ entries }: { entries: LeaderboardEntry[] }) {
  const top3 = entries.slice(0, 3)
  // Arrange as 2nd | 1st | 3rd for classic podium look
  const arranged = [top3[1], top3[0], top3[2]].filter(Boolean)
  const cfgMap = { 0: PODIUM_CONFIG[1], 1: PODIUM_CONFIG[0], 2: PODIUM_CONFIG[2] } as const

  return (
    <div className="flex items-end justify-center gap-3 sm:gap-4 mt-8 mb-6 px-2">
      {arranged.map((entry, i) => entry && (
        <PodiumCard
          key={entry.architecture_id}
          entry={entry}
          cfg={cfgMap[i as 0 | 1 | 2]}
        />
      ))}
    </div>
  )
}

// ─── Metric cell ─────────────────────────────────────────────────────────────

function L2Cell({ value }: { value: number | null }) {
  if (value == null) return <span className="text-text-muted">—</span>
  const color = value < 0.01
    ? 'text-emerald-600'
    : value < 0.1
      ? 'text-amber-600'
      : 'text-red-500'
  return <span className={clsx('font-mono text-xs', color)}>{fmt(value)}</span>
}

function R2Cell({ value }: { value: number | null }) {
  if (value == null) return <span className="text-text-muted">—</span>
  const color = value > 0.99
    ? 'text-emerald-600'
    : value > 0.95
      ? 'text-amber-600'
      : 'text-red-500'
  return <span className={clsx('font-mono text-xs', color)}>{fmt(value, 3)}</span>
}

function MapeCell({ value }: { value: number | null | undefined }) {
  if (value == null) return <span className="text-text-muted">—</span>
  const color = value < 1
    ? 'text-emerald-600'
    : value < 10
      ? 'text-amber-600'
      : 'text-red-500'
  return <span className={clsx('font-mono text-xs', color)}>{value.toFixed(1)}%</span>
}

function ConvBar({ rate }: { rate: number }) {
  const pctVal = Math.round(rate * 100)
  const color = pctVal >= 80 ? 'bg-emerald-500' : pctVal >= 50 ? 'bg-amber-400' : 'bg-red-400'
  return (
    <div className="flex items-center gap-1.5">
      <div className="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden min-w-[40px]">
        <div className={clsx('h-full rounded-full', color)} style={{ width: `${pctVal}%` }} />
      </div>
      <span className="text-[10px] text-text-muted w-7 text-right">{pctVal}%</span>
    </div>
  )
}

// ─── RankingsTable ────────────────────────────────────────────────────────────

const RANK_MEDAL: Record<number, string> = { 1: '🥇', 2: '🥈', 3: '🥉' }

function RankingsTable({
  entries,
  showPodium,
}: {
  entries: LeaderboardEntry[]
  showPodium: boolean
}) {
  const [showAll, setShowAll] = useState(false)
  const startRank = showPodium ? 4 : 1
  const tableEntries = entries.filter((e) => e.rank >= startRank)
  const displayed = showAll ? tableEntries : tableEntries.slice(0, 10)
  const hidden = tableEntries.length - displayed.length

  if (tableEntries.length === 0) return null

  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <h2 className="text-sm font-semibold text-text-primary">
          {showPodium ? 'Full Rankings' : 'All Architectures'}
        </h2>
        {showPodium && (
          <span className="text-xs text-text-muted">Showing ranks 4+</span>
        )}
      </div>

      <div className="overflow-x-auto rounded-xl border border-border bg-white shadow-card">
        <table className="w-full">
          <thead className="bg-slate-50 border-b border-border">
            <tr>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap">Rank</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap">Architecture</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap">Avg L2 Error</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap hidden sm:table-cell">Avg MAPE</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap hidden sm:table-cell">Avg R²</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap hidden sm:table-cell">Avg RMSE</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap">Win Rate</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap hidden md:table-cell">Convergence</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap hidden md:table-cell">Experiments</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {displayed.map((entry, i) => {
              const family = FAMILY_CONFIG[entry.family as keyof typeof FAMILY_CONFIG]
              return (
                <motion.tr
                  key={entry.architecture_id}
                  initial={{ opacity: 0, x: -8 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.03 }}
                  className="hover:bg-slate-50 transition-colors"
                >
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-1.5">
                      {RANK_MEDAL[entry.rank] ? (
                        <span className="text-base leading-none">{RANK_MEDAL[entry.rank]}</span>
                      ) : (
                        <span className="w-6 h-6 rounded-full bg-slate-100 flex items-center justify-center text-[11px] font-bold text-text-muted">
                          {entry.rank}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex flex-col gap-1">
                      <span className="text-xs font-semibold text-text-primary leading-tight">
                        {entry.architecture_name}
                      </span>
                      {family && (
                        <Badge
                          variant={family.variant}
                          size="xs"
                          style={{ color: family.color, borderColor: `${family.color}30`, backgroundColor: `${family.color}10` } as React.CSSProperties}
                        >
                          {family.label}
                        </Badge>
                      )}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <L2Cell value={entry.avg_l2_error} />
                  </td>
                  <td className="py-3 px-4 hidden sm:table-cell">
                    <MapeCell value={entry.avg_mape} />
                  </td>
                  <td className="py-3 px-4 hidden sm:table-cell">
                    <R2Cell value={entry.avg_r2_score} />
                  </td>
                  <td className="py-3 px-4 hidden sm:table-cell">
                    <span className="font-mono text-xs text-text-secondary">{fmt(entry.avg_rmse)}</span>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-1">
                      <Trophy size={11} className="text-amber-500 flex-shrink-0" />
                      <span className="text-xs font-medium text-text-primary">{pct(entry.win_rate)}</span>
                      <span className="text-[10px] text-text-muted">({entry.n_wins}w)</span>
                    </div>
                  </td>
                  <td className="py-3 px-4 hidden md:table-cell min-w-[100px]">
                    <ConvBar rate={entry.convergence_rate} />
                  </td>
                  <td className="py-3 px-4 hidden md:table-cell">
                    <div className="flex items-center gap-1 text-xs text-text-secondary">
                      <FlaskConical size={11} />
                      {entry.n_experiments}
                      <span className="text-text-muted ml-1">/ {entry.problems_count}p</span>
                    </div>
                  </td>
                </motion.tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Show more / show less */}
      {tableEntries.length > 10 && (
        <button
          onClick={() => setShowAll((v) => !v)}
          className="mt-3 w-full flex items-center justify-center gap-1.5 py-2.5 rounded-xl border border-dashed border-border hover:border-accent hover:text-accent text-text-muted text-xs font-medium transition-colors"
        >
          {showAll ? (
            <><ChevronUp size={14} /> Show less</>
          ) : (
            <><ChevronDown size={14} /> Show {hidden} more architectures</>
          )}
        </button>
      )}
    </div>
  )
}

// ─── StatsBar ────────────────────────────────────────────────────────────────

function StatsBar({
  totalExperiments,
  totalArchitectures,
  problemsCovered,
}: {
  totalExperiments: number
  totalArchitectures: number
  problemsCovered: number
}) {
  const stats = [
    { icon: <FlaskConical size={16} className="text-accent" />, label: 'Total Experiments', value: totalExperiments },
    { icon: <Cpu size={16} className="text-purple-500" />, label: 'Architectures Ranked', value: totalArchitectures },
    { icon: <Target size={16} className="text-emerald-500" />, label: 'Problems Covered', value: problemsCovered },
  ]
  return (
    <div className="grid grid-cols-3 gap-3">
      {stats.map((s) => (
        <div
          key={s.label}
          className="bg-white border border-border rounded-xl p-3 sm:p-4 flex flex-col items-center text-center shadow-card"
        >
          {s.icon}
          <div className="text-xl sm:text-2xl font-bold text-text-primary mt-1">{s.value}</div>
          <div className="text-[10px] sm:text-xs text-text-muted mt-0.5">{s.label}</div>
        </div>
      ))}
    </div>
  )
}

// ─── EmptyState ───────────────────────────────────────────────────────────────

function EmptyState({ hasFilters }: { hasFilters: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center">
      <div className="text-5xl mb-4">{hasFilters ? '🔍' : '🏆'}</div>
      <h3 className="text-base font-semibold text-text-primary mb-2">
        {hasFilters ? 'No results for these filters' : 'No data yet'}
      </h3>
      <p className="text-sm text-text-muted max-w-sm">
        {hasFilters
          ? 'Try removing some filters to see more architectures.'
          : 'No completed experiments yet. Run an experiment to start building the leaderboard.'
        }
      </p>
    </div>
  )
}

// ─── Multi-Objective: helpers ─────────────────────────────────────────────────

const SCORE_DIMENSIONS = [
  { key: 'accuracy',        label: 'Accuracy',         color: '#10b981', icon: <Target size={11} /> },
  { key: 'training_cost',   label: 'Training Cost',    color: '#6366f1', icon: <Clock size={11} /> },
  { key: 'convergence',     label: 'Convergence',      color: '#8b5cf6', icon: <Activity size={11} /> },
  { key: 'memory',          label: 'Memory',           color: '#f59e0b', icon: <HardDrive size={11} /> },
  { key: 'inference_speed', label: 'Inference Speed',  color: '#ef4444', icon: <Zap size={11} /> },
] as const

const RADAR_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

function overallScore(scores: MultiObjectiveScores | null): number | null {
  if (!scores) return null
  const vals = [scores.accuracy, scores.training_cost, scores.convergence, scores.memory, scores.inference_speed]
    .filter((v): v is number => v != null)
  return vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : null
}

// ─── Multi-Objective: Radar chart ─────────────────────────────────────────────

function MultiObjectiveRadar({ entries }: { entries: LeaderboardEntry[] }) {
  const top = entries.slice(0, Math.min(5, entries.length)).filter((e) => e.scores)
  if (top.length === 0) return null

  const radarData = SCORE_DIMENSIONS.map(({ key, label }) => ({
    dimension: label,
    ...top.reduce<Record<string, number>>((acc, e) => {
      acc[e.architecture_name] = (e.scores as unknown as Record<string, number | null>)[key] ?? 0
      return acc
    }, {}),
  }))

  return (
    <div className="bg-white border border-border rounded-xl p-4 shadow-card">
      <h3 className="text-xs font-semibold text-text-primary mb-3 flex items-center gap-1.5">
        <RadarIcon size={13} className="text-accent" />
        Top {top.length} Architectures — Multi-Objective Comparison
      </h3>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
          <PolarGrid stroke="rgb(var(--color-border))" />
          <PolarAngleAxis
            dataKey="dimension"
            tick={{ fontSize: 10, fill: 'rgb(var(--color-text-secondary))' }}
          />
          {top.map((e, i) => (
            <Radar
              key={e.architecture_id}
              name={e.architecture_name}
              dataKey={e.architecture_name}
              stroke={RADAR_COLORS[i % RADAR_COLORS.length]}
              fill={RADAR_COLORS[i % RADAR_COLORS.length]}
              fillOpacity={0.12}
              strokeWidth={1.5}
            />
          ))}
          <Tooltip
            contentStyle={{ fontSize: 11, borderRadius: 8 }}
            formatter={(value: number) => [`${value}`, '']}
          />
          <Legend wrapperStyle={{ fontSize: 10 }} />
        </RadarChart>
      </ResponsiveContainer>
    </div>
  )
}

// ─── Multi-Objective: Score bar ───────────────────────────────────────────────

function ScoreBar({
  label, score, color, icon,
}: {
  label: string
  score: number | null
  color: string
  icon: React.ReactNode
}) {
  return (
    <div>
      <div className="flex items-center justify-between text-[10px] mb-1">
        <span className="flex items-center gap-1 text-text-muted" style={{ color }}>
          {icon}
          {label}
        </span>
        <span className="font-mono font-semibold text-text-primary">
          {score != null ? score : '—'}
        </span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: `${color}22` }}>
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${score ?? 0}%`, backgroundColor: color }}
        />
      </div>
    </div>
  )
}

// ─── Multi-Objective: Architecture score card ─────────────────────────────────

function ArchScoreCard({ entry }: { entry: LeaderboardEntry }) {
  const family = FAMILY_CONFIG[entry.family as keyof typeof FAMILY_CONFIG]
  const overall = overallScore(entry.scores)

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      className="bg-white border border-border rounded-xl p-4 shadow-card flex flex-col gap-3"
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-2">
        <div className="flex flex-col gap-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="w-5 h-5 rounded-full bg-slate-100 flex items-center justify-center text-[10px] font-bold text-text-muted flex-shrink-0">
              {entry.rank}
            </span>
            <span className="text-xs font-bold text-text-primary leading-tight truncate">
              {entry.architecture_name}
            </span>
          </div>
          {family && (
            <Badge
              variant={family.variant}
              size="xs"
              style={{ color: family.color, borderColor: `${family.color}30`, backgroundColor: `${family.color}10` } as React.CSSProperties}
            >
              {family.label}
            </Badge>
          )}
        </div>
        {overall != null && (
          <div className="flex-shrink-0 flex flex-col items-center">
            <div className="text-xl font-black text-accent leading-none">{overall}</div>
            <div className="text-[9px] text-text-muted mt-0.5">overall</div>
          </div>
        )}
      </div>

      {/* Score bars */}
      <div className="space-y-2">
        {SCORE_DIMENSIONS.map(({ key, label, color, icon }) => (
          <ScoreBar
            key={key}
            label={label}
            score={entry.scores ? (entry.scores as unknown as Record<string, number | null>)[key] : null}
            color={color}
            icon={icon}
          />
        ))}
      </div>

      {/* Footer stats */}
      <div className="flex items-center gap-3 text-[10px] text-text-muted pt-1 border-t border-border">
        <span className="flex items-center gap-0.5">
          <FlaskConical size={9} /> {entry.n_experiments} exp
        </span>
        <span className="flex items-center gap-0.5">
          <Trophy size={9} className="text-amber-500" /> {pct(entry.win_rate)} win
        </span>
        {entry.avg_l2_error != null && (
          <span className="font-mono">L2 {fmt(entry.avg_l2_error)}</span>
        )}
      </div>
    </motion.div>
  )
}

// ─── Multi-Objective: Full view ───────────────────────────────────────────────

function MultiObjectiveView({ entries }: { entries: LeaderboardEntry[] }) {
  return (
    <div className="space-y-5">
      {/* Dimension key */}
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        {SCORE_DIMENSIONS.map(({ key, label, color, icon }) => (
          <div key={key} className="flex items-center gap-1 text-[11px] text-text-secondary">
            <span style={{ color }}>{icon}</span>
            <span>{label}</span>
          </div>
        ))}
      </div>

      {/* Radar chart */}
      <MultiObjectiveRadar entries={entries} />

      {/* Score cards grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {entries.map((entry) => (
          <ArchScoreCard key={entry.architecture_id} entry={entry} />
        ))}
      </div>
    </div>
  )
}

// ─── LeaderboardPage ──────────────────────────────────────────────────────────

export function LeaderboardPage() {
  const [filters, setFilters] = useState<LeaderboardFilters>({})
  const [showFilters, setShowFilters] = useState(false)

  const hasFilters = Object.values(filters).some((v) => v != null)

  const { data: leaderboard, isLoading } = useQuery({
    queryKey: ['leaderboard', filters],
    queryFn: () => leaderboardApi.get(filters).then((r) => r.data),
    staleTime: 30_000,
  })

  const { data: problems } = useQuery({
    queryKey: ['problems'],
    queryFn: () => problemsApi.list().then((r) => r.data),
    staleTime: 60_000,
  })

  const problemOptions = useMemo(() => {
    if (!problems) return []
    return problems.map((p) => ({ id: p.id, name: p.name }))
  }, [problems])

  const entries = leaderboard?.entries ?? []
  const hasPodium = entries.length >= 3

  function setFilter<K extends keyof LeaderboardFilters>(key: K, value: LeaderboardFilters[K] | undefined) {
    setFilters((prev) => {
      const next = { ...prev }
      if (value == null) delete next[key]
      else next[key] = value
      return next
    })
  }

  function clearFilters() {
    setFilters({})
  }

  return (
    <div className="px-4 sm:px-8 py-4 sm:py-6 space-y-6 bg-background min-h-full pb-16">
      {/* Header */}
      <div className="flex items-start justify-between gap-4 flex-wrap">
        <div>
          <div className="flex items-center gap-2 mb-1">
            <Trophy size={22} className="text-amber-500" />
            <h1 className="text-2xl font-bold text-text-primary">Global Leaderboard</h1>
          </div>
          <p className="text-sm text-text-secondary">
            Aggregated performance across all completed experiments — accuracy rankings and multi-objective breakdown.
          </p>
        </div>

        <button
          onClick={() => setShowFilters((v) => !v)}
          className={clsx(
            'flex items-center gap-2 px-3 py-2 rounded-lg border text-xs font-medium transition-colors',
            showFilters || hasFilters
              ? 'bg-accent text-white border-accent'
              : 'bg-white border-border text-text-secondary hover:text-text-primary hover:bg-surface-elevated',
          )}
        >
          <Filter size={13} />
          Filters
          {hasFilters && (
            <span className="bg-white/20 text-white rounded-full w-4 h-4 flex items-center justify-center text-[9px] font-bold">
              {Object.values(filters).filter((v) => v != null).length}
            </span>
          )}
        </button>
      </div>

      {/* Filter panel */}
      <AnimatePresence>
        {showFilters && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="overflow-hidden"
          >
            <div className="bg-white border border-border rounded-xl p-4 space-y-4 shadow-card">
              {/* Problem selector */}
              <div>
                <div className="text-xs font-medium text-text-muted mb-2">Specific Problem</div>
                <select
                  value={filters.problem_id ?? ''}
                  onChange={(e) => setFilter('problem_id', e.target.value || undefined)}
                  className="w-full sm:w-80 text-xs border border-border rounded-lg px-3 py-2 bg-white text-text-primary focus:outline-none focus:ring-2 focus:ring-accent/30"
                >
                  <option value="">All problems</option>
                  {problemOptions.map((p) => (
                    <option key={p.id} value={p.id}>{p.name}</option>
                  ))}
                </select>
              </div>

              {/* Domain */}
              <div>
                <div className="text-xs font-medium text-text-muted mb-2">Physics Domain</div>
                <div className="flex flex-wrap gap-2">
                  {DOMAINS.map((d) => {
                    const cfg = DOMAIN_CONFIG[d]
                    return (
                      <FilterChip
                        key={d}
                        label={cfg.label}
                        active={filters.domain === d}
                        onClick={() => setFilter('domain', filters.domain === d ? undefined : d)}
                        icon={cfg.icon}
                        color={cfg.color}
                      />
                    )
                  })}
                </div>
              </div>

              {/* Dimension + Difficulty + Geometry */}
              <div className="flex flex-wrap gap-6">
                <div>
                  <div className="text-xs font-medium text-text-muted mb-2">Dimension</div>
                  <div className="flex gap-2">
                    {DIMENSIONS.map((d) => (
                      <FilterChip
                        key={d}
                        label={DIMENSION_LABELS[d]}
                        active={filters.dimension === d}
                        onClick={() => setFilter('dimension', filters.dimension === d ? undefined : d)}
                      />
                    ))}
                  </div>
                </div>

                <div>
                  <div className="text-xs font-medium text-text-muted mb-2">Difficulty</div>
                  <div className="flex gap-2">
                    {DIFFICULTIES.map((d) => {
                      const cfg = DIFFICULTY_CONFIG[d]
                      return (
                        <FilterChip
                          key={d}
                          label={cfg.label}
                          active={filters.difficulty === d}
                          onClick={() => setFilter('difficulty', filters.difficulty === d ? undefined : d)}
                        />
                      )
                    })}
                  </div>
                </div>

                <div>
                  <div className="text-xs font-medium text-text-muted mb-2">Geometry</div>
                  <div className="flex gap-2">
                    <FilterChip
                      label="With geometry"
                      active={filters.requires_geometry === true}
                      onClick={() => setFilter('requires_geometry', filters.requires_geometry === true ? undefined : true)}
                    />
                    <FilterChip
                      label="No geometry"
                      active={filters.requires_geometry === false}
                      onClick={() => setFilter('requires_geometry', filters.requires_geometry === false ? undefined : false)}
                    />
                  </div>
                </div>
              </div>

              {hasFilters && (
                <button
                  onClick={clearFilters}
                  className="flex items-center gap-1 text-xs text-accent hover:text-accent-hover"
                >
                  <X size={12} /> Clear all filters
                </button>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Active filter chips summary */}
      {hasFilters && !showFilters && (
        <div className="flex flex-wrap items-center gap-2 text-xs text-text-muted">
          <Filter size={11} />
          <span>Filtered by:</span>
          {filters.problem_id && (
            <span className="bg-accent/10 text-accent px-2 py-0.5 rounded-full flex items-center gap-1">
              Problem: {problemOptions.find((p) => p.id === filters.problem_id)?.name ?? filters.problem_id}
              <button onClick={() => setFilter('problem_id', undefined)}><X size={10} /></button>
            </span>
          )}
          {filters.domain && (
            <span className="bg-accent/10 text-accent px-2 py-0.5 rounded-full flex items-center gap-1">
              {DOMAIN_CONFIG[filters.domain as keyof typeof DOMAIN_CONFIG]?.label ?? filters.domain}
              <button onClick={() => setFilter('domain', undefined)}><X size={10} /></button>
            </span>
          )}
          {filters.dimension && (
            <span className="bg-accent/10 text-accent px-2 py-0.5 rounded-full flex items-center gap-1">
              {DIMENSION_LABELS[filters.dimension as keyof typeof DIMENSION_LABELS] ?? filters.dimension}
              <button onClick={() => setFilter('dimension', undefined)}><X size={10} /></button>
            </span>
          )}
          {filters.difficulty && (
            <span className="bg-accent/10 text-accent px-2 py-0.5 rounded-full flex items-center gap-1">
              {DIFFICULTY_CONFIG[filters.difficulty as keyof typeof DIFFICULTY_CONFIG]?.label ?? filters.difficulty}
              <button onClick={() => setFilter('difficulty', undefined)}><X size={10} /></button>
            </span>
          )}
          {filters.requires_geometry != null && (
            <span className="bg-accent/10 text-accent px-2 py-0.5 rounded-full flex items-center gap-1">
              {filters.requires_geometry ? 'With geometry' : 'No geometry'}
              <button onClick={() => setFilter('requires_geometry', undefined)}><X size={10} /></button>
            </span>
          )}
        </div>
      )}

      {/* Main content */}
      {isLoading ? (
        <PageLoading label="Computing global rankings…" />
      ) : entries.length === 0 ? (
        <EmptyState hasFilters={hasFilters} />
      ) : (
        <>
          {/* Stats */}
          <StatsBar
            totalExperiments={leaderboard?.total_experiments ?? 0}
            totalArchitectures={leaderboard?.total_architectures ?? 0}
            problemsCovered={leaderboard?.problems_covered ?? 0}
          />

          {/* Podium — only when there are at least 3 entries */}
          {hasPodium && (
            <div>
              <h2 className="text-sm font-semibold text-text-primary mb-2 flex items-center gap-1.5">
                <Medal size={15} className="text-amber-500" />
                Top 3 Architectures
              </h2>
              <Podium entries={entries} />
            </div>
          )}

          {/* Accuracy rankings table */}
          <RankingsTable entries={entries} showPodium={hasPodium} />

          {/* Multi-objective breakdown */}
          <div>
            <h2 className="text-sm font-semibold text-text-primary mb-1 flex items-center gap-1.5">
              <RadarIcon size={15} className="text-accent" />
              Benchmarking Beyond Accuracy
            </h2>
            <p className="text-xs text-text-muted mb-4">
              Multi-objective evaluation across 5 dimensions. Scores are normalized 0–100 within the current dataset (higher = better).
            </p>
            <MultiObjectiveView entries={entries} />
          </div>
        </>
      )}
    </div>
  )
}
