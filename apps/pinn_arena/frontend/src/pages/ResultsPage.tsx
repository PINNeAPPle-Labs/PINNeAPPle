import React, { useState, useMemo } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Trophy, ChevronUp, ChevronDown, ChevronRight, Download, Plus,
  CheckCircle, XCircle, Crown, Medal, Award, BarChart3, TrendingDown,
  Clock, Layers, ChevronDown as ChevronDownIcon, Microscope, Activity,
  ShieldCheck, ShieldAlert, ShieldX, AlertTriangle, Info,
  Zap, HardDrive, Target, Radar as RadarIcon,
} from 'lucide-react'
import {
  RadarChart, PolarGrid, PolarAngleAxis, Radar,
  ResponsiveContainer, Tooltip, Legend,
} from 'recharts'
import { useQueries, useQuery } from '@tanstack/react-query'
import { useExperimentResults } from '../hooks/useExperiment'
import { useExperimentStore } from '../store/experimentStore'
import type { LucideIcon } from 'lucide-react'
import type { ModelResult, SortColumn, SortConfig, ResultTab, ArchitectureFamily, InverseResult, PredictionQualityLevel } from '../types'
import { LossCurvesChart, ErrorDistributionChart, TrainingScatterPlot, FieldComparisonChart } from '../components/ComparisonChart'
import { Badge } from '../components/ui/Badge'
import { Button } from '../components/ui/Button'
import { Card, CardContent } from '../components/ui/Card'
import { PageLoading } from '../components/ui/LoadingSpinner'
import { FAMILY_CONFIG } from '../components/ArchitectureCard'
import { experimentsApi, problemsApi, architecturesApi } from '../lib/api'
import ArchitectureDetailModal from '../components/ArchitectureDetailModal'
import type { ArchitectureInfo } from '../types'
import { ScientificHeatmap, buildHeatmapFromScatter } from '../components/ScientificHeatmap'
import type { ColormapName } from '../components/ScientificHeatmap'
import { SurfaceMeshView } from '../components/SurfaceMeshView'
import type { SurfaceMeshData } from '../components/SurfaceMeshView'

const RANK_ICONS: Record<number, React.ReactNode> = {
  1: <Crown size={16} className="text-yellow-500" />,
  2: <Medal size={16} className="text-slate-400" />,
  3: <Award size={16} className="text-amber-600" />,
}

function getRankStyle(rank: number) {
  if (rank === 1) return 'border-yellow-300 bg-yellow-50'
  if (rank === 2) return 'border-slate-300 bg-slate-50'
  if (rank === 3) return 'border-amber-300 bg-amber-50'
  return ''
}

function formatMetric(value: number, decimals = 4): string {
  if (Math.abs(value) < 1e-3 || Math.abs(value) > 1e4) {
    return value.toExponential(decimals)
  }
  return value.toFixed(decimals)
}

function MetricCell({ value, isBest, isWorst }: { value: number; isBest: boolean; isWorst: boolean }) {
  return (
    <span className={
      isBest ? 'text-success font-semibold' :
      isWorst ? 'text-danger' :
      'text-text-primary'
    }>
      {formatMetric(value)}
      {isBest && <span className="ml-1 text-success/60 text-[10px]">▲</span>}
    </span>
  )
}

interface SortableHeaderProps {
  label: string
  column: SortColumn
  currentSort: SortConfig
  onSort: (col: SortColumn) => void
}

function SortableHeader({ label, column, currentSort, onSort }: SortableHeaderProps) {
  const isActive = currentSort.column === column
  return (
    <th
      className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 cursor-pointer hover:text-text-primary transition-colors whitespace-nowrap"
      onClick={() => onSort(column)}
    >
      <div className="flex items-center gap-1">
        {label}
        <span className="flex flex-col">
          <ChevronUp size={10} className={isActive && currentSort.direction === 'asc' ? 'text-accent' : 'text-slate-300'} />
          <ChevronDown size={10} className={isActive && currentSort.direction === 'desc' ? 'text-accent' : 'text-slate-300'} />
        </span>
      </div>
    </th>
  )
}

// ---------------------------------------------------------------------------
// Prediction Quality Badge
// ---------------------------------------------------------------------------

const QUALITY_CONFIG: Record<PredictionQualityLevel, {
  label: string
  color: string
  bg: string
  border: string
  Icon: LucideIcon
}> = {
  excellent:   { label: 'Excellent',   color: '#16a34a', bg: '#f0fdf4', border: '#bbf7d0', Icon: ShieldCheck },
  good:        { label: 'Good',        color: '#0891b2', bg: '#ecfeff', border: '#a5f3fc', Icon: ShieldCheck },
  fair:        { label: 'Fair',        color: '#d97706', bg: '#fffbeb', border: '#fde68a', Icon: ShieldAlert },
  poor:        { label: 'Poor',        color: '#dc2626', bg: '#fef2f2', border: '#fecaca', Icon: ShieldX },
  unreliable:  { label: 'Unreliable', color: '#7c3aed', bg: '#faf5ff', border: '#ddd6fe', Icon: ShieldX },
  unknown:     { label: 'N/A',         color: '#94a3b8', bg: '#f8fafc', border: '#e2e8f0', Icon: Info },
}

const FLAG_LABELS: Record<string, string> = {
  nan_or_inf:           'Output contains NaN or Inf values',
  constant_output:      'Model predicts a constant value everywhere',
  near_constant_output: 'Prediction has near-zero variance',
  zero_output:          'Model outputs near-zero everywhere (trivial solution)',
  below_mean_baseline:  'R² < 0: worse than predicting the mean',
  high_relative_error:  'L2 relative error > 1.0 (worse than trivial)',
}

function PredictionQualityBadge({ result }: { result: ModelResult }) {
  const [open, setOpen] = useState(false)
  const level: PredictionQualityLevel = result.quality_level ?? 'unknown'
  const cfg = QUALITY_CONFIG[level]
  const { Icon } = cfg
  const flags = result.quality_flags ?? []
  const score = result.quality_score

  return (
    <div className="relative inline-block">
      <button
        onClick={(e) => { e.stopPropagation(); setOpen((v) => !v) }}
        onBlur={() => setTimeout(() => setOpen(false), 150)}
        className="flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-semibold border transition-colors cursor-pointer"
        style={{ color: cfg.color, backgroundColor: cfg.bg, borderColor: cfg.border }}
        title="Prediction quality — click for details"
      >
        <Icon size={11} className="" />
        {cfg.label}
        {score != null && (
          <span className="opacity-60 font-normal ml-0.5">{Math.round(score * 100)}%</span>
        )}
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -4 }}
            transition={{ duration: 0.15 }}
            className="absolute z-50 left-0 top-full mt-1.5 w-64 rounded-xl border border-border bg-white shadow-lg p-3"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="text-xs font-semibold text-text-primary mb-1.5">
              Prediction Quality
            </div>

            {/* Score bar */}
            {score != null && (
              <div className="mb-2">
                <div className="flex justify-between text-[10px] text-text-muted mb-0.5">
                  <span>Score</span><span style={{ color: cfg.color }}>{Math.round(score * 100)}/100</span>
                </div>
                <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${Math.round(score * 100)}%`, backgroundColor: cfg.color }}
                  />
                </div>
              </div>
            )}

            {/* std ratio */}
            {result.pred_std_ratio != null && (
              <div className="text-[10px] text-text-muted mb-2">
                Output variance ratio: <span className="font-mono text-text-primary">{result.pred_std_ratio.toFixed(2)}×</span>
                <span className="ml-1 opacity-60">(ideal ≈ 1.0)</span>
              </div>
            )}

            {/* Flags */}
            {flags.length > 0 ? (
              <div className="space-y-1">
                <div className="text-[10px] font-semibold text-danger uppercase tracking-wide mb-1">Issues detected</div>
                {flags.map((f) => (
                  <div key={f} className="flex items-start gap-1.5 text-[10px] text-text-secondary">
                    <AlertTriangle size={10} className="text-danger mt-0.5 shrink-0" />
                    {FLAG_LABELS[f] ?? f}
                  </div>
                ))}
              </div>
            ) : (
              <div className="text-[10px] text-success flex items-center gap-1">
                <CheckCircle size={10} /> No issues detected
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

// ─── Experiment Multi-Objective Comparison ────────────────────────────────────

const EXP_SCORE_DIMS = [
  { key: 'accuracy',        label: 'Accuracy',        color: '#10b981', icon: <Target   size={11} /> },
  { key: 'training_cost',   label: 'Training Cost',   color: '#6366f1', icon: <Clock    size={11} /> },
  { key: 'convergence',     label: 'Convergence',     color: '#8b5cf6', icon: <Activity size={11} /> },
  { key: 'memory',          label: 'Memory',          color: '#f59e0b', icon: <HardDrive size={11} /> },
  { key: 'inference_speed', label: 'Inference Speed', color: '#ef4444', icon: <Zap      size={11} /> },
] as const

const EXP_RADAR_COLORS = ['#6366f1', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316']

type ExpScores = { accuracy: number | null; training_cost: number | null; convergence: number; memory: number | null; inference_speed: number | null }
type ScoredModel = ModelResult & { expScores: ExpScores }

function computeExpScores(models: ModelResult[]): ScoredModel[] {
  const normLower = (vals: (number | null | undefined)[], v: number | null | undefined): number | null => {
    if (v == null) return null
    const valid = vals.filter((x): x is number => x != null)
    if (!valid.length) return null
    const lo = Math.min(...valid), hi = Math.max(...valid)
    if (hi === lo) return 100
    return Math.round((1 - (v - lo) / (hi - lo)) * 1000) / 10
  }

  const times   = models.map(m => m.training_time_s)
  const mems    = models.map(m => (m.peak_gpu_memory_mb ?? m.peak_cpu_memory_mb) ?? null)
  const infers  = models.map(m => m.inference_time_ms ?? null)

  return models.map(m => {
    const l2  = m.metrics.l2_relative_error
    const mem = m.peak_gpu_memory_mb ?? m.peak_cpu_memory_mb ?? null
    return {
      ...m,
      expScores: {
        accuracy:        l2  != null ? Math.max(0, Math.round((1 - l2) * 1000) / 10) : null,
        training_cost:   normLower(times,  m.training_time_s),
        convergence:     m.converged || (l2 != null && l2 < 0.1) ? 100 : 0,
        memory:          normLower(mems,   mem),
        inference_speed: normLower(infers, m.inference_time_ms ?? null),
      } satisfies ExpScores,
    }
  })
}

function ExperimentRadar({ models }: { models: ScoredModel[] }) {
  const radarData = EXP_SCORE_DIMS.map(({ key, label }) => ({
    dimension: label,
    ...models.reduce<Record<string, number>>((acc, m) => {
      acc[m.architecture_name] = (m.expScores as Record<string, number | null>)[key] ?? 0
      return acc
    }, {}),
  }))

  return (
    <ResponsiveContainer width="100%" height={280}>
      <RadarChart data={radarData} margin={{ top: 10, right: 30, bottom: 10, left: 30 }}>
        <PolarGrid stroke="rgb(var(--color-border))" />
        <PolarAngleAxis dataKey="dimension" tick={{ fontSize: 10, fill: 'rgb(var(--color-text-secondary))' }} />
        {models.map((m, i) => (
          <Radar
            key={m.architecture_id}
            name={m.architecture_name}
            dataKey={m.architecture_name}
            stroke={EXP_RADAR_COLORS[i % EXP_RADAR_COLORS.length]}
            fill={EXP_RADAR_COLORS[i % EXP_RADAR_COLORS.length]}
            fillOpacity={0.12}
            strokeWidth={1.5}
          />
        ))}
        <Tooltip contentStyle={{ fontSize: 11, borderRadius: 8 }} formatter={(v: number) => [`${v}`, '']} />
        <Legend wrapperStyle={{ fontSize: 10 }} />
      </RadarChart>
    </ResponsiveContainer>
  )
}

function ExpScoreBar({ label, score, color, icon }: { label: string; score: number | null; color: string; icon: React.ReactNode }) {
  return (
    <div>
      <div className="flex items-center justify-between text-[10px] mb-1">
        <span className="flex items-center gap-1" style={{ color }}>{icon}{label}</span>
        <span className="font-mono font-semibold text-text-primary">{score != null ? score : '—'}</span>
      </div>
      <div className="h-1.5 rounded-full overflow-hidden" style={{ backgroundColor: `${color}22` }}>
        <div className="h-full rounded-full transition-all duration-500" style={{ width: `${score ?? 0}%`, backgroundColor: color }} />
      </div>
    </div>
  )
}

function ExperimentMultiObjectiveView({ modelResults }: { modelResults: ModelResult[] }) {
  const completed = modelResults.filter(m => m.metrics?.l2_relative_error != null)
  if (completed.length < 2) return null

  const scored = useMemo(() => computeExpScores(completed), [completed])

  return (
    <div className="space-y-4">
      <h2 className="text-base font-bold text-text-primary flex items-center gap-2">
        <RadarIcon size={18} className="text-accent" />
        Multi-Objective Comparison
      </h2>

      {/* Dimension key */}
      <div className="flex flex-wrap gap-x-4 gap-y-1.5">
        {EXP_SCORE_DIMS.map(({ key, label, color, icon }) => (
          <div key={key} className="flex items-center gap-1 text-[11px] text-text-secondary">
            <span style={{ color }}>{icon}</span>
            <span>{label}</span>
          </div>
        ))}
      </div>

      {/* Radar */}
      <div className="bg-white border border-border rounded-xl p-4 shadow-card">
        <ExperimentRadar models={scored} />
      </div>

      {/* Score cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {scored.map((m, i) => {
          const vals = Object.values(m.expScores).filter((v): v is number => v != null)
          const overall = vals.length ? Math.round(vals.reduce((a, b) => a + b, 0) / vals.length) : null
          const family = FAMILY_CONFIG[m.architecture_family as keyof typeof FAMILY_CONFIG]
          return (
            <motion.div
              key={m.architecture_id}
              initial={{ opacity: 0, y: 12 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: i * 0.04 }}
              className="bg-white border border-border rounded-xl p-4 shadow-card flex flex-col gap-3"
            >
              <div className="flex items-start justify-between gap-2">
                <div className="flex flex-col gap-1 min-w-0">
                  <span className="text-xs font-bold text-text-primary leading-tight truncate">{m.architecture_name}</span>
                  {family && (
                    <Badge variant={family.variant} size="xs"
                      style={{ color: family.color, borderColor: `${family.color}30`, backgroundColor: `${family.color}10` } as React.CSSProperties}>
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
              <div className="space-y-2">
                {EXP_SCORE_DIMS.map(({ key, label, color, icon }) => (
                  <ExpScoreBar key={key} label={label}
                    score={(m.expScores as Record<string, number | null>)[key]}
                    color={color} icon={icon} />
                ))}
              </div>
              <div className="flex items-center gap-3 text-[10px] text-text-muted pt-1 border-t border-border">
                <span className="font-mono">L2 {m.metrics.l2_relative_error.toFixed(3)}</span>
                <span>{m.training_time_s.toFixed(1)}s train</span>
                {m.inference_time_ms != null && <span>{m.inference_time_ms.toFixed(0)}ms infer</span>}
              </div>
            </motion.div>
          )
        })}
      </div>
    </div>
  )
}

function RankingTable({
  modelResults,
  ranking,
  bestModelId,
  experimentId,
  archInfoMap,
  onDetails,
}: {
  modelResults: ModelResult[]
  ranking: string[]
  bestModelId: string
  experimentId?: string
  archInfoMap: Record<string, ArchitectureInfo>
  onDetails: (arch: ArchitectureInfo) => void
}) {
  const [sort, setSort] = useState<SortConfig>({ column: 'l2_relative_error', direction: 'asc' })
  const [expandedId, setExpandedId] = useState<string | null>(null)

  const sorted = useMemo(() => {
    const ranked = modelResults.map((m) => ({
      ...m,
      rank: (ranking.indexOf(m.architecture_id) + 1) || modelResults.length,
    }))
    return [...ranked].sort((a, b) => {
      let av: number | string, bv: number | string
      switch (sort.column) {
        case 'rank': av = a.rank; bv = b.rank; break
        case 'architecture_name': av = a.architecture_name; bv = b.architecture_name; break
        case 'l2_relative_error': av = a.metrics.l2_relative_error; bv = b.metrics.l2_relative_error; break
        case 'rmse': av = a.metrics.rmse; bv = b.metrics.rmse; break
        case 'max_error': av = a.metrics.max_error; bv = b.metrics.max_error; break
        case 'r2_score': av = a.metrics.r2_score; bv = b.metrics.r2_score; break
        case 'training_time_s': av = a.training_time_s; bv = b.training_time_s; break
        case 'converged': av = a.converged ? 1 : 0; bv = b.converged ? 1 : 0; break
        default: av = a.rank; bv = b.rank
      }
      const cmp = typeof av === 'string' ? av.localeCompare(bv as string) : (av as number) - (bv as number)
      return sort.direction === 'asc' ? cmp : -cmp
    })
  }, [modelResults, ranking, sort])

  const handleSort = (col: SortColumn) => {
    setSort((prev) => ({
      column: col,
      direction: prev.column === col && prev.direction === 'asc' ? 'desc' : 'asc',
    }))
  }

  // Compute best/worst for error columns (lower = better)
  const minL2 = Math.min(...modelResults.map((m) => m.metrics.l2_relative_error))
  const maxL2 = Math.max(...modelResults.map((m) => m.metrics.l2_relative_error))
  const minRmse = Math.min(...modelResults.map((m) => m.metrics.rmse))
  const maxRmse = Math.max(...modelResults.map((m) => m.metrics.rmse))
  const minMax = Math.min(...modelResults.map((m) => m.metrics.max_error))
  const maxMax = Math.max(...modelResults.map((m) => m.metrics.max_error))
  // R2: higher = better
  const maxR2 = Math.max(...modelResults.map((m) => m.metrics.r2_score))
  const minR2 = Math.min(...modelResults.map((m) => m.metrics.r2_score))

  return (
    <div className="overflow-x-auto rounded-xl border border-border bg-white shadow-card">
      <table className="w-full">
        <thead className="bg-slate-50 border-b border-border">
          <tr>
            <SortableHeader label="Rank" column="rank" currentSort={sort} onSort={handleSort} />
            <SortableHeader label="Architecture" column="architecture_name" currentSort={sort} onSort={handleSort} />
            <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4 whitespace-nowrap">Quality</th>
            <SortableHeader label="L2 Error" column="l2_relative_error" currentSort={sort} onSort={handleSort} />
            <SortableHeader label="RMSE" column="rmse" currentSort={sort} onSort={handleSort} />
            <SortableHeader label="Max Error" column="max_error" currentSort={sort} onSort={handleSort} />
            <SortableHeader label="R² Score" column="r2_score" currentSort={sort} onSort={handleSort} />
            <SortableHeader label="Time (s)" column="training_time_s" currentSort={sort} onSort={handleSort} />
            <SortableHeader label="Converged" column="converged" currentSort={sort} onSort={handleSort} />
            <th className="w-10" />
          </tr>
        </thead>
        <tbody className="divide-y divide-border">
          {sorted.map((m) => {
            const isBest = m.architecture_id === bestModelId
            const isExpanded = expandedId === m.architecture_id
            const family = FAMILY_CONFIG[m.architecture_name as unknown as ArchitectureFamily]

            return (
              <React.Fragment key={m.architecture_id}>
                <tr
                  className={`transition-colors hover:bg-slate-50 cursor-pointer ${isBest ? getRankStyle(1) : m.rank <= 3 ? getRankStyle(m.rank) : ''}`}
                  onClick={() => setExpandedId(isExpanded ? null : m.architecture_id)}
                >
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-1.5">
                      {RANK_ICONS[m.rank] ?? (
                        <span className="text-sm font-bold text-text-muted w-4 text-center">{m.rank}</span>
                      )}
                      {isBest && <span className="text-[10px] text-yellow-600 font-semibold">BEST</span>}
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <div className="flex items-center gap-2">
                      <div>
                        <div className="text-sm font-semibold text-text-primary">{m.architecture_name}</div>
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
                    </div>
                  </td>
                  <td className="py-3 px-4">
                    <PredictionQualityBadge result={m} />
                  </td>
                  <td className="py-3 px-4 font-mono text-sm">
                    <MetricCell
                      value={m.metrics.l2_relative_error}
                      isBest={m.metrics.l2_relative_error === minL2}
                      isWorst={m.metrics.l2_relative_error === maxL2}
                    />
                  </td>
                  <td className="py-3 px-4 font-mono text-sm">
                    <MetricCell
                      value={m.metrics.rmse}
                      isBest={m.metrics.rmse === minRmse}
                      isWorst={m.metrics.rmse === maxRmse}
                    />
                  </td>
                  <td className="py-3 px-4 font-mono text-sm">
                    <MetricCell
                      value={m.metrics.max_error}
                      isBest={m.metrics.max_error === minMax}
                      isWorst={m.metrics.max_error === maxMax}
                    />
                  </td>
                  <td className="py-3 px-4 font-mono text-sm">
                    <MetricCell
                      value={m.metrics.r2_score}
                      isBest={m.metrics.r2_score === maxR2}
                      isWorst={m.metrics.r2_score === minR2}
                    />
                  </td>
                  <td className="py-3 px-4 font-mono text-sm text-text-secondary">
                    {m.training_time_s.toFixed(1)}s
                  </td>
                  <td className="py-3 px-4">
                    {m.converged ? (
                      <CheckCircle size={16} className="text-success" />
                    ) : (
                      <XCircle size={16} className="text-slate-300" />
                    )}
                  </td>
                  <td className="py-3 px-4">
                    <ChevronDownIcon
                      size={14}
                      className={`text-slate-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
                    />
                  </td>
                </tr>

                {/* Expanded row */}
                <AnimatePresence>
                  {isExpanded && (
                    <tr>
                      <td colSpan={10} className="p-0">
                        <motion.div
                          initial={{ height: 0, opacity: 0 }}
                          animate={{ height: 'auto', opacity: 1 }}
                          exit={{ height: 0, opacity: 0 }}
                          transition={{ duration: 0.2 }}
                          className="overflow-hidden bg-slate-50"
                        >
                          <div className="px-6 py-4 grid grid-cols-2 md:grid-cols-4 gap-4">
                            {[
                              { label: 'L1 Error', value: m.metrics.l1_error },
                              { label: 'L2 Error', value: m.metrics.l2_relative_error },
                              { label: 'RMSE', value: m.metrics.rmse },
                              { label: 'Max Error', value: m.metrics.max_error },
                              { label: 'R² Score', value: m.metrics.r2_score },
                              { label: 'Final Loss', value: m.final_loss },
                              { label: 'Train Time', value: m.training_time_s, unit: 's', noExp: true },
                              { label: 'History Pts', value: m.training_history.length, noExp: true },
                            ].map((item) => (
                              <div key={item.label}>
                                <div className="text-xs text-text-muted">{item.label}</div>
                                <div className="font-mono text-sm text-text-primary mt-0.5">
                                  {item.noExp ? item.value : formatMetric(item.value as number)}
                                  {item.unit && <span className="text-text-muted text-xs ml-1">{item.unit}</span>}
                                </div>
                              </div>
                            ))}
                            <div className="col-span-2 md:col-span-4 flex flex-wrap gap-2 mt-2 pt-2 border-t border-border">
                              {archInfoMap[m.architecture_id] && (
                                <button
                                  onClick={(e) => { e.stopPropagation(); onDetails(archInfoMap[m.architecture_id]) }}
                                  className="flex items-center gap-1.5 rounded-lg border border-accent/30 bg-accent/5 px-3 py-1.5 text-xs font-semibold text-accent hover:bg-accent/10 transition-colors"
                                >
                                  <Layers size={12} /> Details
                                </button>
                              )}
                              {experimentId && (
                                <button
                                  onClick={(e) => { e.stopPropagation(); experimentsApi.downloadModel(experimentId, m.architecture_id) }}
                                  className="flex items-center gap-1.5 rounded-lg border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-600 hover:bg-slate-50 transition-colors"
                                >
                                  <Download size={12} /> Download {m.architecture_name}.pt
                                </button>
                              )}
                            </div>
                          </div>
                        </motion.div>
                      </td>
                    </tr>
                  )}
                </AnimatePresence>
              </React.Fragment>
            )
          })}
        </tbody>
      </table>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Uncertainty Quantification Panel
// ---------------------------------------------------------------------------

interface ModelPlotDataForUQ {
  y_pred: unknown[]
  y_true: unknown[]
  X_eval: number[][]
}

function flattenPred(y: unknown[]): number[] {
  if (!y.length) return []
  const first = y[0]
  if (typeof first === 'number') return y as number[]
  if (Array.isArray(first)) return (y as number[][]).map((v) => (v[0] ?? 0) as number)
  return y.map((v) => (typeof v === 'number' ? v : 0))
}

// ── Per-model residual UQ ───────────────────────────────────────────────────

function PerModelUQCard({
  plotData,
  modelName,
}: {
  plotData: ModelPlotDataForUQ
  modelName: string
}) {
  const preds  = flattenPred(plotData.y_pred as unknown[])
  const truths = flattenPred(plotData.y_true as unknown[])
  const nPts   = Math.min(preds.length, truths.length)

  const residuals = Array.from({ length: nPts }, (_, i) => preds[i] - truths[i])
  const meanRes   = residuals.reduce((a, b) => a + b, 0) / nPts
  const stdRes    = Math.sqrt(residuals.reduce((a, b) => a + (b - meanRes) ** 2, 0) / nPts)
  const mae       = residuals.reduce((a, b) => a + Math.abs(b), 0) / nPts

  const cov1 = (residuals.filter((r) => Math.abs(r) <= stdRes).length / nPts) * 100
  const cov2 = (residuals.filter((r) => Math.abs(r) <= 2 * stdRes).length / nPts) * 100

  // ── Residual histogram ───────────────────────────────────────────────────
  const N_BINS = 28
  const histRange = Math.max(Math.abs(Math.min(...residuals)), Math.abs(Math.max(...residuals)), stdRes * 3.2, 1e-12)
  const binWidth  = (histRange * 2) / N_BINS
  const bins: number[] = Array(N_BINS).fill(0)
  residuals.forEach((r) => {
    const idx = Math.min(N_BINS - 1, Math.max(0, Math.floor((r + histRange) / binWidth)))
    bins[idx]++
  })
  const maxCount = Math.max(...bins, 1)

  const W = 420; const H = 120
  const pad = { l: 36, r: 12, t: 8, b: 28 }
  const pw = W - pad.l - pad.r; const ph = H - pad.t - pad.b

  const bw   = pw / N_BINS                           // pixel width per bin
  const bx   = (i: number) => pad.l + i * bw        // left edge of bin i
  const barH = (count: number) => (count / maxCount) * ph
  // sigma marker positions
  const sigPx = (sigma: number) => pad.l + ((sigma + histRange) / (histRange * 2)) * pw
  const zeroPx = sigPx(0)

  return (
    <div className="bg-slate-50 rounded-xl border border-border p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-sm font-semibold text-text-primary">{modelName}</span>
        <div className="flex gap-3 text-xs font-mono text-text-muted">
          <span>bias: <span className={meanRes >= 0 ? 'text-orange-500' : 'text-blue-500'}>{meanRes >= 0 ? '+' : ''}{meanRes.toExponential(2)}</span></span>
          <span>σ: <span className="text-accent">{stdRes.toExponential(2)}</span></span>
          <span>MAE: <span className="text-text-primary">{mae.toExponential(2)}</span></span>
        </div>
      </div>

      {/* Residual histogram */}
      <div className="overflow-x-auto">
        <svg width={W} height={H} className="font-mono">
          {/* ±2σ shaded region */}
          <rect
            x={sigPx(-2 * stdRes)} y={pad.t}
            width={sigPx(2 * stdRes) - sigPx(-2 * stdRes)} height={ph}
            fill="#3b82f612"
          />
          {/* ±1σ shaded region */}
          <rect
            x={sigPx(-stdRes)} y={pad.t}
            width={sigPx(stdRes) - sigPx(-stdRes)} height={ph}
            fill="#3b82f622"
          />
          {/* Histogram bars */}
          {bins.map((count, i) => {
            const binCenter = -histRange + (i + 0.5) * binWidth
            const isOutlier = Math.abs(binCenter) > 2 * stdRes
            const h = barH(count)
            return (
              <rect
                key={i}
                x={bx(i) + 0.5} y={pad.t + ph - h}
                width={Math.max(1, bw - 1)} height={h}
                fill={isOutlier ? '#ef4444' : '#3b82f6'}
                fillOpacity={0.75}
                rx={1}
              />
            )
          })}
          {/* Zero / mean lines */}
          <line x1={zeroPx} y1={pad.t} x2={zeroPx} y2={pad.t + ph} stroke="#94a3b8" strokeWidth={1} strokeDasharray="3 2" />
          <line x1={sigPx(meanRes)} y1={pad.t} x2={sigPx(meanRes)} y2={pad.t + ph} stroke="#f97316" strokeWidth={1.5} />
          {/* σ tick lines */}
          {[-2, -1, 1, 2].map((k) => (
            <line key={k} x1={sigPx(k * stdRes)} y1={pad.t + ph - 6} x2={sigPx(k * stdRes)} y2={pad.t + ph} stroke="#3b82f6" strokeWidth={1} />
          ))}
          {/* Axes */}
          <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + ph} stroke="#cbd5e1" strokeWidth={1} />
          <line x1={pad.l} y1={pad.t + ph} x2={pad.l + pw} y2={pad.t + ph} stroke="#cbd5e1" strokeWidth={1} />
          {/* X labels */}
          {[-2, -1, 0, 1, 2].map((k) => (
            <text key={k} x={sigPx(k * stdRes)} y={H - 4} textAnchor="middle" fontSize={7} fill="#94a3b8">
              {k === 0 ? '0' : `${k > 0 ? '+' : ''}${k}σ`}
            </text>
          ))}
          <text x={pad.l - 4} y={pad.t + ph / 2} textAnchor="middle" fontSize={7} fill="#94a3b8" transform={`rotate(-90,${pad.l - 14},${pad.t + ph / 2})`}>count</text>
          {/* Legend */}
          <line x1={pad.l + pw - 60} y1={pad.t + 10} x2={pad.l + pw - 50} y2={pad.t + 10} stroke="#f97316" strokeWidth={1.5} />
          <text x={pad.l + pw - 46} y={pad.t + 13} fontSize={7} fill="#94a3b8">mean</text>
        </svg>
      </div>

      {/* Coverage bars */}
      <div className="grid grid-cols-2 gap-2">
        {[
          { label: '±1σ coverage', val: cov1, expected: 68.3 },
          { label: '±2σ coverage', val: cov2, expected: 95.4 },
        ].map(({ label, val, expected }) => {
          const diff  = val - expected
          const color = Math.abs(diff) < 5 ? 'text-success' : Math.abs(diff) < 15 ? 'text-yellow-600' : 'text-danger'
          return (
            <div key={label} className="bg-white rounded-lg border border-border px-3 py-2">
              <div className="flex items-center justify-between mb-1">
                <span className="text-xs text-text-muted">{label}</span>
                <span className="text-xs font-mono font-semibold text-accent">{val.toFixed(1)}%</span>
              </div>
              <div className="h-1.5 bg-slate-200 rounded-full overflow-hidden">
                <div className="h-1.5 rounded-full bg-accent" style={{ width: `${Math.min(100, val)}%` }} />
              </div>
              <div className={`text-[10px] font-mono mt-1 ${color}`}>
                {diff >= 0 ? '+' : ''}{diff.toFixed(1)}% vs {expected}% ideal
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function UncertaintyPanel({
  plotQueries,
  modelResults,
}: {
  plotQueries: { data?: ModelPlotDataForUQ; isLoading: boolean; isError: boolean }[]
  modelResults: ModelResult[]
}) {
  const [uqView, setUqView] = useState<'per_model' | 'ensemble'>('per_model')

  const valid = useMemo(() => {
    return plotQueries
      .map((q, i) => ({ plot: q.data, model: modelResults[i] }))
      .filter((d) => d.plot && Array.isArray(d.plot.y_pred) && d.plot.y_pred.length > 0)
  }, [plotQueries, modelResults])

  if (plotQueries.some((q) => q.isLoading)) {
    return (
      <div className="flex items-center gap-3 py-10 text-sm text-text-muted">
        <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
        Loading field data for UQ analysis…
      </div>
    )
  }

  if (valid.length === 0) {
    return (
      <div className="py-10 text-center text-sm text-text-muted">
        No field data available yet.
      </div>
    )
  }

  // ── Per-model view ───────────────────────────────────────────────────────
  const perModelView = (
    <div className="space-y-4">
      <p className="text-xs text-text-muted">
        Per-model uncertainty estimated from empirical residuals (ŷ − y). σ = std of residuals;
        coverage calibration compares observed coverage vs Gaussian ideal.
      </p>
      {valid.map((d) => (
        <PerModelUQCard key={d.model.architecture_id} plotData={d.plot!} modelName={d.model.architecture_name} />
      ))}
    </div>
  )

  if (valid.length < 2) {
    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2">
          <Activity size={14} className="text-accent" />
          <span className="text-sm font-semibold text-text-primary">Per-Model Uncertainty</span>
          <span className="text-xs text-text-muted ml-1">(add a second model to enable ensemble UQ)</span>
        </div>
        {perModelView}
      </div>
    )
  }

  // ── Ensemble statistics (2+ models) ─────────────────────────────────────
  const refPlot = valid[0].plot!
  const X_eval  = refPlot.X_eval as number[][]
  const y_true  = flattenPred(refPlot.y_true as unknown[])
  const allPreds = valid.map((d) => flattenPred(d.plot!.y_pred as unknown[]))
  const nPts     = Math.min(X_eval.length, ...allPreds.map((p) => p.length), y_true.length)

  const ensMean = Array.from({ length: nPts }, (_, i) => {
    const vals = allPreds.map((p) => p[i]).filter(isFinite)
    return vals.length ? vals.reduce((a, b) => a + b, 0) / vals.length : 0
  })
  const ensStd = Array.from({ length: nPts }, (_, i) => {
    const vals = allPreds.map((p) => p[i]).filter(isFinite)
    if (vals.length < 2) return 0
    const m = ensMean[i]
    return Math.sqrt(vals.reduce((a, b) => a + (b - m) ** 2, 0) / vals.length)
  })

  const sortedIdx = Array.from({ length: nPts }, (_, i) => i)
    .sort((a, b) => (X_eval[a]?.[0] ?? 0) - (X_eval[b]?.[0] ?? 0))

  const xs       = sortedIdx.map((i) => X_eval[i]?.[0] ?? 0)
  const meanVals = sortedIdx.map((i) => ensMean[i])
  const stdVals  = sortedIdx.map((i) => ensStd[i])
  const trueVals = sortedIdx.map((i) => y_true[i] ?? 0)

  const calibration = [1, 2, 3].map((k) => {
    const covered = sortedIdx.filter((i) => Math.abs((y_true[i] ?? 0) - ensMean[i]) <= k * ensStd[i]).length
    return { k, expected: [68.3, 95.4, 99.7][k - 1], actual: (covered / nPts) * 100 }
  })

  const modelErrors = valid.map((d) => {
    const preds = flattenPred(d.plot!.y_pred as unknown[])
    const mae   = sortedIdx.reduce((s, i) => s + Math.abs((preds[i] ?? 0) - ensMean[i]), 0) / nPts
    const rmse  = Math.sqrt(sortedIdx.reduce((s, i) => s + ((preds[i] ?? 0) - ensMean[i]) ** 2, 0) / nPts)
    return { name: d.model.architecture_name, mae: mae.toFixed(5), rmse: rmse.toFixed(5) }
  })

  const W = 560; const H = 200
  const pad = { l: 40, r: 20, t: 10, b: 30 }
  const pw = W - pad.l - pad.r; const ph = H - pad.t - pad.b

  const allY = [...trueVals, ...meanVals, ...meanVals.map((m, i) => m + 2 * stdVals[i]), ...meanVals.map((m, i) => m - 2 * stdVals[i])].filter(isFinite)
  const xMin = xs[0] ?? 0; const xMax = xs[xs.length - 1] ?? 1
  const yMin = Math.min(...allY); const yMax = Math.max(...allY)
  const xR2 = xMax - xMin || 1; const yR = yMax - yMin || 1

  const epx = (x: number) => pad.l + ((x - xMin) / xR2) * pw
  const epy = (y: number) => pad.t + ph - ((y - yMin) / yR) * ph

  const lineD = (vals: number[]) =>
    vals.map((v, i) => `${i === 0 ? 'M' : 'L'}${epx(xs[i]).toFixed(1)},${epy(v).toFixed(1)}`).join(' ')

  const bandPath = (upper: number[], lower: number[]) => {
    const top = upper.map((v, i) => `${i === 0 ? 'M' : 'L'}${epx(xs[i]).toFixed(1)},${epy(v).toFixed(1)}`).join(' ')
    const bot = lower.map((v, i) => `L${epx(xs[lower.length - 1 - i]).toFixed(1)},${epy(v).toFixed(1)}`).reverse().join(' ')
    return `${top} ${bot} Z`
  }

  const mean1up = meanVals.map((m, i) => m + stdVals[i])
  const mean1dn = meanVals.map((m, i) => m - stdVals[i])
  const mean2up = meanVals.map((m, i) => m + 2 * stdVals[i])
  const mean2dn = meanVals.map((m, i) => m - 2 * stdVals[i])
  const meanStd = ensStd.reduce((a, b) => a + b, 0) / ensStd.length

  const ensembleView = (
    <div className="space-y-6">
      <div className="grid grid-cols-3 gap-4">
        {[
          { label: 'Models in ensemble', value: valid.length },
          { label: 'Evaluation points', value: nPts },
          { label: 'Mean ensemble σ', value: meanStd.toExponential(3) },
        ].map((m) => (
          <div key={m.label} className="bg-slate-50 rounded-xl border border-border p-4 text-center">
            <div className="text-xl font-bold text-accent font-mono">{m.value}</div>
            <div className="text-xs text-text-muted mt-0.5">{m.label}</div>
          </div>
        ))}
      </div>

      <div>
        <h3 className="text-sm font-semibold text-text-primary mb-3 flex items-center gap-2">
          <Activity size={14} className="text-accent" />
          Ensemble Prediction vs Ground Truth
          <span className="text-xs font-normal text-text-muted">(sorted by first coordinate)</span>
        </h3>
        <div className="overflow-x-auto">
          <svg width={W} height={H} className="font-mono">
            <path d={bandPath(mean2up, mean2dn)} fill="#3b82f620" />
            <path d={bandPath(mean1up, mean1dn)} fill="#3b82f640" />
            <path d={lineD(trueVals)} fill="none" stroke="#1e293b" strokeWidth={1.5} strokeDasharray="4 2" />
            <path d={lineD(meanVals)} fill="none" stroke="#3b82f6" strokeWidth={2} />
            <line x1={pad.l} y1={pad.t} x2={pad.l} y2={pad.t + ph} stroke="#cbd5e1" strokeWidth={1} />
            <line x1={pad.l} y1={pad.t + ph} x2={pad.l + pw} y2={pad.t + ph} stroke="#cbd5e1" strokeWidth={1} />
            <text x={pad.l + pw / 2} y={H - 4} textAnchor="middle" fontSize={9} fill="#94a3b8">x₁</text>
            <text x={pad.l - 5} y={pad.t + ph / 2} textAnchor="end" fontSize={9} fill="#94a3b8" transform={`rotate(-90,${pad.l - 5},${pad.t + ph / 2})`}>u</text>
            <g transform={`translate(${pad.l + 10},${pad.t + 8})`}>
              <line x1={0} y1={0} x2={16} y2={0} stroke="#1e293b" strokeWidth={1.5} strokeDasharray="4 2" /><text x={20} y={4} fontSize={8} fill="#475569">Ground truth</text>
              <line x1={0} y1={14} x2={16} y2={14} stroke="#3b82f6" strokeWidth={2} /><text x={20} y={18} fontSize={8} fill="#475569">Ensemble mean</text>
              <rect x={0} y={24} width={16} height={8} fill="#3b82f640" /><text x={20} y={32} fontSize={8} fill="#475569">±1σ band</text>
              <rect x={0} y={38} width={16} height={8} fill="#3b82f620" /><text x={20} y={46} fontSize={8} fill="#475569">±2σ band</text>
            </g>
          </svg>
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-text-primary mb-3">
          Ensemble Calibration
          <span className="text-xs font-normal text-text-muted ml-2">— % of true values covered by ±kσ interval</span>
        </h3>
        <div className="flex gap-4">
          {calibration.map(({ k, expected, actual }) => {
            const diff  = actual - expected
            const color = Math.abs(diff) < 5 ? 'text-success' : Math.abs(diff) < 15 ? 'text-yellow-600' : 'text-danger'
            return (
              <div key={k} className="flex-1 bg-slate-50 rounded-xl border border-border p-4">
                <div className="text-xs text-text-muted mb-2 font-semibold">±{k}σ interval</div>
                <div className="flex items-end gap-2">
                  <div className="text-2xl font-bold font-mono text-accent">{actual.toFixed(1)}%</div>
                  <div className="text-xs text-text-muted mb-1">covered</div>
                </div>
                <div className={`text-xs font-mono mt-1 ${color}`}>
                  {diff >= 0 ? '+' : ''}{diff.toFixed(1)}% vs {expected}% expected
                </div>
                <div className="mt-2 h-2 bg-slate-200 rounded-full overflow-hidden">
                  <div className="h-2 rounded-full bg-accent transition-all" style={{ width: `${Math.min(100, actual)}%` }} />
                </div>
              </div>
            )
          })}
        </div>
      </div>

      <div>
        <h3 className="text-sm font-semibold text-text-primary mb-3">Model Deviation from Ensemble Mean</h3>
        <div className="overflow-x-auto rounded-xl border border-border">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 border-b border-border">
              <tr>
                <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">Architecture</th>
                <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">MAE vs Ensemble</th>
                <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">RMSE vs Ensemble</th>
                <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">Agreement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {modelErrors
                .sort((a, b) => parseFloat(a.mae) - parseFloat(b.mae))
                .map(({ name, mae, rmse }) => {
                  const maeVal = parseFloat(mae)
                  const maxMae = Math.max(...modelErrors.map((e) => parseFloat(e.mae)))
                  const agreeW = maxMae > 0 ? (1 - maeVal / maxMae) * 100 : 100
                  return (
                    <tr key={name} className="hover:bg-slate-50 transition-colors">
                      <td className="py-2.5 px-4 font-medium text-text-primary">{name}</td>
                      <td className="py-2.5 px-4 font-mono text-text-secondary">{mae}</td>
                      <td className="py-2.5 px-4 font-mono text-text-secondary">{rmse}</td>
                      <td className="py-2.5 px-4">
                        <div className="flex items-center gap-2">
                          <div className="flex-1 h-1.5 bg-slate-200 rounded-full overflow-hidden">
                            <div className="h-1.5 rounded-full bg-success transition-all" style={{ width: `${agreeW}%` }} />
                          </div>
                          <span className="text-xs font-mono text-text-muted w-10 text-right">{agreeW.toFixed(0)}%</span>
                        </div>
                      </td>
                    </tr>
                  )
                })}
            </tbody>
          </table>
        </div>
      </div>

      <p className="text-xs text-text-muted">
        UQ computed as a cross-architecture ensemble. Calibration assumes Gaussian errors — divergence from expected coverage
        indicates model overconfidence (actual &lt; expected) or underconfidence (actual &gt; expected).
      </p>
    </div>
  )

  return (
    <div className="space-y-4">
      {/* Tab switcher */}
      <div className="flex gap-1 border-b border-border">
        {([
          { id: 'per_model' as const, label: 'Per-Model Uncertainty' },
          { id: 'ensemble' as const, label: `Ensemble UQ (${valid.length} models)` },
        ]).map((tab) => (
          <button
            key={tab.id}
            onClick={() => setUqView(tab.id)}
            className={`px-4 py-2 text-xs font-medium transition-all border-b-2 -mb-px ${
              uqView === tab.id
                ? 'text-accent border-accent'
                : 'text-text-muted border-transparent hover:text-text-primary'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>
      {uqView === 'per_model' ? perModelView : ensembleView}
    </div>
  )
}

// ── Inverse Results Panel ────────────────────────────────────────────────────

function InverseResultsPanel({ modelResults }: { modelResults: ModelResult[] }) {
  const inverseModels = modelResults.filter(
    (m) => m.inverse_result && !m.inverse_result.error && m.inverse_result.identified_params
  )

  if (inverseModels.length === 0) return null

  // Collect all parameter names across all models
  const allParams = Array.from(
    new Set(inverseModels.flatMap((m) => Object.keys(m.inverse_result!.identified_params ?? {})))
  )

  // Pick first model's true_params as reference (they should all be the same)
  const trueParams: Record<string, number | null> =
    inverseModels[0].inverse_result?.true_params ?? {}

  const fmtParam = (v: number | null | undefined): string => {
    if (v == null) return '—'
    if (Math.abs(v) < 1e-3 || Math.abs(v) > 1e4) return v.toExponential(4)
    return v.toFixed(6).replace(/\.?0+$/, '')
  }

  const relErr = (pred: number, truth: number | null | undefined): string => {
    if (truth == null || truth === 0) return '—'
    return `${(((pred - truth) / Math.abs(truth)) * 100).toFixed(2)}%`
  }

  const errColor = (pred: number, truth: number | null | undefined): string => {
    if (truth == null || truth === 0) return 'text-text-muted'
    const pct = Math.abs(((pred - truth) / Math.abs(truth)) * 100)
    if (pct < 1) return 'text-success'
    if (pct < 5) return 'text-yellow-600'
    return 'text-danger'
  }

  return (
    <div className="space-y-4">
      {/* Solver metadata */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Solver', value: inverseModels[0].inverse_result?.solver_method ?? '—' },
          { label: 'Observations', value: inverseModels[0].inverse_result?.n_obs ?? '—' },
          { label: 'Noise model', value: inverseModels[0].inverse_result?.noise_model ?? '—' },
          {
            label: 'Converged',
            value: inverseModels[0].inverse_result?.converged ? 'Yes' : 'No',
            highlight: inverseModels[0].inverse_result?.converged,
          },
        ].map((item) => (
          <div key={item.label} className="bg-slate-50 rounded-xl border border-border p-3 text-center">
            <div className={`text-sm font-bold font-mono ${
              item.highlight === true ? 'text-success' :
              item.highlight === false ? 'text-danger' : 'text-accent'
            }`}>
              {String(item.value)}
            </div>
            <div className="text-xs text-text-muted mt-0.5">{item.label}</div>
          </div>
        ))}
      </div>

      {/* Parameter comparison table */}
      <div className="overflow-x-auto rounded-xl border border-border">
        <table className="w-full text-sm">
          <thead className="bg-slate-50 border-b border-border">
            <tr>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4">Parameter</th>
              <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4">True Value</th>
              {inverseModels.map((m) => (
                <th key={m.architecture_id} className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4">
                  {m.architecture_name}
                </th>
              ))}
              {inverseModels.length > 1 && (
                <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-3 px-4">Error %</th>
              )}
            </tr>
          </thead>
          <tbody className="divide-y divide-border">
            {allParams.map((param) => {
              const truth = trueParams[param]
              return (
                <tr key={param} className="hover:bg-slate-50 transition-colors">
                  <td className="py-3 px-4 font-mono text-sm text-text-primary font-semibold">{param}</td>
                  <td className="py-3 px-4 font-mono text-sm text-text-secondary">
                    {truth != null ? fmtParam(truth) : <span className="text-text-muted italic">unknown</span>}
                  </td>
                  {inverseModels.map((m) => {
                    const pred = m.inverse_result?.identified_params?.[param]
                    return (
                      <td key={m.architecture_id} className="py-3 px-4">
                        <div className="font-mono text-sm text-text-primary">{fmtParam(pred)}</div>
                        {pred != null && truth != null && inverseModels.length === 1 && (
                          <div className={`text-xs font-mono mt-0.5 ${errColor(pred, truth)}`}>
                            {relErr(pred, truth)}
                          </div>
                        )}
                      </td>
                    )
                  })}
                  {inverseModels.length > 1 && (
                    <td className="py-3 px-4">
                      <div className="flex flex-col gap-0.5">
                        {inverseModels.map((m) => {
                          const pred = m.inverse_result?.identified_params?.[param]
                          return pred != null ? (
                            <span key={m.architecture_id} className={`text-xs font-mono ${errColor(pred, truth)}`}>
                              {m.architecture_name.split(' ')[0]}: {relErr(pred, truth)}
                            </span>
                          ) : null
                        })}
                      </div>
                    </td>
                  )}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Final residual */}
      {inverseModels.some((m) => m.inverse_result?.final_residual != null) && (
        <div className="text-xs text-text-muted font-mono bg-slate-50 border border-border rounded-lg px-4 py-2">
          Final residuals:{' '}
          {inverseModels
            .filter((m) => m.inverse_result?.final_residual != null)
            .map((m) => `${m.architecture_name}: ${Number(m.inverse_result!.final_residual).toExponential(3)}`)
            .join('  |  ')}
        </div>
      )}
    </div>
  )
}

const TAB_CONFIG: { id: ResultTab; label: string; icon: React.ReactNode }[] = [
  { id: 'loss_curves',       label: 'Loss Curves',       icon: <TrendingDown size={14} /> },
  { id: 'field_comparison',  label: 'Field Comparison',  icon: <Layers size={14} /> },
  { id: 'error_distribution',label: 'Error Distribution',icon: <BarChart3 size={14} /> },
  { id: 'scatter_plot',      label: 'Scatter Plot',      icon: <ChevronRight size={14} /> },
  { id: 'uncertainty',       label: 'Uncertainty (UQ)',  icon: <Activity size={14} /> },
]

function ExportButton({ data, filename, label }: { data: object; filename: string; label: string }) {
  const handleExport = () => {
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = filename
    a.click()
    URL.revokeObjectURL(url)
  }
  return (
    <Button variant="outline" size="sm" leftIcon={<Download size={14} />} onClick={handleExport}>
      {label}
    </Button>
  )
}

function exportToCsv(modelResults: ModelResult[]): string {
  const headers = ['Architecture', 'L2 Error', 'L1 Error', 'RMSE', 'Max Error', 'R2 Score', 'Training Time (s)', 'Converged']
  const rows = modelResults.map((m) => [
    m.architecture_name,
    m.metrics.l2_relative_error,
    m.metrics.l1_error,
    m.metrics.rmse,
    m.metrics.max_error,
    m.metrics.r2_score,
    m.training_time_s,
    m.converged ? 'true' : 'false',
  ])
  return [headers.join(','), ...rows.map((r) => r.join(','))].join('\n')
}

// ---------------------------------------------------------------------------
// Colormap helpers
// ---------------------------------------------------------------------------

function rdbu(t: number): string {
  t = Math.max(0, Math.min(1, t))
  const blue: [number, number, number] = [33, 102, 172]
  const white: [number, number, number] = [247, 247, 247]
  const red: [number, number, number] = [178, 24, 43]
  const [a, b, s] = t <= 0.5 ? [blue, white, t * 2] : [white, red, (t - 0.5) * 2]
  return `rgb(${Math.round(a[0] + s * (b[0] - a[0]))},${Math.round(a[1] + s * (b[1] - a[1]))},${Math.round(a[2] + s * (b[2] - a[2]))})`
}

function inferno(t: number): string {
  t = Math.max(0, Math.min(1, t))
  const stops: [number, number, number][] = [
    [0, 0, 4], [87, 16, 110], [188, 55, 84], [249, 142, 9], [252, 255, 164],
  ]
  const seg = t * 4
  const i = Math.min(Math.floor(seg), 3)
  const s = seg - i
  const a = stops[i]
  const b = stops[i + 1]
  return `rgb(${Math.round(a[0] + s * (b[0] - a[0]))},${Math.round(a[1] + s * (b[1] - a[1]))},${Math.round(a[2] + s * (b[2] - a[2]))})`
}

function toNum(v: unknown): number {
  if (Array.isArray(v)) return Number(v[0])
  return Number(v)
}

// ---------------------------------------------------------------------------
// MiniHeatmap SVG scatter plot
// ---------------------------------------------------------------------------

interface MiniHeatmapProps {
  coords: number[][]
  values: number[]
  colorFn: (t: number) => string
  width?: number
  height?: number
  title: string
  xLabel?: string
  yLabel?: string
}

function MiniHeatmap({
  coords,
  values,
  colorFn,
  width = 220,
  height = 160,
  title,
  xLabel = 'x',
  yLabel = 't',
}: MiniHeatmapProps) {
  if (!coords.length || !values.length) {
    return (
      <div className="flex items-center justify-center bg-slate-50 rounded border border-border text-xs text-text-muted" style={{ width, height }}>
        No data
      </div>
    )
  }

  const xs = coords.map((c) => c[0])
  const ys = coords.map((c) => c.length > 1 ? c[1] : 0)
  const xMin = Math.min(...xs), xMax = Math.max(...xs)
  const yMin = Math.min(...ys), yMax = Math.max(...ys)
  const vMin = Math.min(...values), vMax = Math.max(...values)
  const vRange = vMax - vMin || 1

  const r = Math.max(1.5, Math.min(4, 200 / Math.sqrt(coords.length)))
  const pad = { l: 30, r: 10, t: 22, b: 28 }
  const plotW = width - pad.l - pad.r
  const plotH = height - pad.t - pad.b

  return (
    <svg width={width} height={height}>
      <text x={width / 2} y={14} textAnchor="middle" fontSize={9} fill="#475569" fontWeight={600}>{title}</text>
      {coords.map((c, i) => {
        const px = pad.l + ((c[0] - xMin) / (xMax - xMin || 1)) * plotW
        const py = pad.t + plotH - (((c.length > 1 ? c[1] : 0) - yMin) / (yMax - yMin || 1)) * plotH
        const tv = (values[i] - vMin) / vRange
        return <circle key={i} cx={px} cy={py} r={r} fill={colorFn(tv)} opacity={0.85} />
      })}
      <text x={pad.l + plotW / 2} y={height - 4} textAnchor="middle" fontSize={8} fill="#94a3b8">{xLabel}</text>
      <text
        transform={`translate(9,${pad.t + plotH / 2})rotate(-90)`}
        textAnchor="middle"
        fontSize={8}
        fill="#94a3b8"
      >
        {yLabel}
      </text>
      <text x={pad.l} y={height - 4} fontSize={7} fill="#94a3b8">{vMin.toFixed(2)}</text>
      <text x={width - pad.r} y={height - 4} fontSize={7} fill="#94a3b8" textAnchor="end">{vMax.toFixed(2)}</text>
    </svg>
  )
}

// ---------------------------------------------------------------------------
// Plot data type
// ---------------------------------------------------------------------------

interface ModelPlotData {
  experiment_id: string
  model_id: string
  status: string
  metrics: Record<string, number>
  loss_history: number[]
  X_eval: number[][]
  y_true: unknown[]
  y_pred: unknown[]
  training_time_s: number
  rank?: number
}

// ---------------------------------------------------------------------------
// InferenceComparisonCard
// ---------------------------------------------------------------------------

interface InferenceComparisonCardProps {
  plotData: ModelPlotData
  architectureName: string
  l2Error: number
  r2Score: number
  rank: number
  problemId?: string
}

function FieldPanels({
  coords,
  yTrue,
  yPred,
  absErr,
  problemId,
}: {
  coords: number[][]
  yTrue: number[]
  yPred: number[]
  absErr: number[]
  problemId?: string
}) {
  const dim = coords[0]?.length ?? 1

  // Auto-colormap: viridis for non-negative, rdbu for signed
  const vminTrue = Math.min(...yTrue.filter(isFinite))
  const vmaxTrue = Math.max(...yTrue.filter(isFinite))
  const vmaxErr  = Math.max(...absErr.filter(isFinite))
  const fieldCm: ColormapName = vminTrue < -0.05 ? 'rdbu' : 'viridis'

  // ── 3D point cloud ──────────────────────────────────────────────────────
  if (dim === 3) {
    return <FieldPanels3D coords={coords} yTrue={yTrue} yPred={yPred} absErr={absErr} problemId={problemId} />
  }

  // ── 2D scatter heatmap (space-time or 2D spatial) ───────────────────────
  const hmTrue = buildHeatmapFromScatter(coords.map((c) => [c[0], c[1] ?? 0]), yTrue)
  const hmPred = buildHeatmapFromScatter(coords.map((c) => [c[0], c[1] ?? 0]), yPred)
  const hmErr  = buildHeatmapFromScatter(coords.map((c) => [c[0], c[1] ?? 0]), absErr)

  const ax1 = 'x'
  const ax2 = coords[0]?.length === 2 ? 't' : 'y'

  if (!hmTrue || !hmPred || !hmErr) return <div className="text-xs text-text-muted py-4 text-center">Cannot build heatmap from evaluation points</div>

  return (
    <div className="flex flex-wrap gap-4 justify-start overflow-x-auto">
      <div>
        <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1 text-center">True Solution</div>
        <ScientificHeatmap
          data={hmTrue.data} xValues={hmTrue.xValues} yValues={hmTrue.yValues}
          xLabel={ax1} yLabel={ax2} colormap={fieldCm}
          vmin={vminTrue} vmax={vmaxTrue}
          symmetric={fieldCm === 'rdbu'}
          width={260} height={180}
        />
      </div>
      <div>
        <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1 text-center">Prediction</div>
        <ScientificHeatmap
          data={hmPred.data} xValues={hmPred.xValues} yValues={hmPred.yValues}
          xLabel={ax1} yLabel={ax2} colormap={fieldCm}
          vmin={vminTrue} vmax={vmaxTrue}
          symmetric={fieldCm === 'rdbu'}
          width={260} height={180}
        />
      </div>
      <div>
        <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1 text-center">Absolute Error |u − û|</div>
        <ScientificHeatmap
          data={hmErr.data} xValues={hmErr.xValues} yValues={hmErr.yValues}
          xLabel={ax1} yLabel={ax2} colormap="viridis"
          vmin={0} vmax={vmaxErr}
          width={260} height={180}
        />
      </div>
    </div>
  )
}

/** Nearest-neighbour lookup: for each mesh vertex, find the closest X_eval point. */
function nnInterpolate(meshVerts: number[][], evalCoords: number[][], evalValues: number[]): number[] {
  return meshVerts.map((v) => {
    let bestDist = Infinity, bestVal = 0
    for (let j = 0; j < evalCoords.length; j++) {
      const e = evalCoords[j]
      const d = (v[0]-e[0])**2 + (v[1]-e[1])**2 + (v[2]-e[2])**2
      if (d < bestDist) { bestDist = d; bestVal = evalValues[j] }
    }
    return bestVal
  })
}

function FieldPanels3D({
  coords,
  yTrue,
  yPred,
  absErr,
  problemId,
}: {
  coords: number[][]
  yTrue: number[]
  yPred: number[]
  absErr: number[]
  problemId?: string
}) {
  const [field, setField] = useState<'true' | 'pred' | 'error'>('true')

  // Fetch the problem's dataset preview to get the surface mesh topology
  const { data: preview } = useQuery({
    queryKey: ['dataset-preview-mesh', problemId],
    queryFn: () => problemsApi.previewDataset(problemId!).then((r) => r.data),
    enabled: !!problemId,
    staleTime: 5 * 60 * 1000,
  })

  const surfaceMesh = preview?.surface_mesh as { vertices: number[][]; faces: number[][]; field?: number[] } | undefined

  // Interpolate eval predictions onto mesh vertices
  const { meshTrue, meshPred, meshErr } = useMemo(() => {
    if (!surfaceMesh?.vertices?.length || !coords.length) return { meshTrue: null, meshPred: null, meshErr: null }
    const meshTrue = nnInterpolate(surfaceMesh.vertices, coords, yTrue)
    const meshPred = nnInterpolate(surfaceMesh.vertices, coords, yPred)
    const meshErr  = nnInterpolate(surfaceMesh.vertices, coords, absErr)
    return { meshTrue, meshPred, meshErr }
  }, [surfaceMesh, coords, yTrue, yPred, absErr])

  const vminTrue = Math.min(...yTrue.filter(isFinite))
  const vmaxTrue = Math.max(...yTrue.filter(isFinite))
  const vmaxErr  = Math.max(...absErr.filter(isFinite))
  const fieldCm: ColormapName = vminTrue < -0.05 ? 'rdbu' : 'viridis'

  // Pre-compute fallback heatmap (used when no surface mesh) — must be before any early return
  const fallbackActiveValues = field === 'true' ? yTrue : field === 'pred' ? yPred : absErr
  const fallbackHm = useMemo(
    () => (coords.length ? buildHeatmapFromScatter(coords.map((c) => [c[0], c[1] ?? 0]), fallbackActiveValues) : null),
    [coords, fallbackActiveValues],
  )

  // If surface mesh available, show SurfaceMeshView
  if (surfaceMesh && meshTrue && meshPred && meshErr) {
    const activeField = field === 'true' ? meshTrue : field === 'pred' ? meshPred : meshErr
    const activeCm: ColormapName = field === 'error' ? 'viridis' : fieldCm
    const activeLabel = field === 'true' ? 'u (true)' : field === 'pred' ? 'û (pred)' : '|error|'
    const vmin = field === 'error' ? 0 : vminTrue
    const vmax = field === 'error' ? vmaxErr : vmaxTrue

    const mesh: SurfaceMeshData = {
      vertices: surfaceMesh.vertices,
      faces: surfaceMesh.faces,
      field: activeField,
      field_name: activeLabel,
    }

    return (
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          {(['true', 'pred', 'error'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setField(f)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
                field === f ? 'bg-accent text-white' : 'bg-slate-100 text-text-muted hover:bg-slate-200'
              }`}
            >
              {f === 'true' ? 'True Solution' : f === 'pred' ? 'Prediction' : 'Absolute Error'}
            </button>
          ))}
        </div>
        <SurfaceMeshView
          mesh={mesh}
          colormap={activeCm}
          title={activeLabel}
          fieldLabel={activeLabel}
          vmin={vmin}
          vmax={vmax}
          width={560}
          height={360}
        />
      </div>
    )
  }

  // Fallback: 2D projected heatmap (x vs y) when mesh not available
  const activeCm2: ColormapName = field === 'error' ? 'viridis' : fieldCm
  const activeLabel2 = field === 'true' ? 'u (true)' : field === 'pred' ? 'û (pred)' : '|error|'
  const activeVmin = field === 'error' ? 0 : vminTrue
  const activeVmax = field === 'error' ? vmaxErr : vmaxTrue

  const fieldToggle = (
    <div className="flex items-center gap-2">
      {(['true', 'pred', 'error'] as const).map((f) => (
        <button
          key={f}
          onClick={() => setField(f)}
          className={`px-3 py-1 rounded-md text-xs font-medium transition-colors ${
            field === f ? 'bg-accent text-white' : 'bg-slate-100 text-text-muted hover:bg-slate-200'
          }`}
        >
          {f === 'true' ? 'True Solution' : f === 'pred' ? 'Prediction' : 'Absolute Error'}
        </button>
      ))}
    </div>
  )

  if (!problemId || preview === undefined) {
    // Still loading the preview — show a spinner instead of "Loading surface mesh…"
    return (
      <div className="space-y-3">
        {fieldToggle}
        <div className="h-[180px] flex items-center justify-center text-xs text-text-muted gap-2">
          <div className="w-3 h-3 border-2 border-accent border-t-transparent rounded-full animate-spin" />
          Loading field data…
        </div>
      </div>
    )
  }

  if (fallbackHm) {
    return (
      <div className="space-y-3">
        {fieldToggle}
        <div>
          <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1 text-center">{activeLabel2}</div>
          <ScientificHeatmap
            data={fallbackHm.data}
            xValues={fallbackHm.xValues}
            yValues={fallbackHm.yValues}
            xLabel="x" yLabel="y"
            colormap={activeCm2}
            vmin={activeVmin} vmax={activeVmax}
            symmetric={activeCm2 === 'rdbu'}
            width={560} height={260}
          />
          <div className="text-[10px] text-text-muted mt-1 text-center">3D projection: x–y plane (z/t averaged)</div>
        </div>
      </div>
    )
  }

  // Last resort: plain scatter dots via MiniHeatmap
  return (
    <div className="space-y-3">
      {fieldToggle}
      <MiniHeatmap
        coords={coords.map((c) => [c[0], c[1] ?? 0])}
        values={fallbackActiveValues}
        colorFn={activeCm2 === 'rdbu' ? rdbu : inferno}
        width={560} height={260}
        title={activeLabel2}
        xLabel="x" yLabel="y"
      />
    </div>
  )
}

function InferenceComparisonCard({
  plotData,
  architectureName,
  l2Error,
  r2Score,
  rank,
  problemId,
}: InferenceComparisonCardProps) {
  const coords: number[][] = plotData.X_eval ?? []
  const yTrue: number[] = (plotData.y_true ?? []).map(toNum)
  const yPred: number[] = (plotData.y_pred ?? []).map(toNum)
  const absErr: number[] = yTrue.map((v, i) => Math.abs(v - (yPred[i] ?? 0)))

  const hasData = coords.length > 0 && yTrue.length > 0 && yPred.length > 0

  return (
    <Card className="overflow-hidden">
      {/* Card header */}
      <div className="flex items-center justify-between px-5 py-3 border-b border-border bg-slate-50">
        <div className="flex items-center gap-2.5">
          <div className="flex items-center gap-1.5">
            {RANK_ICONS[rank] ?? (
              <span className="text-sm font-bold text-text-muted">#{rank}</span>
            )}
          </div>
          <span className="text-sm font-semibold text-text-primary">{architectureName}</span>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="default" size="xs">
            L2: <span className="font-mono ml-1">{formatMetric(l2Error)}</span>
          </Badge>
          <Badge variant={r2Score >= 0.9 ? 'success' : r2Score >= 0.5 ? 'warning' : 'danger'} size="xs">
            R² <span className="font-mono ml-1">{r2Score.toFixed(4)}</span>
          </Badge>
        </div>
      </div>

      <CardContent className="py-4">
        {!hasData ? (
          <div className="flex items-center justify-center h-24 text-sm text-text-muted">
            No field data available for this model
          </div>
        ) : (
          <FieldPanels coords={coords} yTrue={yTrue} yPred={yPred} absErr={absErr} problemId={problemId} />
        )}
      </CardContent>
    </Card>
  )
}

// ---------------------------------------------------------------------------
// ResultsPage
// ---------------------------------------------------------------------------

export function ResultsPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { selectedProblem, reset } = useExperimentStore()
  const [activeTab, setActiveTab] = useState<ResultTab>('loss_curves')

  const { data: results, isLoading, error } = useExperimentResults(id)
  const isInverse = results?.problem_type === 'inverse'

  const [detailArch, setDetailArch] = useState<ArchitectureInfo | null>(null)

  // Pre-fetch ArchitectureInfo for every model so Details button works immediately
  const archQueries = useQueries({
    queries: (results?.model_results ?? []).map((m) => ({
      queryKey: ['architecture', m.architecture_id],
      queryFn: () => architecturesApi.get(m.architecture_id).then((r) => r.data as ArchitectureInfo),
      enabled: !!results,
      staleTime: 600_000,
    })),
  })
  const archInfoMap = useMemo<Record<string, ArchitectureInfo>>(() => {
    const map: Record<string, ArchitectureInfo> = {}
    archQueries.forEach((q, i) => {
      const id = results?.model_results[i]?.architecture_id
      if (id && q.data) map[id] = q.data
    })
    return map
  }, [archQueries, results?.model_results])

  const plotQueries = useQueries({
    queries: (results?.model_results ?? []).map((m) => ({
      queryKey: ['model-plots', id, m.architecture_id],
      queryFn: () =>
        experimentsApi.getModelPlots(id!, m.architecture_id).then((r) => r.data as ModelPlotData),
      enabled: !!id && (results?.status === 'COMPLETED' || (results?.status as string) === 'completed'),
      staleTime: 60_000,
    })),
  })

  if (isLoading) return <PageLoading label="Loading experiment results…" />

  if (error || !results) {
    const errMsg = (() => {
      if (!error) return null
      const e = error as any
      // 422 = experiment failed (training error); extract the detail message
      const detail = e?.response?.data?.detail ?? e?.message ?? null
      return detail
    })()
    const isFailed = (error as any)?.response?.status === 422

    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center max-w-lg">
          {isFailed
            ? <XCircle size={40} className="text-danger mx-auto mb-3" />
            : <Trophy size={40} className="text-slate-300 mx-auto mb-3" />
          }
          <p className="text-text-secondary font-medium">
            {isFailed ? 'Training failed' : 'Results not found or experiment still running'}
          </p>
          {errMsg && (
            <p className="text-xs text-text-muted mt-2 font-mono bg-slate-50 border border-slate-200 rounded px-3 py-2 text-left break-all">
              {errMsg.replace(/^Experiment failed:\s*/i, '')}
            </p>
          )}
          <div className="flex gap-2 justify-center mt-4">
            <Button variant="outline" size="sm" onClick={() => navigate(`/experiment/${id}/training`)}>
              Back to Training
            </Button>
            <Button variant="outline" size="sm" onClick={() => navigate('/problems')}>
              Go to Problems
            </Button>
          </div>
        </div>
      </div>
    )
  }

  const bestModel = results.model_results.find((m) => m.architecture_id === results.best_model_id)
  const completedAt = results.completed_at ? new Date(results.completed_at) : null

  const handleNewExperiment = () => {
    reset()
    navigate('/experiment/setup')
  }

  const csvData = exportToCsv(results.model_results)
  const csvBlob = () => {
    const blob = new Blob([csvData], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `experiment_${id}_results.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  return (
    <div className="px-4 sm:px-8 py-4 sm:py-6 space-y-8 pb-16 bg-background min-h-full">
      {/* Top Section */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2 text-xs text-text-muted mb-1.5">
            <span>Experiment</span>
            <ChevronRight size={12} />
            <span className="font-mono">{id}</span>
          </div>
          <div className="flex items-center gap-3 mb-2">
            <CheckCircle size={22} className="text-success flex-shrink-0" />
            <h1 className="text-2xl font-bold text-text-primary">Experiment Complete</h1>
          </div>
          {selectedProblem && (
            <p className="text-sm text-text-secondary">
              Problem: <span className="text-text-primary font-medium">{selectedProblem.name}</span>
            </p>
          )}
          {completedAt && (
            <div className="flex items-center gap-1.5 text-xs text-text-muted mt-1">
              <Clock size={12} />
              <span>Completed {completedAt.toLocaleString()}</span>
            </div>
          )}
        </div>

        <div className="flex items-center gap-3 flex-wrap">
          <Button variant="outline" size="sm" leftIcon={<Download size={14} />} onClick={csvBlob}>
            Export CSV
          </Button>
          <ExportButton data={results} filename={`experiment_${id}_results.json`} label="Export JSON" />
          <Button
            variant="outline"
            size="sm"
            leftIcon={<Download size={14} />}
            onClick={() => experimentsApi.downloadAllModels(id!, results?.problem_id ?? 'experiment')}
          >
            Download All Models (.zip)
          </Button>
          <Button
            variant="primary"
            size="sm"
            leftIcon={<Plus size={14} />}
            onClick={handleNewExperiment}
          >
            New Experiment
          </Button>
        </div>
      </div>

      {/* Winner banner */}
      {bestModel && (
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          className={`bg-gradient-to-r border rounded-2xl p-5 flex items-center gap-5 ${
            isInverse
              ? 'from-purple-50 via-violet-50 to-white border-purple-200'
              : 'from-yellow-50 via-amber-50 to-white border-yellow-200'
          }`}
        >
          <div className={`w-14 h-14 rounded-xl border flex items-center justify-center flex-shrink-0 ${
            isInverse ? 'bg-purple-100 border-purple-300' : 'bg-yellow-100 border-yellow-300'
          }`}>
            {isInverse
              ? <Microscope size={28} className="text-purple-500" />
              : <Crown size={28} className="text-yellow-500" />
            }
          </div>
          <div className="flex-1 min-w-0">
            <div className={`text-xs font-semibold uppercase tracking-wider mb-0.5 ${
              isInverse ? 'text-purple-600' : 'text-yellow-600'
            }`}>
              {isInverse ? 'Inverse Problem Solved' : 'Best Performing Model'}
            </div>
            <h2 className="text-lg font-bold text-text-primary">{bestModel.architecture_name}</h2>
            <div className="flex items-center gap-4 mt-1.5 flex-wrap">
              {(isInverse ? [
                { label: 'Final Loss', value: formatMetric(bestModel.final_loss) },
                { label: 'Time', value: `${bestModel.training_time_s.toFixed(1)}s` },
              ] : [
                { label: 'L2 Error', value: formatMetric(bestModel.metrics.l2_relative_error) },
                { label: 'R²', value: bestModel.metrics.r2_score.toFixed(4) },
                { label: 'RMSE', value: formatMetric(bestModel.metrics.rmse) },
                { label: 'Time', value: `${bestModel.training_time_s.toFixed(1)}s` },
              ]).map((m) => (
                <div key={m.label} className="text-xs">
                  <span className="text-text-muted">{m.label}: </span>
                  <span className="font-mono font-semibold text-success">{m.value}</span>
                </div>
              ))}
              {bestModel.converged && <Badge variant="success" size="xs">Converged</Badge>}
            </div>
          </div>
          <div className={`text-5xl font-black select-none ${isInverse ? 'text-purple-100' : 'text-yellow-200'}`}>#1</div>
        </motion.div>
      )}

      {/* ── Inverse Problem Layout ── */}
      {isInverse ? (
        <>
          {/* Parameter Recovery — main result */}
          <div>
            <h2 className="text-base font-bold text-text-primary flex items-center gap-2 mb-4">
              <Microscope size={18} className="text-purple-500" />
              Parameter Recovery Results
              <Badge variant="default" size="sm">Inverse Problem</Badge>
            </h2>
            <Card>
              <CardContent className="py-5">
                <InverseResultsPanel modelResults={results.model_results} />
              </CardContent>
            </Card>
          </div>

          {/* Training Loss */}
          <div>
            <h2 className="text-base font-bold text-text-primary flex items-center gap-2 mb-4">
              <TrendingDown size={18} className="text-accent" />
              Training Loss
            </h2>
            <Card>
              <CardContent className="py-5">
                <LossCurvesChart modelResults={results.model_results} />
              </CardContent>
            </Card>
          </div>

          {/* Model cards — simplified for inverse */}
          <div>
            <h2 className="text-base font-bold text-text-primary flex items-center gap-2 mb-4">
              <Layers size={18} className="text-accent" />
              Model Details
            </h2>
            <div className="space-y-3">
              {results.model_results.map((m, i) => (
                <ModelDetailCard key={m.architecture_id} model={m} rank={results.ranking.indexOf(m.architecture_id) + 1} index={i} archInfo={archInfoMap[m.architecture_id]} onDetails={setDetailArch} />
              ))}
            </div>
          </div>
        </>
      ) : (
        <>
          {/* ── Forward Problem Layout ── */}

          {/* Ranking Table */}
          <div>
            <h2 className="text-base font-bold text-text-primary flex items-center gap-2 mb-4">
              <Trophy size={18} className="text-yellow-500" />
              Model Rankings
              <Badge variant="default" size="sm">{results.model_results.length} models</Badge>
            </h2>

            {/* Validation alerts */}
            {(() => {
              const bad = results.model_results.filter(
                (m) => m.quality_level === 'unreliable' || m.quality_level === 'poor'
              )
              if (!bad.length) return null
              return (
                <div className="mb-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 flex gap-3">
                  <AlertTriangle size={16} className="text-amber-500 shrink-0 mt-0.5" />
                  <div className="text-sm text-amber-800">
                    <span className="font-semibold">
                      {bad.length} model{bad.length > 1 ? 's' : ''} flagged:
                    </span>{' '}
                    {bad.map((m) => m.architecture_name).join(', ')} produced
                    degenerate or unreliable predictions. Check the Quality column for details.
                  </div>
                </div>
              )
            })()}

            <RankingTable
              modelResults={results.model_results}
              ranking={results.ranking}
              bestModelId={results.best_model_id}
              experimentId={id}
              archInfoMap={archInfoMap}
              onDetails={setDetailArch}
            />
          </div>

          {/* Multi-Objective Comparison */}
          <ExperimentMultiObjectiveView modelResults={results.model_results} />

          {/* Comparative Plots */}
          <div>
            <h2 className="text-base font-bold text-text-primary flex items-center gap-2 mb-4">
              <BarChart3 size={18} className="text-accent" />
              Comparative Analysis
            </h2>

            <div className="flex gap-1 mb-4 border-b border-border">
              {TAB_CONFIG.map((tab) => (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium transition-all duration-150 border-b-2 -mb-px ${
                    activeTab === tab.id
                      ? 'text-accent border-accent'
                      : 'text-text-muted border-transparent hover:text-text-primary hover:border-slate-300'
                  }`}
                >
                  {tab.icon}
                  {tab.label}
                </button>
              ))}
            </div>

            <AnimatePresence mode="wait">
              <motion.div
                key={activeTab}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.15 }}
              >
                <Card>
                  <CardContent className="py-5">
                    {activeTab === 'loss_curves' && (
                      <LossCurvesChart modelResults={results.model_results} />
                    )}
                    {activeTab === 'field_comparison' && (
                      <FieldComparisonChart modelResults={results.model_results} />
                    )}
                    {activeTab === 'error_distribution' && (
                      <ErrorDistributionChart modelResults={results.model_results} />
                    )}
                    {activeTab === 'scatter_plot' && (
                      <TrainingScatterPlot modelResults={results.model_results} />
                    )}
                    {activeTab === 'uncertainty' && (
                      <UncertaintyPanel
                        plotQueries={plotQueries}
                        modelResults={results.model_results}
                      />
                    )}
                  </CardContent>
                </Card>
              </motion.div>
            </AnimatePresence>
          </div>

          {/* Detailed Model Cards */}
          <div>
            <h2 className="text-base font-bold text-text-primary flex items-center gap-2 mb-4">
              <Layers size={18} className="text-accent" />
              Detailed Model Reports
            </h2>
            <div className="space-y-3">
              {results.model_results.map((m, i) => (
                <ModelDetailCard key={m.architecture_id} model={m} rank={results.ranking.indexOf(m.architecture_id) + 1} index={i} archInfo={archInfoMap[m.architecture_id]} onDetails={setDetailArch} />
              ))}
            </div>
          </div>
        </>
      )}

      {/* Architecture Detail Modal */}
      <ArchitectureDetailModal architecture={detailArch} onClose={() => setDetailArch(null)} />

      {/* Inference Comparison — forward mode only */}
      {!isInverse && <div>
        <h2 className="text-base font-bold text-text-primary flex items-center gap-2 mb-4">
          <Microscope size={18} className="text-accent" />
          Field Comparison
          <Badge variant="default" size="sm">Inference</Badge>
        </h2>

        {plotQueries.some((q) => q.isLoading) ? (
          <div className="flex items-center gap-3 py-8 text-sm text-text-muted">
            <div className="w-4 h-4 border-2 border-accent border-t-transparent rounded-full animate-spin" />
            Loading field data…
          </div>
        ) : (
          <div className="space-y-4">
            {results.model_results.map((m, i) => {
              const q = plotQueries[i]
              const plotData = q?.data
              const rank = (results.ranking.indexOf(m.architecture_id) + 1) || results.model_results.length

              if (!plotData) {
                return (
                  <Card key={m.architecture_id} className="px-5 py-4 text-sm text-text-muted">
                    {m.architecture_name}: field data unavailable
                    {q?.isError && (
                      <span className="ml-2 text-danger text-xs">(fetch failed)</span>
                    )}
                  </Card>
                )
              }

              return (
                <motion.div
                  key={m.architecture_id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.05 }}
                >
                  <InferenceComparisonCard
                    plotData={plotData}
                    architectureName={m.architecture_name}
                    l2Error={m.metrics.l2_relative_error}
                    r2Score={m.metrics.r2_score}
                    rank={rank}
                    problemId={selectedProblem?.id ?? results.problem_id}
                  />
                </motion.div>
              )
            })}
          </div>
        )}
      </div>}
    </div>
  )
}

function ModelDetailCard({
  model, rank, index, archInfo, onDetails,
}: {
  model: ModelResult
  rank: number
  index: number
  archInfo?: ArchitectureInfo
  onDetails?: (arch: ArchitectureInfo) => void
}) {
  const [expanded, setExpanded] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: index * 0.05 }}
    >
      <Card className={rank <= 3 ? getRankStyle(rank) : ''}>
        <button
          className="w-full px-5 py-4 flex items-center justify-between text-left"
          onClick={() => setExpanded(!expanded)}
        >
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-1.5">
              {RANK_ICONS[rank] ?? (
                <span className="text-sm font-bold text-text-muted"># {rank}</span>
              )}
            </div>
            <div>
              <div className="text-sm font-semibold text-text-primary">{model.architecture_name}</div>
              <div className="flex items-center gap-3 mt-1 text-xs">
                <span className="text-text-muted">L2: <span className="text-success font-mono">{formatMetric(model.metrics.l2_relative_error)}</span></span>
                <span className="text-text-muted">R²: <span className="text-accent font-mono">{model.metrics.r2_score.toFixed(4)}</span></span>
                <span className="text-text-muted">Time: <span className="font-mono text-text-primary">{model.training_time_s.toFixed(1)}s</span></span>
                {model.converged && <Badge variant="success" size="xs">Converged</Badge>}
              </div>
            </div>
          </div>
          <ChevronDownIcon
            size={16}
            className={`text-slate-400 transition-transform ${expanded ? 'rotate-180' : ''}`}
          />
        </button>

        <AnimatePresence>
          {expanded && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden border-t border-border"
            >
              <div className="px-5 py-4 bg-slate-50">
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                  {[
                    { label: 'L1 Error', value: formatMetric(model.metrics.l1_error) },
                    { label: 'L2 Relative Error', value: formatMetric(model.metrics.l2_relative_error) },
                    { label: 'RMSE', value: formatMetric(model.metrics.rmse) },
                    { label: 'Max Error', value: formatMetric(model.metrics.max_error) },
                    { label: 'R² Score', value: model.metrics.r2_score.toFixed(6) },
                    { label: 'Final Loss', value: formatMetric(model.final_loss) },
                    { label: 'Training Time', value: `${model.training_time_s.toFixed(2)}s` },
                    { label: 'History Points', value: model.training_history.length.toString() },
                  ].map((item) => (
                    <div key={item.label}>
                      <div className="text-xs text-text-muted mb-0.5">{item.label}</div>
                      <div className="text-sm font-mono text-text-primary">{item.value}</div>
                    </div>
                  ))}
                </div>
                {archInfo && onDetails && (
                  <div className="pt-3 border-t border-border">
                    <button
                      onClick={() => onDetails(archInfo)}
                      className="flex items-center gap-1.5 rounded-lg border border-accent/30 bg-accent/5 px-3 py-1.5 text-xs font-semibold text-accent hover:bg-accent/10 transition-colors"
                    >
                      <Layers size={12} /> Architecture Details
                    </button>
                  </div>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </Card>
    </motion.div>
  )
}
