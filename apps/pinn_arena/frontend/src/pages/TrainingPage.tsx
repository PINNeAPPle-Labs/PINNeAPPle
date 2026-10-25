import React, { useEffect, useRef, useState } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import {
  AlertTriangle, CheckCircle2, ChevronRight, Terminal,
  Clock, XCircle, BarChart3, RefreshCw,
} from 'lucide-react'
import { useExperimentStatus } from '../hooks/useExperiment'
import { useExperimentStore } from '../store/experimentStore'
import { experimentsApi } from '../lib/api'
import { TrainingProgressCard } from '../components/TrainingProgressCard'
import ArchitectureDetailModal from '../components/ArchitectureDetailModal'
import { Button } from '../components/ui/Button'
import { ProgressBar } from '../components/ui/ProgressBar'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { Badge } from '../components/ui/Badge'
import { PageLoading } from '../components/ui/LoadingSpinner'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { FAMILY_CONFIG } from '../components/ArchitectureCard'
import type { ArchitectureFamily, ArchitectureInfo } from '../types'

const MODEL_COLORS = [
  '#2563eb', '#059669', '#d97706', '#dc2626', '#a855f7',
  '#0891b2', '#ea580c', '#db2777', '#ca8a04', '#0d9488',
]

function ProgressRing({ progress, size = 60, color = '#2563eb' }: { progress: number; size?: number; color?: string }) {
  const r = (size - 8) / 2
  const circumference = 2 * Math.PI * r
  const offset = circumference - (progress / 100) * circumference

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="-rotate-90">
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="#e2e8f0" strokeWidth="4" />
      <circle
        cx={size / 2}
        cy={size / 2}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth="4"
        strokeDasharray={circumference}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: 'stroke-dashoffset 0.5s ease' }}
      />
    </svg>
  )
}

export function TrainingPage() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { selectedProblem, selectedArchitectures } = useExperimentStore()
  const logEndRef = useRef<HTMLDivElement>(null)
  const [isCancelling, setIsCancelling] = useState(false)
  const [detailArch, setDetailArch] = useState<ArchitectureInfo | null>(null)

  // Persist training history across page reloads and backend restarts.
  // We keep the longer of (cached, incoming) so pre-crash history stays visible.
  const [historyCache, setHistoryCache] = useState<Record<string, any[]>>(() => {
    if (!id) return {}
    try {
      const stored = sessionStorage.getItem(`tph_${id}`)
      return stored ? JSON.parse(stored) : {}
    } catch { return {} }
  })

  const { data: status, isLoading, error } = useExperimentStatus(id)

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [status?.logs])

  // Navigate to results on completion
  useEffect(() => {
    if (status?.status === 'COMPLETED') {
      const timer = setTimeout(() => navigate(`/experiment/${id}/results`), 1500)
      return () => clearTimeout(timer)
    }
  }, [status?.status, id, navigate])

  // Merge incoming training history into cache — survives backend restarts
  useEffect(() => {
    if (!status?.model_statuses || !id) return
    setHistoryCache(prev => {
      const next = { ...prev }
      let changed = false
      for (const [archId, ms] of Object.entries(status.model_statuses)) {
        const incoming = ms.training_history ?? []
        if (incoming.length > (prev[archId]?.length ?? 0)) {
          next[archId] = incoming
          changed = true
        }
      }
      if (changed) {
        try { sessionStorage.setItem(`tph_${id}`, JSON.stringify(next)) } catch { /* quota */ }
      }
      return changed ? next : prev
    })
  }, [status?.model_statuses, id])

  const handleCancel = async () => {
    if (!id) return
    setIsCancelling(true)
    try {
      await experimentsApi.cancel(id)
    } finally {
      setIsCancelling(false)
    }
  }

  // All architectures to display: those in model_statuses + selected ones not yet started
  const displayArchEntries = React.useMemo(() => {
    const fromStatus = Object.entries(status?.model_statuses ?? {})
    const statusIds = new Set(fromStatus.map(([archId]) => archId))
    const pending = selectedArchitectures
      .filter(a => !statusIds.has(a.id))
      .map(a => [a.id, { status: 'waiting' as const, progress: 0, training_history: [] as any[] }] as [string, any])
    return [...fromStatus, ...pending]
  }, [status?.model_statuses, selectedArchitectures])

  // Build live chart data from accumulated history (survives crashes)
  const liveChartData = React.useMemo(() => {
    const allIds = [...new Set([
      ...Object.keys(historyCache),
      ...Object.keys(status?.model_statuses ?? {}),
    ])]
    const maxLen = Math.max(...allIds.map(aid => historyCache[aid]?.length ?? 0), 0)
    if (maxLen === 0) return []

    const archNames = selectedArchitectures.reduce<Record<string, string>>((acc, a) => {
      acc[a.id] = a.name
      return acc
    }, {})

    return Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, number | string> = { epoch: i }
      allIds.forEach(archId => {
        const h = historyCache[archId]?.[i]
        if (h) {
          const name = archNames[archId] ?? archId
          point[name] = h.total_loss
        }
      })
      return point
    })
  }, [historyCache, status?.model_statuses, selectedArchitectures])

  if (isLoading) return <PageLoading label="Connecting to experiment…" />

  if (error) {
    return (
      <div className="flex items-center justify-center min-h-[400px]">
        <div className="text-center">
          <AlertTriangle size={40} className="text-danger mx-auto mb-3" />
          <p className="text-text-secondary">Failed to load experiment status</p>
          <p className="text-sm text-text-muted mt-1">Experiment ID: {id}</p>
          <Button variant="outline" size="sm" className="mt-4" onClick={() => navigate('/problems')}>
            Go to Problems
          </Button>
        </div>
      </div>
    )
  }

  const isCompleted = status?.status === 'COMPLETED'
  const isFailed = status?.status === 'FAILED'
  const isRunning = status?.status === 'RUNNING'

  const overallProgress = (status?.progress ?? 0) * 100   // backend sends 0.0–1.0, convert to 0–100

  return (
    <div className="px-4 sm:px-8 py-4 sm:py-6 space-y-6 bg-background min-h-full">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <div className="flex items-center gap-2 text-xs text-text-muted mb-1">
            <span>Experiment</span>
            <ChevronRight size={12} />
            <span className="font-mono">{id}</span>
          </div>
          <h1 className="text-xl font-bold text-text-primary">
            {selectedProblem?.name ?? 'Training Experiment'}
          </h1>
          <p className="text-sm text-text-secondary mt-1">
            Benchmarking {selectedArchitectures.length || Object.keys(status?.model_statuses ?? {}).length} architectures
          </p>
        </div>

        <div className="flex items-center gap-3">
          {/* Status badge */}
          {isCompleted && (
            <Badge variant="success" size="md">
              <CheckCircle2 size={13} className="mr-1" />
              Completed
            </Badge>
          )}
          {isFailed && (
            <Badge variant="danger" size="md">
              <XCircle size={13} className="mr-1" />
              Failed
            </Badge>
          )}
          {isRunning && (
            <Badge variant="info" size="md">
              <RefreshCw size={13} className="mr-1 animate-spin" />
              Running
            </Badge>
          )}

          {isRunning && (
            <Button
              variant="danger"
              size="sm"
              loading={isCancelling}
              onClick={handleCancel}
            >
              Cancel
            </Button>
          )}
          {isCompleted && (
            <Button
              variant="primary"
              size="sm"
              rightIcon={<ChevronRight size={14} />}
              onClick={() => navigate(`/experiment/${id}/results`)}
            >
              View Results
            </Button>
          )}
        </div>
      </div>

      {/* Overall progress */}
      <Card>
        <CardContent className="py-4">
          <div className="flex items-center justify-between mb-3">
            <div>
              <div className="text-sm font-medium text-text-primary">
                {isCompleted ? 'Training Complete' : isFailed ? 'Training Failed' : 'Overall Progress'}
              </div>
              {status?.current_architecture && !isCompleted && (
                <div className="text-xs text-text-secondary mt-0.5">
                  Currently training: <span className="text-accent font-medium">{status.current_architecture}</span>
                  {status.current_epoch && status.total_epochs && (
                    <span className="text-text-muted ml-2">
                      Epoch {status.current_epoch.toLocaleString()} / {status.total_epochs.toLocaleString()}
                    </span>
                  )}
                </div>
              )}
            </div>
            <div className="flex items-center gap-3">
              {status?.started_at && (
                <div className="flex items-center gap-1 text-xs text-text-muted">
                  <Clock size={12} />
                  <span>{new Date(status.started_at).toLocaleTimeString()}</span>
                </div>
              )}
              <div className="text-sm font-mono font-bold text-accent">{Math.round(overallProgress)}%</div>
            </div>
          </div>
          <ProgressBar
            value={overallProgress}
            max={100}
            variant={isCompleted ? 'success' : isFailed ? 'danger' : 'default'}
            size="lg"
            animated={isRunning}
          />
        </CardContent>
      </Card>

      {/* Completion animation */}
      <AnimatePresence>
        {isCompleted && (
          <motion.div
            initial={{ opacity: 0, scale: 0.95 }}
            animate={{ opacity: 1, scale: 1 }}
            className="bg-green-50 border border-green-200 rounded-xl p-4 flex items-center gap-3"
          >
            <CheckCircle2 size={20} className="text-success flex-shrink-0" />
            <div>
              <div className="text-sm font-semibold text-success">All models trained successfully!</div>
              <div className="text-xs text-text-muted mt-0.5">Redirecting to results page…</div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
        {/* Model cards */}
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-text-primary flex items-center gap-1.5">
            <BarChart3 size={15} className="text-accent" />
            Model Status
          </h2>
          <div className="space-y-3">
            {displayArchEntries.map(([archId, ms], i) => {
              const arch = selectedArchitectures.find((a) => a.id === archId)
              const history = historyCache[archId] ?? ms.training_history ?? []
              const currentLoss = history[history.length - 1]?.total_loss

              return (
                <motion.div
                  key={archId}
                  initial={{ opacity: 0, x: -10 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: i * 0.05 }}
                >
                  <TrainingProgressCard
                    architectureId={archId}
                    architectureName={arch?.name ?? archId}
                    architectureFamily={arch?.family as ArchitectureFamily}
                    status={ms.status}
                    progress={(ms.progress ?? 0) * 100}
                    currentLoss={currentLoss}
                    trainingHistory={history}
                    finalLoss={ms.status === 'completed' ? currentLoss : undefined}
                    converged={ms.status === 'completed' ? (currentLoss !== undefined && currentLoss < 1e-3) : undefined}
                    onDetails={arch ? () => setDetailArch(arch) : undefined}
                  />
                </motion.div>
              )
            })}
          </div>
        </div>

        {/* Live loss chart */}
        <div className="space-y-3">
          <h2 className="text-sm font-semibold text-text-primary flex items-center gap-1.5">
            <BarChart3 size={15} className="text-accent" />
            Live Loss Curves
          </h2>
          <Card>
            <CardContent className="py-4">
              {liveChartData.length > 0 ? (
                <ResponsiveContainer width="100%" height={320}>
                  <LineChart data={liveChartData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                    <XAxis
                      dataKey="epoch"
                      tick={{ fill: '#64748b', fontSize: 10 }}
                      axisLine={{ stroke: '#e2e8f0' }}
                      tickLine={false}
                    />
                    <YAxis
                      scale="log"
                      domain={['auto', 'auto']}
                      tick={{ fill: '#64748b', fontSize: 10 }}
                      axisLine={{ stroke: '#e2e8f0' }}
                      tickLine={false}
                      tickFormatter={(v: number) => v.toExponential(1)}
                    />
                    <Tooltip
                      contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 8, color: '#0f172a', fontSize: 11, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
                      formatter={(v: number) => [v.toExponential(4), '']}
                    />
                    <Legend wrapperStyle={{ color: '#64748b', fontSize: 11 }} />
                    {displayArchEntries.map(([archId], i) => {
                      const arch = selectedArchitectures.find((a) => a.id === archId)
                      const name = arch?.name ?? archId
                      return (
                        <Line
                          key={archId}
                          type="monotone"
                          dataKey={name}
                          stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                          strokeWidth={2}
                          dot={false}
                          isAnimationActive={false}
                        />
                      )
                    })}
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="h-64 flex flex-col items-center justify-center text-text-muted gap-3">
                  <ProgressRing progress={overallProgress} size={60} />
                  <p className="text-sm">Waiting for training data…</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>

      {/* Log console */}
      {(status?.logs?.length ?? 0) > 0 && (
        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-text-primary flex items-center gap-1.5">
              <Terminal size={14} className="text-success" />
              Training Logs
            </h3>
          </CardHeader>
          <CardContent className="p-0">
            <div className="bg-slate-900 rounded-b-xl font-mono text-xs p-4 max-h-48 overflow-y-auto space-y-0.5">
              {status?.logs?.map((log, i) => (
                <div key={i} className="text-slate-300 leading-relaxed">
                  <span className="text-slate-500 select-none mr-2">[{String(i).padStart(3, '0')}]</span>
                  <span className={
                    log.includes('Error') || log.includes('FAILED') ? 'text-red-400' :
                    log.includes('Complete') || log.includes('SUCCESS') ? 'text-green-400' :
                    log.includes('Training') || log.includes('Epoch') ? 'text-blue-400' :
                    'text-slate-300'
                  }>
                    {log}
                  </span>
                </div>
              ))}
              <div ref={logEndRef} />
            </div>
          </CardContent>
        </Card>
      )}

      <ArchitectureDetailModal architecture={detailArch} onClose={() => setDetailArch(null)} />
    </div>
  )
}
