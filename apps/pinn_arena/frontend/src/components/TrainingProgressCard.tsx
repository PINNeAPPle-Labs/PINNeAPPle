import React from 'react'
import clsx from 'clsx'
import { CheckCircle, XCircle, Clock, Loader2, Info } from 'lucide-react'
import { Badge } from './ui/Badge'
import { ProgressBar } from './ui/ProgressBar'
import { FAMILY_CONFIG } from './ArchitectureCard'
import type { ArchitectureFamily, TrainingMetrics } from '../types'

type ModelStatus = 'waiting' | 'training' | 'evaluating' | 'inverting' | 'completed' | 'failed'

interface TrainingProgressCardProps {
  architectureId: string
  architectureName: string
  architectureFamily?: ArchitectureFamily
  status: ModelStatus
  progress: number
  currentLoss?: number
  trainingHistory?: TrainingMetrics[]
  finalLoss?: number
  converged?: boolean
  onDetails?: () => void
}

export function TrainingProgressCard({
  architectureName,
  architectureFamily,
  status,
  progress,
  currentLoss,
  finalLoss,
  converged,
  onDetails,
}: TrainingProgressCardProps) {
  const family = architectureFamily ? FAMILY_CONFIG[architectureFamily] : null

  const statusConfig: Record<ModelStatus, { icon: React.ReactNode; label: string; color: string }> = {
    waiting: {
      icon: <Clock size={14} className="text-slate-400" />,
      label: 'Waiting',
      color: 'text-slate-400',
    },
    training: {
      icon: <Loader2 size={14} className="text-accent animate-spin" />,
      label: 'Training',
      color: 'text-accent',
    },
    evaluating: {
      icon: <Loader2 size={14} className="text-violet-500 animate-spin" />,
      label: 'Evaluating',
      color: 'text-violet-500',
    },
    inverting: {
      icon: <Loader2 size={14} className="text-amber-500 animate-spin" />,
      label: 'Inverse solver',
      color: 'text-amber-500',
    },
    completed: {
      icon: <CheckCircle size={14} className="text-success" />,
      label: 'Complete',
      color: 'text-success',
    },
    failed: {
      icon: <XCircle size={14} className="text-danger" />,
      label: 'Failed',
      color: 'text-danger',
    },
  }

  // Fallback for any unexpected status string coming from the backend
  const sc = statusConfig[status] ?? statusConfig.training

  return (
    <div
      className={clsx(
        'bg-white border rounded-xl shadow-card p-4 transition-all duration-300',
        (status === 'training' || status === 'evaluating' || status === 'inverting') && 'border-accent/50 shadow-glow-blue',
        status === 'completed' && 'border-green-200',
        status === 'failed' && 'border-red-200',
        status === 'waiting' && 'opacity-60'
      )}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2 min-w-0">
          {family && (
            <Badge
              variant={family.variant}
              size="xs"
              style={{ color: family.color, borderColor: `${family.color}30`, backgroundColor: `${family.color}10` } as React.CSSProperties}
            >
              {family.label}
            </Badge>
          )}
          <h4 className="text-sm font-semibold text-text-primary truncate">{architectureName}</h4>
          {onDetails && (
            <button
              onClick={onDetails}
              title="View architecture details"
              className="flex-shrink-0 p-0.5 rounded text-text-muted hover:text-accent hover:bg-accent/10 transition-colors"
            >
              <Info size={13} />
            </button>
          )}
        </div>
        <div className={clsx('flex items-center gap-1.5 flex-shrink-0', sc.color)}>
          {sc.icon}
          <span className="text-xs font-medium">{sc.label}</span>
        </div>
      </div>

      {(status === 'training' || status === 'evaluating' || status === 'inverting') && (
        <>
          <ProgressBar
            value={progress}
            max={100}
            variant="default"
            size="sm"
            showPercentage
            animated
            className="mb-2"
          />
          {currentLoss !== undefined && status === 'training' && (
            <div className="flex items-center justify-between text-xs">
              <span className="text-text-muted">Current Loss</span>
              <span className="text-accent font-mono">{currentLoss.toExponential(4)}</span>
            </div>
          )}
          {status === 'evaluating' && (
            <div className="text-xs text-text-muted">Computing metrics on held-out test set…</div>
          )}
          {status === 'inverting' && (
            <div className="text-xs text-amber-600">Running inverse parameter identification…</div>
          )}
        </>
      )}

      {status === 'waiting' && (
        <div className="h-1.5 w-full bg-slate-200 rounded-full">
          <div className="h-full w-0 bg-slate-300 rounded-full" />
        </div>
      )}

      {status === 'completed' && (
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {finalLoss !== undefined && (
              <div className="text-xs">
                <div className="text-text-muted">Final Loss</div>
                <div className="text-success font-mono font-semibold">{finalLoss.toExponential(4)}</div>
              </div>
            )}
          </div>
          {converged !== undefined && (
            <Badge variant={converged ? 'success' : 'warning'} size="xs">
              {converged ? 'Converged' : 'Not Converged'}
            </Badge>
          )}
        </div>
      )}

      {status === 'failed' && (
        <div className="text-xs text-red-500">Training failed. Check logs for details.</div>
      )}
    </div>
  )
}
