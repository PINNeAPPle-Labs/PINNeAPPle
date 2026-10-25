import React from 'react'
import clsx from 'clsx'

interface ProgressBarProps {
  value: number
  max?: number
  label?: string
  showPercentage?: boolean
  variant?: 'default' | 'success' | 'warning' | 'danger'
  size?: 'sm' | 'md' | 'lg'
  animated?: boolean
  className?: string
}

const variantColors: Record<string, string> = {
  default: 'bg-accent',
  success: 'bg-success',
  warning: 'bg-warning',
  danger: 'bg-danger',
}

const sizeHeights: Record<string, string> = {
  sm: 'h-1',
  md: 'h-2',
  lg: 'h-3',
}

export function ProgressBar({
  value,
  max = 100,
  label,
  showPercentage = false,
  variant = 'default',
  size = 'md',
  animated = false,
  className,
}: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))

  return (
    <div className={clsx('space-y-1.5', className)}>
      {(label || showPercentage) && (
        <div className="flex items-center justify-between text-xs">
          {label && <span className="text-text-secondary">{label}</span>}
          {showPercentage && (
            <span className="text-text-muted font-mono">{Math.round(pct)}%</span>
          )}
        </div>
      )}
      <div className={clsx('w-full bg-slate-200 rounded-full overflow-hidden', sizeHeights[size])}>
        <div
          className={clsx(
            'h-full rounded-full transition-all duration-500 ease-out',
            variantColors[variant],
            animated && 'animate-pulse-slow'
          )}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
