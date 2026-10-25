import React from 'react'
import clsx from 'clsx'

interface MetricBadgeProps {
  label: string
  value: number | string
  unit?: string
  goodDirection?: 'low' | 'high'
  thresholds?: { good: number; bad: number }
  className?: string
  size?: 'sm' | 'md'
}

export function MetricBadge({
  label,
  value,
  unit,
  goodDirection = 'low',
  thresholds,
  className,
  size = 'sm',
}: MetricBadgeProps) {
  let colorClass = 'text-text-secondary'

  if (thresholds && typeof value === 'number') {
    if (goodDirection === 'low') {
      if (value <= thresholds.good) colorClass = 'text-success'
      else if (value >= thresholds.bad) colorClass = 'text-danger'
      else colorClass = 'text-warning'
    } else {
      if (value >= thresholds.good) colorClass = 'text-success'
      else if (value <= thresholds.bad) colorClass = 'text-danger'
      else colorClass = 'text-warning'
    }
  }

  return (
    <div className={clsx('flex flex-col gap-0.5', className)}>
      <span className="text-xs text-text-muted font-medium uppercase tracking-wider">{label}</span>
      <span className={clsx('font-mono font-semibold', colorClass, size === 'sm' ? 'text-sm' : 'text-base')}>
        {typeof value === 'number' ? value.toExponential(3) : value}
        {unit && <span className="text-text-muted text-xs ml-1">{unit}</span>}
      </span>
    </div>
  )
}
