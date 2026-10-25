import React from 'react'
import clsx from 'clsx'

interface LoadingSpinnerProps {
  size?: 'sm' | 'md' | 'lg' | 'xl'
  className?: string
  label?: string
}

const sizeMap = {
  sm: 16,
  md: 24,
  lg: 36,
  xl: 48,
}

export function LoadingSpinner({ size = 'md', className, label }: LoadingSpinnerProps) {
  const px = sizeMap[size]

  return (
    <div className={clsx('flex flex-col items-center justify-center gap-3', className)}>
      <svg
        width={px}
        height={px}
        viewBox="0 0 24 24"
        fill="none"
        className="animate-spin"
      >
        <circle
          cx="12"
          cy="12"
          r="10"
          stroke="#e2e8f0"
          strokeWidth="3"
        />
        <path
          d="M12 2a10 10 0 0 1 10 10"
          stroke="#2563eb"
          strokeWidth="3"
          strokeLinecap="round"
        />
      </svg>
      {label && <p className="text-sm text-text-secondary">{label}</p>}
    </div>
  )
}

interface PageLoadingProps {
  label?: string
}

export function PageLoading({ label = 'Loading...' }: PageLoadingProps) {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <LoadingSpinner size="xl" label={label} />
    </div>
  )
}
