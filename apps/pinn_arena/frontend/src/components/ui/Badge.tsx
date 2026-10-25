import React from 'react'
import clsx from 'clsx'

export type BadgeVariant = 'default' | 'success' | 'warning' | 'danger' | 'info' | 'accent' | 'purple' | 'orange' | 'pink' | 'cyan' | 'yellow'
export type BadgeSize = 'xs' | 'sm' | 'md'

interface BadgeProps {
  variant?: BadgeVariant
  size?: BadgeSize
  children: React.ReactNode
  className?: string
  style?: React.CSSProperties
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-slate-100 text-slate-700 border-slate-200',
  success: 'bg-green-50 text-green-700 border-green-200',
  warning: 'bg-amber-50 text-amber-700 border-amber-200',
  danger: 'bg-red-50 text-red-700 border-red-200',
  info: 'bg-blue-50 text-blue-700 border-blue-200',
  accent: 'bg-blue-50 text-blue-700 border-blue-200',
  purple: 'bg-purple-50 text-purple-700 border-purple-200',
  orange: 'bg-orange-50 text-orange-700 border-orange-200',
  pink: 'bg-pink-50 text-pink-700 border-pink-200',
  cyan: 'bg-cyan-50 text-cyan-700 border-cyan-200',
  yellow: 'bg-yellow-50 text-yellow-700 border-yellow-200',
}

const sizeStyles: Record<BadgeSize, string> = {
  xs: 'text-[10px] px-1.5 py-0.5',
  sm: 'text-xs px-2 py-0.5',
  md: 'text-sm px-2.5 py-1',
}

export function Badge({ variant = 'default', size = 'sm', children, className, style }: BadgeProps) {
  return (
    <span
      className={clsx(
        'inline-flex items-center rounded-full border font-medium',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
      style={style}
    >
      {children}
    </span>
  )
}
