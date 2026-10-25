import React from 'react'
import clsx from 'clsx'

interface FilterChipProps {
  label: string
  active?: boolean
  onClick?: () => void
  icon?: React.ReactNode
  count?: number
  color?: string
  size?: 'sm' | 'md' | 'xs'
}

export function FilterChip({ label, active, onClick, icon, count, color, size }: FilterChipProps) {
  const sizeClass = size === 'xs'
    ? 'px-2 py-0.5 text-[11px] gap-1'
    : size === 'sm'
    ? 'px-2.5 py-1 text-xs gap-1'
    : 'px-3 py-1.5 text-sm gap-1.5'

  return (
    <button
      onClick={onClick}
      className={clsx(
        'inline-flex items-center rounded-full font-medium border transition-all duration-200 whitespace-nowrap',
        sizeClass,
        active
          ? 'bg-accent text-white border-accent shadow-sm'
          : 'bg-white text-slate-600 border-slate-200 hover:border-slate-300 hover:text-text-primary'
      )}
      style={active && color ? { backgroundColor: color, borderColor: color, color: 'white' } : undefined}
    >
      {icon && <span className="text-base leading-none">{icon}</span>}
      {label}
      {count !== undefined && (
        <span
          className={clsx(
            'rounded-full px-1 min-w-[16px] text-center',
            size === 'xs' ? 'text-[9px]' : 'text-xs px-1.5 py-0.5 min-w-[20px]',
            active ? 'bg-white/25' : 'bg-slate-100 text-slate-500'
          )}
        >
          {count}
        </span>
      )}
    </button>
  )
}
