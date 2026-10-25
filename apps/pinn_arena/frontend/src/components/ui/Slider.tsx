import React, { useState, useRef } from 'react'
import clsx from 'clsx'

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  onChange: (value: number) => void
  unit?: string
  description?: string
  className?: string
  formatValue?: (v: number) => string
}

export function Slider({
  label,
  value,
  min,
  max,
  step = 0.01,
  onChange,
  unit,
  description,
  className,
  formatValue,
}: SliderProps) {
  const pct = Math.max(0, Math.min(100, ((value - min) / (max - min)) * 100))
  const [draft, setDraft] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const commit = (raw: string) => {
    const parsed = parseFloat(raw)
    if (isFinite(parsed)) {
      const stepped = step >= 1 ? Math.round(parsed / step) * step : parsed
      onChange(Math.max(min, Math.min(max, stepped)))
    }
    setDraft(null)
  }

  return (
    <div className={clsx('space-y-2', className)}>
      <div className="flex items-center justify-between gap-3">
        <label className="text-sm font-medium text-text-primary leading-none">{label}</label>

        <div className="flex items-center gap-1">
          <input
            ref={inputRef}
            type="number"
            min={min}
            max={max}
            step={step}
            value={draft !== null ? draft : value}
            onFocus={() => setDraft(String(value))}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={(e) => commit(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { commit((e.target as HTMLInputElement).value); inputRef.current?.blur() }
              if (e.key === 'Escape') { setDraft(null); inputRef.current?.blur() }
            }}
            title="Click to type a value directly"
            className={clsx(
              'font-mono text-sm text-accent text-right bg-transparent',
              'border-b border-transparent hover:border-accent/40 focus:border-accent focus:outline-none',
              'transition-colors w-24 px-0.5 [appearance:textfield]',
              '[&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none'
            )}
          />
          {unit && <span className="text-text-muted text-xs flex-shrink-0">{unit}</span>}
        </div>
      </div>

      {description && <p className="text-xs text-text-muted">{description}</p>}

      <div className="relative">
        <input
          type="range"
          min={min}
          max={max}
          step={step}
          value={value}
          onChange={(e) => onChange(parseFloat(e.target.value))}
          className="w-full h-1.5 appearance-none rounded-full cursor-pointer"
          style={{ background: `linear-gradient(to right, #2563eb ${pct}%, #e2e8f0 ${pct}%)` }}
        />
      </div>

      <div className="flex justify-between text-xs text-text-muted font-mono">
        <span>{formatValue ? formatValue(min) : min}{unit}</span>
        <span>{formatValue ? formatValue(max) : max}{unit}</span>
      </div>
    </div>
  )
}
