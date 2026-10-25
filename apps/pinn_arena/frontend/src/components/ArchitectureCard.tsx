import React from 'react'
import clsx from 'clsx'
import { motion } from 'framer-motion'
import { Check, Zap, ExternalLink, Search, Microscope } from 'lucide-react'
import type { ArchitectureInfo, ArchitectureFamily } from '../types'
import { Badge } from './ui/Badge'
import type { BadgeVariant } from './ui/Badge'

interface ArchitectureCardProps {
  architecture: ArchitectureInfo
  selected?: boolean
  onClick?: () => void
  /** Called when the info icon is clicked — opens detail modal */
  onInfo?: () => void
  /** Highlight as recommended for inverse problems */
  inverseRecommended?: boolean
}

export const FAMILY_CONFIG: Record<ArchitectureFamily, { label: string; variant: BadgeVariant; color: string }> = {
  PINN: { label: 'PINN', variant: 'info', color: '#3b82f6' },
  NEURAL_OPERATOR: { label: 'Neural Op.', variant: 'purple', color: '#a855f7' },
  TRANSFORMER: { label: 'Transformer', variant: 'cyan', color: '#06b6d4' },
  RECURRENT: { label: 'Recurrent', variant: 'orange', color: '#f97316' },
  CONVOLUTIONAL: { label: 'Conv.', variant: 'pink', color: '#ec4899' },
  CONTINUOUS: { label: 'Continuous', variant: 'success', color: '#059669' },
  GRAPH: { label: 'Graph', variant: 'yellow', color: '#d97706' },
  AUTOENCODER: { label: 'AutoEnc.', variant: 'warning', color: '#d97706' },
  RESERVOIR: { label: 'Reservoir', variant: 'danger', color: '#dc2626' },
  ROM: { label: 'ROM', variant: 'default', color: '#64748b' },
  CLASSICAL: { label: 'Classical', variant: 'default', color: '#64748b' },
  PHYSICS_AWARE: { label: 'Physics-Aware', variant: 'success', color: '#059669' },
}

export function ArchitectureCard({ architecture, selected, onClick, onInfo, inverseRecommended }: ArchitectureCardProps) {
  const family = FAMILY_CONFIG[architecture.family]

  return (
    <motion.div
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.98 }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      onClick={onClick}
      className={clsx(
        'bg-white border rounded-xl shadow-card p-4 cursor-pointer transition-all duration-200 group relative',
        selected
          ? 'border-accent shadow-glow-blue ring-1 ring-accent/20'
          : inverseRecommended
            ? 'border-purple-400 ring-1 ring-purple-200'
            : 'border-border hover:border-border-strong hover:shadow-card-hover'
      )}
    >
      {/* Top-right controls: Analyze + Recommended + Checkmark stacked */}
      <div className="absolute top-2.5 right-2.5 flex flex-col items-end gap-1.5">
        {onInfo && (
          <button
            onClick={(e) => { e.stopPropagation(); onInfo() }}
            className="flex items-center gap-1 text-[10px] font-semibold text-accent bg-accent/8 border border-accent/20 hover:bg-accent/15 hover:border-accent/40 px-2 py-1 rounded-md transition-colors"
          >
            <Microscope size={10} />
            Analyze
          </button>
        )}
        {inverseRecommended && !selected && (
          <div className="text-[9px] font-bold bg-purple-100 text-purple-700 px-1.5 py-0.5 rounded-full flex items-center gap-0.5">
            <Search size={8} /> Recommended
          </div>
        )}
        {selected && (
          <div className="w-5 h-5 rounded-full bg-accent flex items-center justify-center">
            <Check size={11} strokeWidth={3} className="text-white" />
          </div>
        )}
      </div>

      {/* Header */}
      <div className="mb-3 pr-20">
        <div className="flex items-center gap-2 flex-wrap mb-1.5">
          <Badge
            variant={family.variant}
            size="xs"
            style={{ color: family.color, borderColor: `${family.color}30`, backgroundColor: `${family.color}10` } as React.CSSProperties}
          >
            {family.label}
          </Badge>
          {architecture.supports_physics_loss && (
            <Badge variant="success" size="xs">
              <Zap size={9} className="mr-0.5" />
              Physics Loss
            </Badge>
          )}
          {architecture.supports_3d && (
            <Badge variant="default" size="xs">3D</Badge>
          )}
          {architecture.inverse_only && (
            <Badge variant="purple" size="xs">
              <Search size={9} className="mr-0.5" />
              Inverse
            </Badge>
          )}
        </div>
        <h3 className="text-sm font-semibold text-text-primary group-hover:text-accent transition-colors leading-tight">
          {architecture.name}
        </h3>
      </div>

      {/* Description */}
      <p className="text-xs text-text-secondary leading-relaxed mb-3 line-clamp-2">
        {architecture.description}
      </p>

      {/* Best for tags */}
      {architecture.best_for.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {architecture.best_for.slice(0, 3).map((tag) => (
            <span
              key={tag}
              className="text-[10px] text-slate-500 bg-slate-100 border border-slate-200 px-1.5 py-0.5 rounded-full"
            >
              {tag}
            </span>
          ))}
          {architecture.best_for.length > 3 && (
            <span className="text-[10px] text-slate-400">+{architecture.best_for.length - 3}</span>
          )}
        </div>
      )}

      {/* Footer: paper ref */}
      {architecture.paper_reference && (
        <div className="mt-2 pt-2 border-t border-border">
          <div className="flex items-center gap-1 text-[10px] text-slate-400 min-w-0">
            <ExternalLink size={9} className="flex-shrink-0" />
            <span className="truncate">{architecture.paper_reference}</span>
          </div>
        </div>
      )}
    </motion.div>
  )
}
