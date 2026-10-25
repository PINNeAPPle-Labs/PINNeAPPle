import React from 'react'
import clsx from 'clsx'
import { motion } from 'framer-motion'
import { ChevronRight, Star } from 'lucide-react'
import type { PhysicsProblem, PhysicsArea, Difficulty } from '../types'
import { Badge } from './ui/Badge'
import type { BadgeVariant } from './ui/Badge'

interface ProblemCardProps {
  problem: PhysicsProblem
  selected?: boolean
  compact?: boolean
  onClick?: () => void
}

export const DOMAIN_CONFIG: Record<PhysicsArea, { label: string; icon: string; color: string; variant: BadgeVariant }> = {
  FLUID_MECHANICS: { label: 'Fluid Mechanics', icon: '🌊', color: '#3b82f6', variant: 'info' },
  SOLID_MECHANICS: { label: 'Solid Mechanics', icon: '⚙️', color: '#f97316', variant: 'orange' },
  ELECTROMAGNETISM: { label: 'Electromagnetism', icon: '⚡', color: '#a855f7', variant: 'purple' },
  QUANTUM_MECHANICS: { label: 'Quantum Mechanics', icon: '⚛️', color: '#ec4899', variant: 'pink' },
  THERMODYNAMICS: { label: 'Thermodynamics', icon: '🌡️', color: '#ef4444', variant: 'danger' },
  WAVE_PHYSICS: { label: 'Wave Physics', icon: '〰️', color: '#06b6d4', variant: 'cyan' },
  NONLINEAR_DYNAMICS: { label: 'Nonlinear Dynamics', icon: '🌀', color: '#eab308', variant: 'yellow' },
}

export const DIMENSION_LABELS: Record<string, string> = {
  ONE_D: '1D',
  TWO_D: '2D',
  THREE_D: '3D',
}

export const DIFFICULTY_CONFIG: Record<Difficulty, { label: string; stars: number; variant: BadgeVariant }> = {
  BEGINNER: { label: 'Beginner', stars: 1, variant: 'success' },
  INTERMEDIATE: { label: 'Intermediate', stars: 2, variant: 'warning' },
  ADVANCED: { label: 'Advanced', stars: 3, variant: 'danger' },
}

export function ProblemCard({ problem, selected, compact, onClick }: ProblemCardProps) {
  const domain = DOMAIN_CONFIG[problem.domain]
  const dim = DIMENSION_LABELS[problem.dimension]
  const diff = DIFFICULTY_CONFIG[problem.difficulty]

  return (
    <motion.div
      whileHover={{ y: -2, scale: 1.01 }}
      whileTap={{ scale: 0.99 }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      onClick={onClick}
      className={clsx(
        'bg-white border rounded-xl shadow-card cursor-pointer transition-all duration-200 group',
        selected
          ? 'border-accent shadow-glow-blue ring-1 ring-accent/20'
          : 'border-border hover:border-border-strong hover:shadow-card-hover',
        compact ? 'p-3' : 'p-5'
      )}
    >
      {/* Header */}
      <div className={clsx('flex items-start justify-between', compact ? 'mb-2' : 'mb-3')}>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap mb-1.5">
            <span className="text-base">{domain.icon}</span>
            <Badge
              variant={domain.variant}
              size={compact ? 'xs' : 'sm'}
              style={{ color: domain.color, borderColor: `${domain.color}30`, backgroundColor: `${domain.color}10` } as React.CSSProperties}
            >
              {compact ? problem.domain.split('_').map(w => w[0]).join('') : domain.label}
            </Badge>
            <Badge variant="default" size={compact ? 'xs' : 'sm'}>
              {dim}
            </Badge>
            {problem.requires_geometry && (
              <Badge variant="info" size={compact ? 'xs' : 'sm'}>
                3D Geometry
              </Badge>
            )}
          </div>
          <h3
            className={clsx(
              'font-semibold text-text-primary group-hover:text-accent transition-colors truncate',
              compact ? 'text-sm' : 'text-base'
            )}
          >
            {problem.name}
          </h3>
        </div>
        {selected && (
          <div className="flex-shrink-0 w-5 h-5 rounded-full bg-accent flex items-center justify-center ml-2">
            <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
              <path d="M1 4L3.5 6.5L9 1" stroke="white" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
            </svg>
          </div>
        )}
      </div>

      {/* Description */}
      {!compact && (
        <p className="text-sm text-text-secondary mb-3 line-clamp-2 leading-relaxed">
          {problem.description}
        </p>
      )}

      {/* First equation */}
      {problem.equations.length > 0 && (
        <div className={clsx('font-mono text-xs bg-slate-50 border border-slate-200 rounded-lg overflow-hidden', compact ? 'px-2 py-1.5 mb-2' : 'px-3 py-2 mb-4')}>
          <span className="text-slate-400 select-none mr-1">∂ </span>
          <span className="text-accent truncate">{problem.equations[0]}</span>
        </div>
      )}

      {/* Footer */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 flex-wrap">
          {/* Difficulty stars */}
          <div className="flex items-center gap-1">
            {Array.from({ length: 3 }).map((_, i) => (
              <Star
                key={i}
                size={compact ? 10 : 12}
                className={i < diff.stars ? 'text-amber-500 fill-amber-500' : 'text-slate-200'}
              />
            ))}
          </div>
          {!compact && problem.tags.slice(0, 2).map((tag) => (
            <span key={tag} className="text-xs text-slate-500 bg-slate-100 px-2 py-0.5 rounded-full border border-slate-200">
              {tag}
            </span>
          ))}
        </div>
        {!compact && onClick && (
          <ChevronRight size={16} className="text-slate-400 group-hover:text-accent transition-colors flex-shrink-0" />
        )}
      </div>
    </motion.div>
  )
}
