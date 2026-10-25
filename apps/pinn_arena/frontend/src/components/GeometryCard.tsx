import React from 'react'
import clsx from 'clsx'
import { motion } from 'framer-motion'
import { Check } from 'lucide-react'
import type { GeometryPreset } from '../types'
import { Badge } from './ui/Badge'

interface GeometryCardProps {
  geometry: GeometryPreset
  selected?: boolean
  onClick?: () => void
}

const GEOMETRY_SVGS: Record<string, React.ReactNode> = {
  sphere: (
    <svg viewBox="0 0 80 80" className="w-full h-full" fill="none">
      <ellipse cx="40" cy="40" rx="30" ry="30" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb18" />
      <ellipse cx="40" cy="40" rx="30" ry="10" stroke="#2563eb40" strokeWidth="1" strokeDasharray="3 3" />
      <ellipse cx="40" cy="40" rx="10" ry="30" stroke="#2563eb40" strokeWidth="1" strokeDasharray="3 3" />
    </svg>
  ),
  cylinder: (
    <svg viewBox="0 0 80 80" className="w-full h-full" fill="none">
      <rect x="15" y="25" width="50" height="40" rx="2" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb18" />
      <ellipse cx="40" cy="25" rx="25" ry="8" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb28" />
      <ellipse cx="40" cy="65" rx="25" ry="8" stroke="#2563eb40" strokeWidth="1" />
    </svg>
  ),
  cube: (
    <svg viewBox="0 0 80 80" className="w-full h-full" fill="none">
      <rect x="20" y="30" width="40" height="35" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb18" />
      <path d="M20 30 L35 18 L75 18 L75 53 L60 65" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb12" />
      <path d="M60 65 L60 30 L20 30" stroke="#2563eb40" strokeWidth="1" />
      <path d="M60 30 L75 18" stroke="#2563eb" strokeWidth="1.5" />
    </svg>
  ),
  pipe: (
    <svg viewBox="0 0 80 80" className="w-full h-full" fill="none">
      <ellipse cx="20" cy="40" rx="10" ry="20" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb18" />
      <ellipse cx="60" cy="40" rx="10" ry="20" stroke="#2563eb40" strokeWidth="1" />
      <line x1="20" y1="20" x2="60" y2="20" stroke="#2563eb" strokeWidth="1.5" />
      <line x1="20" y1="60" x2="60" y2="60" stroke="#2563eb" strokeWidth="1.5" />
      <ellipse cx="20" cy="40" rx="5" ry="12" stroke="#2563eb40" strokeWidth="1" strokeDasharray="2 2" />
      <ellipse cx="60" cy="40" rx="5" ry="12" stroke="#2563eb40" strokeWidth="1" strokeDasharray="2 2" />
    </svg>
  ),
  torus: (
    <svg viewBox="0 0 80 80" className="w-full h-full" fill="none">
      <ellipse cx="40" cy="40" rx="28" ry="12" stroke="#2563eb" strokeWidth="1.5" fill="none" />
      <ellipse cx="40" cy="40" rx="14" ry="6" stroke="#2563eb40" strokeWidth="1" strokeDasharray="3 3" />
      <circle cx="40" cy="28" r="7" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb18" />
      <circle cx="40" cy="52" r="7" stroke="#2563eb40" strokeWidth="1" fill="#2563eb10" />
    </svg>
  ),
  channel: (
    <svg viewBox="0 0 80 80" className="w-full h-full" fill="none">
      <rect x="10" y="25" width="60" height="30" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb18" rx="2" />
      <line x1="10" y1="40" x2="70" y2="40" stroke="#2563eb40" strokeWidth="1" strokeDasharray="4 3" />
      <path d="M15 32 Q25 28 35 32 Q45 36 55 32 Q65 28 70 32" stroke="#059669" strokeWidth="1" fill="none" />
      <path d="M15 44 Q25 40 35 44 Q45 48 55 44 Q65 40 70 44" stroke="#059669" strokeWidth="1" fill="none" />
    </svg>
  ),
}

function getGeometrySvg(geometryType: string): React.ReactNode {
  const key = geometryType.toLowerCase()
  for (const [k, v] of Object.entries(GEOMETRY_SVGS)) {
    if (key.includes(k)) return v
  }
  return (
    <svg viewBox="0 0 80 80" className="w-full h-full" fill="none">
      <rect x="15" y="15" width="50" height="50" stroke="#2563eb" strokeWidth="1.5" fill="#2563eb18" rx="4" />
      <line x1="15" y1="40" x2="65" y2="40" stroke="#2563eb40" strokeWidth="1" strokeDasharray="4 3" />
      <line x1="40" y1="15" x2="40" y2="65" stroke="#2563eb40" strokeWidth="1" strokeDasharray="4 3" />
    </svg>
  )
}

export function GeometryCard({ geometry, selected, onClick }: GeometryCardProps) {
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
          : 'border-border hover:border-border-strong hover:shadow-card-hover'
      )}
    >
      {selected && (
        <div className="absolute top-3 right-3 w-5 h-5 rounded-full bg-accent flex items-center justify-center z-10">
          <Check size={11} strokeWidth={3} className="text-white" />
        </div>
      )}

      {/* Geometry visualization */}
      <div className="w-full h-24 mb-3 flex items-center justify-center bg-slate-50 rounded-lg border border-slate-200 overflow-hidden">
        <div className="w-20 h-20">
          {getGeometrySvg(geometry.geometry_type)}
        </div>
      </div>

      {/* Info */}
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Badge variant="default" size="xs">{geometry.geometry_type}</Badge>
        </div>
        <h3 className="text-sm font-semibold text-text-primary group-hover:text-accent transition-colors mb-1">
          {geometry.name}
        </h3>
        <p className="text-xs text-text-secondary leading-relaxed line-clamp-2">
          {geometry.description}
        </p>
      </div>

      {/* Dimensions */}
      {Object.keys(geometry.dimensions).length > 0 && (
        <div className="mt-3 pt-2 border-t border-border grid grid-cols-2 gap-1">
          {Object.entries(geometry.dimensions).slice(0, 4).map(([key, val]) => (
            <div key={key} className="flex items-center justify-between text-xs">
              <span className="text-slate-400 capitalize">{key.replace(/_/g, ' ')}</span>
              <span className="text-text-secondary font-mono">{val}</span>
            </div>
          ))}
        </div>
      )}

      <div className="mt-2 text-xs text-slate-400">
        Mesh res: <span className="text-accent font-mono">{geometry.mesh_resolution_default}</span>
      </div>
    </motion.div>
  )
}
