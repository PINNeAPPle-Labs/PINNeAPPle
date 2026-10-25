import React, { useState, useRef, useMemo, useEffect, useCallback, useLayoutEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import { motion, AnimatePresence } from 'framer-motion'
import {
  ChevronLeft, ChevronRight, Rocket, Filter, Zap, CheckCircle,
  Settings, Database, Layers, Play, BarChart3, Cpu, Star, ExternalLink,
  RefreshCw, Upload, X, ArrowRight, FlaskConical, Search, GitMerge,
  AlertCircle,
} from 'lucide-react'
import { problemsApi, geometriesApi, architecturesApi, experimentsApi, datasetsApi } from '../lib/api'
import type { PhysicsProblem, ArchitectureFamily, ArchitectureInfo, UploadedDataset, ColumnMapping, InverseConfig, ProblemType, ColumnHints } from '../types'
import ArchitectureDetailModal from '../components/ArchitectureDetailModal'
import { useExperimentStore, WIZARD_STEPS, STEP_LABELS, type WizardStep } from '../store/experimentStore'
import { ProblemCard, DOMAIN_CONFIG, DIMENSION_LABELS, DIFFICULTY_CONFIG } from '../components/ProblemCard'
import { ArchitectureCard, FAMILY_CONFIG } from '../components/ArchitectureCard'
import { GeometryCard } from '../components/GeometryCard'
import { StepIndicator } from '../components/ui/StepIndicator'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { Card, CardContent, CardHeader } from '../components/ui/Card'
import { Slider } from '../components/ui/Slider'
import { SearchInput } from '../components/ui/SearchInput'
import { FilterChip } from '../components/ui/FilterChip'
import { PageLoading } from '../components/ui/LoadingSpinner'
import { ProgressBar } from '../components/ui/ProgressBar'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  Legend,
} from 'recharts'
import { ScientificHeatmap, buildHeatmapFromGrid, buildHeatmapFromScatter } from '../components/ScientificHeatmap'
import { MathEq } from '../components/MathEq'
import type { ColormapName } from '../components/ScientificHeatmap'
import { Dataset3DView } from '../components/Dataset3DView'
import { SurfaceMeshView } from '../components/SurfaceMeshView'
import type { SurfaceMeshData } from '../components/SurfaceMeshView'

// ─── Step 0: Problem Type ────────────────────────────────────────────────────

function StepProblemType() {
  const { problemType, setProblemType } = useExperimentStore()

  type CardDef = {
    type: ProblemType
    label: string
    tagline: string
    description: string
    icon: React.ReactNode
    badge: string
    badgeColor: string
    features: string[]
    recommended: boolean
  }

  const cards: CardDef[] = [
    {
      type: 'forward',
      label: 'Forward Problem',
      tagline: 'Given parameters → predict the solution',
      description:
        'Train neural networks to approximate the solution of a known PDE with fixed parameters. '
        + 'The model learns to map coordinates to field values (e.g. velocity, temperature, pressure).',
      icon: <ArrowRight size={28} />,
      badge: 'Standard',
      badgeColor: 'bg-blue-100 text-blue-700',
      features: [
        'All 93 architectures available',
        'Physics-informed loss (PDE residual)',
        'Benchmarking & comparison',
        'Optional dataset upload for data-driven mode',
      ],
      recommended: true,
    },
    {
      type: 'inverse',
      label: 'Inverse Problem',
      tagline: 'Given observations → recover parameters',
      description:
        'Use sensor measurements to identify unknown PDE parameters (e.g. viscosity, diffusivity). '
        + 'PINNeAPPle\'s inverse infrastructure (EKI, L-BFGS, Adam) identifies the governing parameters '
        + 'while enforcing physics consistency.',
      icon: <Search size={28} />,
      badge: 'Inverse',
      badgeColor: 'bg-purple-100 text-purple-700',
      features: [
        'InversePINN + InverseProblemSolver',
        'Adam / L-BFGS / EKI / TEKI solvers',
        'Gaussian / Huber / Cauchy noise models',
        'Upload experimental observations',
        'Tikhonov regularisation',
      ],
      recommended: false,
    },
  ]

  return (
    <div className="max-w-3xl mx-auto">
      <div className="mb-6 text-center">
        <p className="text-sm text-text-secondary">
          Select whether you want to solve a <strong>forward problem</strong> (predict the solution given known physics)
          or an <strong>inverse problem</strong> (identify unknown physical parameters from measurements).
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
        {cards.map((card) => {
          const isSelected = problemType === card.type
          return (
            <motion.button
              key={card.type}
              onClick={() => setProblemType(card.type)}
              className={[
                'relative text-left rounded-2xl border-2 p-6 transition-all',
                'hover:shadow-lg focus:outline-none',
                isSelected
                  ? 'border-accent bg-accent/5 shadow-md'
                  : 'border-border bg-white hover:border-accent/40',
              ].join(' ')}
              whileHover={{ y: -2 }}
              whileTap={{ scale: 0.98 }}
            >
              {/* Check mark */}
              {isSelected && (
                <div className="absolute top-4 right-4 w-6 h-6 rounded-full bg-accent flex items-center justify-center">
                  <CheckCircle size={14} className="text-white" />
                </div>
              )}

              {/* Icon + badge */}
              <div className="flex items-start gap-3 mb-4">
                <div className={[
                  'w-12 h-12 rounded-xl flex items-center justify-center',
                  isSelected ? 'bg-accent text-white' : 'bg-slate-100 text-slate-600',
                ].join(' ')}>
                  {card.icon}
                </div>
                <div>
                  <span className={`inline-block text-xs font-semibold px-2 py-0.5 rounded-full ${card.badgeColor}`}>
                    {card.badge}
                  </span>
                  {card.recommended && (
                    <span className="ml-1.5 inline-block text-xs font-semibold px-2 py-0.5 rounded-full bg-green-100 text-green-700">
                      Recommended
                    </span>
                  )}
                </div>
              </div>

              <h3 className="text-base font-bold text-text-primary mb-1">{card.label}</h3>
              <p className="text-xs font-medium text-accent mb-2">{card.tagline}</p>
              <p className="text-xs text-text-secondary leading-relaxed mb-4">{card.description}</p>

              <ul className="space-y-1">
                {card.features.map((f) => (
                  <li key={f} className="flex items-start gap-1.5 text-xs text-text-secondary">
                    <span className="text-green-500 mt-0.5">✓</span>
                    {f}
                  </li>
                ))}
              </ul>
            </motion.button>
          )
        })}
      </div>
    </div>
  )
}

// ─── Step 1: Problem ─────────────────────────────────────────────────────────

function ProblemDetailCompact({ problem }: { problem: PhysicsProblem }) {
  const domain = DOMAIN_CONFIG[problem.domain]
  const diff = DIFFICULTY_CONFIG[problem.difficulty]

  return (
    <motion.div
      initial={{ opacity: 0, x: 20 }}
      animate={{ opacity: 1, x: 0 }}
      className="bg-white border border-border rounded-xl shadow-card overflow-y-auto max-h-[520px]"
    >
      {/* Header */}
      <div className="p-4 border-b border-border bg-slate-50">
        <div className="flex items-center gap-2 flex-wrap mb-1.5">
          <span className="text-lg">{domain.icon}</span>
          <Badge
            variant={domain.variant}
            size="sm"
            style={{ color: domain.color, borderColor: `${domain.color}30`, backgroundColor: `${domain.color}10` } as React.CSSProperties}
          >
            {domain.label}
          </Badge>
          <Badge variant="default" size="sm">{DIMENSION_LABELS[problem.dimension]}</Badge>
          {problem.requires_geometry && <Badge variant="info" size="xs">3D Geometry</Badge>}
        </div>
        <h3 className="text-sm font-bold text-text-primary">{problem.name}</h3>
        <p className="text-xs text-text-secondary mt-1 leading-relaxed">{problem.description}</p>
        <div className="flex items-center gap-1.5 mt-2">
          {Array.from({ length: 3 }).map((_, i) => (
            <Star key={i} size={11} className={i < diff.stars ? 'text-amber-500 fill-amber-500' : 'text-slate-200'} />
          ))}
          <span className="text-xs text-text-muted ml-1">{diff.label}</span>
        </div>
      </div>

      <div className="p-4 space-y-4">
        {/* Equations */}
        {problem.equations.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 mb-2">
              <Layers size={11} className="text-text-muted" />
              <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">Governing Equations</span>
            </div>
            <div className="space-y-1.5">
              {(problem.latex_equations ?? problem.equations).map((eq, i) => (
                <div key={i} className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 overflow-x-auto">
                  <span className="text-[10px] text-slate-400 select-none flex-shrink-0 tabular-nums w-4 text-right self-start pt-0.5">{i + 1}</span>
                  {problem.latex_equations ? (
                    <MathEq latex={eq} display={false} className="text-sm min-w-0" />
                  ) : (
                    <span className="font-mono text-xs text-accent">{eq}</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* BCs and ICs */}
        {problem.boundary_conditions && (
          <div>
            <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1.5">Boundary Conditions</div>
            <div className="font-mono text-xs bg-green-50 border border-green-200 rounded-lg px-3 py-2 text-green-700 leading-relaxed">
              {problem.boundary_conditions}
            </div>
          </div>
        )}
        {problem.initial_conditions && (
          <div>
            <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1.5">Initial Conditions</div>
            <div className="font-mono text-xs bg-amber-50 border border-amber-200 rounded-lg px-3 py-2 text-amber-700 leading-relaxed">
              {problem.initial_conditions}
            </div>
          </div>
        )}

        {/* Applications */}
        {problem.applications.length > 0 && (
          <div>
            <div className="flex items-center gap-1.5 mb-1.5">
              <ExternalLink size={10} className="text-text-muted" />
              <span className="text-[10px] font-semibold text-text-muted uppercase tracking-wider">Applications</span>
            </div>
            <ul className="space-y-1">
              {problem.applications.slice(0, 4).map((app, i) => (
                <li key={i} className="text-xs text-text-secondary flex items-start gap-1.5">
                  <span className="text-accent mt-0.5 flex-shrink-0">•</span>
                  {app}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Parameters summary */}
        {problem.parameters.length > 0 && (
          <div>
            <div className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-1.5">Parameters</div>
            <div className="space-y-1">
              {problem.parameters.map((p) => (
                <div key={p.name} className="flex items-center justify-between text-xs py-1 border-b border-slate-100">
                  <span className="font-mono text-accent">{p.name}</span>
                  <span className="font-mono text-text-secondary">{p.default} <span className="text-text-muted">{p.unit}</span></span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </motion.div>
  )
}

function StepProblem() {
  const { selectedProblem, setSelectedProblem } = useExperimentStore()
  const [search, setSearch] = useState('')
  const [domainFilter, setDomainFilter] = useState<string | null>(null)

  const { data: problems, isLoading } = useQuery({
    queryKey: ['problems'],
    queryFn: () => problemsApi.list().then((r) => r.data),
  })

  const filtered = useMemo(() => {
    if (!problems) return []
    return problems.filter((p) => {
      if (domainFilter && p.domain !== domainFilter) return false
      if (search) {
        const q = search.toLowerCase()
        return p.name.toLowerCase().includes(q) || p.description.toLowerCase().includes(q)
      }
      return true
    })
  }, [problems, domainFilter, search])

  const domains = useMemo(() => {
    if (!problems) return []
    return [...new Set(problems.map((p) => p.domain))]
  }, [problems])

  if (isLoading) return <PageLoading label="Loading problems…" />

  return (
    <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
      {/* Problem selector (left 2/3) */}
      <div className="xl:col-span-2 space-y-4">
        <div className="flex gap-3 flex-wrap">
          <SearchInput value={search} onChange={setSearch} placeholder="Search problems…" className="flex-1 min-w-48" />
          <div className="flex gap-2 flex-wrap">
            <FilterChip label="All" active={domainFilter === null} onClick={() => setDomainFilter(null)} />
            {domains.map((d) => {
              const cfg = DOMAIN_CONFIG[d as keyof typeof DOMAIN_CONFIG]
              return (
                <FilterChip
                  key={d}
                  label={cfg?.icon ?? d}
                  active={domainFilter === d}
                  onClick={() => setDomainFilter(domainFilter === d ? null : d)}
                  color={cfg?.color}
                />
              )
            })}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 max-h-[500px] overflow-y-auto pr-1">
          {filtered.map((p) => (
            <ProblemCard
              key={p.id}
              problem={p}
              selected={selectedProblem?.id === p.id}
              onClick={() => setSelectedProblem(selectedProblem?.id === p.id ? null : p)}
            />
          ))}
        </div>

        {selectedProblem && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-blue-50 border border-blue-200 rounded-xl p-4"
          >
            <div className="flex items-center gap-3">
              <CheckCircle size={18} className="text-accent flex-shrink-0" />
              <div>
                <div className="text-sm font-semibold text-accent">Selected: {selectedProblem.name}</div>
                <div className="text-xs text-text-muted mt-0.5">
                  {DOMAIN_CONFIG[selectedProblem.domain]?.label} •{' '}
                  {DIMENSION_LABELS[selectedProblem.dimension]} •{' '}
                  {DIFFICULTY_CONFIG[selectedProblem.difficulty]?.label}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </div>

      {/* Detail panel (right 1/3) */}
      <div className="xl:col-span-1">
        <AnimatePresence mode="wait">
          {selectedProblem ? (
            <ProblemDetailCompact key={selectedProblem.id} problem={selectedProblem} />
          ) : (
            <motion.div
              key="empty"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="bg-slate-50 border border-dashed border-slate-300 rounded-xl p-8 flex flex-col items-center justify-center text-center min-h-[300px]"
            >
              <div className="text-3xl mb-3">🔬</div>
              <p className="text-sm text-text-muted font-medium">Select a problem</p>
              <p className="text-xs text-slate-400 mt-1">Details and equations will appear here</p>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}

// ─── Step 2: Parameters ───────────────────────────────────────────────────────

function StepParameters() {
  const { selectedProblem, customParameters, setCustomParameter } = useExperimentStore()

  if (!selectedProblem) {
    return <div className="text-text-muted text-center py-8">No problem selected. Go back to Step 1.</div>
  }

  if (selectedProblem.parameters.length === 0) {
    return (
      <div className="text-center py-8">
        <div className="text-text-muted mb-2">No configurable parameters for this problem.</div>
        <div className="text-sm text-text-secondary">You can proceed to the next step.</div>
      </div>
    )
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="space-y-6">
        {selectedProblem.parameters.map((param) => (
          <Card key={param.name}>
            <CardContent className="pt-4">
              <Slider
                label={param.name}
                value={customParameters[param.name] ?? param.default}
                min={param.min_val}
                max={param.max_val}
                step={(param.max_val - param.min_val) / 100}
                unit={param.unit}
                description={param.description}
                onChange={(v) => setCustomParameter(param.name, v)}
                formatValue={(v) => v.toFixed(4)}
              />
            </CardContent>
          </Card>
        ))}
      </div>

      <div className="space-y-4">
        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-text-primary">Current Parameters</h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {selectedProblem.parameters.map((param) => (
                <div key={param.name} className="flex items-center justify-between text-sm border-b border-slate-100 pb-2">
                  <span className="font-mono text-accent text-xs">{param.name}</span>
                  <div className="text-right">
                    <span className="font-mono text-text-primary">
                      {(customParameters[param.name] ?? param.default).toFixed(4)}
                    </span>
                    <span className="text-text-muted text-xs ml-1">{param.unit}</span>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {selectedProblem.equations.length > 0 && (
          <Card>
            <CardHeader>
              <h3 className="text-sm font-semibold text-text-primary flex items-center gap-1.5">
                <Layers size={14} className="text-accent" />
                Governing Equations
              </h3>
            </CardHeader>
            <CardContent>
              <div className="space-y-2">
                {(selectedProblem.latex_equations ?? selectedProblem.equations).map((eq, i) => (
                  <div key={i} className="flex items-center gap-2 bg-slate-50 border border-slate-200 rounded-lg px-3 py-2 overflow-x-auto">
                    <span className="text-[10px] text-slate-400 select-none flex-shrink-0 tabular-nums w-4 text-right self-start pt-0.5">{i + 1}</span>
                    {selectedProblem.latex_equations ? (
                      <MathEq latex={eq} display={false} className="text-sm min-w-0" />
                    ) : (
                      <span className="font-mono text-xs text-accent">{eq}</span>
                    )}
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

// ─── Step 3: Geometry ─────────────────────────────────────────────────────────

function StepGeometry() {
  const { selectedProblem, selectedGeometry, setSelectedGeometry } = useExperimentStore()

  const { data: geometries, isLoading, isError, refetch, isFetching } = useQuery({
    queryKey: ['geometries', selectedProblem?.id],
    queryFn: () => geometriesApi.list(selectedProblem?.id).then((r) => r.data),
    enabled: !!selectedProblem,
    retry: 1,
  })

  if (isLoading) return <PageLoading label="Loading geometries…" />

  if (isError) {
    return (
      <div className="rounded-lg bg-red-50 border border-red-200 p-6 text-center">
        <p className="text-sm font-medium text-red-700 mb-2">Failed to load geometries</p>
        <p className="text-xs text-red-500 mb-4">Make sure the backend is running.</p>
        <button
          onClick={() => refetch()}
          disabled={isFetching}
          className="text-xs underline text-red-600 hover:text-red-700 disabled:opacity-50"
        >
          {isFetching ? 'Retrying…' : 'Try again'}
        </button>
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <p className="text-sm text-text-secondary">
        Select a 3D geometry preset for the problem. The geometry defines the spatial domain, boundary shapes, and mesh.
      </p>
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
        {geometries?.map((g) => (
          <GeometryCard
            key={g.id}
            geometry={g}
            selected={selectedGeometry?.id === g.id}
            onClick={() => setSelectedGeometry(selectedGeometry?.id === g.id ? null : g)}
          />
        ))}
        {(!geometries || geometries.length === 0) && (
          <div className="col-span-full text-center py-8 text-text-muted">
            No geometries available for this problem
          </div>
        )}
      </div>
    </div>
  )
}

// ─── Step 4: Dataset Preview ──────────────────────────────────────────────────

// Normalize any solution entry (scalar or vector) to a single number.
// Single-element lists like [-0.195] are unwrapped preserving sign.
// Multi-component vectors [u, v, p] take the first component (not magnitude).
function toScalar(v: number | number[]): number {
  if (Array.isArray(v)) {
    if (v.length === 0) return 0
    if (v.length === 1) return Number(v[0])  // single-element list → unwrap, preserve sign
    return v[0]  // multi-component: take first component (e.g., u from [u,v,p])
  }
  return Number(v)
}

// Ensure coords is always number[][]
function toCoords(raw: any[]): number[][] {
  return raw.map((c) => (Array.isArray(c) ? c.map(Number) : [Number(c), 0]))
}

// Flatten solution to scalar[]
function toScalarArray(raw: any[]): number[] {
  return raw.map(toScalar)
}

// Snapshot line colors for time-series overlays
const SNAPSHOT_COLORS = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6']

// Slice view for 3D data using ScientificHeatmap
function SliceView3D({
  coords,
  solution,
  axis,
  sliceValue,
  label,
  colormap,
}: {
  coords: number[][]
  solution: number[]
  axis: 'x' | 'y' | 'z'
  sliceValue: number
  label: string
  colormap: ColormapName
}) {
  const axisIdx = axis === 'x' ? 0 : axis === 'y' ? 1 : 2
  const eps = 0.15

  const filtered = coords
    .map((c, i) => ({ c, val: solution[i] }))
    .filter(({ c }) => Math.abs((c[axisIdx] ?? 0) - sliceValue) < eps)

  if (filtered.length === 0) {
    return (
      <div className="text-center py-4 text-xs text-text-muted">
        No points near {axis}={sliceValue.toFixed(2)}
      </div>
    )
  }

  const sliceCoords = filtered.map(({ c }) => [c[(axisIdx + 1) % 3] ?? 0, c[(axisIdx + 2) % 3] ?? 0])
  const sliceSolution = filtered.map(({ val }) => val)
  const hm = buildHeatmapFromScatter(sliceCoords, sliceSolution)
  if (!hm) return null

  const ax1 = ['y', 'z', 'x'][(axisIdx + 1) % 3]
  const ax2 = ['y', 'z', 'x'][(axisIdx + 2) % 3]

  return (
    <div>
      <div className="text-xs font-medium text-text-muted mb-1">{label}</div>
      <div style={{ overflowX: 'auto' }}>
        <ScientificHeatmap
          data={hm.data}
          xValues={hm.xValues}
          yValues={hm.yValues}
          xLabel={ax1}
          yLabel={ax2}
          colormap={colormap}
          width={460}
          height={200}
          symmetric={colormap === 'rdbu'}
        />
      </div>
    </div>
  )
}

// ── Animated mesh timeline ─────────────────────────────────────────────────

interface TimeFrame {
  tau: number | null
  t_s: number | null
  label: string
  field: number[]
}

interface TimeAnimation {
  vertices: number[][]
  faces: number[][]
  field_name: string
  frames: TimeFrame[]
}

interface AnimFrame2D {
  t_s: number
  label: string
  grid: number[][]   // vorticity, shape [nx][ny], grid[xi][yi] convention
  u?: number[][]     // x-velocity on coarser grid (optional, for streamlines)
  v?: number[][]     // y-velocity on coarser grid
  xv?: number[]      // x-coords of velocity grid
  yv?: number[]      // y-coords of velocity grid
}

// ── Canvas-based vorticity renderer ──────────────────────────────────────────

function rdbuRgb(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  const blue  = [33,  102, 172]
  const white = [247, 247, 247]
  const red   = [178,  24,  43]
  let r, g, b
  if (t <= 0.5) {
    const s = t * 2
    r = blue[0] + s * (white[0] - blue[0])
    g = blue[1] + s * (white[1] - blue[1])
    b = blue[2] + s * (white[2] - blue[2])
  } else {
    const s = (t - 0.5) * 2
    r = white[0] + s * (red[0] - white[0])
    g = white[1] + s * (red[1] - white[1])
    b = white[2] + s * (red[2] - white[2])
  }
  return [Math.round(r), Math.round(g), Math.round(b)]
}

function bilinInterp(
  field: number[][], xArr: number[], yArr: number[],
  x: number, y: number,
): number {
  const nx = xArr.length, ny = yArr.length
  const fx = ((x - xArr[0]) / (xArr[nx-1] - xArr[0])) * (nx - 1)
  const fy = ((y - yArr[0]) / (yArr[ny-1] - yArr[0])) * (ny - 1)
  const ix = Math.max(0, Math.min(Math.floor(fx), nx - 2))
  const iy = Math.max(0, Math.min(Math.floor(fy), ny - 2))
  const tx = fx - ix, ty = fy - iy
  return (
    (field[ix]?.[iy] ?? 0) * (1-tx)*(1-ty) +
    (field[ix+1]?.[iy] ?? 0) * tx*(1-ty) +
    (field[ix]?.[iy+1] ?? 0) * (1-tx)*ty +
    (field[ix+1]?.[iy+1] ?? 0) * tx*ty
  )
}

function drawStreamlines(
  ctx: CanvasRenderingContext2D,
  uGrid: number[][], vGrid: number[][],
  xvArr: number[], yvArr: number[],
  toPx: (x: number) => number, toPy: (y: number) => number,
  plotX0: number, plotY0: number, plotW: number, plotH: number,
  xMin: number, xMax: number, yMin: number, yMax: number,
) {
  const R = 0.5   // cylinder radius (D=1)
  const ds = 0.22 // arc-length step in domain units

  // Seed on a regular grid to the left, plus a few top/bottom rows
  const seedRows: [number, number][] = []
  const nRows = 36
  for (let i = 0; i < nRows; i++) {
    const y = yMin + (i + 0.5) / nRows * (yMax - yMin)
    seedRows.push([xMin + 0.05, y])
  }

  ctx.save()
  ctx.beginPath()
  ctx.rect(plotX0, plotY0, plotW, plotH)
  ctx.clip()
  ctx.strokeStyle = 'rgba(255,255,255,0.52)'
  ctx.lineWidth = 0.9
  ctx.lineCap = 'round'

  for (const [sx, sy] of seedRows) {
    if (sx * sx + sy * sy < (R * 1.15) ** 2) continue

    let px = sx, py = sy
    ctx.beginPath()
    let moved = false

    for (let step = 0; step < 600; step++) {
      if (px < xMin || px > xMax || py < yMin || py > yMax) break
      if (px * px + py * py < (R * 1.08) ** 2) break

      const screenX = toPx(px), screenY = toPy(py)
      if (!moved) { ctx.moveTo(screenX, screenY); moved = true }
      else ctx.lineTo(screenX, screenY)

      const u = bilinInterp(uGrid, xvArr, yvArr, px, py)
      const v = bilinInterp(vGrid, xvArr, yvArr, px, py)
      const speed = Math.sqrt(u * u + v * v) + 1e-9
      px += (u / speed) * ds
      py += (v / speed) * ds
    }
    if (moved) ctx.stroke()
  }
  ctx.restore()
}

function drawVorticityCanvas(
  canvas: HTMLCanvasElement,
  grid: number[][],
  xArr: number[], yArr: number[],
  vmin: number, vmax: number,
  uGrid?: number[][], vGrid?: number[][],
  xvArr?: number[], yvArr?: number[],
) {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  const W = canvas.width, H = canvas.height
  const nx = xArr.length, ny = yArr.length
  const xMin = xArr[0], xMax = xArr[nx - 1]
  const yMin = yArr[0], yMax = yArr[ny - 1]
  const range = vmax - vmin || 1

  const PL = 50, PR = 76, PT = 26, PB = 32
  const plotW = W - PL - PR
  const plotH = H - PT - PB

  // Background
  ctx.fillStyle = '#f8fafc'
  ctx.fillRect(0, 0, W, H)
  ctx.fillStyle = '#ffffff'
  ctx.fillRect(PL, PT, plotW, plotH)

  // Heatmap pixels via bilinear interpolation
  const imgData = ctx.createImageData(plotW, plotH)
  const buf = imgData.data
  for (let py = 0; py < plotH; py++) {
    for (let px = 0; px < plotW; px++) {
      const fx = (px / Math.max(plotW - 1, 1)) * (nx - 1)
      const fy = ((plotH - 1 - py) / Math.max(plotH - 1, 1)) * (ny - 1)
      const ix = Math.min(Math.floor(fx), nx - 2)
      const iy = Math.min(Math.floor(fy), ny - 2)
      const tx = fx - ix, ty = fy - iy
      const v00 = grid[ix]?.[iy] ?? 0
      const v10 = grid[ix + 1]?.[iy] ?? 0
      const v01 = grid[ix]?.[iy + 1] ?? 0
      const v11 = grid[ix + 1]?.[iy + 1] ?? 0
      const val = v00*(1-tx)*(1-ty) + v10*tx*(1-ty) + v01*(1-tx)*ty + v11*tx*ty
      const [r, g, b] = rdbuRgb((val - vmin) / range)
      const idx = (py * plotW + px) * 4
      buf[idx] = r; buf[idx+1] = g; buf[idx+2] = b; buf[idx+3] = 255
    }
  }
  ctx.putImageData(imgData, PL, PT)

  // Helper: domain → pixel
  const toPx = (x: number) => PL + ((x - xMin) / (xMax - xMin)) * plotW
  const toPy = (y: number) => PT + ((yMax - y) / (yMax - yMin)) * plotH

  // Streamlines (drawn on top of heatmap, before cylinder + axes)
  if (uGrid && vGrid && xvArr && yvArr) {
    drawStreamlines(ctx, uGrid, vGrid, xvArr, yvArr, toPx, toPy, PL, PT, plotW, plotH, xMin, xMax, yMin, yMax)
  }

  // Subtle grid lines
  ctx.strokeStyle = 'rgba(203,213,225,0.35)'
  ctx.lineWidth = 0.5
  ;[-2, 0, 2, 6, 10, 14, 18].forEach(xv => {
    if (xv < xMin || xv > xMax) return
    const px = toPx(xv)
    ctx.beginPath(); ctx.moveTo(px, PT); ctx.lineTo(px, PT + plotH); ctx.stroke()
  })
  ;[-4, -2, 0, 2, 4].forEach(yv => {
    if (yv < yMin || yv > yMax) return
    const py = toPy(yv)
    ctx.beginPath(); ctx.moveTo(PL, py); ctx.lineTo(PL + plotW, py); ctx.stroke()
  })

  // Cylinder at (0, 0) with radius = 0.5 (D/2 when D=1)
  const cR = 0.5
  const cxPx = toPx(0)
  const cyPx = toPy(0)
  const rPxX = (cR / (xMax - xMin)) * plotW
  const rPxY = (cR / (yMax - yMin)) * plotH
  ctx.beginPath()
  ctx.ellipse(cxPx, cyPx, rPxX, rPxY, 0, 0, 2 * Math.PI)
  ctx.fillStyle = 'rgba(15,23,42,0.88)'
  ctx.fill()
  ctx.strokeStyle = '#e2e8f0'
  ctx.lineWidth = 1.5
  ctx.stroke()

  // Plot border
  ctx.strokeStyle = '#cbd5e1'
  ctx.lineWidth = 1
  ctx.strokeRect(PL, PT, plotW, plotH)

  // X-axis
  ctx.fillStyle = '#64748b'
  ctx.font = '9px "JetBrains Mono", monospace'
  ctx.textAlign = 'center'
  ;[-2, 0, 2, 6, 10, 14, 18].forEach(xv => {
    if (xv < xMin || xv > xMax) return
    const px = toPx(xv)
    ctx.beginPath(); ctx.moveTo(px, PT+plotH); ctx.lineTo(px, PT+plotH+4)
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1; ctx.stroke()
    ctx.fillText(xv.toString(), px, PT + plotH + 14)
  })
  ctx.font = '10px system-ui, sans-serif'
  ctx.fillStyle = '#475569'
  ctx.fillText('x / D', PL + plotW / 2, H - 4)

  // Y-axis
  ctx.textAlign = 'right'
  ctx.font = '9px "JetBrains Mono", monospace'
  ;[-4, -2, 0, 2, 4].forEach(yv => {
    if (yv < yMin || yv > yMax) return
    const py = toPy(yv)
    ctx.beginPath(); ctx.moveTo(PL-4, py); ctx.lineTo(PL, py)
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1; ctx.stroke()
    ctx.fillStyle = '#64748b'
    ctx.fillText(yv.toString(), PL - 6, py + 3)
  })
  ctx.save()
  ctx.translate(12, PT + plotH / 2)
  ctx.rotate(-Math.PI / 2)
  ctx.textAlign = 'center'
  ctx.font = '10px system-ui, sans-serif'
  ctx.fillStyle = '#475569'
  ctx.fillText('y / D', 0, 0)
  ctx.restore()

  // Colorbar
  const cbX = PL + plotW + 10, cbW = 12, cbY = PT
  for (let i = 0; i < plotH; i++) {
    const [r, g, b] = rdbuRgb(1 - i / (plotH - 1))
    ctx.fillStyle = `rgb(${r},${g},${b})`
    ctx.fillRect(cbX, cbY + i, cbW, 1)
  }
  ctx.strokeStyle = '#cbd5e1'; ctx.lineWidth = 1
  ctx.strokeRect(cbX, cbY, cbW, plotH)
  ctx.textAlign = 'left'
  ctx.font = '8px "JetBrains Mono", monospace'
  ctx.fillStyle = '#64748b'
  for (let i = 0; i <= 4; i++) {
    const t = i / 4
    const val = vmin + t * (vmax - vmin)
    const py = cbY + (1 - t) * plotH
    ctx.beginPath(); ctx.moveTo(cbX+cbW, py); ctx.lineTo(cbX+cbW+3, py)
    ctx.strokeStyle = '#94a3b8'; ctx.lineWidth = 1; ctx.stroke()
    ctx.fillText(val.toFixed(2), cbX + cbW + 5, py + 3)
  }
}

// ── Benchmark reference table ────────────────────────────────────────────────

function CylinderBenchmarkTable({ Re }: { Re: number }) {
  const refs = [
    { Re: 40,  Cd: 1.54, St: null  },
    { Re: 60,  Cd: 1.43, St: 0.136 },
    { Re: 80,  Cd: 1.40, St: 0.153 },
    { Re: 100, Cd: 1.37, St: 0.164 },
    { Re: 140, Cd: 1.32, St: 0.174 },
    { Re: 180, Cd: 1.28, St: 0.179 },
  ]
  return (
    <div className="mt-3 border border-slate-200 rounded-lg overflow-hidden">
      <div className="flex items-center gap-1.5 px-3 py-1.5 bg-slate-50 border-b border-slate-200">
        <span className="text-[10px] font-bold text-text-muted uppercase tracking-wider">
          Benchmark Reference
        </span>
        <span className="text-[10px] text-text-muted">— Williamson 1988 · Liu et al. 1998</span>
      </div>
      <table className="w-full text-[10px] font-mono">
        <thead>
          <tr className="border-b border-slate-100 bg-slate-50/60">
            <th className="text-left px-3 py-1 font-semibold text-text-muted">Re</th>
            <th className="text-right px-3 py-1 font-semibold text-text-muted">Cd (lit.)</th>
            <th className="text-right px-3 py-1 font-semibold text-text-muted">St (lit.)</th>
            <th className="text-right px-3 py-1 font-semibold text-text-muted">St (Williamson)</th>
          </tr>
        </thead>
        <tbody>
          {refs.map(r => {
            const stW = r.St !== null ? (0.198 * (1 - 19.7 / r.Re)).toFixed(3) : '—'
            const isActive = Math.abs(r.Re - Re) < 15
            return (
              <tr
                key={r.Re}
                className={`border-b border-slate-100 last:border-0 ${isActive ? 'bg-blue-50' : ''}`}
              >
                <td className={`px-3 py-0.5 ${isActive ? 'font-bold text-accent' : 'text-text-secondary'}`}>
                  {r.Re}
                </td>
                <td className="text-right px-3 py-0.5 text-text-secondary">{r.Cd.toFixed(2)}</td>
                <td className="text-right px-3 py-0.5 text-text-secondary">
                  {r.St !== null ? r.St.toFixed(3) : '—'}
                </td>
                <td className="text-right px-3 py-0.5 text-text-secondary">{stW}</td>
              </tr>
            )
          })}
        </tbody>
      </table>
      <div className="px-3 py-1 text-[9px] text-text-muted italic">
        Current Re = {Re.toFixed(0)} (highlighted row = closest reference point)
      </div>
    </div>
  )
}

// ── Animated heatmap (canvas) ─────────────────────────────────────────────────

function AnimatedHeatmap2D({
  xArr, yArr, frames, vmin, vmax, metadata,
}: {
  xArr: number[]
  yArr: number[]
  frames: AnimFrame2D[]
  vmin: number
  vmax: number
  metadata?: Record<string, any>
}) {
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying]   = useState(false)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  // Canvas dimensions — keep 2:1 aspect matching the 20D × 10D domain
  const CANVAS_W = 612
  const CANVAS_H = 306

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      setFrameIdx(prev => {
        if (prev >= frames.length - 1) { setPlaying(false); return prev }
        return prev + 1
      })
    }, 700)
    return () => clearInterval(id)
  }, [playing, frames.length])

  useLayoutEffect(() => {
    const canvas = canvasRef.current
    const f = frames[frameIdx]
    if (!canvas || !f) return
    drawVorticityCanvas(
      canvas, f.grid, xArr, yArr, vmin, vmax,
      f.u as number[][] | undefined,
      f.v as number[][] | undefined,
      f.xv as number[] | undefined,
      f.yv as number[] | undefined,
    )
  }, [frameIdx, frames, xArr, yArr, vmin, vmax])

  const frame = frames[frameIdx]
  const Re = Number(metadata?.Re ?? 100)

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">
        Vorticity field (ω) — Kármán vortex street animation
      </div>

      <div style={{ overflowX: 'auto' }}>
        <div className="text-[10px] font-mono text-center text-text-secondary mb-1.5 font-semibold tracking-wide">
          ω (vorticity) · {frame?.label ?? ''}
        </div>
        <canvas
          ref={canvasRef}
          width={CANVAS_W}
          height={CANVAS_H}
          style={{ display: 'block', borderRadius: 6, border: '1px solid #e2e8f0' }}
        />
      </div>

      {/* Timeline controls */}
      <div className="flex items-center gap-3 px-1">
        <button
          onClick={() => {
            if (frameIdx >= frames.length - 1) setFrameIdx(0)
            setPlaying(p => !p)
          }}
          className="w-8 h-8 rounded-full bg-accent text-white flex items-center justify-center flex-shrink-0 hover:bg-accent/80 transition-colors"
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? (
            <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
              <rect x="0" y="0" width="3" height="12" rx="1" />
              <rect x="7" y="0" width="3" height="12" rx="1" />
            </svg>
          ) : (
            <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
              <polygon points="0,0 10,6 0,12" />
            </svg>
          )}
        </button>
        <input
          type="range"
          min={0} max={frames.length - 1} value={frameIdx}
          onChange={e => { setPlaying(false); setFrameIdx(Number(e.target.value)) }}
          className="flex-1 accent-accent h-1.5 cursor-pointer"
        />
        <span className="text-xs font-mono text-text-secondary w-24 text-right flex-shrink-0">
          {frame?.label ?? ''}
        </span>
      </div>

      {/* Frame ticks */}
      <div className="flex gap-0.5 px-11">
        {frames.map((_, i) => (
          <button
            key={i}
            onClick={() => { setPlaying(false); setFrameIdx(i) }}
            className={`flex-1 h-1.5 rounded-sm transition-colors ${i === frameIdx ? 'bg-accent' : 'bg-slate-200 hover:bg-slate-300'}`}
            title={frames[i].label}
          />
        ))}
      </div>

      <CylinderBenchmarkTable Re={Re} />
    </div>
  )
}

function AnimatedMeshView({ anim, title, subtitle }: { anim: TimeAnimation; title: string; subtitle?: string }) {
  const [frameIdx, setFrameIdx] = useState(0)
  const [playing, setPlaying] = useState(false)

  useEffect(() => {
    if (!playing) return
    const id = setInterval(() => {
      setFrameIdx((prev) => {
        if (prev >= anim.frames.length - 1) {
          setPlaying(false)
          return prev
        }
        return prev + 1
      })
    }, 700)
    return () => clearInterval(id)
  }, [playing, anim.frames.length])

  // Lock color range to the steady-state frame so colours are consistent across time
  const [globalVmin, globalVmax] = useMemo(() => {
    const steadyField = anim.frames[anim.frames.length - 1].field.filter(isFinite)
    if (!steadyField.length) return [0, 1]
    const mn = Math.min(...steadyField), mx = Math.max(...steadyField)
    return mn === mx ? [mn - 0.5, mx + 0.5] : [mn, mx]
  }, [anim])

  const frame = anim.frames[frameIdx]
  const mesh: SurfaceMeshData = {
    vertices: anim.vertices,
    faces: anim.faces,
    field: frame.field,
    field_name: anim.field_name,
  }

  return (
    <div className="space-y-3">
      <div className="text-xs font-semibold text-text-muted uppercase tracking-wider">
        {subtitle ?? 'Surface Mesh — Transient Startup'} · drag to rotate
      </div>

      <SurfaceMeshView
        mesh={mesh}
        colormap="auto"
        title={`${title}  ·  ${frame.label}`}
        fieldLabel={anim.field_name}
        width={560}
        height={420}
        vmin={globalVmin}
        vmax={globalVmax}
      />

      {/* Timeline controls */}
      <div className="flex items-center gap-3 px-1">
        {/* Play / Pause */}
        <button
          onClick={() => {
            if (frameIdx >= anim.frames.length - 1) setFrameIdx(0)
            setPlaying((p) => !p)
          }}
          className="w-8 h-8 rounded-full bg-accent text-white flex items-center justify-center flex-shrink-0 hover:bg-accent/80 transition-colors"
          title={playing ? 'Pause' : 'Play'}
        >
          {playing ? (
            <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
              <rect x="0" y="0" width="3" height="12" rx="1" />
              <rect x="7" y="0" width="3" height="12" rx="1" />
            </svg>
          ) : (
            <svg width="10" height="12" viewBox="0 0 10 12" fill="currentColor">
              <polygon points="0,0 10,6 0,12" />
            </svg>
          )}
        </button>

        {/* Slider */}
        <input
          type="range"
          min={0}
          max={anim.frames.length - 1}
          value={frameIdx}
          onChange={(e) => { setPlaying(false); setFrameIdx(Number(e.target.value)) }}
          className="flex-1 accent-accent h-1.5 cursor-pointer"
        />

        {/* Time label */}
        <span className="text-xs font-mono text-text-secondary w-28 text-right flex-shrink-0">
          {frame.label}
          {frame.t_s !== null && frame.t_s !== undefined && (
            <span className="text-text-muted ml-1">({frame.t_s.toFixed(2)} s)</span>
          )}
        </span>
      </div>

      {/* Frame ticks */}
      <div className="flex gap-0.5 px-11">
        {anim.frames.map((f, i) => (
          <button
            key={i}
            onClick={() => { setPlaying(false); setFrameIdx(i) }}
            className={`flex-1 h-1.5 rounded-sm transition-colors ${
              i === frameIdx ? 'bg-accent' : 'bg-slate-200 hover:bg-slate-300'
            }`}
            title={f.label}
          />
        ))}
      </div>
    </div>
  )
}

function DatasetChart({ preview }: { preview: any }) {
  const metadata = preview.metadata ?? {}
  const dim = metadata.dim ?? 1
  const problem = (metadata.problem ?? '').toLowerCase()

  // Choose colormap based on problem type
  const colormap: ColormapName = (() => {
    if (problem.includes('navier') || problem.includes('burgers') || problem.includes('cylinder')) return 'rdbu'
    if (problem.includes('heat') || problem.includes('thermal') || problem.includes('schrodinger')) return 'viridis'
    return 'rdbu'
  })()

  // ── Time-dependent 1D problems (Burgers, heat_1d) ──────────────────────────
  // These come back with preview.x (nx), preview.t (nt), preview.u ([nt][nx])
  // Guard dim !== 3: 3D+time problems (heat_eq_2d) use the animated mesh branch, not this one
  const hasTimeSeries1D = Array.isArray(preview.x) && Array.isArray(preview.t) && Array.isArray(preview.u) && dim !== 3

  if (hasTimeSeries1D) {
    const xArr: number[] = preview.x.map(Number)
    const tArr: number[] = preview.t.map(Number)
    const uGrid: number[][] = (preview.u as any[][]).map((row: any[]) => row.map(Number))

    const nt = tArr.length
    const fieldLabel = problem.includes('heat') ? 'T(x, t)' : 'u(x, t)'

    // Build heatmap: rows = time (t), cols = position (x)
    // uGrid is [nt][nx], xValues = x positions, yValues = time values
    const hm = buildHeatmapFromGrid(xArr, tArr, uGrid)

    // Snapshot line chart at 5 time indices
    const snapIndices = [0, Math.floor(nt / 4), Math.floor(nt / 2), Math.floor((3 * nt) / 4), nt - 1]
    const snapshotData = xArr.map((x, xi) => {
      const point: Record<string, number> = { x }
      snapIndices.forEach((ti) => {
        point[`t=${tArr[ti].toFixed(2)}`] = uGrid[ti]?.[xi] ?? 0
      })
      return point
    })
    const snapshotKeys = snapIndices.map((ti) => `t=${tArr[ti].toFixed(2)}`)

    return (
      <div className="space-y-4">
        {/* Heatmap */}
        <div style={{ overflowX: 'auto' }}>
          <ScientificHeatmap
            data={hm.data}
            xValues={hm.xValues}
            yValues={hm.yValues}
            xLabel="x"
            yLabel="t"
            title={fieldLabel}
            colormap={colormap}
            symmetric={colormap === 'rdbu'}
            width={480}
            height={280}
          />
        </div>

        {/* Multi-snapshot line chart */}
        <div>
          <div className="text-xs font-medium text-text-muted mb-1">Snapshots at selected time steps</div>
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={snapshotData} margin={{ top: 5, right: 16, bottom: 16, left: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis
                dataKey="x"
                tick={{ fill: '#64748b', fontSize: 9 }}
                axisLine={{ stroke: '#e2e8f0' }}
                tickLine={false}
                label={{ value: 'x', position: 'insideBottom', offset: -8, fill: '#475569', fontSize: 10 }}
              />
              <YAxis tick={{ fill: '#64748b', fontSize: 9 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
              <Tooltip
                contentStyle={{ background: '#fff', border: '1px solid #e2e8f0', borderRadius: 8, fontSize: 10 }}
                formatter={(v: number, name: string) => [v.toFixed(4), name]}
              />
              <Legend iconSize={8} wrapperStyle={{ fontSize: 9 }} />
              {snapshotKeys.map((key, ki) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={SNAPSHOT_COLORS[ki % SNAPSHOT_COLORS.length]}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    )
  }

  // ── 2D spatial problems (Laplace, Poisson, NS2D) ───────────────────────────
  // These come back with preview.x, preview.y, preview.u (and optionally v, p, omega)
  // Guard dim !== 3: 3D problems that happen to carry x/y/u keys route to the 3D branch instead
  const hasSpatialGrid2D = Array.isArray(preview.x) && Array.isArray(preview.y) && Array.isArray(preview.u) && dim !== 3

  if (hasSpatialGrid2D) {
    const xArr: number[] = preview.x.map(Number)
    const yArr: number[] = preview.y.map(Number)
    const uRaw = preview.u as number[][]

    // ── Animated 2D heatmap (cylinder flow, etc.) ─────────────────────────
    const animFrames2D = preview.animated_frames as AnimFrame2D[] | undefined
    if (animFrames2D?.length) {
      const vmin = typeof preview.vmin === 'number' ? preview.vmin : undefined
      const vmax = typeof preview.vmax === 'number' ? preview.vmax : undefined
      // Derive symmetric range from data if not provided by backend
      const allAbs: number[] = animFrames2D.flatMap(f => f.grid.flatMap(row => row.map(Math.abs)))
      const autoRange = allAbs.reduce((acc, v) => Math.max(acc, v), 0) || 1
      return (
        <AnimatedHeatmap2D
          xArr={xArr}
          yArr={yArr}
          frames={animFrames2D}
          vmin={vmin ?? -autoRange}
          vmax={vmax ??  autoRange}
          metadata={metadata}
        />
      )
    }

    // Transpose: backend sends u[xi][yi] (indexing="ij"), heatmap needs data[row=yi][col=xi]
    const transposed: number[][] = yArr.map((_: any, yi: number) =>
      xArr.map((_: any, xi: number) => uRaw[xi]?.[yi] ?? 0)
    )
    const hmData = { data: transposed, xValues: xArr, yValues: yArr }

    const hasV = Array.isArray(preview.v) && (preview.v as any[]).length > 0
    const hasP = Array.isArray(preview.p) && (preview.p as any[]).length > 0
    const hasOmega = Array.isArray(preview.omega) && (preview.omega as any[]).length > 0

    if (hasV || hasP || hasOmega) {
      // NS2D / cavity: show u, v, p, omega panels
      const fieldPanels = [
        hasV ? { label: 'v (y-velocity)', grid: preview.v as number[][], cm: 'rdbu' as const } : null,
        hasP ? { label: 'p (pressure)', grid: preview.p as number[][], cm: 'viridis' as const } : null,
        hasOmega ? { label: 'ω (vorticity)', grid: preview.omega as number[][], cm: 'rdbu' as const } : null,
      ].filter(Boolean) as { label: string; grid: number[][]; cm: ColormapName }[]

      return (
        <div className="space-y-4">
          <div style={{ overflowX: 'auto' }}>
            <ScientificHeatmap
              data={hmData.data}
              xValues={hmData.xValues}
              yValues={hmData.yValues}
              xLabel="x"
              yLabel="y"
              title="u (x-velocity)"
              colormap={colormap}
              symmetric={colormap === 'rdbu'}
              width={480}
              height={260}
            />
          </div>
          <div className="grid grid-cols-2 gap-4">
            {fieldPanels.map(fp => {
              const transposedField: number[][] = yArr.map((_: any, yi: number) =>
                xArr.map((_: any, xi: number) => fp.grid[xi]?.[yi] ?? 0)
              )
              return (
                <div key={fp.label} style={{ overflowX: 'auto' }}>
                  <ScientificHeatmap
                    data={transposedField}
                    xValues={xArr}
                    yValues={yArr}
                    xLabel="x"
                    yLabel="y"
                    title={fp.label}
                    colormap={fp.cm}
                    symmetric={fp.cm === 'rdbu'}
                    width={230}
                    height={180}
                  />
                </div>
              )
            })}
          </div>
        </div>
      )
    }

    return (
      <div style={{ overflowX: 'auto' }}>
        <ScientificHeatmap
          data={hmData.data}
          xValues={hmData.xValues}
          yValues={hmData.yValues}
          xLabel="x"
          yLabel="y"
          title="u(x, y)"
          colormap={colormap}
          symmetric={colormap === 'rdbu'}
          width={480}
          height={280}
        />
      </div>
    )
  }

  // ── 3D geometry — surface mesh (ParaView-style) or point cloud ──────────
  if (dim === 3) {
    const problemTitle = metadata.description ?? metadata.problem ?? '3D Problem'
    const surfaceMeshRaw = preview.surface_mesh as SurfaceMeshData | undefined

    // ── Time animation — checked FIRST regardless of whether there's also a static mesh ──
    const timeAnim = preview.time_animation as TimeAnimation | undefined
    if (timeAnim?.frames?.length && timeAnim.vertices?.length) {
      const hasVolume = preview.coords && preview.solution
      const coords: number[][] = hasVolume ? toCoords(preview.coords as any[]) : []
      const solution: number[] = hasVolume ? toScalarArray(preview.solution as any[]) : []
      const midX = coords.length ? (Math.min(...coords.map(c=>c[0])) + Math.max(...coords.map(c=>c[0]))) / 2 : 0
      const midY = coords.length ? (Math.min(...coords.map(c=>c[1])) + Math.max(...coords.map(c=>c[1]))) / 2 : 0
      const midZ = coords.length ? (Math.min(...coords.map(c=>c[2])) + Math.max(...coords.map(c=>c[2]))) / 2 : 0
      return (
        <div className="space-y-5">
          <AnimatedMeshView
            anim={timeAnim}
            title={problemTitle}
            subtitle={timeAnim.field_name ? `${timeAnim.field_name} — time evolution` : undefined}
          />
          {coords.length > 0 && (
            <details className="group">
              <summary className="cursor-pointer text-xs font-semibold text-text-muted uppercase tracking-wider select-none list-none flex items-center gap-1">
                <span className="group-open:rotate-90 transition-transform inline-block">▶</span>
                Volume cross-section slices
              </summary>
              <div className="mt-3 space-y-3">
                <SliceView3D coords={coords} solution={solution} axis="z" sliceValue={midZ} label={`XY plane  z ≈ ${midZ.toFixed(2)}`} colormap={colormap} />
                <SliceView3D coords={coords} solution={solution} axis="y" sliceValue={midY} label={`XZ plane  y ≈ ${midY.toFixed(2)}`} colormap={colormap} />
                <SliceView3D coords={coords} solution={solution} axis="x" sliceValue={midX} label={`YZ plane  x ≈ ${midX.toFixed(2)}`} colormap={colormap} />
              </div>
            </details>
          )}
        </div>
      )
    }

    // ── Surface mesh (ParaView-style solid surface) ──────────────────────
    if (surfaceMeshRaw?.vertices?.length && surfaceMeshRaw?.faces?.length) {
      const mesh: SurfaceMeshData = surfaceMeshRaw
      const fieldName = mesh.field_name ?? (problem.includes('navier') ? 'Cp' : 'u')

      // Also prepare scatter slices if volume data is present
      const hasVolume = preview.coords && preview.solution
      const coords: number[][] = hasVolume ? toCoords(preview.coords as any[]) : []
      const solution: number[] = hasVolume ? toScalarArray(preview.solution as any[]) : []
      const midX = coords.length ? (Math.min(...coords.map(c=>c[0])) + Math.max(...coords.map(c=>c[0]))) / 2 : 0
      const midY = coords.length ? (Math.min(...coords.map(c=>c[1])) + Math.max(...coords.map(c=>c[1]))) / 2 : 0
      const midZ = coords.length ? (Math.min(...coords.map(c=>c[2])) + Math.max(...coords.map(c=>c[2]))) / 2 : 0

      return (
        <div className="space-y-5">
          {/* Solid surface mesh — PRIMARY */}
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
              Surface Mesh — drag to rotate · scroll to zoom
            </div>
            <SurfaceMeshView
              mesh={mesh}
              colormap="auto"
              title={problemTitle}
              fieldLabel={fieldName}
              width={560}
              height={440}
            />
          </div>

          {/* Volume slice views — secondary, collapsed */}
          {coords.length > 0 && (
            <details className="group">
              <summary className="cursor-pointer text-xs font-semibold text-text-muted uppercase tracking-wider select-none list-none flex items-center gap-1">
                <span className="group-open:rotate-90 transition-transform inline-block">▶</span>
                Volume cross-section slices
              </summary>
              <div className="mt-3 space-y-3">
                <SliceView3D coords={coords} solution={solution} axis="z" sliceValue={midZ} label={`XY plane  z ≈ ${midZ.toFixed(2)}`} colormap={colormap} />
                <SliceView3D coords={coords} solution={solution} axis="y" sliceValue={midY} label={`XZ plane  y ≈ ${midY.toFixed(2)}`} colormap={colormap} />
                <SliceView3D coords={coords} solution={solution} axis="x" sliceValue={midX} label={`YZ plane  x ≈ ${midX.toFixed(2)}`} colormap={colormap} />
              </div>
            </details>
          )}
        </div>
      )
    }

    // ── Point cloud fallback (no surface mesh) ───────────────────────────
    if (preview.coords && preview.solution) {
      const coords: number[][] = toCoords(preview.coords as any[])
      const solution: number[] = toScalarArray(preview.solution as any[])

      const xs = coords.map((c) => c[0])
      const ys = coords.map((c) => c[1])
      const zs = coords.map((c) => c[2])
      const midX = (Math.min(...xs) + Math.max(...xs)) / 2
      const midY = (Math.min(...ys) + Math.max(...ys)) / 2
      const midZ = (Math.min(...zs) + Math.max(...zs)) / 2

      const fieldLabel3D = problem.includes('heat') ? 'T' : problem.includes('navier') ? 'u_x' : 'u'

      return (
        <div className="space-y-5">
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
              3D Point Cloud — drag to rotate · scroll to zoom
            </div>
            <Dataset3DView
              coords={coords}
              values={solution}
              xLabel="x" yLabel="y" zLabel="z"
              fieldLabel={fieldLabel3D}
              title={problemTitle}
              colormap={colormap}
              width={520}
              height={420}
            />
          </div>
          <details className="group">
            <summary className="cursor-pointer text-xs font-semibold text-text-muted uppercase tracking-wider select-none list-none flex items-center gap-1">
              <span className="group-open:rotate-90 transition-transform inline-block">▶</span>
              Cross-section slices
            </summary>
            <div className="mt-3 space-y-3">
              <SliceView3D coords={coords} solution={solution} axis="z" sliceValue={midZ} label={`XY plane  z ≈ ${midZ.toFixed(2)}`} colormap={colormap} />
              <SliceView3D coords={coords} solution={solution} axis="y" sliceValue={midY} label={`XZ plane  y ≈ ${midY.toFixed(2)}`} colormap={colormap} />
              <SliceView3D coords={coords} solution={solution} axis="x" sliceValue={midX} label={`YZ plane  x ≈ ${midX.toFixed(2)}`} colormap={colormap} />
            </div>
          </details>
        </div>
      )
    }
  }

  // ── 2D scatter fallback (coords + solution, no structured grid) ────────────
  if ((dim === 2 || preview.sample_y) && preview.coords && preview.solution) {
    const coords: number[][] = toCoords(preview.coords)
    const solution: number[] = toScalarArray(preview.solution)
    const hm = buildHeatmapFromScatter(coords, solution)
    if (hm) {
      return (
        <div style={{ overflowX: 'auto' }}>
          <ScientificHeatmap
            data={hm.data}
            xValues={hm.xValues}
            yValues={hm.yValues}
            xLabel="x"
            yLabel="y"
            title="u(x, y)"
            colormap={colormap}
            symmetric={colormap === 'rdbu'}
            width={480}
            height={280}
          />
        </div>
      )
    }
  }

  // ── 1D chart: bernoulli (u+p+A), darcy (u+p), beam (w+M), BEC (ψ+V), stokes/poiseuille (u) ──
  // Supports both preview.x (most 1D problems) and preview.y (stokes_flow, channel_poiseuille)
  const xKey1d = Array.isArray(preview.x) && !Array.isArray(preview.t)
    ? 'x'
    : Array.isArray(preview.y) && !Array.isArray(preview.t) && !Array.isArray(preview.x)
    ? 'y'
    : null
  if (xKey1d && dim !== 3) {
    const xArr1d = ((preview as any)[xKey1d] as any[]).map(Number)
    // Dynamic x-axis formatter: use exponential notation when range is tiny (physical units)
    const xRange = xArr1d.length > 1 ? xArr1d[xArr1d.length - 1] - xArr1d[0] : 1
    const xFmt = (v: number) => xRange < 1e-3 ? v.toExponential(2) : Number(v.toFixed(4)).toString()
    const candidates: [string, string, string][] = [
      ['u', 'u(x)', '#059669'],
      ['p', 'p(x)', '#2563eb'],
      ['w', 'w(x)', '#059669'],
      ['M', 'M(x)', '#d97706'],
      ['A', 'A(x)', '#a855f7'],
      ['psi', 'ψ(x)', '#059669'],
      ['V', 'V(x)', '#dc2626'],
    ]
    const mf: { label: string; data: number[]; color: string }[] = []
    for (const [key, label, color] of candidates) {
      const arr = (preview as any)[key]
      if (Array.isArray(arr) && (arr as any[]).length === xArr1d.length) {
        mf.push({ label, data: (arr as any[]).map(Number), color })
      }
    }
    if (mf.length >= 1) {
      const mfData = xArr1d.map((x, i) => {
        const pt: Record<string, number> = { x }
        mf.forEach(f => { pt[f.label] = f.data[i] ?? 0 })
        return pt
      })
      return (
        <ResponsiveContainer width="100%" height={220}>
          <LineChart data={mfData} margin={{ top: 5, right: 10, bottom: 16, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
            <XAxis dataKey="x" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} tickFormatter={xFmt} />
            <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
            <Tooltip contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 8, color: '#0f172a', fontSize: 11, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }} formatter={(v: number, name: string) => [v.toFixed(4), name]} />
            {mf.length > 1 && <Legend wrapperStyle={{ fontSize: 10 }} iconSize={8} />}
            {mf.map(f => (
              <Line key={f.label} type="monotone" dataKey={f.label} stroke={f.color} strokeWidth={2} dot={false} isAnimationActive={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      )
    }
  }

  // ── 1D line chart (fallback) ───────────────────────────────────────────────
  const chartData = (() => {
    if (preview.sample_x) {
      return (preview.sample_x as any[]).map((x: any, i: number) => ({
        x: Number(Number(x).toFixed(4)),
        value: Number(toScalar(preview.sample_values?.[i] ?? 0).toFixed(6)),
      }))
    }
    if (preview.coords && preview.solution) {
      const coords2 = toCoords(preview.coords)
      const sol2 = toScalarArray(preview.solution)
      return coords2
        .map((c, i) => ({ x: Number((c[0] ?? 0).toFixed(4)), value: Number(sol2[i].toFixed(6)) }))
        .sort((a, b) => a.x - b.x)
    }
    return []
  })()

  if (chartData.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-text-muted text-sm">
        No preview data available
      </div>
    )
  }

  const yLabel = problem.includes('heat') ? 'T(x)' : 'u(x)'
  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={chartData} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis dataKey="x" tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
        <YAxis tick={{ fill: '#64748b', fontSize: 10 }} axisLine={{ stroke: '#e2e8f0' }} tickLine={false} />
        <Tooltip
          contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 8, color: '#0f172a', fontSize: 11, boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
        />
        <Line type="monotone" dataKey="value" stroke="#059669" strokeWidth={2} dot={false} name={yLabel} />
      </LineChart>
    </ResponsiveContainer>
  )
}

// ─── Expected Format Box ──────────────────────────────────────────────────────

function ExpectedFormatBox({ hints, onApplyHints }: { hints: ColumnHints; onApplyHints?: () => void }) {
  return (
    <div className="rounded-xl border border-blue-200 bg-blue-50 p-4 space-y-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h5 className="text-xs font-bold text-blue-800 mb-0.5">Expected Column Format</h5>
          <p className="text-[11px] text-blue-700">
            Prepare your file with these columns for this problem.
          </p>
        </div>
        {onApplyHints && (
          <button
            onClick={onApplyHints}
            className="text-[11px] font-semibold text-blue-700 hover:text-blue-900 bg-blue-100 hover:bg-blue-200 px-2 py-1 rounded-md transition-colors whitespace-nowrap"
          >
            Auto-apply
          </button>
        )}
      </div>

      {/* CSV header example */}
      <div>
        <span className="text-[10px] font-semibold text-blue-600 uppercase tracking-wider">CSV header</span>
        <div className="mt-1 font-mono text-xs bg-white border border-blue-200 rounded px-3 py-1.5 text-blue-900 select-all">
          {hints.csv_header}
        </div>
      </div>

      {/* Column breakdown */}
      <div className="grid grid-cols-2 gap-3 text-[11px]">
        <div>
          <span className="font-semibold text-blue-700 block mb-1">Coordinates (coord_cols)</span>
          <div className="flex flex-wrap gap-1">
            {hints.coord_cols.map((c) => (
              <span key={c} className="font-mono bg-white border border-blue-300 text-blue-800 px-1.5 py-0.5 rounded">
                {c}
              </span>
            ))}
            {hints.time_col && (
              <span className="font-mono bg-purple-50 border border-purple-300 text-purple-700 px-1.5 py-0.5 rounded" title="time column">
                {hints.time_col} <span className="text-[9px]">(t)</span>
              </span>
            )}
          </div>
        </div>
        <div>
          <span className="font-semibold text-blue-700 block mb-1">Outputs (output_cols)</span>
          <div className="flex flex-wrap gap-1">
            {hints.output_cols.map((c) => (
              <span key={c} className="font-mono bg-green-50 border border-green-300 text-green-800 px-1.5 py-0.5 rounded">
                {c}
              </span>
            ))}
          </div>
        </div>
      </div>

      {hints.notes && (
        <p className="text-[11px] text-blue-700 italic">{hints.notes}</p>
      )}
    </div>
  )
}

// ─── Dataset Upload sub-component ────────────────────────────────────────────

function DatasetUploadZone({
  onUploaded,
  current,
  onAutoMap,
}: {
  onUploaded: (ds: UploadedDataset) => void
  current: UploadedDataset | null
  onAutoMap?: (ds: UploadedDataset) => void
}) {
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFile = useCallback(async (file: File) => {
    setUploadError(null)
    setUploading(true)
    try {
      const res = await datasetsApi.upload(file)
      onUploaded(res.data)
      onAutoMap?.(res.data)
    } catch (err: any) {
      setUploadError(err?.response?.data?.detail || err?.message || 'Upload failed')
    } finally {
      setUploading(false)
    }
  }, [onUploaded, onAutoMap])

  const onDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file) handleFile(file)
  }, [handleFile])

  return (
    <div className="space-y-3">
      {/* Drop zone */}
      <div
        className={[
          'border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer',
          uploading ? 'border-accent/60 bg-accent/5' : 'border-slate-300 hover:border-accent/50 hover:bg-slate-50',
        ].join(' ')}
        onDragOver={(e) => e.preventDefault()}
        onDrop={onDrop}
        onClick={() => !uploading && inputRef.current?.click()}
      >
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.tsv,.json,.npz,.txt"
          className="hidden"
          onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
        />
        {uploading ? (
          <div className="flex flex-col items-center gap-2">
            <RefreshCw size={24} className="text-accent animate-spin" />
            <p className="text-sm font-medium text-accent">Uploading…</p>
          </div>
        ) : (
          <div className="flex flex-col items-center gap-2">
            <Upload size={24} className="text-slate-400" />
            <p className="text-sm font-medium text-text-primary">
              Drop a file here, or <span className="text-accent underline">browse</span>
            </p>
            <p className="text-xs text-text-muted">CSV, TSV, JSON (array-of-objects), NPZ — up to 50 MB</p>
          </div>
        )}
      </div>

      {uploadError && (
        <div className="flex items-start gap-2 rounded-lg bg-red-50 border border-red-200 px-3 py-2 text-red-700 text-xs">
          <AlertCircle size={14} className="mt-0.5 flex-shrink-0" />
          {uploadError}
        </div>
      )}

      {current && (
        <div className="rounded-lg bg-green-50 border border-green-200 px-3 py-2 flex items-center gap-2 text-xs text-green-800">
          <CheckCircle size={14} className="flex-shrink-0" />
          <span>
            <strong>{current.original_filename}</strong> — {current.n_rows.toLocaleString()} rows × {current.n_cols} cols
          </span>
        </div>
      )}
    </div>
  )
}

// ─── Column Mapping sub-component ────────────────────────────────────────────

function ColumnMappingPanel({
  dataset,
  mapping,
  onChange,
}: {
  dataset: UploadedDataset
  mapping: ColumnMapping
  onChange: (m: Partial<ColumnMapping>) => void
}) {
  const { columns, col_stats, preview } = dataset

  const toggleCoord = (col: string) => {
    const next = mapping.coord_cols.includes(col)
      ? mapping.coord_cols.filter((c) => c !== col)
      : [...mapping.coord_cols, col]
    onChange({ coord_cols: next })
  }

  const toggleOutput = (col: string) => {
    const next = mapping.output_cols.includes(col)
      ? mapping.output_cols.filter((c) => c !== col)
      : [...mapping.output_cols, col]
    onChange({ output_cols: next })
  }

  return (
    <div className="space-y-4">
      <div>
        <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1">Column Mapping</h4>
        <p className="text-xs text-text-secondary">
          Assign each column as a coordinate input, output field, or time axis.
        </p>
      </div>

      {/* Column table */}
      <div className="overflow-x-auto rounded-lg border border-border">
        <table className="w-full text-xs">
          <thead className="bg-slate-50 border-b border-border">
            <tr>
              <th className="text-left px-3 py-2 font-semibold text-text-muted">Column</th>
              <th className="text-right px-3 py-2 font-semibold text-text-muted">Min</th>
              <th className="text-right px-3 py-2 font-semibold text-text-muted">Max</th>
              <th className="text-right px-3 py-2 font-semibold text-text-muted">Mean</th>
              <th className="text-center px-3 py-2 font-semibold text-text-muted">Coord</th>
              <th className="text-center px-3 py-2 font-semibold text-text-muted">Output</th>
              <th className="text-center px-3 py-2 font-semibold text-text-muted">Time</th>
            </tr>
          </thead>
          <tbody>
            {columns.map((col, i) => {
              const stat = col_stats[i]
              const isCoord = mapping.coord_cols.includes(col)
              const isOutput = mapping.output_cols.includes(col)
              const isTime = mapping.time_col === col
              return (
                <tr key={col} className="border-b border-border/50 hover:bg-slate-50/50">
                  <td className="px-3 py-2 font-mono font-medium text-text-primary">{col}</td>
                  <td className="px-3 py-2 text-right font-mono text-text-muted">{stat?.min?.toFixed(4) ?? '—'}</td>
                  <td className="px-3 py-2 text-right font-mono text-text-muted">{stat?.max?.toFixed(4) ?? '—'}</td>
                  <td className="px-3 py-2 text-right font-mono text-text-muted">{stat?.mean?.toFixed(4) ?? '—'}</td>
                  <td className="px-3 py-2 text-center">
                    <input
                      type="checkbox"
                      checked={isCoord}
                      onChange={() => {
                        if (isTime) onChange({ time_col: null })
                        toggleCoord(col)
                      }}
                      className="accent-accent"
                    />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <input
                      type="checkbox"
                      checked={isOutput}
                      onChange={() => toggleOutput(col)}
                      className="accent-accent"
                    />
                  </td>
                  <td className="px-3 py-2 text-center">
                    <input
                      type="radio"
                      name="time_col"
                      checked={isTime}
                      onChange={() => onChange({ time_col: isTime ? null : col })}
                      className="accent-purple-600"
                    />
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>

      {/* Data preview */}
      {preview && preview.length > 0 && (
        <div>
          <h5 className="text-xs font-semibold text-text-muted mb-1">Data Preview (first 5 rows)</h5>
          <div className="overflow-x-auto rounded-lg border border-border">
            <table className="text-xs">
              <thead className="bg-slate-50 border-b border-border">
                <tr>
                  {columns.map((c) => (
                    <th key={c} className="px-3 py-1.5 text-left font-mono text-text-muted whitespace-nowrap">{c}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {preview.slice(0, 5).map((row, i) => (
                  <tr key={i} className="border-b border-border/40">
                    {columns.map((c) => (
                      <td key={c} className="px-3 py-1.5 font-mono text-text-primary whitespace-nowrap">
                        {typeof row[c] === 'number' ? (row[c] as number).toFixed(5) : '—'}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

// ─── Inverse Config sub-component ────────────────────────────────────────────

function InverseConfigPanel({
  config,
  onChange,
  problem,
}: {
  config: InverseConfig
  onChange: (c: Partial<InverseConfig>) => void
  problem?: PhysicsProblem | null
}) {
  const availableParams = problem?.parameters ?? []

  const toggle = (name: string, defaultVal: number) => {
    if (config.params.includes(name)) {
      const guesses = { ...config.initial_guesses }
      delete guesses[name]
      onChange({ params: config.params.filter((p) => p !== name), initial_guesses: guesses })
    } else {
      onChange({
        params: [...config.params, name],
        initial_guesses: { ...config.initial_guesses, [name]: defaultVal },
      })
    }
  }

  const setGuess = (name: string, val: number) => {
    onChange({ initial_guesses: { ...config.initial_guesses, [name]: val } })
  }

  return (
    <div className="space-y-5">
      {/* Parameters to identify */}
      <div>
        <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
          PDE Parameters to Identify
        </h4>
        <p className="text-xs text-text-secondary mb-3">
          Select which physical parameters are unknown. Provide an initial guess for each selected parameter.
        </p>

        {availableParams.length === 0 ? (
          <div className="text-xs text-text-muted italic py-2">
            No parameter definitions found for this problem.
          </div>
        ) : (
          <div className="space-y-2">
            {availableParams.map((param) => {
              const selected = config.params.includes(param.name)
              return (
                <div
                  key={param.name}
                  className={`rounded-lg border px-3 py-2.5 transition-colors cursor-pointer ${
                    selected
                      ? 'border-accent/60 bg-accent/5'
                      : 'border-border bg-white hover:border-accent/30 hover:bg-slate-50'
                  }`}
                  onClick={() => toggle(param.name, param.default)}
                >
                  <div className="flex items-center gap-3">
                    {/* Checkbox indicator */}
                    <div
                      className={`w-4 h-4 rounded flex-shrink-0 flex items-center justify-center border transition-colors ${
                        selected ? 'bg-accent border-accent' : 'border-slate-300'
                      }`}
                    >
                      {selected && (
                        <svg width="10" height="8" viewBox="0 0 10 8" fill="none">
                          <path d="M1 4L3.5 6.5L9 1" stroke="white" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"/>
                        </svg>
                      )}
                    </div>

                    {/* Param info */}
                    <div className="flex-1 min-w-0">
                      <div className="flex items-baseline gap-2">
                        <code className="text-xs font-mono font-semibold text-text-primary">{param.name}</code>
                        {param.unit && (
                          <span className="text-xs text-text-muted">({param.unit})</span>
                        )}
                        <span className="text-xs text-text-secondary truncate">{param.description}</span>
                      </div>
                      <div className="text-xs text-text-muted mt-0.5">
                        default: <span className="font-mono">{param.default}</span>
                        {param.min_val !== undefined && param.max_val !== undefined && (
                          <span className="ml-2">range: [{param.min_val}, {param.max_val}]</span>
                        )}
                      </div>
                    </div>

                    {/* Initial guess input — only when selected */}
                    {selected && (
                      <div
                        className="flex items-center gap-1.5 flex-shrink-0"
                        onClick={(e) => e.stopPropagation()}
                      >
                        <span className="text-xs text-text-muted whitespace-nowrap">Initial guess:</span>
                        <input
                          type="number"
                          value={config.initial_guesses[param.name] ?? param.default}
                          onChange={(e) => setGuess(param.name, parseFloat(e.target.value) || 0)}
                          className="w-24 text-xs font-mono border border-border rounded px-2 py-1 focus:outline-none focus:ring-1 focus:ring-accent bg-white"
                        />
                      </div>
                    )}
                  </div>
                </div>
              )
            })}
          </div>
        )}

        {config.params.length === 0 && availableParams.length > 0 && (
          <p className="text-xs text-amber-600 mt-2 flex items-center gap-1">
            <span>⚠</span> Select at least one parameter to identify.
          </p>
        )}
      </div>

      {/* Solver settings */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="text-xs font-semibold text-text-muted block mb-1">Solver Method</label>
          <select
            value={config.method}
            onChange={(e) => onChange({ method: e.target.value as InverseConfig['method'] })}
            className="w-full text-xs border border-border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-accent bg-white"
          >
            <option value="adam">Adam (fast, general)</option>
            <option value="lbfgs">L-BFGS (smooth, deterministic)</option>
            <option value="eki">Ensemble Kalman Inversion — EKI</option>
            <option value="teki">Tikhonov EKI — TEKI</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold text-text-muted block mb-1">Noise Model</label>
          <select
            value={config.noise_model}
            onChange={(e) => onChange({ noise_model: e.target.value as InverseConfig['noise_model'] })}
            className="w-full text-xs border border-border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-accent bg-white"
          >
            <option value="gaussian">Gaussian (standard)</option>
            <option value="huber">Huber (robust, outliers)</option>
            <option value="cauchy">Cauchy (heavy-tailed)</option>
            <option value="student_t">Student-t</option>
          </select>
        </div>

        <div>
          <label className="text-xs font-semibold text-text-muted block mb-1">
            Observation Points <span className="font-normal text-text-muted">(n_obs)</span>
          </label>
          <input
            type="number"
            value={config.n_obs}
            min={1}
            onChange={(e) => onChange({ n_obs: Math.max(1, parseInt(e.target.value) || 1) })}
            className="w-full text-xs font-mono border border-border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-accent"
          />
        </div>

        <div>
          <label className="text-xs font-semibold text-text-muted block mb-1">
            Noise Std <span className="font-normal text-text-muted">(σ)</span>
          </label>
          <input
            type="number"
            value={config.noise_std}
            min={0}
            step={0.001}
            onChange={(e) => onChange({ noise_std: Math.max(0, parseFloat(e.target.value) || 0) })}
            className="w-full text-xs font-mono border border-border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-accent"
          />
        </div>

        <div>
          <label className="text-xs font-semibold text-text-muted block mb-1">Solver Iterations</label>
          <input
            type="number"
            value={config.n_iters}
            min={1}
            onChange={(e) => onChange({ n_iters: Math.max(1, parseInt(e.target.value) || 1) })}
            className="w-full text-xs font-mono border border-border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-accent"
          />
        </div>

        <div>
          <label className="text-xs font-semibold text-text-muted block mb-1">
            Regularisation λ
          </label>
          <input
            type="number"
            value={config.lambda_reg}
            min={0}
            step={0.0001}
            onChange={(e) => onChange({ lambda_reg: Math.max(0, parseFloat(e.target.value) || 0) })}
            className="w-full text-xs font-mono border border-border rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-accent"
          />
        </div>
      </div>
    </div>
  )
}

// ─── Step 4: Dataset ──────────────────────────────────────────────────────────

function StepDataset() {
  const {
    selectedProblem, customParameters, selectedGeometry,
    problemType,
    uploadedDataset, setUploadedDataset,
    columnMapping, setColumnMapping,
    inverseConfig, setInverseConfig,
  } = useExperimentStore()

  const hints = selectedProblem?.column_hints ?? null

  // Auto-map columns: match uploaded dataset columns against problem hints
  const autoMapFromHints = useCallback((ds: UploadedDataset) => {
    if (!hints) return
    const cols = new Set(ds.columns)
    const coord = hints.coord_cols.filter((c) => cols.has(c))
    const output = hints.output_cols.filter((c) => cols.has(c))
    const time = hints.time_col && cols.has(hints.time_col) ? hints.time_col : null
    if (coord.length > 0 || output.length > 0) {
      setColumnMapping({ coord_cols: coord, output_cols: output, time_col: time })
    }
  }, [hints, setColumnMapping])

  // Apply hints to current mapping (for the "Auto-apply" button when dataset already uploaded)
  const applyHintsToExisting = useCallback(() => {
    if (uploadedDataset) autoMapFromHints(uploadedDataset)
  }, [uploadedDataset, autoMapFromHints])

  const dim = selectedProblem?.dimension
  const nSamples = dim === 'THREE_D' ? 2000 : dim === 'TWO_D' ? 1000 : 500

  const { data: preview, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['dataset-preview', selectedProblem?.id, customParameters, nSamples, selectedGeometry?.id],
    queryFn: () => problemsApi.previewDataset(
      selectedProblem!.id,
      {
        ...customParameters,
        n_samples: nSamples,
        ...(selectedGeometry?.id ? { geometry_id: selectedGeometry.id } : {}),
      }
    ).then((r) => r.data),
    enabled: !!selectedProblem && problemType === 'forward',
    staleTime: 0,
    retry: 1,
  })

  const metadata = (preview?.metadata ?? {}) as Record<string, any>
  const gridSize = metadata.grid_size ?? preview?.n_samples
  const domainBounds = (preview?.domain_bounds ?? metadata.domain_bounds) as Record<string, [number, number]> | undefined

  const valueRange: [number, number] | null = (() => {
    if (!preview?.solution) return null
    const scalars = toScalarArray(preview.solution as any[]).filter(isFinite)
    if (scalars.length === 0) return null
    scalars.sort((a, b) => a - b)
    const lo = scalars[Math.floor(scalars.length * 0.01)]
    const hi = scalars[Math.ceil(scalars.length * 0.99) - 1]
    return [lo, hi]
  })()

  // ── Inverse mode ────────────────────────────────────────────────────────────
  if (problemType === 'inverse') {
    return (
      <div className="space-y-6 max-w-3xl">
        {/* Tip banner */}
        <div className="flex items-start gap-3 rounded-xl bg-purple-50 border border-purple-200 px-4 py-3">
          <FlaskConical size={16} className="text-purple-600 mt-0.5 flex-shrink-0" />
          <div className="text-xs text-purple-800">
            <strong>Inverse mode:</strong> Upload your experimental observations (sensor readings, measurements)
            and configure which PDE parameters to identify. PINNeAPPle will use its inverse problem
            infrastructure (EKI, L-BFGS, Adam) to recover the unknown parameters while enforcing physics.
          </div>
        </div>

        {/* Expected format + upload */}
        <Card>
          <CardHeader>
            <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <Upload size={14} /> Observation Data
            </h4>
          </CardHeader>
          <CardContent className="space-y-4">
            {hints && (
              <ExpectedFormatBox
                hints={hints}
                onApplyHints={uploadedDataset ? applyHintsToExisting : undefined}
              />
            )}
            <DatasetUploadZone
              onUploaded={setUploadedDataset}
              current={uploadedDataset}
              onAutoMap={autoMapFromHints}
            />
          </CardContent>
        </Card>

        {/* Column mapping */}
        {uploadedDataset && (
          <Card>
            <CardHeader>
              <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
                <GitMerge size={14} /> Column Mapping
              </h4>
            </CardHeader>
            <CardContent>
              <ColumnMappingPanel
                dataset={uploadedDataset}
                mapping={columnMapping}
                onChange={setColumnMapping}
              />
            </CardContent>
          </Card>
        )}

        {/* Inverse config */}
        <Card>
          <CardHeader>
            <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <Search size={14} /> Inverse Problem Configuration
            </h4>
          </CardHeader>
          <CardContent>
            <InverseConfigPanel config={inverseConfig} onChange={setInverseConfig} problem={selectedProblem} />
          </CardContent>
        </Card>
      </div>
    )
  }

  // ── Forward mode — reference dataset preview (original) ─────────────────────
  return (
    <div className="space-y-6">
      {/* Expected column format (always shown for forward) */}
      {hints && !uploadedDataset && (
        <ExpectedFormatBox hints={hints} />
      )}

      {/* Optional upload banner */}
      {!uploadedDataset ? (
        <div className="flex items-start gap-3 rounded-xl bg-blue-50 border border-blue-200 px-4 py-3">
          <Upload size={14} className="text-blue-600 mt-0.5 flex-shrink-0" />
          <div className="text-xs text-blue-800 flex-1">
            <strong>Optional:</strong> Upload your own dataset (CSV, NPZ, JSON) to train with real data
            alongside the physics loss. Columns will be auto-mapped based on the format above.{' '}
            <button
              className="underline font-medium"
              onClick={() => {
                const input = document.createElement('input')
                input.type = 'file'
                input.accept = '.csv,.tsv,.json,.npz,.txt'
                input.onchange = async (e) => {
                  const file = (e.target as HTMLInputElement).files?.[0]
                  if (file) {
                    try {
                      const res = await datasetsApi.upload(file)
                      setUploadedDataset(res.data)
                      autoMapFromHints(res.data)
                    } catch { /* ignore */ }
                  }
                }
                input.click()
              }}
            >
              Upload a file
            </button>
          </div>
        </div>
      ) : (
        <div className="flex items-center gap-3 rounded-xl bg-green-50 border border-green-200 px-4 py-3">
          <CheckCircle size={14} className="text-green-600 flex-shrink-0" />
          <div className="text-xs text-green-800 flex-1">
            Using uploaded dataset: <strong>{uploadedDataset.original_filename}</strong>
            &nbsp;({uploadedDataset.n_rows.toLocaleString()} rows)
            {hints && (
              <button onClick={applyHintsToExisting} className="ml-2 underline text-green-700 hover:text-green-900">
                Re-apply hints
              </button>
            )}
          </div>
          <button onClick={() => setUploadedDataset(null)} className="text-green-600 hover:text-green-800">
            <X size={14} />
          </button>
        </div>
      )}

      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-text-primary">Reference Dataset Preview</h3>
          <p className="text-xs text-text-muted mt-0.5">Analytical or high-fidelity reference solution for benchmarking</p>
        </div>
        <Button
          variant="outline"
          size="sm"
          leftIcon={isFetching ? <RefreshCw size={14} className="animate-spin" /> : <Database size={14} />}
          loading={isFetching}
          onClick={() => refetch()}
        >
          {isFetching ? 'Generating…' : 'Regenerate'}
        </Button>
      </div>

      {/* Error state */}
      {error && (
        <div className="rounded-lg bg-red-50 border border-red-200 p-4 text-red-700">
          <p className="font-medium text-sm">Failed to generate dataset preview</p>
          <p className="text-xs mt-1 text-red-600">{(error as any)?.message || 'Unknown error'}</p>
          <button onClick={() => refetch()} className="mt-2 text-xs underline text-red-600 hover:text-red-700">
            Try again
          </button>
        </div>
      )}

      {isLoading && !error && <PageLoading label="Generating dataset preview…" />}

      {preview && !isLoading && (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          {/* Stats */}
          <Card>
            <CardHeader>
              <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider">Dataset Statistics</h4>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                <div>
                  <div className="text-xs text-text-muted">Total Samples</div>
                  <div className="text-lg font-mono font-bold text-accent">{preview.n_samples?.toLocaleString()}</div>
                </div>
                {gridSize && gridSize !== preview.n_samples && (
                  <div>
                    <div className="text-xs text-text-muted">Grid Size</div>
                    <div className="text-sm font-mono text-text-primary">{gridSize}</div>
                  </div>
                )}
                <div>
                  <div className="text-xs text-text-muted">Solution Range</div>
                  <div className="text-sm font-mono text-text-primary">
                    {valueRange
                      ? `[${valueRange[0].toFixed(4)}, ${valueRange[1].toFixed(4)}]`
                      : '—'}
                  </div>
                </div>
                {domainBounds && Object.entries(domainBounds).map(([k, v]: [string, any]) => (
                  <div key={k}>
                    <div className="text-xs text-text-muted capitalize">{k} bounds</div>
                    <div className="text-sm font-mono text-text-primary">[{v[0].toFixed(3)}, {v[1].toFixed(3)}]</div>
                  </div>
                ))}
                {metadata.dim && (
                  <div>
                    <div className="text-xs text-text-muted">Dimensionality</div>
                    <Badge variant="info" size="sm">{metadata.dim}D</Badge>
                  </div>
                )}
                {preview.description && (
                  <p className="text-xs text-text-secondary leading-relaxed">{preview.description}</p>
                )}
              </div>
            </CardContent>
          </Card>

          {/* Chart */}
          <Card className="lg:col-span-2">
            <CardHeader>
              <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider">Solution Preview</h4>
            </CardHeader>
            <CardContent>
              <DatasetChart preview={preview} />
            </CardContent>
          </Card>
        </div>
      )}

      {/* Column mapping for uploaded dataset */}
      {uploadedDataset && (
        <Card>
          <CardHeader>
            <h4 className="text-sm font-semibold text-text-primary flex items-center gap-2">
              <GitMerge size={14} /> Column Mapping
            </h4>
          </CardHeader>
          <CardContent>
            <ColumnMappingPanel
              dataset={uploadedDataset}
              mapping={columnMapping}
              onChange={setColumnMapping}
            />
          </CardContent>
        </Card>
      )}
    </div>
  )
}

// ─── Step 5: Architecture Selection ──────────────────────────────────────────

const ALL_FAMILIES: ArchitectureFamily[] = [
  'PINN', 'NEURAL_OPERATOR', 'TRANSFORMER', 'RECURRENT', 'CONVOLUTIONAL',
  'CONTINUOUS', 'GRAPH', 'AUTOENCODER', 'RESERVOIR', 'ROM', 'CLASSICAL', 'PHYSICS_AWARE',
]

function StepArchitectures() {
  const { selectedProblem, selectedArchitectures, toggleArchitecture, selectAllArchitectures, clearArchitectures, problemType } = useExperimentStore()
  const [search, setSearch] = useState('')
  const [familyFilter, setFamilyFilter] = useState<ArchitectureFamily | null>(null)
  const [detailArch, setDetailArch] = useState<ArchitectureInfo | null>(null)

  const { data: architectures, isLoading } = useQuery({
    queryKey: ['architectures'],
    queryFn: () => architecturesApi.list().then((r) => r.data),
  })

  // Architectures that only make sense for 1-D time-series inputs.
  // They are hidden when the selected problem is 2-D or 3-D.
  const TEMPORAL_ONLY_IDS = new Set([
    'arima', 'var',
    'kalman_filter', 'extended_kalman_filter',
    'unscented_kalman_filter', 'ensemble_kalman_filter',
  ])

  const filtered = useMemo(() => {
    if (!architectures) return []
    const dim = selectedProblem?.dimension
    return architectures.filter((a) => {
      // Dimension-based visibility
      if (dim === 'THREE_D' && !a.supports_3d) return false
      if (dim === 'TWO_D' && TEMPORAL_ONLY_IDS.has(a.id)) return false
      // ONE_D: show everything

      // Problem-type visibility: inverse-only architectures only appear in inverse mode
      if (a.inverse_only && problemType !== 'inverse') return false

      if (familyFilter && a.family !== familyFilter) return false
      if (search) {
        const q = search.toLowerCase()
        return (
          a.name.toLowerCase().includes(q) ||
          a.description.toLowerCase().includes(q) ||
          a.family.toLowerCase().includes(q)
        )
      }
      return true
    })
  }, [architectures, familyFilter, search, selectedProblem, problemType])

  const familyCounts = useMemo(() => {
    if (!architectures) return {} as Record<string, number>
    const dim = selectedProblem?.dimension
    return ALL_FAMILIES.reduce<Record<string, number>>((acc, f) => {
      acc[f] = architectures.filter((a) => {
        if (a.family !== f) return false
        if (dim === 'THREE_D' && !a.supports_3d) return false
        if (dim === 'TWO_D' && TEMPORAL_ONLY_IDS.has(a.id)) return false
        if (a.inverse_only && problemType !== 'inverse') return false
        return true
      }).length
      return acc
    }, {})
  }, [architectures, selectedProblem, problemType])

  const totalCompatible = useMemo(() => {
    if (!architectures) return 0
    const dim = selectedProblem?.dimension
    return architectures.filter((a) => {
      if (dim === 'THREE_D' && !a.supports_3d) return false
      if (dim === 'TWO_D' && TEMPORAL_ONLY_IDS.has(a.id)) return false
      if (a.inverse_only && problemType !== 'inverse') return false
      return true
    }).length
  }, [architectures, selectedProblem, problemType])

  const selectedIds = new Set(selectedArchitectures.map((a) => a.id))

  if (isLoading) return <PageLoading label="Loading architectures…" />

  return (
    <div className="space-y-4">
      {/* Inverse mode banner */}
      {problemType === 'inverse' && (
        <div className="flex items-start gap-3 rounded-xl bg-purple-50 border border-purple-200 px-4 py-3 text-xs text-purple-800">
          <Search size={14} className="text-purple-600 mt-0.5 flex-shrink-0" />
          <span>
            <strong>Inverse mode:</strong> The <code>inverse_pinn</code> architecture is recommended for
            parameter identification. Other PINNs also work — they will be wrapped with PINNeAPPle's
            InverseProblemSolver during training.
          </span>
        </div>
      )}

      {/* Controls */}
      <div className="flex items-center gap-2 flex-wrap">
        <SearchInput value={search} onChange={setSearch} placeholder="Search architectures…" className="flex-1 min-w-0" />
        <div className="flex items-center gap-2 flex-shrink-0">
          <Badge variant="info" size="md">
            {selectedArchitectures.length} selected
          </Badge>
          <Button
            variant="outline"
            size="sm"
            onClick={() => filtered.length > 0 && selectAllArchitectures(filtered)}
          >
            Select All
          </Button>
          {selectedArchitectures.length > 0 && (
            <Button variant="ghost" size="sm" onClick={clearArchitectures}>
              Clear
            </Button>
          )}
        </div>
      </div>

      {/* Family filter tabs — horizontal scroll on mobile */}
      <div className="flex gap-2 overflow-x-auto pb-1 flex-nowrap [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
        <FilterChip label="All" active={familyFilter === null} onClick={() => setFamilyFilter(null)} count={totalCompatible} />
        {ALL_FAMILIES.filter((f) => (familyCounts[f] ?? 0) > 0).map((f) => {
          const cfg = FAMILY_CONFIG[f]
          return (
            <FilterChip
              key={f}
              label={cfg.label}
              active={familyFilter === f}
              onClick={() => setFamilyFilter(familyFilter === f ? null : f)}
              count={familyCounts[f]}
              color={cfg.color}
            />
          )
        })}
      </div>

      {/* Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-3 max-h-[55vh] md:max-h-[480px] overflow-y-auto pr-1">
        {filtered.map((arch) => (
          <ArchitectureCard
            key={arch.id}
            architecture={arch}
            selected={selectedIds.has(arch.id)}
            onClick={() => toggleArchitecture(arch)}
            onInfo={() => setDetailArch(arch)}
            inverseRecommended={problemType === 'inverse' && arch.inverse_only}
          />
        ))}
        {filtered.length === 0 && (
          <div className="col-span-full text-center py-10 space-y-2">
            <p className="text-text-muted text-sm">No architectures match your filters</p>
            {selectedProblem?.dimension === 'THREE_D' && (
              <p className="text-text-muted text-xs">
                3D problems require architectures with native 3D support.{' '}
                <button className="text-accent underline" onClick={() => setFamilyFilter(null)}>
                  Show all compatible
                </button>
              </p>
            )}
          </div>
        )}
      </div>

      {/* Architecture detail modal */}
      <ArchitectureDetailModal architecture={detailArch} onClose={() => setDetailArch(null)} />

      {/* Selected summary */}
      {selectedArchitectures.length > 0 && (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="bg-slate-50 border border-slate-200 rounded-xl p-4">
          <h4 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Selected Architectures</h4>
          <div className="flex flex-wrap gap-2">
            {selectedArchitectures.map((a) => {
              const cfg = FAMILY_CONFIG[a.family]
              return (
                <Badge
                  key={a.id}
                  variant={cfg.variant}
                  size="sm"
                  style={{ color: cfg.color, borderColor: `${cfg.color}30`, backgroundColor: `${cfg.color}10` } as React.CSSProperties}
                >
                  {a.name}
                </Badge>
              )
            })}
          </div>
        </motion.div>
      )}
    </div>
  )
}

// ─── Step 6: Training Config ──────────────────────────────────────────────────

function StepTrainingConfig() {
  const { trainingConfig, setTrainingConfig, selectedProblem, selectedArchitectures } = useExperimentStore()

  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
      <div className="space-y-6">
        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-text-primary flex items-center gap-1.5">
              <Settings size={14} className="text-accent" />
              Training Parameters
            </h3>
          </CardHeader>
          <CardContent className="space-y-5">
            <Slider
              label="Epochs"
              value={trainingConfig.epochs}
              min={100}
              max={50000}
              step={100}
              onChange={(v) => setTrainingConfig({ epochs: Math.round(v) })}
              formatValue={(v) => Math.round(v).toLocaleString()}
            />
            <Slider
              label="Learning Rate"
              value={trainingConfig.learning_rate}
              min={1e-5}
              max={1e-1}
              step={1e-5}
              onChange={(v) => setTrainingConfig({ learning_rate: v })}
              formatValue={(v) => v.toExponential(2)}
            />
            <Slider
              label="Batch Size"
              value={trainingConfig.batch_size}
              min={64}
              max={8192}
              step={64}
              onChange={(v) => setTrainingConfig({ batch_size: Math.round(v) })}
              formatValue={(v) => Math.round(v).toLocaleString()}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-text-primary flex items-center gap-1.5">
              <Database size={14} className="text-accent" />
              Training Points
            </h3>
          </CardHeader>
          <CardContent className="space-y-5">
            <Slider
              label="Collocation Points"
              value={trainingConfig.n_train_col}
              min={1000}
              max={100000}
              step={1000}
              onChange={(v) => setTrainingConfig({ n_train_col: Math.round(v) })}
              formatValue={(v) => Math.round(v).toLocaleString()}
              description="Interior domain points for PDE residual"
            />
            <Slider
              label="Boundary Points"
              value={trainingConfig.n_train_bc}
              min={100}
              max={10000}
              step={100}
              onChange={(v) => setTrainingConfig({ n_train_bc: Math.round(v) })}
              formatValue={(v) => Math.round(v).toLocaleString()}
            />
            <Slider
              label="Initial Condition Points"
              value={trainingConfig.n_train_ic}
              min={100}
              max={10000}
              step={100}
              onChange={(v) => setTrainingConfig({ n_train_ic: Math.round(v) })}
              formatValue={(v) => Math.round(v).toLocaleString()}
            />
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-text-primary">Loss Weights</h3>
          </CardHeader>
          <CardContent className="space-y-5">
            {(['pde', 'bc', 'ic', 'data'] as const).map((key) => (
              <Slider
                key={key}
                label={`${key.toUpperCase()} Weight`}
                value={trainingConfig.loss_weights[key]}
                min={0}
                max={100}
                step={0.5}
                onChange={(v) => setTrainingConfig({ loss_weights: { ...trainingConfig.loss_weights, [key]: v } })}
                formatValue={(v) => v.toFixed(1)}
              />
            ))}
          </CardContent>
        </Card>
      </div>

      {/* Right: summary */}
      <div className="space-y-4">
        <Card>
          <CardHeader>
            <h3 className="text-sm font-semibold text-text-primary flex items-center gap-1.5">
              <BarChart3 size={14} className="text-accent" />
              Experiment Summary
            </h3>
          </CardHeader>
          <CardContent>
            <div className="space-y-3 text-sm">
              <div>
                <div className="text-xs text-text-muted mb-1">Problem</div>
                <div className="text-text-primary font-medium">{selectedProblem?.name ?? '—'}</div>
              </div>
              <div>
                <div className="text-xs text-text-muted mb-1">Architectures ({selectedArchitectures.length})</div>
                <div className="flex flex-wrap gap-1">
                  {selectedArchitectures.slice(0, 5).map((a) => (
                    <Badge key={a.id} variant="default" size="xs">{a.name}</Badge>
                  ))}
                  {selectedArchitectures.length > 5 && (
                    <Badge variant="default" size="xs">+{selectedArchitectures.length - 5} more</Badge>
                  )}
                </div>
              </div>
              <div className="border-t border-border pt-3 space-y-1.5">
                {[
                  ['Epochs', trainingConfig.epochs.toLocaleString()],
                  ['Learning Rate', trainingConfig.learning_rate.toExponential(2)],
                  ['Batch Size', trainingConfig.batch_size.toLocaleString()],
                  ['Collocation Pts', trainingConfig.n_train_col.toLocaleString()],
                  ['BC Points', trainingConfig.n_train_bc.toLocaleString()],
                  ['IC Points', trainingConfig.n_train_ic.toLocaleString()],
                ].map(([label, value]) => (
                  <div key={label} className="flex justify-between text-xs">
                    <span className="text-text-muted">{label}</span>
                    <span className="font-mono text-text-primary">{value}</span>
                  </div>
                ))}
              </div>
              <div className="border-t border-border pt-3">
                <div className="text-xs text-text-muted mb-1.5">Loss Weights</div>
                {Object.entries(trainingConfig.loss_weights).map(([k, v]) => (
                  <div key={k} className="mb-1">
                    <div className="flex justify-between text-xs mb-0.5">
                      <span className="text-text-muted uppercase">{k}</span>
                      <span className="font-mono text-text-primary">{v.toFixed(1)}</span>
                    </div>
                    <ProgressBar value={v} max={100} size="sm" />
                  </div>
                ))}
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Estimated time */}
        <Card>
          <CardContent className="py-3">
            <div className="text-xs text-text-muted mb-1">Estimated training time</div>
            <div className="text-lg font-mono font-bold text-warning">
              ~{Math.round((trainingConfig.epochs * selectedArchitectures.length) / 1000)} min
            </div>
            <div className="text-xs text-text-muted mt-0.5">Based on average GPU throughput</div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

// ─── Step 7: Launch ───────────────────────────────────────────────────────────

function StepLaunch() {
  const { selectedProblem, selectedGeometry, selectedArchitectures, trainingConfig } = useExperimentStore()
  const { DOMAIN_CONFIG: DC, DIMENSION_LABELS: DL } = { DOMAIN_CONFIG, DIMENSION_LABELS }

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="text-center">
        <div className="w-16 h-16 rounded-2xl bg-accent/10 border border-accent/20 flex items-center justify-center mx-auto mb-4">
          <Rocket size={28} className="text-accent" />
        </div>
        <h2 className="text-xl font-bold text-text-primary">Ready to Launch</h2>
        <p className="text-sm text-text-secondary mt-1">Review your configuration before starting the experiment</p>
      </div>

      <Card>
        <CardHeader>
          <h3 className="text-sm font-semibold text-text-primary">Experiment Configuration</h3>
        </CardHeader>
        <CardContent className="space-y-4">
          {/* Problem */}
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Problem</div>
            {selectedProblem ? (
              <div className="flex items-center gap-3">
                <span className="text-xl">{DC[selectedProblem.domain]?.icon}</span>
                <div>
                  <div className="text-sm font-semibold text-text-primary">{selectedProblem.name}</div>
                  <div className="text-xs text-text-muted">{DC[selectedProblem.domain]?.label} • {DL[selectedProblem.dimension]}</div>
                </div>
              </div>
            ) : <span className="text-danger text-sm">No problem selected</span>}
          </div>

          {/* Geometry */}
          {selectedGeometry && (
            <div>
              <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Geometry</div>
              <div className="text-sm text-text-primary">{selectedGeometry.name}</div>
            </div>
          )}

          {/* Architectures */}
          <div>
            <div className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">
              Architectures ({selectedArchitectures.length})
            </div>
            <div className="flex flex-wrap gap-2">
              {selectedArchitectures.map((a) => (
                <Badge key={a.id} variant="info" size="sm">
                  {a.name}
                </Badge>
              ))}
              {selectedArchitectures.length === 0 && (
                <span className="text-danger text-sm">No architectures selected</span>
              )}
            </div>
          </div>

          {/* Config summary */}
          <div className="grid grid-cols-2 gap-2 text-xs pt-2 border-t border-border">
            {[
              ['Epochs', trainingConfig.epochs.toLocaleString()],
              ['LR', trainingConfig.learning_rate.toExponential(2)],
              ['Batch Size', trainingConfig.batch_size.toLocaleString()],
              ['Collocation Pts', trainingConfig.n_train_col.toLocaleString()],
            ].map(([l, v]) => (
              <div key={l} className="flex justify-between bg-slate-50 border border-slate-200 rounded px-2 py-1.5">
                <span className="text-text-muted">{l}</span>
                <span className="font-mono text-text-primary">{v}</span>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {(!selectedProblem || selectedArchitectures.length === 0) && (
        <p className="text-xs text-danger text-center">
          {!selectedProblem ? 'Select a problem first.' : 'Select at least one architecture.'}
        </p>
      )}
    </div>
  )
}

// ─── Main Wizard Page ─────────────────────────────────────────────────────────

export function ExperimentSetupPage() {
  const navigate = useNavigate()
  const {
    currentStep, nextStep, prevStep, setCurrentStep, selectedProblem, selectedArchitectures, buildExperimentConfig,
  } = useExperimentStore()

  const visibleSteps = useMemo<WizardStep[]>(() => {
    return WIZARD_STEPS.filter((s) => {
      if (s === 'geometry' && selectedProblem && !selectedProblem.requires_geometry) return false
      return true
    })
  }, [selectedProblem])

  const stepObjects = visibleSteps.map((s) => ({ id: s, label: STEP_LABELS[s] }))
  const currentIdx = visibleSteps.indexOf(currentStep)
  const completedSteps = visibleSteps.slice(0, currentIdx)

  const isLastStep = currentStep === 'launch'
  const isFirstStep = currentStep === 'problem_type'

  const canProceed = useMemo(() => {
    switch (currentStep) {
      case 'problem': return !!selectedProblem
      case 'architectures': return selectedArchitectures.length > 0
      default: return true
    }
  }, [currentStep, selectedProblem, selectedArchitectures])

  const launchMutation = useMutation({
    mutationFn: async () => {
      const config = buildExperimentConfig()
      if (!config) throw new Error('Invalid config')
      const res = await experimentsApi.create(config)
      return res.data
    },
    onSuccess: (data) => {
      navigate(`/experiment/${data.experiment_id}/training`)
    },
  })

  const stepIcons: Record<WizardStep, React.ReactNode> = {
    problem_type: <FlaskConical size={14} />,
    problem: <Filter size={14} />,
    parameters: <Settings size={14} />,
    geometry: <Layers size={14} />,
    dataset: <Database size={14} />,
    architectures: <Cpu size={14} />,
    training_config: <BarChart3 size={14} />,
    launch: <Rocket size={14} />,
  }

  const renderStep = () => {
    switch (currentStep) {
      case 'problem_type': return <StepProblemType />
      case 'problem': return <StepProblem />
      case 'parameters': return <StepParameters />
      case 'geometry': return <StepGeometry />
      case 'dataset': return <StepDataset />
      case 'architectures': return <StepArchitectures />
      case 'training_config': return <StepTrainingConfig />
      case 'launch': return <StepLaunch />
    }
  }

  const problemType = useExperimentStore((s) => s.problemType)

  const stepHeaderConfig: Record<WizardStep, { title: string; description: string; icon: React.ReactNode }> = {
    problem_type: { title: 'Problem Type', description: 'Choose whether this is a forward or inverse physics problem', icon: <FlaskConical size={18} /> },
    problem: { title: 'Select Problem', description: 'Choose a physics problem to benchmark neural architectures on', icon: <Filter size={18} /> },
    parameters: { title: 'Configure Parameters', description: 'Adjust the problem parameters (e.g. Reynolds number, frequency)', icon: <Settings size={18} /> },
    geometry: { title: 'Select Geometry', description: 'Choose a 3D geometry preset for the spatial domain', icon: <Layers size={18} /> },
    dataset: {
      title: problemType === 'inverse' ? 'Observations & Inverse Config' : 'Dataset Preview',
      description: problemType === 'inverse'
        ? 'Upload sensor observations and configure the inverse problem solver'
        : 'Preview the reference dataset used for evaluation',
      icon: <Database size={18} />,
    },
    architectures: { title: 'Select Architectures', description: 'Choose one or more neural network architectures to compare', icon: <Zap size={18} /> },
    training_config: { title: 'Training Configuration', description: 'Set training hyperparameters and loss weights', icon: <Settings size={18} /> },
    launch: { title: 'Launch Experiment', description: 'Review and start the benchmarking experiment', icon: <Rocket size={18} /> },
  }

  const header = stepHeaderConfig[currentStep]

  return (
    <div className="min-h-full flex flex-col bg-background">
      {/* Step indicator */}
      <div className="px-4 sm:px-8 pt-4 sm:pt-6 pb-4 border-b border-border bg-white overflow-x-auto">
        <StepIndicator
          steps={stepObjects}
          currentStep={currentStep}
          completedSteps={completedSteps}
          onStepClick={(s) => {
            const idx = visibleSteps.indexOf(s as WizardStep)
            const curr = visibleSteps.indexOf(currentStep)
            if (idx < curr) setCurrentStep(s as WizardStep)
          }}
        />
      </div>

      {/* Step content */}
      <div className="flex-1 px-4 sm:px-8 py-4 sm:py-6">
        {/* Step header */}
        <div className="mb-6">
          <div className="flex items-center gap-2.5 mb-1">
            <div className="w-8 h-8 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center text-accent">
              {header.icon}
            </div>
            <h2 className="text-lg font-bold text-text-primary">{header.title}</h2>
          </div>
          <p className="text-sm text-text-secondary ml-10">{header.description}</p>
        </div>

        <AnimatePresence mode="wait">
          <motion.div
            key={currentStep}
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -20 }}
            transition={{ duration: 0.2 }}
          >
            {renderStep()}
          </motion.div>
        </AnimatePresence>
      </div>

      {/* Navigation — always visible */}
      <div className="px-4 sm:px-8 py-4 border-t border-border flex items-center justify-between bg-white sticky bottom-0 z-10">
        <Button
          variant="ghost"
          size="md"
          leftIcon={<ChevronLeft size={16} />}
          onClick={prevStep}
          disabled={isFirstStep}
        >
          Back
        </Button>
        <div className="text-xs text-text-muted">
          Step {currentIdx + 1} of {visibleSteps.length}
        </div>
        {isLastStep ? (
          <Button
            variant="primary"
            size="md"
            leftIcon={<Play size={16} />}
            loading={launchMutation.isPending}
            onClick={() => launchMutation.mutate()}
            disabled={!selectedProblem || selectedArchitectures.length === 0 || launchMutation.isPending}
          >
            {launchMutation.isPending ? 'Launching…' : 'Launch Experiment'}
          </Button>
        ) : (
          <Button
            variant="primary"
            size="md"
            rightIcon={<ChevronRight size={16} />}
            onClick={nextStep}
            disabled={!canProceed}
          >
            Continue
          </Button>
        )}
      </div>

      {launchMutation.isError && (
        <div className="px-4 sm:px-8 py-3 bg-red-50 border-t border-red-200">
          <p className="text-sm text-danger">
            Failed to launch experiment. Make sure the backend is running.
          </p>
        </div>
      )}
    </div>
  )
}
