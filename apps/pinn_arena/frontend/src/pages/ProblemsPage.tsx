import React, { useState, useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import { motion, AnimatePresence } from 'framer-motion'
import { X, Layers, Star, Cpu, ExternalLink, PlayCircle, ArrowRight, Search, SlidersHorizontal, ChevronDown } from 'lucide-react'
import { problemsApi } from '../lib/api'
import type { PhysicsProblem, PhysicsArea, Dimension, Difficulty, ProblemType } from '../types'
import { ProblemCard, DOMAIN_CONFIG, DIMENSION_LABELS, DIFFICULTY_CONFIG } from '../components/ProblemCard'
import { SearchInput } from '../components/ui/SearchInput'
import { FilterChip } from '../components/ui/FilterChip'
import { Button } from '../components/ui/Button'
import { Badge } from '../components/ui/Badge'
import { PageLoading } from '../components/ui/LoadingSpinner'
import { useExperimentStore } from '../store/experimentStore'
import { MathEq } from '../components/MathEq'

const DOMAINS: PhysicsArea[] = [
  'FLUID_MECHANICS', 'SOLID_MECHANICS', 'ELECTROMAGNETISM',
  'QUANTUM_MECHANICS', 'THERMODYNAMICS', 'WAVE_PHYSICS', 'NONLINEAR_DYNAMICS',
]
const DIMENSIONS: Dimension[] = ['ONE_D', 'TWO_D', 'THREE_D']
const DIFFICULTIES: Difficulty[] = ['BEGINNER', 'INTERMEDIATE', 'ADVANCED']

// ─── Detail panel ─────────────────────────────────────────────────────────────

function ProblemDetailPanel({
  problem,
  onClose,
  onStartExperiment,
}: {
  problem: PhysicsProblem
  onClose: () => void
  onStartExperiment: (p: PhysicsProblem, type: ProblemType) => void
}) {
  const domain = DOMAIN_CONFIG[problem.domain]
  const diff = DIFFICULTY_CONFIG[problem.difficulty]
  const [problemType, setProblemType] = useState<ProblemType>('forward')

  return (
    <motion.div
      initial={{ x: 40, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      exit={{ x: 40, opacity: 0 }}
      transition={{ type: 'spring', stiffness: 300, damping: 28 }}
      className="fixed inset-0 z-50 overflow-y-auto bg-white md:static md:inset-auto md:z-auto md:w-96 md:flex-shrink-0 md:h-[calc(100vh-6.5rem)] md:overflow-y-auto md:rounded-l-2xl md:border-l md:border-border md:shadow-xl"
    >
      {/* Header */}
      <div className="sticky top-0 bg-white border-b border-border p-5 z-10">
        <div className="flex items-start justify-between gap-3">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-2">
              <span className="text-lg">{domain.icon}</span>
              <Badge variant={domain.variant} size="sm" style={{ color: domain.color, borderColor: `${domain.color}30`, backgroundColor: `${domain.color}10` } as React.CSSProperties}>
                {domain.label}
              </Badge>
              <Badge variant="default" size="sm">{DIMENSION_LABELS[problem.dimension]}</Badge>
            </div>
            <h2 className="text-base font-bold text-text-primary leading-snug">{problem.name}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg hover:bg-slate-100 text-slate-400 hover:text-text-primary transition-colors flex-shrink-0"
          >
            <X size={16} />
          </button>
        </div>
      </div>

      <div className="p-5 space-y-5">
        <p className="text-sm text-text-secondary leading-relaxed">{problem.description}</p>

        {/* Difficulty */}
        <div className="flex items-center gap-2">
          <div className="flex gap-1">
            {Array.from({ length: 3 }).map((_, i) => (
              <Star key={i} size={13} className={i < diff.stars ? 'text-amber-500 fill-amber-500' : 'text-slate-200'} />
            ))}
          </div>
          <span className="text-sm text-text-secondary">{diff.label}</span>
          {problem.requires_geometry && (
            <Badge variant="info" size="xs" className="ml-auto">Requires 3D Geometry</Badge>
          )}
        </div>

        {/* Equations */}
        {problem.equations.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <Layers size={12} /> Governing Equations
            </h3>
            <div className="space-y-1.5">
              {(problem.latex_equations ?? problem.equations).map((eq: string, i: number) => {
                const isLatex = !!problem.latex_equations
                return (
                  <div
                    key={i}
                    className="flex items-center gap-3 bg-slate-50 border border-slate-200 rounded-xl px-4 py-3 overflow-x-auto"
                  >
                    <span className="text-[10px] font-semibold text-slate-400 select-none flex-shrink-0 tabular-nums w-4 text-right self-start pt-1">
                      {i + 1}
                    </span>
                    {isLatex ? (
                      <MathEq latex={eq} display={true} className="min-w-0" />
                    ) : (
                      <span
                        className="text-sm text-slate-800 leading-relaxed tracking-wide"
                        style={{ fontFamily: "'Georgia', 'Times New Roman', serif", letterSpacing: '0.01em' }}
                      >
                        {eq}
                      </span>
                    )}
                  </div>
                )
              })}
            </div>
          </div>
        )}

        {/* Boundary & Initial Conditions */}
        <div className="grid grid-cols-1 gap-3">
          {problem.boundary_conditions && (
            <div>
              <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Boundary Conditions</h3>
              <div className="text-sm bg-green-50 border border-green-200 rounded-xl px-4 py-3 text-green-800 leading-relaxed" style={{ fontFamily: "'Georgia', 'Times New Roman', serif" }}>{problem.boundary_conditions}</div>
            </div>
          )}
          {problem.initial_conditions && (
            <div>
              <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-1.5">Initial Conditions</h3>
              <div className="text-sm bg-amber-50 border border-amber-200 rounded-xl px-4 py-3 text-amber-800 leading-relaxed" style={{ fontFamily: "'Georgia', 'Times New Roman', serif" }}>{problem.initial_conditions}</div>
            </div>
          )}
        </div>

        {/* Parameters */}
        {problem.parameters.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2">Parameters</h3>
            <div className="space-y-1.5">
              {problem.parameters.map((p) => (
                <div key={p.name} className="flex items-center justify-between gap-3 text-sm py-1.5 border-b border-slate-100">
                  <div className="flex-1 min-w-0">
                    <span className="font-mono text-accent text-xs">{p.name}</span>
                    <p className="text-xs text-text-muted mt-0.5 truncate">{p.description}</p>
                  </div>
                  <div className="text-right flex-shrink-0">
                    <div className="text-xs font-mono text-text-primary">{p.default}</div>
                    <div className="text-[10px] text-text-muted">{p.unit}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Applications */}
        {problem.applications.length > 0 && (
          <div>
            <h3 className="text-xs font-semibold text-text-muted uppercase tracking-wider mb-2 flex items-center gap-1.5">
              <ExternalLink size={12} /> Applications
            </h3>
            <ul className="space-y-1.5">
              {problem.applications.map((app, i) => (
                <li key={i} className="text-xs text-text-secondary flex items-start gap-2">
                  <span className="text-accent mt-0.5 flex-shrink-0">•</span>{app}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Tags */}
        {problem.tags.length > 0 && (
          <div className="flex flex-wrap gap-1.5">
            {problem.tags.map((tag) => (
              <span key={tag} className="text-xs bg-slate-100 text-slate-500 border border-slate-200 px-2 py-1 rounded-full">{tag}</span>
            ))}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="sticky bottom-0 bg-white border-t border-border p-4 space-y-3">
        <p className="text-[11px] font-semibold text-text-muted uppercase tracking-wider">Problem Type</p>
        <div className="grid grid-cols-2 gap-2">
          <button
            onClick={() => setProblemType('forward')}
            className={['flex flex-col items-start gap-1 rounded-lg border-2 px-3 py-2.5 text-left transition-all',
              problemType === 'forward' ? 'border-accent bg-accent/5' : 'border-border hover:border-accent/40 bg-white'].join(' ')}
          >
            <div className="flex items-center gap-1.5">
              <div className={`w-4 h-4 rounded flex items-center justify-center ${problemType === 'forward' ? 'bg-accent text-white' : 'bg-slate-100 text-slate-500'}`}>
                <ArrowRight size={10} />
              </div>
              <span className="text-xs font-semibold text-text-primary">Forward</span>
              {problemType === 'forward' && <span className="text-[9px] font-bold text-green-600 bg-green-50 px-1 rounded">✓</span>}
            </div>
            <p className="text-[10px] text-text-muted leading-snug">Predict solution from known parameters</p>
          </button>

          <button
            onClick={() => setProblemType('inverse')}
            className={['flex flex-col items-start gap-1 rounded-lg border-2 px-3 py-2.5 text-left transition-all',
              problemType === 'inverse' ? 'border-purple-500 bg-purple-50' : 'border-border hover:border-purple-300 bg-white'].join(' ')}
          >
            <div className="flex items-center gap-1.5">
              <div className={`w-4 h-4 rounded flex items-center justify-center ${problemType === 'inverse' ? 'bg-purple-500 text-white' : 'bg-slate-100 text-slate-500'}`}>
                <Search size={10} />
              </div>
              <span className="text-xs font-semibold text-text-primary">Inverse</span>
              {problemType === 'inverse' && <span className="text-[9px] font-bold text-purple-600 bg-purple-100 px-1 rounded">✓</span>}
            </div>
            <p className="text-[10px] text-text-muted leading-snug">Identify unknown PDE parameters from data</p>
          </button>
        </div>

        <Button variant="primary" size="md" className="w-full" leftIcon={<PlayCircle size={16} />}
          onClick={() => onStartExperiment(problem, problemType)}>
          Start {problemType === 'inverse' ? 'Inverse' : 'Forward'} Experiment
        </Button>
      </div>
    </motion.div>
  )
}

// ─── Main page ────────────────────────────────────────────────────────────────

export function ProblemsPage() {
  const navigate = useNavigate()
  const { setSelectedProblem, setProblemType, setCurrentStep, reset } = useExperimentStore()

  const [search, setSearch] = useState('')
  const [activeDomain, setActiveDomain] = useState<PhysicsArea | null>(null)
  const [activeDimension, setActiveDimension] = useState<Dimension | null>(null)
  const [activeDifficulty, setActiveDifficulty] = useState<Difficulty | null>(null)
  const [requiresGeometry, setRequiresGeometry] = useState<boolean | null>(null)
  const [selectedProblem, setSelectedProblemLocal] = useState<PhysicsProblem | null>(null)
  const [filtersOpen, setFiltersOpen] = useState(false)

  const { data: problems, isLoading, error } = useQuery({
    queryKey: ['problems'],
    queryFn: () => problemsApi.list().then((r) => r.data),
  })

  const filtered = useMemo(() => {
    if (!problems) return []
    return problems.filter((p) => {
      if (activeDomain && p.domain !== activeDomain) return false
      if (activeDimension && p.dimension !== activeDimension) return false
      if (activeDifficulty && p.difficulty !== activeDifficulty) return false
      if (requiresGeometry !== null && p.requires_geometry !== requiresGeometry) return false
      if (search) {
        const q = search.toLowerCase()
        return (
          p.name.toLowerCase().includes(q) ||
          p.description.toLowerCase().includes(q) ||
          p.tags.some((t) => t.toLowerCase().includes(q)) ||
          DOMAIN_CONFIG[p.domain].label.toLowerCase().includes(q)
        )
      }
      return true
    })
  }, [problems, activeDomain, activeDimension, activeDifficulty, requiresGeometry, search])

  const domainCounts = useMemo(() => {
    if (!problems) return {} as Record<string, number>
    return DOMAINS.reduce<Record<string, number>>((acc, d) => {
      acc[d] = problems.filter((p) => p.domain === d).length
      return acc
    }, {})
  }, [problems])

  const hasFilters = !!(activeDomain || activeDimension || activeDifficulty || requiresGeometry !== null)

  const handleStartExperiment = (problem: PhysicsProblem, type: ProblemType) => {
    reset()
    setSelectedProblem(problem)
    setProblemType(type)
    setCurrentStep('parameters')
    navigate('/experiment/setup')
  }

  const clearFilters = () => {
    setActiveDomain(null); setActiveDimension(null)
    setActiveDifficulty(null); setRequiresGeometry(null)
  }

  return (
    <div className="flex h-full bg-background">
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">

        {/* ── Header ── */}
        <div className="px-4 sm:px-8 pt-5 pb-3 border-b border-border bg-white flex-shrink-0">
          <div className="flex items-end gap-3 flex-wrap">
            <div className="flex-1 min-w-0">
              <h1 className="text-xl font-bold text-text-primary">Physics Problems</h1>
              <p className="text-sm text-text-secondary mt-0.5">
                {problems?.length ?? '…'} canonical benchmarks — select one to start an experiment.
              </p>
            </div>
          </div>
          <div className="mt-3 flex items-center gap-2">
            <SearchInput
              value={search}
              onChange={setSearch}
              placeholder="Search by name, domain, tags…"
              className="flex-1 max-w-lg"
            />
            {/* Filter toggle */}
            <button
              onClick={() => setFiltersOpen((v) => !v)}
              className={[
                'flex items-center gap-1.5 px-3 py-1.5 rounded-lg border text-sm font-medium transition-colors flex-shrink-0',
                filtersOpen || hasFilters
                  ? 'border-accent text-accent bg-accent/5'
                  : 'border-border text-text-muted hover:text-text-primary hover:border-slate-300 bg-white',
              ].join(' ')}
            >
              <SlidersHorizontal size={14} />
              <span className="hidden sm:inline">Filters</span>
              {hasFilters && <span className="w-1.5 h-1.5 rounded-full bg-accent" />}
              <ChevronDown size={12} className={`transition-transform ${filtersOpen ? 'rotate-180' : ''}`} />
            </button>
          </div>
        </div>

        {/* ── Collapsible filters ── */}
        <AnimatePresence initial={false}>
          {filtersOpen && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden flex-shrink-0"
            >
              <div className="px-4 sm:px-8 py-3 border-b border-border bg-slate-50 space-y-3">
                {/* Domain */}
                <div>
                  <div className="text-xs font-medium text-text-muted mb-2">Physics Area</div>
                  <div className="flex gap-2 flex-wrap">
                    <FilterChip label="All" active={activeDomain === null} onClick={() => setActiveDomain(null)} count={problems?.length} />
                    {DOMAINS.map((d) => {
                      const cfg = DOMAIN_CONFIG[d]
                      return (
                        <FilterChip key={d} label={cfg.label} active={activeDomain === d}
                          onClick={() => setActiveDomain(activeDomain === d ? null : d)}
                          icon={cfg.icon} count={domainCounts[d]} color={cfg.color} />
                      )
                    })}
                  </div>
                </div>

                <div className="flex flex-wrap gap-6">
                  {/* Dimension */}
                  <div>
                    <div className="text-xs font-medium text-text-muted mb-2">Dimension</div>
                    <div className="flex gap-2">
                      <FilterChip label="All" active={activeDimension === null} onClick={() => setActiveDimension(null)} />
                      {DIMENSIONS.map((d) => (
                        <FilterChip key={d} label={DIMENSION_LABELS[d]} active={activeDimension === d}
                          onClick={() => setActiveDimension(activeDimension === d ? null : d)} />
                      ))}
                    </div>
                  </div>
                  {/* Difficulty */}
                  <div>
                    <div className="text-xs font-medium text-text-muted mb-2">Difficulty</div>
                    <div className="flex gap-2">
                      <FilterChip label="All" active={activeDifficulty === null} onClick={() => setActiveDifficulty(null)} />
                      {DIFFICULTIES.map((d) => {
                        const cfg = DIFFICULTY_CONFIG[d]
                        return (
                          <FilterChip key={d} label={cfg.label} active={activeDifficulty === d}
                            onClick={() => setActiveDifficulty(activeDifficulty === d ? null : d)} />
                        )
                      })}
                    </div>
                  </div>
                  {/* Geometry */}
                  <div>
                    <div className="text-xs font-medium text-text-muted mb-2">Geometry</div>
                    <div className="flex gap-2">
                      <FilterChip label="All" active={requiresGeometry === null} onClick={() => setRequiresGeometry(null)} />
                      <FilterChip label="3D Geometry" active={requiresGeometry === true}
                        onClick={() => setRequiresGeometry(requiresGeometry === true ? null : true)} />
                    </div>
                  </div>
                </div>

                {hasFilters && (
                  <button onClick={clearFilters} className="text-xs text-accent hover:underline flex items-center gap-1">
                    <X size={11} /> Clear all filters
                  </button>
                )}
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── Results bar ── */}
        <div className="px-4 sm:px-8 py-2.5 flex items-center gap-2 text-sm text-text-muted bg-white border-b border-border flex-shrink-0">
          <Cpu size={13} className="text-accent" />
          <span>
            <span className="text-text-primary font-medium">{filtered.length}</span>
            {' '}problem{filtered.length !== 1 ? 's' : ''} found
          </span>
          {search && (
            <button onClick={() => setSearch('')} className="ml-auto text-xs text-accent hover:text-accent-hover flex items-center gap-1">
              <X size={11} /> Clear search
            </button>
          )}
        </div>

        {/* ── Card grid (scrollable) ── */}
        <div className="flex-1 overflow-y-auto px-4 sm:px-8 pb-10 pt-5 bg-background">
          {isLoading && <PageLoading label="Loading physics problems…" />}
          {error && (
            <div className="flex items-center justify-center min-h-[300px] text-center text-text-secondary">
              <div>
                <p className="text-lg mb-2">Failed to load problems</p>
                <p className="text-sm text-text-muted">Make sure the backend is running on port 8000</p>
              </div>
            </div>
          )}
          {!isLoading && !error && (
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {filtered.map((problem, i) => (
                <motion.div
                  key={problem.id}
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: i * 0.025, duration: 0.25 }}
                >
                  <ProblemCard
                    problem={problem}
                    selected={selectedProblem?.id === problem.id}
                    onClick={() => setSelectedProblemLocal(selectedProblem?.id === problem.id ? null : problem)}
                  />
                </motion.div>
              ))}
              {filtered.length === 0 && (
                <div className="col-span-full text-center py-16 text-text-muted">
                  <div className="text-4xl mb-3">🔭</div>
                  <p>No problems match your filters</p>
                </div>
              )}
            </div>
          )}
        </div>
      </div>

      {/* ── Detail panel ── */}
      <AnimatePresence>
        {selectedProblem && (
          <ProblemDetailPanel
            problem={selectedProblem}
            onClose={() => setSelectedProblemLocal(null)}
            onStartExperiment={handleStartExperiment}
          />
        )}
      </AnimatePresence>
    </div>
  )
}
