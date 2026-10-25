import React, { useMemo } from 'react'
import { useQuery } from '@tanstack/react-query'
import { useNavigate, useSearchParams } from 'react-router-dom'
import { motion } from 'framer-motion'
import { Boxes, ExternalLink, GitCompare, Gauge, LayoutGrid, Rows3, X } from 'lucide-react'
import { externalCatalogApi } from '../../lib/api'
import type { ExternalCatalogModel, ExternalCategoryMeta, ExternalDomainMeta } from '../../types'
import { Badge } from '../../components/ui/Badge'
import { SearchInput } from '../../components/ui/SearchInput'
import { FilterChip } from '../../components/ui/FilterChip'
import { Button } from '../../components/ui/Button'
import { PageLoading } from '../../components/ui/LoadingSpinner'
import { INSTALL_STATE_CONFIG, isRestrictiveLicense } from './statusConfig'

// Maturity values come straight from ModelSpec.maturity in the backend registry.
const MATURITY_OPTIONS = [
  { key: 'maintained library', label: 'Maintained library' },
  { key: 'paper code', label: 'Paper code' },
  { key: 'archived', label: 'Archived' },
] as const

type ViewMode = 'grouped' | 'grid'

function matchesSearch(m: ExternalCatalogModel, q: string) {
  if (!q) return true
  return `${m.name} ${m.org} ${m.desc} ${m.tags.join(' ')} ${m.ref}`.toLowerCase().includes(q)
}

// ─── Model card ───────────────────────────────────────────────────────────────

interface ModelCardProps {
  model: ExternalCatalogModel
  // domain whose section this card is rendered in — excluded from the "also:" chips
  contextDomain?: string
  domainMeta: Record<string, ExternalDomainMeta>
  categoryMeta: Record<string, ExternalCategoryMeta>
  onOpen: () => void
}

function ModelCard({ model, contextDomain, domainMeta, categoryMeta, onOpen }: ModelCardProps) {
  const install = INSTALL_STATE_CONFIG[model.install_state]
  const otherDomains = model.domains.filter((d) => d !== contextDomain)

  return (
    <motion.div
      whileHover={{ y: -2 }}
      transition={{ type: 'spring', stiffness: 400, damping: 25 }}
      role="button"
      tabIndex={0}
      onClick={onOpen}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); onOpen() } }}
      className="h-full bg-white border border-border rounded-xl shadow-card p-4 cursor-pointer transition-all duration-200 hover:border-accent hover:shadow-card-hover focus-visible:outline focus-visible:outline-2 focus-visible:outline-accent flex flex-col gap-2.5"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <h3 className="text-sm font-bold text-text-primary leading-tight">{model.name}</h3>
          <p className="text-[11px] text-accent mt-0.5">{model.org}</p>
        </div>
        <Badge variant={install.variant} size="xs" className="flex-shrink-0">{install.label}</Badge>
      </div>

      <p className="text-xs text-text-secondary leading-relaxed line-clamp-4">{model.desc}</p>

      <div className="flex items-center gap-1.5 flex-wrap mt-auto pt-1">
        <Badge variant="accent" size="xs">{categoryMeta[model.arch]?.label ?? model.arch}</Badge>
        {model.maturity === 'paper code' && <Badge variant="yellow" size="xs">Paper code</Badge>}
        {model.maturity === 'archived' && <Badge variant="default" size="xs">Archived</Badge>}
        {model.license && (
          <Badge variant={isRestrictiveLicense(model.license) ? 'danger' : 'success'} size="xs">{model.license}</Badge>
        )}
        {model.heavy && <Badge variant="warning" size="xs">Heavy</Badge>}
        {model.finetunable && <Badge variant="purple" size="xs">Fine-tunable</Badge>}
      </div>

      {model.tags.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {model.tags.map((t) => (
            <span key={t} className="text-[10px] font-mono bg-slate-50 text-slate-500 border border-slate-200 px-1.5 py-0.5 rounded">{t}</span>
          ))}
        </div>
      )}

      {otherDomains.length > 0 && (
        <div className="flex flex-wrap gap-1">
          {otherDomains.map((d) => (
            <span key={d} className="text-[10px] bg-amber-50 text-amber-700 border border-amber-200 px-1.5 py-0.5 rounded">
              {contextDomain ? 'also: ' : ''}{domainMeta[d]?.label ?? d}
            </span>
          ))}
        </div>
      )}

      <div className="flex items-center justify-between border-t border-border pt-2.5 mt-1">
        <a
          href={model.repo}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          className="inline-flex items-center gap-1 text-xs font-semibold text-accent hover:underline"
        >
          Repository <ExternalLink size={11} />
        </a>
        <span className="text-[11px] text-text-muted">Details →</span>
      </div>
      {model.ref && <p className="text-[10.5px] italic text-text-muted leading-snug">{model.ref}</p>}
    </motion.div>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ExternalCatalogPage() {
  const navigate = useNavigate()
  // Filters live in the URL so a filtered view can be linked/shared and survives back-navigation.
  const [params, setParams] = useSearchParams()
  const search = params.get('q') ?? ''
  const activeDomain = params.get('domain')
  const activeCategory = params.get('arch')
  const activeMaturity = params.get('maturity')
  const view: ViewMode = params.get('view') === 'grid' ? 'grid' : 'grouped'

  const setParam = (key: string, value: string | null) => {
    setParams((prev) => {
      const next = new URLSearchParams(prev)
      if (value) next.set(key, value)
      else next.delete(key)
      return next
    }, { replace: true })
  }

  const { data: models, isLoading, error } = useQuery({
    queryKey: ['external-catalog'],
    queryFn: () => externalCatalogApi.list().then((r) => r.data),
  })

  const { data: domainMeta = {} } = useQuery({
    queryKey: ['external-catalog-domains'],
    queryFn: () => externalCatalogApi.listDomains().then((r) => r.data),
    staleTime: 5 * 60_000,
  })

  const { data: categoryMeta = {} } = useQuery({
    queryKey: ['external-catalog-categories'],
    queryFn: () => externalCatalogApi.listCategories().then((r) => r.data),
    staleTime: 5 * 60_000,
  })

  const q = search.trim().toLowerCase()

  // Everything except the architecture filter — used for the per-architecture chip counts.
  const baseFiltered = useMemo(() => (models ?? []).filter((m) =>
    (!activeDomain || m.domains.includes(activeDomain)) &&
    (!activeMaturity || m.maturity === activeMaturity) &&
    matchesSearch(m, q)
  ), [models, activeDomain, activeMaturity, q])

  const filtered = useMemo(
    () => baseFiltered.filter((m) => !activeCategory || m.arch === activeCategory),
    [baseFiltered, activeCategory],
  )

  const archCounts = useMemo(() => {
    const c: Record<string, number> = {}
    baseFiltered.forEach((m) => { c[m.arch] = (c[m.arch] ?? 0) + 1 })
    return c
  }, [baseFiltered])

  // Sections follow the backend's DOMAIN_META order; a multi-domain model appears in each of its sections.
  const sections = useMemo(() => Object.values(domainMeta)
    .filter((d) => !activeDomain || d.key === activeDomain)
    .map((d) => ({ meta: d, items: filtered.filter((m) => m.domains.includes(d.key)) }))
    .filter((s) => s.items.length > 0), [domainMeta, filtered, activeDomain])

  const stats = useMemo(() => {
    const all = models ?? []
    return {
      total: all.length,
      // only domains that actually hold a model — some exist just to label dataset subsets
      domains: new Set(all.flatMap((m) => m.domains)).size,
      archs: Object.keys(categoryMeta).length,
      paperCode: all.filter((m) => m.maturity === 'paper code').length,
      runnable: all.filter((m) => m.pip_package && !m.manual_only).length,
    }
  }, [models, categoryMeta])

  const hasFilters = !!(activeDomain || activeCategory || activeMaturity || search)
  const clearFilters = () => setParams(view === 'grid' ? { view: 'grid' } : {}, { replace: true })

  const cardProps = { domainMeta, categoryMeta }
  const open = (slug: string) => navigate(`/external-catalog/${slug}`)

  return (
    <div className="flex flex-col h-full bg-background">
      {/* Header */}
      <div className="px-4 sm:px-8 pt-5 pb-4 border-b border-border bg-white flex-shrink-0">
        <div className="flex items-end justify-between gap-3 flex-wrap">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Boxes size={18} className="text-accent" />
              <h1 className="text-xl font-bold text-text-primary">External Model Catalog</h1>
            </div>
            <p className="text-sm text-text-secondary mt-1 max-w-3xl">
              Open-source PINNs, neural operators, differentiable simulators and learned physics engines, grouped by the
              physical domain they target. Entries marked <span className="font-semibold text-amber-700">Paper code</span> are
              single-paper research repositories rather than maintained libraries — runnable, but expect less polish and no ongoing support.
            </p>
          </div>
          <div className="flex items-center gap-2 flex-shrink-0">
            <Button variant="outline" size="sm" leftIcon={<GitCompare size={14} />} onClick={() => navigate('/external-catalog/compare')}>
              Compare
            </Button>
            <Button variant="secondary" size="sm" leftIcon={<Gauge size={14} />} onClick={() => navigate('/external-catalog/benchmark')}>
              Benchmark
            </Button>
          </div>
        </div>

        <div className="mt-4 flex gap-6 flex-wrap">
          {[
            [stats.total, 'Models indexed'],
            [stats.domains, 'Physical domains'],
            [stats.archs, 'Architecture families'],
            [stats.runnable, 'pip-installable'],
            [stats.paperCode, 'Paper code'],
          ].map(([value, label]) => (
            <div key={label}>
              <div className="text-2xl font-bold text-text-primary leading-none">{models ? value : '…'}</div>
              <div className="text-[10.5px] uppercase tracking-wider text-text-muted mt-1">{label}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Controls */}
      <div className="px-4 sm:px-8 py-3 border-b border-border bg-slate-50 space-y-2.5 flex-shrink-0">
        <div className="flex items-center gap-2 flex-wrap">
          <SearchInput
            value={search}
            onChange={(v) => setParam('q', v || null)}
            placeholder="Search by name, keyword, backend, author or citation…"
            className="flex-1 min-w-[220px] max-w-xl"
          />
          <select
            value={activeDomain ?? ''}
            onChange={(e) => setParam('domain', e.target.value || null)}
            className="text-sm border border-border rounded-lg bg-white px-2.5 py-1.5 text-text-primary max-w-[260px]"
            aria-label="Physical domain"
          >
            <option value="">All physical domains</option>
            {Object.values(domainMeta).map((d) => (
              <option key={d.key} value={d.key}>{d.label}</option>
            ))}
          </select>
          <div className="flex rounded-lg border border-border bg-white overflow-hidden" role="group" aria-label="View mode">
            <button
              onClick={() => setParam('view', null)}
              className={`px-2.5 py-1.5 ${view === 'grouped' ? 'bg-accent text-white' : 'text-text-muted hover:text-text-primary'}`}
              title="Group by physical domain"
            >
              <Rows3 size={14} />
            </button>
            <button
              onClick={() => setParam('view', 'grid')}
              className={`px-2.5 py-1.5 ${view === 'grid' ? 'bg-accent text-white' : 'text-text-muted hover:text-text-primary'}`}
              title="Flat grid"
            >
              <LayoutGrid size={14} />
            </button>
          </div>
        </div>

        <div className="flex items-start gap-2 flex-wrap">
          <span className="text-[10.5px] uppercase tracking-wider text-amber-700 font-medium w-24 pt-1.5">Architecture</span>
          <div className="flex gap-1.5 flex-wrap flex-1">
            <FilterChip size="sm" label="All" count={baseFiltered.length} active={!activeCategory} onClick={() => setParam('arch', null)} />
            {Object.values(categoryMeta).map((c) => (
              <FilterChip key={c.key} size="sm" label={c.label} count={archCounts[c.key] ?? 0} active={activeCategory === c.key}
                onClick={() => setParam('arch', activeCategory === c.key ? null : c.key)} />
            ))}
          </div>
        </div>

        <div className="flex items-start gap-2 flex-wrap">
          <span className="text-[10.5px] uppercase tracking-wider text-amber-700 font-medium w-24 pt-1.5">Maturity</span>
          <div className="flex gap-1.5 flex-wrap flex-1">
            <FilterChip size="sm" label="All" active={!activeMaturity} onClick={() => setParam('maturity', null)} />
            {MATURITY_OPTIONS.map((o) => (
              <FilterChip key={o.key} size="sm" label={o.label} active={activeMaturity === o.key}
                onClick={() => setParam('maturity', activeMaturity === o.key ? null : o.key)} />
            ))}
          </div>
        </div>
      </div>

      <div className="px-4 sm:px-8 py-2 flex items-center gap-3 text-xs text-text-muted bg-white border-b border-border flex-shrink-0">
        <span>
          Showing <span className="text-accent font-semibold">{filtered.length}</span> of{' '}
          <span className="text-accent font-semibold">{stats.total}</span> models
          {view === 'grouped' && sections.length > 0 && <> in {sections.length} domain{sections.length !== 1 ? 's' : ''}</>}
        </span>
        {hasFilters && (
          <button onClick={clearFilters} className="text-accent hover:underline flex items-center gap-1">
            <X size={11} /> Clear filters
          </button>
        )}
      </div>

      <div className="flex-1 overflow-y-auto px-4 sm:px-8 pb-10 pt-5">
        {isLoading && <PageLoading label="Loading external catalog…" />}
        {error && (
          <div className="flex items-center justify-center min-h-[300px] text-center text-text-secondary">
            <div>
              <p className="text-lg mb-2">Failed to load the external catalog</p>
              <p className="text-sm text-text-muted">Make sure the backend is running and external_catalog is mounted at /api/external-catalog</p>
            </div>
          </div>
        )}

        {!isLoading && !error && filtered.length === 0 && (
          <div className="text-center py-16 text-text-muted text-sm">
            No entries match that search/filter. Try a broader term or clear the filters.
          </div>
        )}

        {!isLoading && !error && view === 'grid' && filtered.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
            {filtered.map((m, i) => (
              <motion.div key={m.slug} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: Math.min(i, 30) * 0.015 }}>
                <ModelCard model={m} {...cardProps} onOpen={() => open(m.slug)} />
              </motion.div>
            ))}
          </div>
        )}

        {!isLoading && !error && view === 'grouped' && sections.map(({ meta, items }) => (
          <section key={meta.key} id={`domain-${meta.key}`} className="mb-10">
            <div className="flex items-baseline gap-3 border-b border-dashed border-slate-300 pb-2.5 flex-wrap">
              <span className="font-mono text-base text-amber-700 min-w-[52px]">{meta.glyph}</span>
              <h2 className="text-lg font-semibold text-text-primary">{meta.label}</h2>
              <span className="text-[11px] font-mono text-text-muted">{items.length} model{items.length !== 1 ? 's' : ''}</span>
            </div>
            <p className="text-[13px] text-text-secondary mt-2 mb-4 max-w-3xl">{meta.desc}</p>
            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
              {items.map((m) => (
                <ModelCard key={m.slug} model={m} contextDomain={meta.key} {...cardProps} onOpen={() => open(m.slug)} />
              ))}
            </div>
          </section>
        ))}

        {!isLoading && !error && (
          <footer className="mt-12 pt-5 border-t border-border text-[11.5px] text-text-muted leading-relaxed max-w-4xl">
            Compiled from public repositories and papers (GitHub, arXiv, PyPI). Verify license terms — several weather-model
            weights are non-commercial (CC-BY-NC-SA) — and current maintenance status before adopting any library into
            production or into a commercial PINNeAPPle product. Not affiliated with the listed projects or their maintainers.
          </footer>
        )}
      </div>
    </div>
  )
}
