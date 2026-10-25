import React, { useMemo, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { ArrowLeft, GitCompare, X } from 'lucide-react'
import { externalCatalogApi, jobErrorMessage, runJob } from '../../lib/api'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { SearchInput } from '../../components/ui/SearchInput'
import { PageLoading } from '../../components/ui/LoadingSpinner'
import { RESULT_STATUS_CONFIG } from './statusConfig'

const MIN_SLUGS = 2
const MAX_SLUGS = 6

export function ExternalComparePage() {
  const [selected, setSelected] = useState<string[]>([])
  const [search, setSearch] = useState('')
  const [includeHeavy, setIncludeHeavy] = useState(false)

  const { data: models, isLoading } = useQuery({
    queryKey: ['external-catalog'],
    queryFn: () => externalCatalogApi.list().then((r) => r.data),
  })

  const filtered = useMemo(() => {
    if (!models) return []
    if (!search) return models
    const q = search.toLowerCase()
    return models.filter((m) => `${m.name} ${m.org}`.toLowerCase().includes(q))
  }, [models, search])

  const compareMutation = useMutation({
    mutationFn: () => runJob(externalCatalogApi.compare(selected, includeHeavy)),
  })

  const toggle = (slug: string) => {
    setSelected((prev) => {
      if (prev.includes(slug)) return prev.filter((s) => s !== slug)
      if (prev.length >= MAX_SLUGS) return prev
      return [...prev, slug]
    })
  }

  const result = compareMutation.data?.result
  const timings = (result?.results ?? [])
    .map((r) => ({ name: r.name, seconds: (r.metrics?.inference_seconds as number | undefined) ?? null, status: r.status }))
  const maxSeconds = Math.max(1e-6, ...timings.map((t) => t.seconds ?? 0))

  return (
    <div className="px-4 sm:px-8 py-5 space-y-5">
      <Link to="/external-catalog" className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary transition-colors w-fit">
        <ArrowLeft size={13} /> Back to catalog
      </Link>

      <div className="flex items-center gap-2">
        <GitCompare size={18} className="text-accent" />
        <h1 className="text-xl font-bold text-text-primary">Compare External Models</h1>
      </div>
      <p className="text-sm text-text-secondary max-w-2xl">
        Pick {MIN_SLUGS}–{MAX_SLUGS} models, run them side by side, and see install status, real output, and timing in one place.
      </p>

      {/* Selected chips */}
      {selected.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {selected.map((slug) => {
            const m = models?.find((x) => x.slug === slug)
            return (
              <span key={slug} className="flex items-center gap-1 text-xs bg-accent/10 text-accent border border-accent/20 rounded-full px-2.5 py-1">
                {m?.name ?? slug}
                <button onClick={() => toggle(slug)}><X size={11} /></button>
              </span>
            )
          })}
        </div>
      )}

      {/* Picker */}
      {isLoading ? (
        <PageLoading label="Loading catalog…" />
      ) : (
        <div className="bg-white border border-border rounded-xl shadow-card p-4">
          <SearchInput value={search} onChange={setSearch} placeholder="Search models to add…" className="mb-3 max-w-md" />
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-2 max-h-72 overflow-y-auto">
            {filtered.map((m) => {
              const isSelected = selected.includes(m.slug)
              const disabled = !isSelected && selected.length >= MAX_SLUGS
              return (
                <button
                  key={m.slug}
                  disabled={disabled}
                  onClick={() => toggle(m.slug)}
                  className={[
                    'text-left px-3 py-2 rounded-lg border text-xs transition-colors',
                    isSelected ? 'border-accent bg-accent/5' : disabled ? 'border-border opacity-40 cursor-not-allowed' : 'border-border hover:border-slate-300',
                  ].join(' ')}
                >
                  <div className="font-medium text-text-primary truncate">{m.name}</div>
                  <div className="text-text-muted truncate">{m.arch} · {m.org}</div>
                </button>
              )
            })}
          </div>
        </div>
      )}

      <div className="flex items-center gap-3">
        <label className="flex items-center gap-2 text-sm text-text-secondary">
          <input type="checkbox" checked={includeHeavy} onChange={(e) => setIncludeHeavy(e.target.checked)} className="rounded" />
          include_heavy
        </label>
        <Button
          variant="primary"
          disabled={selected.length < MIN_SLUGS}
          loading={compareMutation.isPending}
          onClick={() => compareMutation.mutate()}
        >
          Run comparison ({selected.length})
        </Button>
      </div>
      {compareMutation.error && (
        <p className="text-xs text-red-700 bg-red-50 border border-red-200 rounded-lg p-2.5">{jobErrorMessage(compareMutation.error, 'Comparison failed')}</p>
      )}

      {/* Results */}
      {result && (
        <div className="space-y-4">
          <div className="overflow-x-auto rounded-xl border border-border bg-white shadow-card">
            <table className="w-full">
              <thead className="bg-slate-50 border-b border-border">
                <tr>
                  <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">Model</th>
                  <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">Status</th>
                  <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">Note</th>
                  <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">Seconds</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {result.results.map((r) => {
                  const cfg = RESULT_STATUS_CONFIG[r.status]
                  return (
                    <tr key={r.slug}>
                      <td className="py-2.5 px-4 text-xs font-medium text-text-primary">{r.name}</td>
                      <td className="py-2.5 px-4"><Badge variant={cfg.variant} size="xs">{cfg.label}</Badge></td>
                      <td className="py-2.5 px-4 text-xs text-text-secondary max-w-xs truncate">{r.note ?? r.error ?? '—'}</td>
                      <td className="py-2.5 px-4 text-xs font-mono text-text-secondary">
                        {typeof r.metrics?.inference_seconds === 'number' ? `${r.metrics.inference_seconds}s` : '—'}
                      </td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>

          {/* Timing bar chart */}
          <div className="bg-white border border-border rounded-xl shadow-card p-4">
            <h3 className="text-xs font-semibold text-text-primary mb-3">Inference timing</h3>
            <div className="space-y-2">
              {timings.map((t) => (
                <div key={t.name} className="flex items-center gap-2">
                  <span className="text-[11px] text-text-muted w-32 truncate flex-shrink-0">{t.name}</span>
                  <div className="flex-1 h-3 bg-slate-100 rounded-full overflow-hidden">
                    {t.seconds != null && (
                      <div className="h-full bg-accent rounded-full" style={{ width: `${(t.seconds / maxSeconds) * 100}%` }} />
                    )}
                  </div>
                  <span className="text-[11px] font-mono text-text-secondary w-16 text-right flex-shrink-0">
                    {t.seconds != null ? `${t.seconds}s` : '—'}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
