import React, { useMemo, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import { ArrowLeft, Gauge, Download, ArrowUpDown } from 'lucide-react'
import { externalCatalogApi, jobErrorMessage, runJob } from '../../lib/api'
import type { ExternalModelResult } from '../../types'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { PageLoading } from '../../components/ui/LoadingSpinner'
import { RESULT_STATUS_CONFIG } from './statusConfig'

type SortKey = 'name' | 'status' | 'seconds'

function download(filename: string, content: string, mime: string) {
  const blob = new Blob([content], { type: mime })
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(url)
}

function toCsv(results: ExternalModelResult[]): string {
  const header = ['slug', 'name', 'status', 'output_shape', 'note', 'error']
  const rows = results.map((r) => [r.slug, r.name, r.status, r.output_shape ?? '', r.note ?? '', r.error ?? ''])
  return [header, ...rows].map((row) => row.map((v) => `"${String(v).replace(/"/g, '""')}"`).join(',')).join('\n')
}

export function ExternalBenchmarkPage() {
  const [category, setCategory] = useState<string>('')
  const [domain, setDomain] = useState<string>('')
  const [includeHeavy, setIncludeHeavy] = useState(false)
  const [sortKey, setSortKey] = useState<SortKey>('name')
  const [sortDir, setSortDir] = useState<1 | -1>(1)

  const { data: categoryMeta } = useQuery({
    queryKey: ['external-catalog-categories'],
    queryFn: () => externalCatalogApi.listCategories().then((r) => r.data),
    staleTime: 5 * 60_000,
  })
  const { data: domainMeta } = useQuery({
    queryKey: ['external-catalog-domains'],
    queryFn: () => externalCatalogApi.listDomains().then((r) => r.data),
    staleTime: 5 * 60_000,
  })

  const benchmarkMutation = useMutation({
    mutationFn: () => runJob(externalCatalogApi.benchmark({
      category: category || undefined,
      domain: domain || undefined,
      includeHeavy,
    })),
  })

  const result = benchmarkMutation.data?.result
  const results = result?.results ?? []

  const sorted = useMemo(() => {
    const arr = [...results]
    arr.sort((a, b) => {
      let cmp = 0
      if (sortKey === 'name') cmp = a.name.localeCompare(b.name)
      else if (sortKey === 'status') cmp = a.status.localeCompare(b.status)
      else {
        const av = (a.metrics?.inference_seconds as number | undefined) ?? -1
        const bv = (b.metrics?.inference_seconds as number | undefined) ?? -1
        cmp = av - bv
      }
      return cmp * sortDir
    })
    return arr
  }, [results, sortKey, sortDir])

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortDir((d) => (d === 1 ? -1 : 1))
    else { setSortKey(key); setSortDir(1) }
  }

  return (
    <div className="px-4 sm:px-8 py-5 space-y-5">
      <Link to="/external-catalog" className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary transition-colors w-fit">
        <ArrowLeft size={13} /> Back to catalog
      </Link>

      <div className="flex items-center gap-2">
        <Gauge size={18} className="text-accent" />
        <h1 className="text-xl font-bold text-text-primary">Benchmark External Catalog</h1>
      </div>
      <p className="text-sm text-text-secondary max-w-2xl">
        Run the whole catalog (or a category/domain slice) as a smoke-test suite. Every result carries the same
        honest status — nothing here is ever fabricated as "ok".
      </p>

      {/* Controls */}
      <div className="bg-white border border-border rounded-xl shadow-card p-4 flex flex-wrap items-end gap-3">
        <label className="text-xs text-text-muted">
          Category
          <select value={category} onChange={(e) => setCategory(e.target.value)}
            className="mt-1 block w-44 text-sm border border-border rounded-lg px-2.5 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-accent/20">
            <option value="">All categories</option>
            {Object.values(categoryMeta ?? {}).map((c) => <option key={c.key} value={c.key}>{c.label}</option>)}
          </select>
        </label>
        <label className="text-xs text-text-muted">
          Domain
          <select value={domain} onChange={(e) => setDomain(e.target.value)}
            className="mt-1 block w-56 text-sm border border-border rounded-lg px-2.5 py-1.5 bg-white focus:outline-none focus:ring-2 focus:ring-accent/20">
            <option value="">All domains</option>
            {Object.values(domainMeta ?? {}).map((d) => <option key={d.key} value={d.key}>{d.label}</option>)}
          </select>
        </label>
        <label className="flex items-center gap-2 text-sm text-text-secondary pb-1.5">
          <input type="checkbox" checked={includeHeavy} onChange={(e) => setIncludeHeavy(e.target.checked)} className="rounded" />
          include_heavy
        </label>
        <Button variant="primary" loading={benchmarkMutation.isPending} onClick={() => benchmarkMutation.mutate()}>
          Run benchmark
        </Button>
        {benchmarkMutation.error && (
          <span className="text-xs text-red-600">
            {jobErrorMessage(benchmarkMutation.error, 'Benchmark failed')}
          </span>
        )}
      </div>

      {benchmarkMutation.isPending && <PageLoading label="Running benchmark…" />}

      {result && (
        <div className="space-y-4">
          {/* Summary */}
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
            <div className="bg-white border border-border rounded-xl p-3 text-center shadow-card">
              <div className="text-xl font-bold text-text-primary">{result.summary.total}</div>
              <div className="text-[11px] text-text-muted">Models run</div>
            </div>
            <div className="bg-white border border-border rounded-xl p-3 text-center shadow-card">
              <div className="text-xl font-bold text-emerald-600">{result.summary.real_ok_count}</div>
              <div className="text-[11px] text-text-muted">Real "ok" results</div>
            </div>
            <div className="bg-white border border-border rounded-xl p-3 text-center shadow-card">
              <div className="text-xl font-bold text-text-primary">{result.summary.avg_inference_seconds ?? '—'}</div>
              <div className="text-[11px] text-text-muted">Avg seconds</div>
            </div>
            <div className="bg-white border border-border rounded-xl p-3 text-center shadow-card">
              <div className="text-xl font-bold text-text-primary">{result.summary.max_inference_seconds ?? '—'}</div>
              <div className="text-[11px] text-text-muted">Max seconds</div>
            </div>
          </div>

          {/* Status breakdown */}
          <div className="flex flex-wrap gap-2">
            {Object.entries(result.summary.by_status).map(([status, count]) => {
              const cfg = RESULT_STATUS_CONFIG[status as keyof typeof RESULT_STATUS_CONFIG]
              return <Badge key={status} variant={cfg?.variant ?? 'default'} size="sm">{cfg?.label ?? status}: {count}</Badge>
            })}
          </div>

          {/* Export */}
          <div className="flex gap-2">
            <Button variant="outline" size="sm" leftIcon={<Download size={13} />}
              onClick={() => download('benchmark_report.json', JSON.stringify(results, null, 2), 'application/json')}>
              Export JSON
            </Button>
            <Button variant="outline" size="sm" leftIcon={<Download size={13} />}
              onClick={() => download('benchmark_report.csv', toCsv(results), 'text/csv')}>
              Export CSV
            </Button>
          </div>

          {/* Sortable table */}
          <div className="overflow-x-auto rounded-xl border border-border bg-white shadow-card">
            <table className="w-full">
              <thead className="bg-slate-50 border-b border-border">
                <tr>
                  {([['name', 'Model'], ['status', 'Status'], ['seconds', 'Seconds']] as [SortKey, string][]).map(([key, label]) => (
                    <th key={key} className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4 cursor-pointer select-none"
                      onClick={() => toggleSort(key)}>
                      <span className="flex items-center gap-1">{label} <ArrowUpDown size={10} className={sortKey === key ? 'text-accent' : ''} /></span>
                    </th>
                  ))}
                  <th className="text-left text-xs font-semibold text-text-muted uppercase tracking-wider py-2.5 px-4">Note / error</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border">
                {sorted.map((r) => {
                  const cfg = RESULT_STATUS_CONFIG[r.status]
                  return (
                    <tr key={r.slug}>
                      <td className="py-2 px-4 text-xs font-medium text-text-primary">{r.name}</td>
                      <td className="py-2 px-4"><Badge variant={cfg.variant} size="xs">{cfg.label}</Badge></td>
                      <td className="py-2 px-4 text-xs font-mono text-text-secondary">
                        {typeof r.metrics?.inference_seconds === 'number' ? `${r.metrics.inference_seconds}s` : '—'}
                      </td>
                      <td className="py-2 px-4 text-xs text-text-secondary max-w-sm truncate">{r.note ?? r.error ?? '—'}</td>
                    </tr>
                  )
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}
