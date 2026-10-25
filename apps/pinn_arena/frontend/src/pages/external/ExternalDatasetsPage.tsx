import React, { useMemo, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import { Database, ExternalLink, PlayCircle, Terminal } from 'lucide-react'
import { externalCatalogApi, jobErrorMessage, runJob } from '../../lib/api'
import type { ExternalDataset, ExternalSubDataset } from '../../types'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { SearchInput } from '../../components/ui/SearchInput'
import { PageLoading } from '../../components/ui/LoadingSpinner'
import { ResultPanel } from './ExternalModelDetailPage'

// ─── Probe panel — load one real sample of one subset ────────────────────────

function ProbePanel({ dataset, subset }: { dataset: ExternalDataset; subset: ExternalSubDataset }) {
  const [split, setSplit] = useState(dataset.splits[0] ?? 'train')
  const [stream, setStream] = useState(false)

  const probe = useMutation({
    mutationFn: () => runJob(externalCatalogApi.probeDataset(dataset.slug, { subset: subset.name, split, stream })),
  })

  const job = probe.data
  return (
    <div className="bg-slate-50 border-t border-border px-4 py-3 space-y-3">
      <p className="text-xs text-text-secondary">{subset.desc}</p>
      <div className="flex items-center gap-3 flex-wrap text-xs">
        <label className="flex items-center gap-1.5">
          <span className="text-text-muted">Split</span>
          <select value={split} onChange={(e) => setSplit(e.target.value)}
            className="border border-border rounded-md bg-white px-2 py-1">
            {dataset.splits.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </label>
        {dataset.stream_base_path && (
          <label className="flex items-center gap-1.5" title={`Fetches one sample from ${dataset.stream_base_path}`}>
            <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} />
            <span>
              Stream one sample remotely
              {dataset.stream_requires.length > 0 && <span className="text-text-muted"> (needs {dataset.stream_requires.join(', ')})</span>}
              {dataset.local_root_configured && <span className="text-text-muted"> — local copy takes precedence</span>}
            </span>
          </label>
        )}
        <Button size="sm" leftIcon={<PlayCircle size={13} />} onClick={() => probe.mutate()} disabled={probe.isPending}>
          {probe.isPending ? 'Probing…' : 'Probe'}
        </Button>
      </div>
      {probe.error && <p className="text-xs text-red-700">{jobErrorMessage(probe.error, 'Probe failed')}</p>}
      {job?.result && <ResultPanel result={job.result} />}
    </div>
  )
}

// ─── Dataset block ────────────────────────────────────────────────────────────

function DatasetBlock({ dataset, domainLabels, query }: {
  dataset: ExternalDataset
  domainLabels: Record<string, string>
  query: string
}) {
  const [openSubset, setOpenSubset] = useState<string | null>(null)
  const subsets = useMemo(() => dataset.subsets.filter((s) =>
    !query || `${s.name} ${s.label} ${s.desc} ${s.domains.map((d) => domainLabels[d] ?? d).join(' ')}`.toLowerCase().includes(query)
  ), [dataset.subsets, query, domainLabels])

  const downloadCmd = dataset.download_command
    ?.replace('{subset}', openSubset ?? '<subset>')
    .replace('{split}', 'train')

  return (
    <section className="bg-white border border-border rounded-xl shadow-card overflow-hidden">
      <div className="p-5 space-y-3">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div>
            <h2 className="text-lg font-bold text-text-primary">{dataset.name}</h2>
            <p className="text-xs text-accent">{dataset.org}</p>
          </div>
          <div className="flex items-center gap-1.5 flex-wrap">
            <Badge variant={dataset.installed ? 'success' : 'default'} size="xs">
              {dataset.installed ? 'Loader installed' : 'Loader not installed'}
            </Badge>
            <Badge variant="info" size="xs">{dataset.size}</Badge>
            {dataset.license && <Badge variant="success" size="xs">{dataset.license}</Badge>}
          </div>
        </div>
        <p className="text-sm text-text-secondary leading-relaxed max-w-4xl">{dataset.desc}</p>
        <div className="flex flex-wrap gap-1">
          {dataset.tags.map((t) => (
            <span key={t} className="text-[10px] font-mono bg-slate-50 text-slate-500 border border-slate-200 px-1.5 py-0.5 rounded">{t}</span>
          ))}
          {dataset.domains.map((d) => (
            <span key={d} className="text-[10px] bg-amber-50 text-amber-700 border border-amber-200 px-1.5 py-0.5 rounded">{domainLabels[d] ?? d}</span>
          ))}
        </div>
        <div className="flex items-center gap-4 flex-wrap text-xs">
          <a href={dataset.repo} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 font-semibold text-accent hover:underline">
            Repository <ExternalLink size={11} />
          </a>
          {dataset.homepage && (
            <a href={dataset.homepage} target="_blank" rel="noopener noreferrer" className="inline-flex items-center gap-1 text-accent hover:underline">
              Homepage <ExternalLink size={11} />
            </a>
          )}
        </div>
        <p className="text-[11px] italic text-text-muted">{dataset.ref}</p>
        {(dataset.pip_install_command || downloadCmd) && (
          <div className="text-xs">
            <div className="text-text-muted mb-1 flex items-center gap-1"><Terminal size={11} /> Get it locally:</div>
            <code className="block bg-slate-900 text-slate-100 rounded-lg p-2.5 overflow-x-auto whitespace-pre-wrap">
              {[dataset.pip_install_command, downloadCmd].filter(Boolean).join('\n')}
            </code>
          </div>
        )}
      </div>

      <div className="border-t border-border">
        <div className="px-5 py-2 text-[10.5px] uppercase tracking-wider text-text-muted bg-slate-50 flex justify-between">
          <span>Subsets ({subsets.length}{subsets.length !== dataset.subsets.length ? ` of ${dataset.subsets.length}` : ''})</span>
          <span>click a row to probe</span>
        </div>
        <ul>
          {subsets.map((s) => (
            <li key={s.name} className="border-t border-border first:border-t-0">
              <button
                onClick={() => setOpenSubset(openSubset === s.name ? null : s.name)}
                className="w-full text-left px-5 py-2.5 flex items-center gap-3 hover:bg-slate-50"
                aria-expanded={openSubset === s.name}
              >
                <span className="font-mono text-xs text-text-primary min-w-0 flex-1 truncate">{s.name}</span>
                <span className="hidden md:inline text-xs text-text-secondary flex-1 truncate">{s.label}</span>
                <Badge variant="default" size="xs">{s.dims}</Badge>
                <span className="hidden sm:flex gap-1">
                  {s.domains.map((d) => (
                    <span key={d} className="text-[10px] bg-amber-50 text-amber-700 border border-amber-200 px-1.5 py-0.5 rounded whitespace-nowrap">{domainLabels[d] ?? d}</span>
                  ))}
                </span>
              </button>
              {openSubset === s.name && <ProbePanel dataset={dataset} subset={s} />}
            </li>
          ))}
          {subsets.length === 0 && <li className="px-5 py-4 text-xs text-text-muted">No subset matches the search.</li>}
        </ul>
      </div>
    </section>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ExternalDatasetsPage() {
  const [search, setSearch] = useState('')

  const { data: datasets, isLoading, error } = useQuery({
    queryKey: ['external-catalog-datasets'],
    queryFn: () => externalCatalogApi.listDatasets().then((r) => r.data),
  })

  const { data: domainMeta = {} } = useQuery({
    queryKey: ['external-catalog-domains'],
    queryFn: () => externalCatalogApi.listDomains().then((r) => r.data),
    staleTime: 5 * 60_000,
  })

  const domainLabels = useMemo(
    () => Object.fromEntries(Object.values(domainMeta).map((d) => [d.key, d.label])),
    [domainMeta],
  )
  const q = search.trim().toLowerCase()

  return (
    <div className="flex flex-col h-full bg-background">
      <div className="px-4 sm:px-8 pt-5 pb-4 border-b border-border bg-white flex-shrink-0">
        <div className="flex items-center gap-2">
          <Database size={18} className="text-accent" />
          <h1 className="text-xl font-bold text-text-primary">External Datasets</h1>
        </div>
        <p className="text-sm text-text-secondary mt-1 max-w-3xl">
          Open physics-simulation datasets for training and evaluating surrogates. A probe loads one real sample through the
          dataset's own loader and reports its fields and shapes — or says exactly why it couldn't, and how to get the data yourself.
        </p>
        <SearchInput value={search} onChange={setSearch} placeholder="Search subsets by name, physics or domain…" className="mt-3 max-w-xl" />
      </div>

      <div className="flex-1 overflow-y-auto px-4 sm:px-8 py-5 space-y-6">
        {isLoading && <PageLoading label="Loading datasets…" />}
        {error && <p className="text-center text-text-secondary py-16">Failed to load the dataset catalog.</p>}
        {datasets?.map((d) => <DatasetBlock key={d.slug} dataset={d} domainLabels={domainLabels} query={q} />)}
      </div>
    </div>
  )
}
