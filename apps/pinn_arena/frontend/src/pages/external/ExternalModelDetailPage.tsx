import React, { useState } from 'react'
import { useParams, useNavigate, Link } from 'react-router-dom'
import { useQuery, useMutation } from '@tanstack/react-query'
import {
  ArrowLeft, PlayCircle, Wrench, Code2, ExternalLink, Copy, Check,
  Terminal, Clock, Info,
} from 'lucide-react'
import { externalCatalogApi, jobErrorMessage, runJob } from '../../lib/api'
import type { ExternalModelResult } from '../../types'
import { Badge } from '../../components/ui/Badge'
import { Button } from '../../components/ui/Button'
import { PageLoading } from '../../components/ui/LoadingSpinner'
import { RESULT_STATUS_CONFIG, INSTALL_STATE_CONFIG, isRestrictiveLicense } from './statusConfig'

type Tab = 'overview' | 'infer' | 'finetune' | 'api'

// ─── Result panel — shared by infer & finetune ───────────────────────────────

export function ResultPanel({ result }: { result: ExternalModelResult }) {
  const cfg = RESULT_STATUS_CONFIG[result.status]
  return (
    <div className="border border-border rounded-xl p-4 bg-slate-50 space-y-3">
      <div className="flex items-center gap-2">
        <Badge variant={cfg.variant} size="sm">{cfg.icon}<span className="ml-1">{cfg.label}</span></Badge>
        {typeof result.metrics?.inference_seconds === 'number' && (
          <span className="text-xs text-text-muted flex items-center gap-1"><Clock size={11} /> {result.metrics.inference_seconds as number}s</span>
        )}
      </div>
      <p className="text-[11px] text-text-muted">
        {result.category === 'dataset' && result.status === 'ok'
          ? 'The real dataset loader was imported and returned a real sample.'
          : cfg.hint}
      </p>
      {result.note && (
        <div className="text-xs text-text-secondary bg-white border border-border rounded-lg p-2.5">{result.note}</div>
      )}
      {result.error && (
        <div className="text-xs text-red-700 bg-red-50 border border-red-200 rounded-lg p-2.5 font-mono whitespace-pre-wrap">{result.error}</div>
      )}
      {result.output_shape && (
        <div className="text-xs text-text-secondary"><span className="text-text-muted">output_shape: </span><span className="font-mono">{result.output_shape}</span></div>
      )}
      {result.output_summary && (
        <pre className="text-[11px] bg-white border border-border rounded-lg p-2.5 overflow-x-auto">{JSON.stringify(result.output_summary, null, 2)}</pre>
      )}
      {Object.keys(result.metrics ?? {}).length > 0 && (
        <pre className="text-[11px] bg-white border border-border rounded-lg p-2.5 overflow-x-auto">{JSON.stringify(result.metrics, null, 2)}</pre>
      )}
      {result.local_command && (
        <div className="text-xs">
          <div className="text-text-muted mb-1 flex items-center gap-1"><Terminal size={11} /> Run it yourself:</div>
          <code className="block bg-slate-900 text-slate-100 rounded-lg p-2.5 overflow-x-auto whitespace-pre-wrap">{result.local_command}</code>
        </div>
      )}
    </div>
  )
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false)
  return (
    <button
      onClick={() => { navigator.clipboard?.writeText(text).catch(() => {}); setCopied(true); setTimeout(() => setCopied(false), 1500) }}
      className="absolute top-2 right-2 p-1.5 rounded-md bg-white/10 hover:bg-white/20 text-slate-200 transition-colors"
      title="Copy"
    >
      {copied ? <Check size={13} /> : <Copy size={13} />}
    </button>
  )
}

// ─── Page ─────────────────────────────────────────────────────────────────────

export function ExternalModelDetailPage() {
  const { slug = '' } = useParams()
  const navigate = useNavigate()
  const [tab, setTab] = useState<Tab>('overview')
  const [includeHeavy, setIncludeHeavy] = useState(false)
  const [finetuneParams, setFinetuneParams] = useState<Record<string, string>>({ epochs: '', lr: '' })

  const { data: model, isLoading } = useQuery({
    queryKey: ['external-catalog-model', slug],
    queryFn: () => externalCatalogApi.get(slug).then((r) => r.data),
  })

  // Same query keys as the catalog page, so these are usually already cached.
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

  const { data: sample } = useQuery({
    queryKey: ['external-catalog-sample', slug],
    queryFn: () => externalCatalogApi.sampleInput(slug).then((r) => r.data as { name: string; description: string; fields: Record<string, unknown> }),
    enabled: !!slug,
  })

  const { data: snippet } = useQuery({
    queryKey: ['external-catalog-snippet', slug],
    queryFn: () => externalCatalogApi.apiSnippet(slug).then((r) => r.data),
    enabled: tab === 'api',
  })

  const inferMutation = useMutation({
    mutationFn: () => runJob(externalCatalogApi.infer(slug, includeHeavy)),
  })

  const finetuneMutation = useMutation({
    mutationFn: () => {
      const params: Record<string, unknown> = { include_heavy: includeHeavy }
      if (finetuneParams.epochs) params.epochs = Number(finetuneParams.epochs)
      if (finetuneParams.lr) params.lr = Number(finetuneParams.lr)
      return runJob(externalCatalogApi.finetune(slug, params))
    },
  })

  if (isLoading) return <PageLoading label="Loading model…" />
  if (!model) {
    return (
      <div className="p-8 text-center text-text-muted">
        <p>Model not found.</p>
        <Link to="/external-catalog" className="text-accent hover:underline text-sm">Back to catalog</Link>
      </div>
    )
  }

  const install = INSTALL_STATE_CONFIG[model.install_state]
  const inferResult = inferMutation.data?.result
  const finetuneResult = finetuneMutation.data?.result

  return (
    <div className="px-4 sm:px-8 py-5 space-y-5 max-w-4xl">
      <button onClick={() => navigate('/external-catalog')} className="flex items-center gap-1.5 text-xs text-text-muted hover:text-text-primary transition-colors">
        <ArrowLeft size={13} /> Back to catalog
      </button>

      {/* Header */}
      <div className="bg-white border border-border rounded-xl shadow-card p-5">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap mb-1.5">
              <Badge variant="accent" size="sm">{categoryMeta[model.arch]?.label ?? model.arch}</Badge>
              <Badge variant={install.variant} size="sm">{install.label}</Badge>
              {model.heavy && <Badge variant="warning" size="sm">Heavy / GPU</Badge>}
              {model.finetunable && <Badge variant="purple" size="sm">Fine-tunable</Badge>}
              {model.maturity !== 'maintained library' && <Badge variant="yellow" size="sm">{model.maturity}</Badge>}
              {model.license && (
                <Badge variant={isRestrictiveLicense(model.license) ? 'danger' : 'success'} size="sm">{model.license}</Badge>
              )}
            </div>
            <h1 className="text-xl font-bold text-text-primary">{model.name}</h1>
            <p className="text-sm text-text-muted mt-0.5">{model.org}</p>
          </div>
          <a href={model.repo} target="_blank" rel="noreferrer" className="flex items-center gap-1 text-xs text-accent hover:underline flex-shrink-0">
            <ExternalLink size={12} /> Repository
          </a>
        </div>
        <p className="text-sm text-text-secondary leading-relaxed mt-3">{model.desc}</p>
        <div className="flex flex-wrap gap-1.5 mt-3">
          {model.domains.map((d) => (
            <Link key={d} to={`/external-catalog?domain=${d}`}
              className="text-[10px] bg-amber-50 text-amber-700 border border-amber-200 px-2 py-0.5 rounded-full hover:border-amber-400">
              {domainMeta[d]?.label ?? d}
            </Link>
          ))}
        </div>
        {model.ref && <p className="text-[11px] text-text-muted mt-3 italic">{model.ref}</p>}
      </div>

      {/* Tabs */}
      <div className="flex gap-1 border-b border-border">
        {([
          ['overview', 'Overview', <Info key="i" size={13} />],
          ['infer', 'Run Inference', <PlayCircle key="p" size={13} />],
          ['finetune', 'Fine-tune', <Wrench key="w" size={13} />],
          ['api', 'Use as API', <Code2 key="c" size={13} />],
        ] as [Tab, string, React.ReactNode][]).map(([key, label, icon]) => (
          <button
            key={key}
            onClick={() => setTab(key)}
            className={[
              'flex items-center gap-1.5 px-3.5 py-2 text-xs font-medium border-b-2 -mb-px transition-colors',
              tab === key ? 'border-accent text-accent' : 'border-transparent text-text-muted hover:text-text-primary',
            ].join(' ')}
          >
            {icon} {label}
          </button>
        ))}
      </div>

      {/* Overview */}
      {tab === 'overview' && (
        <div className="bg-white border border-border rounded-xl shadow-card p-5 space-y-4">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <div className="text-xs text-text-muted uppercase tracking-wider mb-1">Maturity</div>
              <div className="text-text-primary">{model.maturity}</div>
            </div>
            <div>
              <div className="text-xs text-text-muted uppercase tracking-wider mb-1">License</div>
              <div className="text-text-primary">{model.license ?? '—'}</div>
            </div>
            <div>
              <div className="text-xs text-text-muted uppercase tracking-wider mb-1">Pip package</div>
              <div className="text-text-primary font-mono text-xs">{model.pip_package ?? 'none (manual install)'}</div>
            </div>
            <div>
              <div className="text-xs text-text-muted uppercase tracking-wider mb-1">Tags</div>
              <div className="text-text-secondary text-xs">{model.tags.join(', ') || '—'}</div>
            </div>
          </div>
          {model.manual_note && (
            <div className="text-xs text-blue-800 bg-blue-50 border border-blue-200 rounded-lg p-3">{model.manual_note}</div>
          )}
          {sample && (
            <div>
              <div className="text-xs text-text-muted uppercase tracking-wider mb-2">Demo dataset — {sample.name}</div>
              <p className="text-xs text-text-secondary mb-2">{sample.description}</p>
              <pre className="text-[11px] bg-slate-50 border border-border rounded-lg p-3 overflow-x-auto">{JSON.stringify(sample.fields, null, 2)}</pre>
            </div>
          )}
        </div>
      )}

      {/* Run inference */}
      {tab === 'infer' && (
        <div className="bg-white border border-border rounded-xl shadow-card p-5 space-y-4">
          <label className="flex items-center gap-2 text-sm text-text-secondary">
            <input type="checkbox" checked={includeHeavy} onChange={(e) => setIncludeHeavy(e.target.checked)} className="rounded" />
            include_heavy (required for GPU/heavy dependencies)
          </label>
          <Button variant="primary" size="md" leftIcon={<PlayCircle size={15} />} loading={inferMutation.isPending} onClick={() => inferMutation.mutate()}>
            Run inference
          </Button>
          {inferMutation.error && (
            <p className="text-xs text-red-700 bg-red-50 border border-red-200 rounded-lg p-2.5">{jobErrorMessage(inferMutation.error, 'Inference failed')}</p>
          )}
          {inferResult && <ResultPanel result={inferResult} />}
        </div>
      )}

      {/* Fine-tune */}
      {tab === 'finetune' && (
        <div className="bg-white border border-border rounded-xl shadow-card p-5 space-y-4">
          <p className="text-xs text-text-muted">
            {model.finetunable
              ? 'Runs a real short training/optimization loop, or a real reconfiguration with the parameters below.'
              : 'This model has no in-app fine-tune/reconfigure loop wired up — running it will report finetune_not_supported with the local command instead.'}
          </p>
          <div className="grid grid-cols-2 gap-3 max-w-sm">
            <label className="text-xs text-text-muted">
              Epochs
              <input
                type="number" value={finetuneParams.epochs}
                onChange={(e) => setFinetuneParams((p) => ({ ...p, epochs: e.target.value }))}
                placeholder="default"
                className="mt-1 w-full text-sm border border-border rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-accent/20"
              />
            </label>
            <label className="text-xs text-text-muted">
              Learning rate
              <input
                type="number" step="0.0001" value={finetuneParams.lr}
                onChange={(e) => setFinetuneParams((p) => ({ ...p, lr: e.target.value }))}
                placeholder="default"
                className="mt-1 w-full text-sm border border-border rounded-lg px-2.5 py-1.5 focus:outline-none focus:ring-2 focus:ring-accent/20"
              />
            </label>
          </div>
          <label className="flex items-center gap-2 text-sm text-text-secondary">
            <input type="checkbox" checked={includeHeavy} onChange={(e) => setIncludeHeavy(e.target.checked)} className="rounded" />
            include_heavy
          </label>
          <Button variant="primary" size="md" leftIcon={<Wrench size={15} />} loading={finetuneMutation.isPending} onClick={() => finetuneMutation.mutate()}>
            Fine-tune / reconfigure
          </Button>
          {finetuneMutation.error && (
            <p className="text-xs text-red-700 bg-red-50 border border-red-200 rounded-lg p-2.5">{jobErrorMessage(finetuneMutation.error, 'Fine-tune failed')}</p>
          )}
          {finetuneResult && <ResultPanel result={finetuneResult} />}
        </div>
      )}

      {/* Use as API */}
      {tab === 'api' && (
        <div className="bg-white border border-border rounded-xl shadow-card p-5 space-y-4">
          <p className="text-xs text-text-secondary">
            Every model is reachable directly over HTTP. Full schema at{' '}
            {snippet && <a href={snippet.openapi_docs} target="_blank" rel="noreferrer" className="text-accent hover:underline">{snippet.openapi_docs}</a>}.
          </p>
          {snippet ? (
            <>
              <div>
                <div className="text-xs text-text-muted uppercase tracking-wider mb-1.5">curl</div>
                <div className="relative">
                  <pre className="text-[11px] bg-slate-900 text-slate-100 rounded-lg p-3 pr-10 overflow-x-auto whitespace-pre-wrap">{snippet.curl}</pre>
                  <CopyButton text={snippet.curl} />
                </div>
              </div>
              <div>
                <div className="text-xs text-text-muted uppercase tracking-wider mb-1.5">python</div>
                <div className="relative">
                  <pre className="text-[11px] bg-slate-900 text-slate-100 rounded-lg p-3 pr-10 overflow-x-auto whitespace-pre-wrap">{snippet.python}</pre>
                  <CopyButton text={snippet.python} />
                </div>
              </div>
            </>
          ) : (
            <PageLoading label="Loading snippet…" />
          )}
        </div>
      )}
    </div>
  )
}
