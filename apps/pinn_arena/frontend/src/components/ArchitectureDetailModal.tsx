import { useState, useEffect, useCallback } from 'react'
import type { ArchitectureInfo, DiagramNode, DiagramEdge, ArchitectureLayerSchema } from '../types'
import { architecturesApi } from '../lib/api'

// ─── Diagram colours by node type ────────────────────────────────────────────

const NODE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  input:     { bg: '#1e3a5f', border: '#3b82f6', text: '#93c5fd' },
  output:    { bg: '#1a3a2a', border: '#22c55e', text: '#86efac' },
  linear:    { bg: '#1e293b', border: '#64748b', text: '#cbd5e1' },
  attention: { bg: '#2d1b4e', border: '#a855f7', text: '#d8b4fe' },
  recurrent: { bg: '#1c2f3a', border: '#06b6d4', text: '#67e8f9' },
  conv:      { bg: '#2a1f10', border: '#f97316', text: '#fdba74' },
  block:     { bg: '#1e2a3a', border: '#475569', text: '#94a3b8' },
  norm:      { bg: '#1e2a1e', border: '#4ade80', text: '#86efac' },
  activation:{ bg: '#2a1e10', border: '#fb923c', text: '#fdba74' },
}

const FAMILY_COLORS: Record<string, string> = {
  PINN:          'bg-blue-500/20 text-blue-300 border-blue-500/40',
  NEURAL_OPERATOR:'bg-purple-500/20 text-purple-300 border-purple-500/40',
  TRANSFORMER:   'bg-indigo-500/20 text-indigo-300 border-indigo-500/40',
  RECURRENT:     'bg-cyan-500/20 text-cyan-300 border-cyan-500/40',
  CONVOLUTIONAL: 'bg-orange-500/20 text-orange-300 border-orange-500/40',
  GRAPH:         'bg-green-500/20 text-green-300 border-green-500/40',
  AUTOENCODER:   'bg-yellow-500/20 text-yellow-300 border-yellow-500/40',
  CONTINUOUS:    'bg-teal-500/20 text-teal-300 border-teal-500/40',
  PHYSICS_AWARE: 'bg-pink-500/20 text-pink-300 border-pink-500/40',
  RESERVOIR:     'bg-rose-500/20 text-rose-300 border-rose-500/40',
  ROM:           'bg-sky-500/20 text-sky-300 border-sky-500/40',
  CLASSICAL:     'bg-slate-500/20 text-slate-300 border-slate-500/40',
}

// ─── SVG Diagram renderer ─────────────────────────────────────────────────────

const NODE_W = 180
const NODE_H = 56
const NODE_GAP_Y = 20
const CANVAS_PADDING = 24

function ArchDiagram({ schema }: { schema: ArchitectureLayerSchema }) {
  const { nodes, edges } = schema
  if (!nodes.length) return null

  const totalH = nodes.length * (NODE_H + NODE_GAP_Y) - NODE_GAP_Y + CANVAS_PADDING * 2
  const canvasW = NODE_W + CANVAS_PADDING * 2
  const canvasH = totalH

  const yFor = (idx: number) => CANVAS_PADDING + idx * (NODE_H + NODE_GAP_Y)
  const cx = CANVAS_PADDING + NODE_W / 2

  // Build index map
  const idxOf = Object.fromEntries(nodes.map((n, i) => [n.id, i]))

  return (
    <svg
      width={canvasW}
      height={canvasH}
      viewBox={`0 0 ${canvasW} ${canvasH}`}
      style={{ display: 'block' }}
    >
      {/* Edges — drawn first so they appear below nodes */}
      {edges.map((e: DiagramEdge, i: number) => {
        const srcIdx = idxOf[e.from]
        const dstIdx = idxOf[e.to]
        if (srcIdx === undefined || dstIdx === undefined) return null
        const x1 = cx
        const y1 = yFor(srcIdx) + NODE_H
        const x2 = cx
        const y2 = yFor(dstIdx)
        const my = (y1 + y2) / 2
        return (
          <g key={i}>
            <path
              d={`M ${x1} ${y1} C ${x1} ${my}, ${x2} ${my}, ${x2} ${y2}`}
              stroke="#334155"
              strokeWidth={2}
              fill="none"
              markerEnd="url(#arr)"
            />
          </g>
        )
      })}

      {/* Arrow marker */}
      <defs>
        <marker id="arr" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto">
          <path d="M0,0 L0,6 L8,3 z" fill="#475569" />
        </marker>
      </defs>

      {/* Nodes */}
      {nodes.map((n: DiagramNode, i: number) => {
        const col = NODE_COLORS[n.type] ?? NODE_COLORS.block
        const y = yFor(i)
        return (
          <g key={n.id}>
            <rect
              x={CANVAS_PADDING}
              y={y}
              width={NODE_W}
              height={NODE_H}
              rx={8}
              fill={col.bg}
              stroke={col.border}
              strokeWidth={1.5}
            />
            <text
              x={cx}
              y={y + 22}
              textAnchor="middle"
              fill={col.text}
              fontSize={13}
              fontWeight={600}
              fontFamily="ui-monospace, monospace"
            >
              {n.label}
            </text>
            {n.sublabel && (
              <text
                x={cx}
                y={y + 38}
                textAnchor="middle"
                fill="#64748b"
                fontSize={11}
                fontFamily="ui-monospace, monospace"
              >
                {n.sublabel}
              </text>
            )}
          </g>
        )
      })}
    </svg>
  )
}

// ─── Feedback button ──────────────────────────────────────────────────────────

function FeedbackPanel({ archId }: { archId: string }) {
  const [voted, setVoted] = useState<boolean | null>(null)
  const [comment, setComment] = useState('')
  const [submitting, setSubmitting] = useState(false)
  const [summary, setSummary] = useState<{ thumbs_up: number; thumbs_down: number; total: number } | null>(null)
  const [showComment, setShowComment] = useState(false)

  useEffect(() => {
    architecturesApi.getFeedbackSummary(archId)
      .then((r) => setSummary(r.data))
      .catch(() => {})
  }, [archId])

  const submit = useCallback(async (isPositive: boolean) => {
    setVoted(isPositive)
    setSubmitting(true)
    try {
      await architecturesApi.submitFeedback(archId, isPositive, comment || undefined)
      setSummary((prev) => prev
        ? {
            ...prev,
            thumbs_up: prev.thumbs_up + (isPositive ? 1 : 0),
            thumbs_down: prev.thumbs_down + (isPositive ? 0 : 1),
            total: prev.total + 1,
          }
        : null,
      )
    } catch {
      setVoted(null)
    } finally {
      setSubmitting(false)
      setShowComment(false)
      setComment('')
    }
  }, [archId, comment])

  return (
    <div className="mt-6 rounded-xl border border-white/10 bg-white/5 p-4">
      <p className="mb-3 text-sm font-semibold text-slate-300">Was this architecture useful for your problem?</p>
      <div className="flex items-center gap-3">
        <button
          onClick={() => voted === null && setShowComment(true) ? setShowComment(true) : submit(true)}
          disabled={voted !== null || submitting}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-all
            ${voted === true ? 'bg-green-500/30 text-green-300 border border-green-500/50' : 'bg-white/5 text-slate-300 border border-white/10 hover:bg-green-500/20 hover:text-green-300'}
            disabled:cursor-default`}
        >
          👍 Helpful
        </button>
        <button
          onClick={() => submit(false)}
          disabled={voted !== null || submitting}
          className={`flex items-center gap-2 rounded-lg px-4 py-2 text-sm font-semibold transition-all
            ${voted === false ? 'bg-red-500/30 text-red-300 border border-red-500/50' : 'bg-white/5 text-slate-300 border border-white/10 hover:bg-red-500/20 hover:text-red-300'}
            disabled:cursor-default`}
        >
          👎 Not helpful
        </button>
        {summary && (
          <span className="ml-auto text-xs text-slate-500">
            {summary.thumbs_up}👍  {summary.thumbs_down}👎  · {summary.total} ratings
          </span>
        )}
      </div>
      {showComment && voted === null && (
        <div className="mt-3 flex gap-2">
          <input
            type="text"
            placeholder="Optional comment..."
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            className="flex-1 rounded-lg border border-white/10 bg-white/5 px-3 py-2 text-sm text-slate-200 placeholder-slate-500 focus:outline-none focus:border-blue-500/50"
          />
          <button
            onClick={() => submit(true)}
            className="rounded-lg bg-green-600 px-4 py-2 text-sm font-semibold text-white hover:bg-green-500"
          >
            Submit
          </button>
        </div>
      )}
      {voted !== null && (
        <p className="mt-2 text-xs text-slate-400">Thanks for your feedback!</p>
      )}
    </div>
  )
}

// ─── Main modal ───────────────────────────────────────────────────────────────

interface Props {
  architecture: ArchitectureInfo | null
  onClose: () => void
}

export default function ArchitectureDetailModal({ architecture, onClose }: Props) {
  const [layerSchema, setLayerSchema] = useState<ArchitectureLayerSchema | null>(null)
  const [loadingDiagram, setLoadingDiagram] = useState(false)

  useEffect(() => {
    if (!architecture) return
    setLayerSchema(null)
    setLoadingDiagram(true)
    architecturesApi.getLayers(architecture.id)
      .then((r) => setLayerSchema(r.data))
      .catch(() => {})
      .finally(() => setLoadingDiagram(false))
  }, [architecture?.id])

  if (!architecture) return null

  const familyClass = FAMILY_COLORS[architecture.family] ?? 'bg-slate-500/20 text-slate-300 border-slate-500/40'

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4"
      onClick={onClose}
    >
      <div
        className="relative flex w-full max-w-4xl max-h-[90vh] overflow-hidden rounded-2xl border border-white/10 bg-[#0f1623] shadow-2xl"
        style={{ colorScheme: 'dark' }}
        onClick={(e) => e.stopPropagation()}
      >
        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute right-4 top-4 z-10 rounded-lg p-1.5 text-slate-400 hover:bg-white/10 hover:text-white transition-colors"
        >
          <svg className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        {/* Left: Diagram panel */}
        <div className="hidden md:flex w-60 shrink-0 flex-col items-center gap-4 border-r border-white/10 bg-white/3 p-6 overflow-y-auto overflow-x-hidden">
          <p className="text-xs font-semibold uppercase tracking-widest text-slate-500 mb-2">Architecture</p>
          {loadingDiagram ? (
            <div className="flex items-center justify-center h-40">
              <div className="h-6 w-6 animate-spin rounded-full border-2 border-blue-500 border-t-transparent" />
            </div>
          ) : layerSchema ? (
            <ArchDiagram schema={layerSchema} />
          ) : (
            <div className="text-xs text-slate-500 text-center">Diagram unavailable</div>
          )}
        </div>

        {/* Right: Info panel */}
        <div className="flex-1 overflow-y-auto p-6">
          {/* Header */}
          <div className="mb-5 pr-8">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <span className={`rounded-full border px-2.5 py-0.5 text-xs font-semibold ${familyClass}`}>
                {architecture.family.replace(/_/g, ' ')}
              </span>
              {architecture.inverse_only && (
                <span className="rounded-full border border-purple-500/40 bg-purple-500/20 px-2.5 py-0.5 text-xs font-semibold text-purple-300">
                  Inverse Only
                </span>
              )}
              {architecture.supports_physics_loss && (
                <span className="rounded-full border border-blue-500/30 bg-blue-500/10 px-2.5 py-0.5 text-xs font-semibold text-blue-300">
                  Physics Loss ✓
                </span>
              )}
              {architecture.supports_3d && (
                <span className="rounded-full border border-green-500/30 bg-green-500/10 px-2.5 py-0.5 text-xs font-semibold text-green-300">
                  3D ✓
                </span>
              )}
            </div>
            <h2 className="text-2xl font-bold text-white">{architecture.name}</h2>
          </div>

          {/* Description */}
          <div className="mb-4 rounded-xl border border-white/5 bg-white/5 p-4">
            <p className="text-sm leading-relaxed text-slate-300">{architecture.description}</p>
          </div>

          {/* Detailed description */}
          {architecture.detailed_description && (
            <div className="mb-4 rounded-xl border border-blue-500/20 bg-blue-500/5 p-4">
              <p className="mb-1.5 text-xs font-semibold uppercase tracking-wider text-blue-400">Design Details</p>
              <p className="text-sm leading-relaxed text-slate-300">{architecture.detailed_description}</p>
            </div>
          )}

          {/* Best for */}
          {architecture.best_for?.length > 0 && (
            <div className="mb-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">Best For</p>
              <div className="flex flex-wrap gap-2">
                {architecture.best_for.map((tag) => (
                  <span
                    key={tag}
                    className="rounded-full border border-white/10 bg-white/5 px-2.5 py-1 text-xs text-slate-300"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Parameters */}
          {Object.keys(architecture.parameters).length > 0 && (
            <div className="mb-4">
              <p className="mb-2 text-xs font-semibold uppercase tracking-wider text-slate-500">Default Hyperparameters</p>
              <div className="overflow-x-auto rounded-xl border border-white/10">
                <table className="w-full text-xs">
                  <thead className="bg-white/5">
                    <tr>
                      <th className="px-3 py-2 text-left text-slate-400 font-semibold">Parameter</th>
                      <th className="px-3 py-2 text-left text-slate-400 font-semibold">Default</th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.entries(architecture.parameters).map(([k, v]) => (
                      <tr key={k} className="border-t border-white/5">
                        <td className="px-3 py-2 font-mono text-slate-300">{k}</td>
                        <td className="px-3 py-2 font-mono text-blue-300">{JSON.stringify(v)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Paper reference */}
          {architecture.paper_reference && (
            <div className="mb-4 flex items-start gap-2 rounded-xl border border-white/10 bg-white/3 p-3">
              <svg className="mt-0.5 h-4 w-4 shrink-0 text-slate-500" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.747 0 3.332.477 4.5 1.253v13C19.832 18.477 18.247 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              <p className="text-xs text-slate-400">{architecture.paper_reference}</p>
            </div>
          )}

          {/* Feedback */}
          <FeedbackPanel archId={architecture.id} />
        </div>
      </div>
    </div>
  )
}
