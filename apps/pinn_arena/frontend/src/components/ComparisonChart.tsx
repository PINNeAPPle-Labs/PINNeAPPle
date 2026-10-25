import React, { useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, ZAxis,
  Cell,
} from 'recharts'
import type { ModelResult } from '../types'

const MODEL_COLORS = [
  '#2563eb', '#059669', '#d97706', '#dc2626', '#a855f7',
  '#0891b2', '#ea580c', '#db2777', '#ca8a04', '#0d9488',
  '#7c3aed', '#65a30d',
]

interface LossCurvesChartProps {
  modelResults: ModelResult[]
}

export function LossCurvesChart({ modelResults }: LossCurvesChartProps) {
  const data = useMemo(() => {
    if (!modelResults.length) return []
    const maxLen = Math.max(...modelResults.map((m) => m.training_history.length))
    return Array.from({ length: maxLen }, (_, i) => {
      const point: Record<string, number | string> = { epoch: i }
      modelResults.forEach((m) => {
        if (m.training_history[i]) {
          point[m.architecture_name] = m.training_history[i].total_loss
        }
      })
      return point
    })
  }, [modelResults])

  return (
    <ResponsiveContainer width="100%" height={340}>
      <LineChart data={data} margin={{ top: 5, right: 20, bottom: 5, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="epoch"
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={{ stroke: '#e2e8f0' }}
          tickLine={false}
          label={{ value: 'Epoch', position: 'insideBottom', offset: -2, fill: '#64748b', fontSize: 12 }}
        />
        <YAxis
          scale="log"
          domain={['auto', 'auto']}
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={{ stroke: '#e2e8f0' }}
          tickLine={false}
          tickFormatter={(v: number) => v.toExponential(1)}
          label={{ value: 'Loss (log)', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }}
        />
        <Tooltip
          contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 8, color: '#0f172a', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
          labelStyle={{ color: '#64748b' }}
          formatter={(v: number) => [v.toExponential(4), '']}
        />
        <Legend wrapperStyle={{ color: '#64748b', fontSize: 12 }} />
        {modelResults.map((m, i) => (
          <Line
            key={m.architecture_id}
            type="monotone"
            dataKey={m.architecture_name}
            stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

interface ErrorDistributionChartProps {
  modelResults: ModelResult[]
}

export function ErrorDistributionChart({ modelResults }: ErrorDistributionChartProps) {
  const data = modelResults.map((m, i) => ({
    name: m.architecture_name,
    l2_error: m.metrics.l2_relative_error,
    rmse: m.metrics.rmse,
    max_error: m.metrics.max_error,
    color: MODEL_COLORS[i % MODEL_COLORS.length],
  }))

  return (
    <ResponsiveContainer width="100%" height={340}>
      <BarChart data={data} margin={{ top: 10, right: 20, bottom: 60, left: 10 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
        <XAxis
          dataKey="name"
          tick={{ fill: '#64748b', fontSize: 10 }}
          axisLine={{ stroke: '#e2e8f0' }}
          tickLine={false}
          angle={-35}
          textAnchor="end"
          interval={0}
        />
        <YAxis
          tick={{ fill: '#64748b', fontSize: 11 }}
          axisLine={{ stroke: '#e2e8f0' }}
          tickLine={false}
          tickFormatter={(v: number) => v.toExponential(1)}
        />
        <Tooltip
          contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 8, color: '#0f172a', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
          formatter={(v: number) => [v.toExponential(4), '']}
        />
        <Legend wrapperStyle={{ color: '#64748b', fontSize: 12 }} />
        <Bar dataKey="l2_error" name="L2 Error" radius={[3, 3, 0, 0]}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.color} fillOpacity={0.85} />
          ))}
        </Bar>
        <Bar dataKey="rmse" name="RMSE" fill="#059669" fillOpacity={0.7} radius={[3, 3, 0, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}

interface ScatterPlotProps {
  modelResults: ModelResult[]
}

export function TrainingScatterPlot({ modelResults }: ScatterPlotProps) {
  const data = modelResults.map((m, i) => ({
    name: m.architecture_name,
    x: m.training_time_s,
    y: m.metrics.l2_relative_error,
    color: MODEL_COLORS[i % MODEL_COLORS.length],
  }))

  // ── Single-model view: metrics overview bar chart ─────────────────────────
  if (modelResults.length === 1) {
    const m = modelResults[0]
    const metricRows = [
      { label: 'L2 Relative Error', value: m.metrics.l2_relative_error, fmt: (v: number) => v.toExponential(4), lowerIsBetter: true },
      { label: 'RMSE',              value: m.metrics.rmse,               fmt: (v: number) => v.toExponential(4), lowerIsBetter: true },
      { label: 'Max Error',         value: m.metrics.max_error,          fmt: (v: number) => v.toExponential(4), lowerIsBetter: true },
      { label: 'R² Score',          value: m.metrics.r2_score,           fmt: (v: number) => v.toFixed(6),       lowerIsBetter: false },
      { label: 'Training Time',     value: m.training_time_s,            fmt: (v: number) => `${v.toFixed(1)}s`, lowerIsBetter: true },
    ]
    return (
      <div className="space-y-3 py-2">
        <p className="text-xs text-text-muted text-center mb-4">
          Scatter plot requires 2+ models for comparison. Showing single-model metrics overview.
        </p>
        {metricRows.map(({ label, value, fmt, lowerIsBetter }) => {
          const isGood = lowerIsBetter ? value < 0.05 : value > 0.9
          const color  = isGood ? '#059669' : value < 0.2 || (!lowerIsBetter && value > 0.5) ? '#d97706' : '#dc2626'
          return (
            <div key={label} className="flex items-center gap-3">
              <div className="w-36 text-xs text-text-muted text-right flex-shrink-0">{label}</div>
              <div className="flex-1 h-6 bg-slate-100 rounded-md overflow-hidden relative">
                <div
                  className="h-full rounded-md transition-all"
                  style={{
                    width: label === 'R² Score'
                      ? `${Math.max(0, Math.min(100, value * 100))}%`
                      : `${Math.max(4, Math.min(100, (1 - Math.min(value / 0.5, 1)) * 100))}%`,
                    backgroundColor: color,
                    opacity: 0.7,
                  }}
                />
              </div>
              <div className="w-24 text-xs font-mono text-right flex-shrink-0" style={{ color }}>{fmt(value)}</div>
            </div>
          )
        })}
      </div>
    )
  }

  // ── Multi-model scatter: time vs accuracy with padded domain ─────────────
  const times  = data.map((d) => d.x)
  const errors = data.map((d) => d.y)
  const tPad = (Math.max(...times)  - Math.min(...times))  * 0.2 || 5
  const ePad = (Math.max(...errors) - Math.min(...errors)) * 0.2 || 0.005
  const xDomain: [number, number] = [Math.max(0, Math.min(...times) - tPad), Math.max(...times) + tPad]
  const yDomain: [number, number] = [Math.max(0, Math.min(...errors) - ePad), Math.max(...errors) + ePad]

  return (
    <div className="space-y-4">
      <ResponsiveContainer width="100%" height={320}>
        <ScatterChart margin={{ top: 20, right: 30, bottom: 30, left: 20 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
          <XAxis
            type="number"
            dataKey="x"
            name="Training Time"
            domain={xDomain}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
            tickFormatter={(v: number) => `${v.toFixed(0)}s`}
            label={{ value: 'Training Time (s)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 12 }}
          />
          <YAxis
            type="number"
            dataKey="y"
            name="L2 Error"
            domain={yDomain}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={{ stroke: '#e2e8f0' }}
            tickLine={false}
            tickFormatter={(v: number) => v.toExponential(1)}
            label={{ value: 'L2 Relative Error', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 12 }}
          />
          <ZAxis range={[80, 80]} />
          <Tooltip
            cursor={{ stroke: '#2563eb40' }}
            contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 8, color: '#0f172a', boxShadow: '0 4px 12px rgba(0,0,0,0.08)' }}
            formatter={(v: number, name: string) => [
              name === 'y' ? v.toExponential(4) : `${v.toFixed(1)}s`,
              name === 'y' ? 'L2 Error' : 'Training Time',
            ]}
          />
          {data.map((entry, i) => (
            <Scatter
              key={i}
              name={entry.name}
              data={[entry]}
              fill={entry.color}
              fillOpacity={0.85}
              shape={(props: any) => {
                const { cx, cy, fill } = props
                const s = 14
                return <rect x={cx - s / 2} y={cy - s / 2} width={s} height={s} fill={fill} fillOpacity={0.85} rx={2} />
              }}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      <div className="flex flex-wrap gap-3 justify-center">
        {data.map((d) => (
          <div key={d.name} className="flex items-center gap-1.5 text-xs text-text-secondary">
            <div className="w-3 h-3 rounded-full" style={{ backgroundColor: d.color }} />
            <span>{d.name}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

interface FieldComparisonChartProps {
  modelResults: ModelResult[]
}

export function FieldComparisonChart({ modelResults }: FieldComparisonChartProps) {
  const modelsWithHistory = modelResults.filter((m) => m.training_history.length > 0)

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {modelsWithHistory.map((m, i) => {
        const data = m.training_history.map((h, idx) => ({
          x: idx,
          predicted: h.total_loss,
          pde: h.pde_loss ?? 0,
          bc: h.bc_loss ?? 0,
        }))

        return (
          <div key={m.architecture_id} className="bg-slate-50 rounded-xl border border-slate-200 p-4">
            <h4
              className="text-xs font-semibold mb-3 truncate"
              style={{ color: MODEL_COLORS[i % MODEL_COLORS.length] }}
            >
              {m.architecture_name}
            </h4>
            <ResponsiveContainer width="100%" height={140}>
              <LineChart data={data} margin={{ top: 5, right: 5, bottom: 5, left: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="x" tick={{ fill: '#94a3b8', fontSize: 9 }} axisLine={false} tickLine={false} />
                <YAxis
                  tick={{ fill: '#94a3b8', fontSize: 9 }}
                  axisLine={false}
                  tickLine={false}
                  tickFormatter={(v: number) => v.toExponential(1)}
                />
                <Tooltip
                  contentStyle={{ background: '#ffffff', border: '1px solid #e2e8f0', borderRadius: 6, color: '#0f172a', fontSize: 11, boxShadow: '0 2px 8px rgba(0,0,0,0.06)' }}
                  formatter={(v: number) => [v.toExponential(4), '']}
                />
                <Line
                  type="monotone"
                  dataKey="predicted"
                  name="Total Loss"
                  stroke={MODEL_COLORS[i % MODEL_COLORS.length]}
                  strokeWidth={1.5}
                  dot={false}
                />
                {data[0]?.pde > 0 && (
                  <Line
                    type="monotone"
                    dataKey="pde"
                    name="PDE Loss"
                    stroke="#d97706"
                    strokeWidth={1}
                    dot={false}
                    strokeDasharray="3 3"
                  />
                )}
              </LineChart>
            </ResponsiveContainer>
            <div className="flex justify-between text-xs mt-2">
              <span className="text-text-muted">L2: <span className="text-success font-mono">{m.metrics.l2_relative_error.toExponential(3)}</span></span>
              <span className="text-text-muted">R²: <span className="text-accent font-mono">{m.metrics.r2_score.toFixed(4)}</span></span>
            </div>
          </div>
        )
      })}
    </div>
  )
}
