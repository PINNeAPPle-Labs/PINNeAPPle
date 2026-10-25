import React, { useMemo } from 'react'

// ─── Colormaps ────────────────────────────────────────────────────────────────

// RdBu diverging: Blue → White → Red (for signed quantities: velocity, displacement)
function rdbu(t: number): string {
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
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`
}

// Viridis-like: dark purple → blue → teal → green → yellow (for positive: temperature, pressure)
function viridis(t: number): string {
  t = Math.max(0, Math.min(1, t))
  const stops = [
    [68, 1, 84],
    [59, 82, 139],
    [33, 145, 140],
    [94, 201, 98],
    [253, 231, 37],
  ]
  const seg = t * (stops.length - 1)
  const i = Math.min(Math.floor(seg), stops.length - 2)
  const s = seg - i
  const a = stops[i], b = stops[i + 1]
  return `rgb(${Math.round(a[0] + s*(b[0]-a[0]))},${Math.round(a[1] + s*(b[1]-a[1]))},${Math.round(a[2] + s*(b[2]-a[2]))})`
}

// Inferno: for error/residual maps (black → deep red → orange → yellow)
function inferno(t: number): string {
  t = Math.max(0, Math.min(1, t))
  const stops = [
    [0, 0, 4],
    [87, 16, 110],
    [188, 55, 84],
    [249, 142, 9],
    [252, 255, 164],
  ]
  const seg = t * (stops.length - 1)
  const i = Math.min(Math.floor(seg), stops.length - 2)
  const s = seg - i
  const a = stops[i], b = stops[i + 1]
  return `rgb(${Math.round(a[0] + s*(b[0]-a[0]))},${Math.round(a[1] + s*(b[1]-a[1]))},${Math.round(a[2] + s*(b[2]-a[2]))})`
}

export type ColormapName = 'rdbu' | 'viridis' | 'inferno'

function applyColormap(t: number, name: ColormapName): string {
  if (name === 'rdbu') return rdbu(t)
  if (name === 'viridis') return viridis(t)
  return inferno(t)
}

function formatTick(v: number): string {
  if (Math.abs(v) === 0) return '0'
  if (Math.abs(v) >= 100 || Math.abs(v) < 0.01) return v.toExponential(1)
  return v.toFixed(Math.abs(v) < 1 ? 3 : 2)
}

// ─── ScientificHeatmap ────────────────────────────────────────────────────────

export interface ScientificHeatmapProps {
  /** Row-major 2D grid of values: data[row][col] */
  data: number[][]
  /** Values for the horizontal axis (length = ncols) */
  xValues: number[]
  /** Values for the vertical axis (length = nrows), rendered bottom-up */
  yValues: number[]
  xLabel?: string
  yLabel?: string
  title?: string
  colormap?: ColormapName
  /** Override color range; defaults to [min, max] of data */
  vmin?: number
  vmax?: number
  /** If true, use symmetric color range [-absmax, absmax] */
  symmetric?: boolean
  width?: number
  height?: number
  /** Number of x-axis ticks to show */
  nTicksX?: number
  /** Number of y-axis ticks to show */
  nTicksY?: number
  /** Number of colorbar ticks */
  nColorbarTicks?: number
}

export function ScientificHeatmap({
  data,
  xValues,
  yValues,
  xLabel = 'x',
  yLabel = 'y',
  title = '',
  colormap = 'rdbu',
  vmin,
  vmax,
  symmetric = false,
  width = 520,
  height = 300,
  nTicksX = 5,
  nTicksY = 5,
  nColorbarTicks = 5,
}: ScientificHeatmapProps) {
  // padding (left for y-axis, right for colorbar, top for title, bottom for x-axis)
  const pad = { left: 48, right: 72, top: title ? 32 : 16, bottom: 44 }
  const plotW = width - pad.left - pad.right
  const plotH = height - pad.top - pad.bottom

  const nRows = data.length
  const nCols = data[0]?.length ?? 0

  const { minVal, maxVal } = useMemo(() => {
    let mn = Infinity, mx = -Infinity
    for (const row of data) for (const v of row) {
      if (isFinite(v)) { mn = Math.min(mn, v); mx = Math.max(mx, v) }
    }
    if (!isFinite(mn)) { mn = 0; mx = 1 }
    if (mn === mx) { mn -= 0.5; mx += 0.5 }
    return { minVal: mn, maxVal: mx }
  }, [data])

  const lo = vmin ?? (symmetric ? -Math.max(Math.abs(minVal), Math.abs(maxVal)) : minVal)
  const hi = vmax ?? (symmetric ? Math.max(Math.abs(minVal), Math.abs(maxVal)) : maxVal)
  const range = hi - lo || 1

  const cellW = plotW / Math.max(nCols, 1)
  const cellH = plotH / Math.max(nRows, 1)

  // Colorbar
  const cbX = pad.left + plotW + 10
  const cbW = 12
  const cbH = plotH
  const cbY = pad.top

  // Axis ticks
  const xTicks = useMemo(() => {
    const step = Math.max(1, Math.floor(nCols / nTicksX))
    const indices: number[] = []
    for (let i = 0; i < nCols; i += step) indices.push(i)
    if (indices[indices.length - 1] !== nCols - 1) indices.push(nCols - 1)
    return indices
  }, [nCols, nTicksX])

  const yTicks = useMemo(() => {
    const step = Math.max(1, Math.floor(nRows / nTicksY))
    const indices: number[] = []
    for (let i = 0; i < nRows; i += step) indices.push(i)
    if (indices[indices.length - 1] !== nRows - 1) indices.push(nRows - 1)
    return indices
  }, [nRows, nTicksY])

  const colorbarTicks = useMemo(() => {
    const ticks: number[] = []
    for (let i = 0; i <= nColorbarTicks; i++) {
      ticks.push(lo + (hi - lo) * i / nColorbarTicks)
    }
    return ticks
  }, [lo, hi, nColorbarTicks])

  return (
    <svg width={width} height={height} style={{ fontFamily: "'JetBrains Mono', monospace", overflow: 'visible' }}>
      {/* Title */}
      {title && (
        <text x={pad.left + plotW / 2} y={18} textAnchor="middle" fontSize={11} fill="#475569" fontWeight={600}>
          {title}
        </text>
      )}

      {/* Heatmap cells */}
      <g>
        {data.map((row, ri) =>
          row.map((val, ci) => {
            const t = isFinite(val) ? (val - lo) / range : 0.5
            return (
              <rect
                key={`${ri}-${ci}`}
                x={pad.left + ci * cellW}
                y={pad.top + (nRows - 1 - ri) * cellH}
                width={cellW + 0.5}
                height={cellH + 0.5}
                fill={applyColormap(t, colormap)}
              />
            )
          })
        )}
      </g>

      {/* Border around plot */}
      <rect x={pad.left} y={pad.top} width={plotW} height={plotH}
        fill="none" stroke="#cbd5e1" strokeWidth={1} />

      {/* X-axis ticks and labels */}
      {xTicks.map((ci) => {
        const px = pad.left + (ci + 0.5) * cellW
        const val = xValues[ci] ?? ci
        return (
          <g key={`xt-${ci}`}>
            <line x1={px} y1={pad.top + plotH} x2={px} y2={pad.top + plotH + 4} stroke="#94a3b8" strokeWidth={1} />
            <text x={px} y={pad.top + plotH + 14} textAnchor="middle" fontSize={9} fill="#64748b">
              {formatTick(val)}
            </text>
          </g>
        )
      })}
      <text x={pad.left + plotW / 2} y={height - 4} textAnchor="middle" fontSize={10} fill="#475569" fontWeight={500}>
        {xLabel}
      </text>

      {/* Y-axis ticks and labels */}
      {yTicks.map((ri) => {
        const py = pad.top + (nRows - 1 - ri) * cellH + cellH / 2
        const val = yValues[ri] ?? ri
        return (
          <g key={`yt-${ri}`}>
            <line x1={pad.left - 4} y1={py} x2={pad.left} y2={py} stroke="#94a3b8" strokeWidth={1} />
            <text x={pad.left - 6} y={py + 3} textAnchor="end" fontSize={9} fill="#64748b">
              {formatTick(val)}
            </text>
          </g>
        )
      })}
      <text
        transform={`translate(${12}, ${pad.top + plotH / 2}) rotate(-90)`}
        textAnchor="middle" fontSize={10} fill="#475569" fontWeight={500}
      >
        {yLabel}
      </text>

      {/* Colorbar */}
      {/* Gradient fill using many small rects for SVG compatibility */}
      {Array.from({ length: 100 }, (_, i) => {
        const t = 1 - i / 99  // top = max, bottom = min
        return (
          <rect
            key={`cb-${i}`}
            x={cbX} y={cbY + (i / 100) * cbH}
            width={cbW} height={cbH / 100 + 0.5}
            fill={applyColormap(t, colormap)}
          />
        )
      })}
      <rect x={cbX} y={cbY} width={cbW} height={cbH} fill="none" stroke="#cbd5e1" strokeWidth={1} />
      {colorbarTicks.map((v, i) => {
        const py = cbY + cbH * (1 - (v - lo) / range)
        return (
          <g key={`cbt-${i}`}>
            <line x1={cbX + cbW} y1={py} x2={cbX + cbW + 3} y2={py} stroke="#94a3b8" strokeWidth={1} />
            <text x={cbX + cbW + 5} y={py + 3} fontSize={8} fill="#64748b">
              {formatTick(v)}
            </text>
          </g>
        )
      })}
    </svg>
  )
}

// ─── Helper: build heatmap data from flat coords + solution ──────────────────

export function buildHeatmapFromGrid(
  xArr: number[],
  yArr: number[],
  /** 2D array [ny][nx] or [nt][nx] */
  uGrid: number[][],
): { data: number[][]; xValues: number[]; yValues: number[] } {
  return { data: uGrid, xValues: xArr, yValues: yArr }
}

export function buildHeatmapFromScatter(
  coords: number[][],
  solution: number[],
): { data: number[][]; xValues: number[]; yValues: number[] } | null {
  if (!coords.length) return null

  const SIG = 4
  const r4 = (v: number) => +v.toFixed(SIG)

  const xs = [...new Set(coords.map(c => r4(c[0])))].sort((a, b) => a - b)
  const ys = [...new Set(coords.map(c => r4(c[1])))].sort((a, b) => a - b)

  const map = new Map<string, number>()
  coords.forEach((c, i) => map.set(`${r4(c[0])},${r4(c[1])}`, solution[i]))

  // Pre-collect filled coords for nearest-neighbor on sparse scatter data
  const fx: number[] = [], fy: number[] = [], fv: number[] = []
  map.forEach((v, key) => {
    const [sx, sy] = key.split(',')
    fx.push(+sx); fy.push(+sy); fv.push(v)
  })

  const data = ys.map(y => xs.map(x => {
    const hit = map.get(`${x},${y}`)
    if (hit !== undefined) return hit
    let bestDist = Infinity, bestVal = 0
    for (let k = 0; k < fx.length; k++) {
      const dx = fx[k] - x, dy = fy[k] - y
      const d = dx * dx + dy * dy
      if (d < bestDist) { bestDist = d; bestVal = fv[k] }
    }
    return bestVal
  }))

  return { data, xValues: xs, yValues: ys }
}
