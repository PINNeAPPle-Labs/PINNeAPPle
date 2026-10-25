import React, { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls } from '@react-three/drei'
import * as THREE from 'three'

// ─── Colormaps ────────────────────────────────────────────────────────────────

function rdbu(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  const blue: [number, number, number] = [33, 102, 172]
  const white: [number, number, number] = [247, 247, 247]
  const red: [number, number, number] = [178, 24, 43]
  const [a, b, s] = t <= 0.5 ? [blue, white, t * 2] : [white, red, (t - 0.5) * 2]
  return [Math.round(a[0] + s * (b[0] - a[0])), Math.round(a[1] + s * (b[1] - a[1])), Math.round(a[2] + s * (b[2] - a[2]))]
}

function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  const stops: [number, number, number][] = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]]
  const seg = t * 4
  const i = Math.min(Math.floor(seg), 3)
  const s = seg - i
  const a = stops[i], b = stops[i + 1]
  return [Math.round(a[0] + s * (b[0] - a[0])), Math.round(a[1] + s * (b[1] - a[1])), Math.round(a[2] + s * (b[2] - a[2]))]
}

export type ColormapName = 'rdbu' | 'viridis'

function sampleColormap(t: number, name: ColormapName): [number, number, number] {
  return name === 'rdbu' ? rdbu(t) : viridis(t)
}

// ─── Colorbar (HTML overlay) ──────────────────────────────────────────────────

function Colorbar({
  vmin, vmax, colormap, fieldLabel, height,
}: {
  vmin: number; vmax: number; colormap: ColormapName; fieldLabel: string; height: number
}) {
  const gradientStops = Array.from({ length: 12 }, (_, i) => {
    const t = i / 11
    const [r, g, b] = sampleColormap(1 - t, colormap)
    return `rgb(${r},${g},${b}) ${t * 100}%`
  }).join(', ')

  const ticks = [1, 0.75, 0.5, 0.25, 0]
  const fmt = (v: number) =>
    Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(3)

  const barH = height * 0.6

  return (
    <div style={{
      position: 'absolute', right: 8, top: '50%', transform: 'translateY(-50%)',
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
      pointerEvents: 'none',
    }}>
      <span style={{ fontSize: 9, color: '#64748b', fontFamily: 'monospace', marginBottom: 2 }}>
        {fieldLabel}
      </span>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'stretch', gap: 5 }}>
        <div style={{
          width: 12, height: barH,
          background: `linear-gradient(to bottom, ${gradientStops})`,
          borderRadius: 3,
          border: '1px solid #cbd5e1',
          flexShrink: 0,
        }} />
        <div style={{
          display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: barH,
        }}>
          {ticks.map((t) => (
            <span key={t} style={{ fontSize: 8, color: '#64748b', fontFamily: 'monospace', whiteSpace: 'nowrap', lineHeight: 1 }}>
              {fmt(vmin + t * (vmax - vmin))}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Point Cloud (Three.js geometry) ─────────────────────────────────────────

function PointCloudMesh({
  coords, values, colormap, vmin, vmax,
}: {
  coords: number[][]
  values: number[]
  colormap: ColormapName
  vmin: number
  vmax: number
}) {
  const geometry = useMemo(() => {
    const n = coords.length
    const positions = new Float32Array(n * 3)
    const colors = new Float32Array(n * 3)
    const range = vmax === vmin ? 1 : vmax - vmin

    // Compute bounds for normalization into [-1, 1]³
    let xmin = Infinity, xmax = -Infinity
    let ymin = Infinity, ymax = -Infinity
    let zmin = Infinity, zmax = -Infinity
    for (const c of coords) {
      const x = c[0], y = c[1], z = c[2] ?? 0
      if (x < xmin) xmin = x; if (x > xmax) xmax = x
      if (y < ymin) ymin = y; if (y > ymax) ymax = y
      if (z < zmin) zmin = z; if (z > zmax) zmax = z
    }

    const norm = (v: number, mn: number, mx: number) =>
      mx === mn ? 0 : ((v - mn) / (mx - mn)) * 2 - 1

    for (let i = 0; i < n; i++) {
      const c = coords[i]
      positions[i * 3]     = norm(c[0], xmin, xmax)
      positions[i * 3 + 1] = norm(c[1], ymin, ymax)
      positions[i * 3 + 2] = norm(c[2] ?? 0, zmin, zmax)

      const val = isFinite(values[i]) ? values[i] : vmin
      const t = Math.max(0, Math.min(1, (val - vmin) / range))
      const [r, g, b] = sampleColormap(t, colormap)
      colors[i * 3]     = r / 255
      colors[i * 3 + 1] = g / 255
      colors[i * 3 + 2] = b / 255
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(positions, 3))
    geo.setAttribute('color', new THREE.BufferAttribute(colors, 3))
    return geo
  }, [coords, values, colormap, vmin, vmax])

  // Scale point size inversely with √N so it looks good at all sample counts
  const pointSize = Math.max(0.008, Math.min(0.06, 60 / Math.sqrt(coords.length)))

  return (
    <points geometry={geometry}>
      <pointsMaterial vertexColors size={pointSize} sizeAttenuation transparent opacity={0.9} />
    </points>
  )
}

// ─── Bounding box wireframe ───────────────────────────────────────────────────

function BoundingBoxWireframe() {
  const edges = useMemo(() => new THREE.EdgesGeometry(new THREE.BoxGeometry(2, 2, 2)), [])
  return (
    <lineSegments geometry={edges}>
      <lineBasicMaterial color="#94a3b8" transparent opacity={0.45} />
    </lineSegments>
  )
}

// ─── Props & main export ──────────────────────────────────────────────────────

export interface Dataset3DViewProps {
  coords: number[][]
  values: number[]
  xLabel?: string
  yLabel?: string
  zLabel?: string
  fieldLabel?: string
  title?: string
  colormap?: ColormapName
  width?: number
  height?: number
}

export function Dataset3DView({
  coords,
  values,
  fieldLabel = 'u',
  title = '',
  colormap = 'rdbu',
  width = 540,
  height = 400,
}: Dataset3DViewProps) {
  const [vmin, vmax] = useMemo(() => {
    if (!values.length) return [0, 1]
    const finite = values.filter(isFinite)
    if (!finite.length) return [0, 1]
    const mn = Math.min(...finite), mx = Math.max(...finite)
    return mn === mx ? [mn - 0.5, mx + 0.5] : [mn, mx]
  }, [values])

  if (!coords.length) return null

  return (
    <div style={{
      position: 'relative',
      width,
      height,
      borderRadius: 10,
      border: '1px solid #e2e8f0',
      overflow: 'hidden',
      background: '#f8fafc',
    }}>
      {/* Title */}
      {title && (
        <div style={{
          position: 'absolute', top: 8, left: 0, right: 50, textAlign: 'center',
          fontSize: 11, fontWeight: 600, color: '#475569',
          fontFamily: 'Inter, sans-serif', zIndex: 2, pointerEvents: 'none',
        }}>
          {title}
        </div>
      )}

      {/* Three.js canvas — fills the container */}
      <Canvas
        camera={{ position: [2.5, 1.8, 2.5], fov: 40 }}
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={['#f8fafc']} />
        <ambientLight intensity={0.9} />
        <directionalLight position={[5, 5, 5]} intensity={0.3} />

        <PointCloudMesh
          coords={coords}
          values={values}
          colormap={colormap}
          vmin={vmin}
          vmax={vmax}
        />
        <BoundingBoxWireframe />
        <axesHelper args={[1.15]} />
        <OrbitControls enableZoom enablePan enableRotate makeDefault dampingFactor={0.15} enableDamping />
      </Canvas>

      {/* Colorbar overlay */}
      <Colorbar vmin={vmin} vmax={vmax} colormap={colormap} fieldLabel={fieldLabel} height={height} />

      {/* Interaction hint */}
      <div style={{
        position: 'absolute', bottom: 6, left: 8,
        fontSize: 9, color: '#94a3b8', fontFamily: 'Inter, sans-serif', pointerEvents: 'none',
      }}>
        drag · scroll · right-click pan
      </div>
    </div>
  )
}
