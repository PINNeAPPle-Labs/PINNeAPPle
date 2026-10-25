import React, { useMemo } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, Grid } from '@react-three/drei'
import * as THREE from 'three'

// ─── Colormaps ────────────────────────────────────────────────────────────────

function rdbu(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  const blue: [number, number, number] = [33, 102, 172]
  const white: [number, number, number] = [247, 247, 247]
  const red: [number, number, number] = [178, 24, 43]
  const [a, b, s] = t <= 0.5 ? [blue, white, t * 2] : [white, red, (t - 0.5) * 2]
  return [Math.round(a[0]+s*(b[0]-a[0])), Math.round(a[1]+s*(b[1]-a[1])), Math.round(a[2]+s*(b[2]-a[2]))]
}

function viridis(t: number): [number, number, number] {
  t = Math.max(0, Math.min(1, t))
  const stops: [number, number, number][] = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]]
  const seg = t * 4; const i = Math.min(Math.floor(seg), 3); const s = seg - i
  const a = stops[i], b = stops[i+1]
  return [Math.round(a[0]+s*(b[0]-a[0])), Math.round(a[1]+s*(b[1]-a[1])), Math.round(a[2]+s*(b[2]-a[2]))]
}

export type ColormapName = 'rdbu' | 'viridis' | 'auto'

function sampleColormap(t: number, name: ColormapName): [number, number, number] {
  if (name === 'viridis') return viridis(t)
  return rdbu(t)
}

function resolveColormap(name: ColormapName, field: number[]): 'rdbu' | 'viridis' {
  if (name !== 'auto') return name
  const finite = field.filter(isFinite)
  const vmin = finite.length ? Math.min(...finite) : 0
  return vmin < -0.05 ? 'rdbu' : 'viridis'
}

// ─── Colorbar ─────────────────────────────────────────────────────────────────

function Colorbar({ vmin, vmax, colormap, fieldLabel, height }: {
  vmin: number; vmax: number; colormap: ColormapName; fieldLabel: string; height: number
}) {
  const stops = Array.from({ length: 12 }, (_, i) => {
    const t = i / 11
    const [r, g, b] = sampleColormap(1 - t, colormap)
    return `rgb(${r},${g},${b}) ${t * 100}%`
  }).join(', ')

  const ticks = [1, 0.75, 0.5, 0.25, 0]
  const fmt = (v: number) =>
    Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(1) : v.toFixed(3)

  const barH = height * 0.60

  return (
    <div style={{
      position: 'absolute', right: 10, top: '50%', transform: 'translateY(-50%)',
      display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4,
      pointerEvents: 'none',
    }}>
      <span style={{ fontSize: 9, color: '#94a3b8', fontFamily: 'monospace', marginBottom: 2 }}>
        {fieldLabel}
      </span>
      <div style={{ display: 'flex', flexDirection: 'row', alignItems: 'stretch', gap: 5 }}>
        <div style={{
          width: 14, height: barH,
          background: `linear-gradient(to bottom, ${stops})`,
          borderRadius: 3,
          border: '1px solid rgba(255,255,255,0.12)',
          flexShrink: 0,
        }} />
        <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', height: barH }}>
          {ticks.map((t) => (
            <span key={t} style={{ fontSize: 8, color: '#94a3b8', fontFamily: 'monospace', whiteSpace: 'nowrap', lineHeight: 1 }}>
              {fmt(vmin + t * (vmax - vmin))}
            </span>
          ))}
        </div>
      </div>
    </div>
  )
}

// ─── Colored surface ─────────────────────────────────────────────────────────

function ColoredSurface({
  vertices, faces, field, colormap, vmin, vmax,
}: {
  vertices: number[][]
  faces: number[][]
  field: number[]
  colormap: ColormapName
  vmin: number
  vmax: number
}) {
  const geometry = useMemo(() => {
    const n = vertices.length
    const pos = new Float32Array(n * 3)
    const col = new Float32Array(n * 3)
    const range = vmax === vmin ? 1 : vmax - vmin

    for (let i = 0; i < n; i++) {
      pos[i * 3]     = vertices[i][0]
      pos[i * 3 + 1] = vertices[i][1]
      pos[i * 3 + 2] = vertices[i][2]
      const val = isFinite(field[i]) ? field[i] : vmin
      const t = Math.max(0, Math.min(1, (val - vmin) / range))
      const [r, g, b] = sampleColormap(t, colormap)
      col[i * 3]     = r / 255
      col[i * 3 + 1] = g / 255
      col[i * 3 + 2] = b / 255
    }

    const idx = new Uint32Array(faces.length * 3)
    for (let f = 0; f < faces.length; f++) {
      idx[f * 3]     = faces[f][0]
      idx[f * 3 + 1] = faces[f][1]
      idx[f * 3 + 2] = faces[f][2]
    }

    const geo = new THREE.BufferGeometry()
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
    geo.setAttribute('color',    new THREE.BufferAttribute(col, 3))
    geo.setIndex(new THREE.BufferAttribute(idx, 1))
    geo.computeVertexNormals()
    return geo
  }, [vertices, faces, field, colormap, vmin, vmax])

  return (
    <mesh geometry={geometry}>
      {/* Lambert shading: field colors modulated by light — gives 3D depth */}
      <meshLambertMaterial
        vertexColors
        side={THREE.DoubleSide}
        polygonOffset
        polygonOffsetFactor={1}
        polygonOffsetUnits={1}
      />
    </mesh>
  )
}

// ─── Box wireframe edges ──────────────────────────────────────────────────────

function BoxEdges({ vertices }: { vertices: number[][] }) {
  const geo = useMemo(() => {
    if (!vertices.length) return null
    let xmn = Infinity, xmx = -Infinity, ymn = Infinity, ymx = -Infinity, zmn = Infinity, zmx = -Infinity
    for (const v of vertices) {
      xmn = Math.min(xmn, v[0]); xmx = Math.max(xmx, v[0])
      ymn = Math.min(ymn, v[1]); ymx = Math.max(ymx, v[1])
      zmn = Math.min(zmn, v[2]); zmx = Math.max(zmx, v[2])
    }
    const box = new THREE.BoxGeometry(xmx - xmn, ymx - ymn, zmx - zmn)
    const edges = new THREE.EdgesGeometry(box)
    edges.translate((xmn + xmx) / 2, (ymn + ymx) / 2, (zmn + zmx) / 2)
    return edges
  }, [vertices])
  if (!geo) return null
  return (
    <lineSegments geometry={geo}>
      <lineBasicMaterial color="#0d1f38" />
    </lineSegments>
  )
}

// ─── Props & main export ──────────────────────────────────────────────────────

export interface SurfaceMeshData {
  vertices: number[][]
  faces: number[][]
  field: number[]
  field_name?: string
}

export interface SurfaceMeshViewProps {
  mesh: SurfaceMeshData
  colormap?: ColormapName
  title?: string
  fieldLabel?: string
  width?: number
  height?: number
  vmin?: number
  vmax?: number
  cameraPos?: [number, number, number]
}

export function SurfaceMeshView({
  mesh,
  colormap = 'auto',
  title = '',
  fieldLabel,
  width = 560,
  height = 420,
  vmin: vminProp,
  vmax: vmaxProp,
  cameraPos,
}: SurfaceMeshViewProps) {
  const { vertices, faces, field } = mesh

  const [vminAuto, vmaxAuto] = useMemo(() => {
    const finite = field.filter(isFinite)
    if (!finite.length) return [-1, 1]
    const mn = Math.min(...finite), mx = Math.max(...finite)
    return mn === mx ? [mn - 0.5, mx + 0.5] : [mn, mx]
  }, [field])

  const vmin = vminProp !== undefined ? vminProp : vminAuto
  const vmax = vmaxProp !== undefined ? vmaxProp : vmaxAuto

  const effectiveColormap = useMemo(
    () => resolveColormap(colormap, field),
    [colormap, field],
  )

  const fLabel = fieldLabel ?? mesh.field_name ?? 'Field'

  // Bounding box → camera + grid params
  const { camPos, target, gridY, gridSize, gridCell } = useMemo<{
    camPos: [number, number, number]
    target: [number, number, number]
    gridY: number
    gridSize: number
    gridCell: number
  }>(() => {
    if (!vertices.length) return {
      camPos: [4, 3, 4], target: [0, 0, 0], gridY: -1, gridSize: 10, gridCell: 1,
    }
    let xmn = Infinity, ymn = Infinity, zmn = Infinity
    let xmx = -Infinity, ymx = -Infinity, zmx = -Infinity
    for (const v of vertices) {
      xmn = Math.min(xmn, v[0]); xmx = Math.max(xmx, v[0])
      ymn = Math.min(ymn, v[1]); ymx = Math.max(ymx, v[1])
      zmn = Math.min(zmn, v[2]); zmx = Math.max(zmx, v[2])
    }
    const cx = (xmn + xmx) / 2, cy = (ymn + ymx) / 2, cz = (zmn + zmx) / 2
    const spanX = xmx - xmn, spanY = ymx - ymn, spanZ = zmx - zmn
    const span = Math.max(spanX, spanY, spanZ, 0.1)

    // Camera: isometric diagonal outside the full 6-face box, equal distance
    // from all visible faces so every panel gets equal visual weight
    const camX = cx + span * 1.1
    const camY = cy + span * 1.1
    const camZ = cz + span * 1.1
    if (cameraPos) return {
      camPos: cameraPos, target: [cx, cy, cz],
      gridY: ymn - span * 0.08, gridSize: span * 5, gridCell: span / 4,
    }
    return {
      camPos: [camX, camY, camZ],
      target: [cx, cy, cz],
      gridY: ymn - span * 0.08,
      gridSize: span * 5,
      gridCell: span / 4,
    }
  }, [vertices, cameraPos])

  if (!vertices.length || !faces.length) return null

  return (
    <div style={{
      position: 'relative',
      width,
      height,
      borderRadius: 10,
      border: '1px solid rgba(255,255,255,0.08)',
      overflow: 'hidden',
      background: '#060d18',
    }}>
      {title && (
        <div style={{
          position: 'absolute', top: 8, left: 0, right: 60, textAlign: 'center',
          fontSize: 11, fontWeight: 600, color: '#e2e8f0',
          fontFamily: 'Inter, sans-serif', zIndex: 2, pointerEvents: 'none',
          textShadow: '0 1px 4px rgba(0,0,0,0.8)',
        }}>
          {title}
        </div>
      )}

      <Canvas
        camera={{ position: camPos, fov: 50 }}
        style={{ width: '100%', height: '100%' }}
        gl={{ antialias: true }}
      >
        <color attach="background" args={['#060d18']} />

        {/* Lighting — gives each face a different brightness for 3D depth */}
        <ambientLight intensity={0.55} />
        <directionalLight position={[2, 3, 2]} intensity={0.9} />
        <directionalLight position={[-1.5, -1, -1]} intensity={0.2} color="#2244aa" />

        <ColoredSurface
          vertices={vertices}
          faces={faces}
          field={field}
          colormap={effectiveColormap}
          vmin={vmin}
          vmax={vmax}
        />
        <BoxEdges vertices={vertices} />

        {/* Floor grid — infinite, like the reference */}
        <Grid
          position={[target[0], gridY, target[2]]}
          args={[gridSize, gridSize]}
          cellSize={gridCell}
          cellThickness={0.4}
          cellColor="#162840"
          sectionSize={gridCell * 4}
          sectionThickness={0.8}
          sectionColor="#1a3555"
          fadeDistance={gridSize * 1.5}
          fadeStrength={2.5}
          infiniteGrid
        />

        <OrbitControls
          target={target}
          enableZoom
          enablePan
          enableRotate
          makeDefault
          enableDamping
          dampingFactor={0.10}
        />
      </Canvas>

      <Colorbar vmin={vmin} vmax={vmax} colormap={effectiveColormap} fieldLabel={fLabel} height={height} />

      <div style={{
        position: 'absolute', bottom: 6, left: 8,
        fontSize: 9, color: '#334155', fontFamily: 'Inter, sans-serif', pointerEvents: 'none',
      }}>
        drag · scroll · right-click pan
      </div>
    </div>
  )
}
