import axios from 'axios'
import type {
  PhysicsProblem,
  GeometryPreset,
  ArchitectureInfo,
  ArchitectureLayerSchema,
  ArchitectureFeedbackSummary,
  ExperimentConfig,
  ExperimentResult,
  ExperimentStatusResponse,
  DatasetPreview,
  UploadedDataset,
  PhysicsArea,
  Dimension,
  Difficulty,
  ArchitectureFamily,
  LeaderboardData,
  LeaderboardFilters,
  ExternalCatalogModel,
  ExternalDomainMeta,
  ExternalCategoryMeta,
  ExternalJobResult,
  ExternalModelResult,
  ExternalBenchmarkResult,
  ExternalCompareResult,
  ExternalApiSnippet,
  ExternalCatalogInstallState,
  ExternalDataset,
} from '../types'
import type { AuthUser } from '../store/authStore'

const LOCAL_STORAGE_KEY = 'arena_token'

const api = axios.create({
  baseURL: '/api',
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor — attach the session JWT if present. X-Arena-Token (not Authorization) so the browser keeps
// sending the HTTP Basic login of a private deployment (PINN_ARENA_USER / PINN_ARENA_PASSWORD).
api.interceptors.request.use((config) => {
  const token = localStorage.getItem(LOCAL_STORAGE_KEY)
  if (token) {
    config.headers = config.headers ?? {}
    config.headers['X-Arena-Token'] = token
  }
  return config
})

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => response,
  (error) => {
    console.error('API Error:', error.response?.data || error.message)
    return Promise.reject(error)
  }
)

// ─── Problems ────────────────────────────────────────────────────────────────

export interface ProblemFilters {
  domain?: PhysicsArea
  dimension?: Dimension
  difficulty?: Difficulty
  requires_geometry?: boolean
  tags?: string[]
  search?: string
}

export const problemsApi = {
  list: (filters?: ProblemFilters) =>
    api.get<{ total: number; problems: PhysicsProblem[] }>('/problems', { params: filters })
      .then((r) => ({ ...r, data: r.data.problems })),

  get: (id: string) =>
    api.get<PhysicsProblem>(`/problems/${id}`),

  previewDataset: (id: string, parameters?: Record<string, number | string>) =>
    api.get<DatasetPreview>(`/problems/${id}/dataset-preview`, { params: parameters }),
}

// ─── Geometries ──────────────────────────────────────────────────────────────

export const geometriesApi = {
  list: (problemId?: string) =>
    api.get<{ total: number; geometries: GeometryPreset[] }>('/geometries', { params: { problem_id: problemId } })
      .then((r) => ({ ...r, data: r.data.geometries })),

  get: (id: string) =>
    api.get<GeometryPreset>(`/geometries/${id}`),

  generateMesh: (id: string, resolution: number) =>
    api.post(`/geometries/${id}/mesh`, null, { params: { resolution } }),
}

// ─── Architectures ───────────────────────────────────────────────────────────

export interface ArchitectureFilters {
  family?: ArchitectureFamily
  supports_physics_loss?: boolean
  supports_3d?: boolean
  search?: string
}

export const architecturesApi = {
  list: (filters?: ArchitectureFilters) =>
    api.get<{ total: number; architectures: ArchitectureInfo[] }>('/architectures', { params: filters })
      .then((r) => ({ ...r, data: r.data.architectures })),

  get: (id: string) =>
    api.get<ArchitectureInfo>(`/architectures/${id}`),

  getLayers: (id: string) =>
    api.get<ArchitectureLayerSchema>(`/architectures/${id}/layers`),

  getFeedbackSummary: (id: string) =>
    api.get<ArchitectureFeedbackSummary>(`/architectures/${id}/feedback/summary`),

  submitFeedback: (id: string, is_positive: boolean, comment?: string) =>
    api.post(`/architectures/${id}/feedback`, { is_positive, comment }),

}

// ─── Experiments ─────────────────────────────────────────────────────────────

export const experimentsApi = {
  create: (config: ExperimentConfig) =>
    api.post<{ experiment_id: string }>('/experiments', { config }),

  getStatus: (id: string) =>
    api.get<ExperimentStatusResponse>(`/experiments/${id}/status`),

  getResults: (id: string) =>
    api.get<ExperimentResult>(`/experiments/${id}/results`),

  list: () =>
    api.get<{ total: number; experiments: ExperimentResult[] }>('/experiments')
      .then((r) => ({ ...r, data: r.data.experiments })),

  cancel: (id: string) =>
    api.post(`/experiments/${id}/cancel`),

  getModelPlots: (expId: string, modelId: string) =>
    api.get(`/experiments/${expId}/plots/${modelId}`),

  downloadModel: (expId: string, archId: string) => {
    const url = `/api/experiments/${expId}/models/${archId}/download`
    const a = document.createElement('a')
    a.href = url
    a.download = `${archId}.pt`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  },

  downloadAllModels: (expId: string, problemId: string) => {
    const url = `/api/experiments/${expId}/models/download-all`
    const a = document.createElement('a')
    a.href = url
    a.download = `models_${problemId}_${expId.slice(0, 8)}.zip`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
  },
}

// ─── Datasets ────────────────────────────────────────────────────────────────

export const datasetsApi = {
  upload: (file: File) => {
    const form = new FormData()
    form.append('file', file)
    return api.post<UploadedDataset>('/datasets/upload', form, {
      headers: { 'Content-Type': 'multipart/form-data' },
    })
  },

  get: (id: string) =>
    api.get<UploadedDataset>(`/datasets/${id}`),
}

// ─── Auth ─────────────────────────────────────────────────────────────────────

export const authApi = {
  /** Fetch the current user's profile (requires valid JWT in localStorage). */
  me: () => api.get<AuthUser>('/auth/me'),

  /** Notify the server of logout (token removal happens client-side). */
  logout: () => api.post('/auth/logout'),
}

// ─── Leaderboard ─────────────────────────────────────────────────────────────

export const leaderboardApi = {
  get: (filters?: LeaderboardFilters) =>
    api.get<LeaderboardData>('/leaderboard', { params: filters }),
}

// ─── External catalog ────────────────────────────────────────────────────────
// ~94 third-party physics-ML libraries, ported from the standalone
// physics-based-references app. Mounted at /api/external-catalog by the
// backend (see backend/external_catalog/api.py). Same "real vs honest"
// status contract as the arena's own experiments — see ExternalCatalogStatus.

export interface ExternalCatalogFilters {
  q?: string
  domain?: string
  category?: string
  status?: ExternalCatalogInstallState
}

export const externalCatalogApi = {
  health: () =>
    api.get<{ status: string; models_indexed: number; datasets_indexed: number }>('/external-catalog/health'),

  listDomains: () =>
    api.get<Record<string, ExternalDomainMeta>>('/external-catalog/catalog/domains'),

  listCategories: () =>
    api.get<Record<string, ExternalCategoryMeta>>('/external-catalog/catalog/categories'),

  list: (filters?: ExternalCatalogFilters) =>
    api.get<{ count: number; total: number; models: ExternalCatalogModel[] }>(
      '/external-catalog', { params: filters },
    ).then((r) => ({ ...r, data: r.data.models })),

  get: (slug: string) =>
    api.get<ExternalCatalogModel>(`/external-catalog/models/${slug}`),

  sampleInput: (slug: string) =>
    api.get(`/external-catalog/models/${slug}/sample-input`),

  apiSnippet: (slug: string) =>
    api.get<ExternalApiSnippet>(`/external-catalog/models/${slug}/api-snippet`),

  infer: (slug: string, includeHeavy = false) =>
    api.post<ExternalJobResult<ExternalModelResult>>(
      `/external-catalog/models/${slug}/infer`, { include_heavy: includeHeavy },
    ),

  finetune: (slug: string, params: Record<string, unknown> = {}) =>
    api.post<ExternalJobResult<ExternalModelResult>>(
      `/external-catalog/models/${slug}/finetune`, { params },
    ),

  compare: (slugs: string[], includeHeavy = false) =>
    api.post<ExternalJobResult<ExternalCompareResult>>(
      '/external-catalog/compare', { slugs, include_heavy: includeHeavy },
    ),

  benchmark: (opts: { slugs?: string[]; category?: string; domain?: string; includeHeavy?: boolean }) =>
    api.post<ExternalJobResult<ExternalBenchmarkResult>>('/external-catalog/benchmark', {
      slugs: opts.slugs,
      category: opts.category,
      domain: opts.domain,
      include_heavy: opts.includeHeavy ?? false,
    }),

  getJob: <T,>(jobId: string) =>
    api.get<ExternalJobResult<T>>(`/external-catalog/jobs/${jobId}`),

  listDatasets: () =>
    api.get<{ count: number; datasets: ExternalDataset[] }>('/external-catalog/datasets')
      .then((r) => ({ ...r, data: r.data.datasets })),

  probeDataset: (slug: string, body: { subset: string; split: string; stream: boolean }) =>
    api.post<ExternalJobResult<ExternalModelResult>>(`/external-catalog/datasets/${slug}/probe`, body),
}

// Upper bound on client-side polling. Generous on purpose: a full benchmark
// slice (up to EXTERNAL_CATALOG_MAX_BENCHMARK_MODELS) can legitimately run long.
export const JOB_POLL_TIMEOUT_MS = 30 * 60_000

/** Follow a job past the backend's synchronous wait window until it settles. */
export async function waitForJob<T>(
  job: ExternalJobResult<T>,
  intervalMs = 2000,
  timeoutMs = JOB_POLL_TIMEOUT_MS,
): Promise<ExternalJobResult<T>> {
  const deadline = Date.now() + timeoutMs
  let current = job
  while (current.status === 'running') {
    if (Date.now() >= deadline) {
      throw new Error(
        `Job ${current.job_id} is still running after ${Math.round(timeoutMs / 60_000)} min — ` +
        `stopped waiting; check GET /api/external-catalog/jobs/${current.job_id} later.`,
      )
    }
    await new Promise((resolve) => setTimeout(resolve, intervalMs))
    current = (await externalCatalogApi.getJob<T>(current.job_id)).data
  }
  return current
}

/**
 * Submit an external-catalog job and resolve with its settled result.
 * A job that ends as "failed" (the job itself raised, as opposed to a model
 * result with status "failed") rejects, so it surfaces as the mutation error.
 */
export async function runJob<T>(request: Promise<{ data: ExternalJobResult<T> }>): Promise<ExternalJobResult<T>> {
  const job = await waitForJob((await request).data)
  if (job.status === 'failed') throw new Error(job.error ?? 'Job failed')
  return job
}

/** Human-readable message for a failed external-catalog request or job. */
export function jobErrorMessage(err: unknown, fallback = 'Request failed'): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response?.data?.detail
  if (typeof detail === 'string') return detail
  if (err instanceof Error && err.message) return err.message
  return fallback
}
