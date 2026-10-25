export type PhysicsArea =
  | 'FLUID_MECHANICS'
  | 'SOLID_MECHANICS'
  | 'ELECTROMAGNETISM'
  | 'QUANTUM_MECHANICS'
  | 'THERMODYNAMICS'
  | 'WAVE_PHYSICS'
  | 'NONLINEAR_DYNAMICS'

export type Dimension = 'ONE_D' | 'TWO_D' | 'THREE_D'
export type Difficulty = 'BEGINNER' | 'INTERMEDIATE' | 'ADVANCED'
export type ArchitectureFamily =
  | 'PINN'
  | 'NEURAL_OPERATOR'
  | 'TRANSFORMER'
  | 'RECURRENT'
  | 'CONVOLUTIONAL'
  | 'CONTINUOUS'
  | 'GRAPH'
  | 'AUTOENCODER'
  | 'RESERVOIR'
  | 'ROM'
  | 'CLASSICAL'
  | 'PHYSICS_AWARE'
export type ExperimentStatus = 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED'

export interface ProblemParameter {
  name: string
  default: number
  min_val: number
  max_val: number
  description: string
  unit: string
}

export interface ColumnHints {
  coord_cols: string[]
  output_cols: string[]
  time_col: string | null
  csv_header: string
  notes: string
}

export interface PhysicsProblem {
  id: string
  name: string
  description: string
  domain: PhysicsArea
  dimension: Dimension
  equations: string[]
  applications: string[]
  initial_conditions: string
  boundary_conditions: string
  parameters: ProblemParameter[]
  difficulty: Difficulty
  requires_geometry: boolean
  tags: string[]
  latex_equations?: string[]
  column_hints?: ColumnHints
}

export interface GeometryPreset {
  id: string
  name: string
  description: string
  geometry_type: string
  dimensions: Record<string, number>
  supported_problems: string[]
  mesh_resolution_default: number
}

export interface ArchitectureInfo {
  id: string
  name: string
  family: ArchitectureFamily
  description: string
  detailed_description?: string
  supports_physics_loss: boolean
  supports_3d: boolean
  inverse_only: boolean
  best_for: string[]
  parameters: Record<string, unknown>
  paper_reference?: string
}

export interface DiagramNode {
  id: string
  label: string
  sublabel: string
  type: 'input' | 'linear' | 'activation' | 'norm' | 'attention' | 'conv' | 'recurrent' | 'output' | 'block'
}

export interface DiagramEdge {
  from: string
  to: string
}

export interface ArchitectureLayerSchema {
  architecture_id: string
  family: string
  nodes: DiagramNode[]
  edges: DiagramEdge[]
}

export interface ArchitectureFeedbackSummary {
  arch_id: string
  thumbs_up: number
  thumbs_down: number
  total: number
  recent_comments: Array<{ is_positive: boolean; comment: string; created_at: string }>
}

export interface TrainingMetrics {
  epoch: number
  total_loss: number
  pde_loss?: number
  bc_loss?: number
  val_loss?: number
}

export interface InverseResult {
  identified_params?: Record<string, number>
  true_params?: Record<string, number | null>
  solver_method?: string
  n_iters?: number
  n_obs?: number
  noise_model?: string
  converged?: boolean
  final_residual?: number | null
  error?: string
}

export type PredictionQualityLevel =
  | 'excellent'
  | 'good'
  | 'fair'
  | 'poor'
  | 'unreliable'
  | 'unknown'

export type PredictionQualityFlag =
  | 'nan_or_inf'
  | 'constant_output'
  | 'near_constant_output'
  | 'zero_output'
  | 'below_mean_baseline'
  | 'high_relative_error'

export interface ModelResult {
  /** "failed" when the architecture couldn't train or evaluate — see `error`. */
  status?: 'completed' | 'failed'
  error?: string | null
  architecture_id: string
  architecture_name: string
  architecture_family?: string
  training_time_s: number
  final_loss: number
  metrics: {
    l2_relative_error: number
    l1_error: number
    rmse: number
    max_error: number
    r2_score: number
  }
  converged: boolean
  training_history: TrainingMetrics[]
  physics_loss_enabled?: boolean
  inverse_result?: InverseResult | null
  inference_time_ms?: number | null
  peak_gpu_memory_mb?: number | null
  peak_cpu_memory_mb?: number | null
  quality_score?: number | null
  quality_level?: PredictionQualityLevel
  quality_flags?: PredictionQualityFlag[]
  pred_std_ratio?: number | null
}

export interface ExperimentResult {
  experiment_id: string
  problem_id: string
  problem_type?: ProblemType
  geometry_id?: string
  status: ExperimentStatus
  created_at: string
  completed_at?: string
  model_results: ModelResult[]
  ranking: string[]
  best_model_id: string
  comparison_summary: Record<string, unknown>
}

// ─── Leaderboard ─────────────────────────────────────────────────────────────

export interface MultiObjectiveScores {
  accuracy: number | null
  training_cost: number | null
  convergence: number
  memory: number | null
  inference_speed: number | null
}

export interface LeaderboardEntry {
  rank: number
  architecture_id: string
  architecture_name: string
  family: string
  n_experiments: number
  n_wins: number
  win_rate: number
  avg_l2_error: number | null
  avg_mape: number | null
  avg_rmse: number | null
  avg_r2_score: number | null
  avg_max_error: number | null
  avg_training_time_s: number | null
  convergence_rate: number
  problems_count: number
  avg_n_pde_evaluations: number | null
  avg_peak_memory_mb: number | null
  avg_inference_time_ms: number | null
  scores: MultiObjectiveScores | null
}

export interface LeaderboardData {
  total_experiments: number
  total_architectures: number
  problems_covered: number
  entries: LeaderboardEntry[]
}

export interface LeaderboardFilters {
  problem_id?: string
  domain?: string
  dimension?: string
  difficulty?: string
  requires_geometry?: boolean
}

export type ProblemType = 'forward' | 'inverse'

export type InverseNoiseModel = 'gaussian' | 'huber' | 'cauchy' | 'student_t'
export type InverseSolverMethod = 'adam' | 'lbfgs' | 'eki' | 'teki'

export interface InverseConfig {
  params: string[]
  initial_guesses: Record<string, number>
  n_obs: number
  noise_std: number
  method: InverseSolverMethod
  n_iters: number
  lambda_reg: number
  noise_model: InverseNoiseModel
}

export interface ColumnMapping {
  coord_cols: string[]
  output_cols: string[]
  time_col: string | null
}

export interface UploadedDataset {
  dataset_id: string
  original_filename: string
  n_rows: number
  n_cols: number
  columns: string[]
  col_stats: Array<{ name: string; min: number; max: number; mean: number; std: number }>
  preview: Record<string, number>[]
}

export interface ExperimentConfig {
  problem_id: string
  problem_type?: ProblemType
  geometry_id?: string
  architecture_ids: string[]
  n_train_col: number
  n_train_bc: number
  n_train_ic: number
  epochs: number
  learning_rate: number
  batch_size: number
  loss_weights: Record<string, number>
  custom_parameters: Record<string, number>
  dataset_id?: string
  column_mapping?: ColumnMapping
  inverse_config?: InverseConfig
}

export interface DatasetPreview {
  problem_id: string
  n_samples: number
  // structured grid fields (1D/2D time-series)
  x?: number[]
  y?: number[]
  t?: number[]
  u?: number[][] | number[]
  v?: number[][] | number[]
  p?: number[][] | number[]
  // scatter fields (3D)
  coords?: number[][]
  solution?: (number | number[])[]
  // fallback legacy fields
  domain_bounds?: Record<string, [number, number]>
  sample_x?: number[]
  sample_y?: number[]
  sample_z?: number[]
  sample_values?: number[]
  description?: string
  metadata?: Record<string, unknown>
  [key: string]: unknown
}

export interface ExperimentStatusResponse {
  experiment_id: string
  status: ExperimentStatus
  progress: number
  current_architecture?: string
  current_epoch?: number
  total_epochs?: number
  model_statuses: Record<string, {
    status: 'waiting' | 'training' | 'completed' | 'failed'
    progress: number
    current_loss?: number
    training_history?: TrainingMetrics[]
  }>
  logs: string[]
  started_at?: string
  estimated_completion?: string
}

// ─── External catalog (backend/external_catalog/) ───────────────────────────
// ~94 third-party physics-ML libraries (PINNs, neural operators, differentiable
// simulators, ML potentials) across 41 physical domains, benchmarked alongside
// PINNeAPPle's own architectures above. Every result carries an honest status —
// see ExternalCatalogStatus below — never a fabricated "it works".

export type ExternalCatalogStatus =
  | 'ok'
  | 'not_installed'
  | 'skipped_heavy'
  | 'manual_install'
  | 'failed'
  | 'finetune_not_supported'

export type ExternalCatalogInstallState = 'installed' | 'not_installed' | 'manual_only' | 'heavy'

export interface ExternalDomainMeta {
  key: string
  label: string
  glyph: string
  desc: string
}

export interface ExternalCategoryMeta {
  key: string
  label: string
}

export interface ExternalCatalogModel {
  slug: string
  name: string
  org: string
  arch: string
  domains: string[]
  desc: string
  tags: string[]
  repo: string
  ref: string
  pip_package: string | null
  import_name: string | null
  heavy: boolean
  manual_only: boolean
  manual_note: string | null
  dataset_fn: string | null
  finetunable: boolean
  license: string | null
  maturity: string
  install_state: ExternalCatalogInstallState
  local_command_infer: string
  local_command_finetune: string
  pip_install_command: string | null
}

export interface ExternalModelResult {
  name: string
  slug: string
  category: string
  domains: string[]
  repo: string
  status: ExternalCatalogStatus
  dataset_name: string
  dataset_description: string
  input_shape: string | null
  output_shape: string | null
  output_summary: Record<string, unknown> | null
  metrics: Record<string, unknown>
  error: string | null
  note: string | null
  local_command: string | null
  timestamp: string
}

export interface ExternalJobResult<T> {
  job_id: string
  kind: 'infer' | 'finetune' | 'benchmark' | 'compare' | 'probe'
  status: 'running' | 'done' | 'failed'
  result: T | null
  error: string | null
  created_at: number
  finished_at: number | null
}

export interface ExternalBenchmarkSummary {
  total: number
  by_status: Record<string, number>
  real_ok_count: number
  avg_inference_seconds: number | null
  max_inference_seconds: number | null
}

export interface ExternalBenchmarkResult {
  summary: ExternalBenchmarkSummary
  results: ExternalModelResult[]
}

export interface ExternalCompareResult extends ExternalBenchmarkResult {
  specs: ExternalCatalogModel[]
}

// Dataset catalog — a parallel catalog to the model one (see DatasetSpec in
// backend/external_catalog/registry/schema.py). Probes return an
// ExternalModelResult with category "dataset" and the same status contract.
export interface ExternalSubDataset {
  name: string
  label: string
  domains: string[]
  dims: string
  desc: string
}

export interface ExternalDataset {
  slug: string
  name: string
  org: string
  desc: string
  tags: string[]
  repo: string
  ref: string
  size: string
  subsets: ExternalSubDataset[]
  splits: string[]
  domains: string[]
  pip_package: string | null
  import_name: string | null
  stream_base_path: string | null
  stream_requires: string[]
  download_command: string | null
  license: string | null
  homepage: string | null
  installed: boolean
  local_root_configured: boolean
  pip_install_command: string | null
}

export interface ExternalApiSnippet {
  slug: string
  base_url: string
  curl: string
  python: string
  openapi_docs: string
}

export type SortColumn =
  | 'rank'
  | 'architecture_name'
  | 'l2_relative_error'
  | 'rmse'
  | 'max_error'
  | 'r2_score'
  | 'training_time_s'
  | 'converged'

export interface SortConfig {
  column: SortColumn
  direction: 'asc' | 'desc'
}

export type ResultTab = 'loss_curves' | 'field_comparison' | 'error_distribution' | 'scatter_plot' | 'uncertainty'
