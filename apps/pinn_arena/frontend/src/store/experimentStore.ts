import { create } from 'zustand'
import type {
  PhysicsProblem, GeometryPreset, ArchitectureInfo, ExperimentConfig,
  ProblemType, InverseConfig, ColumnMapping, UploadedDataset,
} from '../types'

export type WizardStep =
  | 'problem_type'
  | 'problem'
  | 'parameters'
  | 'geometry'
  | 'dataset'
  | 'architectures'
  | 'training_config'
  | 'launch'

export const WIZARD_STEPS: WizardStep[] = [
  'problem_type',
  'problem',
  'parameters',
  'geometry',
  'dataset',
  'architectures',
  'training_config',
  'launch',
]

export const STEP_LABELS: Record<WizardStep, string> = {
  problem_type: 'Problem Type',
  problem: 'Problem',
  parameters: 'Parameters',
  geometry: 'Geometry',
  dataset: 'Dataset',
  architectures: 'Architectures',
  training_config: 'Config',
  launch: 'Launch',
}

export interface TrainingConfig {
  n_train_col: number
  n_train_bc: number
  n_train_ic: number
  epochs: number
  learning_rate: number
  batch_size: number
  loss_weights: {
    pde: number
    bc: number
    ic: number
    data: number
  }
}

const DEFAULT_TRAINING_CONFIG: TrainingConfig = {
  n_train_col: 10000,
  n_train_bc: 1000,
  n_train_ic: 500,
  epochs: 5000,
  learning_rate: 0.001,
  batch_size: 1024,
  loss_weights: {
    pde: 1.0,
    bc: 10.0,
    ic: 5.0,
    data: 1.0,
  },
}

const DEFAULT_INVERSE_CONFIG: InverseConfig = {
  params: [],
  initial_guesses: {},
  n_obs: 200,
  noise_std: 0.01,
  method: 'adam',
  n_iters: 1000,
  lambda_reg: 1e-4,
  noise_model: 'gaussian',
}

export interface ExperimentState {
  // Wizard state
  currentStep: WizardStep

  // Problem type (forward / inverse)
  problemType: ProblemType

  // Selected data
  selectedProblem: PhysicsProblem | null
  customParameters: Record<string, number>
  selectedGeometry: GeometryPreset | null
  selectedArchitectures: ArchitectureInfo[]
  trainingConfig: TrainingConfig

  // Dataset upload (optional; replaces generated data when provided)
  uploadedDataset: UploadedDataset | null
  columnMapping: ColumnMapping

  // Inverse problem config
  inverseConfig: InverseConfig

  // Actions
  setCurrentStep: (step: WizardStep) => void
  nextStep: () => void
  prevStep: () => void
  setProblemType: (type: ProblemType) => void
  setSelectedProblem: (problem: PhysicsProblem | null) => void
  setCustomParameter: (name: string, value: number) => void
  setCustomParameters: (params: Record<string, number>) => void
  setSelectedGeometry: (geometry: GeometryPreset | null) => void
  toggleArchitecture: (architecture: ArchitectureInfo) => void
  selectAllArchitectures: (architectures: ArchitectureInfo[]) => void
  clearArchitectures: () => void
  setTrainingConfig: (config: Partial<TrainingConfig>) => void
  setUploadedDataset: (dataset: UploadedDataset | null) => void
  setColumnMapping: (mapping: Partial<ColumnMapping>) => void
  setInverseConfig: (config: Partial<InverseConfig>) => void
  buildExperimentConfig: () => ExperimentConfig | null
  reset: () => void
}

function getVisibleSteps(problem: PhysicsProblem | null): WizardStep[] {
  const steps = WIZARD_STEPS.filter((s) => {
    if (s === 'geometry' && problem && !problem.requires_geometry) return false
    return true
  })
  return steps
}

export const useExperimentStore = create<ExperimentState>((set, get) => ({
  currentStep: 'problem_type',
  problemType: 'forward',
  selectedProblem: null,
  customParameters: {},
  selectedGeometry: null,
  selectedArchitectures: [],
  trainingConfig: DEFAULT_TRAINING_CONFIG,
  uploadedDataset: null,
  columnMapping: { coord_cols: [], output_cols: [], time_col: null },
  inverseConfig: DEFAULT_INVERSE_CONFIG,

  setCurrentStep: (step) => set({ currentStep: step }),

  nextStep: () => {
    const { currentStep, selectedProblem } = get()
    const steps = getVisibleSteps(selectedProblem)
    const idx = steps.indexOf(currentStep)
    if (idx < steps.length - 1) {
      set({ currentStep: steps[idx + 1] })
    }
  },

  prevStep: () => {
    const { currentStep, selectedProblem } = get()
    const steps = getVisibleSteps(selectedProblem)
    const idx = steps.indexOf(currentStep)
    if (idx > 0) {
      set({ currentStep: steps[idx - 1] })
    }
  },

  setProblemType: (type) => set({ problemType: type }),

  setSelectedProblem: (problem) => {
    const defaultParams: Record<string, number> = {}
    if (problem) {
      problem.parameters.forEach((p) => {
        defaultParams[p.name] = p.default
      })
    }
    set({
      selectedProblem: problem,
      customParameters: defaultParams,
      selectedGeometry: null,
    })
  },

  setCustomParameter: (name, value) =>
    set((state) => ({
      customParameters: { ...state.customParameters, [name]: value },
    })),

  setCustomParameters: (params) => set({ customParameters: params }),

  setSelectedGeometry: (geometry) => set({ selectedGeometry: geometry }),

  toggleArchitecture: (architecture) =>
    set((state) => {
      const exists = state.selectedArchitectures.some((a) => a.id === architecture.id)
      return {
        selectedArchitectures: exists
          ? state.selectedArchitectures.filter((a) => a.id !== architecture.id)
          : [...state.selectedArchitectures, architecture],
      }
    }),

  selectAllArchitectures: (architectures) =>
    set((state) => {
      const existingIds = new Set(state.selectedArchitectures.map((a) => a.id))
      const newOnes = architectures.filter((a) => !existingIds.has(a.id))
      return { selectedArchitectures: [...state.selectedArchitectures, ...newOnes] }
    }),

  clearArchitectures: () => set({ selectedArchitectures: [] }),

  setTrainingConfig: (config) =>
    set((state) => ({
      trainingConfig: { ...state.trainingConfig, ...config },
    })),

  setUploadedDataset: (dataset) => set({ uploadedDataset: dataset }),

  setColumnMapping: (mapping) =>
    set((state) => ({
      columnMapping: { ...state.columnMapping, ...mapping },
    })),

  setInverseConfig: (config) =>
    set((state) => ({
      inverseConfig: { ...state.inverseConfig, ...config },
    })),

  buildExperimentConfig: (): ExperimentConfig | null => {
    const {
      selectedProblem, selectedGeometry, selectedArchitectures,
      trainingConfig, customParameters, problemType,
      uploadedDataset, columnMapping, inverseConfig,
    } = get()
    if (!selectedProblem || selectedArchitectures.length === 0) return null

    const config: ExperimentConfig = {
      problem_id: selectedProblem.id,
      problem_type: problemType,
      geometry_id: selectedGeometry?.id,
      architecture_ids: selectedArchitectures.map((a) => a.id),
      n_train_col: trainingConfig.n_train_col,
      n_train_bc: trainingConfig.n_train_bc,
      n_train_ic: trainingConfig.n_train_ic,
      epochs: trainingConfig.epochs,
      learning_rate: trainingConfig.learning_rate,
      batch_size: trainingConfig.batch_size,
      loss_weights: {
        pde: trainingConfig.loss_weights.pde,
        bc: trainingConfig.loss_weights.bc,
        ic: trainingConfig.loss_weights.ic,
        data: trainingConfig.loss_weights.data,
      },
      custom_parameters: customParameters,
    }

    if (uploadedDataset) {
      config.dataset_id = uploadedDataset.dataset_id
      config.column_mapping = columnMapping
    }

    if (problemType === 'inverse') {
      config.inverse_config = inverseConfig
    }

    return config
  },

  reset: () =>
    set({
      currentStep: 'problem_type',
      problemType: 'forward',
      selectedProblem: null,
      customParameters: {},
      selectedGeometry: null,
      selectedArchitectures: [],
      trainingConfig: DEFAULT_TRAINING_CONFIG,
      uploadedDataset: null,
      columnMapping: { coord_cols: [], output_cols: [], time_col: null },
      inverseConfig: DEFAULT_INVERSE_CONFIG,
    }),
}))
