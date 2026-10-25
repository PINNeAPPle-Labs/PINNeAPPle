import { useQuery } from '@tanstack/react-query'
import { experimentsApi } from '../lib/api'
import type { ExperimentStatusResponse, ExperimentResult } from '../types'

export function useExperimentStatus(experimentId: string | undefined, enabled = true) {
  return useQuery<ExperimentStatusResponse>({
    queryKey: ['experiment-status', experimentId],
    queryFn: async () => {
      if (!experimentId) throw new Error('No experiment ID')
      const res = await experimentsApi.getStatus(experimentId)
      return res.data
    },
    enabled: !!experimentId && enabled,
    refetchInterval: (query) => {
      const data = query.state.data
      if (!data) return 2000
      if (data.status === 'COMPLETED' || data.status === 'FAILED') return false
      return 2000
    },
    staleTime: 0,
  })
}

export function useExperimentResults(experimentId: string | undefined) {
  return useQuery<ExperimentResult>({
    queryKey: ['experiment-results', experimentId],
    queryFn: async () => {
      if (!experimentId) throw new Error('No experiment ID')
      const res = await experimentsApi.getResults(experimentId)
      return res.data
    },
    enabled: !!experimentId,
    staleTime: 60_000,
  })
}

export function useExperimentList() {
  return useQuery<ExperimentResult[]>({
    queryKey: ['experiments'],
    queryFn: async () => {
      const res = await experimentsApi.list()
      return res.data
    },
  })
}
