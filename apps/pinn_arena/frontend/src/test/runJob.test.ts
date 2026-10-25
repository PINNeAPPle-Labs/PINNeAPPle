import { describe, it, expect, vi, afterEach } from 'vitest'
import { externalCatalogApi, jobErrorMessage, runJob, waitForJob } from '../lib/api'
import type { ExternalJobResult } from '../types'

function job(status: ExternalJobResult<string>['status'], extra: Partial<ExternalJobResult<string>> = {}): ExternalJobResult<string> {
  return { job_id: 'j1', kind: 'infer', status, result: null, error: null, created_at: 0, finished_at: null, ...extra }
}

afterEach(() => { vi.restoreAllMocks(); vi.useRealTimers() })

describe('runJob — jobs that outlive the backend sync window', () => {
  it('returns a job that finished within the sync window without polling', async () => {
    const getJob = vi.spyOn(externalCatalogApi, 'getJob')
    const done = await runJob(Promise.resolve({ data: job('done', { result: 'ok' }) }))
    expect(done.result).toBe('ok')
    expect(getJob).not.toHaveBeenCalled()
  })

  it('polls a still-running job until it settles', async () => {
    vi.useFakeTimers()
    const getJob = vi.spyOn(externalCatalogApi, 'getJob')
      .mockResolvedValueOnce({ data: job('running') } as never)
      .mockResolvedValueOnce({ data: job('done', { result: 'late' }) } as never)
    const pending = runJob(Promise.resolve({ data: job('running') }))
    await vi.runAllTimersAsync()
    const done = await pending
    expect(done.result).toBe('late')
    expect(getJob).toHaveBeenCalledTimes(2)
  })

  it('gives up after the timeout instead of polling forever', async () => {
    vi.useFakeTimers()
    vi.spyOn(externalCatalogApi, 'getJob').mockResolvedValue({ data: job('running') } as never)
    const pending = waitForJob(job('running'), 1000, 5000)
    const assertion = expect(pending).rejects.toThrow(/still running after/)
    await vi.runAllTimersAsync()
    await assertion
  })

  it('rejects when the job itself failed, carrying its error', async () => {
    await expect(runJob(Promise.resolve({ data: job('failed', { error: 'RuntimeError: boom' }) })))
      .rejects.toThrow('RuntimeError: boom')
  })
})

describe('jobErrorMessage', () => {
  it('prefers the backend detail over the generic message', () => {
    expect(jobErrorMessage({ response: { data: { detail: 'Unknown model slug: x' } } })).toBe('Unknown model slug: x')
    expect(jobErrorMessage(new Error('RuntimeError: boom'))).toBe('RuntimeError: boom')
    expect(jobErrorMessage(null, 'fallback')).toBe('fallback')
  })
})
