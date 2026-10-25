import React from 'react'
import { CheckCircle2, CircleDashed, MinusCircle, Clock3, XCircle, HelpCircle } from 'lucide-react'
import type { BadgeVariant } from '../../components/ui/Badge'
import type { ExternalCatalogInstallState, ExternalCatalogStatus } from '../../types'

/**
 * The "real vs honest" status contract preserved from physics-based-references —
 * see backend/external_catalog/core/result.py and the arena README. Every result
 * from /infer, /finetune, /benchmark, /compare carries one of these; none of
 * them are ever silently upgraded to "ok" by the UI.
 */
export const RESULT_STATUS_CONFIG: Record<
  ExternalCatalogStatus,
  { label: string; variant: BadgeVariant; icon: React.ReactNode; hint: string }
> = {
  ok: {
    label: 'OK — real result',
    variant: 'success',
    icon: <CheckCircle2 size={12} />,
    hint: 'The real library was imported and a real forward pass / simulation ran.',
  },
  not_installed: {
    label: 'Not installed',
    variant: 'default',
    icon: <CircleDashed size={12} />,
    hint: 'Pip-installable, just not installed in this environment.',
  },
  skipped_heavy: {
    label: 'Skipped (heavy)',
    variant: 'warning',
    icon: <Clock3 size={12} />,
    hint: 'GPU/heavy dependency — opt in with include_heavy.',
  },
  manual_install: {
    label: 'Manual install',
    variant: 'info',
    icon: <MinusCircle size={12} />,
    hint: 'No pip package exists — git-only or non-Python.',
  },
  failed: {
    label: 'Failed',
    variant: 'danger',
    icon: <XCircle size={12} />,
    hint: 'It ran and raised — see the exception below.',
  },
  finetune_not_supported: {
    label: 'Fine-tune not supported',
    variant: 'default',
    icon: <HelpCircle size={12} />,
    hint: 'Fine-tuning isn’t wired up in-app for this model yet.',
  },
}

/** Licenses that must not reach commercial PINNeAPPle products (rendered in red). */
export function isRestrictiveLicense(license: string | null | undefined): boolean {
  return !!license && /non-commercial|NC-SA|NC\b|military/i.test(license)
}

export const INSTALL_STATE_CONFIG: Record<
  ExternalCatalogInstallState,
  { label: string; variant: BadgeVariant }
> = {
  installed: { label: 'Installed', variant: 'success' },
  not_installed: { label: 'Not installed', variant: 'default' },
  manual_only: { label: 'Manual install', variant: 'info' },
  heavy: { label: 'Heavy / GPU', variant: 'warning' },
}
