import React from 'react'
import clsx from 'clsx'
import { Check } from 'lucide-react'

interface Step {
  id: string
  label: string
}

interface StepIndicatorProps {
  steps: Step[]
  currentStep: string
  completedSteps: string[]
  onStepClick?: (stepId: string) => void
}

export function StepIndicator({ steps, currentStep, completedSteps, onStepClick }: StepIndicatorProps) {
  return (
    <div className="flex items-center w-full overflow-x-auto pb-2">
      {steps.map((step, index) => {
        const isActive = step.id === currentStep
        const isCompleted = completedSteps.includes(step.id)
        const isClickable = isCompleted && onStepClick

        return (
          <React.Fragment key={step.id}>
            <div className="flex flex-col items-center flex-shrink-0">
              <button
                onClick={() => isClickable && onStepClick(step.id)}
                disabled={!isClickable}
                className={clsx(
                  'w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold transition-all duration-200',
                  isActive && 'bg-accent text-white ring-4 ring-accent/20',
                  isCompleted && !isActive && 'bg-accent text-white cursor-pointer hover:bg-accent-hover',
                  !isActive && !isCompleted && 'bg-slate-200 text-slate-400 border border-slate-300'
                )}
              >
                {isCompleted && !isActive ? (
                  <Check size={14} />
                ) : (
                  <span>{index + 1}</span>
                )}
              </button>
              <span
                className={clsx(
                  'text-xs mt-1.5 font-medium whitespace-nowrap',
                  isActive && 'text-accent',
                  isCompleted && !isActive && 'text-accent',
                  !isActive && !isCompleted && 'text-text-muted'
                )}
              >
                {step.label}
              </span>
            </div>

            {index < steps.length - 1 && (
              <div
                className={clsx(
                  'flex-1 h-0.5 mx-2 transition-all duration-300 min-w-4',
                  isCompleted ? 'bg-accent' : 'bg-slate-200'
                )}
              />
            )}
          </React.Fragment>
        )
      })}
    </div>
  )
}
