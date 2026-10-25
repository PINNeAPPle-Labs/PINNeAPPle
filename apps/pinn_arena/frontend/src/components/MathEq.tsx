import { useMemo } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

interface MathEqProps {
  latex: string
  display?: boolean
  className?: string
}

export function MathEq({ latex, display = true, className }: MathEqProps) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(latex, {
        displayMode: display,
        throwOnError: false,
        strict: false,
      })
    } catch {
      return null
    }
  }, [latex, display])

  if (!html) {
    return <span className={className}>{latex}</span>
  }

  return (
    <span
      className={className}
      dangerouslySetInnerHTML={{ __html: html }}
    />
  )
}
