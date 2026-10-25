import React from 'react'
import clsx from 'clsx'

interface CardProps {
  children: React.ReactNode
  className?: string
  hover?: boolean
  selected?: boolean
  onClick?: () => void
}

export function Card({ children, className, hover, selected, onClick }: CardProps) {
  return (
    <div
      className={clsx(
        'bg-white border border-border rounded-xl shadow-card transition-all duration-200',
        hover && 'hover:shadow-card-hover hover:border-border-strong cursor-pointer',
        selected && 'border-accent shadow-glow-blue',
        onClick && 'cursor-pointer',
        className
      )}
      onClick={onClick}
    >
      {children}
    </div>
  )
}

interface CardHeaderProps {
  children: React.ReactNode
  className?: string
}

export function CardHeader({ children, className }: CardHeaderProps) {
  return (
    <div className={clsx('px-5 py-4 border-b border-border', className)}>
      {children}
    </div>
  )
}

interface CardContentProps {
  children: React.ReactNode
  className?: string
}

export function CardContent({ children, className }: CardContentProps) {
  return (
    <div className={clsx('px-5 py-4', className)}>
      {children}
    </div>
  )
}

interface CardFooterProps {
  children: React.ReactNode
  className?: string
}

export function CardFooter({ children, className }: CardFooterProps) {
  return (
    <div className={clsx('px-5 py-4 border-t border-border', className)}>
      {children}
    </div>
  )
}
