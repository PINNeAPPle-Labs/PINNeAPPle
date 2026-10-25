/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // All custom tokens use CSS variables so switching .dark on <html>
        // automatically re-themes the whole app without touching components.
        background:         'rgb(var(--color-background) / <alpha-value>)',
        surface:            'rgb(var(--color-surface) / <alpha-value>)',
        'surface-elevated': 'rgb(var(--color-surface-elevated) / <alpha-value>)',
        border:             'rgb(var(--color-border) / <alpha-value>)',
        'border-strong':    'rgb(var(--color-border-strong) / <alpha-value>)',
        accent:             'rgb(var(--color-accent) / <alpha-value>)',
        'accent-hover':     'rgb(var(--color-accent-hover) / <alpha-value>)',
        success:            'rgb(var(--color-success) / <alpha-value>)',
        warning:            'rgb(var(--color-warning) / <alpha-value>)',
        danger:             'rgb(var(--color-danger) / <alpha-value>)',
        muted:              'rgb(var(--color-muted) / <alpha-value>)',
        'text-primary':     'rgb(var(--color-text-primary) / <alpha-value>)',
        'text-secondary':   'rgb(var(--color-text-secondary) / <alpha-value>)',
        'text-muted':       'rgb(var(--color-text-muted) / <alpha-value>)',
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'spin-slow': 'spin 3s linear infinite',
      },
      boxShadow: {
        'glow-blue': '0 0 15px rgba(37, 99, 235, 0.15)',
        'glow-green': '0 0 15px rgba(5, 150, 105, 0.15)',
        'card': '0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04)',
        'card-hover': '0 4px 12px rgba(0,0,0,0.08)',
      }
    },
  },
  plugins: [],
}
