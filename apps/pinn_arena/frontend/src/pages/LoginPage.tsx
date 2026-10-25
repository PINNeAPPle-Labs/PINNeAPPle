import { Github, FlaskConical, BarChart3, Layers } from 'lucide-react'

// ─── LoginPage ────────────────────────────────────────────────────────────────

export function LoginPage() {
  const handleGitHubLogin = () => {
    window.location.href = '/api/auth/github/login'
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-6">
      <div className="w-full max-w-md">
        {/* Card */}
        <div className="bg-white border border-border rounded-2xl shadow-sm overflow-hidden">
          {/* Header band */}
          <div className="bg-accent/5 border-b border-border px-8 py-7 text-center">
            <div className="flex items-center justify-center gap-3 mb-3">
              <img
                src="/logo.svg"
                alt="PINNeAPPle Arena logo"
                className="w-12 h-12 flex-shrink-0"
              />
              <div className="text-left">
                <div className="text-lg font-bold text-accent leading-tight">PINNeAPPle</div>
                <div className="text-sm font-semibold text-slate-600 leading-tight">Arena</div>
              </div>
            </div>
            <p className="text-xs text-text-muted leading-relaxed max-w-xs mx-auto">
              Physics-Informed Neural Network Benchmarking Platform
            </p>
          </div>

          {/* Body */}
          <div className="px-8 py-7">
            <h1 className="text-base font-semibold text-text-primary mb-1 text-center">
              Sign in to your account
            </h1>
            <p className="text-xs text-text-muted text-center mb-6">
              Your experiments will be saved and synced across sessions.
            </p>

            {/* GitHub sign-in button */}
            <button
              onClick={handleGitHubLogin}
              className="w-full flex items-center justify-center gap-2.5 bg-slate-900 hover:bg-slate-800 text-white rounded-lg px-4 py-3 text-sm font-medium transition-colors duration-200"
            >
              <Github size={17} />
              Sign in with GitHub
            </button>

            {/* Divider */}
            <div className="flex items-center gap-3 my-5">
              <div className="flex-1 h-px bg-border" />
              <span className="text-[10px] text-text-muted uppercase tracking-wider">or</span>
              <div className="flex-1 h-px bg-border" />
            </div>

            {/* Guest access note */}
            <p className="text-[11px] text-text-muted text-center leading-relaxed">
              You can also{' '}
              <button
                onClick={() => window.location.href = '/'}
                className="text-accent hover:underline font-medium"
              >
                continue as guest
              </button>
              {' '}without signing in.
              <br />
              Guest experiments are stored in memory and lost on server restart.
            </p>
          </div>

          {/* Feature bullets */}
          <div className="border-t border-border px-8 py-5 bg-surface-elevated/40">
            <p className="text-[10px] font-semibold text-text-muted uppercase tracking-wider mb-3">
              With an account you get
            </p>
            <ul className="space-y-2">
              {[
                { icon: <FlaskConical size={12} />, text: 'Experiments saved to database' },
                { icon: <BarChart3 size={12} />, text: 'Compare results across sessions' },
                { icon: <Layers size={12} />, text: 'Full experiment history' },
              ].map((item) => (
                <li key={item.text} className="flex items-center gap-2 text-[11px] text-text-secondary">
                  <span className="text-accent flex-shrink-0">{item.icon}</span>
                  {item.text}
                </li>
              ))}
            </ul>
          </div>
        </div>

        <p className="text-center text-[10px] text-text-muted mt-4">
          Built on top of the{' '}
          <a
            href="https://github.com/PINNeAPPle"
            target="_blank"
            rel="noopener noreferrer"
            className="text-accent hover:underline"
          >
            PINNeAPPle
          </a>{' '}
          library
        </p>
      </div>
    </div>
  )
}
