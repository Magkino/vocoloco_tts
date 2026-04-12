/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ['./*.{html,js}', './workers/*.js'],
  theme: {
    extend: {
      colors: {
        'omni-bg': '#0f131a',
        'omni-panel': '#1a2028',
        'omni-border': '#2d3748',
        'omni-neon': '#4ade80',
        'omni-neon-muted': 'rgba(74, 222, 128, 0.2)',
        'omni-text-muted': '#94a3b8',
        'omni-active-bg': '#273c38',
        'omni-btn-bg': '#1e293b',
      },
      boxShadow: {
        'neon': '0 0 10px rgba(74, 222, 128, 0.3)',
        'neon-strong': '0 0 20px rgba(74, 222, 128, 0.5)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      },
      backgroundImage: {
        'circuit-pattern': "url('data:image/svg+xml,%3Csvg width=\"60\" height=\"60\" viewBox=\"0 0 60 60\" xmlns=\"http://www.w3.org/2000/svg\"%3E%3Cg fill=\"none\" fill-rule=\"evenodd\"%3E%3Cg fill=\"%232d3748\" fill-opacity=\"0.1\"%3E%3Cpath d=\"M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z\"/%3E%3C/g%3E%3C/g%3E%3C/svg%3E')",
      }
    }
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries'),
  ],
}
