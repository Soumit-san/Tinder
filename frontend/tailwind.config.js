/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        tinder: {
          pink: '#FE3C72',
          orange: '#FF7854',
          dark: '#111418',
          card: '#1E2228'
        },
        sentiment: {
          positive: '#22C55E',
          neutral: '#94A3B8',
          negative: '#EF4444'
        }
      }
    },
  },
  plugins: [],
}
