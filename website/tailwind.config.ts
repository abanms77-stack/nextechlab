import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
    "./lib/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        ink: {
          50: "#f6f7f9",
          100: "#eceef2",
          200: "#d6dae2",
          300: "#b2b9c7",
          400: "#8791a6",
          500: "#66708a",
          600: "#515a72",
          700: "#42495c",
          800: "#3a3f4e",
          900: "#0e1116",
        },
        accent: {
          400: "#4f8cff",
          500: "#2f6bf0",
          600: "#1f55d6",
        },
        risk: {
          low: "#1f9d6b",
          med: "#d18b26",
          high: "#d64545",
        },
      },
      fontFamily: {
        mono: ["var(--font-mono)", "ui-monospace", "SFMono-Regular", "monospace"],
      },
      boxShadow: {
        panel: "0 1px 2px rgba(14,17,22,0.06), 0 8px 24px rgba(14,17,22,0.06)",
      },
      backgroundImage: {
        grid: "linear-gradient(to right, rgba(14,17,22,0.045) 1px, transparent 1px), linear-gradient(to bottom, rgba(14,17,22,0.045) 1px, transparent 1px)",
      },
      backgroundSize: {
        grid: "32px 32px",
      },
    },
  },
  plugins: [],
};

export default config;