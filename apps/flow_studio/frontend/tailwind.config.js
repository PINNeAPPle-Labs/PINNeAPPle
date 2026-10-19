/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: { extend: { colors: { brand: { 50: "#fff8e6", 100: "#ffefc2", 500: "#f2a900", 600: "#d48f00", 700: "#a86f00" } } } },
  plugins: [],
};
