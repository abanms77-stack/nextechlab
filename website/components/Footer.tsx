import Link from "next/link";
import { Logo } from "./Nav";

const cols = [
  {
    title: "Product",
    links: [
      { href: "/product", label: "Context Profiler" },
      { href: "/product#optimizer", label: "Context Optimizer" },
      { href: "/product#before-after", label: "Before / After" },
      { href: "/playground", label: "Playground" },
    ],
  },
  {
    title: "Developers",
    links: [
      { href: "/developers", label: "API docs" },
      { href: "/developers#sdk", label: "SDK" },
      { href: "/developers#examples", label: "Examples" },
      { href: "/dashboard", label: "Dashboard" },
    ],
  },
  {
    title: "Research",
    links: [
      { href: "/research", label: "Overview" },
      { href: "/research#methodology", label: "Methodology" },
      { href: "/research#results", label: "Results" },
      { href: "/about", label: "About" },
    ],
  },
  {
    title: "Company",
    links: [
      { href: "/pricing", label: "Pricing" },
      { href: "/about", label: "About" },
      { href: "/trust", label: "Trust & security" },
      { href: "/status", label: "Status" },
    ],
  },
];

export function Footer() {
  return (
    <footer className="mt-24 border-t border-ink-200 bg-ink-50/40">
      <div className="container-tight py-14">
        <div className="grid gap-10 md:grid-cols-6">
          <div className="md:col-span-2">
            <Logo />
            <p className="mt-4 max-w-xs text-sm leading-relaxed text-ink-500">
              Infrastructure that makes LLM context reliable by measuring and
              reducing interference between competing passages.
            </p>
          </div>
          {cols.map((c) => (
            <div key={c.title}>
              <div className="eyebrow mb-3">{c.title}</div>
              <ul className="space-y-2">
                {c.links.map((l) => (
                  <li key={l.href}>
                    <Link
                      href={l.href}
                      className="text-sm text-ink-600 transition hover:text-ink-900"
                    >
                      {l.label}
                    </Link>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </div>
        <div className="mt-12 flex flex-col gap-3 border-t border-ink-200 pt-6 sm:flex-row sm:items-center sm:justify-between">
          <p className="mono text-[11px] text-ink-400">
            © {new Date().getFullYear()} ContextLab. All rights reserved.
          </p>
          <p className="mono text-[11px] text-ink-400">
            Research-backed · retrieval-interference theory
          </p>
        </div>
      </div>
    </footer>
  );
}