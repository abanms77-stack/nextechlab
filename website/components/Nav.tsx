"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";

const links = [
  { href: "/product", label: "Product" },
  { href: "/research", label: "Research" },
  { href: "/developers", label: "Developers" },
  { href: "/pricing", label: "Pricing" },
];

export function Logo() {
  return (
    <Link href="/" className="group inline-flex items-center gap-2.5">
      <span className="relative grid h-7 w-7 place-items-center rounded-md bg-ink-900">
        <span className="absolute h-3.5 w-3.5 rounded-[3px] border border-white/70" />
        <span className="absolute h-1.5 w-1.5 rounded-full bg-accent-400" />
      </span>
      <span className="text-[15px] font-semibold tracking-tight text-ink-900">
        ContextLab
      </span>
    </Link>
  );
}

export function Nav() {
  const pathname = usePathname();
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-ink-200 bg-white/85 backdrop-blur">
      <div className="container-tight flex h-16 items-center justify-between">
        <Logo />

        <nav className="hidden items-center gap-1 md:flex">
          {links.map((l) => {
            const active = pathname === l.href;
            return (
              <Link
                key={l.href}
                href={l.href}
                className={
                  "rounded-md px-3 py-2 text-sm transition " +
                  (active
                    ? "text-ink-900"
                    : "text-ink-600 hover:text-ink-900")
                }
              >
                {l.label}
              </Link>
            );
          })}
        </nav>

        <div className="hidden items-center gap-2 md:flex">
          <Link href="/dashboard" className="btn-ghost">
            Sign in
          </Link>
          <Link href="/playground" className="btn-primary">
            Try Playground
          </Link>
        </div>

        <button
          aria-label="Menu"
          className="grid h-9 w-9 place-items-center rounded-md border border-ink-200 md:hidden"
          onClick={() => setOpen((v) => !v)}
        >
          <span className="space-y-1">
            <span className="block h-0.5 w-4 bg-ink-800" />
            <span className="block h-0.5 w-4 bg-ink-800" />
          </span>
        </button>
      </div>

      {open && (
        <div className="border-t border-ink-200 bg-white md:hidden">
          <div className="container-tight flex flex-col py-3">
            {[...links, { href: "/playground", label: "Playground" }, { href: "/dashboard", label: "Dashboard" }].map(
              (l) => (
                <Link
                  key={l.href}
                  href={l.href}
                  onClick={() => setOpen(false)}
                  className="rounded-md px-3 py-2.5 text-sm text-ink-700 hover:bg-ink-50"
                >
                  {l.label}
                </Link>
              ),
            )}
          </div>
        </div>
      )}
    </header>
  );
}