import type { Metadata } from "next";
import Link from "next/link";
import { Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "Pricing",
  description: "Simple plans for context profiling and optimization.",
};

const plans = [
  {
    name: "Free",
    price: "$0",
    per: "forever",
    items: ["100 analyses / month", "Playground", "Basic profiling", "Community support"],
    cta: "Start free",
    href: "/playground",
    featured: false,
  },
  {
    name: "Developer",
    price: "$49",
    per: "/ month",
    items: ["API access", "10,000 analyses / month", "Optimization", "Usage analytics", "Email support"],
    cta: "Start building",
    href: "/dashboard",
    featured: true,
  },
  {
    name: "Team",
    price: "$299",
    per: "/ month",
    items: ["Higher limits", "Team dashboard", "Evaluation suite", "Integrations", "Priority support"],
    cta: "Contact us",
    href: "/about",
    featured: false,
  },
  {
    name: "Enterprise",
    price: "Custom",
    per: "",
    items: ["Private deployment", "Security review", "SLA", "Custom models", "Large-scale inference"],
    cta: "Talk to us",
    href: "/about",
    featured: false,
  },
];

export default function PricingPage() {
  return (
    <>
      <section className="border-b border-ink-200">
        <div className="container-tight py-14 text-center">
          <div className="eyebrow mb-3">pricing</div>
          <h1 className="text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
            Pay for reliability, not tokens.
          </h1>
          <p className="lead mx-auto mt-4 max-w-2xl">
            Start free in the playground. Move to the API when you want it in
            production. Prices below are starting points, not commitments.
          </p>
        </div>
      </section>

      <section className="container-tight py-14">
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
          {plans.map((p) => (
            <div
              key={p.name}
              className={
                "card flex flex-col p-5 " +
                (p.featured ? "ring-2 ring-ink-900" : "")
              }
            >
              {p.featured && (
                <span className="tag mb-3 border-ink-900 text-ink-900">Most popular</span>
              )}
              <div className="text-sm font-semibold text-ink-900">{p.name}</div>
              <div className="mono mt-3 text-[30px] font-semibold leading-none text-ink-900">
                {p.price}
                <span className="text-[13px] font-normal text-ink-400"> {p.per}</span>
              </div>
              <ul className="mt-5 flex-1 space-y-2.5">
                {p.items.map((it) => (
                  <li key={it} className="flex gap-2 text-[13px] text-ink-600">
                    <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-ink-300" />
                    {it}
                  </li>
                ))}
              </ul>
              <Link
                href={p.href}
                className={(p.featured ? "btn-primary" : "btn-ghost") + " mt-6"}
              >
                {p.cta}
              </Link>
            </div>
          ))}
        </div>
      </section>

      <section className="border-t border-ink-200 bg-ink-50/40">
        <Section eyebrow="FAQ" title="The usual questions.">
          <div className="mt-8 grid gap-4 sm:grid-cols-2">
            {[
              { q: "Do you store my context?", a: "The profiling endpoint returns previews only and does not persist raw passage text." },
              { q: "Which models are supported?", a: "The profiler is model-agnostic. Calibrated risk estimates are available per model tier." },
              { q: "Can I self-host?", a: "Yes — private deployment is available on Enterprise." },
              { q: "Do I need to change my RAG stack?", a: "No. It is a single call you can insert before inference." },
            ].map((f) => (
              <div key={f.q} className="card p-5">
                <div className="text-sm font-semibold text-ink-900">{f.q}</div>
                <p className="mt-1.5 text-[13px] leading-relaxed text-ink-600">{f.a}</p>
              </div>
            ))}
          </div>
        </Section>
      </section>
    </>
  );
}