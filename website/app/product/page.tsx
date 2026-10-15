import type { Metadata } from "next";
import Link from "next/link";
import { Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "Product",
  description:
    "Context Profiler, Context Optimizer, Before/After verification, and API.",
};

const pipeline = [
  { t: "Your application", d: "Query + retrieved passages" },
  { t: "Context Profiler", d: "Measure · Predict · Explain" },
  { t: "Optimization Engine", d: "Prune · Reorder · Compress" },
  { t: "Before / After", d: "Real LLM result, measured" },
];

const profilerPoints = [
  "Per-passage similarity between each passage and the query.",
  "Competing-item count and an aggregate interference load.",
  "Retrieval-risk estimate and calibration, before inference.",
  "Token footprint and share of the model window.",
];

const optimizerPoints = [
  "Prune passages that compete with the query but add no answer.",
  "Reorder so the decisive passage sits where recall is strongest.",
  "Compress redundant context while preserving the answer.",
  "Never mutates without showing the Before/After estimate.",
];

export default function ProductPage() {
  return (
    <>
      <section className="border-b border-ink-200">
        <div className="container-tight py-14">
          <div className="eyebrow mb-3">product</div>
          <h1 className="text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
            One layer between retrieval and inference.
          </h1>
          <p className="lead mt-4 max-w-2xl">
            ContextLab sits inside your RAG pipeline. It profiles the context,
            predicts retrieval risk, optimizes it, and verifies the result — so
            reliability is engineered, not hoped for.
          </p>
        </div>
      </section>

      <section className="container-tight py-14">
        <div className="grid gap-3 md:grid-cols-4">
          {pipeline.map((p, i) => (
            <div key={p.t} className="card relative p-5">
              <div className="mono text-[11px] text-ink-400">stage {i + 1}</div>
              <div className="mt-2 text-sm font-semibold text-ink-900">{p.t}</div>
              <p className="mt-1 text-[13px] text-ink-600">{p.d}</p>
            </div>
          ))}
        </div>
      </section>

      <Section id="profiler" eyebrow="Context Profiler" title="See the risk before you run the model.">
        <ul className="mt-8 grid gap-3 sm:grid-cols-2">
          {profilerPoints.map((p) => (
            <li key={p} className="card flex gap-3 p-4 text-[13px] text-ink-700">
              <span className="mt-[6px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent-500" />
              {p}
            </li>
          ))}
        </ul>
        <Link href="/playground" className="btn-primary mt-8">
          Open the profiler
        </Link>
      </Section>

      <section id="optimizer" className="border-y border-ink-200 bg-ink-50/40">
        <Section eyebrow="Context Optimizer" title="Then fix it, automatically.">
          <ul className="mt-8 grid gap-3 sm:grid-cols-2">
            {optimizerPoints.map((p) => (
              <li key={p} className="card flex gap-3 p-4 text-[13px] text-ink-700">
                <span className="mt-[6px] h-1.5 w-1.5 shrink-0 rounded-full bg-ink-900" />
                {p}
              </li>
            ))}
          </ul>
        </Section>
      </section>

      <Section id="before-after" eyebrow="Before / After" title="Prove the improvement.">
        <p className="lead mt-4 max-w-2xl">
          Every optimization run shows the context before and after, plus the
          estimated change in retrieval reliability. In controlled research,
          pruning by measured interference recovered roughly a third of lost
          accuracy versus random pruning.
        </p>
        <div className="mt-8 overflow-hidden rounded-xl border border-ink-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-ink-50">
              <tr>
                <th className="px-4 py-3 font-medium text-ink-700">Condition</th>
                <th className="px-4 py-3 font-medium text-ink-700">Result</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-100">
              <tr><td className="px-4 py-3 text-ink-600">Full context</td><td className="px-4 py-3 font-medium text-risk-high">Degraded</td></tr>
              <tr><td className="px-4 py-3 text-ink-600">Random pruning</td><td className="px-4 py-3 font-medium text-risk-med">Partial recovery</td></tr>
              <tr><td className="px-4 py-3 text-ink-600">Profiler-guided pruning</td><td className="px-4 py-3 font-medium text-risk-low">Best recovery</td></tr>
            </tbody>
          </table>
        </div>
      </Section>

      <section id="api" className="border-t border-ink-200 bg-ink-50/40">
        <Section eyebrow="API" title="One endpoint.">
          <p className="lead mt-4 max-w-2xl">
            Send a query and context, get back risk, features and
            recommendations.
          </p>
          <pre className="mono mt-8 overflow-x-auto rounded-xl border border-ink-200 bg-ink-900 p-5 text-[12.5px] leading-relaxed text-ink-100">
{`curl -X POST /api/profile \\
  -H "content-type: application/json" \\
  -d '{"query":"...","context":"passage 1\\n\\npassage 2"}'`}
          </pre>
          <Link href="/developers" className="btn-ghost mt-6">
            Read the API docs
          </Link>
        </Section>
      </section>
    </>
  );
}