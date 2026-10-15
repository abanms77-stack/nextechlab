import type { Metadata } from "next";
import { Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "Developers",
  description: "ContextLab API reference, SDK and examples.",
};

const sampleResponse = `{
  "risk": "medium",
  "predictedSuccess": 0.32,
  "confidence": 0.85,
  "tokensPct": 12.4,
  "features": {
    "passages": 4,
    "tokens": 842,
    "queryTokens": 5,
    "relevanceMean": 0.75,
    "relevantPassages": 4,
    "competitors": 3,          // D - passages in a competing cluster
    "competitorSimilarity": 0.67, // S - mean similarity of competing pairs
    "interference": 0.40,      // I = min(1, D/5) * S
    "lengthFactor": 0.11
  },
  "competingPairs": [
    { "a": "p1", "b": "p2", "similarity": 0.66 },
    { "a": "p1", "b": "p3", "similarity": 0.68 }
  ],
  "highRiskIds": ["p1", "p2", "p3"],
  "recommendations": [
    "3 of 4 passages form 1 competing group (mean competitor similarity 67%).",
    "Keep the most query-relevant passage in each group and prune the rest."
  ]
}`;

export default function DevelopersPage() {
  return (
    <>
      <section className="border-b border-ink-200">
        <div className="container-tight py-14">
          <div className="eyebrow mb-3">developers</div>
          <h1 className="text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
            Integrate context reliability in minutes.
          </h1>
          <p className="lead mt-4 max-w-2xl">
            A single endpoint profiles a context. Drop it into your RAG pipeline
            to gate, warn on, or optimize retrieved context.
          </p>
        </div>
      </section>

      <Section eyebrow="API reference" title="POST /api/profile">
        <div className="mt-8 grid gap-6 lg:grid-cols-2">
          <div className="card p-5">
            <div className="eyebrow mb-3">request</div>
            <pre className="mono overflow-x-auto rounded-lg bg-ink-900 p-4 text-[12px] leading-relaxed text-ink-100">
{`{
  "query": "string (required)",
  "context": "string (optional)",
  "passages": ["string[] (optional)"],
  "modelWindow": 128000
}`}
            </pre>
            <p className="mt-3 text-[12px] text-ink-500">
              Provide either <span className="mono">context</span> (blank-line
              separated) or <span className="mono">passages</span>.
            </p>
          </div>
          <div className="card p-5">
            <div className="eyebrow mb-3">response</div>
            <pre className="mono overflow-x-auto rounded-lg bg-ink-900 p-4 text-[12px] leading-relaxed text-ink-100">
              {sampleResponse}
            </pre>
          </div>
        </div>
      </Section>

      <Section id="sdk" eyebrow="SDK" title="Two lines to a risk score.">
        <pre className="mono mt-8 overflow-x-auto rounded-xl border border-ink-200 bg-ink-900 p-5 text-[12.5px] leading-relaxed text-ink-100">
{`const res = await fetch("/api/profile", {
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify({ query, context }),
});
const { risk, predictedSuccess, recommendations } = await res.json();`}
        </pre>
      </Section>

      <section id="examples" className="border-y border-ink-200 bg-ink-50/40">
        <Section eyebrow="Examples" title="Common integrations.">
          <div className="mt-8 grid gap-4 sm:grid-cols-3">
            {[
              { t: "Gate retrieval", d: "Block or retry when risk is high." },
              { t: "Warn in UI", d: "Surface reliability to end users." },
              { t: "Auto-optimize", d: "Prune and re-profile before inference." },
            ].map((x) => (
              <div key={x.t} className="card p-5">
                <div className="text-sm font-semibold text-ink-900">{x.t}</div>
                <p className="mt-1.5 text-[13px] leading-relaxed text-ink-600">{x.d}</p>
              </div>
            ))}
          </div>
        </Section>
      </section>

      <Section eyebrow="Limits & errors" title="Predictable behaviour.">
        <div className="mt-8 overflow-hidden rounded-xl border border-ink-200">
          <table className="w-full text-left text-sm">
            <thead className="bg-ink-50">
              <tr>
                <th className="px-4 py-3 font-medium text-ink-700">Code</th>
                <th className="px-4 py-3 font-medium text-ink-700">Meaning</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-ink-100">
              <tr><td className="mono px-4 py-3 text-ink-600">400</td><td className="px-4 py-3 text-ink-600">Missing query or context</td></tr>
              <tr><td className="mono px-4 py-3 text-ink-600">413</td><td className="px-4 py-3 text-ink-600">More than 200 passages</td></tr>
              <tr><td className="mono px-4 py-3 text-ink-600">429</td><td className="px-4 py-3 text-ink-600">Rate limit exceeded (plan-dependent)</td></tr>
            </tbody>
          </table>
        </div>
        <p className="mt-4 text-[12px] text-ink-400">
          Raw passage text is not stored by the profiling endpoint; only short
          previews are returned.
        </p>
      </Section>
    </>
  );
}