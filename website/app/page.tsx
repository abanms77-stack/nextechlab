import Link from "next/link";
import { Profiler } from "@/components/Profiler";

const steps = [
  { k: "Analyze", d: "Measure which passages compete with the query." },
  { k: "Predict", d: "Estimate retrieval reliability before inference." },
  { k: "Optimize", d: "Prune, reorder, or rewrite the risky context." },
  { k: "Verify", d: "Compare Before vs After on the real model." },
];

const chain = [
  "More retrieved documents",
  "More competing information",
  "Higher interference",
  "Lower retrieval reliability",
];

const evidence = [
  { v: "3", l: "model families tested (Qwen, Llama, GPT-2)" },
  { v: "0.95", l: "cross-model correlation of the frozen law" },
  { v: "0.05", l: "calibration error (ECE) on a held-out model" },
  { v: "+0.35", l: "accuracy recovered by pruning on measured interference" },
];

export default function Home() {
  return (
    <>
      {/* hero */}
      <section className="relative overflow-hidden border-b border-ink-200">
        <div className="pointer-events-none absolute inset-0 bg-grid opacity-50" />
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-transparent via-white/40 to-white" />
        <div className="container-tight relative py-20 sm:py-28">
          <div className="max-w-3xl">
            <span className="tag">Research-backed · retrieval-interference</span>
            <h1 className="h1 mt-5">Make AI context reliable.</h1>
            <p className="lead mt-5 max-w-2xl">
              ContextLab detects competing information inside your LLM context,
              estimates retrieval risk, and automatically optimizes context
              before inference.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link href="/playground" className="btn-primary">
                Try Context Profiler
              </Link>
              <Link href="/research" className="btn-ghost">
                View Research
              </Link>
            </div>
          </div>

          <div className="mt-14 grid gap-px overflow-hidden rounded-xl border border-ink-200 bg-ink-200 sm:grid-cols-4">
            {steps.map((s, i) => (
              <div key={s.k} className="bg-white p-5">
                <div className="mono text-[11px] text-ink-400">0{i + 1}</div>
                <div className="mt-2 text-sm font-semibold text-ink-900">{s.k}</div>
                <p className="mt-1 text-[13px] leading-relaxed text-ink-600">{s.d}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* live demo */}
      <section id="demo" className="border-b border-ink-200 bg-ink-50/40">
        <div className="container-tight py-16 sm:py-20">
          <div className="mb-8 flex flex-wrap items-end justify-between gap-4">
            <div>
              <div className="eyebrow mb-3">live playground</div>
              <h2 className="h2">Profile a context in one click.</h2>
            </div>
            <p className="max-w-md text-[13px] text-ink-500">
              Runs in your browser session. Context is analysed for structure
              only and is never stored by the endpoint.
            </p>
          </div>
          <Profiler />
        </div>
      </section>

      {/* problem */}
      <section className="border-b border-ink-200">
        <div className="container-tight py-16 sm:py-20">
          <div className="eyebrow mb-3">the problem</div>
          <h2 className="h2 max-w-3xl">
            More context doesn&apos;t always mean better context.
          </h2>
          <p className="lead mt-4 max-w-2xl">
            Retrieval adds documents. Documents add competing information. Once
            several passages compete for the same answer, reliability falls —
            even when the answer is present.
          </p>

          <div className="mt-10 grid gap-3 md:grid-cols-4">
            {chain.map((c, i) => (
              <div key={c} className="card relative p-5">
                <div className="mono text-[11px] text-ink-400">step {i + 1}</div>
                <div className="mt-2 text-[14px] font-medium text-ink-800">{c}</div>
                {i < chain.length - 1 && (
                  <span className="absolute -right-2 top-1/2 hidden -translate-y-1/2 text-ink-300 md:block">
                    →
                  </span>
                )}
              </div>
            ))}
          </div>

          <div className="mt-10 overflow-hidden rounded-xl border border-ink-200">
            <table className="w-full text-left text-sm">
              <thead className="bg-ink-50">
                <tr>
                  <th className="px-4 py-3 font-medium text-ink-700">Context</th>
                  <th className="px-4 py-3 font-medium text-ink-700">Result</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-100">
                <tr>
                  <td className="px-4 py-3 text-ink-600">One relevant passage</td>
                  <td className="px-4 py-3 font-medium text-risk-low">Reliable</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 text-ink-600">+ unrelated passages</td>
                  <td className="px-4 py-3 font-medium text-risk-med">Slight loss</td>
                </tr>
                <tr>
                  <td className="px-4 py-3 text-ink-600">+ similar competing passages</td>
                  <td className="px-4 py-3 font-medium text-risk-high">Fails</td>
                </tr>
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[12px] text-ink-400">
            ContextLab focuses on context-induced retrieval failures — not on
            hallucinations in general.
          </p>
        </div>
      </section>
      {/* evidence */}
      <section className="border-b border-ink-200 bg-ink-50/40">
        <div className="container-tight py-16 sm:py-20">
          <div className="eyebrow mb-3">evidence (public summary)</div>
          <h2 className="h2 max-w-3xl">Built on controlled research.</h2>
          <div className="mt-10 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {evidence.map((e) => (
              <div key={e.l} className="card p-5">
                <div className="mono text-[28px] font-semibold text-ink-900">{e.v}</div>
                <p className="mt-2 text-[13px] leading-relaxed text-ink-600">{e.l}</p>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[12px] text-ink-400">
            High-level only. Methodology, full report, dataset and exact model
            are private.{" "}
            <Link href="/research" className="underline">
              Read the public summary →
            </Link>
          </p>
        </div>
      </section>

      {/* cta */}
      <section className="container-tight py-20 text-center">
        <h2 className="h2">Analyze. Optimize. Compare.</h2>
        <p className="lead mx-auto mt-4 max-w-xl">
          Paste your context and see the interference risk in seconds — then let
          the optimizer fix it and measure the Before vs After.
        </p>
        <div className="mt-8 flex justify-center gap-3">
          <Link href="/playground" className="btn-primary">
            Try ContextLab
          </Link>
          <Link href="/product" className="btn-ghost">
            See the product
          </Link>
        </div>
      </section>
    </>
  );
}