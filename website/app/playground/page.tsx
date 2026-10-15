import type { Metadata } from "next";
import { Profiler } from "@/components/Profiler";

export const metadata: Metadata = {
  title: "Playground",
  description:
    "Paste a query and context to profile interference risk and optimize it.",
};

export default function PlaygroundPage() {
  return (
    <>
      <section className="border-b border-ink-200">
        <div className="container-tight py-14">
          <div className="eyebrow mb-3">playground</div>
          <h1 className="text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
            Profile and optimize a context.
          </h1>
          <p className="lead mt-4 max-w-2xl">
            Paste your query and context, run the profiler, then optimize. The
            profiler measures which passages compete with the query and estimates
            retrieval reliability; the optimizer prunes the risky passages and
            re-profiles so you can see the Before vs After.
          </p>
        </div>
      </section>

      <section className="bg-ink-50/40">
        <div className="container-tight py-12">
          <Profiler />
        </div>
      </section>

      <section className="container-tight py-14">
        <div className="grid gap-4 sm:grid-cols-3">
          {[
            { t: "Analyze", d: "Per-passage similarity, competitor count, interference load, token footprint." },
            { t: "Optimize", d: "The optimizer proposes which passages to prune, rewrite or reorder." },
            { t: "Compare", d: "Before/After view shows the estimated change in retrieval reliability." },
          ].map((x) => (
            <div key={x.t} className="card p-5">
              <div className="text-sm font-semibold text-ink-900">{x.t}</div>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-600">{x.d}</p>
            </div>
          ))}
        </div>
        <p className="mt-6 text-[12px] text-ink-400">
          The playground uses the public profiler engine. Production deployments
          use the calibrated model and evaluation harness (private).
        </p>
      </section>
    </>
  );
}