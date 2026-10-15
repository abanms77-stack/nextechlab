import type { Metadata } from "next";
import { Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "Research",
  description:
    "Public, high-level summary of the retrieval-interference research behind ContextLab.",
};

const results = [
  { v: "3", l: "model families" },
  { v: "~0.95", l: "cross-model correlation" },
  { v: "0.05", l: "calibration error (ECE)" },
  { v: "+0.35", l: "accuracy recovered by pruning" },
];

export default function ResearchPage() {
  return (
    <>
      <section className="border-b border-ink-200">
        <div className="container-tight py-14">
          <div className="eyebrow mb-3">research · public summary</div>
          <h1 className="text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
            Retrieval interference, measured.
          </h1>
          <p className="lead mt-4 max-w-2xl">
            ContextLab began as a controlled research project on how competing
            information affects LLM retrieval. This page is a deliberately
            high-level summary.
          </p>
        </div>
      </section>

      <Section id="methodology" eyebrow="Methodology" title="A controlled, quality-blind benchmark.">
        <p className="lead mt-4 max-w-3xl">
          We build contexts token-by-token with known structure: a target fact,
          a variable number of competing passages, a measured similarity between
          each competitor and the target, and a controlled context length. The
          answer is a single token from a fixed set, so scoring is deterministic.
          We then vary competition count, similarity, and length independently
          and measure retrieval accuracy.
        </p>
        <p className="mt-4 max-w-3xl text-[14px] leading-relaxed text-ink-600">
          Full methodology, the evaluation set, and the calibrated model are
          private.
        </p>
      </Section>

      <section id="results" className="border-y border-ink-200 bg-ink-50/40">
        <Section eyebrow="Results" title="What the research found.">
          <div className="mt-8 grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {results.map((r) => (
              <div key={r.l} className="card p-5">
                <div className="mono text-[28px] font-semibold text-ink-900">{r.v}</div>
                <p className="mt-2 text-[13px] leading-relaxed text-ink-600">{r.l}</p>
              </div>
            ))}
          </div>
          <ul className="mt-8 space-y-3">
            {[
              "Interference — not raw context length — is the dominant control on whether a fact stays retrievable.",
              "Retrieval accuracy tracks a multiplicative interaction between competitor count and measured similarity.",
              "A law fitted on one model, then frozen, predicted an unseen model's cells without refitting.",
              "Pruning the most similar competing passages recovered substantially more accuracy than random pruning.",
            ].map((t) => (
              <li key={t} className="flex gap-3 text-[14px] leading-relaxed text-ink-700">
                <span className="mt-[8px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent-500" />
                {t}
              </li>
            ))}
          </ul>
        </Section>
      </section>

      <Section eyebrow="Limitations" title="Where the claim stops.">
        <ul className="mt-6 space-y-3">
          {[
            "The validated finding is a retrieval-interference result; cross-task transfer was not established.",
            "Models tested were relatively small; very long contexts (8K–128K) were not the focus.",
            "The mechanism is an interpretation, not a proven internal quantity.",
          ].map((t) => (
            <li key={t} className="flex gap-3 text-[14px] leading-relaxed text-ink-700">
              <span className="mt-[8px] h-1.5 w-1.5 shrink-0 rounded-full bg-ink-900" />
              {t}
            </li>
          ))}
        </ul>
        <p className="mt-8 text-[12px] text-ink-400">
          The full report, raw data, evaluation set and exact model are private
          to the company.
        </p>
      </Section>
    </>
  );
}