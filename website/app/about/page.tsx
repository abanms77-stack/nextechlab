import type { Metadata } from "next";
import { Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "About",
  description:
    "ContextLab started as a controlled research project on LLM retrieval interference.",
};

const arc = ["Research", "Measurement", "Prediction", "Intervention", "Product"];

export default function AboutPage() {
  return (
    <>
      <section className="border-b border-ink-200">
        <div className="container-tight py-14">
          <div className="eyebrow mb-3">about</div>
          <h1 className="text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
            From measurement to infrastructure.
          </h1>
          <p className="lead mt-4 max-w-2xl">
            ContextLab started as a controlled research project studying how
            competing information affects LLM retrieval. It became a product when
            the research showed two things: the effect is predictable, and acting
            on it improves results.
          </p>
        </div>
      </section>

      <Section eyebrow="How we got here" title="Research → Product.">
        <div className="mt-8 grid gap-3 sm:grid-cols-5">
          {arc.map((a, i) => (
            <div key={a} className="card p-4">
              <div className="mono text-[11px] text-ink-400">0{i + 1}</div>
              <div className="mt-1.5 text-sm font-medium text-ink-800">{a}</div>
            </div>
          ))}
        </div>
        <p className="mt-8 max-w-3xl text-[14px] leading-relaxed text-ink-600">
          The path was deliberate. We measured the phenomenon, found a predictive
          relationship, tested whether it generalised, and only then built the
          intervention and the product. We did not start from a feature idea.
        </p>
      </Section>

      <Section eyebrow="Principles" title="How we work.">
        <ul className="mt-6 grid gap-3 sm:grid-cols-2">
          {[
            { t: "Evidence first", d: "Every product claim traces to a controlled measurement." },
            { t: "Honest limits", d: "We state what is not established, not just what is." },
            { t: "Measured, not guessed", d: "Similarity and interference are computed, not hand-labelled." },
            { t: "Actionable", d: "Diagnosis without intervention is not the product." },
          ].map((p) => (
            <li key={p.t} className="card p-5">
              <div className="text-sm font-semibold text-ink-900">{p.t}</div>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-600">{p.d}</p>
            </li>
          ))}
        </ul>
      </Section>

      <section className="border-t border-ink-200 bg-ink-50/40">
        <Section eyebrow="Contact" title="Work with us.">
          <p className="lead mt-4 max-w-2xl">
            For enterprise deployments, security review, or research
            collaboration, get in touch.
          </p>
          <a href="mailto:hello@contextlab.dev" className="btn-primary mt-6">
            hello@contextlab.dev
          </a>
        </Section>
      </section>
    </>
  );
}