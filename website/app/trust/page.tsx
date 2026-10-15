import type { Metadata } from "next";
import { Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "Trust & security",
  description: "How ContextLab handles your context data.",
};

export default function TrustPage() {
  return (
    <>
      <section className="border-b border-ink-200">
        <div className="container-tight py-14">
          <div className="eyebrow mb-3">trust & security</div>
          <h1 className="text-3xl font-semibold tracking-tight text-ink-900 sm:text-4xl">
            Your context, handled carefully.
          </h1>
          <p className="lead mt-4 max-w-2xl">
            A context profiler sees sensitive retrieved text. Here is exactly how
            we treat it.
          </p>
        </div>
      </section>

      <Section eyebrow="Data handling" title="What we do with your context.">
        <ul className="mt-6 grid gap-3 sm:grid-cols-2">
          {[
            { t: "No raw retention", d: "The profiling endpoint does not persist raw passage text." },
            { t: "Previews only", d: "Responses contain short previews, not full passages." },
            { t: "Transport security", d: "All traffic is served over TLS." },
            { t: "Deletion", d: "Account data and stored runs can be deleted on request." },
            { t: "Isolation", d: "Enterprise deployments can run in your own environment." },
            { t: "Model providers", d: "The profiler is model-agnostic; no context is shared with providers by default." },
          ].map((x) => (
            <li key={x.t} className="card p-5">
              <div className="text-sm font-semibold text-ink-900">{x.t}</div>
              <p className="mt-1.5 text-[13px] leading-relaxed text-ink-600">{x.d}</p>
            </li>
          ))}
        </ul>
        <p className="mt-8 text-[12px] text-ink-400">
          We do not claim certifications we have not obtained. Compliance
          statements will be published here as they are achieved.
        </p>
      </Section>

      <section className="border-t border-ink-200 bg-ink-50/40">
        <Section eyebrow="Responsible use" title="What this product is not.">
          <p className="lead mt-4 max-w-2xl">
            ContextLab reduces context-induced retrieval failures. It is not a
            guarantee against hallucination, and it does not replace evaluation
            of your own task.
          </p>
        </Section>
      </section>
    </>
  );
}