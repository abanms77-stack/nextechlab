import type { Metadata } from "next";
import { Section } from "@/components/ui";

export const metadata: Metadata = {
  title: "Status",
  description: "ContextLab service status.",
};

const services = [
  { name: "Profiler API", state: "Operational", ok: true },
  { name: "Optimizer", state: "Operational", ok: true },
  { name: "Dashboard", state: "Operational", ok: true },
];

export default function StatusPage() {
  return (
    <Section eyebrow="status" title="All systems operational.">
      <div className="mt-8 overflow-hidden rounded-xl border border-ink-200">
        <table className="w-full text-left text-sm">
          <thead className="bg-ink-50">
            <tr>
              <th className="px-4 py-3 font-medium text-ink-700">Service</th>
              <th className="px-4 py-3 font-medium text-ink-700">State</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-ink-100">
            {services.map((s) => (
              <tr key={s.name}>
                <td className="px-4 py-3 text-ink-700">{s.name}</td>
                <td className="px-4 py-3">
                  <span className="inline-flex items-center gap-2 text-risk-low">
                    <span className="h-2 w-2 rounded-full bg-risk-low" />
                    {s.state}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </Section>
  );
}