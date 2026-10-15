import type { Risk } from "@/lib/profiler";

export const riskColor: Record<Risk, string> = {
  low: "text-risk-low border-risk-low/30 bg-risk-low/5",
  medium: "text-risk-med border-risk-med/30 bg-risk-med/5",
  high: "text-risk-high border-risk-high/30 bg-risk-high/5",
};

export function RiskBadge({ risk }: { risk: Risk }) {
  return (
    <span
      className={
        "mono rounded-full border px-2.5 py-1 text-[11px] font-medium uppercase tracking-wider " +
        riskColor[risk]
      }
    >
      {risk} risk
    </span>
  );
}

export function Bar({ value, tone = "accent" }: { value: number; tone?: "accent" | "risk" }) {
  const pct = Math.round(Math.min(1, Math.max(0, value)) * 100);
  const color =
    tone === "risk"
      ? value >= 0.5
        ? "bg-risk-high"
        : value >= 0.25
          ? "bg-risk-med"
          : "bg-risk-low"
      : "bg-accent-500";
  return (
    <div className="h-1.5 w-full overflow-hidden rounded-full bg-ink-100">
      <div className={"h-full rounded-full " + color} style={{ width: pct + "%" }} />
    </div>
  );
}

export function Metric({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-ink-200 bg-white px-3.5 py-3">
      <div className="eyebrow mb-1">{label}</div>
      <div className="mono text-[19px] font-semibold leading-none text-ink-900">{value}</div>
      {hint && <div className="mt-1 text-[11px] text-ink-400">{hint}</div>}
    </div>
  );
}

export function Section({
  id,
  eyebrow,
  title,
  children,
  className = "",
}: {
  id?: string;
  eyebrow?: string;
  title?: string;
  children?: React.ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={"container-tight py-16 sm:py-20 " + className}>
      {eyebrow && <div className="eyebrow mb-3">{eyebrow}</div>}
      {title && <h2 className="h2 max-w-3xl">{title}</h2>}
      {children}
    </section>
  );
}