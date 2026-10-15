"use client";

import { useState } from "react";
import { Profiler } from "./Profiler";

const TABS = ["Overview", "Analyze", "Runs", "API keys", "Settings"] as const;
type Tab = (typeof TABS)[number];

const runs = [
  { id: "#1827", query: "enterprise refund policy", risk: "high", before: "43%", after: "81%", tokens: "1,204", latency: "38 ms" },
  { id: "#1826", query: "termination notice period", risk: "medium", before: "67%", after: "88%", tokens: "864", latency: "31 ms" },
  { id: "#1825", query: "refund window", risk: "low", before: "94%", after: "94%", tokens: "212", latency: "12 ms" },
];

const riskTone: Record<string, string> = {
  high: "text-risk-high",
  medium: "text-risk-med",
  low: "text-risk-low",
};

export function Dashboard() {
  const [tab, setTab] = useState<Tab>("Overview");

  return (
    <div>
      <div className="flex flex-wrap gap-1 border-b border-ink-200 pb-3">
        {TABS.map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={
              "rounded-md px-3 py-2 text-sm transition " +
              (tab === t ? "bg-ink-900 text-white" : "text-ink-600 hover:bg-ink-50")
            }
          >
            {t}
          </button>
        ))}
      </div>

      <div className="mt-6">
        {tab === "Overview" && (
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            {[
              { l: "Analyses this month", v: "8,412" },
              { l: "Mean risk", v: "medium" },
              { l: "Mean gain (optimized)", v: "+22 pts" },
              { l: "Median latency", v: "31 ms" },
            ].map((k) => (
              <div key={k.l} className="card p-5">
                <div className="eyebrow">{k.l}</div>
                <div className="mono mt-2 text-[22px] font-semibold text-ink-900">{k.v}</div>
              </div>
            ))}
          </div>
        )}

        {tab === "Analyze" && <Profiler />}

        {tab === "Runs" && (
          <div className="overflow-hidden rounded-xl border border-ink-200">
            <table className="w-full text-left text-sm">
              <thead className="bg-ink-50">
                <tr>
                  {["Run", "Query", "Risk", "Before", "After", "Tokens", "Latency"].map((h) => (
                    <th key={h} className="px-4 py-3 font-medium text-ink-700">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-ink-100">
                {runs.map((r) => (
                  <tr key={r.id}>
                    <td className="mono px-4 py-3 text-ink-500">{r.id}</td>
                    <td className="px-4 py-3 text-ink-700">{r.query}</td>
                    <td className={"px-4 py-3 font-medium " + riskTone[r.risk]}>{r.risk}</td>
                    <td className="mono px-4 py-3 text-ink-600">{r.before}</td>
                    <td className="mono px-4 py-3 text-ink-600">{r.after}</td>
                    <td className="mono px-4 py-3 text-ink-600">{r.tokens}</td>
                    <td className="mono px-4 py-3 text-ink-600">{r.latency}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        {tab === "API keys" && (
          <div className="card p-5">
            <div className="eyebrow mb-3">keys</div>
            <div className="flex items-center justify-between gap-4 rounded-lg border border-ink-200 px-4 py-3">
              <div>
                <div className="mono text-[13px] text-ink-800">cl_live_••••••••••••4f2a</div>
                <div className="text-[11px] text-ink-400">created 2 days ago · last used 3 min ago</div>
              </div>
              <button className="btn-ghost">Copy</button>
            </div>
            <button className="btn-primary mt-4">Create new key</button>
          </div>
        )}

        {tab === "Settings" && (
          <div className="grid gap-4 sm:grid-cols-2">
            {[
              { l: "Default risk threshold", v: "medium" },
              { l: "Auto-optimize", v: "on" },
              { l: "Model window", v: "128,000" },
              { l: "Data retention", v: "none (ephemeral)" },
            ].map((s) => (
              <div key={s.l} className="card flex items-center justify-between p-4">
                <div className="text-[13px] text-ink-700">{s.l}</div>
                <div className="mono text-[13px] text-ink-900">{s.v}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}