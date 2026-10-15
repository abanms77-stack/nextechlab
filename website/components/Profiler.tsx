"use client";

import { useState } from "react";
import type { ProfileResult } from "@/lib/profiler";
import { SAMPLES } from "@/lib/samples";
import { ResultPanel } from "./ResultPanel";

function parsePassages(raw: string): string[] {
  return raw
    .split(/\n\s*\n+/)
    .map((s) => s.replace(/\s+/g, " ").trim())
    .filter(Boolean);
}

export function Profiler({ compact = false }: { compact?: boolean }) {
  const [query, setQuery] = useState(SAMPLES[0].query);
  const [context, setContext] = useState(SAMPLES[0].context);
  const [result, setResult] = useState<ProfileResult | null>(null);
  const [before, setBefore] = useState<ProfileResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  async function call(nextQuery: string, nextContext: string): Promise<ProfileResult> {
    const res = await fetch("/api/profile", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ query: nextQuery, context: nextContext }),
    });
    if (!res.ok) {
      const j = await res.json().catch(() => ({}));
      throw new Error((j as { error?: string }).error ?? "request failed");
    }
    return (await res.json()) as ProfileResult;
  }

  async function run() {
    setBusy(true);
    setErr(null);
    try {
      setResult(await call(query, context));
      setBefore(null);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  async function optimize() {
    if (!result) return;
    setBusy(true);
    setErr(null);
    try {
      const keep = new Set(result.optimized.passages.map((p) => p.index));
      const pruned = parsePassages(context).filter((_, i) => keep.has(i));
      const newContext = pruned.join("\n\n");
      setBefore(result);
      setResult(await call(query, newContext));
      setContext(newContext);
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(false);
    }
  }

  const gain = before && result ? result.predictedSuccess - before.predictedSuccess : null;

  return (
    <div className={compact ? "" : "grid gap-6 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)]"}>
      <div className="card p-5">
        <div className="mb-3 flex flex-wrap gap-2">
          {SAMPLES.map((s) => (
            <button
              key={s.label}
              className="tag hover:border-ink-300 hover:text-ink-800"
              onClick={() => {
                setQuery(s.query);
                setContext(s.context);
                setResult(null);
                setBefore(null);
              }}
            >
              {s.label}
            </button>
          ))}
        </div>

        <label className="eyebrow mb-1.5 block">query</label>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full rounded-lg border border-ink-200 px-3 py-2.5 text-sm outline-none focus:border-accent-500"
          placeholder="What are you asking the model?"
        />

        <label className="eyebrow mb-1.5 mt-5 block">context (blank line = new passage)</label>
        <textarea
          value={context}
          onChange={(e) => setContext(e.target.value)}
          rows={compact ? 8 : 14}
          className="mono w-full resize-y rounded-lg border border-ink-200 p-3 text-[12.5px] leading-relaxed outline-none focus:border-accent-500"
          placeholder="Paste retrieved passages…"
        />

        <div className="mt-4 flex items-center justify-between gap-3">
          <button className="btn-primary" onClick={run} disabled={busy}>
            {busy ? "Analyzing…" : "Profile context"}
          </button>
          <span className="mono text-[11px] text-ink-400">
            {parsePassages(context).length} passages
          </span>
        </div>
        {err && <p className="mt-3 text-[12px] text-risk-high">{err}</p>}
      </div>

      <div className={compact ? "mt-6 space-y-6" : "space-y-6"}>
        {gain !== null && before && result && (
          <div className="card flex items-center justify-between gap-4 px-5 py-4">
            <div>
              <div className="eyebrow">before → after (estimated success)</div>
              <div className="mono mt-1 text-[18px] font-semibold text-ink-900">
                {Math.round(before.predictedSuccess * 100)}% → {Math.round(result.predictedSuccess * 100)}%
              </div>
            </div>
            <div className="text-right">
              <div className="eyebrow">gain</div>
              <div className="mono text-[18px] font-semibold text-risk-low">
                {gain >= 0 ? "+" : ""}
                {Math.round(gain * 100)} pts
              </div>
            </div>
          </div>
        )}

        {result ? (
          <ResultPanel result={result} onOptimize={optimize} busy={busy} />
        ) : (
          <div className="card grid min-h-[280px] place-items-center p-8 text-center">
            <div>
              <p className="text-sm font-medium text-ink-700">No profile yet</p>
              <p className="mt-1 text-[13px] text-ink-500">
                Paste a query and context, then run the profiler to see the
                interference risk, competing passages, and recommendations.
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}