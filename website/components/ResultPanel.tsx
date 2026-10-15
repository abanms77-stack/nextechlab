import type { ProfileResult } from "@/lib/profiler";
import { Bar, Metric, RiskBadge, riskColor } from "./ui";

const pct = (x: number) => Math.round(x * 100) + "%";

export function ResultPanel({
  result,
  onOptimize,
  busy,
  showOptimize = true,
}: {
  result: ProfileResult;
  onOptimize?: () => void;
  busy?: boolean;
  showOptimize?: boolean;
}) {
  const f = result.features;
  const dist = result.similarityDistribution;
  const maxCount = Math.max(1, ...dist.map((d) => d.count));

  return (
    <div className="card overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-ink-200 px-5 py-4">
        <div className="flex flex-wrap items-center gap-3">
          <RiskBadge risk={result.risk} />
          {result.hasConflict && (
            <span className="mono rounded-full border border-risk-high/40 bg-risk-high/5 px-2.5 py-1 text-[11px] font-medium uppercase tracking-wider text-risk-high">
              ⚠ conflict
            </span>
          )}
          <span className="mono text-[11px] text-ink-400">{result.engineVersion}</span>
        </div>
        <div className="text-right">
          <div className="eyebrow">predicted retrieval success</div>
          <div className="mono text-[26px] font-semibold leading-none text-ink-900">
            {pct(result.predictedSuccess)}
          </div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 p-5 sm:grid-cols-4">
        <Metric label="competitors (D)" value={String(f.competitors)} hint={`of ${f.passages} passages`} />
        <Metric label="competitor sim (S)" value={pct(f.competitorSimilarity)} hint="passage ↔ passage" />
        <Metric label="interference (D×S)" value={f.interference.toFixed(2)} hint="aggregate load" />
        <Metric label="tokens" value={f.tokens.toLocaleString()} hint={`${result.tokensPct.toFixed(1)}% of window`} />
      </div>

      <div className="px-5 pb-2">
        <div className="eyebrow mb-2">context relationships</div>
        <div className="flex flex-wrap gap-2">
          {[
            { k: "contradictory", v: f.contradictoryPairs, tone: "text-risk-high border-risk-high/30 bg-risk-high/5" },
            { k: "redundant", v: f.redundantPairs, tone: "text-risk-med border-risk-med/30 bg-risk-med/5" },
            { k: "supporting", v: f.supportingPairs, tone: "text-risk-low border-risk-low/30 bg-risk-low/5" },
            { k: "related", v: f.relatedPairs, tone: "text-ink-600 border-ink-200 bg-ink-50" },
          ].map((r) => (
            <span key={r.k} className={"mono rounded-full border px-2.5 py-1 text-[11px] " + r.tone}>
              {r.k} {r.v}
            </span>
          ))}
        </div>
      </div>

      <div className="grid gap-5 px-5 pb-5 sm:grid-cols-2">
        <div>
          <div className="eyebrow mb-2">pairwise competition distribution</div>
          <div className="space-y-1.5">
            {dist.length === 0 && (
              <p className="text-[12px] text-ink-400">Only one passage — no pairs to compare.</p>
            )}
            {dist.map((d) => (
              <div key={d.bucket} className="flex items-center gap-3">
                <span className="mono w-16 shrink-0 text-[11px] text-ink-500">{d.bucket}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-ink-100">
                  <div className="h-full rounded-full bg-ink-700" style={{ width: Math.round((d.count / maxCount) * 100) + "%" }} />
                </div>
                <span className="mono w-6 text-right text-[11px] text-ink-500">{d.count}</span>
              </div>
            ))}
          </div>
        </div>
        <div>
          <div className="eyebrow mb-2">competing pairs</div>
          {result.competingPairs.length === 0 ? (
            <p className="text-[12px] text-ink-400">No competing pairs detected.</p>
          ) : (
            <ul className="space-y-1.5">
              {result.competingPairs.slice(0, 6).map((p, i) => (
                <li key={i} className="mono flex items-center justify-between text-[12px] text-ink-600">
                  <span>
                    {p.a} ↔ {p.b}
                  </span>
                  <span className="text-ink-900">{pct(p.similarity)}</span>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
      <div className="border-t border-ink-200 px-5 py-5">
        <div className="eyebrow mb-3">passages</div>
        <ul className="space-y-3">
          {result.passages.map((p) => (
            <li key={p.id} className="rounded-lg border border-ink-200 px-3.5 py-3">
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="mono text-[11px] font-medium text-ink-500">{p.id}</span>
                    <span className={"mono rounded border px-1.5 py-0.5 text-[10px] uppercase " + riskColor[p.risk]}>
                      {p.risk}
                    </span>
                    <span className="mono rounded border border-ink-200 bg-white px-1.5 py-0.5 text-[10px] uppercase text-ink-500">
                      {p.relation}
                    </span>
                    {!p.keep && (
                      <span className="mono rounded border border-ink-200 bg-ink-50 px-1.5 py-0.5 text-[10px] uppercase text-ink-500">
                        prune
                      </span>
                    )}
                  </div>
                  <p className="mt-1.5 text-[13px] leading-relaxed text-ink-600">{p.preview}</p>
                </div>
                <div className="w-32 shrink-0 space-y-2">
                  <div>
                    <Bar value={p.relevance} />
                    <div className="mono mt-1 text-right text-[10px] text-ink-400">
                      relevance {pct(p.relevance)}
                    </div>
                  </div>
                  <div>
                    <Bar value={p.competition} tone="risk" />
                    <div className="mono mt-1 text-right text-[10px] text-ink-400">
                      competition {pct(p.competition)}
                    </div>
                  </div>
                  <div className="mono text-right text-[10px] text-ink-400">
                    nearest {pct(p.nearestSimilarity)}
                  </div>
                </div>
              </div>
            </li>
          ))}
        </ul>
      </div>

      <div className="border-t border-ink-200 bg-ink-50/50 px-5 py-5">
        <div className="eyebrow mb-3">recommendations</div>
        <ul className="space-y-2">
          {result.recommendations.map((r, i) => (
            <li key={i} className="flex gap-2 text-[13px] leading-relaxed text-ink-700">
              <span className="mt-[7px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent-500" />
              <span>{r}</span>
            </li>
          ))}
        </ul>
      </div>

      {showOptimize && onOptimize && (
        <div className="flex flex-wrap items-center justify-between gap-3 border-t border-ink-200 px-5 py-4">
          <div className="text-[12px] text-ink-500">
            Optimizer will keep {result.optimized.passages.length} and prune{" "}
            {result.optimized.removed} passage{result.optimized.removed === 1 ? "" : "s"}.
          </div>
          <button className="btn-primary" onClick={onOptimize} disabled={busy}>
            {busy ? "Optimizing…" : "Optimize context →"}
          </button>
        </div>
      )}
    </div>
  );
}