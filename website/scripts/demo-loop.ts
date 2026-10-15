// END-TO-END DEMO: Detect -> Explain -> Optimize -> Re-profile.
// Every number is MEASURED from the engine, none is predetermined.
//
//   npx tsx scripts/demo-loop.ts

import { profileContext, splitPassages } from "../lib/profiler";

const pct = (x: number) => (x * 100).toFixed(0) + "%";

function show(label: string, r: ReturnType<typeof profileContext>) {
  console.log(
    `${label.padEnd(12)} D=${String(r.features.competitors).padStart(2)}  ` +
      `S=${pct(r.features.competitorSimilarity).padStart(4)}  ` +
      `I=${r.features.interference.toFixed(2)}  ` +
      `pred=${pct(r.predictedSuccess).padStart(4)}  risk=${r.risk.padEnd(6)}  ` +
      `kept=${r.optimized.passages.length}/${r.features.passages} removed=${r.optimized.removed}`,
  );
}

// ---- the exact scenario that previously reported D=0 / 94% ---------------
const QUERY = "What is the enterprise refund policy?";
const CLEAN = `The enterprise refund policy allows a full refund within 30 days of purchase, provided the licence has not been activated.

Refunds for annual plans are prorated for the unused period.`;

const COMPETING = `The enterprise refund policy allows a full refund within 30 days of purchase.

The enterprise refund policy allows a full refund within 14 days of purchase.

The enterprise refund policy allows a full refund within 60 days of purchase.

The enterprise refund policy allows a full refund within 7 days of purchase.

The enterprise refund policy requires approval from the account manager.

Refunds for annual plans are prorated for the unused period.

The enterprise refund policy is described in the supplier handbook.

The enterprise refund policy may be changed at any time.`;

console.log("=== ContextLab end-to-end loop (all values measured) ===\n");

const clean = profileContext({ query: QUERY, passages: splitPassages(CLEAN) });
show("CLEAN", clean);

const competing = profileContext({
  query: QUERY,
  passages: splitPassages(COMPETING),
});
show("COMPETING", competing);

// Optimize: keep the passages the engine decided to keep, re-profile them.
const optimizedPassages = splitPassages(COMPETING).filter((_, i) =>
  competing.optimized.passages.some((p) => p.index === i),
);
const optimized = profileContext({ query: QUERY, passages: optimizedPassages });
show("OPTIMIZED", optimized);

console.log("\n--- detail: competing context ---");
console.log(`recommendations:`);
for (const rec of competing.recommendations) console.log(`  · ${rec}`);
console.log(`competing pairs (top 6):`);
for (const p of competing.competingPairs.slice(0, 6)) {
  console.log(
    `  ${p.a} ↔ ${p.b}  sim=${pct(p.similarity)}  relA=${pct(p.relevanceA)} relB=${pct(p.relevanceB)}  strength=${pct(p.strength)}`,
  );
}

console.log("\n--- summary ---");
console.log(`predicted success: ${pct(clean.predictedSuccess)} -> ${pct(competing.predictedSuccess)} -> ${pct(optimized.predictedSuccess)}`);
console.log(`risk:              ${clean.risk} -> ${competing.risk} -> ${optimized.risk}`);
const recovered = optimized.predictedSuccess - competing.predictedSuccess;
console.log(`recovered by optimization: ${recovered >= 0 ? "+" : ""}${pct(recovered)} (engine-estimated)`);